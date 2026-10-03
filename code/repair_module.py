#!/usr/bin/env python3
"""
受约束的修复模块（Constrained Repair Module）

三层修复策略，由便宜到昂贵：
- L1 确定性修复：纯静态规则，无 LLM 调用
- L2 编译器级重生成：把 reference 签名喂给 DSLv2Compiler 的 LLM 提示
- L3 受约束 LLM 补丁：用 LLM 写一个 IR-preserving 的 step_1 重写

每次修复尝试都记录到 repair_log，便于论文里做失败模式分析。
"""
import os, sys, re, ast, json, time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# LLM Prompt 模板
# ============================================================================

L3_SYSTEM_PROMPT = """你是 Python 函数重写专家。
你的任务：根据给定的"任务描述"和"期望函数签名"，写一个 Python 函数体。

严格约束：
1. 函数签名必须与给定的"期望函数签名"完全一致（参数名、参数顺序）
2. 函数体必须确定性：无 await、无 I/O、无 random、无 time.sleep
3. 不要 import 任何非标准库
4. 不允许输出任何解释、注释（仅代码外的 docstring 除外）
5. 如果参考信息里有可用的辅助函数，可以用
6. 返回值类型由任务描述决定
7. 处理边界条件：None、负数、空列表等
8. 只输出 Python 代码（用一个 ```python 代码块包裹），其他什么都不要输出
"""

# P0-1 增强 (2026-06-12): 在 L3 prompt 中显式列出常见边界条件 + 错误模式
# 目的: pilot 实验中 ours FPR (65%) 略低于 dsl (70%), 5 个失败任务中有
#       3 个集中在"边界值未显式处理"(如 None, 0, 负数) 和 "类型隐式转换"
# 预期: ΔFPR 从 -2 pp 转正到 +3-5 pp
L3_BOUNDARY_HINTS = """
## 容易出错的边界条件 (请显式处理)

A. 空值与缺失:
   - 输入可能为 None: 使用 `if x is None: return <默认>` 提前返回
   - 字符串为空: `if not s: return <默认>`
   - 列表/字典为空: `if not lst: return <默认>`

B. 数值边界:
   - 除零保护: 除法前检查分母
   - 负数场景: 价格、数量不能为负
   - 零值: 0% 折扣、0 元订单是合法值
   - 浮点精度: 涉及金额时用 `round(x, 2)` 截断

C. 字符串匹配:
   - 严格用 `==` 而非 `in` 匹配枚举值 (避免 'Gold' 误匹配 'GoldPlus')
   - 字符串去前后空格: `member_level.strip()`

D. 多分支互斥:
   - if/elif/else 必须完备, 不留隐式 fallthrough
   - 同一变量不能既走 if 又走 elif

E. 类型转换:
   - 显式 `int()` / `float()` 转换, 不要依赖 Python 自动转换
   - 字典 `.get(key, default)` 替代 `dict[key]`, 避免 KeyError

F. 复合条件:
   - 多个条件组合用括号明确优先级: `(a > 0) and (b > 0)`
"""

L3_USER_TEMPLATE = """## 任务描述
{prompt}

## 期望函数签名
{signature}

## 上下文（IR 已生成的辅助函数，可选用）
{context}

{boundary_hints}

## 你的输出
请写一个 Python 函数，签名完全匹配期望函数签名，函数体实现任务描述里的规则。
"""

# RQ2 baseline: 通用 LLM 修复（无签名约束）
L_UNCONSTRAINED_SYSTEM = """你是 Python 程序员。给你一个原始代码（可能不工作）和它的任务描述，
请你重写一个能正确运行的 Python 版本。

要求：
- 可以自由选择函数名、参数、签名
- 只需要让任务描述里的功能正确
- 输出一个完整的 Python 代码块
"""

L_UNCONSTRAINED_USER = """## 任务描述
{prompt}

## 原始代码（不保证正确）
{code}

## 你的输出
请重写这个任务，输出一个能正确运行的 Python 代码（用 ```python 代码块包裹）。
"""


# ============================================================================
# 工具函数
# ============================================================================

def _strip_code_block(text: str) -> str:
    """从 LLM 输出里抽取 ```python ... ``` 代码块。"""
    if not text:
        return ""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 没找到代码块，整段当代码返回
    return text.strip()


def _parse_signature_simple(sig: str):
    """解析 'def solve(a, b, c) -> int:' → ('solve', ['a','b','c'], 'int')"""
    if not sig:
        return ("solve", [], "")
    try:
        m = re.match(r"^\s*def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^:]+))?\s*:\s*$", sig.strip(), re.DOTALL)
        if m:
            name = m.group(1)
            params_str = m.group(2).strip()
            retty = (m.group(3) or "").strip()
            params = [p.strip().split("=")[0].strip().split(":")[0].strip() for p in params_str.split(",") if p.strip()]
            return (name, params, retty)
    except Exception:
        pass
    return ("solve", [], "")


def _validate_code_syntax(code: str) -> tuple:
    """检查代码是否能被 ast 解析，并提取出 solve 函数定义。"""
    if not code:
        return (False, "empty code")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (False, f"SyntaxError: {e}")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return (True, node.name)
    return (False, "no function def")


# ============================================================================
# 修复模块
# ============================================================================

class RepairModule:
    """三阶段修复模块。

    Usage:
        rm = RepairModule(llm_client, model_name="deepseek-v3")
        result = rm.repair(compile_result, prompt, reference_signature)
        # result = {success, code, level, attempts, log}
    """

    def __init__(self, llm_client=None, model_name="deepseek-v3", max_attempts: int = 1,
                 unconstrained: bool = False,
                 enable_l1: bool = True, enable_l3: bool = True,
                 boundary_hints_enabled: bool = True):
        """初始化修复模块。

        unconstrained=True 时（RQ2 baseline 用）：
        - L1 跳过（不做签名约束的修整）
        - L3 用通用 LLM 重写 prompt，不强制签名匹配，不做 verify
        - 用于对比：受约束修复 vs 通用 LLM 修复

        enable_l1 / enable_l3: ablation 开关。两者都为 True 是 "full" 路径；
        RQ5 ablation 用 enable_l1=False 或 enable_l3=False 观察 L1/L3 的贡献。
        boundary_hints_enabled: P0-1 开关, 默认 True, 可关闭以做 A/B 对比.
        """
        self.llm = llm_client
        self.model = model_name
        self.max_attempts = max_attempts
        self.unconstrained = unconstrained
        self.enable_l1 = enable_l1
        self.enable_l3 = enable_l3
        # P0-1 新增: 边界条件提示开关
        self.boundary_hints_enabled = boundary_hints_enabled
        self.log = []

    def _log(self, level: str, action: str, detail: str = ""):
        entry = {"level": level, "action": action, "detail": detail, "t": time.time()}
        self.log.append(entry)

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def repair(self, compile_result: dict, prompt: str, reference_signature: str) -> dict:
        """按 L1 → L2 → L3 顺序尝试，返回第一个成功的或最终 fallback。"""
        self.log = []
        self._log("info", "start", f"ref_sig={reference_signature[:60]}, unconstrained={self.unconstrained}")

        if not compile_result.get("success"):
            self._log("warn", "skip", "compile_result.success=False")
            return {
                "success": False, "code": compile_result.get("code", ""),
                "level": "none", "attempts": [], "log": self.log,
                "reason": "compile_failed",
            }

        # RQ2 baseline 路径：unconstrained 模式
        if self.unconstrained:
            return self._l_unconstrained(compile_result, prompt)

        # 解析 reference 签名
        ref_name, ref_params, _ = _parse_signature_simple(reference_signature)
        if not ref_params:
            self._log("warn", "no_params", f"无法从 '{reference_signature}' 解析参数")
            return {
                "success": False, "code": compile_result.get("code", ""),
                "level": "none", "attempts": [], "log": self.log,
                "reason": "no_reference_params",
            }

        # 当前 IR 的产物是否已经满足 reference 签名
        if _signature_matches(compile_result.get("code", ""), ref_name, ref_params):
            self._log("info", "no_repair_needed", "已匹配 reference 签名")
            return {
                "success": True, "code": compile_result.get("code", ""),
                "level": "none", "attempts": [], "log": self.log,
                "reason": "matched",
            }

        # L1 尝试（ablation 可关）
        if self.enable_l1:
            l1 = self._l1_deterministic(compile_result, ref_name, ref_params)
            if l1.get("success"):
                self._log("info", "l1_success", l1.get("reason", ""))
                return {
                    "success": True, "code": l1["code"], "level": "L1",
                    "attempts": [l1], "log": self.log,
                }
        else:
            l1 = {"success": False, "reason": "l1_disabled_ablation"}
            self._log("info", "l1_skipped", "ablation: enable_l1=False")

        # L2 跳过（与 L3 重叠且成本高，暂用 L3 替代）

        # L3 尝试（ablation 可关）
        if self.enable_l3:
            l3 = self._l3_llm_patch(compile_result, prompt, reference_signature, ref_name, ref_params)
            if l3.get("success"):
                self._log("info", "l3_success", l3.get("reason", ""))
                return {
                    "success": True, "code": l3["code"], "level": "L3",
                    "attempts": [l1, l3], "log": self.log,
                }
        else:
            l3 = {"success": False, "reason": "l3_disabled_ablation"}
            self._log("info", "l3_skipped", "ablation: enable_l3=False")

        # 全部失败
        return {
            "success": False, "code": compile_result.get("code", ""),
            "level": "unfixable", "attempts": [l1, l3], "log": self.log,
            "reason": "all_levels_failed",
        }

    # ------------------------------------------------------------------
    # L1: 确定性修复
    # ------------------------------------------------------------------
    def _l1_deterministic(self, compile_result: dict, ref_name: str, ref_params: list) -> dict:
        """L1 策略：纯静态规则，零 LLM 调用。

        L1.1 强制重新包装：用 reference 的签名直接改写 wrapper，参数转发
        L1.2 step_4 提代：当 step_1 是空壳时，尝试用 step_4

        P0-2 增强 (2026-06-12): 加 smoke_call gate.
        目的: 防止 L1 "过修正" (改写前能 smoke 跑通, 改写后跑不通, 应回滚).
        之前 pilot 实验中 ours Adapt% (81.83%) 反而低于 dsl (85.78%), 一个主因
        就是 L1.1 偶发把通过测试的 wrapper 改成不工作版本.
        """
        self._log("L1", "start", "")

        original_code = compile_result.get("code", "")
        if not original_code:
            return {"success": False, "reason": "empty_code"}

        # P0-2: 改写前 smoke gate
        before_smoke = _smoke_call(original_code, ref_name, ref_params)

        # L1.1: 重新包装 - 把 wrapper 改成 *args, **kwargs 形式
        l1_1 = self._l1_1_relax_wrapper(original_code, ref_name, ref_params)
        if l1_1.get("success"):
            # P0-2: 改写后 smoke gate
            after_smoke = _smoke_call(l1_1.get("code", ""), ref_name, ref_params)
            if before_smoke and not after_smoke:
                # 改写前能跑、改写后跑不了 → 过修正, 回滚
                self._log("L1.1", "gate_rollback", "before=pass after=fail")
                return {"success": False, "reason": "l1_gate_rollback_smoke_fail"}
            self._log("L1.1", "ok", l1_1.get("reason", "") + f"|gate(before={before_smoke},after={after_smoke})")
            return l1_1

        # L1.2: step_4 提代
        l1_2 = self._l1_2_promote_step4(original_code, ref_name, ref_params)
        if l1_2.get("success"):
            after_smoke = _smoke_call(l1_2.get("code", ""), ref_name, ref_params)
            if before_smoke and not after_smoke:
                self._log("L1.2", "gate_rollback", "before=pass after=fail")
                return {"success": False, "reason": "l1_gate_rollback_smoke_fail"}
            self._log("L1.2", "ok", l1_2.get("reason", "") + f"|gate(before={before_smoke},after={after_smoke})")
            return l1_2

        return {"success": False, "reason": "no_l1_strategy_matched"}

    def _l1_1_relax_wrapper(self, code: str, ref_name: str, ref_params: list) -> dict:
        """L1.1: 把 wrapper 改成 *args/**kwargs 形式，让签名 mismatch 消失。"""
        if not ref_params:
            return {"success": False, "reason": "no_ref_params"}

        # 把 def solve(a, b, c): 改成 def solve(*args, **kwargs):
        # body 用 *args/**kwargs 转发
        new_code = re.sub(
            rf"def\s+{re.escape(ref_name)}\s*\([^)]*\)\s*:",
            f"def {ref_name}(*args, **kwargs):",
            code,
            count=1,
        )
        if new_code == code:
            return {"success": False, "reason": "no_wrapper_to_relax"}

        # 找到 wrapper 函数体里的 return 调用，改成 *args/**kwargs
        m = re.search(
            rf"(def\s+{re.escape(ref_name)}\s*\(\*args\s*,\s*\*\*kwargs\)\s*:\s*(?:\"\"\"[^\"]*\"\"\")?\s*)(return\s+\w+\([^)]*\))",
            new_code,
            re.DOTALL,
        )
        if m:
            call_m = re.search(r"return\s+(\w+)\(", m.group(2))
            if call_m:
                step_name = call_m.group(1)
                new_code = new_code.replace(
                    m.group(2),
                    f"return {step_name}(*args, **kwargs)",
                    1,
                )

        # 验证语法
        ok, info = _validate_code_syntax(new_code)
        if not ok:
            return {"success": False, "reason": f"syntax_error: {info}"}

        # 真实 smoke call
        if not _smoke_call(new_code, ref_name, ref_params):
            return {"success": False, "reason": "smoke_call_failed"}

        return {"success": True, "code": new_code, "reason": "wrapper_relaxed", "strategy": "L1.1"}

    def _l1_2_promote_step4(self, code: str, ref_name: str, ref_params: list) -> dict:
        """L1.2: 用 step_4 替代 step_1 作为 wrapper 的目标。"""
        if not ref_params:
            return {"success": False, "reason": "no_ref_params"}

        # 检查是否有 step_4_xxx 函数
        m = re.search(r"def\s+(step_4_\w+)\s*\(([^)]*)\)\s*:", code)
        if not m:
            return {"success": False, "reason": "no_step4"}

        step4_name = m.group(1)
        step4_params = m.group(2).strip()
        if step4_params:  # step_4 本身有参数，不能直接替代
            return {"success": False, "reason": "step4_has_params"}

        # 替换 wrapper 中的 main function 名
        new_code = re.sub(
            rf"def\s+{re.escape(ref_name)}\s*\([^)]*\)\s*:[^\n]*\n(?:    \"\"\"[^\"]*\"\"\"\s*\n)?(?:    [^\n]*\n)*?    return\s+(\w+)\(",
            lambda m: f"def {ref_name}({', '.join(ref_params)}):\n    return {step4_name}(",
            code,
            count=1,
        )
        if new_code == code:
            return {"success": False, "reason": "no_wrapper_to_patch"}

        ok, info = _validate_code_syntax(new_code)
        if not ok:
            return {"success": False, "reason": f"syntax_error: {info}"}

        # 真实 smoke call
        if not _smoke_call(new_code, ref_name, ref_params):
            return {"success": False, "reason": "smoke_call_failed"}

        return {"success": True, "code": new_code, "reason": "step4_promoted", "strategy": "L1.2"}

    # ------------------------------------------------------------------
    # L3: 受约束 LLM 补丁
    # ------------------------------------------------------------------
    def _l3_llm_patch(self, compile_result: dict, prompt: str,
                      reference_signature: str, ref_name: str, ref_params: list) -> dict:
        """L3: 用 LLM 重写 step_1（或直接写一个 solve），IR 其他部分保留作为上下文。"""
        self._log("L3", "start", f"ref_params={ref_params}")

        if self.llm is None:
            return {"success": False, "reason": "no_llm_client"}

        # 构造 context：把 IR 已生成的函数作为可用 helpers
        context = self._build_context(compile_result.get("code", ""))

        # P0-1 增强: 在 L3 prompt 中嵌入边界条件提示
        # 开关: L3_BOUNDARY_HINTS_ENABLED (默认 True, 可关闭以做 A/B)
        boundary_hints = L3_BOUNDARY_HINTS if getattr(self, "boundary_hints_enabled", True) else ""

        user_msg = L3_USER_TEMPLATE.format(
            prompt=prompt,
            signature=reference_signature,
            context=context or "（无可用辅助函数）",
            boundary_hints=boundary_hints,
        )

        try:
            t0 = time.time()
            resp = self.llm.call(
                system_prompt=L3_SYSTEM_PROMPT,
                user_content=user_msg,
                temperature=0.0,
                max_tokens=2048,
            )
            elapsed = time.time() - t0
            self._log("L3", "llm_returned", f"elapsed={elapsed:.1f}s, len={len(resp.content)}")
        except Exception as e:
            self._log("L3", "llm_error", str(e)[:200])
            return {"success": False, "reason": f"llm_error: {e}"}

        new_code = _strip_code_block(resp.content)
        if not new_code:
            return {"success": False, "reason": "empty_llm_output"}

        # 验证：必须包含 def solve(...) 且签名匹配
        ok, info = self._verify_repaired_code(new_code, ref_name, ref_params)
        if not ok:
            self._log("L3", "verify_fail", info)
            return {"success": False, "reason": f"verify_fail: {info}"}

        # 把修复后的 solve 与 IR 已有 helpers 合并（保证 IR 结构保留）
        merged = self._merge_with_ir_helpers(new_code, compile_result.get("code", ""))
        return {"success": True, "code": merged, "reason": "llm_rewrite", "strategy": "L3.1"}

    def _l_unconstrained(self, compile_result: dict, prompt: str) -> dict:
        """RQ2 baseline: 通用 LLM 修复（无签名约束、无 verify、无 IR 合并）。

        与 L3 的区别：
        - 不强制函数名/签名
        - 不做 verify（让 LLM 自由写）
        - 不合并 IR helpers（避免再引入编译上下文）
        - 用例：RQ2 对比"受约束修复" vs "通用 LLM 修复"
        """
        self._log("L_UNCONSTRAINED", "start", "RQ2 baseline: 通用 LLM 修复")
        if self.llm is None:
            return {"success": False, "level": "L_UNCONSTRAINED",
                    "attempts": [], "log": self.log, "reason": "no_llm_client"}

        original_code = compile_result.get("code", "")
        user_msg = L_UNCONSTRAINED_USER.format(prompt=prompt, code=original_code)
        attempts = []  # RQ5: 记录每次 LLM 调用

        try:
            t0 = time.time()
            resp = self.llm.call(
                system_prompt=L_UNCONSTRAINED_SYSTEM,
                user_content=user_msg,
                temperature=0.0,
                max_tokens=2048,
            )
            elapsed = time.time() - t0
            attempts.append({
                "level": "L_UNCONSTRAINED",
                "elapsed_s": round(elapsed, 3),
                "input_chars": len(user_msg),
                "output_chars": len(resp.content or ""),
                "strategy": "generic_rewrite",
            })
            self._log("L_UNCONSTRAINED", "llm_returned", f"elapsed={elapsed:.1f}s, len={len(resp.content)}")
        except Exception as e:
            self._log("L_UNCONSTRAINED", "llm_error", str(e)[:200])
            return {"success": False, "level": "L_UNCONSTRAINED",
                    "attempts": attempts, "log": self.log, "reason": f"llm_error: {e}"}

        new_code = _strip_code_block(resp.content)
        if not new_code:
            return {"success": False, "level": "L_UNCONSTRAINED",
                    "attempts": attempts, "log": self.log, "reason": "empty_llm_output"}

        # 仅做语法检查，不做签名验证
        ok, info = _validate_code_syntax(new_code)
        if not ok:
            self._log("L_UNCONSTRAINED", "syntax_fail", info)
            return {"success": False, "level": "L_UNCONSTRAINED",
                    "attempts": attempts, "log": self.log, "reason": f"syntax_error: {info}"}

        return {"success": True, "code": new_code, "reason": "unconstrained_rewrite",
                "strategy": "L_UNCONSTRAINED", "level": "L_UNCONSTRAINED",
                "attempts": attempts, "log": self.log}

    def _build_context(self, code: str) -> str:
        """提取 IR 里的辅助函数作为 context。"""
        if not code:
            return ""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ""
        out = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                # 只保留非 solve/wrapper 函数
                if "wrapper" in (ast.get_docstring(node) or "").lower():
                    continue
                src = ast.get_source_segment(code, node) or ast.unparse(node)
                out.append(src)
        return "\n\n".join(out[:6])  # 最多 6 个，避免 prompt 过长

    def _verify_repaired_code(self, code: str, ref_name: str, ref_params: list) -> tuple:
        """验证 LLM 产物：必须能 import，且函数签名匹配。"""
        ok, info = _validate_code_syntax(code)
        if not ok:
            return (False, f"syntax: {info}")

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return (False, f"parse: {e}")

        # 找 ref_name 函数
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == ref_name:
                actual_params = [a.arg for a in node.args.args]
                if actual_params == ref_params:
                    return (True, "ok")
                return (False, f"param_mismatch: expected {ref_params}, got {actual_params}")
        return (False, f"function '{ref_name}' not found")

    def _merge_with_ir_helpers(self, repaired_solve: str, ir_code: str) -> str:
        """把修复后的 solve 与 IR 已有 helpers 合并。"""
        if not ir_code:
            return repaired_solve
        try:
            ir_tree = ast.parse(ir_code)
        except SyntaxError:
            return repaired_solve

        # 收集 IR 中所有函数定义名
        ir_funcs = set()
        for node in ir_tree.body:
            if isinstance(node, ast.FunctionDef):
                ir_funcs.add(node.name)

        # 收集 repaired_solve 里的函数名
        try:
            rep_tree = ast.parse(repaired_solve)
        except SyntaxError:
            return repaired_solve
        rep_funcs = set()
        for node in rep_tree.body:
            if isinstance(node, ast.FunctionDef):
                rep_funcs.add(node.name)

        # 提取 IR 中被 repaired 引用但未在 repaired 中重定义的函数
        repaired_src = repaired_solve
        # 简单做法：把 IR 中所有"非 ref_name 且不在 repaired 中"的函数拼到末尾
        helpers_to_add = []
        for node in ir_tree.body:
            if isinstance(node, ast.FunctionDef):
                if node.name in rep_funcs:
                    continue
                src = ast.get_source_segment(ir_code, node)
                if src:
                    helpers_to_add.append(src)

        if not helpers_to_add:
            return repaired_solve
        return repaired_solve + "\n\n" + "\n\n".join(helpers_to_add)


# ============================================================================
# 签名匹配检测
# ============================================================================

def _signature_matches(code: str, ref_name: str, ref_params: list) -> bool:
    """检测产物里 def <ref_name>(<ref_params>): 是否已存在且调用正确。

    实际执行代码 + 用占位参数调用 solve，看是否报 signature error。
    """
    if not code or not ref_name or not ref_params:
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    found = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == ref_name:
            actual = [a.arg for a in node.args.args]
            if actual != ref_params:
                return False
            found = True
    if not found:
        return False
    # 真实试一下：用占位参数调用
    return _smoke_call(code, ref_name, ref_params)


def _smoke_call(code: str, fn_name: str, ref_params: list) -> bool:
    """实际执行 code 并调用 fn_name(占位参数...)，看是否通过。

    占位参数根据参数名启发式给出：
    - str → ""
    - int/float → 0
    - bool → False
    - 其他 → None
    """
    import importlib.util, sys as _sys
    if not code or not fn_name:
        return False
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    # 准备占位参数
    placeholders = []
    for p in ref_params:
        pn = p.lower()
        if any(k in pn for k in ("str", "name", "level", "type", "status", "code", "mode", "role", "dtype", "purpose", "consent", "scene", "agent")):
            placeholders.append("")
        elif any(k in pn for k in ("bool", "is_", "has_", "can_", "ovld")):
            placeholders.append(False)
        elif any(k in pn for k in ("float", "amount", "rate", "score", "weight", "total", "lim", "burst", "qps", "pri", "dur", "dist")):
            placeholders.append(0.0)
        elif any(k in pn for k in ("int", "age", "count", "month", "day", "zip", "stock", "tenant", "member")):
            placeholders.append(0)
        elif any(k in pn for k in ("list", "arr", "vec", "items")):
            placeholders.append([])
        elif any(k in pn for k in ("dict", "map", "obj")):
            placeholders.append({})
        else:
            placeholders.append(None)
    # 动态执行
    mod = _sys.modules.get(f"_smoke_{id(code)}")
    if mod is None:
        mod = type(_sys)("_smoke")
        _sys.modules[mod.__name__] = mod
    try:
        exec(code, mod.__dict__)
        fn = getattr(mod, fn_name, None)
        if fn is None:
            return False
        fn(*placeholders)
        return True
    except Exception:
        return False


# ============================================================================
# 直接测试
# ============================================================================

if __name__ == "__main__":
    # 演示
    sample_code = '''
def step_1_compute_scene(user_input):
    if "default" in user_input:
        return 0, "未知智能体"

def step_4_compute_scene():
    return 0, "未知"

def solve(order_amount, member_level, has_coupon):
    """..."""
    return step_1_compute_scene(order_amount, member_level, has_coupon)
'''
    rm = RepairModule(llm_client=None)
    r = rm.repair(
        {"success": True, "code": sample_code},
        prompt="设计一个订单折扣系统。",
        reference_signature="def solve(order_amount, member_level, has_coupon):",
    )
    print(json.dumps(r, ensure_ascii=False, indent=2))
