"""
DSL 执行引擎 (方案 C) — 轻量 DSL 解释器。

方案 C 范围：
- CONTAINS / NOT CONTAINS 关键词匹配
- IF / ELIF / ELSE / ENDIF 控制流
- {{x}} = value 赋值
- CALL sb_xxx → mock 为成功
- RETURN 拦截并记录

不处理：
- 数值比较 (>, <, ==)
- 循环 (FOR)
- 外部函数真实调用
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================================
# 数据结构
# ============================================================================


@dataclass
class CaseResult:
    """一个 case 的执行结果。"""
    case_index: int = 0
    execution_path: list[str] = field(default_factory=list)
    expected_path: list[str] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    all_blocks_tried: list[str] = field(default_factory=list)
    passed: bool = True


@dataclass
class ExecutionResults:
    """全部 case 的执行结果总和。"""
    results: list[CaseResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_blocks(self) -> set[str]:
        """收集所有失败的 BLOCK ID。

        核心逻辑：取"期望路径里有但实际没命中的"block，
        而非"实际执行路径里的"block。
        这样才能让 PipelineV2 的升级循环正确定位需要升级的 BLOCK。
        """
        failed = set()
        for r in self.results:
            if not r.passed:
                actual_set = set(r.execution_path)
                for bid in r.expected_path:
                    if bid not in actual_set:
                        failed.add(bid)
                if not r.execution_path:
                    # 什么都没匹配到，期望路径里的全部失败
                    for bid in r.expected_path:
                        failed.add(bid)
        return failed


# ============================================================================
# BLOCK 解析
# ============================================================================


class ParsedCondition:
    """解析后的单个 IF 条件节点。"""
    def __init__(self, kind: str, **kwargs):
        self.kind = kind          # "contains" | "not_contains" | "always"
        self.kwargs = kwargs      # {"var": "user_input", "keywords": ["a","b"]}

    def evaluate(self, ctx: dict) -> bool:
        if self.kind == "always":
            return True
        var_name = self.kwargs.get("var", "user_input")
        input_text = str(ctx.get(var_name, ""))
        keywords = self.kwargs.get("keywords", [])
        if self.kind == "contains":
            return any(kw in input_text for kw in keywords)
        elif self.kind == "not_contains":
            return not any(kw in input_text for kw in keywords)
        return False

    def __repr__(self):
        return f"ParsedCondition({self.kind}, {self.kwargs})"


class ParsedBlock:
    """解析后的单个 BLOCK 节点。"""
    def __init__(self, block_id: str):
        self.block_id = block_id
        self.condition_chains: list[list[ParsedCondition]] = []
        # condition_chains[i] 对应 IF / ELIF i 的条件列表
        self.actions: list[list[dict]] = []
        # actions[i] 对应 IF / ELIF i 命中后的操作
        self.has_return: bool = False


class DSLExecutor:
    """DSL 执行引擎。每次 run() 执行一个 case。"""

    def run(self, dsl_code: str, input_values: dict) -> CaseResult:
        """执行一段 DSL 对一个输入 case 的匹配。

        Args:
            dsl_code: DSL 1.2 语法的文本
            input_values: case 的 INPUT 字典

        Returns:
            CaseResult 包含命中的执行路径和输出
        """
        blocks = self._parse_blocks(dsl_code)
        result = CaseResult()
        ctx = dict(input_values)

        for block in blocks:
            result.all_blocks_tried.append(block.block_id)
            matched = self._try_block(block, ctx, result)
            if matched and block.has_return:
                break  # RETURN 后不再执行后续 BLOCK

        # 从 ctx 中提取 scene, agent, response 等输出变量
        result.output = {}
        for key in ("scene", "agent", "response", "reply", "status"):
            if key in ctx:
                result.output[key] = ctx[key]
        return result

    def run_examples(
        self,
        dsl_code: str,
        examples: list[dict],
    ) -> ExecutionResults:
        """对 DSL 执行一组 EXAMPLES。

        Args:
            dsl_code: DSL 代码
            examples: 示例列表，每项含 input, expected, execution_path

        Returns:
            ExecutionResults 包含所有 case 的执行结果
        """
        results = []
        for idx, case in enumerate(examples):
            result = self.run(dsl_code, case.get("input", {}))
            result.case_index = idx
            expected_path = case.get("execution_path", [])
            result.expected_path = list(expected_path)
            # 比对 EXECUTION_PATH
            if expected_path and result.execution_path != expected_path:
                result.passed = False
            result.output = case.get("expected", {})
            results.append(result)
        return ExecutionResults(results=results)

    # ------------------------------------------------------------------
    # BLOCK 解析
    # ------------------------------------------------------------------

    def _parse_blocks(self, dsl_code: str) -> list[ParsedBlock]:
        """从 DSL 文本中提取所有 BLOCK。"""
        blocks = []
        current_block = None
        lines = dsl_code.split("\n")
        in_block = False
        i = 0

        while i < len(lines):
            raw_line = lines[i]
            stripped = raw_line.strip()

            # 检测 BLOCK 开始
            m = re.match(r'^BLOCK\s+(\w+)\s+"', stripped)
            if m:
                current_block = ParsedBlock(m.group(1))
                # 不创建初始空链——等遇到 IF 再创建
                in_block = True
                i += 1
                continue

            if not in_block:
                i += 1
                continue

            # 检测 BLOCK 结束
            if stripped == "ENDBLOCK":
                if current_block:
                    blocks.append(current_block)
                current_block = None
                in_block = False
                i += 1
                continue

            # 计算缩进级别
            indent = len(raw_line) - len(raw_line.lstrip())
            # 缩进 0 或 4 为外层 IF/ELIF（新分支链），
            # 缩进 8+ 为嵌套 IF（追加到当前链）
            is_outer_if = indent <= 4

            # 解析外层 IF / ELIF（新分支链）
            if is_outer_if and (stripped.startswith("IF ") or stripped.startswith("ELIF ")):
                current_block.condition_chains.append([])
                current_block.actions.append([])
                self._parse_condition_to_current(stripped, current_block)
                i += 1
                continue

            # 解析 ELSE (无条件匹配，外层)
            if is_outer_if and stripped.startswith("ELSE"):
                current_block.condition_chains.append([ParsedCondition("always")])
                current_block.actions.append([])
                i += 1
                continue

            if stripped == "ENDIF":
                i += 1
                continue

            # 解析嵌套 IF（缩进的 IF — 追加到当前链）
            m_if_nested = re.match(r'IF\s+\{\{(\w+)\}\}\s+(NOT\s+)?CONTAINS', stripped)
            if m_if_nested:
                chain_idx = len(current_block.condition_chains) - 1
                if chain_idx >= 0:
                    self._parse_condition_to_current(stripped, current_block)
                i += 1
                continue

            # 解析赋值语句 {{x}} = value
            m_assign = re.match(r'\{\{(\w+)\}\}\s*=\s*(.+)', stripped)
            if m_assign and "CALL" not in stripped:
                chain_idx = len(current_block.condition_chains) - 1
                if chain_idx >= 0:
                    current_block.actions[chain_idx].append({
                        "type": "assign",
                        "var": m_assign.group(1),
                        "value": m_assign.group(2).strip(),
                    })
                i += 1
                continue

            # 解析 CALL 语句
            if "CALL" in stripped and "sb_" in stripped:
                m_call = re.match(r'\{\{(\w+)\}\}\s*=\s*CALL\s+(\w+)\(', stripped)
                if m_call:
                    chain_idx = len(current_block.condition_chains) - 1
                    if chain_idx >= 0:
                        current_block.actions[chain_idx].append({
                            "type": "call_mock",
                            "var": m_call.group(1),
                            "func": m_call.group(2),
                        })
                i += 1
                continue

            # 解析 CALL 新语法: CALL sb_xxx INPUT {...} OUTPUT {...}
            m_call_new = re.match(r'CALL\s+(\w+)\s+INPUT', stripped)
            if m_call_new:
                chain_idx = len(current_block.condition_chains) - 1
                if chain_idx >= 0:
                    current_block.actions[chain_idx].append({
                        "type": "call_mock",
                        "func": m_call_new.group(1),
                        "var": "",
                    })
                i += 1
                continue

            # 解析 IF {{var}} != "value" (CALL 后的门控)
            m_gate = re.match(r'\{\{(\w+)\}\}\s*(!=|==)\s*"([^"]*)"', stripped)
            if m_gate:
                # CALL mock 成功后，门控条件自动满足
                chain_idx = len(current_block.condition_chains) - 1
                current_block.condition_chains[chain_idx].append(
                    ParsedCondition("always")
                )
                i += 1
                continue

            # 解析 RETURN
            if stripped.startswith("RETURN"):
                current_block.has_return = True
                chain_idx = len(current_block.condition_chains) - 1
                if chain_idx >= 0:
                    current_block.actions[chain_idx].append({
                        "type": "return",
                    })
                i += 1
                continue

            i += 1

        return blocks

    def _parse_condition_to_current(self, line: str, block: ParsedBlock):
        """解析 IF/ELIF 条件并添加到当前条件链。"""
        chain_idx = len(block.condition_chains) - 1
        # CONTAINS
        m_contains = re.search(r'\{\{(\w+)\}\}\s+CONTAINS\s+\[([^\]]*)\]', line)
        if m_contains:
            var = m_contains.group(1)
            keywords = [kw.strip().strip('"') for kw in m_contains.group(2).split(",") if kw.strip()]
            block.condition_chains[chain_idx].append(
                ParsedCondition("contains", var=var, keywords=keywords)
            )
            return

        # NOT CONTAINS
        m_not = re.search(r'\{\{(\w+)\}\}\s+NOT\s+CONTAINS\s+\[([^\]]*)\]', line)
        if m_not:
            var = m_not.group(1)
            keywords = [kw.strip().strip('"') for kw in m_not.group(2).split(",") if kw.strip()]
            block.condition_chains[chain_idx].append(
                ParsedCondition("not_contains", var=var, keywords=keywords)
            )
            return

    # ------------------------------------------------------------------
    # BLOCK 执行
    # ------------------------------------------------------------------

    def _try_block(self, block: ParsedBlock, ctx: dict, result: CaseResult) -> bool:
        """尝试执行一个 BLOCK。返回是否匹配并进入。"""
        # 如果没有条件链（如 fallback block），默认匹配
        if not block.condition_chains:
            result.execution_path.append(block.block_id)
            # 执行所有 action
            for actions in block.actions:
                self._execute_actions(actions, ctx)
            return block.has_return

        for chain_idx, chain in enumerate(block.condition_chains):
            if not chain:
                continue  # 跳过空链
            # 检查这一条 IF/ELIF 链的所有条件是否都满足
            all_ok = all(cond.evaluate(ctx) for cond in chain)
            if not all_ok:
                continue

            # 执行该链的动作
            actions = block.actions[chain_idx] if chain_idx < len(block.actions) else []
            self._execute_actions(actions, ctx)
            result.execution_path.append(block.block_id)

            if block.has_return:
                return True
            return True

        return False

    def _execute_actions(self, actions: list[dict], ctx: dict):
        """执行一组动作。"""
        for action in actions:
            if action["type"] == "assign":
                ctx[action["var"]] = self._coerce_value(action["value"])
            elif action["type"] == "call_mock":
                # 方案 C: CALL sb_xxx 直接 mock 为成功
                ctx[action.get("var", action.get("func", ""))] = "success"
            elif action["type"] == "return":
                pass

    @staticmethod
    def _coerce_value(value_str: str) -> Any:
        """将字符串值转为合适的 Python 类型。"""
        if value_str.isdigit():
            return int(value_str)
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str
        if value_str.startswith("{{") and value_str.endswith("}}"):
            return value_str
        return value_str
