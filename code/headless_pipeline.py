#!/usr/bin/env python3
"""
无头编译管道：将 prompt3.0 真实编译器包装为简单函数调用接口。
核心路径: PromptPreprocessor(标准化) → DSLv2Compiler(DSL编译) → WaActCompiler(Python代码生成)

被 batch_run.py 的 ours 方法调用。
"""

import sys, os, io, json, re, ast, time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_DEVNULL = io.StringIO()
_TERM_MAPPING = {
    "帮我": "请", "那个": "", "嘛": "", "吧": "", "这些": "", "那套": "等工具",
    "RAG": "检索增强生成", "LLM": "大型语言模型",
    "chain": "处理链", "K8s": "Kubernetes", "ELK": "ELK日志系统",
    "QPS": "每秒查询率(QPS)", "top-3": "前3个结果", "top-5": "前5个结果",
    "向量索引": "向量索引(VI)", "知识图谱": "知识图谱(KG)",
    "缓存命中率": "缓存命中率(CRH)",
}


def compile_prompt_to_code(prompt_text: str, output_mode: str = "agent",
                           function_signature: str = None,
                           enable_repair: bool = False,
                           repair_module = None,
                           reference_prompt: str = None) -> dict:
    """
    Compile a prompt to code.

    Args:
        prompt_text: 原始自然语言 prompt
        output_mode:
            - "agent": 原始 agent 编排代码（含 step_*/sb_fallback）—— 默认
            - "function": 把产物包装成 eval 期望的 solve() 接口
        function_signature:
            当 output_mode="function" 时，给出参考签名 "def solve(a, b, c):"
    """
    # === RQ5 成本统计：开始计时 ===
    _cost_t0 = time.time()
    _cost_stages = []
    _cost_llm_calls = 0
    _cost_in_tokens = 0
    _cost_out_tokens = 0

    def _record_stage(name: str, llm: int = 0, ms: int = 0, in_tok: int = 0, out_tok: int = 0):
        nonlocal _cost_llm_calls, _cost_in_tokens, _cost_out_tokens
        _cost_llm_calls += llm
        _cost_in_tokens += in_tok
        _cost_out_tokens += out_tok
        _cost_stages.append({"stage": name, "llm_calls": llm, "ms": ms,
                             "approx_input_tokens": in_tok, "approx_output_tokens": out_tok})

    result = {"success": False, "code": "", "dsl_code": "", "modules": [], "error": "", "compile_details": {}, "wrap_info": {}}
    stdout_bak, stderr_bak = sys.stdout, sys.stderr

    try:
        sys.stdout = _DEVNULL
        sys.stderr = _DEVNULL
        _mute_logger()

        from data_models import ProcessingMode
        from prompt_preprocessor import PromptPreprocessor
        from dsl_v2 import PromptCompiler as DSLv2Compiler
        from prompt_codegenetate import WaActCompiler

        # 阶段1: 预处理
        pp = PromptPreprocessor(
            mode=ProcessingMode.HYBRID,
            term_mapping=_TERM_MAPPING,
            ambiguity_blacklist=["这个", "那个", "它", "他们", "某些"],
            use_mock_llm=True,
            enable_deep_check=False,
        )
        p10 = pp.process(prompt_text, save_history=False, show_comparison=False)
        processed = p10.processed_text or prompt_text
        p10_id = getattr(p10, 'id', '')

        # 阶段2.0: PromptStructurizer（实体抽取 → Prompt 2.0 模板）
        # 阶段2.5: Prompt 2.5 语义结构
        # 与 demo_full_pipeline.py 的 5 阶段管线对齐
        dsl_input = None
        prompt25_result = None
        compile_input = processed
        compile_label = "原始文本"

        def _template_quality_ok(tpl: str, original: str) -> bool:
            """模板质量检查：模板必须保留原 prompt 中的关键业务词（避免 mock 模式
            下被过度抽象为 "CALL generate_output" 的废模板）。"""
            if not tpl or len(tpl.strip()) < 20:
                return False
            # 提取原 prompt 中长度 >= 2 的中文/英文关键词
            import re as _re
            orig_words = set(w for w in _re.findall(r'[\u4e00-\u9fa5]{2,}|[A-Za-z_]{3,}', original) if len(w) >= 2)
            if not orig_words:
                return True  # 没法判断则放行
            tpl_lower = tpl
            kept = sum(1 for w in orig_words if w in tpl_lower)
            # 至少保留 30% 的关键词，或至少 3 个
            return kept >= max(3, int(0.3 * len(orig_words)))

        try:
            from prompt_structurizer import PromptStructurizer, VariableMeta
            from data_models import create_prompt20_result, convert_prompt20_to_dsl_input

            # use_mock 优先 False（真实 LLM，对齐 demo 行为）。
            # 如果环境没有 LLM_API_KEY 等环境变量，create_llm_client 会用空 key，
            # 实际调用时失败——但我们的 except 会兜住并回退到 use_mock=True。
            use_mock_for_stage2 = False
            try:
                import os as _os
                if not (_os.environ.get("LLM_API_KEY") or _os.environ.get("QWEN_API_KEY")
                        or _os.environ.get("DEEPSEEK_API_KEY") or _os.environ.get("ANTHROPIC_API_KEY")):
                    use_mock_for_stage2 = True
            except Exception:
                use_mock_for_stage2 = True

            structurizer = PromptStructurizer(use_mock=use_mock_for_stage2)
            _stage2_t0 = time.time()
            try:
                prompt_structure, _opt = structurizer.process(processed)
                template = prompt_structure.template_text
                variable_registry = prompt_structure.variable_registry
                extraction_log = prompt_structure.extraction_log
            except Exception as e_proc:
                # 真实 LLM 失败 → 回退到 mock 再试一次
                if not use_mock_for_stage2:
                    try:
                        structurizer = PromptStructurizer(use_mock=True)
                        prompt_structure, _opt = structurizer.process(processed)
                        template = prompt_structure.template_text
                        variable_registry = prompt_structure.variable_registry
                        extraction_log = prompt_structure.extraction_log
                        result.setdefault("stage_warnings", []).append(
                            f"stage2_real_llm_failed_fallback_mock: {str(e_proc)[:80]}"
                        )
                    except Exception as e_mock:
                        raise RuntimeError(f"stage2_process_failed: {e_proc} | {e_mock}")
                else:
                    raise
            _record_stage("stage2_structurize",
                          llm=0 if use_mock_for_stage2 else 1,
                          ms=int((time.time() - _stage2_t0) * 1000),
                          in_tok=len(processed) // 4,
                          out_tok=len(template) // 4)

            # 把 variable_registry (List[Dict]) 转为 variable_metas (List[VariableMeta])
            variable_metas = []
            for var_dict in (variable_registry or []):
                var_meta = VariableMeta(
                    name=var_dict.get('variable', ''),
                    original_text=var_dict.get('original_text', ''),
                    value=var_dict.get('value', ''),
                    data_type=var_dict.get('type', ''),
                    start_index=0,
                    end_index=0,
                )
                variable_metas.append(var_meta)

            # 构建 Prompt20Result + dsl_input（带质量检查）
            if variable_metas and _template_quality_ok(template, processed):
                prompt20_result = create_prompt20_result(
                    source_prompt10_id=p10_id,
                    original_text=processed,
                    template_text=template,
                    variables=variable_metas,
                    processing_time_ms=0,
                )
                dsl_input = convert_prompt20_to_dsl_input(prompt20_result)
                compile_label = f"Prompt 2.0 模板 ({len(variable_metas)} 变量)"
            else:
                result.setdefault("stage_warnings", []).append(
                    f"stage2_template_quality_poor_fallback_processed"
                )

            # 阶段2.5: Prompt 2.5 语义结构
            _stage25_t0 = time.time()
            try:
                prompt25_result = structurizer.process_to_prompt25(p10)
                _record_stage("stage25_semantic",
                              llm=0 if use_mock_for_stage2 else 1,
                              ms=int((time.time() - _stage25_t0) * 1000),
                              in_tok=len(processed) // 4,
                              out_tok=len(getattr(prompt25_result, 'intent', '') or '') // 4)
            except Exception as e25:
                prompt25_result = None
                result.setdefault("stage_warnings", []).append(f"prompt25_failed: {str(e25)[:120]}")

        except Exception as e2:
            # Stage 2 整体失败不阻塞后续（demo 中也有回退路径）
            dsl_input = None
            prompt25_result = None
            result.setdefault("stage_warnings", []).append(f"stage2_failed: {str(e2)[:120]}")

        # 决策：使用 Prompt 2.5 / Prompt 2.0 / 原始文本 作为 DSL 编译输入
        if prompt25_result is not None:
            use_prompt25 = (
                getattr(prompt25_result, 'intent', None)
                and len(prompt25_result.intent) > 10
                and (len(getattr(prompt25_result, 'operations', []) or []) > 0
                     or len(getattr(prompt25_result, 'branches', []) or []) > 0)
                and getattr(prompt25_result, 'confidence', 0) >= 0.7
            )
            if use_prompt25:
                compile_input = prompt25_result.intent
                compile_label = f"Prompt 2.5 语义 (conf={prompt25_result.confidence:.2f})"
        if compile_label == "原始文本" and dsl_input is not None:
            compile_input = dsl_input.get('prompt_text', processed)
            compile_label = "Prompt 2.0 模板"

        result["compile_input"] = compile_input
        result["compile_label"] = compile_label

        # 阶段3: DSL 编译（这里是 LLM 决策点：IR 阶段）
        dsl_compiler = DSLv2Compiler(max_retries=3, auto_fix_threshold=3, use_mock=False)
        _stage3_t0 = time.time()
        dsl_r = dsl_compiler.compile(compile_input)
        _stage3_ms = int((time.time() - _stage3_t0) * 1000)
        dsl_code = getattr(dsl_r, 'dsl_code', '') or ''
        # DSLv2 compile 内部可能调 N 次 LLM（修复 cycle），但我们没有暴露 N；记 1 次保守估计
        _record_stage("stage3_dsl_compile",
                      llm=1 if getattr(dsl_r, 'success', False) else 1,
                      ms=_stage3_ms,
                      in_tok=len(str(compile_input)) // 4,
                      out_tok=len(dsl_code) // 4)

        if not getattr(dsl_r, 'success', False):
            errs = []
            if hasattr(dsl_r, 'validation_result') and dsl_r.validation_result:
                errs = getattr(dsl_r.validation_result, 'errors', ['unknown'])
            result["error"] = f"DSL compile failed: {errs}"
            result["dsl_code"] = dsl_code
            if output_mode == "function":
                result["code"] = _build_function_stub(function_signature, "DSL compile failed (IR 阶段 LLM 失败)")
            return result

        result["dsl_code"] = dsl_code

        # 阶段3: 代码生成（这里是 deterministic emit 阶段，无 LLM）
        _stage4_t0 = time.time()
        code_compiler = WaActCompiler()
        modules, main_code, compile_details = code_compiler.compile(dsl_code, clustering_strategy="hybrid", visualize=False)
        _record_stage("stage4_emit",
                      llm=0,
                      ms=int((time.time() - _stage4_t0) * 1000),
                      in_tok=0,
                      out_tok=len(main_code or "") // 4)

        # 如果生成的是编排代码（非独立函数），提取模块函数体
        code = main_code or ""
        if not _has_def(code) and modules:
            parts = []
            for m in modules:
                if hasattr(m, 'body_code') and m.body_code:
                    parts.append(m.body_code)
            if parts:
                code = "\n\n".join(parts)

        has_semantic_block = _has_semantic_block(code, modules)

        if output_mode == "function":
            target_name, target_params, target_retty = _parse_signature(function_signature)
            code, wrap_info = _wrap_as_function(code, modules, target_name, target_params, target_retty, has_semantic_block)
            result["wrap_info"] = wrap_info
        else:
            result["wrap_info"] = {"mode": "agent", "semantic_block_detected": has_semantic_block}

        result["success"] = True
        result["code"] = code
        result["modules"] = [getattr(m, 'name', 'unknown') for m in modules] if modules else []
        result["compile_details"] = compile_details or {}

        # 修复钩子：当 output_mode=function 且失败/不匹配时尝试 L1/L3 修复
        if enable_repair and output_mode == "function" and function_signature and repair_module is not None:
            ref_prompt = reference_prompt or prompt_text
            _repair_t0 = time.time()
            repair_r = repair_module.repair(result, ref_prompt, function_signature)
            _repair_ms = int((time.time() - _repair_t0) * 1000)
            # 修复阶段调用的 LLM 次数（从 attempts / log 里推算）
            _repair_llm = len(repair_r.get("attempts", []))
            _repair_out_chars = len(repair_r.get("code", "") or "")
            result["repair_info"] = {
                "level": repair_r.get("level"),
                "success": repair_r.get("success"),
                "reason": repair_r.get("reason"),
                "attempts_count": _repair_llm,
                "log": repair_r.get("log", []),
            }
            _record_stage("repair_module",
                          llm=_repair_llm,
                          ms=_repair_ms,
                          in_tok=len(ref_prompt) // 4,
                          out_tok=_repair_out_chars // 4)
            if repair_r.get("success"):
                result["code"] = repair_r["code"]
                result["repair_applied"] = True

    except Exception as e:
        result["error"] = f"Pipeline error: {str(e)[:500]}"
        if output_mode == "function":
            result["code"] = _build_function_stub(function_signature, f"Pipeline exception: {str(e)[:120]}")
    finally:
        sys.stdout = stdout_bak
        sys.stderr = stderr_bak

        # === RQ5 成本统计：写入 cost_stats 字段 ===
        _cost_total_ms = int((time.time() - _cost_t0) * 1000)
        result["cost_stats"] = {
            "pipeline_total_ms": _cost_total_ms,
            "total_llm_calls": _cost_llm_calls,
            "approx_input_tokens": _cost_in_tokens,
            "approx_output_tokens": _cost_out_tokens,
            "stages": _cost_stages,
        }

    return result


# ============================================================================
# output_mode="function" 包装层
# ============================================================================

def _has_semantic_block(code: str, modules) -> bool:
    """检测产物里是否含 SEMANTIC_BLOCK（需要 LLM 兜底的占位）。

    注意：sb_fallback / await sb_xxx 是 WaAct 架构的正常运行时 LLM 兜底，
    不是失败占位。所以这里只检测明确的失败标记。
    """
    text = (code or "")
    for m in (modules or []):
        if hasattr(m, 'body_code') and m.body_code:
            text += "\n" + m.body_code
    if not text:
        return False
    # 真正表示"我无法完成"的占位
    patterns = [
        r"raise\s+NotImplementedError",
        r"^\s*pass\s*$",                      # 空函数体
        r"#\s*SEMANTIC_BLOCK.*needs?\s+repair",
        r"Auto-generated stub",
    ]
    for p in patterns:
        if re.search(p, text, re.MULTILINE):
            return True
    return False


def _parse_signature(sig: str):
    """解析 'def solve(a, b, c) -> int:' → ('solve', 'a, b, c', 'int')
    解析失败时返回 ('solve', '', '')。"""
    if not sig:
        return ("solve", "", "")
    try:
        # 标准化：把 '->' 后面的部分隔开
        m = re.match(r"^\s*def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*[^:]+)?\s*:\s*$", sig.strip(), re.DOTALL)
        if m:
            return (m.group(1), m.group(2).strip(), "")
        # ast 解析整段
        tree = ast.parse(sig.strip())
        if tree.body and isinstance(tree.body[0], ast.FunctionDef):
            fn = tree.body[0]
            params = ", ".join(a.arg for a in fn.args.args)
            return (fn.name, params, "")
    except Exception:
        pass
    return ("solve", "", "")


def _build_function_stub(signature: str, reason: str) -> str:
    """生成兼容签名的 stub，raise RuntimeError 让 eval 知道这是 raw baseline 失败。"""
    name, params, _ = _parse_signature(signature)
    params = params or "*args, **kwargs"
    return (
        f"def {name}({params}):\n"
        f"    \"\"\"Auto-generated stub: {reason}\n"
        f"    This is the expected raw baseline behavior for tasks where\n"
        f"    the IR stage could not produce executable code. The constrained\n"
        f"    repair module (L3) is responsible for fixing these cases in Ours-Full.\"\"\"\n"
        f"    raise NotImplementedError({reason!r})\n"
    )


def _wrap_as_function(code: str, modules, target_name: str, target_params: str, target_ret: str, has_semantic_block: bool):
    """包装代码为 eval 期望的 target_name(*target_params) 形式。

    策略:
    1. 如果 main_code 或 modules 里恰好有 def target_name(...) → 直接用（按需重命名）
    2. 如果有 def target_name(args) 但参数不同 → 写 wrapper
    3. 如果没有 target_name 但有 def 其他名 → 选第一个 non-helper 函数当 main，写 wrapper
    4. 如果有 SEMANTIC_BLOCK → emit stub (raise)，让 raw 干净 fail
    5. 如果什么都没有 → emit stub (raise)

    Returns: (wrapped_code, wrap_info_dict)
    """
    if has_semantic_block:
        stub = _build_function_stub(f"def {target_name}({target_params}):", "SEMANTIC_BLOCK detected in raw IR; needs L3 repair")
        return stub, {"mode": "function", "strategy": "semantic_block_stub", "target": target_name, "reason": "SEMANTIC_BLOCK"}

    if not code and not modules:
        stub = _build_function_stub(f"def {target_name}({target_params}):", "empty compiler output")
        return stub, {"mode": "function", "strategy": "empty_stub", "target": target_name}

    target_params_eff = target_params or "*args, **kwargs"
    target_sig = f"def {target_name}({target_params_eff}):"

    # 收集所有函数定义
    fn_defs = _find_function_defs(code or "")
    # 找名字匹配的
    matched = [d for d in fn_defs if d["name"] == target_name]
    if matched:
        # 名字匹配，按需调整参数
        d = matched[0]
        if d["params"].strip() == target_params_eff.strip():
            # 完全匹配，直接返回
            return code, {"mode": "function", "strategy": "direct_match", "target": target_name}
        # 名字匹配但参数不同，把原函数改名，写 wrapper
        if target_params_eff.strip() == "*args, **kwargs":
            call_expr = f"_{target_name}(*args, **kwargs)"
        else:
            call_expr = f"_{target_name}({target_params_eff})"
        wrapper = (
            f"{target_sig}\n"
            f"    \"\"\"Auto-generated wrapper: name matches, signature adapted.\"\"\"\n"
            f"    return {call_expr}\n"
        )
        renamed = _rename_def(code, target_name, f"_{target_name}")
        return renamed + "\n\n" + wrapper, {"mode": "function", "strategy": "name_match_wrapper", "target": target_name}

    # 没有同名函数，但有其他函数 → 选第一个 non-helper 函数当 main
    if fn_defs:
        main = fn_defs[0]
        main_name = main["name"]
        # 写 wrapper 调用 main_name。
        # 关键：wrapper 函数体里不能再用 *args/**kwargs，因为签名已经是固定参数列表。
        # 用显式传参或 *args/**kwargs 取决于 target_params_eff。
        if target_params_eff.strip() == "*args, **kwargs":
            call_expr = f"{main_name}(*args, **kwargs)"
        else:
            # 用 target_params_eff 的参数名列表（逗号分隔）
            call_expr = f"{main_name}({target_params_eff})"
        wrapper = (
            f"{target_sig}\n"
            f"    \"\"\"Auto-generated wrapper: orchestrator IR detected, exposing first step as solve().\n"
            f"    Original main function: {main_name}\"\"\"\n"
            f"    return {call_expr}\n"
        )
        return code + "\n\n" + wrapper, {"mode": "function", "strategy": "first_step_wrapper", "target": target_name, "main": main_name}

    # 没有 def 任何函数（极少见），emit stub
    stub = _build_function_stub(f"def {target_name}({target_params_eff}):", "no function def found in compiled code")
    return stub, {"mode": "function", "strategy": "no_def_stub", "target": target_name}


def _find_function_defs(code: str):
    """从代码字符串中提取 def <name>(<params>): 的所有顶层定义（粗略解析）。"""
    if not code:
        return []
    out = []
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                params = ", ".join(a.arg for a in node.args.args)
                out.append({"name": node.name, "params": params, "lineno": node.lineno})
    except SyntaxError:
        # ast 解析失败，fallback 到 regex
        for m in re.finditer(r"^def\s+(\w+)\s*\(([^)]*)\)\s*:", code, re.MULTILINE):
            out.append({"name": m.group(1), "params": m.group(2).strip(), "lineno": -1})
    return out


def _rename_def(code: str, old_name: str, new_name: str) -> str:
    """简单地把 def old_name( 替换为 def new_name(。"""
    return re.sub(rf"\bdef\s+{re.escape(old_name)}\s*\(", f"def {new_name}(", code)


def _has_def(code):
    if not code:
        return False
    for line in code.split("\n"):
        if line.strip().startswith("def "):
            return True
    return False


def _mute_logger():
    try:
        import logger as _l
        for fn in ['info', 'warning', 'error', 'debug']:
            setattr(_l, fn, lambda *a, **k: None)
    except Exception:
        pass


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    r = compile_prompt_to_code(prompt)
    print(json.dumps(r, ensure_ascii=False, indent=2))
