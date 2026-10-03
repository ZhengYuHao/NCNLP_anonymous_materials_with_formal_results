"""
DSL v2 自动修复引擎
基于 prompt_dslcompiler.py 的 _auto_fix_syntax_errors 实现
"""

import re
from typing import List, Tuple, Dict, Any, Optional


class AutoFixError:
    """自动修复错误信息"""
    def __init__(self, line_number: int, error_type: str, message: str, suggestion: Optional[str] = None, severity: str = "P2"):
        self.line_number = line_number
        self.error_type = error_type
        self.message = message
        self.suggestion = suggestion
        self.severity = severity

    def __str__(self):
        suggestion_text = f"\n  建议: {self.suggestion}" if self.suggestion else ""
        severity_icon = {"P0": "🔴", "P1": "🟡", "P2": "⚪"}.get(self.severity, "")
        return f"{severity_icon}[第{self.line_number}行] {self.error_type}: {self.message}{suggestion_text}"


class AutoFixEngine:
    """
    DSL 自动修复引擎

    支持的修复类型：
    - IF 缺少条件 → IF True
    - 未闭合的控制流 → 自动添加 ENDIF/ENDFOR/ENDWHILE
    - 未定义的变量 → 自动添加 DEFINE 语句
    - CALL 函数缺少实现 → 注入 invoke_function
    - 多余的空行 → 清理
    """

    KNOWN_FUNCTION_IMPLS = {
        "vector_retrieval": "await vector_db.query(collection='documents', text={query}, similarity_threshold={similarity_threshold}, top_k={top_k})",
        "vector_search": "await vector_db.query(collection='documents', text={query}, top_k={top_k})",
        "rerank_model": "await reranker.rank(query={query}, documents={docs}, top_k={top_k})",
        "hybrid_retrieval": "await hybrid_engine.retrieve(query={query})",
        "graph_database_retrieval": "await knowledge_graph.search(intent={intent}, hops={hop_count})",
        "graph_database_search": "await knowledge_graph.search(intent={intent}, hops={hop_count})",
        "graph_search": "await knowledge_graph.search(intent={intent}, hops={hops})",
        "full_search": "await full_text_search.query(query={query})",
        "default_search": "await full_text_search.query(query={query})",
        "decompose_intent": "await intent_decomposer.decompose(text={query})",
        "determine_query_type": "await classifier.determine_query_type(text={query})",
        "handle_unknown_query": "await fallback_handler.handle(query={query})",
        "return_entities_and_relations": "await knowledge_graph.extract_entities_and_relations(items={filtered_results})",
        "handle_no_match": "await fallback_handler.no_match(item={filtered_results})",
        "get_related_entities": "await knowledge_graph.get_related_entities(items={filtered_results})",
        "cache_results": "await cache.store(key={query}, ttl={duration})",
        "code_analysis_engine": "await code_analyzer.analyze(language={language}, lines={line_threshold}, target={target})",
        "static_analysis": "await code_analyzer.static_analysis(target={target})",
        "semantic_analysis": "await code_analyzer.semantic_analysis(target={target}, generate_document={generate_document})",
        "generate_explanation_document": "await doc_generator.generate(target={target})",
        "feedback_loop": "await llm.feedback_loop(stronger_model={stronger_model})",
        "strong_reasoning_model": "await reasoning_model.invoke(prompt={prompt}, strength=True)",
        "manual_review_queue": "await review_queue.enqueue(item={item})",
        "concurrent_request_handler": "await concurrency_handler.set_limit(max_concurrent={max_concurrent})",
        "sensitive_word_filter": "await content_filter.filter(text={text})",
        "identity_confirmation": "await auth.verify(identity={identity})",
        "response_time_monitor": "await monitor.observe_response_time(vector={vector_retrieval}, graph={graph_db}, code={code_analysis}, total={total})",
        "response_time_control": "await sla.set_response_time_limit(vector={vector_retrieval}, graph={graph_db}, code={code_analysis}, total={total})",
        "service_downgrade": "await ops.downgrade_service()",
        "log_management": "await log_manager.rotate(error_log={error_log}, access_log={access_log}, debug_log={debug_log})",
        "central_log_upload": "await log_uploader.upload()",
        "log_violation": "await audit.log_violation(text={text})",
        "monitoring_metrics": "await monitor.set_metrics(QPS_threshold={QPS_threshold}, response_time_threshold={response_time_threshold})",
        "alert_trigger": "await monitor.trigger_alert()",
        "scaling_trigger": "await autoscaler.trigger_scaling()",
        "trigger_alert": "await monitor.trigger_alert()",
        "trigger_scaling": "await autoscaler.trigger_scaling()",
        "compute_count_1": "await data_processor.count(entities={entities}, field={field})",
        "compute_query_type": "await intent_classifier.classify(query={query})",
        "vector_index": "await vector_db.create_index(dimension={dimension})",
        "concurrency_control": "await rate_limiter.set_limit(max_requests={max_requests})",
        "compute_user_priority": "await user_scorer.calculate_priority(user={user}, factors={factors})",
        "compute_data_source": "await data_router.select_source(query={query})",
        "logging": "await log_manager.configure(error_log={error_log}, access_log={access_log}, debug_log={debug_log})",
        "monitoring": "await monitor.set_metrics(QPS_threshold={QPS_threshold}, avg_response_time={avg_response_time}, 95th_response_time={95th_response_time}, CRH={CRH}, accuracy={accuracy})",
    }

    def __init__(self):
        """初始化自动修复引擎"""
        self.defined_vars: set = set()

    def _get_impl_for_call(self, func_name: str, args_str: str) -> str:
        """获取函数实现"""
        if func_name in self.KNOWN_FUNCTION_IMPLS:
            template = self.KNOWN_FUNCTION_IMPLS[func_name]
            arg_map = {}
            for arg in args_str.split(','):
                arg = arg.strip()
                if not arg or '=' not in arg:
                    continue
                name, value = arg.split('=', 1)
                arg_map[name.strip()] = value.strip()
            normalized_template = re.sub(r'\{\{(\w+)\}\}', r'{\1}', template)
            try:
                return normalized_template.format(**arg_map)
            except (KeyError, TypeError):
                pass

        arg_parts = []
        if args_str:
            for arg in args_str.split(','):
                arg = arg.strip()
                if not arg:
                    continue
                if '=' in arg:
                    name, value = arg.split('=', 1)
                    name = name.strip()
                    value = value.strip()
                    if not value.startswith('{{'):
                        arg_parts.append(f"{name}={value}")
                    else:
                        arg_parts.append(f"{name}={value.strip('{} ')}")
                else:
                    v = arg.strip('{} ')
                    arg_parts.append(f"{v}={v}")

        return f"await invoke_function('{func_name}'" + (", " + ", ".join(arg_parts) if arg_parts else "") + ")"

    def _infer_variable_type(self, dsl_code: str, var_name: str) -> str:
        """根据使用上下文推断变量类型"""
        var_usage = re.findall(rf'\{{\{{\s*{var_name}\s*\}}}}', dsl_code)

        if not var_usage:
            return 'Any'

        for line in dsl_code.split('\n'):
            if f'{{{var_name}}}' in line or f'{{{var_name}}}' in line:
                if any(op in line for op in ['>', '<', '>=', '<=']):
                    return 'Integer'
                if '==' in line and re.search(r'"\d+"', line):
                    return 'String'
                if 'IN' in line:
                    return 'List'

        return 'String'

    def fix(self, dsl_code: str, errors: List[Any]) -> Tuple[str, int]:
        """
        自动修复 DSL 代码中的语法错误

        Args:
            dsl_code: DSL 代码字符串
            errors: 错误列表

        Returns:
            (修复后的代码, 修复数量)
        """
        lines = dsl_code.split('\n')
        fix_count = 0
        self.defined_vars = set()

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('DEFINE'):
                match = re.match(r'DEFINE\s+\{\{(\w+)\}\}\s*:', line)
                if match:
                    var_name = match.group(1)
                    self.defined_vars.add(var_name)

        final_lines = []
        control_stack = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if final_lines and final_lines[-1].strip():
                    final_lines.append(line)
                continue

            if stripped == 'IF' or (stripped.startswith('IF ') and len(stripped) == 2):
                final_lines.append(line.replace('IF', 'IF True'))
                fix_count += 1
                continue

            if stripped == 'IF' or stripped.startswith('IF '):
                control_stack.append('IF')
            elif stripped == 'FOR' or stripped.startswith('FOR '):
                control_stack.append('FOR')
            elif stripped == 'WHILE' or stripped.startswith('WHILE '):
                control_stack.append('WHILE')
            elif stripped == 'ENDIF':
                if control_stack and control_stack[-1] == 'IF':
                    control_stack.pop()
            elif stripped == 'ENDFOR':
                if control_stack and control_stack[-1] == 'FOR':
                    control_stack.pop()
            elif stripped == 'ENDWHILE':
                if control_stack and control_stack[-1] == 'WHILE':
                    control_stack.pop()

            var_pattern = re.compile(r'\{\{(\w+)\}\}')
            for error in errors:
                error_type = getattr(error, 'error_type', 'unknown')
                message = getattr(error, 'message', '')

                if error_type == "未定义变量" or "未定义变量" in message:
                    var_match = re.search(r'\{\{(\w+)\}\}', message)
                    if var_match:
                        var_name = var_match.group(1)
                        if var_name not in self.defined_vars:
                            self.defined_vars.add(var_name)
                            var_type = self._infer_variable_type(dsl_code, var_name)
                            define_line = f'DEFINE {{{{var_name}}}}: {var_type}'
                            final_lines.insert(0, define_line)
                            fix_count += 1

            m_result = re.match(r'^\s*\{\{(\w+)\}\}\s*=\s*CALL\s+(\w+)\s*\(([^)]*)\)\s*$', line)
            if m_result:
                result_var = m_result.group(1)
                func_name = m_result.group(2)
                args_str = m_result.group(3)
                arg_parts = []
                if args_str.strip():
                    for arg in args_str.split(','):
                        arg = arg.strip()
                        if not arg:
                            continue
                        if '=' in arg:
                            name, value = arg.split('=', 1)
                            name = name.strip()
                            value = value.strip()
                            if not value.startswith('{{'):
                                value = '{{' + value.strip('{} ') + '}}'
                            arg_parts.append(f"{name}={value}")
                        else:
                            v = arg.strip('{} ')
                            arg_parts.append(f"{v}={{{{{v}}}}}")
                impl = self._get_impl_for_call(func_name, args_str)
                fixed_line = f"{{{{{result_var}}}}} = CALL {func_name}({args_str}) = {impl}"
                final_lines.append(fixed_line)
                fix_count += 1
                continue

            m_call = re.match(r'^\s*CALL\s+(\w+)\s*\(([^)]*)\)\s*$', line)
            if m_call:
                func_name = m_call.group(1)
                args_str = m_call.group(2)
                impl = self._get_impl_for_call(func_name, args_str)
                fixed_line = f"CALL {func_name}({args_str}) = {impl}"
                final_lines.append(fixed_line)
                fix_count += 1
                continue

            final_lines.append(line)

        while control_stack:
            block_type = control_stack.pop()
            if block_type == 'IF':
                final_lines.append('ENDIF')
            elif block_type == 'FOR':
                final_lines.append('ENDFOR')
            elif block_type == 'WHILE':
                final_lines.append('ENDWHILE')
            fix_count += 1

        cleaned_lines = []
        prev_empty = False
        for line in final_lines:
            if not line.strip():
                if not prev_empty:
                    cleaned_lines.append(line)
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False

        return '\n'.join(cleaned_lines), fix_count

    def analyze_errors(self, errors: List[Any]) -> Dict[str, int]:
        """分析错误严重程度"""
        analysis = {'p0_count': 0, 'p1_count': 0, 'p2_count': 0, 'total': len(errors)}

        for error in errors:
            severity = getattr(error, 'severity', 'P2')
            if severity == 'P0':
                analysis['p0_count'] += 1
            elif severity == 'P1':
                analysis['p1_count'] += 1
            else:
                analysis['p2_count'] += 1

        return analysis
