"""渲染器: (AgentSpec | ClassifiedSpec) → DSL v1.2 文本。

这一层是**完全确定性**的:给定相同 spec,输出 byte-exact 相同的 DSL。
所有"判断"都在 extractor + classifier 里完成,renderer 只做模板填充。

两个入口:
- render(spec: AgentSpec) — 旧入口,保留向后兼容
- render_from_classifiedspec(spec: ClassifiedSpec) — 新入口,StrategyClassifier 产出的直接输入
"""
from __future__ import annotations

import json
import re as _re
from io import StringIO
from typing import Any

from .schema import (
    AgentSpec,
    ExampleCase,
    InputSpec,
    OutputSpec,
    Scene,
    SemanticHook,
    SubScenario,
)

# 注意: ClassifiedSpec / SceneClassification 在 render_from_classifiedspec 内延迟导入，
# 避免 files/__init__.py → converter.py → renderer.py → dsl_v2 → compiler.py → files/converter 的循环依赖


INDENT = "    "  # 4 空格缩进(符合 DSL v1.2 语法规范)


class DSLRenderer:
    """把 (AgentSpec | ClassifiedSpec) 渲染为 DSL v1.2 源码。"""

    def render(self, spec: AgentSpec) -> str:
        buf = StringIO()
        self._write_header(buf, spec)
        buf.write("\n")
        self._write_persona(buf, spec)
        buf.write("\n")
        if spec.constraints:
            self._write_constraints(buf, spec)
            buf.write("\n")
        if spec.inputs:
            self._write_inputs(buf, spec)
            buf.write("\n")
        if spec.outputs:
            self._write_outputs(buf, spec)
            buf.write("\n")
        self._write_variable_defs(buf, spec)
        buf.write("\n")
        self._write_semantic_blocks(buf, spec)
        buf.write("\n")
        self._write_scenes(buf, spec)
        self._write_fallback(buf, spec)
        if spec.examples:
            buf.write("\n")
            self._write_examples(buf, spec)
        buf.write("\nENDAGENT\n")
        return buf.getvalue()

    # --- 元数据 ---

    def _write_header(self, buf: StringIO, spec: AgentSpec) -> None:
        buf.write(f'AGENT {{{{{spec.agent_name}}}}} "{spec.agent_description}"\n')

    def _write_persona(self, buf: StringIO, spec: AgentSpec) -> None:
        buf.write("PERSONA:\n")
        buf.write(f"{INDENT}ROLE: {spec.persona.role}\n")
        for cap in spec.persona.capabilities:
            key = self._infer_capability_key(cap)
            buf.write(f"{INDENT}{key}: {cap}\n")
        buf.write("ENDPERSONA\n")

    @staticmethod
    def _infer_capability_key(capability_text: str) -> str:
        """给每条能力推一个 UPPER_SNAKE_CASE 键。简单启发式。"""
        lowered = capability_text.lower()
        if "语义" in capability_text or "semantic" in lowered:
            return "SEMANTIC_ANALYSIS"
        if "上下文" in capability_text or "context" in lowered:
            return "CONTEXT_UNDERSTANDING"
        if "优先级" in capability_text or "priority" in lowered:
            return "PRIORITY_JUDGMENT"
        if "排除" in capability_text or "exclusion" in lowered:
            return "EXCLUSION_CHECK"
        return "CAPABILITY"

    def _write_constraints(self, buf: StringIO, spec: AgentSpec) -> None:
        buf.write("CONSTRAINTS:\n")
        for c in spec.constraints:
            buf.write(f"{INDENT}{c.key}: {c.description}\n")
        buf.write("ENDCONSTRAINTS\n")

    def _write_inputs(self, buf: StringIO, spec: AgentSpec) -> None:
        buf.write("INPUTS:\n")
        for i in spec.inputs:
            buf.write(INDENT + self._format_input(i) + "\n")
        buf.write("ENDINPUTS\n")

    @staticmethod
    def _format_input(i: InputSpec) -> str:
        req = "REQUIRED" if i.required else "OPTIONAL"
        base = f"{req} {{{{{i.name}}}}}: {i.type}"
        if i.default is not None:
            base += f" = {i.default}"
        return base

    def _write_outputs(self, buf: StringIO, spec: AgentSpec) -> None:
        buf.write("OUTPUTS:\n")
        for o in spec.outputs:
            buf.write(f"{INDENT}{{{{{o.name}}}}}: {o.type}\n")
        buf.write("ENDOUTPUTS\n")

    def _write_variable_defs(self, buf: StringIO, spec: AgentSpec) -> None:
        """为每个输出变量生成一个 DEFINE(初始化)。"""
        for o in spec.outputs:
            init = self._default_init_value(o)
            buf.write(f"DEFINE {{{{{o.name}}}}}: {o.type} = {init}\n")

    @staticmethod
    def _default_init_value(o: OutputSpec) -> str:
        return {
            "String": '""',
            "Integer": "0",
            "Float": "0.0",
            "Boolean": "False",
            "List": "[]",
            "Dict": "{}",
        }.get(o.type, '""')

    # --- 场景块 ---

    def _write_scenes(self, buf: StringIO, spec: AgentSpec) -> None:
        for scene in spec.scenes:
            self._write_scene(buf, scene, spec)
            buf.write("\n")

    def _write_scene(self, buf: StringIO, scene: Scene, spec: AgentSpec) -> None:
        buf.write(f'BLOCK {scene.block_id} "{scene.block_description}"\n')

        # 有多个子场景 → 用 ELIF 串起来
        # 只有一个子场景 → 简单 IF
        if len(scene.sub_scenarios) == 1:
            self._write_sub_scenario_as_if(
                buf, scene.sub_scenarios[0], scene, spec, indent_level=1
            )
        else:
            self._write_sub_scenarios_as_elif_chain(buf, scene, spec)

        buf.write("ENDBLOCK\n")

    def _write_sub_scenario_as_if(
        self,
        buf: StringIO,
        sub: SubScenario,
        scene: Scene,
        spec: AgentSpec,
        indent_level: int,
    ) -> None:
        """单子场景: 直接 IF(带可选 NOT CONTAINS 嵌套 IF 与 semantic)。"""
        pad = INDENT * indent_level
        user_var = self._primary_input_var(spec)

        buf.write(
            f"{pad}IF {{{{{user_var}}}}} CONTAINS "
            f"{self._render_keyword_list(sub.trigger_keywords)}\n"
        )

        inner_pad = pad + INDENT
        has_exclusion = bool(sub.exclusion_keywords)

        if has_exclusion:
            buf.write(
                f"{inner_pad}IF {{{{{user_var}}}}} NOT CONTAINS "
                f"{self._render_keyword_list(sub.exclusion_keywords)}\n"
            )
            body_pad = inner_pad + INDENT
        else:
            body_pad = inner_pad

        self._write_scene_body(buf, scene, sub, body_pad)

        if has_exclusion:
            buf.write(f"{inner_pad}ENDIF\n")
        buf.write(f"{pad}ENDIF\n")

    def _write_sub_scenarios_as_elif_chain(
        self, buf: StringIO, scene: Scene, spec: AgentSpec
    ) -> None:
        """多子场景: IF / ELIF / ELIF / ENDIF 链。
        注意: ELIF 下如果有 NOT CONTAINS,仍需嵌套 IF。
        """
        pad = INDENT
        inner_pad = pad + INDENT
        body_pad = inner_pad + INDENT
        user_var = self._primary_input_var(spec)

        for idx, sub in enumerate(scene.sub_scenarios):
            kw = "IF" if idx == 0 else "ELIF"
            buf.write(
                f"{pad}{kw} {{{{{user_var}}}}} CONTAINS "
                f"{self._render_keyword_list(sub.trigger_keywords)}\n"
            )
            if sub.exclusion_keywords:
                buf.write(
                    f"{inner_pad}IF {{{{{user_var}}}}} NOT CONTAINS "
                    f"{self._render_keyword_list(sub.exclusion_keywords)}\n"
                )
                self._write_scene_body(buf, scene, sub, body_pad)
                buf.write(f"{inner_pad}ENDIF\n")
            else:
                self._write_scene_body(buf, scene, sub, inner_pad)
        buf.write(f"{pad}ENDIF\n")

    def _write_semantic_blocks(self, buf: StringIO, spec: AgentSpec) -> None:
        """渲染所有 SEMANTIC_BLOCK 定义。"""
        collected: dict[str, SemanticHook] = {}
        for scene in spec.scenes:
            for sub in scene.sub_scenarios:
                if sub.semantic is not None:
                    fn = sub.semantic.function_name
                    if fn not in collected:
                        collected[fn] = sub.semantic

        for hook in collected.values():
            buf.write(f"SEMANTIC_BLOCK {hook.function_name}\n")
            buf.write(f"{INDENT}MODEL: {hook.model}\n")
            buf.write(f"{INDENT}TASK: {hook.task}\n")

            buf.write(f"{INDENT}INPUT:\n")
            for iv in hook.input_vars:
                buf.write(f"{INDENT*2}{iv.name}: {iv.type}\n")

            buf.write(f"{INDENT}OUTPUT:\n")
            for ov in hook.output_vars:
                buf.write(f"{INDENT*2}{ov.name}: {ov.type}\n")

            buf.write(f"{INDENT}PROMPT:\n")
            for line in hook.prompt_template.strip().splitlines():
                buf.write(f"{INDENT*2}{line}\n")
            buf.write("ENDSEMANTIC_BLOCK\n")

    def _write_scene_body(
        self,
        buf: StringIO,
        scene: Scene,
        sub: SubScenario,
        pad: str,
    ) -> None:
        """写场景命中后的赋值+RETURN(带可选 semantic 门控)。"""
        if sub.semantic is not None:
            s = sub.semantic
            args = ", ".join(f"{{{{{a}}}}}" for a in s.arguments)
            buf.write(
                f"{pad}{{{{{s.result_var}}}}} = "
                f"CALL {s.function_name}({args})\n"
            )
            buf.write(f"{pad}IF {{{{{s.result_var}}}}} {s.gate_condition}\n")
            inner = pad + INDENT
            buf.write(f"{inner}{{{{scene}}}} = {scene.scene_id}\n")
            buf.write(f"{inner}{{{{agent}}}} = {scene.agent_expression}\n")
            buf.write(f"{inner}RETURN {{{{scene}}}}, {{{{agent}}}}\n")
            buf.write(f"{pad}ENDIF\n")
        else:
            buf.write(f"{pad}{{{{scene}}}} = {scene.scene_id}\n")
            buf.write(f"{pad}{{{{agent}}}} = {scene.agent_expression}\n")
            buf.write(f"{pad}RETURN {{{{scene}}}}, {{{{agent}}}}\n")

    @staticmethod
    def _render_keyword_list(keywords: list[str]) -> str:
        parts = [f'"{k}"' for k in keywords]
        return "[" + ", ".join(parts) + "]"

    @staticmethod
    def _primary_input_var(spec: AgentSpec) -> str:
        """用作 CONTAINS 匹配的主输入变量。默认找 'user_input',找不到用第一个 required。"""
        for i in spec.inputs:
            if i.name == "user_input":
                return i.name
        for i in spec.inputs:
            if i.required:
                return i.name
        # 兜底
        return "user_input"

    # --- 兜底与示例 ---

    def _write_fallback(self, buf: StringIO, spec: AgentSpec) -> None:
        buf.write('BLOCK b_fallback "未匹配任何场景的兜底"\n')
        buf.write(f"{INDENT}{{{{scene}}}} = {spec.fallback_scene_id}\n")
        buf.write(f"{INDENT}{{{{agent}}}} = {spec.fallback_agent_expression}\n")
        buf.write(f"{INDENT}RETURN {{{{scene}}}}, {{{{agent}}}}\n")
        buf.write("ENDBLOCK\n")

    def _write_examples(self, buf: StringIO, spec: AgentSpec) -> None:
        buf.write("EXAMPLES:\n")
        for case in spec.examples:
            self._write_case(buf, case)
        buf.write("ENDEXAMPLES\n")

    def _write_case(self, buf: StringIO, case: ExampleCase) -> None:
        buf.write(f"{INDENT}CASE:\n")
        buf.write(f"{INDENT*2}INPUT: {self._fmt_dict(case.input)}\n")
        buf.write(f"{INDENT*2}EXPECTED: {self._fmt_dict(case.expected)}\n")
        path_items = ", ".join(case.execution_path)
        buf.write(f"{INDENT*2}EXECUTION_PATH: [{path_items}]\n")
        buf.write(f"{INDENT}ENDCASE\n")

    @staticmethod
    def _fmt_dict(d: dict) -> str:
        parts = []
        for k, v in d.items():
            if isinstance(v, str):
                parts.append(f'{k}: "{v}"')
            else:
                parts.append(f"{k}: {v}")
        return "{" + ", ".join(parts) + "}"

    # ====================================================================
    # 新入口: ClassifiedSpec → DSL v1.2
    # ====================================================================

    def render_from_classifiedspec(self, spec: Any, llm_call=None) -> str:
        """将 ClassifiedSpec 渲染为 DSL 1.2 文本。

        Args:
            spec: ClassifiedSpec 实例
            llm_call: 可选的 LLM 调用回调，用于 SemanticBlockRenderer
        """
        from dsl_v2.fact_types import ClassifiedSpec, SceneClassification
        from dsl_v2.semantic_renderer import SemanticBlockRenderer
        lines = []

        # AGENT 头
        lines.append(f'AGENT {{{{{spec.agent_name}}}}} "{spec.agent_description}"')

        # PERSONA
        if spec.persona_role:
            lines.append("PERSONA:")
            lines.append(f"{INDENT}ROLE: {spec.persona_role}")
            for cap in spec.persona_capabilities:
                lines.append(f"{INDENT}{self._infer_cap_key(cap)}: {cap}")
            lines.append("ENDPERSONA")

        # CONSTRAINTS
        if spec.constraints:
            lines.append("CONSTRAINTS:")
            for c in spec.constraints:
                lines.append(f"{INDENT}{c.get('key', 'CONSTRAINT')}: {c.get('description', '')}")
            lines.append("ENDCONSTRAINTS")

        # CONFIG（路径B新增：配置项）
        configs = getattr(spec, 'configs', None)
        if configs:
            lines.append("CONFIG:")
            for cfg in configs:
                lines.append(f"{INDENT}{cfg.key}: {cfg.value}")
            lines.append("ENDCONFIG")

        # POLICY（路径B新增：策略项）
        policies = getattr(spec, 'policies', None)
        if policies:
            lines.append("POLICY:")
            for pol in policies:
                policy_line = f"{INDENT}{pol.key}: {pol.description}"
                if pol.trigger_condition:
                    policy_line += f"，触发条件: {pol.trigger_condition}"
                if pol.action:
                    policy_line += f"→{pol.action}"
                lines.append(policy_line)
            lines.append("ENDPOLICY")

        # INPUTS
        if spec.inputs:
            lines.append("INPUTS:")
            for inp in spec.inputs:
                req = "REQUIRED" if inp.get("required", True) else "OPTIONAL"
                iname = inp.get("name", "")
                itype = inp.get("type", "String")
                base = f"{INDENT}{req} {{{{{iname}}}}}: {itype}"
                if inp.get("default"):
                    base += f" = {inp['default']}"
                lines.append(base)
            lines.append("ENDINPUTS")

        # OUTPUTS
        if spec.outputs:
            lines.append("OUTPUTS:")
            for out in spec.outputs:
                oname = out.get("name", "")
                otype = out.get("type", "String")
                lines.append(f"{INDENT}{{{{{oname}}}}}: {otype}")
            lines.append("ENDOUTPUTS")

        # DEFINE 变量
        for out in spec.outputs:
            init = self._default_init_val(out.get("type", "String"))
            oname = out.get("name", "")
            otype = out.get("type", "String")
            lines.append(f"DEFINE {{{{{oname}}}}}: {otype} = {init}")

        # SEMANTIC_BLOCK 定义（按 block_id 去重）——使用 SemanticBlockRenderer
        # 需要 SEMANTIC_BLOCK 的场景：路由或执行任一需要 LLM
        action_contract = self._build_action_contract(spec)
        if action_contract:
            lines.append("# ACTION_CONTRACT_JSON: " + json.dumps(action_contract, ensure_ascii=False, sort_keys=True))

        semantic_renderer = SemanticBlockRenderer(llm_call=llm_call)
        seen_semantic = set()
        for scene in spec.scenes:
            # 跳过 ABSORBED 场景
            if scene.trigger_strategy == "ABSORBED":
                continue
            # 判断是否需要 SEMANTIC_BLOCK（路由或执行任一需要LLM）
            needs_semantic = (
                scene.trigger_strategy.startswith("SEMANTIC_BLOCK") or
                scene.trigger_strategy in ("RAW_SEMANTIC", "HYBRID") or
                scene.action_strategy.startswith("SEMANTIC_BLOCK")
            )
            if needs_semantic:
                sb_id = f"sb_{scene.block_id}"
                if sb_id in seen_semantic:
                    continue  # 跳过重复 block_id 的 SEMANTIC_BLOCK
                seen_semantic.add(sb_id)
                lines.append("")
                lines.append(f"# [auto] strategy={scene.trigger_strategy}+{scene.action_strategy}")
                lines.append(f"# [auto] source={scene.source_ref}, rule={scene.meta.get('rule_matched', 'unknown')}")
                if "upgraded_from" in scene.meta:
                    lines.append(f"# [upgrade] from {scene.meta['upgraded_from']}: {scene.meta.get('upgrade_reason', '')}")
                # 使用 SemanticBlockRenderer 渲染（LLM 或降级模板）
                sb_content = semantic_renderer.render(scene)
                # SemanticBlockRenderer 已经生成了完整的 SEMANTIC_BLOCK ... ENDSEMANTIC_BLOCK
                # 缩进追加到 lines
                for sb_line in sb_content.split("\n"):
                    lines.append(sb_line)

        # BLOCK 定义（相同 block_id 合并为 IF/ELIF 链）
        lines.append("")
        from collections import defaultdict
        blocks_by_id = defaultdict(list)
        for scene in spec.scenes:
            blocks_by_id[scene.block_id].append(scene)

        for block_id, scenes in blocks_by_id.items():
            # 跳过 ABSORBED 场景（场景簇合并后标记）
            if scenes[0].trigger_strategy == "ABSORBED":
                continue

            lines.append(f"BLOCK {block_id} \"{scenes[0].block_description}\"")
            lines.append("")

            # 判断路由方式：是否使用纯语义路由
            # 语义路由/RAW_SEMANTIC 的渲染方法自带所有 ENDIF
            # HYBRID = 关键词粗筛+LLM执行，渲染方法自带内部 ENDIF，但需要外层 CONTAINS 的 ENDIF
            # 关键词路由的渲染方法不带外层 ENDIF，由这里统一加
            is_pure_semantic_routing = (
                scenes[0].trigger_strategy.startswith("SEMANTIC_BLOCK") or
                scenes[0].trigger_strategy == "RAW_SEMANTIC"
            )
            needs_outer_endif = not is_pure_semantic_routing

            if is_pure_semantic_routing:
                # 纯语义路由：渲染方法自带所有 ENDIF
                self._render_classified_scene_body(lines, scenes[0], spec, branch="IF")
            else:
                # 关键词路由/HYBRID：用 IF/ELIF 链
                for i, scene in enumerate(scenes):
                    branch = "IF" if i == 0 else "ELIF"
                    self._render_classified_scene_body(lines, scene, spec, branch=branch)
                # 关闭外层 IF/ELIF 链
                if scenes:
                    lines.append(f"{INDENT}ENDIF")
            lines.append("ENDBLOCK")
            lines.append("")

        # sb_fallback SEMANTIC_BLOCK 定义——运行时 LLM 分类兜底
        seen_fallback_ids = set()
        scene_descriptions = []
        for scene in spec.scenes:
            if scene.block_id in seen_fallback_ids:
                continue
            seen_fallback_ids.add(scene.block_id)
            scene_descriptions.append(
                f"- 场景{scene.scene_id} ({scene.block_id}): {scene.block_description}"
            )
        scene_list_text = "\n".join(scene_descriptions) if scene_descriptions else "（无已知场景）"

        fallback_prompt = (
            f"你是场景分类器。根据用户输入判断属于哪个场景。\n\n"
            f"## 已知场景列表\n{scene_list_text}\n\n"
            f"## 输出格式\n"
            f'输出严格的 JSON 格式：{{"is_match": true/false, "confidence": 0.0-1.0, "scene": 场景编号}}\n'
            f"如果用户输入不属于任何已知场景，返回 is_match=false, scene=0。"
        )

        lines.append(f'SEMANTIC_BLOCK sb_fallback "运行时LLM分类兜底"')
        lines.append(f"{INDENT}MODEL gpt-4o")
        lines.append(f"{INDENT}TEMPERATURE 0.1")
        lines.append(f"{INDENT}MAX_TOKENS 2048")
        lines.append(f"{INDENT}TASK classification")
        lines.append(f"{INDENT}INPUT:")
        lines.append(f"{INDENT*2}user_input: String")
        lines.append(f"{INDENT}OUTPUT:")
        lines.append(f"{INDENT*2}is_match: Boolean")
        lines.append(f"{INDENT*2}confidence: Float")
        lines.append(f"{INDENT*2}scene: Integer")
        lines.append(f"{INDENT}PROMPT \"\"\"")
        for p_line in fallback_prompt.split("\n"):
            lines.append(f"{INDENT*2}{p_line}")
        lines.append(f"{INDENT*2}\"\"\"")
        lines.append(f"ENDSEMANTIC_BLOCK")
        lines.append("")

        # 兜底 BLOCK——调用 sb_fallback 做运行时分类
        lines.append(f'BLOCK b_fallback "未匹配任何场景的兜底"')
        lines.append(f"{INDENT}{{{{classification}}}} = CALL sb_fallback(user_input={{{{user_input}}}})")
        lines.append(f"{INDENT}IF {{{{classification}}}}.is_match == true")
        lines.append(f"{INDENT*2}{{{{scene}}}} = {{{{classification}}}}.scene")
        fallback_agent = spec.fallback_agent_expression.strip('"')
        lines.append(f"{INDENT*2}{{{{agent}}}} = \"{fallback_agent}\"")
        lines.append(f"{INDENT*2}RETURN {{{{scene}}}}, {{{{agent}}}}")
        lines.append(f"{INDENT}ENDIF")
        lines.append(f"{INDENT}{{{{scene}}}} = {spec.fallback_scene_id}")
        lines.append(f"{INDENT}{{{{agent}}}} = {spec.fallback_agent_expression}")
        lines.append(f"{INDENT}RETURN {{{{scene}}}}, {{{{agent}}}}")
        lines.append("ENDBLOCK")

        # EXAMPLES
        if spec.examples:
            # 构建有效 block_id 集合，用于修正 execution_path 中的过期引用
            valid_block_ids = set()
            for scene in spec.scenes:
                if scene.trigger_strategy != "ABSORBED" and scene.block_id:
                    valid_block_ids.add(scene.block_id)

            # 构建旧→新 block_id 映射（处理流水线重命名、场景合并等）
            block_id_mapping = {}
            for scene in spec.scenes:
                if scene.trigger_strategy == "ABSORBED" and scene.block_id:
                    # ABSORBED 场景：被合并进 _cluster 块
                    absorbed_into = scene.meta.get("absorbed_into", "") if scene.meta else ""
                    if absorbed_into and absorbed_into in valid_block_ids:
                        block_id_mapping[scene.block_id] = absorbed_into
                meta = scene.meta or {}
                if "upgraded_from" in meta:
                    # 升级场景：从原始 block_id 升级
                    upgraded_from = meta["upgraded_from"]
                    # upgraded_from 格式: "cluster(b1_xxx, b2_yyy)" 或单个 block_id
                    cluster_match = _re.search(r'cluster\(([^)]+)\)', upgraded_from)
                    if cluster_match:
                        for old_id in cluster_match.group(1).split(","):
                            old_id = old_id.strip()
                            if old_id != scene.block_id and scene.block_id in valid_block_ids:
                                block_id_mapping[old_id] = scene.block_id
                    elif upgraded_from != scene.block_id and scene.block_id in valid_block_ids:
                        block_id_mapping[upgraded_from] = scene.block_id

            lines.append("EXAMPLES:")
            for case in spec.examples:
                inp = case.get("input", {})
                exp = case.get("expected", {})
                path = case.get("execution_path", [])
                # 修正 execution_path 中的过期 block_id
                fixed_path = []
                for p in path:
                    if p in valid_block_ids:
                        fixed_path.append(p)
                    elif p in block_id_mapping:
                        fixed_path.append(block_id_mapping[p])
                    else:
                        # 模糊匹配：尝试在有效 block_id 中找到包含原 block_id 关键部分的
                        matched = False
                        p_parts = p.split("_", 1)
                        if len(p_parts) >= 1:
                            for vid in valid_block_ids:
                                # 如果有效 block_id 包含原 block_id 的主要部分
                                if p_parts[0] == vid.split("_")[0] and vid != "b_fallback":
                                    fixed_path.append(vid)
                                    matched = True
                                    break
                        if not matched:
                            # 无法匹配，保留原始值（验证器会报告）
                            fixed_path.append(p)
                lines.append(f"{INDENT}CASE:")
                lines.append(f"{INDENT*2}INPUT: {self._fmt_dict(inp)}")
                lines.append(f"{INDENT*2}EXPECTED: {self._fmt_dict(exp)}")
                lines.append(f"{INDENT*2}EXECUTION_PATH: [{', '.join(fixed_path)}]")
                lines.append(f"{INDENT}ENDCASE")
            lines.append("ENDEXAMPLES")

        lines.append("ENDAGENT")
        return "\n".join(lines)

    @staticmethod
    def _build_action_contract(spec: Any) -> dict[str, Any]:
        """Build a deterministic action contract embedded in DSL comments."""
        scenes = []
        for scene in getattr(spec, "scenes", []) or []:
            if getattr(scene, "trigger_strategy", "") == "ABSORBED":
                continue
            action = getattr(scene, "action", None)
            structured_op = getattr(action, "structured_op", None) if action else None
            public_operations = (
                structured_op.get("public_operations", [])
                if isinstance(structured_op, dict) else []
            )
            local_program = (
                structured_op.get("local_program", [])
                if isinstance(structured_op, dict) else []
            )
            local_program_diagnostic = (
                structured_op.get("local_program_diagnostic")
                if isinstance(structured_op, dict) else None
            )
            scenes.append({
                "scene_id": getattr(scene, "scene_id", 0),
                "block_id": getattr(scene, "block_id", ""),
                "description": getattr(scene, "block_description", ""),
                "raw_requirement_text": getattr(scene, "raw_requirement_text", ""),
                "logic_flow": getattr(scene, "logic_flow", ""),
                "action_raw_text": getattr(action, "raw_text", "") if action else "",
                "action_type": getattr(action, "action_type", "") if action else "",
                "trigger_strategy": getattr(scene, "trigger_strategy", ""),
                "action_strategy": getattr(scene, "action_strategy", ""),
                "outputs": list(getattr(action, "outputs", []) or []) if action else [],
                "side_effects": list(getattr(scene, "side_effects", []) or []),
                "action_sequence": list(getattr(scene, "action_sequence", []) or []),
                "public_operations": list(public_operations or []),
                "local_program": list(local_program or []),
                "local_program_diagnostic": local_program_diagnostic,
            })
        if not scenes:
            return {}
        return {"version": 6, "source": "DSLRenderer.render_from_classifiedspec", "scenes": scenes}

    def _render_classified_scene_body(
        self, lines: list[str], scene: Any, spec: Any, branch: str = "IF"
    ) -> None:
        """渲染 BLOCK 主体（从 ClassifiedSpec）。

        核心原则：路由（trigger_strategy）和执行（action_strategy）是独立维度。
        - trigger_strategy 决定"如何进入BLOCK"（关键词匹配/LLM分类/数值比较）
        - action_strategy 决定"进入后做什么"（代码计算/LLM生成/代码赋值）
        两者独立渲染，互不覆盖。
        """
        # 判断是否需要语义路由（LLM决定入口）
        # HYBRID = 关键词粗筛 + LLM执行，不算纯语义路由
        is_semantic_routing = scene.trigger_strategy.startswith("SEMANTIC_BLOCK") or scene.trigger_strategy == "RAW_SEMANTIC"
        is_hybrid = scene.trigger_strategy == "HYBRID"
        # 判断是否需要语义执行（LLM执行动作）
        is_semantic_execution = scene.action_strategy.startswith("SEMANTIC_BLOCK")

        if is_semantic_routing and is_semantic_execution:
            # 路由+执行都是LLM：单次CALL完成路由+执行（最高效）
            self._render_semantic_routing_semantic_execution(lines, scene, branch)
        elif is_semantic_routing and not is_semantic_execution:
            # LLM路由 + 代码执行：CALL做路由，然后代码做执行
            self._render_semantic_routing_code_execution(lines, scene, branch)
        elif is_hybrid:
            # HYBRID：关键词粗筛 + LLM执行（特殊的"关键词路由+LLM执行"）
            self._render_keyword_routing_semantic_execution(lines, scene, spec, branch)
        elif not is_semantic_routing and is_semantic_execution:
            # 关键词路由 + LLM执行：IF关键词 → CALL做执行
            self._render_keyword_routing_semantic_execution(lines, scene, spec, branch)
        else:
            # 关键词路由 + 代码执行：最标准的情况
            self._render_keyword_routing_code_execution(lines, scene, spec, branch)

    # ====================================================================
    # 4种路由×执行组合的渲染方法
    # ====================================================================

    def _render_semantic_routing_semantic_execution(
        self, lines: list[str], scene: Any, branch: str = "IF"
    ) -> None:
        """路由+执行都是LLM：单次CALL完成路由+执行。"""
        sb_id = f"sb_{scene.block_id}"
        call_pad = INDENT
        lines.append(f"{call_pad}{{{{result}}}} = CALL {sb_id}(user_input={{{{user_input}}}})")
        lines.append(f"{call_pad}IF {{{{result}}}}.is_match == true")
        result_pad = call_pad + INDENT
        lines.append(f"{result_pad}{{{{scene}}}} = {{{{result}}}}.scene")
        agent_expr = scene.agent_expression.strip('"')
        lines.append(f"{result_pad}{{{{agent}}}} = \"{agent_expr}\"")
        # 输出业务字段
        business_outputs = scene.action.outputs or []
        for out_name in business_outputs:
            if out_name not in ("scene", "agent", "is_match", "confidence"):
                lines.append(f"{result_pad}{{{{{out_name}}}}} = {{{{result}}}}.{out_name}")
        # fallback 处理（如果有的话）
        if scene.fallback:
            lines.append(f"{result_pad}# fallback: {scene.fallback}")
        lines.append(f"{result_pad}RETURN {{{{scene}}}}, {{{{agent}}}}")
        lines.append(f"{call_pad}ENDIF")

    def _render_semantic_routing_code_execution(
        self, lines: list[str], scene: Any, branch: str = "IF"
    ) -> None:
        """LLM路由 + 代码执行：CALL做路由决策，然后代码做执行。"""
        sb_id = f"sb_{scene.block_id}"
        call_pad = INDENT
        lines.append(f"{call_pad}{{{{result}}}} = CALL {sb_id}(user_input={{{{user_input}}}})")
        lines.append(f"{call_pad}IF {{{{result}}}}.is_match == true")
        body_pad = call_pad + INDENT
        # 代码执行：根据 action_strategy 生成不同代码
        self._render_code_execution_body(lines, scene, body_pad)
        lines.append(f"{call_pad}ENDIF")

    def _render_keyword_routing_semantic_execution(
        self, lines: list[str], scene: Any, spec: Any, branch: str = "IF"
    ) -> None:
        """关键词路由 + LLM执行：IF关键词 → CALL做执行。"""
        user_var = self._primary_var(spec)
        keywords = scene.triggers.extracted_keywords or ["default"]
        exclusion_kw = scene.exclusions.extracted_keywords or []

        # 渲染路由条件
        self._render_keyword_condition(lines, user_var, keywords, exclusion_kw, branch)
        if exclusion_kw:
            body_pad = INDENT * 3
        else:
            body_pad = INDENT * 2

        # LLM执行
        sb_id = f"sb_{scene.block_id}"
        lines.append(f"{body_pad}{{{{result}}}} = CALL {sb_id}(user_input={{{{user_input}}}})")
        lines.append(f"{body_pad}IF {{{{result}}}}.is_match == true")
        inner = body_pad + INDENT
        lines.append(f"{inner}{{{{scene}}}} = {{{{result}}}}.scene")
        agent_expr = scene.agent_expression.strip('"')
        lines.append(f"{inner}{{{{agent}}}} = \"{agent_expr}\"")
        # 输出业务字段
        business_outputs = scene.action.outputs or []
        for out_name in business_outputs:
            if out_name not in ("scene", "agent", "is_match", "confidence"):
                lines.append(f"{inner}{{{{{out_name}}}}} = {{{{result}}}}.{out_name}")
        if scene.fallback:
            lines.append(f"{inner}# fallback: {scene.fallback}")
        lines.append(f"{inner}RETURN {{{{scene}}}}, {{{{agent}}}}")
        lines.append(f"{body_pad}ENDIF")

        # 关闭排除条件（NOT CONTAINS 的 ENDIF）
        if exclusion_kw:
            lines.append(f"{INDENT*2}ENDIF")
        # 注意：外层 CONTAINS 的 ENDIF 由 BLOCK 定义代码统一管理

    def _render_keyword_routing_code_execution(
        self, lines: list[str], scene: Any, spec: Any, branch: str = "IF"
    ) -> None:
        """关键词路由 + 代码执行：最标准的情况。"""
        user_var = self._primary_var(spec)
        keywords = scene.triggers.extracted_keywords or ["default"]
        exclusion_kw = scene.exclusions.extracted_keywords or []

        # BLOCK_COMPARE 特殊处理：数值比较路由
        if scene.trigger_strategy == "BLOCK_COMPARE":
            comp = scene.triggers.numeric_comparisons or {}
            field = comp.get("field", "unknown")
            op = comp.get("op", ">")
            val = comp.get("value", 0)
            # 标准化字段名
            from dsl_v2.pipeline import _normalize_var_name
            field_name = _normalize_var_name(field)
            if field_name == "result":
                field_name = "input_value"
            lines.append(f"{INDENT}{branch} {{{{{field_name}}}}} {op} {val}")
            # 如果有附加条件（additional_conditions），用 AND/OR 连接
            additional = comp.get("additional_conditions", [])
            for add_cond in additional:
                add_field = add_cond.get("field", "unknown")
                add_op = add_cond.get("op", ">")
                add_val = add_cond.get("value", 0)
                add_field_name = _normalize_var_name(add_field)
                if add_field_name == "result":
                    add_field_name = "input_value"
                lines.append(f"{INDENT}OR {{{{{add_field_name}}}}} {add_op} {add_val}")
            self._render_code_execution_body(lines, scene, INDENT * 2)
            lines.append(f"{INDENT}ENDIF")
            return

        # 渲染路由条件
        self._render_keyword_condition(lines, user_var, keywords, exclusion_kw, branch)
        if exclusion_kw:
            body_pad = INDENT * 3
        else:
            body_pad = INDENT * 2

        # 代码执行
        self._render_code_execution_body(lines, scene, body_pad)

        # 关闭排除条件（NOT CONTAINS 的 ENDIF）
        if exclusion_kw:
            lines.append(f"{INDENT*2}ENDIF")
        # 注意：外层 CONTAINS 的 ENDIF 由 BLOCK 定义代码统一管理

    # ====================================================================
    # 路由条件渲染
    # ====================================================================

    def _render_keyword_condition(
        self, lines: list[str], user_var: str, keywords: list[str],
        exclusion_kw: list[str], branch: str
    ) -> None:
        """渲染关键词路由条件：IF/ELIF + 可选的 NOT CONTAINS。"""
        lines.append(f"{INDENT}{branch} {{{{{user_var}}}}} CONTAINS {self._render_kw_list(keywords)}")
        if exclusion_kw:
            lines.append(f"{INDENT*2}IF {{{{{user_var}}}}} NOT CONTAINS {self._render_kw_list(exclusion_kw)}")

    # ====================================================================
    # 代码执行体渲染
    # ====================================================================

    def _render_code_execution_body(
        self, lines: list[str], scene: Any, pad: str
    ) -> None:
        """根据 action_strategy 渲染代码执行体。"""
        if scene.action_strategy == "CODE_COMPUTE":
            self._render_compute_block(lines, scene, pad)
        elif scene.action_strategy == "CODE_CALL_FUNC":
            self._render_lookup_block(lines, scene, pad)
        elif scene.action_strategy == "CODE_ASSIGN":
            self._render_assign_block(lines, scene, pad)
        elif scene.action_strategy == "CODE_TEMPLATE":
            self._render_template_block(lines, scene, pad)
        else:
            # 兜底：简单赋值
            lines.append(f"{pad}{{{{scene}}}} = {scene.scene_id}")
            agent_expr = scene.agent_expression.strip('"')
            lines.append(f"{pad}{{{{agent}}}} = \"{agent_expr}\"")
            lines.append(f"{pad}RETURN {{{{scene}}}}, {{{{agent}}}}")

    def _render_assign_block(
        self, lines: list[str], scene: Any, pad: str
    ) -> None:
        """渲染赋值类BLOCK——简单赋值。"""
        lines.append(f"{pad}{{{{scene}}}} = {scene.scene_id}")
        agent_expr = scene.agent_expression
        if not agent_expr.startswith("{{"):
            agent_expr = agent_expr.strip('"')
            agent_expr = f'"{agent_expr}"'
        lines.append(f"{pad}{{{{agent}}}} = {agent_expr}")
        # 如果有业务输出字段，在BLOCK内直接赋值
        business_outputs = scene.action.outputs or []
        for out_name in business_outputs:
            if out_name not in ("scene", "agent"):
                out_val = self._infer_code_value(out_name, scene)
                lines.append(f"{pad}{{{{{out_name}}}}} = {out_val}")
        lines.append(f"{pad}RETURN {{{{scene}}}}, {{{{agent}}}}")



    def _render_compute_block(self, lines: list[str], scene: Any, pad: str) -> None:
        """渲染计算类BLOCK——从原文/嵌套逻辑提取规则表，生成 IF/ELIF 代码。"""
        import re
        raw_text = scene.raw_requirement_text or scene.block_description or ""

        # 确定比较变量名和输出变量名（通用推断，不硬编码领域字段）
        compare_var, output_var = self._infer_compute_vars(scene)

        # 优先从 nested_logic 结构生成代码（更精确）
        if scene.nested_logic is not None:
            self._render_nested_logic(lines, scene.nested_logic, compare_var, output_var, pad)
        else:
            # 从原文提取规则表（通用模式）
            rules = self._extract_rules_from_text(raw_text)
            if rules:
                self._render_rules_table(lines, rules, compare_var, output_var, pad)
            else:
                # 无规则表，使用通用计算
                business_outputs = scene.action.outputs or []
                for out_name in business_outputs:
                    if out_name not in ("scene", "agent"):
                        lines.append(f"{pad}{{{{{out_name}}}}} = computed_value  # 需要根据业务逻辑补充")

        # fallback 处理
        if scene.fallback:
            lines.append(f"{pad}# fallback: {scene.fallback}")

        lines.append(f"{pad}{{{{scene}}}} = {scene.scene_id}")
        agent_expr = scene.agent_expression.strip('"')
        lines.append(f"{pad}{{{{agent}}}} = \"{agent_expr}\"")
        lines.append(f"{pad}RETURN {{{{scene}}}}, {{{{agent}}}}")

    def _render_nested_logic(self, lines, logic: dict, compare_var: str, output_var: str, pad: str, depth: int = 0) -> None:
        """递归渲染嵌套条件逻辑树。"""
        if not isinstance(logic, dict):
            return
        condition = logic.get("condition", "")
        then_branch = logic.get("then")
        else_branch = logic.get("else")

        # 解析条件为 DSL 比较表达式
        cond_expr = self._parse_condition_to_dsl(condition, compare_var)
        kw = "IF" if depth == 0 else "IF"

        lines.append(f"{pad}{kw} {cond_expr}")
        inner_pad = pad + INDENT

        if isinstance(then_branch, dict) and "condition" in then_branch:
            # 嵌套 then 分支
            self._render_nested_logic(lines, then_branch, compare_var, output_var, inner_pad, depth + 1)
        elif isinstance(then_branch, dict) and "value" in then_branch:
            lines.append(f"{inner_pad}{{{{{output_var}}}}} = {then_branch['value']}")
        elif then_branch:
            lines.append(f"{inner_pad}{{{{{output_var}}}}} = {then_branch}")

        if else_branch is not None:
            if isinstance(else_branch, dict) and "condition" in else_branch:
                lines.append(f"{pad}ELSE")
                self._render_nested_logic(lines, else_branch, compare_var, output_var, inner_pad, depth + 1)
            elif isinstance(else_branch, dict) and "value" in else_branch:
                lines.append(f"{pad}ELSE")
                lines.append(f"{inner_pad}{{{{{output_var}}}}} = {else_branch['value']}")
            elif else_branch:
                lines.append(f"{pad}ELSE")
                lines.append(f"{inner_pad}{{{{{output_var}}}}} = {else_branch}")

        lines.append(f"{pad}ENDIF")

    def _parse_condition_to_dsl(self, condition: str, compare_var: str) -> str:
        """将自然语言条件解析为 DSL 比较表达式。

        支持复合条件（或/且连接词）和中文阈值表达式。
        """
        import re

        # 处理复合条件：用"或"连接的多个子条件
        if "或" in condition:
            sub_conditions = condition.split("或")
            sub_exprs = []
            for sub in sub_conditions:
                sub = sub.strip()
                if sub:
                    sub_exprs.append(self._parse_condition_to_dsl(sub, compare_var))
            if sub_exprs:
                return " OR ".join(sub_exprs)

        # 处理复合条件：用"且"/"和"连接的多个子条件
        if "且" in condition or ("和" in condition and not any(
            kw in condition for kw in ["折", "元", "块", "个", "件"]
        )):
            connector = "且" if "且" in condition else "和"
            sub_conditions = condition.split(connector)
            sub_exprs = []
            for sub in sub_conditions:
                sub = sub.strip()
                if sub:
                    sub_exprs.append(self._parse_condition_to_dsl(sub, compare_var))
            if sub_exprs:
                return " AND ".join(sub_exprs)

        # "X == Y" / "X 等于 Y" / "X 是 Y"
        m = re.match(r'(.+?)\s*(==|等于|是)\s*(.+)', condition)
        if m:
            var = m.group(1).strip()
            val = m.group(3).strip().strip('"').strip("'")
            var_name = self._condition_var_to_name(var, compare_var)
            return f'{{{{{var_name}}}}} == "{val}"'

        # "X >= Y" / "X 大于等于 Y" / "X ≥ Y" / "X 不小于 Y"
        m = re.match(r'(.+?)\s*(>=|≥|大于等于?|不小于|以上)\s*(.+)', condition)
        if m:
            var = m.group(1).strip()
            val = self._parse_numeric_value(m.group(3).strip())
            var_name = self._condition_var_to_name(var, compare_var)
            return f'{{{{{var_name}}}}} >= {val}'

        # "X > Y" / "X 大于 Y" / "X 超过 Y"
        m = re.match(r'(.+?)\s*(>|大于|超过)\s*(.+)', condition)
        if m:
            var = m.group(1).strip()
            val = self._parse_numeric_value(m.group(3).strip())
            var_name = self._condition_var_to_name(var, compare_var)
            return f'{{{{{var_name}}}}} > {val}'

        # "X <= Y" / "X 小于等于 Y" / "X ≤ Y" / "X 不大于 Y" / "X 以下"
        m = re.match(r'(.+?)\s*(<=|≤|小于等于?|不大于|以下)\s*(.+)', condition)
        if m:
            var = m.group(1).strip()
            val = self._parse_numeric_value(m.group(3).strip())
            var_name = self._condition_var_to_name(var, compare_var)
            return f'{{{{{var_name}}}}} <= {val}'

        # "X < Y" / "X 小于 Y" / "X 低于 Y"
        m = re.match(r'(.+?)\s*(<|小于|低于)\s*(.+)', condition)
        if m:
            var = m.group(1).strip()
            val = self._parse_numeric_value(m.group(3).strip())
            var_name = self._condition_var_to_name(var, compare_var)
            return f'{{{{{var_name}}}}} < {val}'

        # 默认：用原始条件字符串
        return f'{{{{{compare_var}}}}} == "{condition}"'

    @staticmethod
    def _condition_var_to_name(var_text: str, default_var: str) -> str:
        """将条件中的变量文本转为DSL变量名。"""
        from dsl_v2.pipeline import _normalize_var_name
        if not var_text or var_text in ("条件", "值", "输入"):
            return default_var
        normalized = _normalize_var_name(var_text)
        if normalized and normalized != "result":
            return normalized
        return default_var

    @staticmethod
    def _parse_numeric_value(val_text: str) -> str:
        """解析中文数值表达式为DSL数值。

        例如：
        - "月均1次" → "1"
        - "30%" → "0.3"
        - "100" → "100"
        - "0.85" → "0.85"
        """
        import re
        val_text = val_text.strip().rstrip('）)]')

        # 百分比："30%" → 0.3
        m = re.match(r'([\d.]+)\s*%', val_text)
        if m:
            return str(float(m.group(1)) / 100)

        # "月均X次"/"年均X次"/"日均X次" → X
        m = re.match(r'(?:月均|年均|日均|平均)\s*([\d.]+)\s*(?:次|件|单|笔)', val_text)
        if m:
            return m.group(1)

        # 纯数字
        m = re.match(r'([\d.]+)', val_text)
        if m:
            return m.group(1)

        # 无法解析，返回原始文本（加引号作为字符串比较）
        return f'"{val_text}"'

    def _render_rules_table(self, lines, rules: list, compare_var: str, output_var: str, pad: str) -> None:
        """渲染规则表为 IF/ELIF/ELSE 代码块。

        rules 元素可以是：
        - 二元组 (condition, value) — 枚举映射，如 ("VIP", 0.8)
        - 三元组 (field, op, value) — 数值比较，如 ("购买频率", ">", "1")
        """
        has_numeric = any(len(r) == 3 for r in rules)

        if has_numeric:
            # 数值比较条件：每个条件独立渲染
            for i, rule in enumerate(rules):
                kw = "IF" if i == 0 else "ELIF"
                if len(rule) == 3:
                    # 三元组 (field, op, value) — 数值比较
                    field_raw, op, value = rule
                    from dsl_v2.pipeline import _normalize_var_name
                    var_name = _normalize_var_name(field_raw)
                    if var_name == "result":
                        var_name = compare_var
                    lines.append(f"{pad}{kw} {{{{{var_name}}}}} {op} {value}")
                    lines.append(f"{pad}{INDENT}{{{{{output_var}}}}} = True  # 满足条件")
                elif len(rule) == 2:
                    # 二元组 (condition, value) — 枚举映射
                    lines.append(f"{pad}{kw} {{{{{compare_var}}}}} == \"{rule[0]}\"")
                    lines.append(f"{pad}{INDENT}{{{{{output_var}}}}} = {rule[1]}")
            lines.append(f"{pad}ELSE")
            lines.append(f"{pad}{INDENT}{{{{{output_var}}}}} = False  # 默认不满足")
            lines.append(f"{pad}ENDIF")
        else:
            # 纯枚举映射规则
            for i, (condition, value) in enumerate(rules):
                kw = "IF" if i == 0 else "ELIF"
                lines.append(f"{pad}{kw} {{{{{compare_var}}}}} == \"{condition}\"")
                lines.append(f"{pad}{INDENT}{{{{{output_var}}}}} = {value}")
            lines.append(f"{pad}ELSE")
            lines.append(f"{pad}{INDENT}{{{{{output_var}}}}} = 0  # 默认值")
            lines.append(f"{pad}ENDIF")

    @staticmethod
    def _infer_compute_vars(scene: Any) -> tuple[str, str]:
        """推断计算块的比较变量名和输出变量名（通用推断）。"""
        raw_text = scene.raw_requirement_text or scene.block_description or ""
        business_outputs = scene.action.outputs or []

        # 输出变量：优先从 action.outputs 获取
        output_var = "result"
        for out_name in business_outputs:
            if out_name not in ("scene", "agent"):
                output_var = out_name
                break

        # 比较变量：从原文推断
        import re
        # 模式："根据X计算Y" → 比较变量是X
        m = re.search(r'根据(.+?)(?:计算|确定|判断)', raw_text)
        if m:
            from dsl_v2.pipeline import _normalize_var_name
            compare_var = _normalize_var_name(m.group(1).strip())
            return compare_var, output_var

        # 模式："X打Y折" → 比较变量推断（不硬编码member_level）
        if re.search(r'\S+?打\d+折', raw_text):
            # 从原文提取"X打Y折"中的X部分
            m = re.search(r'(\S+?)打\d+折', raw_text)
            if m:
                from dsl_v2.pipeline import _normalize_var_name
                first_level = m.group(1).strip().rstrip('，,')
                # 推断比较变量名（通常是等级、类别等）
                return _normalize_var_name(first_level) if first_level else "level", output_var
            return "level", output_var

        # 兜底
        return "input_value", output_var

    def _render_lookup_block(self, lines: list[str], scene: Any, pad: str) -> None:
        """渲染查询类BLOCK——生成函数调用代码。"""
        raw_text = scene.raw_requirement_text or scene.block_description or ""
        business_outputs = scene.action.outputs or []

        # 通用推断函数名（从动作描述提取动词+名词）
        func_name = self._infer_lookup_func_generic(raw_text)

        # 生成 CALL 语句
        if business_outputs:
            out_names = [f"{{{{{n}}}}}" for n in business_outputs if n not in ("scene", "agent")]
            if out_names:
                lhs = ", ".join(out_names)
                lines.append(f"{pad}{lhs} = CALL {func_name}(user_input={{{{user_input}}}})")
            else:
                lines.append(f"{pad}{{{{query_result}}}} = CALL {func_name}(user_input={{{{user_input}}}})")
        else:
            lines.append(f"{pad}{{{{query_result}}}} = CALL {func_name}(user_input={{{{user_input}}}})")

        # fallback 处理
        if scene.fallback:
            lines.append(f"{pad}# fallback: {scene.fallback}")

        lines.append(f"{pad}{{{{scene}}}} = {scene.scene_id}")
        agent_expr = scene.agent_expression.strip('"')
        lines.append(f"{pad}{{{{agent}}}} = \"{agent_expr}\"")
        lines.append(f"{pad}RETURN {{{{scene}}}}, {{{{agent}}}}")

    def _render_template_block(self, lines: list[str], scene: Any, pad: str) -> None:
        """Render the routing body for a template action.

        Template execution belongs to the typed local action in
        ACTION_CONTRACT_JSON.  Inventing ``template_<output>`` here silently
        introduced an undeclared input and made the router disagree with the
        executable workflow plan.
        """
        lines.append(f"{pad}# template_fill is executed by the compiled local action contract")
        lines.append(f"{pad}{{{{scene}}}} = {scene.scene_id}")
        agent_expr = scene.agent_expression.strip('"')
        lines.append(f"{pad}{{{{agent}}}} = \"{agent_expr}\"")
        lines.append(f"{pad}RETURN {{{{scene}}}}, {{{{agent}}}}")

    @staticmethod
    def _infer_lookup_func_generic(text: str) -> str:
        """通用推断查询函数名——从动作描述提取，不硬编码领域字段。"""
        import re
        # 模式："查询X"/"获取X"/"检索X" → query_xxx
        m = re.search(r'(?:查询|获取|检索|调用)([\w]{1,8})', text)
        if m:
            from dsl_v2.pipeline import _normalize_var_name
            return f"query_{_normalize_var_name(m.group(1))}"
        # 兜底
        return "query_system"

    @staticmethod
    def _extract_rules_from_text(text: str) -> list:
        """通用的规则表提取——支持多种自然语言规则表达。"""
        import re
        rules = []

        # 中文标点排除列表——防止"折扣：VIP"被匹配为"折扣：VIP"而非"VIP"
        _CN_PUNCT = r'：:。.！!？?（）()《》【】\[\]""\"\"\'\u3000'

        # 模式1："A打X折"（折扣类）——A是等级名称
        # 关键修复：排除中文标点，避免"折扣：VIP"被整体匹配为level
        for match in re.finditer(rf'([^\s，,；;{_CN_PUNCT}]+?)打(\d+)折', text):
            level = match.group(1).strip().rstrip('，,')
            discount_num = int(match.group(2))
            # 中文折扣：8折=0.8, 75折=0.75, 85折=0.85
            rate = discount_num / 100 if discount_num > 10 else discount_num / 10
            rules.append((level, rate))

        if rules:
            return rules

        # 模式4（优先于模式2）：数值比较条件
        # 提取结构化比较条件，用于生成 BLOCK_COMPARE 代码
        # 支持多种中文数值表达式：
        #   "购买频率<月均1次" → field=购买频率, op=<, value=1
        #   "退货率>30%" → field=退货率, op=>, value=0.3
        #   "频率<1" → field=频率, op=<, value=1
        numeric_rules = []

        # 模式4a：中文字段 <中文前缀+数值>，如 "购买频率<月均1次"
        for match in re.finditer(
            r'([\w\u4e00-\u9fff]+?)\s*([<>＞＜≥≤>=<]+|大于等于?|小于等于?|超过|低于|不小于|不大于|以上|以下)\s*(?:月均|年均|日均|平均)?\s*([\d.]+)\s*(?:次|件|单|笔|%)?',
            text
        ):
            field_raw = match.group(1).strip()
            # 去掉前缀的连接词"或"/"且"/"和"
            field_raw = re.sub(r'^(或|且|和)', '', field_raw)
            op_raw = match.group(2).strip()
            value_raw = match.group(3).strip()
            # 检查是否有百分比后缀
            full_match = match.group(0)
            is_percent = '%' in full_match[full_match.index(value_raw):]

            # 标准化操作符
            op_map = {
                ">": ">", "<": "<", "＞": ">", "＜": "<",
                ">=": ">=", "<=": "<=", "≥": ">=", "≤": "<=",
                "大于": ">", "大于等于": ">=", "超过": ">", "以上": ">=",
                "小于": "<", "小于等于": "<=", "低于": "<", "不小于": ">=", "不大于": "<=",
                "以下": "<=",
            }
            op = op_map.get(op_raw, op_raw)

            # 处理百分比：30% → 0.3
            if is_percent:
                try:
                    val = float(value_raw) / 100
                    value_str = str(val)
                except ValueError:
                    value_str = value_raw
            else:
                value_str = value_raw

            numeric_rules.append((field_raw, op, value_str))

        if numeric_rules:
            return numeric_rules  # 返回三元组 (field, op, value) 表示数值比较条件

        # 模式2："A是X, B是Y"（映射类）——排除中文标点
        for match in re.finditer(rf'([^\s，,；;{_CN_PUNCT}]+?)[=→是]\s*"?([^，,；;"\s]+)"?\s*[，,；;]', text):
            rules.append((match.group(1), f'"{match.group(2)}"'))

        if rules:
            return rules

        # 模式3："如果是X，则Y"（条件赋值类）
        for match in re.finditer(r'如果(.+?)[，,]则(.+?)[。，;]', text):
            rules.append((match.group(1).strip(), match.group(2).strip()))

        return rules



    @staticmethod
    def _infer_code_value(field_name: str, scene: Any) -> str:
        """推断代码赋值块中字段的值。"""
        raw = scene.raw_requirement_text or scene.block_description or ""
        import re
        # 从原文提取优惠券代码
        coupon_match = re.search(r'([A-Z]+\d+)', raw)
        if "coupon" in field_name.lower() or "优惠券" in raw:
            return f'"{coupon_match.group(1)}"' if coupon_match else '"COUPON"'
        # 布尔值推断
        if field_name.startswith("is_"):
            return "True"
        return '""'

    @staticmethod
    def _primary_var(spec: Any) -> str:
        """从 ClassifiedSpec 获取主输入变量名。"""
        for inp in spec.inputs:
            if inp.get("name") == "user_input":
                return "user_input"
        return spec.inputs[0]["name"] if spec.inputs else "user_input"

    @staticmethod
    def _render_kw_list(keywords: list[str]) -> str:
        parts = [f'"{kw}"' for kw in keywords]
        return "[" + ", ".join(parts) + "]"

    @staticmethod
    def _default_init_val(type_name: str) -> str:
        return {
            "String": '""', "Integer": "0", "Float": "0.0",
            "Boolean": "False", "List": "[]", "Dict": "{}",
        }.get(type_name, '""')

    @staticmethod
    def _infer_cap_key(text: str) -> str:
        lowered = text.lower()
        if "语义" in text or "semantic" in lowered:
            return "SEMANTIC_ANALYSIS"
        if "上下文" in text or "context" in lowered:
            return "CONTEXT_UNDERSTANDING"
        if "优先级" in text or "priority" in lowered:
            return "PRIORITY_JUDGMENT"
        if "排除" in text or "exclusion" in lowered:
            return "EXCLUSION_CHECK"
        return "CAPABILITY"
