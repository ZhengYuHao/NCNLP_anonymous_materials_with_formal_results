"""SemanticBlockRenderer —— 使用 LLM 渲染 SEMANTIC_BLOCK 内容。

核心设计：
1. 输入：SceneClassification（含 raw_requirement_text + logic_flow + side_effects）+ DSL 语法规范
2. LLM 渲染：根据原文+语法规范生成完整的 SEMANTIC_BLOCK 定义
3. 降级模式：无 LLM 时使用模板生成（基于 raw_requirement_text）
4. 代码验证：对 LLM 输出做语法校验

通用性保证：
- LLM 输入是"原文+语法规范"，不是特定场景的模板
- 和具体例子无关，能处理任何类型的复杂逻辑
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from .fact_types import SceneClassification

logger = logging.getLogger(__name__)

# DSL v1.2 语法规范片段（传给 LLM）
DSL_GRAMMAR_SPEC = """
SEMANTIC_BLOCK <block_id> "<描述>"
    MODEL <model_name>
    [ TEMPERATURE <float> ]
    [ MAX_TOKENS <int> ]
    TASK <task_type>     # classification | generation | scoring | similarity | extraction
    INPUT:
        <var_name>: <type>   # String | Integer | Float | Boolean | List | Dict
    OUTPUT:
        <var_name>: <type>
    PROMPT "<prompt内容>"
ENDSEMANTIC_BLOCK

TASK 类型说明:
- classification: 意图分类 (temperature 0.1-0.3, 输出 {intent, confidence})
- generation: 内容生成 (temperature 0.5-0.7, 输出 {result})
- scoring: 多维度评分 (temperature 0.1-0.3, 输出 {score_a, score_b, ...})
- similarity: 语义相似度 (temperature 0.1, 输出 {similarity})
- extraction: 信息抽取 (temperature 0.1-0.3, 输出 {field, ...})
"""

# LLM 渲染 Prompt
SEMANTIC_RENDER_SYSTEM_PROMPT = """你是一个 DSL 编译器。你的任务是根据原始需求文本生成 SEMANTIC_BLOCK 定义。

## DSL v1.2 语法规范

{grammar_spec}

## 生成规则

1. **忠实原文**：PROMPT 内容必须覆盖原始需求文本中的所有逻辑
2. **结构化输出**：OUTPUT 变量必须包含后续 BLOCK 分流所需的字段（至少 is_match/scene）
3. **实用性**：PROMPT 要具体、可执行，不要泛泛而谈
4. **语法正确**：严格遵循上面的语法规范

## 输出格式

输出 JSON 格式，包含以下字段：
```json
{{
    "task": "classification|generation|scoring|similarity|extraction",
    "input_vars": [{{"name": "var_name", "type": "String"}}],
    "output_vars": [{{"name": "var_name", "type": "Boolean"}}, ...],
    "prompt_template": "完整的 prompt 内容..."
}}
```

仅输出 JSON，不要有其他文字。"""

SEMANTIC_RENDER_USER_TEMPLATE = """请为以下场景生成 SEMANTIC_BLOCK 定义：

## 场景信息
- block_id: sb_{block_id}
- 描述: {block_description}
- 触发策略: {trigger_strategy}
- 动作策略: {action_strategy}

## 原始需求文本（最重要的输入）
{raw_requirement_text}

## 逻辑流描述
{logic_flow}

## 副作用
{side_effects}

## 扩展信息
{extension_info}

现在输出 JSON:"""


def _default_llm_call(prompt: str) -> str:
    """默认 LLM 调用——不调用真实 LLM，返回 None 触发降级。"""
    return None


def _fallback_render(scene: SceneClassification) -> str:
    """降级模板渲染——不使用 LLM，基于 raw_requirement_text 生成 SEMANTIC_BLOCK。

    路径A升级：从"分类器"模式升级为"场景执行器"模式。
    - 如果 action.outputs 有业务输出字段，则生成执行器式 SEMANTIC_BLOCK
    - 否则回退到分类器模式
    """
    sb_id = f"sb_{scene.block_id}"
    block_desc = scene.block_description or "未知场景"

    # 根据 trigger_strategy 推断 task 类型
    task = "classification"
    if "EXTRACTION" in scene.trigger_strategy:
        task = "extraction"
    elif "SCORING" in scene.action_strategy:
        task = "scoring"
    elif "GENERATION" in scene.action_strategy:
        task = "generation"
    elif "SIMILARITY" in scene.action_strategy:
        task = "similarity"

    # 构建 prompt——使用原文信息
    raw_text = scene.raw_requirement_text or block_desc
    logic = scene.logic_flow or ""
    effects = scene.side_effects or []

    # ================================================================
    # RAW_SEMANTIC 模式：整段原文透传给 LLM
    # ================================================================
    is_raw_semantic = scene.trigger_strategy == "RAW_SEMANTIC"

    # ================================================================
    # 路径A核心：生成业务输出字段
    # 从 action.outputs 推导，如果为空则使用默认分类输出
    # ================================================================
    business_outputs = scene.action.outputs or []
    has_business_outputs = len(business_outputs) > 0 and business_outputs != ["scene", "agent"]

    if is_raw_semantic:
        # RAW_SEMANTIC 模式：完整原文透传，LLM 拿到完整上下文
        prompt_parts = [
            f"你是场景执行器。请基于以下完整规范处理用户输入。",
            f"",
            f"## 完整规范原文",
            f"{raw_text}",
        ]
        if scene.raw_context:
            prompt_parts.append(f"\n## 相关上下文\n{scene.raw_context}")
        if scene.preconditions:
            prompt_parts.append(f"\n## 前置条件\n" + "\n".join(f"- {p}" for p in scene.preconditions))
        if scene.fallback:
            prompt_parts.append(f"\n## 回退逻辑\n{scene.fallback}")
        if scene.action_sequence:
            prompt_parts.append(f"\n## 执行步骤\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(scene.action_sequence)))
        if scene.nested_logic:
            prompt_parts.append(f"\n## 嵌套条件\n{json.dumps(scene.nested_logic, ensure_ascii=False, indent=2)}")
        if effects:
            prompt_parts.append(f"\n## 副作用\n" + "\n".join(f"- {e}" for e in effects))

        # OUTPUT 字段
        output_vars = [
            {"name": "is_match", "type": "Boolean"},
            {"name": "confidence", "type": "Float"},
            {"name": "scene", "type": "Integer"},
        ]
        for out_name in business_outputs:
            if out_name not in ("scene", "agent", "is_match", "confidence"):
                output_vars.append({"name": out_name, "type": _infer_output_type(out_name)})

        json_fields = {"is_match": "true/false", "confidence": "0.0-1.0", "scene": scene.scene_id}
        for out_name in business_outputs:
            if out_name not in json_fields:
                json_fields[out_name] = _infer_json_example(out_name)
        json_example = json.dumps(json_fields, ensure_ascii=False)

        prompt_parts.extend([
            f"",
            f"## 输出格式",
            f"输出严格的 JSON 格式：",
            f"{json_example}",
            f"如果不属于该场景，is_match 返回 false。",
        ])

        prompt_text = "\n".join(prompt_parts)

        lines = [
            f'SEMANTIC_BLOCK {sb_id} "{block_desc}"',
            f'    MODEL gpt-4o',
            f'    TEMPERATURE 0.1',
            f'    MAX_TOKENS 2048',
            f'    TASK {task}',
            f'    INPUT:',
            f'        user_input: String',
            f'    OUTPUT:',
        ]
        for ov in output_vars:
            lines.append(f'        {ov["name"]}: {ov["type"]}')
        lines.append(f'    PROMPT """')
        for p_line in prompt_text.split("\n"):
            lines.append(f"        {p_line}")
        lines.append('    """')
        lines.append("ENDSEMANTIC_BLOCK")

        return "\n".join(lines)

    if has_business_outputs:
        # 执行器模式：PROMPT 要求 LLM 执行业务逻辑并返回结果
        prompt_parts = [
            f"你是场景执行器。根据用户输入执行\"{block_desc}\"的业务逻辑。",
            f"",
            f"## 原始需求",
            f"{raw_text}",
        ]
        if scene.raw_context:
            prompt_parts.append(f"\n## 上下文（相邻段落）\n{scene.raw_context}")
        if scene.preconditions:
            prompt_parts.append(f"\n## 前置条件\n" + "\n".join(f"- {p}" for p in scene.preconditions))
        if logic:
            prompt_parts.append(f"\n## 执行逻辑\n{logic}")
        if scene.fallback:
            prompt_parts.append(f"\n## 回退逻辑\n{scene.fallback}")
        if scene.action_sequence:
            prompt_parts.append(f"\n## 执行步骤\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(scene.action_sequence)))
        if scene.nested_logic:
            prompt_parts.append(f"\n## 嵌套条件\n{json.dumps(scene.nested_logic, ensure_ascii=False, indent=2)}")
        if effects:
            prompt_parts.append(f"\n## 副作用\n" + "\n".join(f"- {e}" for e in effects))

        # 构建 OUTPUT 字段列表：分类字段 + 业务输出字段
        output_vars = [
            {"name": "is_match", "type": "Boolean"},
            {"name": "confidence", "type": "Float"},
            {"name": "scene", "type": "Integer"},
        ]
        # 推断业务输出字段的类型
        for out_name in business_outputs:
            if out_name in ("scene", "agent", "is_match", "confidence"):
                continue  # 已包含
            out_type = _infer_output_type(out_name)
            output_vars.append({"name": out_name, "type": out_type})

        # 构建 JSON 输出格式示例
        json_fields = {"is_match": "true/false", "confidence": "0.0-1.0", "scene": scene.scene_id}
        for out_name in business_outputs:
            if out_name not in json_fields:
                json_fields[out_name] = _infer_json_example(out_name)
        json_example = json.dumps(json_fields, ensure_ascii=False)

        prompt_parts.extend([
            f"",
            f"## 输出格式",
            f"输出严格的 JSON 格式：",
            f"{json_example}",
        ])

        prompt_text = "\n".join(prompt_parts)

        lines = [
            f'SEMANTIC_BLOCK {sb_id} "{block_desc}"',
            f'    MODEL gpt-4o',
            f'    TEMPERATURE 0.1',
            f'    MAX_TOKENS 2048',
            f'    TASK {task}',
            f'    INPUT:',
            f'        user_input: String',
            f'    OUTPUT:',
        ]
        for ov in output_vars:
            lines.append(f'        {ov["name"]}: {ov["type"]}')
        lines.append(f'    PROMPT """')
        for p_line in prompt_text.split("\n"):
            lines.append(f"        {p_line}")
        lines.append('    """')
        lines.append("ENDSEMANTIC_BLOCK")

    else:
        # 分类器模式（保持原有行为）
        prompt_parts = [
            f"你是场景分类器。判断用户输入是否属于\"{block_desc}\"场景。",
            f"",
            f"## 原始需求",
            f"{raw_text}",
        ]
        if scene.raw_context:
            prompt_parts.append(f"\n## 上下文（相邻段落）\n{scene.raw_context}")
        if scene.preconditions:
            prompt_parts.append(f"\n## 前置条件\n" + "\n".join(f"- {p}" for p in scene.preconditions))
        if logic:
            prompt_parts.append(f"\n## 逻辑流\n{logic}")
        if scene.fallback:
            prompt_parts.append(f"\n## 回退逻辑\n{scene.fallback}")
        if scene.action_sequence:
            prompt_parts.append(f"\n## 执行步骤\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(scene.action_sequence)))
        if scene.nested_logic:
            prompt_parts.append(f"\n## 嵌套条件\n{json.dumps(scene.nested_logic, ensure_ascii=False, indent=2)}")
        if effects:
            prompt_parts.append(f"\n## 副作用\n" + "\n".join(f"- {e}" for e in effects))

        prompt_parts.extend([
            f"",
            f"## 输出格式",
            f"输出严格的 JSON 格式：",
            f'{{"is_match": true/false, "confidence": 0.0-1.0, "scene": {scene.scene_id}}}',
        ])

        prompt_text = "\n".join(prompt_parts)

        lines = [
            f'SEMANTIC_BLOCK {sb_id} "{block_desc}"',
            f'    MODEL gpt-4o',
            f'    TEMPERATURE 0.1',
            f'    MAX_TOKENS 2048',
            f'    TASK {task}',
            f'    INPUT:',
            f'        user_input: String',
            f'    OUTPUT:',
            f'        is_match: Boolean',
            f'        confidence: Float',
            f'        scene: Integer',
            f'    PROMPT """',
        ]
        for p_line in prompt_text.split("\n"):
            lines.append(f"        {p_line}")
        lines.append('    """')
        lines.append("ENDSEMANTIC_BLOCK")

    return "\n".join(lines)


def _infer_output_type(field_name: str) -> str:
    """根据输出字段名推断类型——通用推断，不硬编码领域字段。"""
    name_lower = field_name.lower()
    # Float 类型信号
    if any(kw in name_lower for kw in ["score", "similarity", "confidence", "rate", "ratio",
                                         "price", "amount", "discount", "frequency", "probability"]):
        return "Float"
    # Integer 类型信号
    if any(kw in name_lower for kw in ["count", "num", "total", "size", "id", "scene",
                                         "level", "order_count", "points"]):
        return "Integer"
    # Boolean 类型信号
    if any(kw in name_lower for kw in ["is_", "has_", "match", "valid", "success",
                                         "flag", "risk"]):
        return "Boolean"
    # List 类型信号
    if any(kw in name_lower for kw in ["results", "items", "list", "docs", "entries",
                                         "orders", "messages"]):
        return "List"
    return "String"


def _infer_json_example(field_name: str) -> str:
    """根据输出字段名推断 JSON 示例值。"""
    name_lower = field_name.lower()
    if any(kw in name_lower for kw in ["score", "similarity", "confidence", "rate"]):
        return "0.85"
    if any(kw in name_lower for kw in ["count", "num", "total", "size"]):
        return "3"
    if any(kw in name_lower for kw in ["is_", "has_", "match", "valid"]):
        return "true"
    if any(kw in name_lower for kw in ["results", "items", "list", "docs", "top_results"]):
        return "[...]"
    if any(kw in name_lower for kw in ["mode", "type", "strategy"]):
        return '"direct"'
    return '"..."'


class SemanticBlockRenderer:
    """LLM 驱动的 SEMANTIC_BLOCK 渲染器。

    Args:
        llm_call: LLM 调用回调。接收 prompt 字符串，返回 JSON 字符串。
                  默认使用降级模板（不调用真实 LLM）。
    """

    def __init__(self, llm_call: Optional[Callable[[str], str]] = None):
        self.llm_call = llm_call or _default_llm_call

    def render(self, scene: SceneClassification) -> str:
        """渲染单个 SEMANTIC_BLOCK 场景。

        优先使用 LLM 渲染，失败时降级到模板渲染。
        """
        try:
            result = self._llm_render(scene)
            if result:
                return result
        except Exception as e:
            logger.warning(f"LLM 渲染 SEMANTIC_BLOCK 失败: {e}，降级到模板渲染")

        return _fallback_render(scene)

    def batch_render(self, scenes: list[SceneClassification]) -> list[str]:
        """批量渲染多个 SEMANTIC_BLOCK 场景。"""
        return [self.render(scene) for scene in scenes]

    def _llm_render(self, scene: SceneClassification) -> Optional[str]:
        """使用 LLM 渲染 SEMANTIC_BLOCK。返回 None 表示需要降级。"""
        sb_id = f"sb_{scene.block_id}"
        block_desc = scene.block_description or "未知场景"

        # 构建扩展信息
        ext_parts = []
        if scene.raw_context:
            ext_parts.append(f"- 上下文: {scene.raw_context}")
        if scene.preconditions:
            ext_parts.append(f"- 前置条件: {', '.join(scene.preconditions)}")
        if scene.fallback:
            ext_parts.append(f"- 回退逻辑: {scene.fallback}")
        if scene.action_sequence:
            ext_parts.append(f"- 执行步骤: {' → '.join(scene.action_sequence)}")
        if scene.nested_logic:
            ext_parts.append(f"- 嵌套条件: {json.dumps(scene.nested_logic, ensure_ascii=False)}")
        if scene.meta_rules:
            ext_parts.append(f"- 元规则: {', '.join(scene.meta_rules)}")
        extension_info = "\n".join(ext_parts) if ext_parts else "（无）"

        # 构建 LLM prompt
        system_prompt = SEMANTIC_RENDER_SYSTEM_PROMPT.format(
            grammar_spec=DSL_GRAMMAR_SPEC,
        )
        user_prompt = SEMANTIC_RENDER_USER_TEMPLATE.format(
            block_id=scene.block_id,
            block_description=block_desc,
            trigger_strategy=scene.trigger_strategy,
            action_strategy=scene.action_strategy,
            raw_requirement_text=scene.raw_requirement_text or block_desc,
            logic_flow=scene.logic_flow or "（未提供）",
            side_effects="\n".join(f"- {e}" for e in (scene.side_effects or [])) or "（无）",
            extension_info=extension_info,
        )

        # 调用 LLM
        response = self.llm_call(system_prompt + "\n\n" + user_prompt)
        if response is None:
            return None

        # 解析 LLM 输出
        try:
            llm_data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("LLM 输出非 JSON，降级到模板渲染")
            return None

        # 从 LLM 输出构建 SEMANTIC_BLOCK
        return self._build_from_llm_data(scene, llm_data)

    def _build_from_llm_data(self, scene: SceneClassification, llm_data: dict) -> str:
        """从 LLM 输出的 JSON 数据构建 SEMANTIC_BLOCK 文本。"""
        sb_id = f"sb_{scene.block_id}"
        block_desc = scene.block_description or "未知场景"

        task = llm_data.get("task", "classification")
        input_vars = llm_data.get("input_vars", [{"name": "user_input", "type": "String"}])
        output_vars = llm_data.get("output_vars", [
            {"name": "is_match", "type": "Boolean"},
            {"name": "confidence", "type": "Float"},
            {"name": "scene", "type": "Integer"},
        ])
        prompt_template = llm_data.get("prompt_template", f"判断输入是否属于{block_desc}场景")

        # 确保输出包含 scene（BLOCK 分流需要）
        output_names = [v["name"] for v in output_vars]
        if "scene" not in output_names:
            output_vars.append({"name": "scene", "type": "Integer"})
        if "is_match" not in output_names:
            output_vars.append({"name": "is_match", "type": "Boolean"})

        lines = [
            f'SEMANTIC_BLOCK {sb_id} "{block_desc}"',
            f'    MODEL gpt-4o',
            f'    TEMPERATURE 0.1',
            f'    MAX_TOKENS 2048',
            f'    TASK {task}',
            f'    INPUT:',
        ]
        for iv in input_vars:
            lines.append(f'        {iv["name"]}: {iv["type"]}')
        lines.append(f'    OUTPUT:')
        for ov in output_vars:
            lines.append(f'        {ov["name"]}: {ov["type"]}')
        lines.append(f'    PROMPT """')
        for p_line in prompt_template.split("\n"):
            lines.append(f'        {p_line}')
        lines.append('    """')
        lines.append("ENDSEMANTIC_BLOCK")

        return "\n".join(lines)
