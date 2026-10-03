"""数据模型 — 定义从提示词中提取出的结构化场景信息。

LLM 抽取 → 此 schema 的实例 → 渲染器 → DSL v1.2 代码

这层 schema 是 LLM 必须产出的"合约",也是渲染器的唯一输入。
它扮演了 SPL 和原始 DSL 之间的"结构化中间表示"角色。
"""
from __future__ import annotations
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, field_validator


def _normalize_dsl_type(value: Any) -> Any:
    """Collapse common parameterized LLM type spellings to DSL base types."""
    if not isinstance(value, str):
        return value
    compact = value.strip().replace(" ", "")
    base = compact.split("[", 1)[0].split("<", 1)[0].lower()
    aliases = {
        "str": "String",
        "string": "String",
        "int": "Integer",
        "integer": "Integer",
        "float": "Float",
        "number": "Float",
        "bool": "Boolean",
        "boolean": "Boolean",
        "list": "List",
        "array": "List",
        "dict": "Dict",
        "dictionary": "Dict",
        "object": "Dict",
        "any": "Any",
    }
    return aliases.get(base, value)


class _TypedSpec(BaseModel):
    @field_validator("type", mode="before", check_fields=False)
    @classmethod
    def normalize_type(cls, value: Any) -> Any:
        return _normalize_dsl_type(value)


# ---------- 元数据块 ----------

class PersonaSpec(BaseModel):
    role: str = Field(..., description="核心角色描述(对应 SPL 的 ROLE_ASPECT)")
    capabilities: list[str] = Field(
        default_factory=list,
        description="能力描述,每条一句话",
    )


class ConstraintSpec(BaseModel):
    key: str = Field(..., description="UPPER_SNAKE_CASE 的约束键,如 OUTPUT/PRIORITY")
    description: str


class InputSpec(_TypedSpec):
    name: str
    type: Literal["String", "Integer", "Float", "Boolean", "List", "Dict", "Any"] = "String"
    required: bool = True
    default: Optional[str] = None  # 表达为 DSL 字面量形式,如 '""' 或 '0'


class OutputSpec(_TypedSpec):
    name: str
    type: Literal["String", "Integer", "Float", "Boolean", "List", "Dict", "Any"] = "String"


# ---------- 场景定义 ----------

class InputVarSpec(_TypedSpec):
    """SEMANTIC_BLOCK 的输入变量定义"""
    name: str = Field(..., description="变量名")
    type: Literal["String", "Integer", "Float", "Boolean", "List", "Dict", "Any"] = "String"


class OutputVarSpec(_TypedSpec):
    """SEMANTIC_BLOCK 的输出变量定义"""
    name: str = Field(..., description="变量名")
    type: Literal["String", "Integer", "Float", "Boolean", "List", "Dict", "Any"] = "String"


class SemanticHook(BaseModel):
    """需要调用 LLM 语义判断时使用。对应"LLM + 规则混合调度模式"。

    渲染器会根据这个结构生成 SEMANTIC_BLOCK...ENDSEMANTIC_BLOCK 定义。
    """
    function_name: str = Field(..., description="SEMANTIC_BLOCK 函数名,如 judge_intent_polarity")
    result_var: str = Field(default="polarity", description="接收结果的变量名")
    gate_condition: str = Field(..., description="如 '!= \"abandon\"' 或 '== \"positive\"'")
    arguments: list[str] = Field(default_factory=lambda: ["user_input"], description="传递给函数的参数名列表")

    model: str = Field(default="gpt-4o", description="调用的 LLM 模型,如 gpt-4o")
    task: str = Field(..., description="任务类型,如 classification/generation/scoring/judge")
    input_vars: list[InputVarSpec] = Field(default_factory=list, description="SEMANTIC_BLOCK 输入变量定义")
    output_vars: list[OutputVarSpec] = Field(default_factory=list, description="SEMANTIC_BLOCK 输出变量定义")
    prompt_template: str = Field(..., description="给 LLM 的提示词模板")
    description: str = Field(default="", description="SEMANTIC_BLOCK 描述")


class SubScenario(BaseModel):
    """一个场景内部的一个判定分支(对应业务文档的子场景、子部分)。"""
    name: str = Field(..., description="子场景名称,仅作文档")
    trigger_keywords: list[str] = Field(..., min_length=1)
    exclusion_keywords: list[str] = Field(default_factory=list)
    semantic: Optional[SemanticHook] = None
    fact: Optional[dict] = Field(
        default=None,
        description="LLM 抽取的事实信息(由 StrategyClassifier 使用),"
                    "包含 action_type / trigger_is_explicit_list 等字段",
    )


class Scene(BaseModel):
    """一个完整场景 = 一个 BLOCK。"""
    priority: int = Field(..., description="数值越小优先级越高,决定 BLOCK 书写顺序")
    block_id: str = Field(..., description="唯一 ID,建议 b<priority>_<topic>")
    block_description: str
    scene_id: int = Field(..., description="RETURN 的场景编号")
    agent_expression: str = Field(
        ...,
        description=(
            "返回的 agent 表达式。字面量写作 '\"xx智能体\"' "
            "(带双引号),变量引用写作 '{{previous_agent}}'。"
        ),
    )
    sub_scenarios: list[SubScenario] = Field(..., min_length=1)


# ---------- 示例 ----------

class ExampleCase(BaseModel):
    input: dict[str, Any]
    expected: dict[str, Any]
    execution_path: list[str] = Field(..., description="命中的 block_id 序列")


# ---------- 顶层 Agent ----------

class AgentSpec(BaseModel):
    """对应 DSL v1.2 的 AGENT 顶层。"""

    agent_name: str = Field(..., description="PascalCase 智能体类名")
    agent_description: str

    persona: PersonaSpec
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    inputs: list[InputSpec] = Field(default_factory=list)
    outputs: list[OutputSpec] = Field(default_factory=list)

    # 场景列表。必须按 priority 递增排序,渲染时直接按顺序写 BLOCK
    scenes: list[Scene] = Field(..., min_length=1)

    # 兜底场景。渲染时追加到最后一个 BLOCK
    fallback_scene_id: int = 0
    fallback_agent_expression: str = '"未知"'

    examples: list[ExampleCase] = Field(default_factory=list)

    # 配置项和策略项 (EXTRACTION_SYSTEM_PROMPT 规则 9-10 要求输出)
    configs: list[Dict[str, Any]] = Field(default_factory=list, description="配置项列表")
    policies: list[Dict[str, Any]] = Field(default_factory=list, description="策略项列表")

    def sort_scenes(self) -> AgentSpec:
        self.scenes = sorted(self.scenes, key=lambda s: s.priority)
        return self
