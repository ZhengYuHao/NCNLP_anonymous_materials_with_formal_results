"""
"事实四元组"数据模型 — UnifiedExtractor 的纯事实输出，不含策略决策。

Schema 设计原则：
- LLM 只输出"是什么"，不输出"怎么实现"
- 所有分类决策权交给 StrategyClassifier（纯代码）

每个场景被解析为统一的四元组结构：triggers, exclusions, action, outputs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================================
# 事实层 — LLM 输出
# ============================================================================


@dataclass
class TriggerSpec:
    """触发条件事实。只记录"有什么"，不决定"怎么实现"。

    Fields:
        raw_text: 原文中描述触发条件的原始文本
        extracted_keywords: 从原文提取的关键词列表（精确可匹配的字面量）
        is_explicit_list: 业务文档是否明确列出了关键词列表
        implicit_intent: 如果触发条件是基于语义推理（非关键词列表），
                        记录提炼后的意图描述，如"用户表达不满但未放弃"
        numeric_comparisons: 数值/枚举比较条件，如 {"field": "code_lines", "op": ">", "value": 500}
        context_stage: 上下文阶段判断的描述，如"流程处于稿件修改中"
        multidimensional: 是否涉及多维主观判断
    """
    raw_text: str = ""
    extracted_keywords: list[str] = field(default_factory=list)
    is_explicit_list: bool = False
    implicit_intent: Optional[str] = None
    numeric_comparisons: Optional[dict] = None
    # 结构：{"field": "字段名", "op": ">|<|>=|<=|==|!=", "value": 数值,
    #        "additional_conditions": [{"field": ..., "op": ..., "value": ...}, ...]}
    context_stage: Optional[str] = None
    multidimensional: bool = False


@dataclass
class ExclusionSpec:
    """排除条件事实。"""
    raw_text: str = ""
    extracted_keywords: list[str] = field(default_factory=list)
    is_explicit_list: bool = False


@dataclass
class ActionSpec:
    """动作事实。只记录"要做什么"，不记录"怎么实现"。

    action_type 取值:
        assign — 直接赋值（如 {{scene}} = 3）
        compute — 算术计算（如 {{total}} = {{a}} + {{b}}）
        template_fill — 字符串模板填充
        text_generation — 需要自然语言生成的输出
        scoring — 多维度评分
        extraction — 信息抽取
        similarity — 语义相似度判断
        lookup — 查表/函数调用
    """
    raw_text: str = ""
    action_type: str = "assign"  # assign | compute | template_fill | text_generation | scoring | extraction | similarity | lookup
    structured_op: Optional[dict] = None  # 结构化的操作描述（可选）
    outputs: list[str] = field(default_factory=list)


@dataclass
class ConfigItem:
    """配置项：key-value 对，承载非场景型需求（数据源、基础设施、性能参数等）。

    Fields:
        key: 配置项名称，如 "MAX_CONCURRENCY"
        value: 配置值，如 "50"
        description: 人类可读的描述
        category: 分类，如 "infrastructure" | "data_source" | "performance"
    """
    key: str = ""
    value: str = ""
    description: str = ""
    category: str = ""  # infrastructure | data_source | performance


@dataclass
class PolicyItem:
    """策略项：跨场景的运行时规则（安全、日志、监控等）。

    Fields:
        key: 策略名称，如 "SECURITY" | "LOGGING" | "MONITORING"
        description: 策略描述
        enforcement: 执行时机，如 "pre_check" | "post_check" | "always"
        trigger_condition: 触发条件，如 "检测到违规内容"
        action: 触发后的动作，如 "拒绝服务并记录日志"
    """
    key: str = ""
    description: str = ""
    enforcement: str = ""       # pre_check | post_check | always
    trigger_condition: str = ""
    action: str = ""


@dataclass
class CompressibilityScore:
    """可压缩性评分——由 LLM 在抽取时同时输出，也可由代码 _compute_compressibility 覆盖。

    Fields:
        score: 0.0-1.0，越高越适合压缩为关键词匹配
        lossy_aspects: 哪些方面压缩有损
        recommendation: BLOCK | SEMANTIC_BLOCK | HYBRID | RAW_SEMANTIC
    """
    score: float = 0.7
    lossy_aspects: list[str] = field(default_factory=list)
    recommendation: str = "BLOCK"


@dataclass
class SemanticVector:
    """6维语义向量——LLM感知（抽取），确定性算法决策（路由）。

    6个维度均为 0.0-1.0 浮点数，由 M3 LLM 抽取时同时输出。
    分类器根据加权评分决定任务走 BLOCK（纯代码）还是 SEMANTIC（sensor调用）。

    Dimensions:
        trigger_specificity: 触发条件的明确性 (1.0=精确可枚举, 0.0=模糊需推理)
        trigger_context_dependency: 触发条件对上下文的依赖度 (1.0=独立, 0.0=强依赖上下文)
        action_determinism: 动作结果的确定性 (1.0=输入唯一确定输出, 0.0=需语义判断)
        numeric_complexity: 数值计算复杂度 (1.0=无计算, 0.0=复杂计算)
        output_structuredness: 输出的结构化程度 (1.0=固定结构, 0.0=自由文本)
        exception_handling: 异常处理的代码化程度 (1.0=可穷举, 0.0=需灵活应对)
    """
    trigger_specificity: float = 0.5
    trigger_context_dependency: float = 0.5
    action_determinism: float = 0.5
    numeric_complexity: float = 0.5
    output_structuredness: float = 0.5
    exception_handling: float = 0.5


@dataclass
class FactScene:
    """一个场景的事实四元组。"""
    scene_id: int = 0
    block_id: str = ""             # 原文中的 block_id（若有）
    source_ref: str = ""           # 原文引用位置，如"原文 23-28 行"
    block_description: str = ""     # 场景中文描述
    agent_expression: str = ""      # 命中的智能体表达式
    triggers: TriggerSpec = field(default_factory=TriggerSpec)
    exclusions: ExclusionSpec = field(default_factory=ExclusionSpec)
    action: ActionSpec = field(default_factory=ActionSpec)
    # 富化字段——保留原始需求文本，避免信息丢失
    raw_requirement_text: str = ""  # 原文中该场景的原始描述文本（零信息丢失）
    logic_flow: str = ""           # 该场景的完整逻辑流描述（自然语言）
    side_effects: list[str] = field(default_factory=list)  # 副作用描述（日志、缓存、告警等）
    # === 扩展结构：解决 schema 过窄导致的强行归约 ===
    raw_context: str = ""               # 前后相邻段落，提供跨场景上下文
    preconditions: list[str] = field(default_factory=list)   # 前置状态 ["用户已签约", "已认证"]
    fallback: str = ""                  # 回退逻辑自然语言描述
    action_sequence: list[str] = field(default_factory=list) # 多步动作序列 ["先X", "再Y"]
    nested_logic: Optional[dict] = None # 嵌套条件树（LLM抽取的结构）
    meta_rules: list[str] = field(default_factory=list)      # 元规则
    compressibility: CompressibilityScore = field(default_factory=CompressibilityScore)
    semantic_vector: SemanticVector = field(default_factory=SemanticVector)  # P2: 6维语义向量


@dataclass
class FactSpec:
    """LLM 抽取的完整事实规格——UnifiedExtractor 的输出。

    包含所有场景的"是什么"信息，不含任何"怎么实现"的策略决策。
    """
    agent_name: str = ""
    agent_description: str = ""
    persona_role: str = ""
    persona_capabilities: list[str] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    inputs: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    scenes: list[FactScene] = field(default_factory=list)
    fallback_scene_id: int = 0
    fallback_agent_expression: str = '"未知"'
    examples: list[dict] = field(default_factory=list)
    configs: list[ConfigItem] = field(default_factory=list)
    policies: list[PolicyItem] = field(default_factory=list)


# ============================================================================
# 策略层 — StrategyClassifier 输出
# ============================================================================


@dataclass
class SceneClassification:
    """一个场景的分类结果——包含事实 + 策略决策 + 元数据。

    这是 DSLRenderer 的输入单元。
    """
    scene_id: int = 0
    block_id: str = ""
    block_description: str = ""
    agent_expression: str = ""
    source_ref: str = ""

    # 事实（来自 FactScene）
    triggers: TriggerSpec = field(default_factory=TriggerSpec)
    exclusions: ExclusionSpec = field(default_factory=ExclusionSpec)
    action: ActionSpec = field(default_factory=ActionSpec)

    # 策略决策
    trigger_strategy: str = "BLOCK_CONTAINS"
    #   BLOCK_CONTAINS | BLOCK_COMPARE
    #   | SEMANTIC_BLOCK_CLASSIFICATION | SEMANTIC_BLOCK_EXTRACTION
    #   | RAW_SEMANTIC | HYBRID
    action_strategy: str = "CODE_ASSIGN"
    #   CODE_ASSIGN | CODE_TEMPLATE | CODE_CALL_FUNC | CODE_COMPUTE
    #   | SEMANTIC_BLOCK_GENERATION | SEMANTIC_BLOCK_SCORING
    #   | SEMANTIC_BLOCK_EXTRACTION | SEMANTIC_BLOCK_SIMILARITY
    #   | ABSORBED（场景簇合并后标记）

    # 溯源元数据
    meta: dict = field(default_factory=dict)

    # 富化字段——保留原始需求文本，供 SEMANTIC_BLOCK 渲染使用
    raw_requirement_text: str = ""  # 原文中该场景的原始描述文本
    logic_flow: str = ""           # 该场景的完整逻辑流描述
    side_effects: list[str] = field(default_factory=list)  # 副作用描述
    # === 扩展结构透传 ===
    raw_context: str = ""               # 前后相邻段落，提供跨场景上下文
    preconditions: list[str] = field(default_factory=list)   # 前置状态
    fallback: str = ""                  # 回退逻辑自然语言描述
    action_sequence: list[str] = field(default_factory=list) # 多步动作序列
    nested_logic: Optional[dict] = None # 嵌套条件树
    meta_rules: list[str] = field(default_factory=list)      # 元规则
    compressibility: CompressibilityScore = field(default_factory=CompressibilityScore)
    semantic_vector: SemanticVector = field(default_factory=SemanticVector)  # P2: 6维语义向量


@dataclass
class ClassifiedSpec:
    """策略分类后的完整规格——DSLRenderer 的输入。

    保留所有来自 FactSpec 的元数据 + 每个场景的分类结果。
    """
    agent_name: str = ""
    agent_description: str = ""
    persona_role: str = ""
    persona_capabilities: list[str] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    inputs: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    scenes: list[SceneClassification] = field(default_factory=list)
    fallback_scene_id: int = 0
    fallback_agent_expression: str = '"未知"'
    examples: list[dict] = field(default_factory=list)
    configs: list[ConfigItem] = field(default_factory=list)
    policies: list[PolicyItem] = field(default_factory=list)
