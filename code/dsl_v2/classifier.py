"""
StrategyClassifier — 纯代码策略分类器。

职责：
1. 将 FactSpec（事实四元组）转换为 ClassifiedSpec（含策略分类）
2. 通过决策矩阵查表，零 LLM 调用
3. 每个场景的分类附带完整 _meta 溯源信息

决策矩阵从 config/strategy_matrix.json 加载。
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from .fact_types import (
    TriggerSpec, ExclusionSpec, ActionSpec,
    FactScene, FactSpec,
    SceneClassification, ClassifiedSpec,
    CompressibilityScore,
    SemanticVector,
)


# ============================================================================
# 决策矩阵加载
# ============================================================================

def load_decision_matrix(matrix_path: str) -> list[dict]:
    """从 JSON 文件加载决策矩阵规则列表。

    返回按 id 排序的规则列表，默认规则排在最后。
    """
    with open(matrix_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rules = list(data.get("rules", []))
    # 默认规则排最后（先匹配精确规则，最后兜底）
    rules.sort(key=lambda r: (1 if r["id"].startswith("rule_default") else 0, r["id"]))
    return rules


# ============================================================================
# 条件判断工具
# ============================================================================

def _evaluate_condition(condition_str: str, scene: FactScene) -> bool:
    """评估一条规则的 condition 表达式是否匹配场景。

    支持的表达式模式：
    - triggers.is_explicit_list == true/false
    - triggers.numeric_comparisons != null
    - triggers.implicit_intent != null
    - triggers.context_stage != null
    - triggers.multidimensional == true/false
    - action.action_type == 'xxx'
    - triggers.fallback != ''         （有回退逻辑）
    - triggers.action_sequence != []  （有多步动作序列）
    - triggers.nested_logic != null   （有嵌套条件树）
    - triggers.compressibility.score < 0.4  （低可压缩性）
    - true（永远匹配）
    - 复合条件: A and B（两个子条件都为 true）
    """
    condition_str = condition_str.strip()

    # 永远匹配
    if condition_str == "true":
        return True

    # 复合条件：A and B
    if " and " in condition_str:
        parts = condition_str.split(" and ", 1)
        return _evaluate_condition(parts[0].strip(), scene) and _evaluate_condition(parts[1].strip(), scene)

    # triggers 条件
    if condition_str.startswith("triggers."):
        return _evaluate_trigger_condition(condition_str, scene)

    # action 条件
    if condition_str.startswith("action."):
        return _evaluate_action_condition(condition_str, scene)

    return False


def _evaluate_trigger_condition(condition_str: str, scene: FactScene) -> bool:
    """评估触发条件相关的条件表达式。"""
    triggers = scene.triggers

    if condition_str == "triggers.is_explicit_list == true":
        return triggers.is_explicit_list
    if condition_str == "triggers.is_explicit_list == false":
        return not triggers.is_explicit_list
    if condition_str == "triggers.numeric_comparisons != null":
        return triggers.numeric_comparisons is not None
    if condition_str == "triggers.implicit_intent != null":
        return triggers.implicit_intent is not None
    if condition_str == "triggers.context_stage != null":
        return triggers.context_stage is not None
    if condition_str == "triggers.multidimensional == true":
        return triggers.multidimensional
    if condition_str == "triggers.multidimensional == false":
        return not triggers.multidimensional

    # 新增条件：扩展字段
    if condition_str == "triggers.fallback != ''":
        return scene.fallback != ""
    if condition_str == "triggers.action_sequence != []":
        return len(scene.action_sequence) > 0
    if condition_str == "triggers.nested_logic != null":
        return scene.nested_logic is not None

    # 可压缩性评分条件
    if condition_str.startswith("triggers.compressibility.score"):
        return _evaluate_compressibility_condition(condition_str, scene)

    return False


def _evaluate_compressibility_condition(condition_str: str, scene: FactScene) -> bool:
    """评估可压缩性评分条件。"""
    comp = scene.compressibility
    # 支持格式: triggers.compressibility.score < 0.4
    #           triggers.compressibility.score >= 0.4
    if condition_str == "triggers.compressibility.score < 0.4":
        return comp.score < 0.4
    if condition_str == "triggers.compressibility.score >= 0.4":
        return comp.score >= 0.4
    if condition_str == "triggers.compressibility.score < 0.7":
        return comp.score < 0.7
    if condition_str == "triggers.compressibility.score >= 0.7":
        return comp.score >= 0.7
    return False


def _evaluate_action_condition(condition_str: str, scene: FactScene) -> bool:
    """评估动作相关的条件表达式。"""
    action = scene.action

    # action.action_type == 'xxx'
    if condition_str.startswith("action.action_type == '") and condition_str.endswith("'"):
        expected = condition_str[len("action.action_type == '"):-1]
        return action.action_type == expected

    return False


# ============================================================================
# 决策矩阵匹配
# ============================================================================

def _match_rules(rules: list[dict], scene: FactScene) -> tuple[Optional[str], Optional[str]]:
    """对场景匹配所有规则，返回 (trigger_strategy, action_strategy)。

    先匹配 trigger 规则，再匹配 action 规则。
    按规则在数组中的顺序匹配，先匹配到的优先。
    """
    trigger_strategy = None
    action_strategy = None

    for rule in rules:
        if not _evaluate_condition(rule["condition"], scene):
            continue

        ts = rule.get("trigger_strategy")
        as_ = rule.get("action_strategy")

        if ts is not None and trigger_strategy is None:
            trigger_strategy = (ts, rule["id"], rule["reason"])
        if as_ is not None and action_strategy is None:
            action_strategy = (as_, rule["id"], rule["reason"])

        # 两个策略都找到就可以提前终止
        if trigger_strategy is not None and action_strategy is not None:
            break

    return trigger_strategy, action_strategy


# ============================================================================
# StrategyClassifier
# ============================================================================

class StrategyClassifier:
    """纯代码策略分类器。

    用法:
        classifier = StrategyClassifier()
        classified_spec = classifier.classify(fact_spec)
    """

    def __init__(self, matrix_path: Optional[str] = None):
        if matrix_path is None:
            matrix_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config", "strategy_matrix.json"
            )
        self.rules = load_decision_matrix(matrix_path)

    def classify(self, fact_spec: FactSpec) -> ClassifiedSpec:
        """将完整 FactSpec 分类为 ClassifiedSpec。"""
        classified_scenes = [
            self.classify_scene(scene) for scene in fact_spec.scenes
        ]
        return ClassifiedSpec(
            agent_name=fact_spec.agent_name,
            agent_description=fact_spec.agent_description,
            persona_role=fact_spec.persona_role,
            persona_capabilities=list(fact_spec.persona_capabilities),
            constraints=list(fact_spec.constraints),
            inputs=list(fact_spec.inputs),
            outputs=list(fact_spec.outputs),
            scenes=classified_scenes,
            fallback_scene_id=fact_spec.fallback_scene_id,
            fallback_agent_expression=fact_spec.fallback_agent_expression,
            examples=list(fact_spec.examples),
            configs=list(fact_spec.configs),
            policies=list(fact_spec.policies),
        )

    def classify_scene(self, scene: FactScene) -> SceneClassification:
        """对单个场景应用决策矩阵，返回分类结果。

        在矩阵分类后，执行语义性检测（Semantic Override）：
        如果 BLOCK_CONTAINS 的触发条件实际上是语义性描述（非字面用户输入），
        自动升级为 SEMANTIC_BLOCK_CLASSIFICATION。
        """
        trigger_strategy_info, action_strategy_info = _match_rules(self.rules, scene)

        # 提取策略值
        trigger_strategy = trigger_strategy_info[0] if trigger_strategy_info else "SEMANTIC_BLOCK_CLASSIFICATION"
        ts_rule_id = trigger_strategy_info[1] if trigger_strategy_info else "rule_default_trigger"
        ts_reason = trigger_strategy_info[2] if trigger_strategy_info else "默认：未知场景→LLM处理"

        action_strategy = action_strategy_info[0] if action_strategy_info else "SEMANTIC_BLOCK_GENERATION"
        as_rule_id = action_strategy_info[1] if action_strategy_info else "rule_default_action"
        as_reason = action_strategy_info[2] if action_strategy_info else "默认：未知动作→LLM生成"

        # ================================================================
        # 语义性覆盖检测（Semantic Override）
        # 当 BLOCK_CONTAINS 的关键词是文档式描述而非用户真实输入时，
        # 自动升级为 SEMANTIC_BLOCK_CLASSIFICATION
        # ================================================================
        semantic_override = False
        semantic_override_reason = ""
        if trigger_strategy == "BLOCK_CONTAINS" and scene.triggers.extracted_keywords:
            semantic_override, semantic_override_reason = _check_semantic_keywords(
                scene.triggers.extracted_keywords,
                scene.triggers.numeric_comparisons,
                scene.action.action_type,
            )
            if semantic_override:
                trigger_strategy = "SEMANTIC_BLOCK_CLASSIFICATION"
                ts_rule_id = "semantic_override"

        # ================================================================
        # 可压缩性评分——代码计算 + LLM 抽取，取更严格值（更低 score）
        # ================================================================
        code_compress = _compute_compressibility(scene)
        if scene.compressibility and scene.compressibility.score < code_compress.score:
            # LLM 给了更低的评分（更严格），用 LLM 的
            final_compress = scene.compressibility
        else:
            # 代码评分更严格或相等，用代码的
            final_compress = code_compress

        # ================================================================
        # 注意：fallback / nested_logic 不再覆盖 trigger_strategy
        # 原因：fallback 是"执行兜底"（如 try/except），不是"路由升级"
        # 嵌套逻辑是"执行复杂度"，不是"路由复杂度"
        # 这些由渲染器根据 action_strategy 决定如何处理
        # ================================================================

        # ================================================================
        # 方案C2：业务输出感知覆盖
        # 如果 action.outputs 包含业务字段（非 scene/agent），
        # 且 action_type 是 LLM 类任务，确保 action_strategy 反映实际执行逻辑
        # ================================================================
        business_outputs = scene.action.outputs or []
        has_business_outputs = len(business_outputs) > 0 and business_outputs != ["scene", "agent"]
        if has_business_outputs and not semantic_override:
            action_type = scene.action.action_type or ""
            # 有业务输出的 LLM 任务：确保 action_strategy 不是 CODE_ASSIGN
            if action_type in ("text_generation",) and not action_strategy.startswith("SEMANTIC_BLOCK"):
                action_strategy = "SEMANTIC_BLOCK_GENERATION"
                as_rule_id = "business_output_override"
            elif action_type in ("scoring",) and not action_strategy.startswith("SEMANTIC_BLOCK"):
                action_strategy = "SEMANTIC_BLOCK_SCORING"
                as_rule_id = "business_output_override"

        # ================================================================
        # compressibility 后置覆盖：低可压缩性场景应使用 RAW_SEMANTIC/HYBRID
        # 即使其他规则已经匹配了 SEMANTIC_BLOCK_CLASSIFICATION
        # ================================================================
        if final_compress.score < 0.4:
            trigger_strategy = "RAW_SEMANTIC"
            ts_rule_id = "compressibility_raw_semantic"
            semantic_override = True
            semantic_override_reason = f"可压缩性评分={final_compress.score}<0.4，原文整段透传"
        elif final_compress.score < 0.7:
            if trigger_strategy == "BLOCK_CONTAINS":
                trigger_strategy = "HYBRID"
                ts_rule_id = "compressibility_hybrid"
                semantic_override = True
                semantic_override_reason = f"可压缩性评分={final_compress.score}，BLOCK触发+SEMANTIC_BLOCK执行"
            elif trigger_strategy.startswith("SEMANTIC_BLOCK"):
                # 中等可压缩性 + SEMANTIC_BLOCK → HYBRID（有粗粒度关键词时更优）
                if scene.triggers.extracted_keywords:
                    trigger_strategy = "HYBRID"
                    ts_rule_id = "compressibility_hybrid_from_semantic"
                    semantic_override = True
                    semantic_override_reason = f"可压缩性评分={final_compress.score}，有部分关键词可用，HYBRID模式"

        # ================================================================
        # P2: 6维语义向量评分覆盖——LLM感知，确定性路由
        # 核心逻辑: 只有SEMANTIC评分(明确需要LLM)才升级策略为sensor
        # HYBRID评分 → 维持原策略不升级(M4自行决定是否生成sensor)
        # BLOCK评分 → 维持决策矩阵结果(纯代码)
        # ================================================================
        sv = scene.semantic_vector
        sv_strategy, sv_score, sv_breakdown = compute_execution_strategy(sv)

        if sv_strategy == "SEMANTIC":
            # 语义向量明确指向需要LLM理解 → 使用sensor
            if trigger_strategy in ("BLOCK_CONTAINS", "BLOCK_COMPARE"):
                trigger_strategy = "SEMANTIC_BLOCK_CLASSIFICATION"
                ts_rule_id = "semantic_vector_semantic"
                semantic_override = True
                semantic_override_reason = f"6维语义向量评分={sv_score}<{_HYBRID_THRESHOLD}，需sensor语义理解"
        # HYBRID: 不升级策略，让M4自行决定是否生成sensor
        # BLOCK: 维持决策矩阵结果

        # 构建 block_id — 优先使用已有的（来自 AgentSpec）
        if scene.block_id:
            block_id = scene.block_id
        else:
            block_id = f"b{scene.scene_id}_{_to_block_topic(scene.block_description)}"

        # 构建 _meta
        override_reason = semantic_override_reason if semantic_override else ""
        meta_decision = override_reason if override_reason else f"触发条件: {ts_reason}; 动作: {as_reason}"
        is_default_fallback = (
            (ts_rule_id == "rule_default_trigger" and not semantic_override) or
            (as_rule_id == "rule_default_action")
        )
        meta = {
            "rule_matched": ts_rule_id,
            "action_rule_matched": as_rule_id,
            "semantic_override": semantic_override,
            "semantic_override_reason": semantic_override_reason,
            "alternatives_considered": self._get_alternatives(scene, ts_rule_id, as_rule_id),
            "decision_reason": meta_decision,
            "source_ref": scene.source_ref or "",
            "is_default_fallback": is_default_fallback,
            "compressibility": {"score": final_compress.score, "lossy": final_compress.lossy_aspects, "recommendation": final_compress.recommendation},
            "semantic_vector_score": sv_score,
            "semantic_vector_strategy": sv_strategy,
            "semantic_vector_breakdown": sv_breakdown,
        }

        return SceneClassification(
            scene_id=scene.scene_id,
            block_id=block_id,
            block_description=scene.block_description,
            agent_expression=scene.agent_expression,
            source_ref=scene.source_ref,
            triggers=scene.triggers,
            exclusions=scene.exclusions,
            action=scene.action,
            trigger_strategy=trigger_strategy,
            action_strategy=action_strategy,
            meta=meta,
            # 富化字段——透传原始需求文本
            raw_requirement_text=scene.raw_requirement_text,
            logic_flow=scene.logic_flow,
            side_effects=list(scene.side_effects),
            # 扩展结构透传
            raw_context=scene.raw_context,
            preconditions=list(scene.preconditions),
            fallback=scene.fallback,
            action_sequence=list(scene.action_sequence),
            nested_logic=scene.nested_logic,
            meta_rules=list(scene.meta_rules),
            compressibility=final_compress,
            semantic_vector=sv,
        )

    def _get_alternatives(self, scene: FactScene, matched_rule: str, matched_action_rule: str) -> list[str]:
        """列出当前场景在相反策略下的备选方案，用于溯源。"""
        alternatives = []

        # trigger 备选
        if matched_rule.startswith("rule_0") and "BLOCK" in matched_rule:
            alternatives.append("SEMANTIC_BLOCK_CLASSIFICATION")  # 如果用语义判断
        elif matched_rule.startswith("rule_0") and "SEMANTIC" in matched_rule:
            alternatives.append("BLOCK_CONTAINS")  # 如果用关键词匹配

        # action 备选
        if matched_action_rule.startswith("rule_06"):
            alternatives.append("SEMANTIC_BLOCK_GENERATION")
        elif matched_action_rule.startswith("rule_10"):
            alternatives.append("CODE_ASSIGN")

        return alternatives


# ============================================================================
# 工具函数
# ============================================================================

# ============================================================================
# 语义性关键词检测
# ============================================================================

# 关键词中的"语义指示词"——出现这些词说明该条件需要 LLM 理解而非关键词匹配
_SEMANTIC_TRIGGER_PATTERNS = [
    # 数值/阈值比较（"超过0.85"、"低于500行"——用户不会输入这些字面量）
    "超过", "低于", "不小于", "大于", "小于", "以上", "以下",
    # 抽象语义/主观判断
    "相似度", "置信度", "匹配度", "相关度", "准确率",
    "不满意", "感兴趣", "抱怨", "放弃", "不满",
    # 复杂推理
    "复杂推理", "意图分解", "语义理解", "上下文", "知识图谱", "推理",
    # 动作性描述（不是用户会说的词，是文档对条件的描述）
    "连续3次", "检测到", "判断是否", "需要判断",
    # LLM 任务类型
    "文本生成", "多维度评分", "评分",
]


def _check_semantic_keywords(
    keywords: list[str],
    numeric_compare: Optional[dict],
    action_type: str,
) -> tuple[bool, str]:
    """检查一组触发关键词是否需要语义理解（关键词匹配不够用）。

    返回 (needs_semantic, reason)。

    关键区分：
    - 可参数化的数值比较（"购买频率 < 1"、"退货率 > 30%"）
      → 不需要语义路由，走 BLOCK_COMPARE + CODE_COMPUTE
    - 不可参数化的语义描述（"风格相似"、"情绪不满"）
      → 需要语义路由，走 SEMANTIC_BLOCK_CLASSIFICATION
    """
    # 如果有数值比较，判断是否可参数化
    if numeric_compare:
        field = numeric_compare.get("field", "unknown")
        # 可参数化的数值比较：field 不是 unknown，有明确的 op 和 value
        if field != "unknown" and numeric_compare.get("op") and numeric_compare.get("value") is not None:
            # 可参数化 → 不需要语义路由，让 rule_02 (BLOCK_COMPARE) 生效
            return False, ""
        else:
            # 不可参数化（缺少具体字段信息）→ 仍需语义路由
            return True, "包含数值/阈值比较，但字段信息不明确，需要语义理解"

    # 注意：action_type 不再影响 trigger_strategy 的升级判断
    # 原因：action_type 决定的是"执行方式"，不是"路由方式"
    # 例如 text_generation 任务可以通过关键词路由（用户说"你好"→生成欢迎语）
    # 执行仍由 action_strategy 决定（CODE_* vs SEMANTIC_BLOCK_*）

    # 检查关键词中是否包含语义指示词
    for kw in keywords:
        for pattern in _SEMANTIC_TRIGGER_PATTERNS:
            if pattern in kw:
                return True, f"关键词 '{kw}' 含语义指示词 '{pattern}'，无法用 CONTAINS 匹配"

    # 如果关键词全是语义描述（没有用户可能说的自然词），也算语义性
    # 注意：这个启发式检查过于保守（基于硬编码词表），容易导致误判
    # 在 is_explicit_list=True 的情况下，LLM 已经判定关键词是明确的，
    # 分类器不应二次质疑。因此只在 is_explicit_list=False 时才做这个检查。
    # 但此处无法直接获取 is_explicit_list，所以简化为：移除此检查。
    # 更精确的语义判断由 numeric_compare 和 _SEMANTIC_TRIGGER_PATTERNS 覆盖。

    return False, ""


# ============================================================================
# 可压缩性评分
# ============================================================================

def _compute_compressibility(scene: FactScene) -> CompressibilityScore:
    """计算场景的可压缩性评分。

    起始分 1.0，每命中一个检测项扣对应分数，最低 0.0。
    - 含模糊副词 → -0.2
    - 含转折/让步 → -0.3
    - 含隐含因果 → -0.2
    - 嵌套条件深度>2 → -0.3
    - 含主观状态描述 → -0.4
    - 含语气要求 → -0.3
    """
    score = 1.0
    lossy = []

    raw_text = scene.raw_requirement_text or ""
    keywords = scene.triggers.extracted_keywords or []

    # 检测项1：模糊副词
    fuzzy_adverbs = ["明显", "稍微", "有点", "大致", "基本", "可能"]
    if any(w in raw_text for w in fuzzy_adverbs):
        score -= 0.2
        lossy.append("含模糊副词")

    # 检测项2：转折/让步
    concession_patterns = ["虽然", "但是", "尽管", "然而", "不过", "仍然"]
    if any(w in raw_text for w in concession_patterns):
        score -= 0.3
        lossy.append("含转折/让步")

    # 检测项3：隐含因果
    causal_patterns = ["考虑到", "鉴于", "毕竟", "既然", "由于"]
    if any(w in raw_text for w in causal_patterns):
        score -= 0.2
        lossy.append("含隐含因果")

    # 检测项4：嵌套深度
    if scene.nested_logic is not None:
        depth = _estimate_nesting_depth(scene.nested_logic)
        if depth > 2:
            score -= 0.3
            lossy.append(f"条件嵌套深度={depth}")

    # 检测项5：主观状态
    subjective = ["情绪", "态度", "意图", "感受", "不满", "不满意"]
    if any(w in raw_text for w in subjective):
        score -= 0.4
        lossy.append("含主观状态描述")

    # 检测项6：语气要求
    tone_words = ["委婉", "强硬", "礼貌", "严肃", "温和"]
    if any(w in raw_text for w in tone_words):
        score -= 0.3
        lossy.append("含语气要求")

    score = max(0.0, round(score, 2))

    if score >= 0.7:
        rec = "BLOCK"
    elif score >= 0.4:
        rec = "HYBRID"
    else:
        rec = "RAW_SEMANTIC"

    return CompressibilityScore(score=score, lossy_aspects=lossy, recommendation=rec)


def _estimate_nesting_depth(logic: dict, depth: int = 0) -> int:
    """估算嵌套条件树的深度。"""
    if not isinstance(logic, dict):
        return depth
    max_child = depth
    for key in ("then", "else"):
        if key in logic and isinstance(logic[key], dict):
            child_depth = _estimate_nesting_depth(logic[key], depth + 1)
            max_child = max(max_child, child_depth)
    return max_child


def _to_block_topic(description: str) -> str:
    """将场景中文描述转为简短英文 topic，用于 block_id。"""
    if not description:
        return "default"
    # 取前 4 个中文字符的拼音首字母简化版——这里用简单的取首字策略
    # 实际可改进为更好的映射
    import re
    # 提取前两个有意义的中文字
    chars = re.findall(r'[\u4e00-\u9fff]', description)
    if chars:
        return ''.join(chars[:4])
    words = description.split()
    return words[0][:8].lower() if words else "default"


# ============================================================================
# P2: 6维语义向量 → 确定性执行策略路由
# ============================================================================

# 加权评分权重: 每个维度对"代码可执行性"的贡献权重
# 高权重=该维度对判断任务是否可纯代码执行影响大
_SEMANTIC_VECTOR_WEIGHTS = {
    "trigger_specificity": 0.25,       # 触发条件越明确, 越适合代码
    "trigger_context_dependency": 0.15, # 触发条件越独立, 越适合代码
    "action_determinism": 0.25,         # 动作越确定, 越适合代码
    "numeric_complexity": 0.10,         # 数值计算越简单, 越适合代码
    "output_structuredness": 0.15,      # 输出越结构化, 越适合代码
    "exception_handling": 0.10,         # 异常处理越可穷举, 越适合代码
}

# 执行策略阈值
_BLOCK_THRESHOLD = 0.55       # 加权分 ≥ 0.55 → BLOCK (纯代码)
_HYBRID_THRESHOLD = 0.35     # 加权分 ≥ 0.35 → HYBRID (代码为主, 可选sensor)
# 加权分 < 0.35 → SEMANTIC (sensor为主)


def compute_execution_strategy(sv: SemanticVector) -> tuple[str, float, dict]:
    """根据6维语义向量的加权评分，确定性决策执行策略。

    Args:
        sv: 6维语义向量 (每个维度 0.0-1.0)

    Returns:
        (strategy, score, breakdown)
        strategy: "BLOCK" | "HYBRID" | "SEMANTIC"
        score: 加权评分 (0.0-1.0, 越高越适合纯代码)
        breakdown: 各维度得分明细 {"trigger_specificity": (0.9, 0.25*0.9), ...}
    """
    dims = {
        "trigger_specificity": sv.trigger_specificity,
        "trigger_context_dependency": sv.trigger_context_dependency,
        "action_determinism": sv.action_determinism,
        "numeric_complexity": sv.numeric_complexity,
        "output_structuredness": sv.output_structuredness,
        "exception_handling": sv.exception_handling,
    }

    breakdown = {}
    weighted_sum = 0.0
    for dim_name, dim_value in dims.items():
        weight = _SEMANTIC_VECTOR_WEIGHTS[dim_name]
        contribution = weight * dim_value
        weighted_sum += contribution
        breakdown[dim_name] = {
            "value": round(dim_value, 2),
            "weight": weight,
            "contribution": round(contribution, 3),
        }

    score = round(weighted_sum, 3)

    if score >= _BLOCK_THRESHOLD:
        strategy = "BLOCK"
    elif score >= _HYBRID_THRESHOLD:
        strategy = "HYBRID"
    else:
        strategy = "SEMANTIC"

    return strategy, score, breakdown
