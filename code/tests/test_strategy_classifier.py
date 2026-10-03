"""
迭代1 TDD：StrategyClassifier 完整决策矩阵测试

覆盖 12+2 条决策规则：
- Rule 01-05: 触发条件策略 (trigger_strategy)
- Rule 06-13: 动作策略 (action_strategy)
- Rule default: 兜底策略

测试原则：
1. 每个测试覆盖一条决策规则
2. 测试只输入事实特征，断言分类结果
3. 不依赖 LLM，纯代码确定性行为
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from dsl_v2.fact_types import (
    TriggerSpec, ExclusionSpec, ActionSpec,
    FactScene, FactSpec, SceneClassification, ClassifiedSpec,
)
from dsl_v2.classifier import StrategyClassifier, load_decision_matrix


# ============================================================================
# 夹具
# ============================================================================

@pytest.fixture
def classifier():
    """每个测试获取一个新的 StrategyClassifier 实例"""
    matrix_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'strategy_matrix.json')
    return StrategyClassifier(matrix_path=matrix_path)


# ============================================================================
# Rule 01: triggers.is_explicit_list == true → BLOCK_CONTAINS
# ============================================================================

def test_rule01_explicit_keyword_list_uses_block_contains(classifier):
    """业务文档列出了明确关键词列表 → BLOCK_CONTAINS"""
    scene = FactScene(
        scene_id=1,
        source_ref="原文 10-15 行",
        block_description="议价场景",
        agent_expression='"议价智能体"',
        triggers=TriggerSpec(
            raw_text="用户提到返点、价格、预算",
            extracted_keywords=["返点", "价格", "预算"],
            is_explicit_list=True,
        ),
        action=ActionSpec(action_type="assign"),
    )
    result = classifier.classify_scene(scene)
    assert result.trigger_strategy == "BLOCK_CONTAINS"
    assert result.meta["rule_matched"] == "rule_01"


# ============================================================================
# Rule 02: triggers.numeric_comparisons != null → BLOCK_COMPARE
# ============================================================================

def test_rule02_numeric_comparison_uses_block_compare(classifier):
    """触发条件含数值/枚举比较 → BLOCK_COMPARE"""
    scene = FactScene(
        scene_id=2,
        source_ref="原文 30-35 行",
        block_description="代码行数判断",
        agent_expression='"代码分析智能体"',
        triggers=TriggerSpec(
            raw_text="代码行数超过500行",
            extracted_keywords=["代码行数超过500"],
            is_explicit_list=False,
            numeric_comparisons={"field": "code_lines", "op": ">", "value": 500},
        ),
        action=ActionSpec(action_type="assign"),
    )
    result = classifier.classify_scene(scene)
    assert result.trigger_strategy == "BLOCK_COMPARE"
    assert result.meta["rule_matched"] == "rule_02"


# ============================================================================
# Rule 03: triggers.implicit_intent != null → SEMANTIC_BLOCK_CLASSIFICATION
# ============================================================================

def test_rule03_implicit_intent_uses_semantic_classification(classifier):
    """需要整句语义推理 → SEMANTIC_BLOCK(classification)"""
    scene = FactScene(
        scene_id=3,
        source_ref="原文 42-48 行",
        block_description="意图判断",
        agent_expression='"意图分析智能体"',
        triggers=TriggerSpec(
            raw_text="用户表达不满但未明确放弃",
            extracted_keywords=[],
            is_explicit_list=False,
            implicit_intent="判断用户是否想放弃对话",
        ),
        action=ActionSpec(action_type="assign"),
    )
    result = classifier.classify_scene(scene)
    assert result.trigger_strategy == "SEMANTIC_BLOCK_CLASSIFICATION"
    assert result.meta["rule_matched"] == "rule_03"


# ============================================================================
# Rule 04: triggers.context_stage != null → SEMANTIC_BLOCK_CLASSIFICATION
# ============================================================================

def test_rule04_context_stage_uses_semantic_classification(classifier):
    """上下文阶段判断 → SEMANTIC_BLOCK(classification)"""
    scene = FactScene(
        scene_id=4,
        source_ref="原文 55-60 行",
        block_description="流程阶段判断",
        agent_expression='"流程判断智能体"',
        triggers=TriggerSpec(
            raw_text="流程处于稿件修改中",
            extracted_keywords=[],
            is_explicit_list=False,
            context_stage="判断当前流程阶段",
        ),
        action=ActionSpec(action_type="assign"),
    )
    result = classifier.classify_scene(scene)
    assert result.trigger_strategy == "SEMANTIC_BLOCK_CLASSIFICATION"
    assert result.meta["rule_matched"] == "rule_04"


# ============================================================================
# Rule 05: triggers.multidimensional == true → SEMANTIC_BLOCK_EXTRACTION
# ============================================================================

def test_rule05_multidimensional_uses_semantic_extraction(classifier):
    """多维主观判断 → SEMANTIC_BLOCK(extraction)"""
    scene = FactScene(
        scene_id=5,
        source_ref="原文 70-78 行",
        block_description="博主匹配度评估",
        agent_expression='"博主匹配智能体"',
        triggers=TriggerSpec(
            raw_text="综合评估博主与商单的匹配度",
            extracted_keywords=[],
            is_explicit_list=False,
            multidimensional=True,
        ),
        action=ActionSpec(action_type="extraction"),
    )
    result = classifier.classify_scene(scene)
    assert result.trigger_strategy == "SEMANTIC_BLOCK_EXTRACTION"
    assert result.action_strategy == "SEMANTIC_BLOCK_EXTRACTION"
    assert result.meta["rule_matched"] == "rule_05"


# ============================================================================
# Rule 06: action.action_type == 'assign' → CODE_ASSIGN
# ============================================================================

def test_rule06_assign_uses_code_assign(classifier):
    """直接赋值操作 → CODE_ASSIGN"""
    scene = FactScene(
        scene_id=6,
        source_ref="原文 12 行",
        block_description="场景赋值",
        agent_expression='"默认智能体"',
        triggers=TriggerSpec(
            raw_text="用户说你好",
            extracted_keywords=["你好"],
            is_explicit_list=True,
        ),
        action=ActionSpec(
            raw_text="设置场景编号为1",
            action_type="assign",
            outputs=["scene"],
        ),
    )
    result = classifier.classify_scene(scene)
    assert result.action_strategy == "CODE_ASSIGN"
    # rule_05f (assign+explicit_list) 比 rule_06 (仅assign) 更具体，优先匹配
    assert result.meta["action_rule_matched"] in ("rule_06", "rule_05f")


# ============================================================================
# Rule 07-09: template_fill / lookup / compute → CODE_*
# ============================================================================

def test_rule07_template_fill_uses_code_template(classifier):
    """模板填充操作 → CODE_TEMPLATE"""
    scene = FactScene(
        scene_id=7, source_ref="", block_description="", agent_expression='""',
        triggers=TriggerSpec(extracted_keywords=["default"], is_explicit_list=True),
        action=ActionSpec(action_type="template_fill"),
    )
    result = classifier.classify_scene(scene)
    assert result.action_strategy == "CODE_TEMPLATE"
    # rule_05d/e/f 系列（action_type+explicit_list）可能比 rule_07-09 更具体
    assert result.meta["action_rule_matched"] in ("rule_07",)


def test_rule08_lookup_uses_code_call_func(classifier):
    """查表/函数调用 → CODE_CALL_FUNC"""
    scene = FactScene(
        scene_id=8, source_ref="", block_description="", agent_expression='""',
        triggers=TriggerSpec(extracted_keywords=["default"], is_explicit_list=True),
        action=ActionSpec(action_type="lookup"),
    )
    result = classifier.classify_scene(scene)
    assert result.action_strategy == "CODE_CALL_FUNC"
    # rule_05e (lookup+explicit_list) 比 rule_08 (仅lookup) 更具体，优先匹配
    assert result.meta["action_rule_matched"] in ("rule_08", "rule_05e")


def test_rule09_compute_uses_code_compute(classifier):
    """算术计算操作 → CODE_COMPUTE"""
    scene = FactScene(
        scene_id=9, source_ref="", block_description="", agent_expression='""',
        triggers=TriggerSpec(extracted_keywords=["default"], is_explicit_list=True),
        action=ActionSpec(action_type="compute"),
    )
    result = classifier.classify_scene(scene)
    assert result.action_strategy == "CODE_COMPUTE"
    # rule_05d (compute+explicit_list) 比 rule_09 (仅compute) 更具体，优先匹配
    assert result.meta["action_rule_matched"] in ("rule_09", "rule_05d")


# ============================================================================
# Rule 10: action.action_type == 'text_generation' → SEMANTIC_BLOCK_GENERATION
# ============================================================================

def test_rule10_text_generation_direct_semantic_generation(classifier):
    """文本生成操作 → 直接 SEMANTIC_BLOCK(generation)"""
    scene = FactScene(
        scene_id=10, source_ref="原文 85 行", block_description="回复生成",
        agent_expression='"生成智能体"',
        triggers=TriggerSpec(
            raw_text="用户询问产品信息",
            extracted_keywords=["询问", "产品"],
            is_explicit_list=True,
        ),
        action=ActionSpec(
            raw_text="生成回复文本",
            action_type="text_generation",
            outputs=["reply"],
        ),
    )
    result = classifier.classify_scene(scene)
    assert result.action_strategy == "SEMANTIC_BLOCK_GENERATION"
    # rule_05g (text_generation) 比 rule_10 更具体，可能优先匹配
    assert result.meta["action_rule_matched"] in ("rule_10", "rule_05g")


# ============================================================================
# Rule 11-13: scoring / extraction / similarity
# ============================================================================

def test_rule11_scoring_uses_semantic_scoring(classifier):
    """评分操作 → SEMANTIC_BLOCK(scoring)"""
    scene = FactScene(
        scene_id=11, source_ref="", block_description="", agent_expression='""',
        triggers=TriggerSpec(extracted_keywords=["评分"], is_explicit_list=True),
        action=ActionSpec(action_type="scoring"),
    )
    result = classifier.classify_scene(scene)
    assert result.action_strategy == "SEMANTIC_BLOCK_SCORING"


def test_rule12_extraction_uses_semantic_extraction(classifier):
    """信息抽取 → SEMANTIC_BLOCK(extraction)"""
    scene = FactScene(
        scene_id=12, source_ref="", block_description="", agent_expression='""',
        triggers=TriggerSpec(extracted_keywords=["提取"], is_explicit_list=True),
        action=ActionSpec(action_type="extraction"),
    )
    result = classifier.classify_scene(scene)
    assert result.action_strategy == "SEMANTIC_BLOCK_EXTRACTION"


def test_rule13_similarity_uses_semantic_similarity(classifier):
    """语义相似度 → SEMANTIC_BLOCK(similarity)"""
    scene = FactScene(
        scene_id=13, source_ref="", block_description="", agent_expression='""',
        triggers=TriggerSpec(extracted_keywords=["相似"], is_explicit_list=True),
        action=ActionSpec(action_type="similarity"),
    )
    result = classifier.classify_scene(scene)
    assert result.action_strategy == "SEMANTIC_BLOCK_SIMILARITY"


# ============================================================================
# 完整 FactSpec → ClassifiedSpec 转换
# ============================================================================

def test_classify_full_spec_produces_correct_count(classifier):
    """完整 FactSpec 分类后，场景数量不变"""
    fact_spec = FactSpec(
        agent_name="TestAgent",
        agent_description="测试智能体",
        scenes=[
            FactScene(
                scene_id=1,
                triggers=TriggerSpec(extracted_keywords=["hello"], is_explicit_list=True),
                action=ActionSpec(action_type="assign"),
            ),
            FactScene(
                scene_id=2,
                triggers=TriggerSpec(extracted_keywords=[], is_explicit_list=False,
                                     implicit_intent="判断意图"),
                action=ActionSpec(action_type="text_generation"),
            ),
        ],
    )
    classified = classifier.classify(fact_spec)
    assert len(classified.scenes) == 2
    assert classified.agent_name == "TestAgent"


# ============================================================================
# 决策矩阵加载
# ============================================================================

def test_load_decision_matrix_from_json():
    """从 JSON 文件加载决策矩阵"""
    matrix_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'strategy_matrix.json')
    rules = load_decision_matrix(matrix_path)
    assert len(rules) >= 14  # 12 条规则 + 2 条默认
    # 验证第一条规则
    rule_01 = next(r for r in rules if r["id"] == "rule_01")
    assert rule_01["trigger_strategy"] == "BLOCK_CONTAINS"


# ============================================================================
# 元数据完整性
# ============================================================================

def test_classification_meta_contains_all_fields(classifier):
    """分类结果 _meta 应包含完整溯源信息"""
    scene = FactScene(
        scene_id=1,
        source_ref="原文 10-15 行",
        block_description="测试场景",
        agent_expression='"测试智能体"',
        triggers=TriggerSpec(
            extracted_keywords=["测试"],
            is_explicit_list=True,
        ),
        action=ActionSpec(action_type="assign"),
    )
    result = classifier.classify_scene(scene)
    meta = result.meta
    assert "rule_matched" in meta
    assert "action_rule_matched" in meta
    assert "alternatives_considered" in meta
    assert "decision_reason" in meta
    assert "source_ref" in meta
    assert meta["source_ref"] == "原文 10-15 行"


# ============================================================================
# 默认兜底
# ============================================================================

def test_unknown_trigger_falls_back_to_semantic_block(classifier):
    """无法判断触发条件时，默认 SEMANTIC_BLOCK_CLASSIFICATION（LLM兜底）"""
    scene = FactScene(
        scene_id=99, source_ref="", block_description="", agent_expression='""',
        triggers=TriggerSpec(extracted_keywords=[], is_explicit_list=False),
        action=ActionSpec(action_type="unknown_type_xyz"),
    )
    result = classifier.classify_scene(scene)
    assert result.trigger_strategy == "SEMANTIC_BLOCK_CLASSIFICATION"
    assert result.action_strategy == "SEMANTIC_BLOCK_GENERATION"


# ============================================================================
# 语义覆盖测试（Semantic Override）
# ============================================================================

def test_semantic_keyword_similarity_overrides_to_semantic(classifier):
    """含'相似度'的语义关键词 → 覆盖为 SEMANTIC_BLOCK"""
    scene = FactScene(
        scene_id=1, source_ref="", block_description="相似度判断",
        agent_expression='"向量检索智能体"',
        triggers=TriggerSpec(
            extracted_keywords=["简单事实查询", "相似度超过0.85"],
            is_explicit_list=True,
        ),
        action=ActionSpec(action_type="assign"),
    )
    result = classifier.classify_scene(scene)
    assert result.trigger_strategy == "SEMANTIC_BLOCK_CLASSIFICATION"
    assert result.meta.get("semantic_override") is True


def test_semantic_keyword_complex_reasoning_overrides(classifier):
    """含'复杂推理'的关键词 → 覆盖为 SEMANTIC_BLOCK"""
    scene = FactScene(
        scene_id=2, source_ref="", block_description="推理查询",
        agent_expression='"图数据库智能体"',
        triggers=TriggerSpec(
            extracted_keywords=["复杂推理查询", "知识图谱", "2跳邻居"],
            is_explicit_list=True,
        ),
        action=ActionSpec(action_type="assign"),
    )
    result = classifier.classify_scene(scene)
    assert result.trigger_strategy == "SEMANTIC_BLOCK_CLASSIFICATION"
    assert result.meta.get("semantic_override") is True


def test_literal_keyword_remains_block_contains(classifier):
    """纯字面关键词（'价格'、'返点'）→ 保持 BLOCK_CONTAINS"""
    scene = FactScene(
        scene_id=3, source_ref="", block_description="议价",
        agent_expression='"议价智能体"',
        triggers=TriggerSpec(
            extracted_keywords=["价格", "返点", "预算"],
            is_explicit_list=True,
        ),
        action=ActionSpec(action_type="assign"),
    )
    result = classifier.classify_scene(scene)
    assert result.trigger_strategy == "BLOCK_CONTAINS"
    assert result.meta.get("semantic_override") is False


def test_mixed_keywords_with_literal_and_semantic(classifier):
    """混合关键词含语义词 → 覆盖为 SEMANTIC_BLOCK"""
    scene = FactScene(
        scene_id=4, source_ref="", block_description="用户反馈",
        agent_expression='"反馈智能体"',
        triggers=TriggerSpec(
            extracted_keywords=["重新生成", "不满意"],
            is_explicit_list=True,
        ),
        action=ActionSpec(action_type="assign"),
    )
    result = classifier.classify_scene(scene)
    assert result.trigger_strategy == "SEMANTIC_BLOCK_CLASSIFICATION"
    assert result.meta.get("semantic_override") is True
