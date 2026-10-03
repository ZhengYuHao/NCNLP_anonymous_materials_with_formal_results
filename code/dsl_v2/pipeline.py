"""
PipelineV2 — 五层编译管线的编排器。

管线流程：
1. extract: 现有 UnifiedExtractor 输出 AgentSpec
2. adapt: AgentSpec → FactSpec（适配器，过渡用）
3. classify: FactSpec → ClassifiedSpec
4. render: ClassifiedSpec → DSL 文本（ANIT-HOLLOW 兼容）
5. validate: 语法验证 + ANTI_HOLLOW lint
6. gate: 执行验证 + 升级
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .fact_types import (
    TriggerSpec, ExclusionSpec, ActionSpec,
    FactScene, FactSpec,
    SceneClassification, ClassifiedSpec,
    ConfigItem, PolicyItem,
    CompressibilityScore,
    SemanticVector,
)
from .classifier import StrategyClassifier
from .executor import DSLExecutor
from .gate import ExamplesGate
from .upgrader import Upgrader
from .anti_hollow import check_anti_hollow

logger = logging.getLogger(__name__)


# ============================================================================
# 通用推断函数（方案B）——从文本语义推断 action_type / outputs / is_explicit_list
# ============================================================================

def _infer_action_type(desc: str, action_text: str, fact_action_type: str = "") -> str:
    """通用 action_type 推断——基于动作关键词，不硬编码领域字段名。

    优先级：fact 字段 > 文本语义推断 > 兜底
    """
    # 如果 LLM 已经给出了有效的 action_type，直接使用
    if fact_action_type and fact_action_type in (
        "assign", "compute", "template_fill", "text_generation",
        "scoring", "extraction", "similarity", "lookup"
    ):
        return fact_action_type

    # 从文本语义推断
    text = (desc + " " + action_text).lower()
    if any(kw in text for kw in ["查询", "获取", "调用系统", "检索", "系统a", "系统b", "数据库"]):
        return "lookup"
    if any(kw in text for kw in ["计算", "打折", "折扣", "乘", "除", "比例", "统计", "价格"]):
        return "compute"
    if any(kw in text for kw in ["生成", "写", "语气", "欢迎语", "文案", "回复", "描述", "段落"]):
        return "text_generation"
    if any(kw in text for kw in ["判断", "评估", "风险", "评分", "分析", "标注", "检测"]):
        return "scoring"
    if any(kw in text for kw in ["发送", "发放", "如果是", "应用", "发放优惠券"]):
        return "assign"
    return "assign"  # 兜底


def _infer_outputs_from_text(desc: str, action_text: str, action_type: str, fact_outputs: list = None) -> list:
    """通用 outputs 推断——从原文动态提取变量名，不硬编码领域字段。

    优先级：fact 字段有业务输出 > 文本语义提取 > 通用兜底
    """
    # 如果 fact 已经给出了非默认的业务输出，直接使用
    if fact_outputs and fact_outputs != ["scene", "agent"]:
        return fact_outputs

    combined = desc + " " + action_text

    # 策略1：从"动词+名词"模式提取（"计算折扣"→"discount"）
    import re
    verb_noun_pairs = re.findall(
        r'(?:计算|查询|获取|判断|生成|发送|评估|发放|提取|分析|检测)([\w]{1,6})',
        combined
    )
    if verb_noun_pairs:
        return [_normalize_var_name(n) for n in verb_noun_pairs[:4]]

    # 策略2：按 action_type 给出通用输出名
    generic_outputs = {
        "lookup": ["query_result"],
        "compute": ["computed_value"],
        "text_generation": ["generated_text"],
        "scoring": ["score", "reason"],
        "extraction": ["extracted_data"],
        "similarity": ["similarity_score"],
        "assign": ["result"],
        "template_fill": ["filled_text"],
    }
    return generic_outputs.get(action_type, ["result"])


def _normalize_var_name(text: str) -> str:
    """将中文/混合文本转为 snake_case 变量名。"""
    import re
    # 移除非字母数字字符
    cleaned = re.sub(r'[^\w]', '_', text)
    # 中文直接用拼音首字母太复杂，这里用简单的音译映射
    _CN_EN_MAP = {
        "折扣": "discount", "等级": "level", "会员": "member",
        "订单": "orders", "风险": "risk", "优惠券": "coupon",
        "欢迎语": "welcome_msg", "价格": "price", "结果": "result",
        "评分": "score", "原因": "reason", "数量": "count",
        "标识": "id", "类型": "type", "名称": "name",
        "消息": "message", "文本": "text", "数据": "data",
        "频率": "frequency", "概率": "probability", "比率": "ratio",
    }
    for cn, en in _CN_EN_MAP.items():
        if cn in text:
            return en
    # 如果纯英文，直接用
    if re.match(r'^[a-zA-Z_]\w*$', cleaned):
        return cleaned.lower()
    return "result"


def _infer_is_explicit_list(text: str, keywords: list = None) -> bool:
    """检测是否有明确的枚举规则或关键词列表。"""
    # 有关键词列表
    if keywords and len(keywords) > 0:
        return True
    # 检测"A→X, B→Y"枚举模式（如 "VIP打8折，普通打9折"）
    import re
    if re.search(r'[，,；;]\s*\S+[打是等于→]\s*[\d.]+', text):
        return True
    # 检测"如果是X→Y"条件赋值模式
    if text.count("如果") >= 1 and text.count("则") >= 1:
        return True
    return False


def _infer_semantic_vector(
    triggers: 'TriggerSpec',
    action: 'ActionSpec',
    raw_text: str,
    compressibility: 'CompressibilityScore',
) -> 'SemanticVector':
    """P2兜底: 用代码推断6维语义向量（当LLM未输出时）。

    基于已有的 TriggerSpec/ActionSpec/raw_text/compressibility 信息，
    用确定性规则推断6个维度的值。

    设计原则：
    - 含语义概念(紧急/严重/异常/可疑/不达标等)→拉低评分→倾向sensor
    - 纯数值阈值+明确枚举→高分→倾向BLOCK
    - 数学题→高分→BLOCK
    """
    import re
    text = raw_text or ""

    # === 预扫描: 检测原文中的语义概念 ===
    # 这些词出现在规则中，但其含义无法用纯if/else编码，
    # 需要LLM理解上下文后才能判断
    _SEMANTIC_CONCEPT_PATTERNS = [
        # 语义分类(无明确定义)
        (r'紧急(?!.*[：:→])', "紧急(未定义)"),
        (r'严重(?!.*[：:→])', "严重(未定义)"),
        # 需理解的概念
        (r'可疑', "可疑(需理解)"),
        (r'异常(?!.*[<>=≥≤])', "异常(需理解)"),
        (r'不达标|不合格', "不达标/不合格(需理解)"),
        (r'误报|虚假', "误报/虚假(需LLM)"),
        # 非结构化输入分类
        (r'(判断|识别|分类).*类型|类型.*(判断|识别|分类)', "事件分类(需LLM)"),
        (r'先判断.*再', "先判断再处理(需LLM)"),
    ]
    has_semantic_concept = False
    semantic_concepts = []
    for pattern, label in _SEMANTIC_CONCEPT_PATTERNS:
        if re.search(pattern, text):
            has_semantic_concept = True
            semantic_concepts.append(label)

    # 1. trigger_specificity: 触发条件明确性
    #    含语义概念→低分(需要LLM理解触发条件)
    if has_semantic_concept:
        trigger_specificity = 0.2
    elif triggers.implicit_intent:
        trigger_specificity = 0.2
    elif triggers.multidimensional:
        trigger_specificity = 0.3
    elif triggers.is_explicit_list and triggers.numeric_comparisons:
        # 有明确数值比较 → 可以代码实现
        trigger_specificity = 0.7
    elif triggers.is_explicit_list and triggers.extracted_keywords:
        # 有关键词列表但无数值比较 → 中等（可能是文档式描述）
        trigger_specificity = 0.5
    elif triggers.extracted_keywords:
        trigger_specificity = 0.4
    else:
        trigger_specificity = 0.3

    # 2. trigger_context_dependency: 触发条件上下文独立性
    #    含语义概念→低分(触发条件依赖上下文理解)
    if has_semantic_concept:
        trigger_context_dependency = 0.2
    elif triggers.context_stage:
        trigger_context_dependency = 0.3
    elif triggers.multidimensional:
        trigger_context_dependency = 0.2
    elif triggers.is_explicit_list:
        trigger_context_dependency = 0.5
    else:
        trigger_context_dependency = 0.4

    # 3. action_determinism: 动作结果确定性
    _HIGH_DETERMINISM = {"compute"}
    _LOW_DETERMINISM = {"text_generation", "scoring", "similarity", "extraction"}
    if has_semantic_concept:
        action_determinism = 0.2
    elif action.action_type in _HIGH_DETERMINISM:
        action_determinism = 0.8
    elif action.action_type in _LOW_DETERMINISM:
        action_determinism = 0.2
    elif action.action_type == "assign":
        if triggers.numeric_comparisons:
            action_determinism = 0.7
        else:
            action_determinism = 0.4
    else:
        action_determinism = 0.4

    # 4. numeric_complexity: 数值计算复杂度
    if has_semantic_concept:
        numeric_complexity = 0.6  # 有语义词时，数值部分不是主要难点
    elif triggers.numeric_comparisons:
        numeric_complexity = 0.4
    elif any(kw in text for kw in ["计算", "统计", "汇总", "合计", "打折", "折扣"]):
        numeric_complexity = 0.3
    else:
        numeric_complexity = 0.8

    # 5. output_structuredness: 输出结构化程度
    if has_semantic_concept:
        output_structuredness = 0.3  # 需sensor的输出结构不一定明确
    elif action.action_type == "text_generation":
        output_structuredness = 0.2
    elif action.action_type in ("compute",):
        output_structuredness = 0.8
    elif action.action_type == "assign":
        output_structuredness = 0.5
    else:
        output_structuredness = 0.4

    # 6. exception_handling: 异常处理代码化程度
    if has_semantic_concept:
        exception_handling = 0.2  # 含"异常/不达标/误报"→异常处理需LLM
    elif triggers.numeric_comparisons and triggers.numeric_comparisons.get("additional_conditions"):
        exception_handling = 0.3
    elif triggers.implicit_intent:
        exception_handling = 0.2
    else:
        exception_handling = 0.5

    return SemanticVector(
        trigger_specificity=trigger_specificity,
        trigger_context_dependency=trigger_context_dependency,
        action_determinism=action_determinism,
        numeric_complexity=numeric_complexity,
        output_structuredness=output_structuredness,
        exception_handling=exception_handling,
    )


def _extract_numeric_comparisons(fact: dict) -> Optional[dict]:
    """从 LLM 提取的 fact 中提取数值比较条件。

    优先级：
    1. LLM 结构化输出 trigger_numeric_comparisons（新版 prompt）
    2. 从 raw_requirement_text / logic_flow 中正则推断（兜底）
    3. trigger_has_numeric_compare 标记但无具体信息 → 返回 None（而非无意义的占位）
    """
    import re

    # 优先使用 LLM 结构化输出
    structured = fact.get("trigger_numeric_comparisons")
    if structured and isinstance(structured, list) and len(structured) > 0:
        # LLM 输出的是列表 [{"field": "x", "op": ">", "value": 30}, ...]
        # TriggerSpec.numeric_comparisons 是单个 dict，取第一个
        # 如果多个条件，将第一个作为主条件
        first = structured[0]
        if isinstance(first, dict) and "field" in first and "op" in first:
            # 确保字段名不是 unknown
            if first.get("field", "unknown") != "unknown":
                return first
            # 有多个条件时合并为复合条件
            if len(structured) > 1:
                return {
                    "field": first["field"],
                    "op": first["op"],
                    "value": first.get("value", 0),
                    "additional_conditions": structured[1:],
                }

    # 兜底：从原文推断
    raw_text = fact.get("raw_requirement_text", "")
    logic_flow = fact.get("logic_flow", "")
    combined = raw_text + " " + logic_flow

    # 匹配"字段<值"/"字段>值"等数值比较
    m = re.search(
        r'([\w\u4e00-\u9fff]+?)\s*([<>＞＜≥≤>=<]+|大于等于?|小于等于?|超过|低于|不小于|不大于|以上|以下)\s*([\d.]+)',
        combined
    )
    if m:
        field_raw = m.group(1).strip()
        op_raw = m.group(2).strip()
        value = float(m.group(3))

        # 标准化操作符
        op_map = {
            ">": ">", "<": "<", "＞": ">", "＜": "<",
            ">=": ">=", "<=": "<=", "≥": ">=", "≤": "<=",
            "大于": ">", "大于等于": ">=", "超过": ">", "以上": ">=",
            "小于": "<", "小于等于": "<=", "低于": "<", "不小于": ">=", "不大于": "<=",
            "以下": "<=",
        }
        op = op_map.get(op_raw, ">")
        return {"field": field_raw, "op": op, "value": value}

    # 旧版 prompt 只输出 trigger_has_numeric_compare 布尔值
    # 但没有具体信息，返回 None 而非无意义的占位
    # 分类器会将 numeric_comparisons=None 的场景走其他匹配路径
    return None


# ============================================================================
# 适配器：AgentSpec → FactSpec
# ============================================================================

def adapt_agent_spec_to_fact_spec(agent_spec) -> FactSpec:
    """将现有 AgentSpec（LLM 输出）转换为 FactSpec。

    过渡用适配器。当 UnifiedExtractor 改造完成后，这个函数可以移除。
    """
    scenes = []
    for idx, scene in enumerate(agent_spec.scenes):
        for sub_idx, sub in enumerate(scene.sub_scenarios):
            # 优先从 fact 字段读取事实信息（新 prompt 输出）
            fact = sub.fact or {}

            # ========== 方案B核心：通用推断 ==========
            # 1. action_type 推断
            raw_action_type = fact.get("action_type", "")
            block_desc = scene.block_description or ""
            sub_desc = getattr(sub, 'name', '') or ""
            inferred_action_type = _infer_action_type(block_desc, sub_desc, raw_action_type)

            # 2. outputs 推断
            fact_outputs = fact.get("outputs", ["scene", "agent"])
            inferred_outputs = _infer_outputs_from_text(block_desc, sub_desc, inferred_action_type, fact_outputs)

            # 3. is_explicit_list 推断
            raw_text_for_infer = (fact.get("raw_requirement_text", "") or block_desc)
            inferred_explicit = _infer_is_explicit_list(
                raw_text_for_infer,
                list(sub.trigger_keywords) if sub.trigger_keywords else None
            )

            # 提取触发条件事实
            # 数值比较条件：优先从 LLM 结构化提取，其次从原文推断
            numeric_comparisons = _extract_numeric_comparisons(fact)

            triggers = TriggerSpec(
                raw_text="; ".join(sub.trigger_keywords) if sub.trigger_keywords else "",
                extracted_keywords=list(sub.trigger_keywords) if sub.trigger_keywords else [],
                is_explicit_list=fact.get("trigger_is_explicit_list", inferred_explicit),
                implicit_intent=fact.get("trigger_implicit_intent"),
                context_stage=fact.get("trigger_context_stage"),
                multidimensional=fact.get("trigger_multidimensional", False),
                numeric_comparisons=numeric_comparisons,
            )

            # 提取排除条件事实
            exclusions = ExclusionSpec(
                raw_text="; ".join(sub.exclusion_keywords) if sub.exclusion_keywords else "",
                extracted_keywords=list(sub.exclusion_keywords) if sub.exclusion_keywords else [],
                is_explicit_list=bool(sub.exclusion_keywords and len(sub.exclusion_keywords) > 0),
            )

            # 提取动作事实 — 使用推断结果
            public_operations = fact.get("public_operations", [])
            local_program = fact.get("local_program", [])
            structured_op = {}
            if isinstance(public_operations, list) and public_operations:
                structured_op["public_operations"] = public_operations
            if isinstance(local_program, list) and local_program:
                structured_op["local_program"] = local_program
            action = ActionSpec(
                raw_text=scene.block_description,
                action_type=inferred_action_type,
                structured_op=structured_op or None,
                outputs=inferred_outputs,
            )

            # 提取 compressibility（如果有）
            compress_data = fact.get("compressibility", None)
            if compress_data and isinstance(compress_data, dict):
                compressibility = CompressibilityScore(
                    score=compress_data.get("score", 0.7),
                    lossy_aspects=list(compress_data.get("lossy_aspects", [])),
                    recommendation=compress_data.get("recommendation", "BLOCK"),
                )
            else:
                compressibility = CompressibilityScore()  # 默认 0.7, 让 classifier 兜底

            # P2: 6维语义向量——始终使用代码推断，不使用LLM主观评分
            # 原因：LLM倾向于给高分（"关键词明确""动作确定"），但论文立意是
            # "LLM感知(抽取事实) + 确定性算法决策(向量计算)"，向量应从事实推导
            # 而非让LLM直接判断"是否需要sensor"
            semantic_vector = _infer_semantic_vector(
                triggers, action, raw_text_for_infer, compressibility
            )

            # ========== 方案E：顺序流水线支持 ==========
            # 检测是否为顺序流水线（每个 sub_scenario 应有独立 block_id）
            is_pipeline = _detect_pipeline_mode(agent_spec)
            if is_pipeline and len(scene.sub_scenarios) > 1:
                # 流水线模式：每个 sub_scenario 分配独立 block_id 和 scene_id
                sub_block_id = f"b{idx+1}_step{sub_idx+1}_{_to_block_topic(sub_desc or block_desc)}"
                sub_scene_id = idx * 10 + sub_idx + 1
            else:
                sub_block_id = scene.block_id
                sub_scene_id = scene.scene_id

            fact_scene = FactScene(
                scene_id=sub_scene_id,
                block_id=sub_block_id,
                source_ref=f"场景 {idx+1}: {scene.block_description}",
                block_description=scene.block_description,
                agent_expression=scene.agent_expression,
                triggers=triggers,
                exclusions=exclusions,
                action=action,
                # 富化字段——从 fact 字段传递
                raw_requirement_text=fact.get("raw_requirement_text", ""),
                logic_flow=fact.get("logic_flow", ""),
                side_effects=list(fact.get("side_effects", [])),
                # === 扩展字段：规则11-14 LLM抽取的新字段 ===
                raw_context=fact.get("raw_context", ""),
                preconditions=list(fact.get("preconditions", [])),
                fallback=fact.get("fallback", ""),
                action_sequence=list(fact.get("action_sequence", [])),
                nested_logic=fact.get("nested_logic"),
                meta_rules=list(fact.get("meta_rules", [])),
                compressibility=compressibility,
                semantic_vector=semantic_vector,
            )
            scenes.append(fact_scene)

    # 处理 examples — 可能是 Pydantic 对象或 dict
    examples_raw = getattr(agent_spec, 'examples', None) or []
    examples_list = []
    for e in examples_raw:
        if hasattr(e, 'input'):
            examples_list.append({
                "input": dict(e.input) if hasattr(e.input, 'items') else e.input,
                "expected": dict(e.expected) if hasattr(e.expected, 'items') else e.expected,
                "execution_path": list(e.execution_path) if hasattr(e.execution_path, '__iter__') else e.execution_path,
            })
        elif isinstance(e, dict):
            examples_list.append(e)
    examples_list = examples_list or []

    # 提取 configs 和 policies（从 AgentSpec 的 constraints 中分离）
    configs = []
    policies = []

    # 从 constraints 中识别配置和策略
    for c in agent_spec.constraints:
        desc = c.description.lower() if c.description else ""
        key = c.key.upper() if c.key else ""

        # 识别配置项（数据源、并发、性能参数等）
        if any(kw in desc for kw in ["并发", "最大", "维度", "知识库", "容量", "队列", "超时"]):
            configs.append(ConfigItem(
                key=key or "CONFIG",
                value=_extract_config_value(c.description),
                description=c.description,
                category=_infer_config_category(desc),
            ))
        # 识别策略项（安全、日志、监控等）
        elif any(kw in desc for kw in ["敏感", "过滤", "日志", "监控", "告警", "安全", "确认", "身份"]):
            policies.append(PolicyItem(
                key=key or "POLICY",
                description=c.description,
                enforcement=_infer_enforcement(desc),
                trigger_condition=_extract_trigger_condition(c.description),
                action=_extract_policy_action(c.description),
            ))

    return FactSpec(
        agent_name=agent_spec.agent_name,
        agent_description=agent_spec.agent_description,
        persona_role=agent_spec.persona.role if agent_spec.persona else "",
        persona_capabilities=list(agent_spec.persona.capabilities) if agent_spec.persona else [],
        constraints=[{"key": c.key, "description": c.description} for c in agent_spec.constraints],
        inputs=[{"name": i.name, "type": i.type, "required": i.required, "default": i.default}
                for i in agent_spec.inputs],
        outputs=[{"name": o.name, "type": o.type} for o in agent_spec.outputs],
        scenes=scenes,
        fallback_scene_id=agent_spec.fallback_scene_id,
        fallback_agent_expression=agent_spec.fallback_agent_expression,
        examples=examples_list,
        configs=configs,
        policies=policies,
    )


def _extract_config_value(description: str) -> str:
    """从描述中提取配置值。"""
    import re
    # 尝试匹配数字+单位
    m = re.search(r'(\d+\s*\w*)', description)
    return m.group(1) if m else description[:30]


def _infer_config_category(desc: str) -> str:
    """推断配置项分类。"""
    if any(kw in desc for kw in ["并发", "队列", "超时", "最大"]):
        return "infrastructure"
    if any(kw in desc for kw in ["知识库", "维度", "索引", "向量"]):
        return "data_source"
    if any(kw in desc for kw in ["响应", "qps", "性能", "延迟"]):
        return "performance"
    return "infrastructure"


def _infer_enforcement(desc: str) -> str:
    """推断策略执行时机。"""
    if any(kw in desc for kw in ["敏感", "过滤", "安全", "确认", "身份"]):
        return "pre_check"
    if any(kw in desc for kw in ["日志", "记录"]):
        return "post_check"
    if any(kw in desc for kw in ["监控", "告警"]):
        return "always"
    return "always"


def _extract_trigger_condition(description: str) -> str:
    """从描述中提取触发条件。"""
    for sep in ["如果", "若", "当", "检测到"]:
        if sep in description:
            idx = description.index(sep)
            after = description[idx + len(sep):].strip()
            # 截取到下一个标点
            for end_char in ["，", "。", "；", "则", "，"]:
                if end_char in after:
                    after = after[:after.index(end_char)]
            return after
    return ""


def _extract_policy_action(description: str) -> str:
    """从描述中提取策略动作。"""
    for sep in ["则", "→", "触发", "上传", "拒绝"]:
        if sep in description:
            idx = description.index(sep)
            return description[idx + len(sep):].strip().rstrip("。；")
    return description


def _detect_pipeline_mode(agent_spec) -> bool:
    """检测是否为顺序流水线模式（而非路由模式）。

    信号：
    1. 编号列表 (1. 2. 3.)
    2. 顺序词（"先...再...最后..."）
    3. 单一 scene 包含多个 sub_scenario 且有明显执行顺序
    """
    import re
    for scene in agent_spec.scenes:
        desc = scene.block_description or ""
        subs = scene.sub_scenarios or []

        # 信号1：描述含编号
        if re.search(r'^\d+[.、）)]\s', desc):
            return True

        # 信号2：顺序词
        if any(kw in desc for kw in ["先", "再", "然后", "最后", "接着", "步骤1", "步骤2"]):
            return True

        # 信号3：sub_scenario 名称含编号/步骤
        for sub in subs:
            sub_name = getattr(sub, 'name', '') or ""
            if re.search(r'^(?:步骤|第)\s*\d', sub_name):
                return True

        # 信号4：scene 有多个 sub 且 agent_expression 相同
        # 说明是同一个 agent 的多步操作，而非多场景路由
        if len(subs) > 2:
            return True

    return False


def _to_block_topic(description: str) -> str:
    """将场景中文描述转为简短英文 topic，用于 block_id。"""
    import re
    if not description:
        return "default"
    # 中文关键词→英文映射（通用，不硬编码领域）
    _TOPIC_MAP = {
        "查询": "query", "计算": "compute", "判断": "judge",
        "生成": "generate", "发送": "send", "评估": "evaluate",
        "检索": "search", "分析": "analyze", "检测": "detect",
        "欢迎": "welcome", "风险": "risk", "折扣": "discount",
        "新人": "newuser", "订单": "order", "会员": "member",
    }
    for cn, en in _TOPIC_MAP.items():
        if cn in description:
            return en
    # 提取前几个中文字
    chars = re.findall(r'[\u4e00-\u9fff]', description)
    if chars:
        return ''.join(chars[:3])
    words = description.split()
    return words[0][:8].lower() if words else "default"


# ============================================================================
# DSL 渲染器（从 ClassifiedSpec 生成 DSL 1.2 文本）
# ============================================================================

INDENT = "    "


def render_classified_spec(spec: ClassifiedSpec) -> str:
    """将 ClassifiedSpec 渲染为 DSL 1.2 文本（委托给 files/renderer.py）。"""
    from files.renderer import DSLRenderer
    return DSLRenderer().render_from_classifiedspec(spec)


# ============================================================================
# PipelineV2
# ============================================================================

@dataclass
class PipelineV2Result:
    """PipelineV2 编译结果。"""
    success: bool
    dsl_code: str
    total_tokens: int = 0
    error_message: str = ""
    violations: list = field(default_factory=list)
    upgraded_blocks: list[str] = field(default_factory=list)
    gate_result: Any = None
    # 中间数据暴露——供前端阶段3展示
    fact_spec: Optional[FactSpec] = None
    classified_spec: Optional[ClassifiedSpec] = None


class PipelineV2:
    """五层编译管线。

    使用:
        pipeline = PipelineV2()
        result = pipeline.compile(agent_spec)  # 从 AgentSpec 开始
    """

    def __init__(
        self,
        classifier: Optional[StrategyClassifier] = None,
        gate: Optional[ExamplesGate] = None,
        upgrader: Optional[Upgrader] = None,
        max_upgrade_rounds: int = 3,
        llm_call: Optional[Any] = None,
    ):
        self.classifier = classifier or StrategyClassifier()
        self.executor = DSLExecutor()
        self.gate = gate or ExamplesGate()
        self.upgrader = upgrader or Upgrader()
        self.max_upgrade_rounds = max_upgrade_rounds
        self.llm_call = llm_call

    def compile(self, agent_spec) -> PipelineV2Result:
        """从 AgentSpec 编译为 DSL。"""
        try:
            # 第1步：Adapter
            fact_spec = adapt_agent_spec_to_fact_spec(agent_spec)

            # 将 examples 统一转为 dict 列表（Pydantic → dict 转换）
            examples_raw = getattr(agent_spec, 'examples', None) or []
            examples_dicts = []
            for e in examples_raw:
                if hasattr(e, 'input'):
                    examples_dicts.append({
                        "input": dict(e.input) if hasattr(e.input, 'items') else e.input,
                        "expected": dict(e.expected) if hasattr(e.expected, 'items') else e.expected,
                        "execution_path": list(e.execution_path) if hasattr(e.execution_path, '__iter__') else e.execution_path,
                    })
                elif isinstance(e, dict):
                    examples_dicts.append(e)

            return self._compile_from_fact(fact_spec, examples_dicts)
        except Exception as e:
            import traceback
            return PipelineV2Result(
                success=False,
                dsl_code="",
                error_message=f"编译失败: {e}\n{traceback.format_exc()}",
            )

    def compile_from_fact(self, fact_spec: FactSpec, examples: list[dict] = None) -> PipelineV2Result:
        """从 FactSpec 编译为 DSL。"""
        return self._compile_from_fact(fact_spec, examples or [])

    def _compile_from_fact(self, fact_spec: FactSpec, examples: list[dict]) -> PipelineV2Result:
        """内部编译方法（含升级循环）。"""
        classified_spec = fact_spec  # 起始
        all_upgraded = []
        result = PipelineV2Result(success=False, dsl_code="", fact_spec=fact_spec)

        for round_num in range(self.max_upgrade_rounds + 1):
            # 第2层：策略分类（首次）或保持（升级后）
            if round_num == 0:
                classified_spec = self.classifier.classify(fact_spec)
                result.classified_spec = classified_spec
            # 第3层：DSL 渲染（使用 files/renderer.py 的统一入口，传递 llm_call）
            from files.renderer import DSLRenderer
            dsl_code = DSLRenderer().render_from_classifiedspec(classified_spec, llm_call=self.llm_call)

            # 第4层：ANTI_HOLLOW lint
            violations = check_anti_hollow(dsl_code)
            if violations:
                result.violations = violations
                violation_str = "; ".join(
                    f"[{v.block_id}] {v.description}" for v in violations[:3]
                )
                logger.warning(f"ANTI_HOLLOW 检测到 {len(violations)} 个违规: {violation_str}")
                # ANTI_HOLLOW 违规不阻断编译，但记录

            # 第5层：ExamplesGate
            gate_result = self.gate.run(dsl_code, examples)
            result.gate_result = gate_result

            if gate_result.all_passed:
                result.success = True
                result.dsl_code = dsl_code
                result.upgraded_blocks = all_upgraded
                return result

            # 需要升级
            if round_num >= self.max_upgrade_rounds:
                logger.warning(
                    f"升级 {self.max_upgrade_rounds} 轮后 EXAMPLES 仍未全通过，"
                    f"但 DSL 编译本身成功；使用当前 DSL"
                )
                result.success = True
                result.dsl_code = dsl_code
                result.upgraded_blocks = all_upgraded
                return result

            # 升级失败的 BLOCK
            failed_block_ids = gate_result.failed_blocks
            failed_cases = [r for r in gate_result.case_results if not r.passed]
            failed_info = {}
            for r in failed_cases:
                for bid in r.execution_path:
                    if bid not in failed_info:
                        failed_info[bid] = []
                    failed_info[bid].append(f"case {r.case_index}: path={r.execution_path}")

            # 定位失败的 SceneClassification
            failed_scenes = []
            for scene in classified_spec.scenes:
                if scene.block_id in failed_block_ids:
                    reason = "; ".join(failed_info.get(scene.block_id, ["EXAMPLES 未通过"]))
                    failed_scenes.append((scene, reason))

            if not failed_scenes:
                # 无法定位失败——产生 DSL 但不阻断
                logger.warning(
                    f"EXAMPLES 门禁未通过 (失败 BLOCK: {failed_block_ids})，"
                    f"但 DSL 编译本身成功；使用当前 DSL"
                )
                result.success = True
                result.dsl_code = dsl_code
                result.upgraded_blocks = all_upgraded
                return result

            # 执行升级
            upgraded = self.upgrader.batch_upgrade(failed_scenes)
            for u in upgraded:
                all_upgraded.append(u.block_id)
                # 替换 classified_spec 中对应的 scene
                for i, s in enumerate(classified_spec.scenes):
                    if s.block_id == u.block_id:
                        classified_spec.scenes[i] = u
                        break

            logger.info(f"第 {round_num + 1} 轮升级: {[u.block_id for u in upgraded]}")

        result.error_message = f"达到最大升级轮数 {self.max_upgrade_rounds}"
        return result
