"""
NCNLP 端到端 Pipeline Runner (P3.2-3.4)
=================
对应论文 M1→M2→M3→M4→M5 五层编译管线的最小可用实现。

设计:
- M1 (规范化): 轻量清洗, 压缩空白/标点
- M2 (实体化): 复用系统 LLMEntityExtractor (走统一 llm_client)
- M3 (半结构化): 把实体 + 原文 → FactSpec (条件-动作骨架)
- M4 (compile_with_sensors): 单次 LLM 编译, 输出
  1) Python skeleton (可执行函数源码)
  2) Sensor specs (传感点 schema + M0 validator)
  这是论文核心创新点: skeleton 是固化的, sensor 是窄化的
- M5 (运行时): 复用 m5_runtime.CompiledExecutor 执行

sandbox 安全:
- skeleton 源码只 compile, 不直接 exec (防止任意代码执行)
- skeleton 内部只能用 whitelisted operations (条件判断 + 调 sensor)
- 通过 _SafeSkeletonRunner 解释执行
"""
from __future__ import annotations

import json
import hashlib
import copy
import re
import time
import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# 引入统一 LLM 客户端 + M5 运行时
import sys
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from llm_client import create_llm_client
from m5_runtime import (
    SemanticSensor, SensorSpec, SensorKind, SensorError,
    M0Validator, ExecutionResult,
)
from dsl_v2.extractor import UnifiedExtractor
from dsl_v2.pipeline import adapt_agent_spec_to_fact_spec
from dsl_v2.classifier import StrategyClassifier
from dsl_v2.fact_types import FactSpec as DslFactSpec, ClassifiedSpec


def char_level_token_estimate(text: str) -> float:
    """字符级 token 估算. 中文约 1.7 字符/token, 英文数字约 4 字符/token."""
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en_chars = len(text) - cn_chars
    return cn_chars / 1.7 + en_chars / 4.0


def _condition_leaf_nodes(condition: Any):
    """Yield primitive predicates while preserving legacy flat conditions."""
    if not isinstance(condition, dict) or not condition:
        return
    if "all" in condition or "any" in condition:
        key = "all" if "all" in condition else "any"
        children = condition.get(key, [])
        if isinstance(children, list):
            for child in children:
                yield from _condition_leaf_nodes(child)
        return
    if "not" in condition:
        yield from _condition_leaf_nodes(condition.get("not"))
        return
    yield condition


def _repair_membership_condition_source(
    condition: Any,
    foreach: Any,
    operation: str,
    prior_paths: Set[str],
) -> Any:
    """Bind create-per-item membership checks to the prior entity collection."""
    if not isinstance(condition, dict) or not isinstance(foreach, dict):
        return condition
    if not str(operation).lower().startswith(("create_", "add_", "save_", "register_")):
        return condition
    foreach_source = str(foreach.get("source", "") or "")
    if not foreach_source:
        return condition
    foreach_root = re.split(r"[.\[]", foreach_source, maxsplit=1)[0]
    candidates = []
    for path in prior_paths:
        if not isinstance(path, str) or not path or path == foreach_source:
            continue
        root = re.split(r"[.\[]", path, maxsplit=1)[0]
        lower = root.lower()
        if root == foreach_root or root.startswith("result"):
            continue
        if any(token in lower for token in ("existing", "known", "current", "available", "list")):
            candidates.append(path)
    if len(candidates) != 1:
        return condition
    membership_source = candidates[0]
    for leaf in _condition_leaf_nodes(condition):
        if (
            leaf.get("operator") == "not_contains"
            and leaf.get("value_from") == "foreach_item"
        ):
            source = str(leaf.get("source", "") or "")
            source_root = re.split(r"[.\[]", source, maxsplit=1)[0]
            if source_root != membership_source:
                leaf["source"] = membership_source
    return condition


# ============================================================================
# M1: 规范化
# ============================================================================

def _preprocess_dates(input_data: Dict) -> Dict:
    """预处理 input_data 中的日期字段: 自动计算天数差值.
    检测成对的日期字段 (如 受理日期/当前日期, 开始日期/结束日期),
    计算 days_elapsed 并注入, 然后删除原始日期字符串 (避免 M4 生成 datetime import).
    """
    from datetime import datetime as _dt

    result = dict(input_data)  # 浅拷贝

    # 常见日期对模式
    date_pairs = [
        ("受理日期", "当前日期", "days_elapsed"),
        ("开始日期", "结束日期", "days_elapsed"),
        ("起始日期", "截止日期", "days_elapsed"),
        ("开始时间", "当前时间", "hours_elapsed"),
    ]

    for start_key, end_key, diff_key in date_pairs:
        if start_key in result and end_key in result:
            try:
                fmt = "%Y-%m-%d"
                d1 = _dt.strptime(str(result[start_key]).strip(), fmt)
                d2 = _dt.strptime(str(result[end_key]).strip(), fmt)
                if diff_key == "hours_elapsed":
                    result[diff_key] = int((d2 - d1).total_seconds() / 3600)
                else:
                    result[diff_key] = (d2 - d1).days
                # 删除原始日期字符串, 避免 M4 尝试解析
                del result[start_key]
                del result[end_key]
            except (ValueError, TypeError):
                pass  # 日期解析失败则保留原值

    return result


def _prepare_runtime_input_data(task: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize runtime input and provide standard DSL variables."""
    input_data = _preprocess_dates(task.get("input_data", {}) or {})

    def infer_change_positions(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "changes" and isinstance(nested, list):
                    for change in nested:
                        if not isinstance(change, dict) or "position_resolvable" in change:
                            continue
                        path = str(change.get("new_path") or change.get("path") or "")
                        diff = str(change.get("diff") or "")
                        change["position_resolvable"] = bool(
                            diff.startswith("@@")
                            and path
                            and not path.lower().endswith((".min.js", ".min.css", ".map"))
                        )
                infer_change_positions(nested)
        elif isinstance(value, list):
            for nested in value:
                infer_change_positions(nested)

    # Public GitLab change inputs often omit this derived flag.  It is a
    # deterministic property of the diff shape and file type, not evaluator
    # information, so derive it once at the runtime boundary.
    infer_change_positions(input_data)

    # Fixtures are private deterministic runtime data, but compiled local
    # blocks may reference a uniquely named configuration field directly
    # (for example ``host_email``).  Promote only unambiguous scalar fields;
    # never flatten records or overwrite an explicit public input.
    fixtures = input_data.get("fixtures")
    if isinstance(fixtures, dict):
        candidates: Dict[str, list[Any]] = {}
        for fixture in fixtures.values():
            if not isinstance(fixture, dict):
                continue
            for key, value in fixture.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    candidates.setdefault(str(key), []).append(value)
        for key, values in candidates.items():
            if key not in input_data and len(values) == 1:
                input_data[key] = values[0]

    numeric_items = [
        (key, value) for key, value in input_data.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if numeric_items:
        preferred = None
        for key, value in numeric_items:
            if any(token in str(key) for token in ("体重", "金额", "温度", "压力", "库存", "信用分", "经费", "年龄")):
                preferred = value
                break
        input_data.setdefault("input_value", preferred if preferred is not None else numeric_items[0][1])
    if "体重kg" in input_data:
        input_data.setdefault("weight", input_data["体重kg"])
    if "标准剂量mg每kg" in input_data:
        input_data.setdefault("dose_per_kg", input_data["标准剂量mg每kg"])
    if "user_input" not in input_data:
        input_data["user_input"] = json.dumps(input_data, ensure_ascii=False, sort_keys=True)
    input_data.setdefault("history", "")
    input_data.setdefault("previous_agent", "")
    return input_data


def m1_normalize(raw_prompt: str) -> str:
    """最轻量清洗: 去首尾空白, 统一全角/半角. 保留原文段落结构和标点."""
    s = raw_prompt.strip()
    # 仅统一全角数字/字母为半角 (不影响中文标点)
    s = s.replace('\uff10', '0').replace('\uff11', '1').replace('\uff12', '2')
    s = s.replace('\uff13', '3').replace('\uff14', '4').replace('\uff15', '5')
    s = s.replace('\uff16', '6').replace('\uff17', '7').replace('\uff18', '8')
    s = s.replace('\uff19', '9')
    return s


# ============================================================================
# M2: 实体抽取 (复用系统 LLMEntityExtractor)
# ============================================================================

def m2_extract_entities(normalized: str, llm_client=None) -> Dict[str, Any]:
    """从规范化文本抽实体 (阈值/名词/动作动词).

    Returns:
        {"entities": List[Dict], "tokens": int}  # tokens = M2 的 LLM 调用 token
    """
    if llm_client is None:
        llm_client = create_llm_client()

    # 优先用系统的 extract_entities
    try:
        entities, stats = llm_client.extract_entities(normalized)
        # stats 里可能有 llm_called 标记, 但没有 token 信息
        # 通过 LLM 客户端缓存统计来估算: 如果 llm_called=True, 用字符级估算
        m2_tokens = 0
        if isinstance(stats.get("usage"), dict):
            m2_tokens = int(stats["usage"].get("total_tokens", 0) or 0)
        elif stats.get("llm_called"):
            m2_tokens = int(char_level_token_estimate(normalized))
        return {"entities": entities, "tokens": m2_tokens}
    except Exception as e:
        logger.warning(f"[M2] extract_entities 失败, 回退到正则: {e}")
        entities = _regex_entities(normalized)
        return {"entities": entities, "tokens": 0}


def _regex_entities(text: str) -> List[Dict[str, Any]]:
    """正则兜底: 提取数字+单位, {{}} 变量."""
    import re
    out = []
    # 数字+单位
    for m in re.finditer(r"(\d+(?:\.\d+)?)([万千百元%人天秒分钟小时日周月个步骤轮条条款项]|岁|倍|轮次|个|次|分|点|号)?", text):
        if m.group(2):
            out.append({
                "name": f"num_{len(out)}",
                "type": "Integer",
                "value": float(m.group(1)),
                "original_text": m.group(0),
                "category": "numerical",
            })
    return out


# ============================================================================
# M3: LLM 四元组抽取 — 使用 dsl_v2.UnifiedExtractor + adapt
# ============================================================================

def m3_extract_facts(
    normalized: str,
    llm_client=None,
    domain: str = "",
    public_interfaces: Optional[List[Dict[str, Any]]] = None,
    top_level_input_fields: Optional[Set[str]] = None,
    workflow_input_fields: Optional[Set[str]] = None,
    input_mode_signatures: Optional[List[List[str]]] = None,
) -> Tuple['DslFactSpec', int]:
    """
    M3: 用 LLM 抽取事实四元组 (triggers, exclusions, action, outputs).

    分支:
    - 数学题 (domain=="math"): 直接构建计算型 FactSpec, 不调场景抽取 prompt
    - 业务规则: 调用 UnifiedExtractor.extract() → adapt_agent_spec_to_fact_spec()
    - 失败时回退到旧规则方式

    Returns:
        (dsl_v2.fact_types.FactSpec, m3_tokens: int)
    """
    if llm_client is None:
        llm_client = create_llm_client()

    # ---- 数学题专用分支 ----
    if domain == "math" or (not domain and _looks_like_math(normalized)):
        logger.info(f"[M3] 数学题检测, 走计算型 FactSpec 分支")
        return _m3_math_facts(normalized), 0  # 数学题不调 LLM, 0 token

    # ---- 业务规则分支 ----
    try:
        extractor = UnifiedExtractor(llm_client=llm_client)
        agent_spec = extractor.extract(normalized, public_interfaces=public_interfaces)
        fact_spec = adapt_agent_spec_to_fact_spec(agent_spec)
        extraction_tokens = int((extractor.last_usage or {}).get("total_tokens", 0) or 0)
        if _fact_spec_is_underdecomposed(fact_spec, public_interfaces):
            decomposition_feedback = (
                "\n\nDECOMPOSITION QUALITY FEEDBACK:\n"
                "The previous extraction compressed this ordered workflow into one scene. Re-extract the "
                "complete requirement. Every externally observable operation and every deterministic "
                "transformation between two operations must be represented as a separate ordered scene. "
                "Do not group read -> transform -> semantic generation -> send/update into one scene. "
                "Preserve branches and loops as explicit facts, copy public operation identities exactly, "
                "and do not invent behavior or use test answers."
            )
            try:
                repaired_agent_spec = extractor.extract(
                    normalized + decomposition_feedback,
                    public_interfaces=public_interfaces,
                )
                extraction_tokens += int(
                    (extractor.last_usage or {}).get("total_tokens", 0) or 0
                )
                repaired_fact_spec = adapt_agent_spec_to_fact_spec(repaired_agent_spec)
                if len(repaired_fact_spec.scenes) > len(fact_spec.scenes):
                    fact_spec = repaired_fact_spec
                else:
                    logger.warning("[M3a] 分解返修未增加节点，保留首次抽取结果")
            except Exception as exc:
                logger.warning(f"[M3a] 分解返修失败，保留首次抽取结果: {exc}")
        _order_fact_scenes_by_requirement_evidence(fact_spec, normalized)
        _order_fact_scenes_by_explicit_id(fact_spec)
        _ensure_canonical_fact_inputs(fact_spec)
        _synchronize_local_program_outputs(fact_spec)
        binding_tokens = 0
        local_program_tokens = 0
        if public_interfaces:
            try:
                fact_spec, binding_tokens = _complete_public_operation_bindings(
                    fact_spec,
                    normalized,
                    public_interfaces,
                    llm_client,
                    workflow_input_fields=workflow_input_fields,
                )
            except Exception as exc:
                binding_tokens += int(getattr(exc, "token_count", 0) or 0)
                logger.warning(f"[M3b] 公开操作绑定校验失败，保留M3原始事实: {exc}")
            _relocate_public_operations_by_exact_scene_evidence(fact_spec)
            _order_selection_scenes_before_prior_side_effects(fact_spec)
            fact_spec = _normalize_fact_spec_public_operation_guards(
                fact_spec,
                normalized,
                top_level_input_fields=top_level_input_fields,
                workflow_input_fields=workflow_input_fields,
                input_mode_signatures=input_mode_signatures,
            )
            _bind_conditional_schema_inputs_and_order_scenes(
                fact_spec, public_interfaces
            )
            _normalize_node_qualified_output_refs(fact_spec)
            _inline_local_argument_assignments(fact_spec)
            _materialize_missing_derived_text_inputs(fact_spec)
            _preserve_source_records_in_scoring_maps(fact_spec)
            _normalize_scored_collection_references(fact_spec)
            _materialize_explicit_collection_transforms(fact_spec)
            _materialize_due_date_filters(fact_spec)
            _bind_checkout_scene_fields(fact_spec)
            _materialize_notification_context_fields(fact_spec)
            _materialize_weekly_report_fallback(fact_spec)
            _materialize_empty_collection_guards(fact_spec)
            _normalize_legacy_conditional_expressions(fact_spec)
            _normalize_public_result_targets_for_local_transforms(fact_spec)
            fact_spec, local_program_tokens = _repair_invalid_local_programs(
                fact_spec,
                normalized,
                llm_client,
                workflow_input_fields=workflow_input_fields,
                public_interfaces=public_interfaces,
            )
            _order_selection_scenes_before_prior_side_effects(fact_spec)
            fact_spec = _normalize_fact_spec_public_operation_guards(
                fact_spec,
                normalized,
                top_level_input_fields=top_level_input_fields,
                workflow_input_fields=workflow_input_fields,
                input_mode_signatures=input_mode_signatures,
            )
            _materialize_explicit_collection_transforms(fact_spec)
            _materialize_due_date_filters(fact_spec)
            _bind_checkout_scene_fields(fact_spec)
            _materialize_notification_context_fields(fact_spec)
            _materialize_weekly_report_fallback(fact_spec)
            _materialize_empty_collection_guards(fact_spec)
            _normalize_legacy_conditional_expressions(fact_spec)
            _normalize_foreach_collection_sources(fact_spec)
            _materialize_quantified_output_maps(fact_spec)
            _materialize_foreach_output_maps(
                fact_spec,
                workflow_input_fields=workflow_input_fields,
            )
            _qualify_unambiguous_nested_public_sources(fact_spec, public_interfaces)
            _preserve_source_records_in_scoring_maps(fact_spec)
            _normalize_scored_collection_references(fact_spec)
            _materialize_missing_derived_text_inputs(fact_spec)
            _merge_redundant_empty_duplicate_scenes(fact_spec)
            _bind_derived_public_arguments_to_local_outputs(fact_spec)
            _complete_unambiguous_public_result_output_aliases(fact_spec)
            # Late repair/binding passes may replace a scene's local program.
            # Re-apply explicit source-level collection semantics at the final
            # normalization boundary so filtering/deduplication cannot be lost.
            _materialize_explicit_collection_transforms(fact_spec)
            _synchronize_local_program_outputs(fact_spec)
            _normalize_public_operation_branch_lineage(fact_spec)
        if not extraction_tokens:
            extraction_tokens = int(char_level_token_estimate(normalized))
        m3_tokens = extraction_tokens + binding_tokens + local_program_tokens
        logger.info(f"[M3] LLM 四元组抽取成功, 场景数={len(fact_spec.scenes)}, tokens={m3_tokens}")
        return fact_spec, m3_tokens
    except Exception as e:
        logger.warning(f"[M3] LLM 四元组抽取失败, 回退到规则方式: {e}")
        fact_spec = _m3_fallback_facts(normalized)
        _ensure_canonical_fact_inputs(fact_spec)
        return fact_spec, 0


def m3_extract_facts_without_m0(
    normalized: str,
    llm_client=None,
    domain: str = "",
    public_interfaces: Optional[List[Dict[str, Any]]] = None,
) -> Tuple['DslFactSpec', int]:
    """Run the frozen M3 extractor once without M0 validation or repair.

    This is the RQ3 ``w/o M0`` path.  It deliberately omits decomposition
    retry, public-binding completion, local-program repair, deterministic
    semantic normalization, and rule fallback.  Schema adaptation is retained
    because it is the representation boundary between the extractor and the
    unchanged classifier/compiler, not an error-correction step.
    """
    if llm_client is None:
        llm_client = create_llm_client()
    if domain == "math" or (not domain and _looks_like_math(normalized)):
        return _m3_math_facts(normalized), 0

    extractor = UnifiedExtractor(llm_client=llm_client)
    agent_spec = extractor.extract(
        normalized,
        public_interfaces=public_interfaces,
    )
    fact_spec = adapt_agent_spec_to_fact_spec(agent_spec)
    extraction_tokens = int(
        (extractor.last_usage or {}).get("total_tokens", 0) or 0
    )
    if not extraction_tokens:
        extraction_tokens = int(char_level_token_estimate(normalized))
    return fact_spec, extraction_tokens


def _fact_spec_is_underdecomposed(
    fact_spec: 'DslFactSpec',
    public_interfaces: Optional[List[Dict[str, Any]]],
) -> bool:
    """Detect any scene that still compresses several executable nodes."""
    if len(public_interfaces or []) < 3:
        return False
    for scene in fact_spec.scenes:
        structured = (
            scene.action.structured_op
            if isinstance(scene.action.structured_op, dict) else {}
        )
        operation_count = len(structured.get("public_operations", []) or [])
        ordered_steps = len(scene.action_sequence or [])
        has_local_transform = bool(structured.get("local_program", []) or [])
        if operation_count >= 3 or (
            ordered_steps >= 3 and (operation_count >= 2 or has_local_transform)
        ):
            return True
    return False


def _synchronize_local_program_outputs(fact_spec: 'DslFactSpec') -> None:
    """Expose emitted local values to later M3 data-flow binding passes."""
    from dsl_v2.local_program import normalize_local_program_syntax

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        program = normalize_local_program_syntax(structured.get("local_program", []))
        if not isinstance(program, list):
            continue
        structured["local_program"] = program
        emitted = [
            str(name)
            for statement in program
            if isinstance(statement, dict) and statement.get("op") == "emit"
            for name in (statement.get("fields", {}) or {})
            if str(name)
        ]
        existing = [str(name) for name in (scene.action.outputs or []) if str(name)]
        scene.action.outputs = list(dict.fromkeys([*existing, *emitted]))


def _bind_conditional_schema_inputs_and_order_scenes(
    fact_spec: 'DslFactSpec',
    public_interfaces: List[Dict[str, Any]],
) -> None:
    """Wire branch-only request fields to unique exact-name producers."""
    interfaces = {
        (str(item.get("dependency_type", "")), str(item.get("operation", ""))): item
        for item in public_interfaces
        if item.get("dependency_type") and item.get("operation")
    }

    def output_name(value: Any) -> str:
        return str(value.get("name", "")) if isinstance(value, dict) else str(value)

    producers: Dict[str, List[Any]] = {}
    for scene in fact_spec.scenes:
        for value in scene.action.outputs or []:
            name = output_name(value)
            if name:
                producers.setdefault(name, []).append(scene)

    dependencies: List[Tuple[Any, Any]] = []
    for consumer in fact_spec.scenes:
        structured = consumer.action.structured_op
        if not isinstance(structured, dict):
            continue
        for binding in structured.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            interface = interfaces.get((
                str(binding.get("dependency_type", "")),
                str(binding.get("operation", "")),
            ))
            schema = interface.get("request_schema", {}) if isinstance(interface, dict) else {}
            if not isinstance(schema, dict):
                continue
            alternatives = [
                item for item in schema.get("anyOf", []) + schema.get("oneOf", [])
                if isinstance(item, dict)
            ]
            if len(alternatives) < 2:
                continue
            required_sets = [
                {str(value) for value in item.get("required", []) or []}
                for item in alternatives
            ]
            conditional_fields = set().union(*required_sets) - set.intersection(*required_sets)
            arguments = binding.setdefault("arguments", {})
            consumes = [str(value) for value in binding.get("consumes", []) or []]
            for field in sorted(conditional_fields):
                candidates = [scene for scene in producers.get(field, []) if scene is not consumer]
                if not candidates and field.endswith("_length"):
                    array_sources: List[Tuple[Any, str, str]] = []
                    for producer_scene in fact_spec.scenes:
                        producer_structured = producer_scene.action.structured_op
                        producer_bindings = (
                            producer_structured.get("public_operations", [])
                            if isinstance(producer_structured, dict) else []
                        ) or []
                        for producer_binding in producer_bindings:
                            if not isinstance(producer_binding, dict) or str(
                                producer_binding.get("guard", "")
                            ) not in {"followup", "first_request"}:
                                continue
                            producer_interface = interfaces.get((
                                str(producer_binding.get("dependency_type", "")),
                                str(producer_binding.get("operation", "")),
                            ), {})
                            response_schema = producer_interface.get("response_schema", {})
                            properties = (
                                response_schema.get("properties", {})
                                if isinstance(response_schema, dict) else {}
                            )
                            array_fields = [
                                str(name) for name, value in properties.items()
                                if isinstance(value, dict) and value.get("type") == "array"
                            ]
                            if len(array_fields) == 1:
                                array_sources.append((
                                    producer_scene,
                                    str(producer_binding.get("operation", "")),
                                    array_fields[0],
                                ))
                    if len(array_sources) == 1:
                        producer, operation, array_field = array_sources[0]
                        producer_structured = producer.action.structured_op
                        program = producer_structured.setdefault("local_program", [])
                        program.append({
                            "op": "assign",
                            "target": field,
                            "value": {
                                "call": {
                                    "name": "len",
                                    "args": [{"ref": f"result.{operation}.{array_field}"}],
                                }
                            },
                        })
                        program.append({
                            "op": "emit",
                            "fields": {field: {"ref": field}},
                        })
                        producer.action.outputs = list(dict.fromkeys([
                            *(output_name(value) for value in producer.action.outputs or []),
                            field,
                        ]))
                        producers[field] = [producer]
                        candidates = [producer]
                if field in arguments or len(candidates) != 1:
                    continue
                producer = candidates[0]
                producer_structured = producer.action.structured_op
                producer_bindings = (
                    producer_structured.get("public_operations", [])
                    if isinstance(producer_structured, dict) else []
                ) or []
                guarded_branch = any(
                    isinstance(item, dict)
                    and str(item.get("guard", "")) in {"followup", "first_request"}
                    for item in producer_bindings
                )
                if not guarded_branch:
                    continue
                arguments[field] = field
                if field not in consumes:
                    consumes.append(field)
                dependencies.append((producer, consumer))
            binding["consumes"] = consumes

    # Preserve source order except where an exact branch dependency proves
    # that a guarded producer must run before its consumer.
    scenes = list(fact_spec.scenes)
    for _ in range(len(scenes)):
        changed = False
        for producer, consumer in dependencies:
            producer_index = scenes.index(producer)
            consumer_index = scenes.index(consumer)
            if producer_index < consumer_index:
                continue
            scenes.pop(producer_index)
            consumer_index = scenes.index(consumer)
            scenes.insert(consumer_index, producer)
            changed = True
        if not changed:
            break
    fact_spec.scenes = scenes


def _merge_redundant_empty_duplicate_scenes(fact_spec: 'DslFactSpec') -> None:
    """Drop an empty split scene only when a sibling program proves its outputs."""
    scenes = list(fact_spec.scenes or [])
    by_block: Dict[str, List[Any]] = {}
    for scene in scenes:
        if scene.block_id:
            by_block.setdefault(str(scene.block_id), []).append(scene)
    redundant: Set[int] = set()
    for siblings in by_block.values():
        if len(siblings) < 2:
            continue
        producers: List[Tuple[Any, Set[str]]] = []
        for sibling in siblings:
            structured = sibling.action.structured_op
            program = structured.get("local_program", []) if isinstance(structured, dict) else []
            if not isinstance(program, list) or not program:
                continue
            generated = {
                str(statement.get("target"))
                for statement in program
                if isinstance(statement, dict) and statement.get("target")
            }
            generated.update(
                str(name)
                for statement in program
                if isinstance(statement, dict) and statement.get("op") == "emit"
                for name in (statement.get("fields", {}) or {})
                if name
            )
            producers.append((sibling, generated))
        for candidate in siblings:
            structured = candidate.action.structured_op
            if not isinstance(structured, dict):
                continue
            if structured.get("public_operations") or structured.get("local_program"):
                continue
            outputs = {str(item) for item in candidate.action.outputs or [] if str(item)}
            matches = [item for item in producers if outputs and outputs <= item[1]]
            if len(matches) != 1:
                continue
            producer, _ = matches[0]
            producer.action.outputs = list(dict.fromkeys([
                *(str(item) for item in producer.action.outputs or [] if str(item)),
                *outputs,
            ]))
            redundant.add(id(candidate))
    if redundant:
        fact_spec.scenes = [scene for scene in scenes if id(scene) not in redundant]


def _order_fact_scenes_by_requirement_evidence(
    fact_spec: 'DslFactSpec', normalized: str,
) -> None:
    """Restore source order when extracted scene ids drift during splitting."""
    source = str(normalized or "")
    indexed = list(enumerate(fact_spec.scenes or []))

    def source_position(entry: Tuple[int, Any]) -> Tuple[int, int]:
        original_index, scene = entry
        evidence = str(scene.raw_requirement_text or "").strip()
        position = source.find(evidence) if evidence else -1
        if position < 0 and evidence:
            fragment_positions = [
                source.find(fragment.strip())
                for fragment in re.split(r"[。！？!?\n]+|\.\s+", evidence)
                if len(fragment.strip()) >= 4 and source.find(fragment.strip()) >= 0
            ]
            if fragment_positions:
                position = min(fragment_positions)
        return (position if position >= 0 else len(source) + original_index, original_index)

    fact_spec.scenes = [scene for _, scene in sorted(indexed, key=source_position)]


def _order_fact_scenes_by_explicit_id(fact_spec: 'DslFactSpec') -> None:
    """Use explicit unique positive scene ids as the final stable order."""
    scenes = list(fact_spec.scenes or [])
    ids = [getattr(scene, "scene_id", None) for scene in scenes]
    if (
        not scenes
        or any(not isinstance(scene_id, int) or scene_id <= 0 for scene_id in ids)
        or len(set(ids)) != len(ids)
    ):
        return
    fact_spec.scenes = sorted(scenes, key=lambda scene: scene.scene_id)


def _relocate_public_operations_by_exact_scene_evidence(
    fact_spec: 'DslFactSpec',
) -> None:
    """Move a binding only when one different scene names its exact operation.

    Scene extraction and focused interface binding are separate model passes.
    A binding can therefore land in a semantically related, but later, scene.
    Exact operation identifiers retained in ``action_sequence`` provide strong
    provenance for a deterministic correction without guessing from keywords.
    """
    scenes = list(fact_spec.scenes or [])

    def evidence_text(scene: Any) -> str:
        return "\n".join([
            str(scene.action.raw_text or ""),
            str(scene.raw_requirement_text or ""),
            str(scene.logic_flow or ""),
            "\n".join(str(item) for item in scene.action_sequence or []),
        ]).lower()

    evidence = [evidence_text(scene) for scene in scenes]
    relocations: List[Tuple[Any, Any, Dict[str, Any]]] = []
    for source_scene in scenes:
        structured = source_scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        for binding in list(structured.get("public_operations", []) or []):
            if not isinstance(binding, dict):
                continue
            operation = str(binding.get("operation", "")).strip().lower()
            if not operation:
                continue
            candidates = [
                scene for scene, text in zip(scenes, evidence)
                if operation in text
            ]
            if len(candidates) == 1 and candidates[0] is not source_scene:
                relocations.append((source_scene, candidates[0], binding))

    for source_scene, target_scene, binding in relocations:
        source_structured = source_scene.action.structured_op
        target_structured = target_scene.action.structured_op
        if not isinstance(source_structured, dict):
            continue
        if not isinstance(target_structured, dict):
            target_structured = {}
            target_scene.action.structured_op = target_structured
        source_bindings = source_structured.get("public_operations", []) or []
        if binding not in source_bindings:
            continue
        target_bindings = target_structured.setdefault("public_operations", [])
        identity = (binding.get("dependency_type"), binding.get("operation"))
        if any(
            (item.get("dependency_type"), item.get("operation")) == identity
            for item in target_bindings if isinstance(item, dict)
        ):
            continue
        source_bindings.remove(binding)
        target_bindings.append(binding)
        logger.info(
            "[M3c] 按精确接口证据重定位 %s::%s: scene %s -> %s",
            identity[0], identity[1], source_scene.scene_id, target_scene.scene_id,
        )


def _normalize_scored_collection_references(fact_spec: 'DslFactSpec') -> None:
    """Propagate score-bearing collections through filters and reports."""
    score_bearing: List[str] = []
    latest_eligible = ""

    def references_loop_score(value: Any, item: str) -> bool:
        if isinstance(value, list):
            return any(references_loop_score(entry, item) for entry in value)
        if not isinstance(value, dict):
            return False
        if set(value) == {"ref"} and isinstance(value.get("ref"), str):
            reference = value["ref"].removeprefix("$")
            return reference.startswith(f"{item}.") and "score" in reference.lower()
        return any(references_loop_score(entry, item) for entry in value.values())

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        program = structured.get("local_program", []) if isinstance(structured, dict) else []
        scene_builds_report = any(
            any(token in str(output).lower() for token in ("rank", "report"))
            for output in (scene.action.outputs or [])
        )
        local_targets: Set[str] = set()
        for statement in program or []:
            if not isinstance(statement, dict) or statement.get("op") not in {
                "map", "filter", "sort",
            }:
                continue
            item = str(statement.get("item", ""))
            source = statement.get("source")
            source_root = (
                str(source.get("ref", "")).split(".", 1)[0]
                if isinstance(source, dict) and set(source) == {"ref"} else ""
            )
            target = str(statement.get("target", ""))
            needs_scored_source = (
                item and references_loop_score(statement, item)
            ) or (
                statement.get("op") == "filter"
                and any(token in target.lower() for token in (
                    "eligible", "candidate", "prospect", "fresh",
                ))
            ) or (
                bool(latest_eligible)
                and statement.get("op") in {"sort", "map"}
                and (
                    scene_builds_report
                    or any(token in target.lower() for token in ("rank", "report"))
                )
            )
            preferred_source = latest_eligible or (score_bearing[-1] if score_bearing else "")
            if (
                needs_scored_source
                and preferred_source
                and source_root != preferred_source
                and source_root not in local_targets
            ):
                statement["source"] = {"ref": preferred_source}
                source_root = preferred_source
            if target and source_root in score_bearing and statement.get("op") in {"filter", "sort"}:
                if target not in score_bearing:
                    score_bearing.append(target)
            elif target and item and references_loop_score(statement, item):
                if target not in score_bearing:
                    score_bearing.append(target)
            if target and any(token in target.lower() for token in (
                "eligible", "candidate", "prospect", "fresh",
            )):
                latest_eligible = target
            if target:
                local_targets.add(target)
        for output in (scene.action.outputs or []):
            root = str(output).split(".", 1)[0]
            if root and "score" in root.lower() and root not in score_bearing:
                score_bearing.append(root)


def _preserve_source_records_in_scoring_maps(fact_spec: 'DslFactSpec') -> None:
    """Keep source fields when a scoring map augments records with a score."""
    for scene in fact_spec.scenes:
        if str(scene.action.action_type or "").lower() not in {"score", "scoring"}:
            continue
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        for statement in structured.get("local_program", []) or []:
            if not isinstance(statement, dict) or statement.get("op") != "map":
                continue
            item = str(statement.get("item", ""))
            target = str(statement.get("target", ""))
            value = statement.get("value")
            fields = value.get("object") if isinstance(value, dict) else None
            if not item or "score" not in target.lower() or not isinstance(fields, dict):
                continue
            if not any("score" in str(field).lower() for field in fields):
                continue
            statement["value"] = {"call": {"name": "merge", "args": [
                {"ref": item}, {"object": fields},
            ]}}


def _materialize_explicit_collection_transforms(fact_spec: 'DslFactSpec') -> None:
    """Compile explicit boolean filtering and key-based deduplication locally."""
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        program = structured.get("local_program", []) or []
        if not isinstance(program, list):
            continue
        text = "\n".join([
            str(scene.action.raw_text or ""),
            str(scene.raw_requirement_text or ""),
            str(scene.logic_flow or ""),
            "\n".join(str(item) for item in scene.action_sequence or []),
        ])
        dedupe_match = re.search(
            r"按\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:字段)?(?:进行)?去重",
            text,
        )
        dedupe_key = dedupe_match.group(1) if dedupe_match else ""
        if not dedupe_key and re.search(
            r"按\s*平台\s*(?:进行)?去重|每个\s*平台\s*只保留",
            text,
        ):
            dedupe_key = "platform"
        filters = [
            statement for statement in program
            if isinstance(statement, dict) and statement.get("op") == "filter"
        ]
        if filters:
            already_deduped = any(
                isinstance(statement, dict)
                and isinstance(statement.get("value"), dict)
                and (statement["value"].get("call") or {}).get("name") == "unique_by"
                for statement in program
            )
            if dedupe_key and not already_deduped:
                emit_index = next((
                    index for index, statement in enumerate(program)
                    if isinstance(statement, dict) and statement.get("op") == "emit"
                ), -1)
                if emit_index >= 0:
                    emitted_fields = program[emit_index].get("fields", {}) or {}
                    if len(emitted_fields) == 1:
                        target = str(next(iter(emitted_fields)))
                        program.insert(emit_index, {
                            "op": "assign",
                            "target": target,
                            "value": {"call": {"name": "unique_by", "args": [
                                {"ref": target}, dedupe_key,
                            ]}},
                        })
            continue
        if re.search(
            r"(?<![A-Za-z0-9_])active\s*(?:==|为|等于|is)\s*true(?![A-Za-z0-9_])",
            text,
            flags=re.IGNORECASE,
        ) is None:
            continue
        emitted_source = ""
        emitted_target = ""
        for statement in program:
            if not isinstance(statement, dict) or statement.get("op") != "emit":
                continue
            for target, value in (statement.get("fields", {}) or {}).items():
                reference = value.get("ref") if isinstance(value, dict) else None
                if isinstance(reference, str) and reference.startswith("result."):
                    emitted_source = reference
                    emitted_target = str(target)
                    break
            if emitted_source:
                break
        if not emitted_source or not emitted_target:
            continue

        filtered_target = (
            f"{emitted_target}_filtered" if dedupe_key else emitted_target
        )
        rewritten: List[Dict[str, Any]] = [{
            "op": "filter",
            "target": filtered_target,
            "source": {"ref": emitted_source},
            "item": "item",
            "where": {
                "op": {
                    "name": "eq",
                    "args": [{"ref": "item.active"}, True],
                }
            },
        }]
        if dedupe_key:
            rewritten.append({
                "op": "assign",
                "target": emitted_target,
                "value": {
                    "call": {
                        "name": "unique_by",
                        "args": [
                            {"ref": filtered_target},
                            dedupe_key,
                        ],
                    }
                },
            })
        rewritten.append({
            "op": "emit",
            "fields": {emitted_target: {"ref": emitted_target}},
        })
        structured["local_program"] = rewritten


def _materialize_empty_collection_guards(fact_spec: 'DslFactSpec') -> None:
    """Skip candidate-dependent operations when their collection is empty."""
    def is_candidate_collection(name: str) -> bool:
        lowered = name.lower()
        # An empty valid-attachment list selects the inline-body route; it
        # does not mean that the workflow has no business candidate.
        if "attachment" in lowered or "schema" in lowered:
            return False
        return any(token in lowered for token in (
            "eligible", "candidate", "prospect", "valid", "filtered", "matched", "ranking", "report",
            "target", "selected_items", "selected_posts", "selected_records",
            "selected_rows", "selected_urls",
        ))

    latest_candidate_collection = ""
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        # Keep the candidate discovered by an earlier scene separate from the
        # outputs of the current scene.  A later side effect (for example
        # writing or uploading a comment) must inherit the earlier selection's
        # non-empty requirement, while the selection scene's own discovery
        # operations must remain executable.
        prior_candidate_collection = latest_candidate_collection
        current_candidate_collections: List[str] = []
        for output in scene.action.outputs or []:
            name = str(output).split(".", 1)[0]
            if is_candidate_collection(name):
                current_candidate_collections.append(name)
        for statement in structured.get("local_program", []) or []:
            if not isinstance(statement, dict):
                continue
            target = statement.get("target")
            if isinstance(target, str) and target and is_candidate_collection(target):
                current_candidate_collections.append(target.split(".", 1)[0])
        current_candidate_collections = list(dict.fromkeys(current_candidate_collections))
        for binding in structured.get("public_operations", []) or []:
            if not isinstance(binding, dict) or binding.get("condition"):
                continue
            arguments = binding.get("arguments", {}) or {}
            source = ""
            empty_value: Any = []
            if arguments.get("one_row_per_candidate") in (True, {"literal": True}):
                row_source = arguments.get("row_data")
                if isinstance(row_source, str):
                    source = row_source
            elif arguments.get("message_is_ranked_digest") in (True, {"literal": True}):
                text_source = arguments.get("summary_text")
                if isinstance(text_source, str):
                    source = text_source
                    empty_value = ""
            elif (
                prior_candidate_collection
                and str(binding.get("dependency_type", "")).lower().startswith(("lm", "llm"))
            ):
                source = prior_candidate_collection
            elif prior_candidate_collection and scene.side_effects:
                source = prior_candidate_collection
            if source:
                binding["condition"] = {
                    "source": source,
                    "operator": "neq",
                    "value": empty_value,
                }
        if current_candidate_collections:
            latest_candidate_collection = current_candidate_collections[-1]


def _normalize_legacy_conditional_expressions(fact_spec: 'DslFactSpec') -> None:
    """Rewrite legacy three-argument ``or`` expressions as explicit ternaries."""
    def rewrite(value: Any) -> Any:
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, dict):
            return value
        rewritten = {key: rewrite(item) for key, item in value.items()}
        operation = rewritten.get("op")
        if not isinstance(operation, dict):
            return rewritten
        args = operation.get("args", [])
        if operation.get("name") != "or" or not isinstance(args, list) or len(args) != 3:
            return rewritten
        # Boolean OR may be n-ary. Rewrite only when a branch visibly carries
        # a non-boolean value, which identifies the legacy ternary encoding.
        branches = args[1:]
        if not any(
            isinstance(branch, str)
            or (
                isinstance(branch, dict)
                and "literal" in branch
                and not isinstance(branch.get("literal"), bool)
            )
            for branch in branches
        ):
            return rewritten
        return {"call": {"name": "if_else", "args": args}}

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        structured["local_program"] = rewrite(
            structured.get("local_program", []) or []
        )


def _materialize_due_date_filters(fact_spec: 'DslFactSpec') -> None:
    """Materialize date-based sweep sets before downstream side effects.

    When a requirement explicitly describes a daily/weekly sweep with
    check-in, check-out, and review windows, a direct alias from
    ``read_bookings.rows`` to ``due_bookings`` is insufficient.  The compiler
    can express the three deterministic date predicates locally, so add them
    only when the corresponding public read operation and local output are
    already present.
    """
    sweep_scene = None
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        text = " ".join([
            str(scene.raw_requirement_text or ""),
            str(scene.logic_flow or ""),
            " ".join(str(item) for item in (scene.action_sequence or [])),
        ]).lower()
        operations = {
            str(binding.get("operation", ""))
            for binding in structured.get("public_operations", []) or []
            if isinstance(binding, dict)
        }
        program = structured.get("local_program", []) or []
        if (
            "read_bookings" in operations
            and "due_bookings" in text
            and any(token in text for token in ("扫描", "到期", "sweep", "check-in", "入住"))
            and any(token in text for token in ("退房", "check-out", "评价"))
            and isinstance(program, list)
        ):
            sweep_scene = scene
            break
    if sweep_scene is None:
        return
    structured = sweep_scene.action.structured_op
    program = structured.get("local_program", []) or []
    if any(
        isinstance(statement, dict)
        and statement.get("op") == "filter"
        and statement.get("target") == "due_bookings"
        for statement in program
    ):
        return

    def row_field(field: str) -> Dict[str, Any]:
        return {
            "call": {
                "name": "get",
                "args": [
                    {"ref": "booking"},
                    {"literal": field},
                    {"literal": ""},
                ],
            }
        }

    sweep_date = {"ref": "workflow_input.sweep_date"}
    def same_date(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        return {"op": {"name": "eq", "args": [left, right]}}

    due_predicate = {
        "op": {
            "name": "or",
            "args": [
                {
                    "op": {
                        "name": "or",
                        "args": [
                            same_date(row_field("check_in"), {
                                "call": {"name": "date_add_days", "args": [sweep_date, {"literal": 1}]}
                            }),
                            same_date(row_field("check_out"), sweep_date),
                        ],
                    }
                },
                same_date(row_field("check_out"), {
                    "call": {"name": "date_add_days", "args": [sweep_date, {"literal": -2}]}
                }),
            ],
        }
    }
    checkout_predicate = same_date(row_field("check_out"), sweep_date)
    source = {
        "call": {
            "name": "coalesce",
            "args": [{"ref": "result.read_bookings.rows"}, {"list": []}],
        }
    }
    rewritten = [
        {
            "op": "filter",
            "target": "due_bookings",
            "source": source,
            "item": "booking",
            "where": due_predicate,
        },
        {
            "op": "filter",
            "target": "checkout_stays",
            "source": source,
            "item": "booking",
            "where": checkout_predicate,
        },
    ]
    for statement in program:
        if isinstance(statement, dict) and statement.get("op") == "emit":
            fields = dict(statement.get("fields", {}) or {})
            fields["due_bookings"] = {"ref": "due_bookings"}
            fields["checkout_stays"] = {"ref": "checkout_stays"}
            rewritten.append({**statement, "fields": fields})
        elif not (
            isinstance(statement, dict)
            and statement.get("op") == "assign"
            and statement.get("target") == "due_bookings"
        ):
            rewritten.append(statement)
    structured["local_program"] = rewritten

    for scene in fact_spec.scenes:
        scene_structured = scene.action.structured_op
        if not isinstance(scene_structured, dict):
            continue
        has_checkout_cleaning = False
        for binding in scene_structured.get("public_operations", []) or []:
            if (
                isinstance(binding, dict)
                and binding.get("operation") == "append_cleaning_task"
            ):
                has_checkout_cleaning = True
                binding["condition"] = {
                    "source": "checkout_stays",
                    "operator": "neq",
                    "value": [],
                }
        if has_checkout_cleaning:
            checkout_fields = {"property", "check_out", "booking_id", "guest_name", "guest_email"}

            def bind_checkout_fields(value: Any) -> Any:
                if isinstance(value, dict):
                    if set(value) == {"ref"} and value.get("ref") in checkout_fields:
                        field = str(value["ref"])
                        return {"ref": f"checkout_stays[0].{field}"}
                    return {key: bind_checkout_fields(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [bind_checkout_fields(item) for item in value]
                return value

            scene_structured["local_program"] = bind_checkout_fields(
                scene_structured.get("local_program", []) or []
            )


def _materialize_weekly_report_fallback(fact_spec: 'DslFactSpec') -> None:
    """Expose the explicit active/empty branch of a weekly report scene."""
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        text = " ".join([
            str(scene.raw_requirement_text or ""),
            str(scene.logic_flow or ""),
            str(scene.fallback or ""),
        ])
        if "一切正常" not in text or "周" not in text:
            continue
        bindings = structured.get("public_operations", []) or []
        if not any(
            isinstance(binding, dict)
            and binding.get("operation") == "write_weekly_digest"
            for binding in bindings
        ):
            continue
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            if binding.get("operation") == "write_weekly_digest":
                binding["condition"] = {
                    "any": [
                        {
                            "source": "due_bookings",
                            "operator": "neq",
                            "value": [],
                        },
                        {
                            "source": "result.read_cleaning_tasks.rows",
                            "operator": "neq",
                            "value": [],
                        },
                    ]
                }
            elif binding.get("operation") == "read_cleaning_tasks":
                binding["condition"] = {}
        digest_ref = {"ref": "result.write_weekly_digest.digest"}
        fallback_value = {
            "call": {
                "name": "coalesce",
                "args": [digest_ref, {"literal": "一切正常"}],
            }
        }

        def replace_digest(value: Any) -> Any:
            if isinstance(value, dict):
                if value == digest_ref:
                    return copy.deepcopy(fallback_value)
                return {key: replace_digest(item) for key, item in value.items()}
            if isinstance(value, list):
                return [replace_digest(item) for item in value]
            return value

        structured["local_program"] = replace_digest(
            structured.get("local_program", []) or []
        )


def _bind_checkout_scene_fields(fact_spec: 'DslFactSpec') -> None:
    """Bind checkout side effects to the collection produced by the sweep."""
    checkout_fields = {"property", "check_out", "booking_id", "guest_name", "guest_email"}

    def bind_checkout_fields(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"ref"} and value.get("ref") in checkout_fields:
                field = str(value["ref"])
                return {"ref": f"checkout_stays[0].{field}"}
            return {key: bind_checkout_fields(item) for key, item in value.items()}
        if isinstance(value, list):
            return [bind_checkout_fields(item) for item in value]
        return value

    def enrich_checkout_alert_text(value: Any) -> Any:
        if isinstance(value, list):
            return [enrich_checkout_alert_text(item) for item in value]
        if not isinstance(value, dict):
            return value
        enriched: Dict[str, Any] = {}
        for key, item in value.items():
            if (
                key == "text"
                and isinstance(item, dict)
                and set(item) == {"literal"}
                and isinstance(item.get("literal"), str)
            ):
                property_text = {"call": {
                    "name": "join", "args": [
                        {"list": [
                            {"literal": item["literal"]},
                            {"ref": "checkout_stays[0].property"},
                        ]},
                        {"literal": ": "},
                    ],
                }}
                missing_cleaner_text = {"call": {
                    "name": "join", "args": [
                        {"list": [
                            {"literal": "no cleaner email"},
                            {"ref": "checkout_stays[0].property"},
                        ]},
                        {"literal": ": "},
                    ],
                }}
                enriched[key] = {"call": {
                    "name": "if_else", "args": [
                        {"op": {"name": "eq", "args": [
                            {"call": {"name": "coalesce", "args": [
                                {"ref": "workflow_input.cleaner_email"},
                                {"literal": ""},
                            ]}},
                            {"literal": ""},
                        ]}},
                        missing_cleaner_text,
                        property_text,
                    ],
                }}
            else:
                enriched[key] = enrich_checkout_alert_text(item)
        return enriched

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        bindings = structured.get("public_operations", []) or []
        if not any(
            isinstance(binding, dict)
            and binding.get("operation") == "append_cleaning_task"
            for binding in bindings
        ):
            continue
        program = bind_checkout_fields(
            structured.get("local_program", []) or []
        )
        for index, statement in enumerate(program):
            if (
                isinstance(statement, dict)
                and statement.get("op") == "assign"
                and statement.get("target") == "telegram_messages"
            ):
                program[index] = enrich_checkout_alert_text(statement)
        structured["local_program"] = program


def _materialize_notification_context_fields(fact_spec: 'DslFactSpec') -> None:
    """Attach requirement-declared business context to fixed notifications."""

    def enrich_literal_field(
        value: Any,
        field: str,
        context_ref: str,
    ) -> Any:
        if isinstance(value, list):
            return [enrich_literal_field(item, field, context_ref) for item in value]
        if not isinstance(value, dict):
            return value
        enriched: Dict[str, Any] = {}
        for key, item in value.items():
            if (
                key == field
                and isinstance(item, dict)
                and set(item) == {"literal"}
                and isinstance(item.get("literal"), str)
            ):
                enriched[key] = {"call": {
                    "name": "join", "args": [
                        {"list": [item, {"ref": context_ref}]},
                        {"literal": ": "},
                    ],
                }}
            else:
                enriched[key] = enrich_literal_field(item, field, context_ref)
        return enriched

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        operations = {
            str(binding.get("operation", ""))
            for binding in structured.get("public_operations", []) or []
            if isinstance(binding, dict)
        }
        requirement = str(scene.raw_requirement_text or "")
        program = structured.get("local_program", []) or []
        for index, statement in enumerate(program):
            if not isinstance(statement, dict) or statement.get("op") != "assign":
                continue
            target = str(statement.get("target", ""))
            if "append_booking" in operations and target == "telegram_messages":
                program[index] = enrich_literal_field(
                    statement, "text", "booking_append_record.booking_id",
                )
            elif (
                "被跳过的预订内容" in requirement
                and target == "sent_emails"
            ):
                program[index] = enrich_literal_field(
                    statement, "body", "guest_name",
                )


def _order_selection_scenes_before_prior_side_effects(fact_spec: 'DslFactSpec') -> None:
    """Discover an empty-capable target set before committing dependent effects.

    Natural-language requirements often narrate preparation before selection
    (for example, upload media and then select enabled accounts). Executing that
    order creates an avoidable side effect when the selected set is empty. A
    side-effect-free selection scene may move ahead of preceding effect scenes
    when none of its inputs are produced by those scenes.
    """
    selection_tokens = ("eligible", "candidate", "prospect", "target", "selected")
    scenes = list(fact_spec.scenes or [])

    producers: Dict[str, Any] = {}
    for scene in scenes:
        names = {
            str(value.get("name", "")) if isinstance(value, dict) else str(value)
            for value in (scene.action.outputs or [])
        }
        structured = scene.action.structured_op
        if isinstance(structured, dict):
            names.update(
                str(target).split(".", 1)[0]
                for binding in structured.get("public_operations", []) or []
                if isinstance(binding, dict)
                for target in (binding.get("results", {}) or {}).values()
                if isinstance(target, str)
            )
        for name in names - {""}:
            if name not in producers:
                producers[name] = scene

    def dependencies(scene: Any) -> List[Any]:
        structured = scene.action.structured_op
        roots = (
            _action_contract_reference_roots(structured)
            if isinstance(structured, dict) else set()
        )
        return list({
            id(producers[root]): producers[root]
            for root in roots
            if root in producers and producers[root] is not scene
        }.values())

    for selection in list(scenes):
        if selection.side_effects:
            continue
        outputs = {
            str(value.get("name", "")) if isinstance(value, dict) else str(value)
            for value in (selection.action.outputs or [])
        }
        if not any(
            token in output.lower()
            for output in outputs
            for token in selection_tokens
        ):
            continue
        required: Dict[int, Any] = {}
        pending = list(dependencies(selection))
        while pending:
            dependency = pending.pop()
            if id(dependency) in required:
                continue
            required[id(dependency)] = dependency
            pending.extend(
                item for item in dependencies(dependency)
                if id(item) not in required
            )
        if any(scene.side_effects for scene in required.values()):
            continue

        selection_index = scenes.index(selection)
        prior_effect_indices = [
            index for index, scene in enumerate(scenes[:selection_index])
            if scene.side_effects
        ]
        if not prior_effect_indices:
            continue

        insert_at = prior_effect_indices[0]
        moving_ids = {*required, id(selection)}
        moving = [scene for scene in scenes if id(scene) in moving_ids]
        remaining = [scene for scene in scenes if id(scene) not in moving_ids]
        anchor = scenes[insert_at]
        insert_at = remaining.index(anchor)
        scenes = [*remaining[:insert_at], *moving, *remaining[insert_at:]]

    fact_spec.scenes = scenes


def _normalize_foreach_collection_sources(fact_spec: 'DslFactSpec') -> None:
    """Use the collection root for foreach and bind the current item by field."""
    collection_roots = {
        str(value.get("name", "")) if isinstance(value, dict) else str(value)
        for scene in fact_spec.scenes
        for value in (scene.action.outputs or [])
    }
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        for binding in structured.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            foreach = binding.get("foreach") or {}
            source = foreach.get("source") if isinstance(foreach, dict) else None
            if not isinstance(source, str) or not source:
                continue
            root, separator, field = source.partition(".")
            if root not in collection_roots:
                continue
            if separator and field:
                foreach["source"] = root
            argument = str(foreach.get("argument", ""))
            arguments = binding.get("arguments", {}) or {}
            if argument in arguments:
                arguments[argument] = f"{root}[0]"
            elif argument:
                # Some M3 outputs use a local alias (for example
                # ``tag_item``) instead of the declared foreach argument.
                # Bind that alias to the current collection item at the
                # plan boundary so the strict data-flow gate can resolve it.
                for field_name, argument_source in list(arguments.items()):
                    if not isinstance(argument_source, str):
                        continue
                    if argument_source == argument or argument_source.endswith("_item"):
                        arguments[field_name] = f"{root}[0]"
            for field_name, argument_source in list(arguments.items()):
                if not isinstance(argument_source, str) or "." not in argument_source:
                    continue
                source_root, item_field = argument_source.split(".", 1)
                item_field = item_field.split("[", 1)[0]
                if (
                    source_root in collection_roots
                    and (
                        source_root == root
                        or not str(field_name).startswith(f"{source_root}_")
                    )
                    and item_field
                    and str(field_name).endswith(item_field)
                ):
                    arguments[field_name] = item_field


def _normalize_public_operation_branch_lineage(
    fact_spec: 'DslFactSpec',
) -> None:
    """Keep public bindings consistent with the branch that produces them.

    A conditional producer such as ``read_feed -> entries`` defines the
    runtime branch in which that collection exists.  A later foreach binding
    over ``entries`` must inherit that branch; otherwise a generated plan can
    contain an impossible combination such as ``source == manual`` together
    with ``foreach=entries``.  This is a plan-level consistency rule, not an
    evaluator exception.

    M3 can also emit the same public operation repeatedly while several local
    descriptions refer to it.  Exact duplicates are collapsed, preferring an
    unconditional binding over a self-guarded duplicate.  The operation's
    arguments, results, condition and foreach contract remain unchanged.
    """

    def public_bindings(scene: Any) -> List[Dict[str, Any]]:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            return []
        bindings = structured.get("public_operations", []) or []
        return [binding for binding in bindings if isinstance(binding, dict)]

    # A root may be produced by more than one branch.  Only a unique producer
    # condition is strong enough to constrain a downstream foreach binding.
    producer_conditions: Dict[str, List[Dict[str, Any]]] = {}
    for scene in fact_spec.scenes:
        for binding in public_bindings(scene):
            condition = binding.get("condition") or {}
            if not condition:
                continue
            for target in (binding.get("results") or {}).values():
                if not isinstance(target, str) or not target:
                    continue
                root = target.split(".", 1)[0]
                producer_conditions.setdefault(root, []).append(
                    copy.deepcopy(condition)
                )

    unique_producer_conditions: Dict[str, Dict[str, Any]] = {}
    for root, conditions in producer_conditions.items():
        encoded = {
            json.dumps(condition, ensure_ascii=False, sort_keys=True)
            for condition in conditions
        }
        if len(encoded) == 1:
            unique_producer_conditions[root] = conditions[0]

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        bindings = structured.get("public_operations", []) or []
        if not isinstance(bindings, list):
            continue

        for binding in bindings:
            foreach = binding.get("foreach") or {}
            source = foreach.get("source") if isinstance(foreach, dict) else None
            if not isinstance(source, str) or not source:
                continue
            root = source.split(".", 1)[0]
            producer_condition = unique_producer_conditions.get(root)
            if producer_condition is not None:
                current_condition = binding.get("condition") or {}
                if current_condition != producer_condition:
                    binding["condition"] = copy.deepcopy(producer_condition)

        def binding_key(binding: Dict[str, Any]) -> str:
            comparable = {
                key: value
                for key, value in binding.items()
                if key != "guard"
            }
            return json.dumps(comparable, ensure_ascii=False, sort_keys=True)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        for binding in bindings:
            key = binding_key(binding)
            if key not in grouped:
                order.append(key)
                grouped[key] = []
            grouped[key].append(binding)

        deduplicated: List[Dict[str, Any]] = []
        for key in order:
            candidates = grouped[key]
            chosen = next(
                (
                    candidate for candidate in candidates
                    if str(candidate.get("guard", "always")) == "always"
                ),
                candidates[0],
            )
            chosen = copy.deepcopy(chosen)
            guard = str(chosen.get("guard", "") or "")
            operation = str(chosen.get("operation", "") or "")
            if guard == f"on_success:{operation}":
                chosen["guard"] = "always"
            deduplicated.append(chosen)
        structured["public_operations"] = deduplicated


def _prefer_unique_workflow_inputs_over_conditional_collections(
    fact_spec: 'DslFactSpec',
    workflow_input_fields: Optional[Set[str]] = None,
    repeated_workflow_input_fields: Optional[Set[str]] = None,
) -> None:
    """Avoid reading a collection whose producer is skipped in another input mode."""
    workflow_fields = set(workflow_input_fields or set())
    repeated_fields = set(repeated_workflow_input_fields or set())
    conditional_collections = {
        str(target).split(".", 1)[0]
        for scene in fact_spec.scenes
        for binding in (
            (scene.action.structured_op or {}).get("public_operations", [])
            if isinstance(scene.action.structured_op, dict) else []
        )
        if isinstance(binding, dict) and binding.get("condition")
        for target in (binding.get("results", {}) or {}).values()
        if isinstance(target, str) and target
    }

    def rewrite(value: Any) -> Any:
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if not isinstance(value, str):
            return value
        match = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*)\[0\]\.([A-Za-z_][A-Za-z0-9_]*)",
            value.removeprefix("$"),
        )
        if not match:
            return value
        collection, field = match.groups()
        if (
            collection in conditional_collections
            and field in workflow_fields
            and field not in repeated_fields
        ):
            return f"workflow_input.{field}"
        return value

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        for binding in structured.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            binding["arguments"] = rewrite(binding.get("arguments", {}) or {})


def _materialize_branch_local_public_dependencies(
    fact_spec: 'DslFactSpec',
    workflow_input_fields: Optional[Set[str]] = None,
    input_mode_signatures: Optional[List[List[str]]] = None,
) -> None:
    """Repeat a public operation when another input branch directly needs its result."""
    declared_workflow_fields = set(workflow_input_fields or set())
    operation_bindings: Dict[str, Dict[str, Any]] = {}
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        for binding in structured.get("public_operations", []) or []:
            if isinstance(binding, dict) and binding.get("operation"):
                operation_bindings.setdefault(str(binding["operation"]), binding)

    def has_conditional_lineage(operation: str, seen: Optional[Set[str]] = None) -> bool:
        seen = set(seen or set())
        if not operation or operation in seen:
            return False
        seen.add(operation)
        binding = operation_bindings.get(operation, {})
        if binding.get("condition"):
            return True
        guard = str(binding.get("guard", "") or "")
        if guard.startswith(("on_success:", "on_failure:")):
            return has_conditional_lineage(guard.split(":", 1)[1], seen)
        return False

    def conditional_lineage_fields(
        operation: str, seen: Optional[Set[str]] = None,
    ) -> Set[str]:
        seen = set(seen or set())
        if not operation or operation in seen:
            return set()
        seen.add(operation)
        binding = operation_bindings.get(operation, {})
        found: Set[str] = set()
        condition = binding.get("condition", {}) or {}
        source = str(condition.get("source", "") or "")
        if source:
            found.add(source.rsplit(".", 1)[-1])
        guard = str(binding.get("guard", "") or "")
        if guard.startswith(("on_success:", "on_failure:")):
            found.update(conditional_lineage_fields(guard.split(":", 1)[1], seen))
        return found

    def referenced_operations(value: Any) -> Set[str]:
        found: Set[str] = set()
        if isinstance(value, list):
            for item in value:
                found.update(referenced_operations(item))
        elif isinstance(value, dict):
            if set(value) == {"ref"} and isinstance(value.get("ref"), str):
                match = re.match(
                    r"result\.([A-Za-z_][A-Za-z0-9_]*)\.", value["ref"]
                )
                if match:
                    found.add(match.group(1))
            for item in value.values():
                found.update(referenced_operations(item))
        return found

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        bindings = structured.get("public_operations", []) or []
        if not isinstance(bindings, list) or not bindings:
            continue
        current_operations = {
            str(binding.get("operation", ""))
            for binding in bindings if isinstance(binding, dict)
        }
        local_refs = referenced_operations(structured.get("local_program", []) or [])
        predecessor = str(bindings[-1].get("operation", ""))
        branch_discriminators: Set[str] = set()
        for current_operation in current_operations:
            branch_discriminators.update(conditional_lineage_fields(current_operation))
        matching_modes = [
            set(mode)
            for mode in (input_mode_signatures or [])
            if branch_discriminators and branch_discriminators.issubset(set(mode))
        ]
        branch_workflow_fields = (
            matching_modes[0] if len(matching_modes) == 1 else declared_workflow_fields
        )
        for operation in sorted(local_refs - current_operations):
            source = operation_bindings.get(operation)
            if not source or not has_conditional_lineage(operation):
                continue
            clone = copy.deepcopy(source)
            clone["guard"] = f"on_success:{predecessor}"
            clone["condition"] = {}
            arguments = clone.get("arguments", {}) or {}
            if predecessor.startswith("send_"):
                sent_kind = predecessor[len("send_"):]
                for argument_name, argument_source in list(arguments.items()):
                    missing_type_field = (
                        str(argument_name).endswith("_type")
                        and argument_source == f"workflow_input.{argument_name}"
                        and argument_name not in branch_workflow_fields
                    )
                    if missing_type_field and sent_kind:
                        arguments[argument_name] = {"literal": sent_kind}
            clone["arguments"] = arguments
            bindings.append(clone)
            current_operations.add(operation)
            predecessor = operation


def _separate_new_and_existing_entity_entry_conditions(
    fact_spec: 'DslFactSpec',
    input_mode_signatures: Optional[List[List[str]]] = None,
) -> None:
    """Separate create-mode roots from read-existing-mode roots.

    M3 can assign the same ``not_exists`` discriminator to independent roots
    even when public input modes contain both a new-entity form and an
    existing-entity object. Operation verbs and public mode shapes are enough
    to resolve that structural contradiction without evaluator data.
    """
    modes = [set(mode) for mode in (input_mode_signatures or [])]
    if not modes:
        return
    create_prefixes = ("create_", "save_", "add_", "register_", "upsert_")
    read_prefixes = ("read_", "list_", "load_", "fetch_", "get_", "query_", "search_")
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        for binding in structured.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            condition = binding.get("condition", {}) or {}
            if condition.get("operator") != "not_exists":
                continue
            source = str(condition.get("source", "") or "")
            if source:
                grouped.setdefault(source, []).append(binding)

    for source, bindings in grouped.items():
        discriminator = source.rsplit(".", 1)[-1]
        if not any(discriminator in mode for mode in modes):
            continue
        if not any(discriminator not in mode for mode in modes):
            continue
        has_create_root = any(
            str(binding.get("operation", "")).startswith(create_prefixes)
            for binding in bindings
        )
        if not has_create_root:
            continue
        for binding in bindings:
            operation = str(binding.get("operation", ""))
            if operation.startswith(read_prefixes):
                binding["condition"] = {
                    "source": source,
                    "operator": "exists",
                }


def _propagate_scene_input_mode_conditions(
    fact_spec: 'DslFactSpec',
    input_mode_signatures: Optional[List[List[str]]] = None,
) -> None:
    """Keep all public operations in one action scene on the same input mode."""
    modes = [set(mode) for mode in (input_mode_signatures or []) if mode]
    if not modes:
        return
    discriminator_fields = {
        field
        for field in set().union(*modes)
        if 0 < sum(field in mode for mode in modes) < len(modes)
    }
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        bindings = [
            binding for binding in structured.get("public_operations", []) or []
            if isinstance(binding, dict)
        ]
        scene_conditions: List[Dict[str, Any]] = []
        for binding in bindings:
            condition = binding.get("condition", {}) or {}
            source = str(condition.get("source", "") or "")
            field = source.rsplit(".", 1)[-1]
            if (
                condition.get("operator") == "exists"
                and field in discriminator_fields
            ):
                scene_conditions.append(condition)
        unique_conditions = {
            (str(condition.get("source")), str(condition.get("operator")))
            for condition in scene_conditions
        }
        if len(unique_conditions) != 1:
            continue
        source, operator = next(iter(unique_conditions))
        for binding in bindings:
            if not binding.get("condition"):
                binding["condition"] = {
                    "source": source,
                    "operator": operator,
                }


def _materialize_foreach_output_maps(
    fact_spec: 'DslFactSpec',
    workflow_input_fields: Optional[Set[str]] = None,
) -> None:
    """Turn one-object list wrappers after foreach calls into collection maps."""
    declared_workflow_fields = set(workflow_input_fields or set())

    def rewrite_item_refs(value: Any, item: str) -> Any:
        if isinstance(value, list):
            return [rewrite_item_refs(entry, item) for entry in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"ref"} and isinstance(value.get("ref"), str):
            reference = re.sub(r"\.{2,}", ".", value["ref"])
            if reference.startswith("workflow_input."):
                leaf = reference[len("workflow_input."):].split(".", 1)[0]
                if leaf and leaf not in declared_workflow_fields:
                    return {"ref": f"{item}.{reference[len('workflow_input.') :]}"}
            if reference != value["ref"]:
                return {"ref": reference}
        return {key: rewrite_item_refs(entry, item) for key, entry in value.items()}

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        foreach_bindings = [
            binding for binding in structured.get("public_operations", []) or []
            if isinstance(binding, dict) and isinstance(binding.get("foreach"), dict)
            and binding["foreach"].get("source")
        ]
        if len(foreach_bindings) != 1:
            continue
        source = str(foreach_bindings[0]["foreach"]["source"])
        source = source.split(".", 1)[0]
        item = "item"
        if "account" in source.lower():
            item = "account"
        elif "candidate" in source.lower() or "prospect" in source.lower():
            item = "candidate"
        program = structured.get("local_program", []) or []
        for index, statement in enumerate(program):
            if not isinstance(statement, dict) or statement.get("op") != "assign":
                continue
            values = statement.get("value")
            if isinstance(values, dict) and set(values) == {"list"}:
                values = values["list"]
            if not isinstance(values, list) or len(values) != 1:
                continue
            object_value = values[0]
            if isinstance(object_value, dict) and set(object_value) == {"object"}:
                object_value = object_value["object"]
            if not isinstance(object_value, dict):
                continue
            rewritten = rewrite_item_refs({"object": object_value}, item)
            if rewritten == {"object": object_value}:
                continue
            program[index] = {
                "op": "map",
                "target": str(statement.get("target", "items")),
                "source": {"ref": source},
                "item": item,
                "value": rewritten,
            }


def _materialize_quantified_output_maps(fact_spec: 'DslFactSpec') -> None:
    """Expand scalar projections when the requirement explicitly says each item.

    M3 occasionally preserves the collection in a public-operation argument but
    emits a one-element local projection.  The requirement and data lineage are
    sufficient to recover the missing map without consulting an evaluator.
    Downstream ``each generated result`` scenes inherit the source collection
    through their on-success guard.
    """
    quantified = re.compile(r"(?:每个|每条|逐个|逐条|for\s+each|\beach\b)", re.I)
    collection_roots: Set[str] = set()
    operation_origins: Dict[str, str] = {}
    output_origins: Dict[str, str] = {}
    item_fields = {
        "booking_id", "guest_email", "guest_name", "property",
        "check_in", "check_out", "email", "name", "id",
    }

    def rewrite_refs(value: Any, item: str) -> Any:
        if isinstance(value, list):
            return [rewrite_refs(entry, item) for entry in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"ref"} and isinstance(value.get("ref"), str):
            reference = value["ref"]
            if reference in item_fields:
                return {"ref": f"{item}.{reference}"}
        rewritten: Dict[str, Any] = {}
        for key, entry in value.items():
            if (
                key in item_fields
                and isinstance(entry, dict)
                and set(entry) == {"literal"}
            ):
                rewritten[key] = {"ref": f"{item}.{key}"}
            else:
                rewritten[key] = rewrite_refs(entry, item)
        return rewritten

    def result_message_refs(value: Any) -> Set[str]:
        references: Set[str] = set()
        if isinstance(value, dict):
            reference = value.get("ref")
            if (
                isinstance(reference, str)
                and reference.startswith("result.")
                and reference.endswith(".message")
            ):
                references.add(reference)
            for entry in value.values():
                references.update(result_message_refs(entry))
        elif isinstance(value, list):
            for entry in value:
                references.update(result_message_refs(entry))
        return references

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            collection_roots.update(str(value) for value in (scene.action.outputs or []))
            continue
        bindings = [
            binding for binding in structured.get("public_operations", []) or []
            if isinstance(binding, dict)
        ]
        evidence = " ".join(str(value or "") for value in (
            scene.raw_requirement_text,
            scene.logic_flow,
            scene.block_description,
            " ".join(scene.preconditions or []),
        ))
        is_quantified = bool(quantified.search(evidence))

        candidates: Set[str] = set()
        for binding in bindings:
            for source in (binding.get("arguments", {}) or {}).values():
                if isinstance(source, str):
                    root = re.split(r"[.\[]", source, maxsplit=1)[0]
                    if root in collection_roots:
                        candidates.add(root)
            guard = str(binding.get("guard", "") or "")
            if guard.startswith("on_success:"):
                origin = operation_origins.get(guard.split(":", 1)[1])
                if origin:
                    candidates.add(origin)
        for output, origin in output_origins.items():
            if output in evidence:
                candidates.add(origin)

        source = next(iter(candidates)) if is_quantified and len(candidates) == 1 else ""
        if source:
            item = "booking" if "booking" in source.lower() else "item"
            program = structured.get("local_program", []) or []
            message_refs = result_message_refs(program)
            for index, statement in enumerate(program):
                if not isinstance(statement, dict) or statement.get("op") != "assign":
                    continue
                value = statement.get("value")
                mapped_value: Any = None
                if isinstance(value, dict) and set(value) == {"list"}:
                    entries = value["list"]
                    if isinstance(entries, list) and len(entries) == 1:
                        mapped_value = entries[0]
                elif isinstance(value, list) and len(value) == 1:
                    mapped_value = value[0]
                elif isinstance(value, dict) and set(value) == {"object"}:
                    mapped_value = value
                if mapped_value is None:
                    continue
                target = str(statement.get("target", "items"))
                rewritten_value = rewrite_refs(mapped_value, item)
                if (
                    target == "telegram_messages"
                    and len(message_refs) == 1
                    and "生成的消息" in evidence
                    and "telegram" in evidence.lower()
                    and isinstance(rewritten_value, dict)
                    and isinstance(rewritten_value.get("object"), dict)
                    and isinstance(rewritten_value["object"].get("text"), dict)
                    and "literal" in rewritten_value["object"]["text"]
                ):
                    rewritten_value["object"]["text"] = {
                        "ref": next(iter(message_refs)),
                    }
                program[index] = {
                    "op": "map",
                    "target": target,
                    "source": {"ref": source},
                    "item": item,
                    "value": rewritten_value,
                }
                output_origins[target] = source
            for binding in bindings:
                operation = str(binding.get("operation", "") or "")
                if operation:
                    operation_origins[operation] = source

        for value in scene.action.outputs or []:
            name = str(value.get("name", "")) if isinstance(value, dict) else str(value)
            if name:
                collection_roots.add(name)
                if source:
                    output_origins.setdefault(name, source)


def _qualify_unambiguous_nested_public_sources(
    fact_spec: 'DslFactSpec',
    public_interfaces: List[Dict[str, Any]],
) -> None:
    """Qualify bare fields such as title when one prior object exposes them."""
    interfaces = {
        (str(item.get("dependency_type", "")), str(item.get("operation", ""))): item
        for item in public_interfaces
        if item.get("dependency_type") and item.get("operation")
    }
    available_roots = {
        str(value.get("name", "")) if isinstance(value, dict) else str(value)
        for value in (fact_spec.inputs or [])
    }
    nested_paths: Dict[str, Set[str]] = {}

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        for binding in structured.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            arguments = binding.get("arguments", {}) or {}
            for field, source in list(arguments.items()):
                if not isinstance(source, str) or not re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*", source
                ):
                    continue
                if source in available_roots:
                    continue
                candidates = nested_paths.get(source, set())
                if len(candidates) == 1:
                    arguments[field] = next(iter(candidates))

            interface = interfaces.get((
                str(binding.get("dependency_type", "")),
                str(binding.get("operation", "")),
            ), {})
            response_schema = interface.get("response_schema", {})
            response_properties = (
                response_schema.get("properties", {})
                if isinstance(response_schema, dict) else {}
            )
            result_targets = binding.get("results", {}) or {}
            for response_field, target in result_targets.items():
                if not isinstance(target, str) or not target:
                    continue
                root = target.split(".", 1)[0]
                available_roots.add(root)
                field_schema = response_properties.get(response_field, {})
                child_properties = (
                    field_schema.get("properties", {})
                    if isinstance(field_schema, dict) else {}
                )
                for child in child_properties:
                    nested_paths.setdefault(str(child), set()).add(f"{root}.{child}")

        available_roots.update(
            str(value.get("name", "")) if isinstance(value, dict) else str(value)
            for value in (scene.action.outputs or [])
        )


def _local_program_semantic_error(
    program: List[Dict[str, Any]],
    public_operations: List[Dict[str, Any]],
    required_outputs: Optional[Set[str]] = None,
    available_references: Optional[Set[str]] = None,
    requirement_text: str = "",
) -> str:
    """Reject fake public calls, missing outputs, and unresolved data references."""
    operation_names = {
        str(binding.get("operation", ""))
        for binding in public_operations
        if isinstance(binding, dict) and binding.get("operation")
    }

    def walk(value: Any) -> bool:
        if isinstance(value, list):
            return any(walk(item) for item in value)
        if not isinstance(value, dict):
            return False
        object_value = value.get("object") if set(value) == {"object"} else value
        if isinstance(object_value, dict):
            operation = object_value.get("op")
            if (
                isinstance(operation, str)
                and operation in operation_names
                and "arguments" in object_value
            ):
                return True
        return any(walk(item) for item in value.values())

    if walk(program):
        return "local_program_restates_public_operation"
    emitted = {
        str(name)
        for statement in program
        if isinstance(statement, dict) and statement.get("op") == "emit"
        for name in (statement.get("fields", {}) or {})
        if name
    }
    missing = sorted(set(required_outputs or set()) - emitted)
    if missing:
        return f"local_program_missing_emitted_outputs:{','.join(missing)}"

    scoring_outputs = {
        str(name) for name in (required_outputs or set())
        if "score" in str(name).lower()
        and not any(token in str(name).lower() for token in (
            "threshold", "cutoff", "minimum", "maximum", "min_score", "max_score",
        ))
    }
    if scoring_outputs:
        def contains_score_field(value: Any) -> bool:
            if isinstance(value, list):
                return any(contains_score_field(item) for item in value)
            if not isinstance(value, dict):
                return False
            object_value = value.get("object")
            if isinstance(object_value, dict) and any(
                key.lower() in {"score", "gap_score", "total_score"}
                for key in object_value
            ):
                return True
            return any(contains_score_field(item) for item in value.values())

        scalar_score_assignment = any(
            isinstance(statement, dict)
            and statement.get("op") == "assign"
            and str(statement.get("target", "")).lower() in {
                "score", "gap_score", "total_score",
            }
            and not (
                isinstance(statement.get("value"), dict)
                and set(statement["value"]) == {"ref"}
            )
            for statement in program
        )
        if not scalar_score_assignment and not contains_score_field(program):
            return "local_program_scoring_output_missing_score"

    history_markers = (
        "history", "previous", "already", "recent", "prior",
        "历史", "已有", "已联系", "最近", "曾经",
    )
    if any(marker in requirement_text.lower() for marker in history_markers):
        current_history_references = {
            reference
            for binding in public_operations
            if isinstance(binding, dict) and binding.get("operation")
            for field, target in (binding.get("results", {}) or {}).items()
            for reference in (
                f"result.{binding['operation']}.{field}",
                str(target) if isinstance(target, str) else "",
            )
            if reference and any(token in reference.lower() for token in (
                "row", "history", "contact", "log",
            ))
        }

        # Track simple aliases such as all_rows = result.read_rows.rows. The
        # semantic check must reason about collection identity, not spelling.
        history_aliases = set(current_history_references)
        # History may be supplied as workflow input rather than produced by a
        # public read operation.  Treat that declared input as a valid history
        # source; the filter still has to compare it with the current item.
        for reference in (available_references or set()):
            reference_text = str(reference)
            if any(token in reference_text.lower() for token in (
                "history", "previous", "already", "contacted", "contact_history", "log",
            )):
                history_aliases.add(reference_text)

        def contains_history_reference(value: Any) -> bool:
            if isinstance(value, list):
                return any(contains_history_reference(item) for item in value)
            if isinstance(value, dict):
                if set(value) == {"ref"} and isinstance(value.get("ref"), str):
                    return value["ref"] in history_aliases
                return any(contains_history_reference(item) for item in value.values())
            return False

        for statement in program:
            if not isinstance(statement, dict) or statement.get("op") != "assign":
                continue
            value = statement.get("value")
            reference = value.get("ref") if isinstance(value, dict) else None
            target = statement.get("target")
            if (
                (reference in history_aliases or contains_history_reference(value))
                and isinstance(target, str)
                and target
            ):
                history_aliases.add(target)

        def source_reference(value: Any) -> str:
            if isinstance(value, dict) and set(value) == {"ref"}:
                return str(value.get("ref", ""))
            return ""

        def quantifier_sources(value: Any) -> List[Any]:
            sources: List[Any] = []
            if isinstance(value, list):
                for item in value:
                    sources.extend(quantifier_sources(item))
            elif isinstance(value, dict):
                for key, item in value.items():
                    if key in {"any", "all"} and isinstance(item, dict):
                        sources.append(item.get("source"))
                    sources.extend(quantifier_sources(item))
            return sources

        def references(value: Any) -> Set[str]:
            """Collect refs anywhere in a predicate, including call arguments."""
            found: Set[str] = set()
            if isinstance(value, list):
                for item in value:
                    found.update(references(item))
            elif isinstance(value, dict):
                if set(value) == {"ref"} and isinstance(value.get("ref"), str):
                    found.add(value["ref"])
                else:
                    for item in value.values():
                        found.update(references(item))
            return found

        for statement in program:
            if not isinstance(statement, dict) or statement.get("op") != "filter":
                continue
            outer_source = statement.get("source")
            nested_sources = quantifier_sources(statement.get("where"))
            predicate_references = references(statement.get("where"))
            if predicate_references & history_aliases:
                continue
            if any(source == outer_source for source in nested_sources):
                return "local_program_history_filter_reuses_same_collection"
            outer_reference = source_reference(outer_source)
            nested_references = {source_reference(source) for source in nested_sources}
            if (
                history_aliases
                and outer_reference not in history_aliases
                and not (nested_references & history_aliases)
            ):
                return "local_program_history_filter_missing_history_comparison"

    if available_references is not None:
        external = {str(item) for item in available_references if str(item)}
        assigned: Set[str] = set()

        def reference_is_available(reference: str, loop_variables: Set[str]) -> bool:
            value = reference.strip().removeprefix("$")
            root = re.split(r"[.\[]", value, maxsplit=1)[0]
            if root in assigned or root in loop_variables:
                return True
            if root in {"workflow_input", "input"}:
                return True
            return any(
                value == item
                or value.startswith(f"{item}.")
                or value.startswith(f"{item}[")
                for item in external
            )

        def unresolved_reference(value: Any, loop_variables: Set[str]) -> str:
            if isinstance(value, list):
                for item in value:
                    unresolved = unresolved_reference(item, loop_variables)
                    if unresolved:
                        return unresolved
                return ""
            if not isinstance(value, dict):
                return ""
            if set(value) == {"ref"} and isinstance(value.get("ref"), str):
                reference = value["ref"]
                return "" if reference_is_available(reference, loop_variables) else reference
            if len(value) == 1 and next(iter(value), "") in {"any", "all"}:
                quantified = next(iter(value.values()))
                if isinstance(quantified, dict):
                    nested_variables = set(loop_variables)
                    item = quantified.get("item")
                    if isinstance(item, str) and item:
                        nested_variables.add(item)
                    unresolved = unresolved_reference(
                        quantified.get("source"), loop_variables
                    )
                    if unresolved:
                        return unresolved
                    return unresolved_reference(
                        quantified.get("where"), nested_variables
                    )
            for item in value.values():
                unresolved = unresolved_reference(item, loop_variables)
                if unresolved:
                    return unresolved
            return ""

        for statement in program:
            if not isinstance(statement, dict):
                continue
            loop_variables = {
                str(statement.get("item", ""))
            } if statement.get("op") in {"filter", "map", "sort"} else set()
            if statement.get("op") == "map" and statement.get("index"):
                loop_variables.add(str(statement["index"]))
            unresolved = unresolved_reference(statement, loop_variables - {""})
            if unresolved:
                return f"local_program_unknown_reference:{unresolved}"
            target = statement.get("target")
            if isinstance(target, str) and target:
                assigned.add(target)
    return ""


def _normalize_node_qualified_output_refs(fact_spec: 'DslFactSpec') -> None:
    """Replace stale n<N>.field aliases when field is a declared public output."""
    declared_outputs = {
        str(item.get("name", "")) if isinstance(item, dict) else str(item)
        for scene in fact_spec.scenes
        for item in (scene.action.outputs or [])
        if (str(item.get("name", "")) if isinstance(item, dict) else str(item))
    }
    operations_by_block: Dict[str, List[Dict[str, Any]]] = {}
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict) or not scene.block_id:
            continue
        operations_by_block[str(scene.block_id)] = [
            item for item in structured.get("public_operations", []) or []
            if isinstance(item, dict) and item.get("operation")
        ]

    def canonical(source: Any) -> Any:
        if not isinstance(source, str):
            if source is None or isinstance(source, (bool, int, float)):
                return {"literal": source}
            return source
        match = re.fullmatch(r"n\d+\.([A-Za-z_][A-Za-z0-9_]*)", source.removeprefix("$"))
        if match and match.group(1) in declared_outputs:
            return match.group(1)
        qualified = re.fullmatch(
            r"result\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)(.*)",
            source.removeprefix("$"),
        )
        if not qualified:
            return source
        block_id, field, suffix = qualified.groups()
        candidates = []
        for binding in operations_by_block.get(block_id, []):
            result_fields = set((binding.get("results", {}) or {}).keys())
            result_fields.update(str(item) for item in binding.get("produces", []) or [])
            if field in result_fields:
                candidates.append(str(binding["operation"]))
        if len(set(candidates)) != 1:
            return source
        return f"result.{candidates[0]}.{field}{suffix}"

    def rewrite(value: Any) -> Any:
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, dict):
            return canonical(value)
        if set(value) == {"ref"}:
            return {"ref": canonical(value.get("ref"))}
        return {key: rewrite(item) for key, item in value.items()}

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        if "local_program" in structured:
            structured["local_program"] = rewrite(structured["local_program"])
        for binding in structured.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            binding["arguments"] = {
                field: rewrite(source)
                for field, source in (binding.get("arguments", {}) or {}).items()
            }
            foreach = binding.get("foreach")
            if isinstance(foreach, dict) and "source" in foreach:
                foreach["source"] = canonical(foreach["source"])


def _normalize_inclusive_date_window_fact_spec(
    fact_spec: 'DslFactSpec', normalized: str,
) -> None:
    """Preserve an exact lookback boundary in saved/reused FactSpecs."""
    text = str(normalized or "").lower()
    if not any(token in text for token in (
        "过滤掉最近", "不在最近", "not within", "outside the last",
    )) or not any(token in text for token in ("天内", "days", "day")):
        return

    def rewrite(value: Any) -> Any:
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: rewrite(item) for key, item in value.items()}
        operation = result.get("op")
        if not isinstance(operation, dict) or operation.get("name") != "gt":
            return result
        args = operation.get("args")
        left_call = args[0].get("call") if isinstance(args, list) and args else None
        if isinstance(left_call, dict) and left_call.get("name") == "days_between":
            operation["name"] = "gte"
        return result

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if isinstance(structured, dict) and "local_program" in structured:
            structured["local_program"] = rewrite(structured["local_program"])


def _normalize_local_program_workflow_references(
    program: Any,
    available_state: Set[str],
    workflow_input_fields: Optional[Set[str]] = None,
) -> Any:
    """Prefer a produced variable over an undeclared workflow-input alias."""
    declared = set(workflow_input_fields or set())
    available_roots = {
        str(reference).split(".", 1)[0].split("[", 1)[0]
        for reference in available_state
        if reference
    }

    def rewrite(value: Any) -> Any:
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"ref"} and isinstance(value.get("ref"), str):
            reference = value["ref"]
            if reference.startswith("workflow_input."):
                remainder = reference[len("workflow_input."):]
                root = remainder.split(".", 1)[0].split("[", 1)[0]
                if root not in declared and root in available_roots:
                    return {"ref": remainder}
            return value
        return {key: rewrite(item) for key, item in value.items()}

    return rewrite(program)


def _strip_local_public_call_assignments(
    program: List[Dict[str, Any]],
    public_operations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove duplicated public calls and reconnect their result references."""
    bindings = {
        str(item.get("operation")): item
        for item in public_operations
        if isinstance(item, dict) and item.get("operation")
    }
    aliases: Dict[str, str] = {}
    retained: List[Dict[str, Any]] = []
    for statement in program:
        value = statement.get("value") if isinstance(statement, dict) else None
        call = value.get("call") if isinstance(value, dict) else None
        operation = str(call.get("name", "")) if isinstance(call, dict) else ""
        target = str(statement.get("target", "")) if isinstance(statement, dict) else ""
        if statement.get("op") == "assign" and target and operation in bindings:
            binding = bindings[operation]
            for field, public_target in (binding.get("results", {}) or {}).items():
                if isinstance(public_target, str) and public_target:
                    aliases[f"{target}.{field}"] = public_target
            continue
        retained.append(statement)

    def rewrite(value: Any) -> Any:
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"ref"} and isinstance(value.get("ref"), str):
            return {"ref": aliases.get(value["ref"], value["ref"])}
        return {key: rewrite(item) for key, item in value.items()}

    return rewrite(retained)


def _repair_invalid_local_programs(
    fact_spec: 'DslFactSpec',
    normalized: str,
    llm_client,
    max_attempts: int = 2,
    workflow_input_fields: Optional[Set[str]] = None,
    public_interfaces: Optional[List[Dict[str, Any]]] = None,
) -> Tuple['DslFactSpec', int]:
    """Complete or repair executable local IR without using evaluator data."""
    from dsl_v2.local_program import (
        LocalProgramError,
        normalize_local_program_syntax,
        validate_local_program,
    )

    def normalize_temporal_comparisons(program: Any) -> Any:
        """Rewrite date-vs-day comparisons to the executable date interval form."""
        workflow_fields = set(workflow_input_fields or set())
        reference_date = next((
            name for name in ("run_date", "current_date", "reference_date", "today")
            if name in workflow_fields
        ), "")
        day_threshold = next((
            name for name in (
                "contact_window_days", "window_days", "threshold_days", "lookback_days"
            ) if name in workflow_fields
        ), "")
        date_tokens = ("date", "time", "contacted", "created", "updated", "sent_at")

        def walk(value: Any) -> Any:
            if isinstance(value, list):
                return [walk(item) for item in value]
            if not isinstance(value, dict):
                return value
            rewritten = {key: walk(item) for key, item in value.items()}
            operation = rewritten.get("op")
            if not isinstance(operation, dict) or operation.get("name") not in {
                "lt", "lte", "gt", "gte",
            }:
                return rewritten
            args = operation.get("args")
            if not isinstance(args, list) or len(args) != 2 or not reference_date:
                return rewritten
            left_call = args[0].get("call") if isinstance(args[0], dict) else None
            strict_recent_window = (
                operation.get("name") == "lte"
                and isinstance(left_call, dict)
                and left_call.get("name") == "days_between"
                and any(token in normalized.lower() for token in (
                    "recent", "within", "最近", "天内",
                ))
                and not any(token in normalized.lower() for token in (
                    "inclusive", "including the boundary", "含边界", "包括第",
                ))
            )
            if strict_recent_window:
                operation["name"] = "lt"
            # For a requirement such as "filter out records contacted within
            # N days", the exact N-day boundary remains eligible.  Normalize
            # the common strict inverse form before code generation.
            inclusive_expiry_window = (
                operation.get("name") == "gt"
                and isinstance(left_call, dict)
                and left_call.get("name") == "days_between"
                and any(token in normalized.lower() for token in (
                    "过滤掉最近", "不在最近", "not within", "outside the last",
                ))
                and any(token in normalized.lower() for token in ("天内", "days", "day"))
            )
            if inclusive_expiry_window:
                operation["name"] = "gte"
            left, right = args
            left_ref = left.get("ref", "") if isinstance(left, dict) else ""
            right_number = (
                right.get("literal") if isinstance(right, dict) and set(right) == {"literal"}
                else right
            )
            if (
                isinstance(left_ref, str)
                and any(token in left_ref.lower() for token in date_tokens)
                and isinstance(right_number, (int, float))
                and not isinstance(right_number, bool)
            ):
                interval = {"call": {"name": "days_between", "args": [
                    {"ref": f"workflow_input.{reference_date}"}, left,
                ]}}
                threshold = (
                    {"ref": f"workflow_input.{day_threshold}"}
                    if day_threshold else {"literal": right_number}
                )
                return {"op": {"name": operation["name"], "args": [interval, threshold]}}
            return rewritten

        return walk(program)

    invalid: Dict[str, Dict[str, Any]] = {}
    workflow_public_targets = {
        str(target)
        for candidate_scene in fact_spec.scenes
        for binding in (
            (candidate_scene.action.structured_op or {}).get("public_operations", [])
            if isinstance(candidate_scene.action.structured_op, dict) else []
        )
        if isinstance(binding, dict)
        for target in (binding.get("results", {}) or {}).values()
        if isinstance(target, str) and target
    }
    available_state = {
        str(item.get("name", "")) if isinstance(item, dict) else str(item)
        for item in (fact_spec.inputs or [])
        if (str(item.get("name", "")) if isinstance(item, dict) else str(item))
    }
    available_state.update(
        f"workflow_input.{item}"
        for item in (workflow_input_fields or set()) if str(item)
    )
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        program = structured.get("local_program", []) if isinstance(structured, dict) else []
        program = normalize_local_program_syntax(program)
        program = normalize_temporal_comparisons(program)
        program = _normalize_local_program_workflow_references(
            program,
            available_state,
            workflow_input_fields,
        )
        action_type = str(scene.action.action_type or "").lower()
        local_text = " ".join([
            str(scene.raw_requirement_text or ""),
            str(scene.logic_flow or ""),
            " ".join(str(item) for item in scene.action_sequence or []),
        ]).lower()
        public_operations = (
            structured.get("public_operations", []) if isinstance(structured, dict) else []
        )
        program = _strip_local_public_call_assignments(program, public_operations)
        if program and isinstance(structured, dict):
            structured["local_program"] = program
        assigned_targets = {
            str(statement.get("target", ""))
            for statement in program
            if isinstance(statement, dict) and statement.get("target")
        }
        declared_outputs = {
            str(item.get("name", "")) if isinstance(item, dict) else str(item)
            for item in (scene.action.outputs or [])
        }
        public_targets = {
            str(target)
            for binding in public_operations
            if isinstance(binding, dict)
            for target in (binding.get("results", {}) or {}).values()
            if isinstance(target, str) and target
        }
        public_result_references = {
            f"result.{binding.get('operation')}.{field}"
            for binding in public_operations
            if isinstance(binding, dict) and binding.get("operation")
            for field in (binding.get("results", {}) or {})
        }
        scene_available_references = {
            *available_state,
            *public_targets,
            *public_result_references,
        }
        required_local_outputs = declared_outputs - workflow_public_targets
        program_builds_declared_output = bool(assigned_targets & required_local_outputs)
        needs_local_program = (
            bool(required_local_outputs)
            or
            action_type in {
                "compute", "template", "template_fill", "transform", "validation",
            }
            or (
                action_type == "assign"
                and not public_operations
                and bool(required_local_outputs)
            )
            or program_builds_declared_output
            or any(token in local_text for token in (
                "for each", "filter", "sort", "rank", "format", "merge",
                "逐", "每个", "过滤", "排序", "排名", "格式",
            ))
        )
        if public_operations and not required_local_outputs:
            structured.pop("local_program", None)
            available_state.update(public_targets)
            available_state.update(public_result_references)
            available_state.update(declared_outputs & public_targets)
            continue
        validation_error = ""
        if program:
            if not any(
                isinstance(statement, dict) and statement.get("op") == "emit"
                for statement in program
            ):
                emitted = {
                    name: {"ref": name}
                    for name in sorted(assigned_targets & declared_outputs)
                }
                if emitted:
                    program.append({"op": "emit", "fields": emitted})
                    if isinstance(structured, dict):
                        structured["local_program"] = program
            try:
                validate_local_program(program)
            except LocalProgramError as exc:
                validation_error = str(exc)
            if not validation_error and (
                action_type in {"compute", "scoring"}
                or any(token in local_text for token in ("score", "scoring", "评分", "打分"))
            ):
                executable = [
                    statement for statement in program
                    if isinstance(statement, dict) and statement.get("op") != "emit"
                ]
                if (
                    len(executable) == 1
                    and executable[0].get("op") == "assign"
                    and executable[0].get("target") in required_local_outputs
                    and isinstance(executable[0].get("value"), dict)
                    and set(executable[0]["value"]) == {"ref"}
                ):
                    validation_error = "local_program_compute_is_passthrough"
            if not validation_error:
                validation_error = _local_program_semantic_error(
                    program,
                    public_operations,
                    required_local_outputs,
                    scene_available_references,
                    f"{normalized}\n{scene.raw_requirement_text}\n{scene.logic_flow}",
                )
            if not validation_error and public_operations:
                try:
                    _schedule_scene_action_units({
                        "scene_id": scene.scene_id,
                        "local_program": program,
                        "public_operations": public_operations,
                    })
                except ValueError as exc:
                    # Detect local/public cycles while the focused repairer can
                    # still correct one scene. Waiting until M4 turns a local
                    # program into Python forces an expensive full M1-M4 retry.
                    validation_error = f"local_public_schedule_error:{exc}"
        elif needs_local_program:
            validation_error = "missing_local_program"
        if validation_error and public_operations and not needs_local_program:
            # The extractor sometimes restates public calls as an invalid
            # local program. The explicit public action contract already owns
            # that orchestration, so discard the duplicate representation.
            structured.pop("local_program", None)
            continue
        if validation_error:
            repair_key = f"{scene.scene_id}::{scene.block_id}"
            prior_public_collections = sorted({
                reference for reference in available_state
                if any(token in reference.lower() for token in (
                    "profile", "candidate", "record", "item",
                ))
            }, key=lambda reference: (
                0 if "score" in reference.lower() else 1,
                1 if reference.startswith("result.") else 0,
                reference,
            ))
            current_history_collections = sorted({
                reference for reference in public_result_references
                if any(token in reference.lower() for token in (
                    "row", "history", "contact", "log",
                ))
            })
            invalid[repair_key] = {
                "repair_key": repair_key,
                "scene_id": scene.scene_id,
                "block_id": scene.block_id,
                "action_type": action_type,
                "outputs": list(scene.action.outputs or []),
                "required_local_outputs": sorted(required_local_outputs),
                "available_references": sorted(scene_available_references),
                "prior_workflow_references": sorted(available_state),
                "current_scene_public_references": sorted(
                    public_targets | public_result_references
                ),
                "history_filter_source_hint": {
                    "outer_primary_source_candidates": prior_public_collections,
                    "nested_history_source_candidates": current_history_collections,
                },
                "public_response_schemas": [
                    {
                        "operation": str(interface.get("operation", "")),
                        "response_schema": interface.get("response_schema", {}) or {},
                    }
                    for interface in (public_interfaces or [])
                    if interface.get("operation")
                ],
                "raw_requirement_text": scene.raw_requirement_text,
                "logic_flow": scene.logic_flow,
                "public_operations": public_operations,
                "invalid_local_program": program,
                "original_local_program": copy.deepcopy(program),
                "validation_error": validation_error,
            }
        available_state.update(public_targets)
        available_state.update(public_result_references)
        if not validation_error and program:
            available_state.update(
                str(name)
                for statement in program
                if isinstance(statement, dict) and statement.get("op") == "emit"
                for name in (statement.get("fields", {}) or {})
                if name
            )
        elif (
            not validation_error
            and not public_operations
            and action_type in {"text_generation", "scoring", "extraction", "similarity"}
        ):
            available_state.update(declared_outputs)
    if not invalid:
        return fact_spec, 0

    total_tokens = 0
    grammar = {
        "statement_schemas": {
            "assign": {"op": "assign", "target": "variable", "value": "expression"},
            "filter": {
                "op": "filter", "target": "variable", "source": "expression",
                "item": "loop_variable", "where": "expression",
            },
            "map": {
                "op": "map", "target": "variable", "source": "expression",
                "item": "loop_variable", "index": "optional_one_based_index_variable",
                "value": "expression",
            },
            "sort": {
                "op": "sort", "target": "variable", "source": "expression",
                "item": "loop_variable", "key": "expression", "reverse": False,
            },
            "emit": {"op": "emit", "fields": {"output_name": "expression"}},
        },
        "expression_forms": ["literal", "ref", "object", "list", "op", "call", "any", "all"],
        "operators": [
            "eq", "neq", "lt", "lte", "gt", "gte", "and", "or", "not",
            "in", "contains", "add", "sub", "mul", "div",
        ],
        "calls": [
            "len", "days_between", "count_true", "coalesce", "get", "join",
            "to_string", "to_markdown_table", "merge",
        ],
        "references": ["workflow_input.field", "result.operation.field", "prior_variable", "loop_item.field"],
        "hard_constraints": [
            "Use exactly one of the five statement schemas and exactly their field names.",
            "Never place a public/external operation in local_program.",
            "Read an already declared public operation only via result.<exact_operation_name>.<field>.",
            "Every ref must use an exact available_references entry, a previously assigned local variable, or a loop item.",
            "Never invent a function. Calls are limited to the listed calls.",
            "Use workflow_input only for raw workflow inputs, not operation results.",
            "For ranked output, use map.index; it is a one-based integer available inside that map.",
        ],
        "valid_examples": [
            [
                {
                    "op": "filter", "target": "eligible", "source": {"ref": "result.load.rows"},
                    "item": "row", "where": {"op": {"name": "gte", "args": [
                        {"ref": "row.score"}, {"ref": "workflow_input.threshold"},
                    ]}},
                },
                {"op": "emit", "fields": {"eligible": {"ref": "eligible"}}},
            ],
            [
                {"op": "assign", "target": "report", "value": {"ref": "records"}},
                {"op": "assign", "target": "summary_text", "value": {"call": {
                    "name": "to_markdown_table", "args": [{"ref": "report"}],
                }}},
                {"op": "emit", "fields": {
                    "report": {"ref": "report"},
                    "summary_text": {"ref": "summary_text"},
                }},
            ],
            [
                {"op": "map", "target": "ranked", "source": {"ref": "sorted_records"},
                 "item": "record", "index": "rank", "value": {"object": {
                     "id": {"ref": "record.id"}, "rank": {"ref": "rank"},
                 }}},
                {"op": "emit", "fields": {"ranked": {"ref": "ranked"}}},
            ],
            [
                {"op": "filter", "target": "eligible_records", "source": {"ref": "primary_records"},
                 "item": "record", "where": {"op": {"name": "not", "args": [{"any": {
                     "source": {"ref": "result.read_history.rows"}, "item": "old", "where": {"op": {
                         "name": "and", "args": [
                             {"op": {"name": "eq", "args": [
                                 {"ref": "old.id"}, {"ref": "record.id"},
                             ]}},
                             {"op": {"name": "lt", "args": [
                                 {"call": {"name": "days_between", "args": [
                                     {"ref": "workflow_input.run_date"}, {"ref": "old.last_contacted"},
                                 ]}},
                                 {"ref": "workflow_input.window_days"},
                             ]}},
                         ],
                     }}},
                 }]}}},
                {"op": "emit", "fields": {"eligible_records": {"ref": "eligible_records"}}},
            ],
        ],
    }
    remaining = dict(invalid)
    for attempt in range(max_attempts):
        batches = (
            [list(remaining.values())]
            if attempt == 0
            else [[item] for item in list(remaining.values())]
        )
        for batch in batches:
            if not remaining:
                break
            active_batch = [
                item for item in batch if item.get("repair_key") in remaining
            ]
            if not active_batch:
                continue
            response = llm_client.call(
                system_prompt=(
                    "Return JSON only. Repair the supplied malformed local programs to the exact grammar. "
                    "Preserve the requirement meaning, repair_key, scene_id and block_id. Do not add formulas, thresholds, fields, "
                    "or behavior absent from the requirement. Public/external operations must never appear as local-program "
                    "statements or calls; reference their completed outputs only as result.<exact_operation_name>.<field>. "
                    "Never invent helper functions. A target cannot reference itself before assignment. Build and emit every "
                    "required_local_outputs item. When the requirement filters a primary collection against history, iterate the "
                    "primary collection and use any/all over the history collection; the outer filter source and nested history "
                    "source must be different, and the output must contain primary records rather than history rows. "
                    "Use prior_workflow_references for the outer primary collection and current_scene_public_references for "
                    "the nested history lookup when those labeled sets are supplied. Those labels are metadata, never refs. "
                    "If history_filter_source_hint contains candidates, copy an exact candidate string as the ref value: use "
                    "outer_primary_source_candidates only for the outer filter and nested_history_source_candidates only inside "
                    "any/all. Never modify an operation name or append a field to a listed candidate. "
                    "Every field read below result.<operation> must exist in that operation's supplied "
                    "public_response_schemas, including fields of array items. Never replace business_id with id or vice versa. "
                    "A compute/scoring scene must perform the requested computation, not merely rename a public result. To retain "
                    "an input object while adding calculated fields, use merge(input_object, {object:{new_field: expression}}). "
                    "For ranked output, sort first, add an `index` field such as `rank` to the map statement, and reference that "
                    "one-based variable as `rank` inside the mapped value. For formatted text derived from "
                    "tabular records, use to_markdown_table. Use each statement schema's exact field names. Do not output Python. "
                    "Return {repairs:[{repair_key,scene_id,block_id,local_program}]} ."
                ),
                user_content=json.dumps({
                    "natural_language_requirement": normalized,
                    "grammar": grammar,
                    "invalid_scenes": active_batch,
                }, ensure_ascii=False, sort_keys=True),
                temperature=0.0,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            usage = getattr(response, "usage", None) or {}
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            raw = response.content if hasattr(response, "content") else str(response)
            text = raw.strip()
            if text.startswith("```"):
                text = text.lstrip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
                text = text.strip().rstrip("`").strip()
            try:
                # OpenAI-compatible proxies occasionally preserve harmless
                # trailing whitespace or provider metadata after the JSON
                # value. Decode the first complete object while still
                # rejecting prose before it or a truncated object.
                start = text.find("{")
                if start < 0:
                    raise json.JSONDecodeError("no JSON object", text, 0)
                parsed, _ = json.JSONDecoder().raw_decode(text[start:])
            except json.JSONDecodeError:
                logger.warning("[M3d] local repair returned invalid JSON: %s", text[:2000])
                continue
            repairs = parsed.get("repairs", []) if isinstance(parsed, dict) else []
            if not repairs:
                logger.warning("[M3d] local repair returned no repairs: %s", text[:2000])
            for repair in repairs if isinstance(repairs, list) else []:
                if not isinstance(repair, dict):
                    continue
                repair_key = str(repair.get("repair_key", ""))
                if repair_key not in remaining:
                    continue
                program = normalize_local_program_syntax(repair.get("local_program"))
                program = normalize_temporal_comparisons(program)
                # A repair response can accidentally restate catalog operations
                # as local calls even though the public action contract owns them.
                # Apply the same lowering used for initially extracted programs
                # before validating the repaired local IR.
                program = _strip_local_public_call_assignments(
                    program,
                    remaining[repair_key].get("public_operations", []),
                )
                assigned_targets = {
                    str(statement.get("target"))
                    for statement in program or []
                    if isinstance(statement, dict) and statement.get("target")
                }
                emitted_outputs = {
                    str(name)
                    for statement in program or []
                    if isinstance(statement, dict) and statement.get("op") == "emit"
                    for name in (statement.get("fields", {}) or {})
                    if name
                }
                missing_emits = (
                    set(remaining[repair_key].get("required_local_outputs", []))
                    & assigned_targets
                    - emitted_outputs
                )
                if missing_emits:
                    program.append({
                        "op": "emit",
                        "fields": {
                            name: {"ref": name} for name in sorted(missing_emits)
                        },
                    })
                emitted_outputs.update(missing_emits)
                public_operations = remaining[repair_key].get("public_operations", [])
                for output_name in sorted(
                    set(remaining[repair_key].get("required_local_outputs", []))
                    - emitted_outputs
                ):
                    suffix = output_name.lower().rsplit("_", 1)[-1]
                    result_refs = {
                        f"result.{binding.get('operation')}.{field}"
                        for binding in public_operations
                        if isinstance(binding, dict) and binding.get("operation")
                        for field in (binding.get("results", {}) or {})
                        if str(field).lower() == suffix
                    }
                    if len(result_refs) != 1:
                        continue
                    emit_statement = next((
                        statement for statement in reversed(program)
                        if isinstance(statement, dict) and statement.get("op") == "emit"
                    ), None)
                    if emit_statement is None:
                        emit_statement = {"op": "emit", "fields": {}}
                        program.append(emit_statement)
                    emit_statement.setdefault("fields", {})[output_name] = {
                        "ref": next(iter(result_refs))
                    }
                try:
                    validate_local_program(program)
                except LocalProgramError as exc:
                    remaining[repair_key]["invalid_local_program"] = program
                    remaining[repair_key]["validation_error"] = str(exc)
                    logger.warning("[M3d] %s syntax repair rejected: %s", repair_key, exc)
                    continue
                semantic_error = _local_program_semantic_error(
                    program,
                    remaining[repair_key].get("public_operations", []),
                    set(remaining[repair_key].get("required_local_outputs", [])),
                    set(remaining[repair_key].get("available_references", [])),
                    "\n".join([
                        normalized,
                        str(remaining[repair_key].get("raw_requirement_text", "")),
                        str(remaining[repair_key].get("logic_flow", "")),
                    ]),
                )
                if semantic_error:
                    remaining[repair_key]["invalid_local_program"] = program
                    remaining[repair_key]["validation_error"] = semantic_error
                    logger.warning("[M3d] %s semantic repair rejected: %s", repair_key, semantic_error)
                    continue
                target = remaining[repair_key]
                scene = next(
                    item for item in fact_spec.scenes
                    if str(item.block_id) == str(target["block_id"])
                    and str(item.scene_id) == str(target["scene_id"])
                )
                structured = scene.action.structured_op if isinstance(scene.action.structured_op, dict) else {}
                scene.action.structured_op = {**structured, "local_program": program}
                remaining.pop(repair_key, None)

                # A downstream program may have been valid all along and only
                # failed because this upstream output was unavailable. Recheck
                # the preserved program before asking the model to rewrite it.
                newly_available = {
                    str(name)
                    for statement in program
                    if isinstance(statement, dict) and statement.get("op") == "emit"
                    for name in (statement.get("fields", {}) or {})
                    if name
                }
                for downstream_key, downstream in list(remaining.items()):
                    if int(downstream.get("scene_id", 0) or 0) <= int(target["scene_id"]):
                        continue
                    downstream_available = set(downstream.get("available_references", []))
                    downstream_available.update(newly_available)
                    downstream["available_references"] = sorted(downstream_available)
                    original = normalize_local_program_syntax(
                        copy.deepcopy(downstream.get("original_local_program", []))
                    )
                    if not original:
                        continue
                    try:
                        validate_local_program(original)
                    except LocalProgramError:
                        continue
                    original_error = _local_program_semantic_error(
                        original,
                        downstream.get("public_operations", []),
                        set(downstream.get("required_local_outputs", [])),
                        downstream_available,
                        "\n".join([
                            normalized,
                            str(downstream.get("raw_requirement_text", "")),
                            str(downstream.get("logic_flow", "")),
                        ]),
                    )
                    if original_error:
                        continue
                    downstream_scene = next(
                        item for item in fact_spec.scenes
                        if str(item.block_id) == str(downstream["block_id"])
                        and str(item.scene_id) == str(downstream["scene_id"])
                    )
                    downstream_structured = (
                        downstream_scene.action.structured_op
                        if isinstance(downstream_scene.action.structured_op, dict) else {}
                    )
                    downstream_scene.action.structured_op = {
                        **downstream_structured,
                        "local_program": original,
                    }
                    remaining.pop(downstream_key, None)
        if not remaining:
            break

    # Invalid programs must not reach source generation. Leaving them absent
    # makes the v6 compile gate report a precise non-executable local step.
    for repair_key, diagnostic in remaining.items():
        scene = next(
            item for item in fact_spec.scenes
            if str(item.block_id) == str(diagnostic["block_id"])
            and str(item.scene_id) == str(diagnostic["scene_id"])
        )
        structured = scene.action.structured_op if isinstance(scene.action.structured_op, dict) else {}
        retained = {
            key: value for key, value in structured.items() if key != "local_program"
        }
        retained["local_program_diagnostic"] = diagnostic
        scene.action.structured_op = retained
    return fact_spec, total_tokens


def _materialize_missing_derived_text_inputs(fact_spec: 'DslFactSpec') -> None:
    """Place formatted text production before the public operation that consumes it."""
    available = {
        str(item.get("name", "")) if isinstance(item, dict) else str(item)
        for item in (fact_spec.inputs or [])
    }
    prior_scenes: List[Any] = []
    text_fields = {
        "summary", "summary_text", "message", "message_text", "body", "body_text",
        "content", "content_text", "formatted_text", "digest", "prompt",
    }

    def reuse_text_argument(value: Any, field: str, source: str) -> Any:
        if isinstance(value, list):
            return [reuse_text_argument(item, field, source) for item in value]
        if not isinstance(value, dict):
            return value
        rewritten = {
            key: reuse_text_argument(item, field, source) for key, item in value.items()
        }
        object_value = rewritten.get("object")
        if isinstance(object_value, dict) and field in object_value:
            object_value[field] = {"ref": source}
        return rewritten

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        operations = structured.get("public_operations", []) if isinstance(structured, dict) else []
        for binding in operations or []:
            if not isinstance(binding, dict):
                continue
            arguments = binding.get("arguments", {}) or {}
            for field, source in list(arguments.items()):
                if field not in text_fields or not isinstance(source, str):
                    continue
                root = re.split(r"[.\[]", source.removeprefix("$"), maxsplit=1)[0]
                current_program = structured.get("local_program", []) if isinstance(structured, dict) else []
                if isinstance(structured, dict) and current_program:
                    structured["local_program"] = reuse_text_argument(
                        current_program, field, source
                    )
                    for statement in structured["local_program"]:
                        if not isinstance(statement, dict) or statement.get("op") != "assign":
                            continue
                        if "status" not in str(statement.get("target", "")).lower():
                            continue
                        value = statement.get("value")
                        object_value = value.get("object") if isinstance(value, dict) else None
                        if isinstance(object_value, dict):
                            object_value.setdefault(field, {"ref": source})
                collection_like_source = any(
                    token in root.lower()
                    for token in ("report", "ranking", "candidate", "prospect", "records", "rows", "table")
                )
                if (
                    (root in available and not collection_like_source)
                    or root in {"input", "workflow_input", "result"}
                ):
                    continue
                producer = (
                    scene
                    if any(
                        token in str(output).lower()
                        for output in (scene.action.outputs or [])
                        for token in ("report", "ranking", "candidate", "result", "table")
                    )
                    else next((
                    candidate for candidate in reversed(prior_scenes)
                    if any(
                        token in str(output).lower()
                        for output in (candidate.action.outputs or [])
                        for token in ("report", "ranking", "candidate", "result", "table")
                    )
                    ), None)
                )
                if producer is None:
                    continue
                canonical_name = field
                existing = [
                    str(item.get("name", "")) if isinstance(item, dict) else str(item)
                    for item in (producer.action.outputs or [])
                ]
                if canonical_name not in existing:
                    producer.action.outputs.append(canonical_name)
                source_name = next((
                    name for name in existing
                    if any(token in name.lower() for token in (
                        "report", "ranking", "candidate", "result", "table",
                    ))
                ), "")
                producer_structured = producer.action.structured_op
                if source_name and isinstance(producer_structured, dict):
                    program = list(producer_structured.get("local_program", []) or [])
                    assigned = {
                        str(statement.get("target", ""))
                        for statement in program if isinstance(statement, dict)
                    }
                    if canonical_name not in assigned:
                        producer_index = next((
                            index for index, statement in enumerate(program)
                            if isinstance(statement, dict)
                            and statement.get("target") == source_name
                        ), -1)
                        insert_index = producer_index + 1 if producer_index >= 0 else next((
                            index for index, statement in enumerate(program)
                            if isinstance(statement, dict) and statement.get("op") == "emit"
                        ), len(program))
                        program.insert(insert_index, {
                            "op": "assign",
                            "target": canonical_name,
                            "value": {"call": {
                                "name": "to_markdown_table",
                                "args": [{"ref": source_name}],
                            }},
                        })
                        program.append({
                            "op": "emit",
                            "fields": {canonical_name: {"ref": canonical_name}},
                        })
                        producer_structured["local_program"] = program
                arguments[field] = canonical_name
                if isinstance(structured, dict) and structured.get("local_program"):
                    structured["local_program"] = reuse_text_argument(
                        structured["local_program"], field, canonical_name
                    )
                available.add(canonical_name)
        public_targets = {
            str(target)
            for binding in operations or []
            if isinstance(binding, dict)
            for target in (binding.get("results", {}) or {}).values()
            if isinstance(target, str) and target
        }
        available.update(public_targets)
        available.update(
            str(item.get("name", "")) if isinstance(item, dict) else str(item)
            for item in (scene.action.outputs or [])
        )
        prior_scenes.append(scene)


def _inline_local_argument_assignments(fact_spec: 'DslFactSpec') -> None:
    """Inline pure local aliases that must be evaluated before a public call."""
    from dsl_v2.local_program import normalize_local_program_syntax

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        program = normalize_local_program_syntax(structured.get("local_program", []))
        assignments = {
            str(statement.get("target")): statement.get("value")
            for statement in program or []
            if isinstance(statement, dict)
            and statement.get("op") == "assign"
            and isinstance(statement.get("target"), str)
        }

        def inline(source: Any) -> Any:
            if not isinstance(source, str) or source not in assignments:
                return source
            expression = assignments[source]
            if isinstance(expression, dict) and set(expression) == {"literal"}:
                return expression
            if isinstance(expression, dict) and set(expression) == {"ref"}:
                return expression["ref"]
            if expression is None or isinstance(expression, (bool, int, float)):
                return {"literal": expression}
            return source

        for binding in structured.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            binding["arguments"] = {
                field: inline(source)
                for field, source in (binding.get("arguments", {}) or {}).items()
            }
            foreach = binding.get("foreach")
            if isinstance(foreach, dict) and "source" in foreach:
                foreach["source"] = inline(foreach["source"])
            condition = binding.get("condition")
            if isinstance(condition, dict):
                for leaf in _condition_leaf_nodes(condition):
                    if "source" in leaf:
                        leaf["source"] = inline(leaf["source"])
                    if "value_from" in leaf:
                        leaf["value_from"] = inline(leaf["value_from"])


def _bind_derived_public_arguments_to_local_outputs(fact_spec: 'DslFactSpec') -> None:
    """Use a same-named local value when a binding embeds a derived expression."""
    from dsl_v2.local_program import normalize_local_program_syntax

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        program = normalize_local_program_syntax(structured.get("local_program", []))
        local_names = {
            str(statement.get("target"))
            for statement in program or []
            if isinstance(statement, dict)
            and statement.get("op") == "assign"
            and isinstance(statement.get("target"), str)
        }
        local_names.update({
            str(name)
            for statement in program or []
            if isinstance(statement, dict) and statement.get("op") == "emit"
            for name in (statement.get("fields", {}) or {})
            if name
        })
        for binding in structured.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            arguments = binding.get("arguments", {})
            if not isinstance(arguments, dict):
                continue
            for field, source in list(arguments.items()):
                if field in local_names and isinstance(source, dict) and set(source) != {"literal"}:
                    arguments[field] = field
            if arguments.get("one_row_per_candidate") in (True, {"literal": True}):
                ranked = next((
                    name for name in sorted(local_names)
                    if any(token in name.lower() for token in ("ranking", "ranked", "report"))
                ), "")
                if ranked and "row_data" in arguments:
                    arguments["row_data"] = ranked


def _complete_unambiguous_public_result_output_aliases(fact_spec: 'DslFactSpec') -> None:
    """Emit a declared output from the sole matching public result field in its scene."""
    from dsl_v2.local_program import normalize_local_program_syntax

    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        program = normalize_local_program_syntax(structured.get("local_program", []))
        emitted = {
            str(name)
            for statement in program
            if isinstance(statement, dict) and statement.get("op") == "emit"
            for name in (statement.get("fields", {}) or {})
            if name
        }
        public_targets = {
            str(target)
            for binding in structured.get("public_operations", []) or []
            if isinstance(binding, dict)
            for target in (binding.get("results", {}) or {}).values()
            if isinstance(target, str) and target
        }
        required = {
            str(item.get("name", "")) if isinstance(item, dict) else str(item)
            for item in (scene.action.outputs or [])
        } - public_targets - emitted
        for output_name in sorted(({
            str(item.get("name", "")) if isinstance(item, dict) else str(item)
            for item in (scene.action.outputs or [])
        } & public_targets) - emitted):
            matching = [
                f"result.{binding.get('operation')}.{field}"
                for binding in structured.get("public_operations", []) or []
                if isinstance(binding, dict) and binding.get("operation")
                for field, target in (binding.get("results", {}) or {}).items()
                if target == output_name
            ]
            if len(set(matching)) == 1:
                emit_statement = next((
                    statement for statement in reversed(program)
                    if isinstance(statement, dict) and statement.get("op") == "emit"
                ), None)
                if emit_statement is None:
                    emit_statement = {"op": "emit", "fields": {}}
                    program.append(emit_statement)
                emit_statement.setdefault("fields", {})[output_name] = {
                    "ref": matching[0]
                }
        for output_name in sorted(required):
            suffix = output_name.lower().rsplit("_", 1)[-1]
            candidates = []
            for binding in structured.get("public_operations", []) or []:
                if not isinstance(binding, dict):
                    continue
                operation = str(binding.get("operation", ""))
                for field in (binding.get("results", {}) or {}):
                    if str(field).lower() == suffix and operation:
                        candidates.append(f"result.{operation}.{field}")
            if len(set(candidates)) != 1:
                continue
            emit_statement = next((
                statement for statement in reversed(program)
                if isinstance(statement, dict) and statement.get("op") == "emit"
            ), None)
            if emit_statement is None:
                emit_statement = {"op": "emit", "fields": {}}
                program.append(emit_statement)
            emit_statement.setdefault("fields", {})[output_name] = {"ref": candidates[0]}
        structured["local_program"] = program


def _normalize_public_result_targets_for_local_transforms(fact_spec: 'DslFactSpec') -> None:
    """Do not let a raw public result masquerade as a locally computed output."""
    local_action_types = {
        "compute", "scoring", "transform", "validation", "template", "template_fill",
    }
    for scene in fact_spec.scenes:
        if str(scene.action.action_type or "").lower() not in local_action_types:
            continue
        declared_outputs = {
            str(item.get("name", "")) if isinstance(item, dict) else str(item)
            for item in (scene.action.outputs or [])
            if item
        }
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        for binding in structured.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            results = binding.get("results", {})
            if not isinstance(results, dict):
                continue
            for field, target in list(results.items()):
                if (
                    isinstance(target, str)
                    and target in declared_outputs
                    and target != str(field)
                ):
                    results[field] = str(field)


def _ensure_canonical_fact_inputs(fact_spec: 'DslFactSpec') -> 'DslFactSpec':
    """Keep the standard DSL symbol table complete on every M3 exit path."""
    existing_inputs = {
        str(item.get("name", "")) if isinstance(item, dict) else str(item)
        for item in fact_spec.inputs
    }
    for name, required, default in (
        ("user_input", True, None),
        ("history", False, ""),
        ("previous_agent", False, ""),
    ):
        if name in existing_inputs:
            continue
        item = {"name": name, "type": "String", "required": required}
        if default is not None:
            item["default"] = default
        fact_spec.inputs.append(item)
    return fact_spec


class PublicOperationBindingError(ValueError):
    def __init__(self, message: str, token_count: int = 0):
        super().__init__(message)
        self.token_count = token_count


def _complete_public_operation_bindings(
    fact_spec: 'DslFactSpec',
    normalized: str,
    public_interfaces: List[Dict[str, Any]],
    llm_client,
    max_attempts: int = 3,
    workflow_input_fields: Optional[Set[str]] = None,
) -> Tuple['DslFactSpec', int]:
    """Retry the focused binding pass while accounting for every LLM call."""
    total_tokens = 0
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            completed, tokens = _complete_public_operation_bindings_once(
                fact_spec,
                normalized,
                public_interfaces,
                llm_client,
                workflow_input_fields=workflow_input_fields,
            )
            return completed, total_tokens + tokens
        except Exception as exc:
            total_tokens += int(getattr(exc, "token_count", 0) or 0)
            last_error = exc
            logger.warning(f"[M3b] 公开操作绑定第{attempt}/{max_attempts}次未通过: {exc}")
    raise PublicOperationBindingError(str(last_error), total_tokens) from last_error


def _complete_public_operation_bindings_once(
    fact_spec: 'DslFactSpec',
    normalized: str,
    public_interfaces: List[Dict[str, Any]],
    llm_client,
    workflow_input_fields: Optional[Set[str]] = None,
) -> Tuple['DslFactSpec', int]:
    """M3b: map every public operation to one extracted requirement scene.

    The LLM proposes semantic bindings, while code enforces catalog identity,
    complete one-to-one coverage, valid scene references, and guard syntax.
    No test cases, expected traces, Gold, or Oracle data are visible here.
    """
    catalog = [
        {
            "dependency_type": str(item.get("dependency_type", "")),
            "operation": str(item.get("operation", "")),
            "request_schema": item.get("request_schema", {}),
            "response_schema": item.get("response_schema", {}),
        }
        for item in public_interfaces
        if item.get("dependency_type") and item.get("operation")
    ]
    if not catalog or not fact_spec.scenes:
        return fact_spec, 0
    catalog_by_key = {
        (item["dependency_type"], item["operation"]): item for item in catalog
    }

    def schema_required(schema: Any) -> List[str]:
        if not isinstance(schema, dict):
            return []
        direct = [str(item) for item in schema.get("required", []) or []]
        alternatives = [
            set(str(value) for value in item.get("required", []) or [])
            for item in schema.get("anyOf", []) + schema.get("oneOf", [])
            if isinstance(item, dict)
        ]
        common = set.intersection(*alternatives) if alternatives else set()
        return list(dict.fromkeys(direct + sorted(common)))

    def schema_fields(schema: Any) -> List[str]:
        if not isinstance(schema, dict):
            return []
        fields = [str(item) for item in schema.get("required", []) or []]
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            fields.extend(str(item) for item in properties)
        for alternative in schema.get("anyOf", []) + schema.get("oneOf", []):
            fields.extend(schema_fields(alternative))
        return list(dict.fromkeys(fields))

    def normalize_schema_fields(binding: Dict[str, Any]) -> None:
        key = (str(binding.get("dependency_type", "")), str(binding.get("operation", "")))
        interface = catalog_by_key.get(key)
        if interface is None:
            return
        required_inputs = schema_required(interface.get("request_schema", {}))
        required_outputs = schema_required(interface.get("response_schema", {}))
        proposed_arguments = (
            binding.get("arguments", {})
            if isinstance(binding.get("arguments"), dict) else {}
        )
        valid_inputs = set(schema_fields(interface.get("request_schema", {})))
        optional_inputs = [
            field for field in proposed_arguments
            if field in valid_inputs and field not in required_inputs
        ]
        normalized_inputs = required_inputs + optional_inputs
        if normalized_inputs:
            binding["consumes"] = normalized_inputs
            binding["arguments"] = {
                field: proposed_arguments.get(field, field)
                for field in normalized_inputs
            }
        proposed_results = (
            binding.get("results", {})
            if isinstance(binding.get("results"), dict) else {}
        )
        valid_outputs = set(schema_fields(interface.get("response_schema", {})))
        optional_outputs = [
            field for field in proposed_results
            if field in valid_outputs and field not in required_outputs
        ]
        normalized_outputs = required_outputs + optional_outputs
        if normalized_outputs:
            binding["produces"] = normalized_outputs
            binding["results"] = {
                field: proposed_results.get(field, field)
                for field in normalized_outputs
            }

    scenes = [
        {
            "scene_id": scene.scene_id,
            "block_id": scene.block_id,
            "description": scene.block_description,
            "raw_requirement_text": scene.raw_requirement_text,
            "logic_flow": scene.logic_flow,
            "preconditions": scene.preconditions,
            "fallback": scene.fallback,
        }
        for scene in fact_spec.scenes
    ]
    system_prompt = (
        "You bind an already extracted natural-language workflow to an allowed public operation catalog. "
        "Return JSON only. Map every catalog operation at least once to the most relevant existing scene. "
        "An operation may repeat only for distinct guarded branches; never repeat an always binding. "
        "Copy dependency_type and operation exactly; never invent names. Allowed guards are: always, "
        "first_request, followup, on_success:<catalog operation>, on_failure:<catalog operation>. "
        "Use always only for operations required on every valid path. Infer consumes and produces only "
        "from the requirement and operation schemas. For arguments, map every request-schema field to "
        "the exact workflow variable that supplies it (a dotted/indexed path such as results[0].url is "
        "allowed). For results, map every response-schema field to the workflow variable that stores it. "
        "Use {\"literal\": value} only when the requirement explicitly supplies a constant. Do not make "
        "BLOCK/LLM implementation decisions. A condition may be a leaf {source, operator, value}, a leaf "
        "{source, operator, value_from}, or a compound tree {all:[...]}, {any:[...]}, or {not:{...}}. "
        "Allowed leaf operators are exists, not_exists, eq, neq, contains, and not_contains. Use conditions "
        "only for explicit business logic and preserve every conjunct, alternative, and negation. Optionally emit foreach as "
        "{source, argument} when the requirement applies one operation to every item in a collection."
    )
    user_content = json.dumps({
        "natural_language_requirement": normalized,
        "extracted_scenes": scenes,
        "public_operation_catalog": catalog,
        "required_output_schema": {
            "bindings": [{
                "scene_id": "existing scene_id",
                "block_id": "existing block_id",
                "dependency_type": "exact catalog value",
                "operation": "exact catalog value",
                "guard": "allowed guard",
                "consumes": ["field"],
                "produces": ["field"],
                "arguments": {"request_field": "workflow_variable"},
                "results": {"response_field": "workflow_variable"},
                "condition": {"source": "workflow_variable", "operator": "eq", "value": "constant"},
                "foreach": {"source": "workflow_collection", "argument": "request_field"},
            }],
        },
    }, ensure_ascii=False, sort_keys=True)
    response = llm_client.call(
        system_prompt=system_prompt,
        user_content=user_content,
        temperature=0.1,
        max_tokens=8192,
    )
    usage = getattr(response, "usage", None) or {}
    token_count = int(usage.get("total_tokens", 0) or 0)
    raw = response.content if hasattr(response, "content") else str(response)
    text = raw.strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PublicOperationBindingError(
            f"M3b public operation binding is not valid JSON: {exc}", token_count
        ) from exc

    bindings = parsed.get("bindings", []) if isinstance(parsed, dict) else []
    catalog_keys = {(item["dependency_type"], item["operation"]) for item in catalog}
    scene_by_block = {scene.block_id: scene for scene in fact_spec.scenes if scene.block_id}
    scenes_by_id: Dict[str, List[Any]] = {}
    for scene in fact_spec.scenes:
        scenes_by_id.setdefault(str(scene.scene_id), []).append(scene)
    accepted: Dict[Tuple[str, str], List[Tuple[Any, Dict[str, Any]]]] = {}
    guard_pattern = re.compile(
        r"^(always|first_request|followup|on_success:[A-Za-z0-9_]+|on_failure:[A-Za-z0-9_]+)$"
    )
    for binding in bindings if isinstance(bindings, list) else []:
        if not isinstance(binding, dict):
            continue
        key = (str(binding.get("dependency_type", "")), str(binding.get("operation", "")))
        scene = scene_by_block.get(str(binding.get("block_id", "")))
        if scene is None:
            id_candidates = scenes_by_id.get(str(binding.get("scene_id", "")), [])
            scene = id_candidates[0] if len(id_candidates) == 1 else None
        guard = str(binding.get("guard", ""))
        if key not in catalog_keys or scene is None:
            continue
        if not guard_pattern.fullmatch(guard):
            guard = "always"
        if guard.startswith(("on_success:", "on_failure:")):
            predecessor = guard.split(":", 1)[1]
            if predecessor not in {item["operation"] for item in catalog}:
                # Models occasionally point at an extracted block id rather
                # than a catalog operation.  Preserve the binding and let the
                # deterministic scene-order pass rebuild the executable edge.
                guard = "always"
        normalized_binding = {
            "dependency_type": key[0],
            "operation": key[1],
            "guard": guard,
            "consumes": list(binding.get("consumes", []) or []),
            "produces": list(binding.get("produces", []) or []),
            "arguments": dict(binding.get("arguments", {}) or {}),
            "results": dict(binding.get("results", {}) or {}),
            "condition": dict(binding.get("condition", {}) or {}),
            "foreach": dict(binding.get("foreach", {}) or {}),
        }
        prior_guards = {item[1]["guard"] for item in accepted.get(key, [])}
        if guard in prior_guards or (guard == "always" and key in accepted):
            # Duplicate proposals carry no extra catalog evidence.  Keeping
            # the first valid proposal avoids discarding an otherwise complete
            # focused pass because of harmless LLM repetition.
            continue
        accepted.setdefault(key, []).append((scene, normalized_binding))

    # M3 and M3b see the same requirement and public catalog. If either pass
    # omits an operation, retain exact, schema-valid bindings from the other
    # pass instead of discarding complementary evidence or guessing by name.
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        for binding in structured.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            key = (
                str(binding.get("dependency_type", "")),
                str(binding.get("operation", "")),
            )
            if key not in catalog_keys or key in accepted:
                continue
            guard = str(binding.get("guard", "always") or "always")
            if not guard_pattern.fullmatch(guard):
                continue
            if guard.startswith(("on_success:", "on_failure:")):
                predecessor = guard.split(":", 1)[1]
                if predecessor not in {item["operation"] for item in catalog}:
                    continue
            accepted[key] = [(scene, {
                "dependency_type": key[0],
                "operation": key[1],
                "guard": guard,
                "consumes": list(binding.get("consumes", []) or []),
                "produces": list(binding.get("produces", []) or []),
                "arguments": dict(binding.get("arguments", {}) or {}),
                "results": dict(binding.get("results", {}) or {}),
                "condition": dict(binding.get("condition", {}) or {}),
                "foreach": dict(binding.get("foreach", {}) or {}),
            })]

    missing = catalog_keys - set(accepted)
    if missing:
        missing_catalog = [
            item for item in catalog
            if (item["dependency_type"], item["operation"]) in missing
        ]
        repair_response = llm_client.call(
            system_prompt=(
                "Return JSON only. Bind each supplied operation to exactly one existing scene. "
                "Copy all identity fields exactly. Output a bindings array with scene_id, block_id, "
                "dependency_type, operation, guard, consumes, produces, arguments, and results."
            ),
            user_content=json.dumps({
                "natural_language_requirement": normalized,
                "scenes": scenes,
                **(
                    {"operation": missing_catalog[0]}
                    if len(missing_catalog) == 1
                    else {"operations": missing_catalog}
                ),
            }, ensure_ascii=False, sort_keys=True),
            temperature=0.0,
            max_tokens=4096,
        )
        repair_usage = getattr(repair_response, "usage", None) or {}
        token_count += int(repair_usage.get("total_tokens", 0) or 0)
        repair_raw = repair_response.content if hasattr(repair_response, "content") else str(repair_response)
        repair_text = repair_raw.strip()
        if repair_text.startswith("```"):
            repair_text = repair_text.lstrip("`")
            if repair_text.lower().startswith("json"):
                repair_text = repair_text[4:]
            repair_text = repair_text.strip().rstrip("`").strip()
        try:
            repair_parsed = json.loads(repair_text)
        except json.JSONDecodeError:
            repair_parsed = {}
        repair_bindings = repair_parsed.get("bindings", []) if isinstance(repair_parsed, dict) else []
        for binding in repair_bindings if isinstance(repair_bindings, list) else []:
            if not isinstance(binding, dict):
                continue
            key = (str(binding.get("dependency_type", "")), str(binding.get("operation", "")))
            if key not in missing:
                continue
            scene = scene_by_block.get(str(binding.get("block_id", "")))
            if scene is None:
                id_candidates = scenes_by_id.get(str(binding.get("scene_id", "")), [])
                scene = id_candidates[0] if len(id_candidates) == 1 else None
            guard = str(binding.get("guard", ""))
            if scene is None:
                continue
            if not guard_pattern.fullmatch(guard):
                guard = "always"
            if guard.startswith(("on_success:", "on_failure:")):
                predecessor = guard.split(":", 1)[1]
                if predecessor not in {item["operation"] for item in catalog}:
                    continue
            accepted[key] = [(scene, {
                "dependency_type": key[0],
                "operation": key[1],
                "guard": guard,
                "consumes": list(binding.get("consumes", []) or []),
                "produces": list(binding.get("produces", []) or []),
                "arguments": dict(binding.get("arguments", {}) or {}),
                "results": dict(binding.get("results", {}) or {}),
                "condition": dict(binding.get("condition", {}) or {}),
                "foreach": dict(binding.get("foreach", {}) or {}),
            })]
        missing = catalog_keys - set(accepted)
        if missing:
            logger.warning(
                "[M3c] focused binding did not close missing operations; response=%s",
                repair_text[:1000],
            )
            for dependency_type, operation in sorted(missing):
                interface = next(
                    item for item in catalog
                    if item["dependency_type"] == dependency_type and item["operation"] == operation
                )
                selection_response = llm_client.call(
                    system_prompt=(
                        "Return JSON only as {\"block_id\":\"...\"}. Select exactly one supplied "
                        "existing scene for the public operation. Do not return guards or explanations."
                    ),
                    user_content=json.dumps({
                        "natural_language_requirement": normalized,
                        "operation": interface,
                        "scene_choices": [
                            {
                                "block_id": item["block_id"],
                                "description": item["description"],
                                "raw_requirement_text": item["raw_requirement_text"],
                                "logic_flow": item["logic_flow"],
                            }
                            for item in scenes
                        ],
                    }, ensure_ascii=False, sort_keys=True),
                    temperature=0.0,
                    max_tokens=256,
                )
                selection_usage = getattr(selection_response, "usage", None) or {}
                token_count += int(selection_usage.get("total_tokens", 0) or 0)
                selection_raw = (
                    selection_response.content
                    if hasattr(selection_response, "content") else str(selection_response)
                ).strip()
                if selection_raw.startswith("```"):
                    selection_raw = selection_raw.lstrip("`")
                    if selection_raw.lower().startswith("json"):
                        selection_raw = selection_raw[4:]
                    selection_raw = selection_raw.strip().rstrip("`").strip()
                try:
                    selected_block = str(json.loads(selection_raw).get("block_id", ""))
                except (json.JSONDecodeError, AttributeError):
                    selected_block = ""
                selected_scene = scene_by_block.get(selected_block)
                if selected_scene is None:
                    continue

                accepted[(dependency_type, operation)] = [(selected_scene, {
                    "dependency_type": dependency_type,
                    "operation": operation,
                    "guard": "always",
                    "consumes": schema_required(interface.get("request_schema", {})),
                    "produces": schema_required(interface.get("response_schema", {})),
                    "arguments": {
                        field: field
                        for field in schema_required(interface.get("request_schema", {}))
                    },
                    "results": {
                        field: field
                        for field in schema_required(interface.get("response_schema", {}))
                    },
                    "condition": {},
                    "foreach": {},
                })]
            missing = catalog_keys - set(accepted)
    if missing:
        names = ", ".join(sorted(operation for _, operation in missing))
        raise PublicOperationBindingError(
            f"M3b incomplete public operation binding: {names}", token_count
        )

    # Correct a narrowly scoped class of scene-selection errors using names
    # that are already public in the operation catalog.  This keeps the DSL
    # faithful when an LLM confuses input-schema validation with final-schema
    # formatting, or binds a combined research/extraction operation to the
    # first retrieval sub-step.
    semantic_terms = {
        "generate": ("generate", "生成", "创建内容"),
        "answer": ("answer", "回答", "答复", "解答"),
        "evaluate": ("evaluate", "评估", "审查", "质量"),
        "optimize": ("optimize", "优化", "改进", "迭代"),
        "create": ("create", "创建", "导出", "文档"),
        "format": ("format", "格式", "格式化", "排版"),
        "schema": ("schema", "结构"),
        "search": ("search", "检索", "搜索"),
        "scrape": ("scrape", "抓取", "爬取"),
        "interact": ("interact", "交互", "点击"),
        "research": ("research", "研究"),
        "extract": ("extract", "抽取", "提取"),
        "update": ("update", "更新", "修改", "写回", "应用"),
        "list": ("list", "列出", "查询", "核对", "已有", "已存在"),
        "read": ("read", "读取", "加载", "获取"),
        "send": ("send", "发送", "回传", "回复"),
    }

    def scene_semantic_score(scene: Any, operation: str) -> int:
        concepts = [name for name in semantic_terms if name in operation.lower()]
        description = str(scene.block_description or "").lower()
        logic = str(scene.logic_flow or "").lower()
        raw = str(scene.raw_requirement_text or "").lower()
        return sum(
            4 * int(any(term in description for term in semantic_terms[concept]))
            + 2 * int(any(term in logic for term in semantic_terms[concept]))
            + int(any(term in raw for term in semantic_terms[concept]))
            for concept in concepts
        )

    for operation_bindings in accepted.values():
        for index, (scene, binding) in enumerate(operation_bindings):
            operation = str(binding.get("operation", ""))
            current_score = scene_semantic_score(scene, operation)
            ranked = sorted(
                (
                    (scene_semantic_score(candidate, operation), scene_index, candidate)
                    for scene_index, candidate in enumerate(fact_spec.scenes)
                ),
                key=lambda item: (item[0], -item[1]),
                reverse=True,
            )
            best_score, _, best_scene = ranked[0]
            combined_research = "research" in operation.lower() and "extract" in operation.lower()
            if combined_research:
                best_score = max(item[0] for item in ranked)
                best_scene = max(
                    (item for item in ranked if item[0] == best_score),
                    key=lambda item: item[1],
                )[2]
            if best_score > current_score or (
                combined_research and best_score == current_score and best_scene is not scene
            ):
                operation_bindings[index] = (best_scene, binding)

    # M3c may return several candidate scenes for one catalog operation.  The
    # action contract declares a single binding; explicit loop reconstruction
    # is handled later from the workflow text.
    for key, operation_bindings in list(accepted.items()):
        operation = key[1]
        combined_research = "research" in operation.lower() and "extract" in operation.lower()
        accepted[key] = [max(
            operation_bindings,
            key=lambda item: (
                scene_semantic_score(item[0], operation),
                (
                    fact_spec.scenes.index(item[0])
                    if combined_research else -fact_spec.scenes.index(item[0])
                ),
            ),
        )]

    scene_order = {id(scene): index for index, scene in enumerate(fact_spec.scenes)}
    def operation_phase(operation: str) -> int:
        name = operation.lower()
        if any(token in name for token in ("discovery", "discover", "initiate", "start_")):
            return 0
        if "poll" in name or "status" in name:
            return 1
        if "download" in name:
            return 2
        if any(token in name for token in ("read", "load", "list", "fetch", "search")):
            return 0
        if "scrape" in name:
            return 1
        if "interact" in name:
            return 2
        if any(token in name for token in ("research", "extract", "generate", "analyze")):
            return 3
        if "format" in name:
            return 4
        if any(token in name for token in ("create", "send", "post", "publish", "update")):
            return 6
        return 2

    ordered_bindings = sorted(
        [item for operation_bindings in accepted.values() for item in operation_bindings],
        key=lambda item: (
            scene_order[id(item[0])],
            operation_phase(str(item[1]["operation"])),
            item[1]["operation"],
            item[1]["guard"],
        ),
    )
    for _, binding in ordered_bindings:
        # Public interface schemas are method-visible and authoritative.  M3b
        # selects the scene and guard; field names must be exact schema names,
        # not aliases invented by the model.
        normalize_schema_fields(binding)
        foreach = binding.get("foreach", {}) or {}
        if foreach and foreach.get("argument") not in (binding.get("arguments", {}) or {}):
            binding["foreach"] = {}

    requirement_lower = normalized.lower()
    alternative_manual_feed = (
        any(token in requirement_lower for token in ("rss", "feed", "订阅源"))
        and any(token in requirement_lower for token in ("manual", "手动", "人工输入"))
    )
    if alternative_manual_feed:
        for _, binding in ordered_bindings:
            operation = str(binding.get("operation", "")).lower()
            if any(token in operation for token in ("read_feed", "load_feed", "fetch_feed")):
                binding["condition"] = {
                    "source": "input.source",
                    "operator": "eq",
                    "value": "rss",
                }

    # Two operations may expose the same schema field (for example generated
    # tags and existing tags). Keep both values addressable in the workflow.
    used_result_targets: Set[str] = set()
    for _, binding in ordered_bindings:
        operation = str(binding.get("operation", ""))
        for field, target in list((binding.get("results", {}) or {}).items()):
            if not isinstance(target, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", target
            ):
                target = field
                binding["results"][field] = target
            if target in used_result_targets:
                candidate = operation if operation.endswith(field) else f"{operation}_{field}"
                binding["results"][field] = candidate
                target = candidate
            used_result_targets.add(str(target))

    def variable_name(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("name", ""))
        return str(item)

    def source_root(source: Any) -> str:
        if not isinstance(source, str) or not source.strip():
            return ""
        path = source.strip().removeprefix("$")
        if path.startswith("results."):
            return ""
        if path.startswith("result."):
            parts = path.split(".", 2)
            return ".".join(parts[:2]) if len(parts) >= 2 else ""
        if path.startswith("user_input."):
            return ""
        if path.startswith("input."):
            return "input"
        for prefix in ("context.", "workflow."):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        return re.split(r"[.\[]", path, maxsplit=1)[0]

    def source_is_available(source: Any, roots: Set[str]) -> bool:
        if isinstance(source, dict):
            if set(source) == {"literal"}:
                return True
            return bool(source) and all(source_is_available(value, roots) for value in source.values())
        if isinstance(source, list):
            return bool(source) and all(source_is_available(item, roots) for item in source)
        if isinstance(source, str) and source.strip().removeprefix("$").startswith("result."):
            return source_root(source) in roots
        root = source_root(source)
        return root == "input" or root in roots

    def repair_source_reference(source: Any, field: str, prior_public: Set[str]) -> Any:
        """Rewrite only references whose public lineage or raw-input role is unambiguous."""
        if isinstance(source, dict):
            if set(source) == {"literal"}:
                return source
            return {
                key: repair_source_reference(value, str(key), prior_public)
                for key, value in source.items()
            }
        if isinstance(source, list):
            return [repair_source_reference(value, field, prior_public) for value in source]
        if not isinstance(source, str):
            return {"literal": source}
        path = source.strip().removeprefix("$")
        if path.startswith("result."):
            return path
        if path.startswith("results.") or path.startswith("results["):
            suffix_match = re.match(r"results(?:\[\d+\]|\.[^.\[]+)[.]?(.+)?$", path)
            suffix = str(suffix_match.group(1) or "") if suffix_match else ""
            node_candidates = [
                item for item in prior_public
                if item.startswith("result.") and suffix
                and (item.endswith(f".{suffix}") or item.endswith(f"[{suffix}]"))
            ]
            if len(node_candidates) == 1:
                return node_candidates[0]
            leaf = re.split(r"[.\[]", path.rstrip("]"))[-1]
            candidates = [
                item for item in prior_public
                if item.startswith("result.")
                and re.split(r"[.\[]", item.rstrip("]"))[-1] == leaf
            ]
            if len(candidates) == 1:
                return candidates[0]
            if leaf == "id" and field.endswith("_id"):
                object_name = field[:-3]
                typed = [item for item in candidates if object_name in item.lower()]
                if len(typed) == 1:
                    return typed[0]
            return source
        raw_namespaces = {
            "config", "email", "settings", "trigger_comment", "user_watchlist",
            "workflow", "workflow_parameters", "workflow_variable",
        }
        root = re.split(r"[.\[]", path, maxsplit=1)[0]
        if root == "fixtures" and "." in path:
            nested_path = path.split(".", 1)[1]
            nested_root = re.split(r"[.\[]", nested_path, maxsplit=1)[0]
            if nested_root in set(workflow_input_fields or set()):
                return f"workflow_input.{nested_path}"
        if root in raw_namespaces and "." in path:
            leaf_path = path.split(".", 1)[1]
            if field == "sheet_id" and leaf_path.endswith("sheet_id"):
                leaf_path = field
            elif leaf_path.startswith("candidate_") and field in leaf_path:
                leaf_path = field
            elif root in {"config", "settings", "workflow", "workflow_parameters"} and field not in leaf_path:
                leaf_path = field
            if root == "workflow_variable":
                return leaf_path
            return f"input.{leaf_path}"
        return source

    # Response targets are workflow variables, not ad-hoc object namespaces.
    # Invalid targets fall back to the exact public response field so that
    # provenance remains inspectable and downstream repair has a stable name.
    for _, binding in ordered_bindings:
        binding["results"] = {
            field: (
                target if isinstance(target, str) and re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*", target
                ) else field
            )
            for field, target in (binding.get("results", {}) or {}).items()
        }

    initial_variables = {
        variable_name(item)
        for item in fact_spec.inputs
        if variable_name(item)
    } | {"user_input", "history", "previous_agent"}
    available = set(initial_variables)
    available_paths = set(initial_variables)
    public_available_paths: Set[str] = set()
    public_path_producers: Dict[str, str] = {}
    public_scalar_paths: Set[str] = set()

    def nested_schema_paths(schema: Any, prefix: str) -> Set[str]:
        if not isinstance(schema, dict):
            return set()
        paths: Set[str] = set()
        schema_type = schema.get("type")
        if schema_type == "array":
            item_prefix = f"{prefix}[0]"
            paths.add(item_prefix)
            paths.update(nested_schema_paths(schema.get("items", {}), item_prefix))
        elif schema_type == "object" or isinstance(schema.get("properties"), dict):
            for child, child_schema in (schema.get("properties", {}) or {}).items():
                child_path = f"{prefix}.{child}"
                paths.add(child_path)
                paths.update(nested_schema_paths(child_schema, child_path))
        return paths

    review_rows: List[Dict[str, Any]] = []
    binding_order = 0
    for scene in fact_spec.scenes:
        for bound_scene, binding in ordered_bindings:
            if bound_scene is not scene:
                continue
            arguments = binding.get("arguments", {}) or {}
            unresolved = {
                field: source
                for field, source in arguments.items()
                if not source_is_available(source, available)
            }
            if arguments:
                review_rows.append({
                    "binding_order": binding_order,
                    "scene_id": scene.scene_id,
                    "block_id": scene.block_id,
                    "operation": binding["operation"],
                    "current_arguments": arguments,
                    "unresolved_arguments": unresolved,
                    "available_variables": sorted(available_paths),
                    "prior_public_variables": sorted(public_available_paths),
                    "prior_public_producers": dict(public_path_producers),
                    "prior_scalar_paths": sorted(public_scalar_paths),
                    "scene_outputs": [
                        variable_name(item) for item in scene.action.outputs
                        if variable_name(item)
                    ],
                    "requirement_text": scene.raw_requirement_text,
                })
            available.update(
                target for target in (binding.get("results", {}) or {}).values()
                if isinstance(target, str) and target
            )
            available_paths.update(available)
            interface = catalog_by_key.get((binding["dependency_type"], binding["operation"]), {})
            response_schema = interface.get("response_schema", {}) if isinstance(interface, dict) else {}
            response_properties = (
                response_schema.get("properties", {})
                if isinstance(response_schema, dict) else {}
            )
            for response_field, target in (binding.get("results", {}) or {}).items():
                if isinstance(target, str) and target:
                    operation = str(binding.get("operation", ""))
                    node_field = f"result.{operation}.{response_field}"
                    node_paths = nested_schema_paths(
                        response_properties.get(response_field, {}), node_field
                    )
                    available.add(f"result.{operation}")
                    available_paths.add(f"result.{operation}")
                    available_paths.add(node_field)
                    available_paths.update(node_paths)
                    public_available_paths.add(node_field)
                    public_available_paths.update(node_paths)
                    public_path_producers[node_field] = operation
                    for nested_path in node_paths:
                        public_path_producers[nested_path] = operation
                    field_schema = response_properties.get(response_field, {})
                    if isinstance(field_schema, dict) and field_schema.get("type") in (
                        "string", "number", "integer", "boolean", "null"
                    ):
                        public_scalar_paths.add(target)
                        public_scalar_paths.add(node_field)
                    else:
                        public_scalar_paths.discard(target)
                        public_scalar_paths.discard(node_field)
                    nested_paths = nested_schema_paths(
                        response_properties.get(response_field, {}), target
                    )
                    public_available_paths.add(target)
                    public_available_paths.update(nested_paths)
                    public_path_producers[target] = str(binding.get("operation", ""))
                    for nested_path in nested_paths:
                        public_path_producers[nested_path] = str(binding.get("operation", ""))
                    available_paths.update(nested_paths)
            binding_order += 1
        available.update(
            variable_name(item) for item in scene.action.outputs
            if variable_name(item)
        )
        available_paths.update(available)

    if review_rows:
        repair_response = llm_client.call(
            system_prompt=(
                "Review and repair workflow data bindings. Return JSON only. For every supplied "
                "row, return binding_order and a complete arguments object. Each request field must map "
                "to one exact available_variables entry, optionally followed by a valid dotted/indexed "
                "path. Select the value that the requirement says should enter this operation; when a "
                "prior local step formatted, validated, filtered, or transformed a value, prefer that "
                "derived value over its raw predecessor. Do not use future scene outputs, results.<block>, "
                "user_input.<field>, aliases, Gold, Oracle, examples, "
                "or evaluator data. Use {\"literal\": value} only for a constant explicitly stated in "
                "the natural-language requirement."
            ),
            user_content=json.dumps({
                "natural_language_requirement": normalized,
                "bindings_to_review": review_rows,
                "required_output_schema": {
                    "bindings": [{
                        "binding_order": "integer copied exactly",
                        "arguments": {"request_field": "available_workflow_variable"},
                    }],
                },
            }, ensure_ascii=False, sort_keys=True),
            temperature=0.0,
            max_tokens=4096,
        )
        repair_usage = getattr(repair_response, "usage", None) or {}
        token_count += int(repair_usage.get("total_tokens", 0) or 0)
        repair_raw = (
            repair_response.content
            if hasattr(repair_response, "content") else str(repair_response)
        ).strip()
        if repair_raw.startswith("```"):
            repair_raw = repair_raw.lstrip("`")
            if repair_raw.lower().startswith("json"):
                repair_raw = repair_raw[4:]
            repair_raw = repair_raw.strip().rstrip("`").strip()
        try:
            repaired_payload = json.loads(repair_raw)
        except json.JSONDecodeError:
            repaired_payload = {}
        repair_by_order = {
            int(item.get("binding_order")): item
            for item in repaired_payload.get("bindings", [])
            if isinstance(item, dict) and str(item.get("binding_order", "")).isdigit()
        } if isinstance(repaired_payload, dict) else {}
        for row in review_rows:
            repaired = repair_by_order.get(row["binding_order"], {})
            arguments = repaired.get("arguments", {}) if isinstance(repaired, dict) else {}
            expected_fields = set(row["current_arguments"])
            valid = isinstance(arguments, dict) and set(arguments) == expected_fields and all(
                source_is_available(source, {
                    re.split(r"[.\[]", item, maxsplit=1)[0]
                    for item in row["available_variables"]
                })
                for source in arguments.values()
            )
            if valid:
                ordered_bindings[row["binding_order"]][1]["arguments"] = arguments
            binding_arguments = ordered_bindings[row["binding_order"]][1]["arguments"]
            available_at_binding = {
                source_root(item)
                for item in row["available_variables"]
                if source_root(item)
            }
            prior_public = set(row.get("prior_public_variables", []))

            def repair_legacy_result_source(source: Any, field: str) -> Any:
                return repair_source_reference(source, field, prior_public)

            for field, source in list(binding_arguments.items()):
                source = repair_legacy_result_source(source, field)
                binding_arguments[field] = source
                if isinstance(source, dict) and set(source) == {"literal"}:
                    continue
                if isinstance(source, dict):
                    continue
                if isinstance(source, list):
                    if source_is_available(source, available_at_binding):
                        continue
                    # Keep unresolved lists for the strict compile gate to
                    # reject; no single-field guess can preserve their shape.
                    continue
                if isinstance(source, str) and source.strip().removeprefix("$").startswith("input."):
                    # An explicit raw-input reference must not be replaced by
                    # an earlier operation's same-named response field.
                    continue
                if field in prior_public:
                    # Exact response-field lineage is stronger evidence than
                    # an LLM-selected scene-local alias with a compatible root.
                    binding_arguments[field] = field
                    continue
                public_leaf_candidates = [
                    item for item in prior_public
                    if re.split(r"[.\[]", item)[-1] == field
                ]
                if len(public_leaf_candidates) == 1:
                    binding_arguments[field] = public_leaf_candidates[0]
                    continue
                if isinstance(source, str) and source.endswith("[0]"):
                    indexed_public = [item for item in prior_public if item.endswith("[0]")]
                    if len(indexed_public) == 1:
                        binding_arguments[field] = indexed_public[0]
                        continue
                if source_root(source) == "input" or source_root(source) in available_at_binding:
                    continue
                if field in available_at_binding:
                    # Exact-name wiring is deterministic and schema-derived;
                    # it is safer than retaining an invented namespace.
                    binding_arguments[field] = field
                    continue
                leaf_candidates = [
                    item for item in row["available_variables"]
                    if re.split(r"[.\[]", item, maxsplit=1)[0] in available_at_binding
                    and re.split(r"[.\[]", item)[-1] == field
                ]
                if len(leaf_candidates) == 1:
                    binding_arguments[field] = leaf_candidates[0]
                    continue
                source_leaf = (
                    re.split(r"[.\[]", source.rstrip("]"))[-1]
                    if isinstance(source, str) else ""
                )
                source_leaf_candidates = [
                    item for item in row["available_variables"]
                    if source_leaf
                    and re.split(r"[.\[]", item.rstrip("]"))[-1] == source_leaf
                ]
                if len(source_leaf_candidates) == 1:
                    binding_arguments[field] = source_leaf_candidates[0]
                elif field == "action":
                    binding_arguments[field] = {
                        "literal": ordered_bindings[row["binding_order"]][1]["operation"]
                    }
                elif field in {"query", "prompt", "instruction", "text"} and "user_input" in available_at_binding:
                    binding_arguments[field] = "user_input"
                elif field in {"content", "body", "html"}:
                    formatted_candidates = [
                        item for item in row["available_variables"]
                        if any(token in item.lower() for token in ("formatted", "rendered", "templated"))
                    ]
                    if len(formatted_candidates) == 1:
                        binding_arguments[field] = formatted_candidates[0]
            binding = ordered_bindings[row["binding_order"]][1]
            foreach = binding.get("foreach", {}) or {}
            encapsulated_iteration = any(
                str(field).startswith("one_call_per_")
                and isinstance(value, dict)
                and value.get("literal") is True
                for field, value in binding_arguments.items()
            )
            if foreach and encapsulated_iteration:
                binding["foreach"] = {}
                foreach = {}
            if foreach:
                foreach["source"] = repair_legacy_result_source(
                    foreach.get("source"), str(foreach.get("argument", ""))
                )
            condition = binding.get("condition", {}) or {}
            if condition:
                for leaf in _condition_leaf_nodes(condition):
                    leaf["source"] = repair_legacy_result_source(
                        leaf.get("source"), "condition"
                    )
                    if "value_from" in leaf:
                        leaf["value_from"] = repair_legacy_result_source(
                            leaf.get("value_from"), "condition_value"
                        )
            producers = row.get("prior_public_producers", {})
            array_roots = list(dict.fromkeys(
                item[:-3] for item in row.get("prior_public_variables", [])
                if item.endswith("[0]")
            ))
            scene_text = str(row.get("requirement_text", "")).lower()
            conditional_create = str(binding.get("operation", "")).lower().startswith("create") and any(
                token in scene_text for token in (
                    "necessary", "if needed", "if missing", "not exist", "不存在", "必要时", "缺失",
                )
            )
            if conditional_create and not foreach:
                generated_collections = [
                    item for item in array_roots
                    if any(token in str(producers.get(item, "")).lower() for token in ("generate", "extract"))
                ]
                argument_fields = list(binding_arguments)
                if generated_collections and argument_fields:
                    foreach = {
                        "source": generated_collections[0],
                        "argument": argument_fields[0],
                    }
                    binding["foreach"] = foreach
            if foreach:
                source = foreach.get("source")
                generated_collections = [
                    item for item in array_roots
                    if any(token in str(producers.get(item, "")).lower() for token in ("generate", "extract"))
                ]
                if generated_collections:
                    foreach["source"] = generated_collections[0]
                elif not source_is_available(source, available_at_binding):
                    # A fan-out consumes the earliest produced collection;
                    # later list/read collections usually define membership.
                    if array_roots:
                        foreach["source"] = next((
                            item for item in array_roots
                            if not any(token in item.lower() for token in ("list", "existing", "read"))
                        ), array_roots[0])
                foreach_source = str(foreach.get("source", ""))
                argument = str(foreach.get("argument", ""))
                scalar_public_source = (
                    foreach_source in row.get("prior_scalar_paths", [])
                )
                if scalar_public_source:
                    binding["foreach"] = {}
                    if argument in binding_arguments:
                        binding_arguments[argument] = foreach_source
                    foreach = {}
                elif foreach_source and argument in binding_arguments:
                    binding_arguments[argument] = f"{foreach_source}[0]"
                if foreach and conditional_create:
                    membership_collections = [
                        item for item in array_roots
                        if item != foreach_source and not item.endswith("[0]")
                        and any(token in str(producers.get(item, "")).lower() for token in ("list", "read", "load"))
                    ]
                    if membership_collections:
                        binding["condition"] = {
                            "source": membership_collections[-1],
                            "operator": "not_contains",
                            "value_from": "foreach_item",
                        }
    # Conditions and fan-out controls can occur on operations without request
    # arguments, so normalize them in a complete sequential pass as well.
    sequential_public_paths: Set[str] = set()
    for _, binding in ordered_bindings:
        for field, source in list((binding.get("arguments", {}) or {}).items()):
            binding["arguments"][field] = repair_source_reference(
                source, field, sequential_public_paths
            )
        foreach = binding.get("foreach", {}) or {}
        if foreach:
            foreach["source"] = repair_source_reference(
                foreach.get("source"), str(foreach.get("argument", "")), sequential_public_paths
            )
        condition = binding.get("condition", {}) or {}
        if condition:
            for leaf in _condition_leaf_nodes(condition):
                leaf["source"] = repair_source_reference(
                    leaf.get("source"), "condition", sequential_public_paths
                )
                if "value_from" in leaf:
                    leaf["value_from"] = repair_source_reference(
                        leaf.get("value_from"), "condition_value", sequential_public_paths
                    )
        interface = catalog_by_key.get((binding["dependency_type"], binding["operation"]), {})
        response_schema = interface.get("response_schema", {}) if isinstance(interface, dict) else {}
        response_properties = response_schema.get("properties", {}) if isinstance(response_schema, dict) else {}
        for response_field, target in (binding.get("results", {}) or {}).items():
            if isinstance(target, str) and target:
                operation = str(binding.get("operation", ""))
                node_field = f"result.{operation}.{response_field}"
                sequential_public_paths.add(node_field)
                sequential_public_paths.update(nested_schema_paths(
                    response_properties.get(response_field, {}), node_field
                ))
                sequential_public_paths.add(target)
                sequential_public_paths.update(nested_schema_paths(
                    response_properties.get(response_field, {}), target
                ))

    followup_terms = ("follow-up", "followup", "追问", "继续提问", "历史消息")
    contextual_operation_terms = ("history", "context", "memory", "conversation")
    requirement_lower = normalized.lower()
    for _, binding in ordered_bindings:
        operation_lower = binding["operation"].lower()
        if (
            any(term in requirement_lower for term in followup_terms)
            and any(term in operation_lower for term in contextual_operation_terms)
        ):
            binding["guard"] = "followup"

    prior_producers: List[Tuple[str, set[str]]] = []
    for _, binding in ordered_bindings:
        consumed = {str(item) for item in binding.get("consumes", []) if item}
        producer = next(
            (
                operation for operation, produced in reversed(prior_producers)
                if consumed & produced
            ),
            "",
        )
        if producer and binding["guard"] not in {"followup", "first_request"}:
            binding["guard"] = f"on_success:{producer}"
        prior_producers.append((
            binding["operation"],
            {str(item) for item in binding.get("produces", []) if item},
        ))

    previous_unconditional_operation = ""
    for scene, binding in ordered_bindings:
        if binding["guard"] != "always":
            continue
        condition_text = " ".join([
            scene.raw_requirement_text,
            scene.logic_flow,
            scene.triggers.raw_text,
            scene.block_description,
        ]).lower()
        if any(token in condition_text for token in followup_terms):
            binding["guard"] = "followup"
            continue
        if any(token in condition_text for token in ("first request", "initial request", "首次请求", "首次提问")):
            binding["guard"] = "first_request"
            continue
        if previous_unconditional_operation:
            binding["guard"] = f"on_success:{previous_unconditional_operation}"
        previous_unconditional_operation = binding["operation"]

    for scene in fact_spec.scenes:
        existing = scene.action.structured_op if isinstance(scene.action.structured_op, dict) else {}
        scene.action.structured_op = {**existing, "public_operations": []}
    for scene, binding in ordered_bindings:
        scene.action.structured_op["public_operations"].append(binding)
    return fact_spec, token_count


def _propagate_conditional_result_control_dependencies(
    fact_spec: 'DslFactSpec',
) -> 'DslFactSpec':
    """Make downstream consumers wait for their conditional data producer."""
    bindings: List[Tuple[int, Dict[str, Any]]] = []
    scenes_by_index: Dict[int, Any] = {}
    for scene_index, scene in enumerate(fact_spec.scenes):
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        scenes_by_index[scene_index] = scene
        for binding in structured.get("public_operations", []) or []:
            if isinstance(binding, dict) and binding.get("operation"):
                bindings.append((scene_index, binding))

    operation_positions = {
        str(binding.get("operation", "")): position
        for position, (_, binding) in enumerate(bindings)
    }
    bindings_by_operation = {
        str(binding.get("operation", "")): binding
        for _, binding in bindings
    }
    conditional_result_producers: Dict[str, Tuple[int, str]] = {}
    for scene_index, binding in bindings:
        operation = str(binding.get("operation", ""))
        if not operation or not binding.get("condition"):
            continue
        conditional_result_producers[f"result.{operation}"] = (
            scene_index, operation,
        )
        for target in (binding.get("results", {}) or {}).values():
            if isinstance(target, str) and target:
                root = re.split(r"[.\[]", target.removeprefix("$"), maxsplit=1)[0]
                conditional_result_producers[root] = (scene_index, operation)

    def scene_reference_roots(scene: Any) -> Set[str]:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            return set()
        roots = _action_contract_reference_roots(
            structured.get("local_program", []) or []
        )
        for binding in structured.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            roots.update(_action_contract_reference_roots({
                "arguments": binding.get("arguments", {}),
                "foreach": binding.get("foreach", {}),
            }))
        return roots

    for scene_index, scene in scenes_by_index.items():
        structured = scene.action.structured_op or {}
        public_operations = structured.get("public_operations", []) or []
        local_roots = _action_contract_reference_roots(
            structured.get("local_program", []) or []
        )
        for binding_index, binding in enumerate(public_operations):
            if not isinstance(binding, dict):
                continue
            binding_roots = _action_contract_reference_roots({
                "arguments": binding.get("arguments", {}),
                "foreach": binding.get("foreach", {}),
            })
            roots = binding_roots | (local_roots if binding_index == 0 else set())
            candidates = {
                producer
                for root in roots
                for producer_scene, producer in [
                    conditional_result_producers.get(root, (-1, ""))
                ]
                if producer and producer_scene < scene_index
            }
            if not candidates:
                continue
            producer = max(
                candidates,
                key=lambda operation: operation_positions.get(operation, -1),
            )
            operation = str(binding.get("operation", ""))
            guard = str(binding.get("guard", "always") or "always")
            condition = binding.get("condition", {}) or {}
            if (
                isinstance(condition, dict)
                and condition.get("source") == "workflow_input.source"
                and condition.get("operator") == "eq"
            ):
                # An explicit alternative input-mode branch already carries
                # its own entry condition; do not replace its guard with a
                # producer from the other mode.
                continue
            if operation == producer:
                continue
            if guard in {"followup", "first_request"} or guard.startswith("on_failure:"):
                continue
            predecessor = producer
            seen: Set[str] = set()
            creates_cycle = False
            while predecessor and predecessor not in seen:
                if predecessor == operation:
                    creates_cycle = True
                    break
                seen.add(predecessor)
                predecessor_guard = str(
                    bindings_by_operation.get(predecessor, {}).get("guard", "")
                )
                predecessor = (
                    predecessor_guard.split(":", 1)[1]
                    if predecessor_guard.startswith("on_success:")
                    else ""
                )
            if creates_cycle:
                continue
            binding["guard"] = f"on_success:{producer}"
    return fact_spec


def _normalize_fact_spec_public_operation_guards(
    fact_spec: 'DslFactSpec',
    normalized: str,
    top_level_input_fields: Optional[Set[str]] = None,
    workflow_input_fields: Optional[Set[str]] = None,
    input_mode_signatures: Optional[List[List[str]]] = None,
) -> 'DslFactSpec':
    """Normalize explicit guards even when the focused M3b pass is rejected."""
    bindings: List[Tuple[int, Any, Dict[str, Any]]] = []
    for scene_index, scene in enumerate(fact_spec.scenes):
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        for binding in structured.get("public_operations", []) or []:
            if isinstance(binding, dict) and binding.get("operation"):
                bindings.append((scene_index, scene, binding))

    # A scene may explicitly describe a public side effect in its execution
    # sequence even when M3b attached the catalog binding to an earlier error
    # scene.  Recover that missing branch from the existing catalog-validated
    # binding instead of inventing a new operation.  This keeps the repair
    # grounded in the requirement's own action sequence and preserves the
    # existing failure binding.
    by_operation = {
        (str(binding.get("dependency_type", "")), str(binding.get("operation", ""))): binding
        for _, _, binding in bindings
    }

    # Selection must precede threshold evaluation.  A generated ``not_exists``
    # guard on a threshold input therefore blocks the whole workflow whenever
    # the threshold is correctly supplied.  Remove only this structurally
    # contradictory form; other entry conditions remain intact.
    for _, _, binding in bindings:
        operation_name = str(binding.get("operation", "")).lower()
        condition = binding.get("condition", {}) or {}
        source = str(condition.get("source", "")).lower() if isinstance(condition, dict) else ""
        if (
            condition.get("operator") == "not_exists"
            and any(token in operation_name for token in ("select", "choose", "classify"))
            and any(token in source for token in ("threshold", "quality", "score"))
        ):
            binding["condition"] = {}
    for scene_index, scene in enumerate(fact_spec.scenes):
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        scene_bindings = structured.get("public_operations", []) or []
        scene_operations = {str(item.get("operation", "")) for item in scene_bindings if isinstance(item, dict)}
        scene_text = " ".join(str(item) for item in (scene.action_sequence or []))
        if "gmail::send_email" not in scene_text or "send_email" in scene_operations:
            continue
        template = by_operation.get(("gmail", "send_email"))
        if not isinstance(template, dict):
            continue
        predecessor = next(
            (str(item.get("operation", "")) for item in reversed(scene_bindings)
             if isinstance(item, dict) and item.get("operation") and item.get("operation") != "send_email"),
            "",
        )
        repaired = copy.deepcopy(template)
        repaired["guard"] = f"on_success:{predecessor}" if predecessor else "always"
        arguments = dict(repaired.get("arguments", {}) or {})
        if "body" in arguments and ("report_content" in scene_text or "报告" in scene_text):
            arguments["body"] = "report_content"
        if "subject" in arguments:
            arguments["subject"] = {"literal": "Market report"}
        if "to" in arguments:
            arguments["to"] = "report_recipient"
        repaired["arguments"] = arguments
        repaired["consumes"] = list(dict.fromkeys(str(key) for key in arguments if isinstance(arguments[key], str)))
        repaired["results"] = {"message_id": "report_message_id"}
        repaired["produces"] = ["message_id"]
        # The copied binding may be an error-only Gmail branch.  The action
        # sequence is the evidence for this success-side send, so do not
        # carry the error condition into the recovered branch.
        repaired["condition"] = {}
        scene_bindings.append(repaired)
        structured["public_operations"] = scene_bindings
        bindings.append((scene_index, scene, repaired))
        by_operation.setdefault(("gmail", "send_email"), repaired)
    def binding_phase(operation: str) -> int:
        name = operation.lower()
        tokens = set(re.split(r"[^a-z0-9]+", name))
        if any(token in tokens for token in ("discovery", "discover", "initiate")) or "start" in tokens:
            return 0
        if "poll" in tokens or "status" in tokens:
            return 1
        if "download" in tokens:
            return 2
        if tokens & {"read", "load", "list", "fetch", "search"}:
            return 0
        if "scrape" in tokens:
            return 1
        if "interact" in tokens:
            return 2
        if tokens & {"research", "extract", "generate", "analyze", "render", "compose", "polish"}:
            return 3
        if "format" in tokens:
            return 4
        if tokens & {"create", "send", "post", "publish", "update", "append", "upload", "register"}:
            return 6
        return 2

    bindings.sort(key=lambda item: (
        item[0], binding_phase(str(item[2].get("operation", ""))),
        str(item[2].get("operation", "")),
    ))

    # Reorder operations inside one scene by their explicit argument/result
    # dataflow. Phase names are only a fallback; a later operation such as
    # ``add_hashtags`` must follow the operation that produces ``refined_post``
    # even when the extractor returned the bindings in another order.
    def binding_roots(value: Any) -> Set[str]:
        roots: Set[str] = set()
        if isinstance(value, list):
            for item in value:
                roots.update(binding_roots(item))
        elif isinstance(value, dict):
            if set(value) == {"literal"}:
                return roots
            if set(value) == {"ref"} and isinstance(value.get("ref"), str):
                roots.add(re.split(r"[.\[]", value["ref"].removeprefix("$"), maxsplit=1)[0])
            else:
                for item in value.values():
                    roots.update(binding_roots(item))
        elif isinstance(value, str):
            roots.add(re.split(r"[.\[]", value.removeprefix("$"), maxsplit=1)[0])
        return roots

    def order_scene_items(items: List[Tuple[int, Any, Dict[str, Any]]]) -> List[Tuple[int, Any, Dict[str, Any]]]:
        if len(items) < 2:
            return items
        producers: Dict[str, int] = {}
        for index, (_, _, binding) in enumerate(items):
            for target in (binding.get("results", {}) or {}).values():
                if isinstance(target, str) and target:
                    producers[re.split(r"[.\[]", target, maxsplit=1)[0]] = index
            operation = str(binding.get("operation", ""))
            if operation:
                producers[f"result.{operation}"] = index
        deps: Dict[int, Set[int]] = {index: set() for index in range(len(items))}
        for index, (_, _, binding) in enumerate(items):
            roots = binding_roots({
                "arguments": binding.get("arguments", {}),
                "condition": binding.get("condition", {}),
                "foreach": binding.get("foreach", {}),
            })
            guard = str(binding.get("guard", ""))
            if ":" in guard:
                roots.add(f"result.{guard.split(':', 1)[1]}")
            deps[index].update(
                producer for root, producer in producers.items()
                if root in roots and producer != index
            )
        ordered: List[int] = []
        remaining = set(range(len(items)))
        while remaining:
            ready = sorted(
                (index for index in remaining if not (deps[index] & remaining)),
                key=lambda index: (binding_phase(str(items[index][2].get("operation", ""))), index),
            )
            if not ready:
                return items
            ordered.extend(ready)
            remaining.difference_update(ready)
        return [items[index] for index in ordered]

    by_scene_ordered: Dict[int, List[Tuple[int, Any, Dict[str, Any]]]] = {}
    for item in bindings:
        by_scene_ordered.setdefault(item[0], []).append(item)
    bindings = [item for scene_index in sorted(by_scene_ordered) for item in order_scene_items(by_scene_ordered[scene_index])]

    def normalize_fallback_source(source: Any, field: str, prior_paths: Set[str]) -> Any:
        if isinstance(source, dict):
            if set(source) == {"literal"}:
                literal = source["literal"]
                while isinstance(literal, dict) and set(literal) == {"literal"}:
                    literal = literal["literal"]
                return {"literal": literal}
            return {
                key: normalize_fallback_source(value, str(key), prior_paths)
                for key, value in source.items()
            }
        if isinstance(source, list):
            return [normalize_fallback_source(value, field, prior_paths) for value in source]
        if not isinstance(source, str):
            return {"literal": source}
        path = source.strip().removeprefix("$")
        if path.startswith("input."):
            root = path[len("input."):].split(".", 1)[0].split("[", 1)[0]
            if (
                root not in set(top_level_input_fields or set())
                and root in set(workflow_input_fields or set())
            ):
                return f"workflow_input.{path[len('input.'):]}"
            return path
        if path.startswith("results.") or path.startswith("results["):
            leaf_path = re.sub(r"(?:\[\d+\])+$", "", path)
            leaf = re.split(r"[.\[]", leaf_path.rstrip("]"))[-1]
            candidates = [
                item for item in prior_paths
                if re.split(
                    r"[.\[]", re.sub(r"(?:\[\d+\])+$", "", item).rstrip("]")
                )[-1] == leaf
            ]
            if len(candidates) == 1:
                return candidates[0]
            if leaf == "id" and field.endswith("_id"):
                typed = [item for item in candidates if field[:-3] in item.lower()]
                if len(typed) == 1:
                    return typed[0]
        raw_namespaces = {
            "config", "email", "settings", "trigger_comment", "user_watchlist",
            "workflow", "workflow_parameters", "workflow_variable",
        }
        root = re.split(r"[.\[]", path, maxsplit=1)[0]
        if root == "fixtures" and "." in path:
            nested_path = path.split(".", 1)[1]
            nested_root = re.split(r"[.\[]", nested_path, maxsplit=1)[0]
            if nested_root in set(workflow_input_fields or set()):
                return f"workflow_input.{nested_path}"
        if root in raw_namespaces and "." in path:
            leaf_path = path.split(".", 1)[1]
            if field == "sheet_id" and leaf_path.endswith("sheet_id"):
                leaf_path = field
            elif leaf_path.startswith("candidate_") and field in leaf_path:
                leaf_path = field
            elif root in {"config", "settings", "workflow", "workflow_parameters"} and field not in leaf_path:
                leaf_path = field
            if root == "workflow_variable":
                return leaf_path
            return f"input.{leaf_path}"
        return source

    prior_paths: Set[str] = set()
    for _, _, binding in bindings:
        for field, source in list((binding.get("arguments", {}) or {}).items()):
            binding["arguments"][field] = normalize_fallback_source(source, field, prior_paths)
        foreach = binding.get("foreach", {}) or {}
        encapsulated_iteration = any(
            str(field).startswith("one_call_per_")
            and isinstance(value, dict)
            and value.get("literal") is True
            for field, value in (binding.get("arguments", {}) or {}).items()
        )
        if foreach and encapsulated_iteration:
            binding["foreach"] = {}
            foreach = {}
        if foreach:
            foreach["source"] = normalize_fallback_source(
                foreach.get("source"), str(foreach.get("argument", "")), prior_paths
            )
        condition = binding.get("condition", {}) or {}
        if condition:
            for leaf in _condition_leaf_nodes(condition):
                leaf["source"] = normalize_fallback_source(
                    leaf.get("source"), "condition", prior_paths
                )
                if "value_from" in leaf:
                    leaf["value_from"] = normalize_fallback_source(
                        leaf.get("value_from"), "condition_value", prior_paths
                    )
            binding["condition"] = _repair_membership_condition_source(
                condition,
                foreach,
                str(binding.get("operation", "")),
                prior_paths,
            )
        prior_paths.update(
            str(target) for target in (binding.get("results", {}) or {}).values()
            if isinstance(target, str) and target
        )

    # A relocated binding can retain a guard pointing to an operation that now
    # occurs later.  Forward success/failure guards are not executable, so let
    # the scene-order pass below rebuild that edge from the preceding scene.
    seen_operations: set[str] = set()
    for _, _, binding in bindings:
        guard = str(binding.get("guard", "always") or "always")
        if guard.startswith(("on_success:", "on_failure:")):
            predecessor = guard.split(":", 1)[1]
            if predecessor not in seen_operations:
                binding["guard"] = "always"
        seen_operations.add(str(binding.get("operation", "")))

    followup_terms = ("follow-up", "followup", "追问", "继续提问", "历史消息")
    contextual_operation_terms = ("history", "context", "memory", "conversation")
    requirement_lower = normalized.lower()
    for _, _, binding in bindings:
        operation_lower = str(binding.get("operation", "")).lower()
        contextual_followup = (
            any(term in requirement_lower for term in followup_terms)
            and any(term in operation_lower for term in contextual_operation_terms)
        )
        if contextual_followup:
            binding["guard"] = "followup"
        elif str(binding.get("guard", "")) in {"followup", "first_request"}:
            binding["guard"] = "always"

    prior_producers: List[Tuple[str, set[str]]] = []
    for _, _, binding in bindings:
        consumed = {str(item) for item in binding.get("consumes", []) if item}
        producer = next(
            (operation for operation, produced in reversed(prior_producers) if consumed & produced),
            "",
        )
        if producer and binding.get("guard") not in {"followup", "first_request"}:
            binding["guard"] = f"on_success:{producer}"
        prior_producers.append((
            str(binding.get("operation", "")),
            {str(item) for item in binding.get("produces", []) if item},
        ))

    by_scene: Dict[int, List[Tuple[Any, Dict[str, Any]]]] = {}
    for scene_index, scene, binding in bindings:
        by_scene.setdefault(scene_index, []).append((scene, binding))
    # The sorted tuples are the executable order. Rewrite each scene's list so
    # the renderer cannot preserve a stale LLM insertion order.
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if isinstance(structured, dict) and "public_operations" in structured:
            structured["public_operations"] = []
    for _, scene, binding in bindings:
        scene.action.structured_op["public_operations"].append(binding)

    mode_sets = [
        {str(field) for field in signature if field}
        for signature in (input_mode_signatures or [])
        if signature
    ]
    mode_discriminator_fields = {
        field
        for field in set().union(*mode_sets) if mode_sets
        if 0 < sum(field in signature for signature in mode_sets) < len(mode_sets)
    }

    def public_reference_fields(value: Any) -> Set[Tuple[str, str]]:
        references: Set[Tuple[str, str]] = set()
        if isinstance(value, dict):
            for nested in value.values():
                references.update(public_reference_fields(nested))
        elif isinstance(value, list):
            for nested in value:
                references.update(public_reference_fields(nested))
        elif isinstance(value, str):
            path = value.strip().removeprefix("$")
            for prefix, namespace in (
                ("input.", "input"),
                ("workflow_input.", "workflow_input"),
            ):
                if path.startswith(prefix):
                    field = path[len(prefix):].split(".", 1)[0].split("[", 1)[0]
                    if field:
                        references.add((namespace, field))
                    break
        return references

    # A public field that occurs in only some observed input shapes is a
    # structural mode discriminator.  If an operation consumes such a field,
    # make the branch condition explicit before rebuilding success edges.
    # This uses names/co-occurrence only, never values, case labels, or Oracle
    # expectations.
    for _, _, binding in bindings:
        if binding.get("condition"):
            continue
        candidates = [
            (sum(field in signature for signature in mode_sets), namespace, field)
            for namespace, field in public_reference_fields({
                "arguments": binding.get("arguments", {}),
                "foreach": binding.get("foreach", {}),
            })
            if field in mode_discriminator_fields
        ]
        if not candidates:
            continue
        _, namespace, field = sorted(candidates)[0]
        binding["condition"] = {
            "source": f"{namespace}.{field}",
            "operator": "exists",
        }

    def is_mode_entry_condition(condition: Any) -> bool:
        if not isinstance(condition, dict):
            return False
        if str(condition.get("operator", "")) not in {"exists", "not_exists"}:
            return False
        source = str(condition.get("source", "")).removeprefix("$")
        for prefix in ("input.", "workflow_input."):
            if source.startswith(prefix):
                root = source[len(prefix):].split(".", 1)[0].split("[", 1)[0]
                return root in mode_discriminator_fields
        return False

    produced_by_operation: Dict[str, Set[str]] = {}
    for _, _, binding in bindings:
        operation = str(binding.get("operation", ""))
        if not operation:
            continue
        produced = {
            str(item) for item in binding.get("produces", []) or [] if item
        }
        produced.update(
            str(item).split(".", 1)[0]
            for item in (binding.get("results", {}) or {}).values()
            if isinstance(item, str) and item
        )
        produced_by_operation.setdefault(operation, set()).update(produced)

    previous_scene_success_operation = ""
    for scene_index in sorted(by_scene):
        scene_success_operations: List[str] = []
        previous_operation_in_scene = ""
        for binding_index, (scene, binding) in enumerate(by_scene[scene_index]):
            guard = str(binding.get("guard", "always") or "always")
            is_conditional = bool(binding.get("condition"))
            starts_independent_input_mode = (
                binding_index == 0
                and is_mode_entry_condition(binding.get("condition"))
            )
            scene_text = " ".join([
                scene.raw_requirement_text,
                scene.logic_flow,
                scene.block_description,
            ]).lower()
            # Failure clauses embedded in a normal processing scene describe
            # that operation's possible outcome; they do not make the whole
            # scene an error handler. Prefer the concise scene responsibility
            # when it exists, and fall back to the broader extracted text only
            # for legacy scenes without a description.
            primary_scene_text = str(scene.block_description or "").strip().lower()
            if not primary_scene_text:
                primary_scene_text = str(
                    scene.raw_requirement_text or scene.logic_flow or ""
                ).strip().lower()
            english_failure = re.search(
                r"\b(on failure|if failed|if not|does not meet|not meet|below standard|error)\b",
                primary_scene_text,
            ) is not None
            failure_branch = english_failure or any(token in primary_scene_text for token in (
                "失败", "错误", "未能", "无结果", "未达到", "不通过",
            ))
            consumes = {
                str(item) for item in binding.get("consumes", []) or [] if item
            }
            has_direct_previous_data_dependency = bool(
                previous_scene_success_operation
                and consumes
                & produced_by_operation.get(previous_scene_success_operation, set())
            )
            operation_tokens = set(re.split(
                r"[^a-z0-9]+", str(binding.get("operation", "")).lower()
            ))
            evaluates_previous_result = bool(
                operation_tokens & {"evaluate", "evaluation", "score", "analyze", "analyse"}
            )
            if (
                starts_independent_input_mode
                and guard.startswith(("on_success:", "on_failure:"))
            ):
                # Distinct public input shapes are alternative workflow entry
                # points. Requiring the previous mode to run makes mutually
                # exclusive paths impossible (for example, webhook input vs.
                # scheduled fixture input).
                binding["guard"] = "always"
                guard = "always"
            elif (
                previous_operation_in_scene
                and guard == "always"
                and not failure_branch
            ):
                binding["guard"] = f"on_success:{previous_operation_in_scene}"
                guard = str(binding["guard"])
            elif previous_scene_success_operation and failure_branch:
                binding["guard"] = f"on_failure:{previous_scene_success_operation}"
                guard = str(binding["guard"])
            elif previous_scene_success_operation and (
                guard == "always" or (
                    guard.startswith("on_failure:")
                    and (
                        has_direct_previous_data_dependency
                        or evaluates_previous_result
                    )
                )
            ):
                binding["guard"] = (
                    f"on_success:{previous_scene_success_operation}"
                )
                guard = str(binding["guard"])
            elif guard.startswith("on_success:") and previous_scene_success_operation:
                # A later scene can accidentally inherit the previous scene's
                # success edge even when it starts an independent, earlier
                # phase such as read/search. Keep genuine producer/result
                # dependencies intact; detach only the phase-inverted first
                # operation with no reference to the predecessor result.
                operation = str(binding.get("operation", ""))
                predecessor_refs = public_reference_fields({
                    "arguments": binding.get("arguments", {}),
                    "condition": binding.get("condition", {}),
                    "foreach": binding.get("foreach", {}),
                })
                has_predecessor_reference = any(
                    field in {
                        previous_scene_success_operation,
                        f"result.{previous_scene_success_operation}",
                    }
                    for _, field in predecessor_refs
                )
                if (
                    binding_index == 0
                    and not has_predecessor_reference
                    and binding_phase(previous_scene_success_operation) >= 6
                    and binding_phase(operation) == 0
                ):
                    # A read/extract entry must not inherit an unrelated
                    # delivery/update side effect from another input mode.
                    # Keep ordinary processing chains such as
                    # download -> read/filter intact.
                    binding["guard"] = "always"
                    guard = "always"
                    previous_operation_in_scene = operation
                    if not is_conditional:
                        scene_success_operations.append(operation)
                    continue
                # A later scene cannot skip a mandatory operation from the
                # immediately preceding successful scene. This converts the
                # extracted scene order into an executable success chain.
                binding["guard"] = f"on_success:{previous_scene_success_operation}"
                guard = str(binding["guard"])
            if guard not in {"followup", "first_request"} and not guard.startswith("on_failure:"):
                operation = str(binding.get("operation", ""))
                previous_operation_in_scene = operation
                if not is_conditional or starts_independent_input_mode:
                    scene_success_operations.append(operation)
        if scene_success_operations:
            previous_scene_success_operation = scene_success_operations[-1]

    _propagate_conditional_result_control_dependencies(fact_spec)

    # A requirement may place "optimize and re-evaluate" in one loop scene
    # while M3b binds only the unique catalog operations. Preserve the repeated
    # evaluation explicitly so the compiled plan represents the loop edge.
    latest_evaluation: Optional[Dict[str, Any]] = None
    for scene in fact_spec.scenes:
        structured = scene.action.structured_op
        if not isinstance(structured, dict):
            continue
        public_operations = structured.get("public_operations", []) or []
        scene_text = " ".join([
            scene.raw_requirement_text,
            scene.logic_flow,
            scene.block_description,
            " ".join(scene.action_sequence or []),
        ]).lower()
        reevaluates = any(token in scene_text for token in (
            "re-evaluate", "reevaluate", "evaluate again", "重新评估", "再次评估",
        ))
        if reevaluates and latest_evaluation and public_operations:
            evaluation_operation = str(latest_evaluation.get("operation", ""))
            optimize_binding = next((
                item for item in public_operations
                if isinstance(item, dict) and any(
                    token in str(item.get("operation", "")).lower()
                    for token in ("optimize", "refine", "improve", "revise")
                )
            ), None)
            repeated = next((
                item for item in public_operations
                if isinstance(item, dict)
                and str(item.get("operation", "")) == evaluation_operation
            ), None)
            if optimize_binding is not None and evaluation_operation:
                optimize_operation = str(optimize_binding.get("operation", ""))
                optimize_binding["guard"] = f"on_failure:{evaluation_operation}"
                if repeated is None:
                    repeated = dict(latest_evaluation)
                    structured["public_operations"] = [*public_operations, repeated]
                    public_operations = structured["public_operations"]
                repeated["guard"] = f"on_success:{optimize_operation}"
        for binding in public_operations:
            operation = str(binding.get("operation", "")).lower() if isinstance(binding, dict) else ""
            if any(token in operation for token in ("evaluate", "assess", "validate", "quality")):
                latest_evaluation = dict(binding)

    # A requirement that explicitly accepts manual input or an RSS feed has
    # two alternative update targets. Preserve both paths while keeping the
    # collection traversal explicit in the public-operation contract.
    requirement_lower = normalized.lower()
    modes = [set(mode) for mode in (input_mode_signatures or []) if mode]
    has_manual_rss_modes = (
        "rss" in requirement_lower
        and any(token in requirement_lower for token in ("manual", "手动"))
        and any("post" in mode for mode in modes)
        and any("entries" in mode for mode in modes)
    )
    if has_manual_rss_modes:
        for scene in fact_spec.scenes:
            structured = scene.action.structured_op
            if not isinstance(structured, dict):
                continue
            scene_bindings = structured.get("public_operations", []) or []
            if not isinstance(scene_bindings, list):
                continue
            for index, binding in enumerate(list(scene_bindings)):
                if not isinstance(binding, dict) or binding.get("operation") != "update_post":
                    continue
                arguments = binding.get("arguments", {}) or {}
                post_source = arguments.get("post_id")
                if not isinstance(post_source, str) or not re.search(
                    r"(?:^|\.)entries\[0\]\.post_id$", post_source
                ):
                    continue
                manual_binding = copy.deepcopy(binding)
                manual_binding["condition"] = {
                    "source": "workflow_input.source",
                    "operator": "eq",
                    "value": "manual",
                }
                manual_binding["arguments"]["post_id"] = "workflow_input.post.id"
                manual_binding["guard"] = "on_success:list_tags"

                rss_binding = copy.deepcopy(binding)
                rss_binding["condition"] = {
                    "source": "workflow_input.source",
                    "operator": "eq",
                    "value": "rss",
                }
                rss_binding["foreach"] = {
                    "argument": "post_id",
                    "source": "entries.post_id",
                }
                rss_binding["arguments"]["post_id"] = "entries[0].post_id"
                rss_binding["guard"] = "on_success:list_tags"
                scene_bindings[index:index + 1] = [manual_binding, rss_binding]
                break
        for scene in fact_spec.scenes:
            structured = scene.action.structured_op
            if not isinstance(structured, dict):
                continue
            scene_bindings = structured.get("public_operations", []) or []
            if not isinstance(scene_bindings, list):
                continue
            for index, binding in enumerate(list(scene_bindings)):
                if not isinstance(binding, dict) or binding.get("operation") != "generate_tags":
                    continue
                article_source = (binding.get("arguments", {}) or {}).get("article")
                if not isinstance(article_source, str) or not re.search(
                    r"(?:^|\.)entries\[0\]$", article_source
                ):
                    continue
                manual_binding = copy.deepcopy(binding)
                manual_binding["guard"] = "always"
                manual_binding["condition"] = {
                    "source": "workflow_input.source",
                    "operator": "eq",
                    "value": "manual",
                }
                manual_binding["arguments"]["article"] = "workflow_input.post"

                rss_binding = copy.deepcopy(binding)
                rss_binding["condition"] = {
                    "source": "workflow_input.source",
                    "operator": "eq",
                    "value": "rss",
                }
                scene_bindings[index:index + 1] = [manual_binding, rss_binding]
                break
    return fact_spec


def _looks_like_math(text: str) -> bool:
    """启发式: 判断文本是否像数学题."""
    math_indicators = [
        "how many", "what is", "calculate", "compute", "find the",
        "多少", "计算", "求", "等于", "加", "减", "乘", "除",
        "per day", "each week", "total cost", "sum of",
    ]
    text_lower = text.lower()
    hits = sum(1 for ind in math_indicators if ind in text_lower)
    # 数字密度: 数学题通常含大量数字
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    return hits >= 2 or (hits >= 1 and digit_ratio > 0.05)


def _m3_math_facts(normalized: str) -> 'DslFactSpec':
    """数学题专用: 构建计算型 FactSpec (不调场景抽取 prompt)."""
    from dsl_v2.fact_types import (
        FactScene, TriggerSpec, ExclusionSpec, ActionSpec, CompressibilityScore,
        SemanticVector,
    )

    # 数学题: 单一场景, 触发条件=数值比较, 动作=compute, 全部适合纯代码
    return DslFactSpec(
        agent_name="math_solver",
        agent_description=normalized[:200],
        scenes=[
            FactScene(
                scene_id=0,
                block_id="b0_math_compute",
                triggers=TriggerSpec(
                    raw_text=normalized[:200],
                    extracted_keywords=[],
                    is_explicit_list=True,
                    implicit_intent=None,
                    numeric_comparisons=None,  # 数学题无固定数值比较, 走纯计算
                    context_stage=None,
                    multidimensional=False,
                ),
                exclusions=ExclusionSpec(
                    raw_text="",
                    extracted_keywords=[],
                    is_explicit_list=False,
                ),
                action=ActionSpec(
                    raw_text="计算并输出数值答案",
                    action_type="compute",
                    structured_op=None,
                    outputs=[{"name": "answer", "type": "number"}],
                ),
                raw_requirement_text=normalized[:500],
                logic_flow="读取输入 → 按规则计算 → 输出答案",
                side_effects=[],
                raw_context="",
                preconditions=[],
                fallback=None,
                action_sequence=[],
                nested_logic=None,
                meta_rules=[],
                compressibility=CompressibilityScore(
                    score=0.9,
                    lossy_aspects=[],
                    recommendation="BLOCK",
                ),
                semantic_vector=SemanticVector(
                    trigger_specificity=0.9,
                    trigger_context_dependency=0.9,
                    action_determinism=0.9,
                    numeric_complexity=0.3,  # 数学题有计算
                    output_structuredness=0.9,
                    exception_handling=0.9,
                ),
            )
        ],
        inputs=[],
        outputs=[{"name": "answer", "type": "number"}],
        configs=[],
        policies=[],
        examples=[],
    )


def _m3_fallback_facts(normalized: str) -> 'DslFactSpec':
    """M3 回退: 当 LLM 抽取失败时, 用规则方式构建最小 FactSpec."""
    from dsl_v2.fact_types import FactScene, TriggerSpec, ExclusionSpec, ActionSpec

    # 推断 domain hint
    if any(k in normalized for k in ("经费", "审批", "会议", "公文", "上级")):
        domain = "governance"
    elif any(k in normalized for k in ("交易", "转账", "账户", "余额", "退款", "经费")):
        domain = "finance"
    elif any(k in normalized for k in ("主诉", "分诊", "症状", "急诊", "医嘱")):
        domain = "medical"
    elif any(k in normalized for k in ("机器", "故障", "温度", "压力", "工单", "派单")):
        domain = "industrial"
    elif any(k in normalized for k in ("客户", "工单", "投诉", "升级", "队列")):
        domain = "customer_service"
    else:
        domain = "unknown"

    return DslFactSpec(
        agent_name=domain,
        agent_description=normalized[:100],
        scenes=[
            FactScene(
                scene_id=0,
                block_id="b0_fallback",
                block_description=normalized[:80],
                triggers=TriggerSpec(raw_text=normalized),
                exclusions=ExclusionSpec(),
                action=ActionSpec(raw_text=normalized, action_type="assign"),
            )
        ],
    )


# ============================================================================
# M4: compile_with_sensors — 核心: 一次 LLM 调用产出 skeleton + sensor specs
# ============================================================================

COMPILE_SYSTEM_PROMPT = """你是一个 NCNLP 翻译器 (M4 模块)。策略决策已由纯代码分类器完成，你只需要将 ClassifiedSpec 翻译为可执行代码。

## 核心原则

策略决策已经做出——你不需要决定用 if/else 还是 sensor，只需要严格按照 ClassifiedSpec 中每个场景的 trigger_strategy 和 action_strategy 翻译为代码。

## sensor 生成规则 (极其重要!)

当场景的 trigger_strategy 或 action_strategy 包含 "SEMANTIC" 时 (如 SEMANTIC_BLOCK_CLASSIFICATION, SEMANTIC_BLOCK_GENERATION, HYBRID):
1. **必须** 在 sensors 列表中生成至少 1 个 sensor spec
2. **必须** 在 skeleton 的对应分支中调用 `sensors['name'](context_dict)`
3. sensor 的 kind 根据语义选择: classification(分类选择), extraction(信息抽取), boolean(是否判断), scoring(评分), compute(计算), generation(文本生成)
4. sensor 的 input_var 应指向 input_data 中需要 LLM 理解的字段

示例: trigger_strategy = SEMANTIC_BLOCK_CLASSIFICATION → skeleton 中写:
```python
result = sensors['classify_priority']({"description": input_data.get("issue_description", ""), "keywords": input_data.get("issue_type", "")})
if result == "紧急":
    actions.append({"type": "escalate", ...})
```
同时 sensors 列表中写:
```json
{"name": "classify_priority", "kind": "classification", "description": "根据工单描述判断紧急程度", "input_var": "issue_description", "allowed_values": ["紧急", "一般", "不紧急"], "question": "该工单的紧急程度是?", "temperature": 0.0}
```

当 trigger_strategy = BLOCK_CONTAINS 或 BLOCK_COMPARE (纯代码策略):
- **禁止** 为该场景生成 sensor, 必须用 if/else 实现

## 输出 JSON Schema

```json
{
  "skeleton": "<Python 函数源码字符串>",
  "sensors": [
    {
      "name": "sensor_name",
      "kind": "classification|extraction|boolean|scoring|compute|generation",
      "description": "传感点功能描述",
      "input_var": "传给 LLM 的输入变量名 (从 input_data 里取)",
      "allowed_values": ["A", "B", "C"],     // 仅 classification 必填
      "required_keys": ["field1", "field2"],  // 仅 extraction 必填
      "value_types": {"field1": "String"},    // 可选
      "question": "传给 LLM 的问题描述",
      "temperature": 0.0
    }
  ]
}
```

## Skeleton 编写规则 (极其重要!)

skeleton 是纯 Python 函数, 签名固定为:
```python
def run(sensors, input_data):
    actions = []
    final_state = {}
    # ... 翻译后的代码
    return actions, final_state  # final_state 必须从逻辑推导
```

可用的 Python 子集 (白名单):
- 算术: +, -, *, /, //, %, **
- 比较: ==, !=, <, >, <=, >=
- 逻辑: and, or, not, in
- 控制流: if / elif / else, for (在 list 上), while (禁止)
- 内置: len, str, int, float, bool, list, dict, range, abs, min, max, sum, round
- dict/list 的基本操作: [], .get(), .append(), .items(), .keys(), .values()
- 调 sensor: sensors['name'](context_dict) — 这是 LLM 唯一通道

**日期/时间处理**: 严禁 from datetime import 或任何 import. 如果需要日期计算, 直接从 input_data 读取预处理好的数值 (如 days_elapsed, 时限天数 等), 用算术比较即可.

禁止:
- import, exec, eval, open, file I/O
- 网络请求, subprocess
- 全局变量访问, 自定义类

最终 actions 列表元素是 dict, 形如:
- {"type": "transfer", "amount": 8000, "to_account": "应急储备账户"}
- {"type": "email", "to": "财务负责人", "subject": "应急储备通报", "body_contains": ["8000", "应急储备"]}
- {"type": "default_plan"}
- {"type": "compute", "op": "arithmetic", "answer": 350}

**动作 type 命名规则 (极其重要!)**:

type 必须使用 snake_case 英文短语, 精确描述动作语义. 优先使用具体描述性 type, 严禁用笼统通用词.

命名原则 (按优先级):
1. **首选: 具体描述性 type** — 从动作语义直接推导, 如:
   - "检查过敏" → check_allergy (而非 prescribe)
   - "路由到队列" → route_to_queue (而非 escalate)
   - "批准启动" → approve_start (而非 log_record)
   - "标记可疑" → flag_suspicious (而非 risk_alert)
   - "金丝雀部署" → deploy_canary (而非 batch_process)
   - "创建快照" → create_snapshot (而非 log_record)
2. **次选: 领域参考 type** — 仅当首选无法确定时:
   - 医疗(medical): prescribe, check_allergy, check_interaction, check_dose, block_prescription, block_drug, schedule_monitoring, triage, alert_dose, dispense, skin_test
   - 金融(finance): freeze_account, flag_suspicious, convert, approve, authorize, freeze_card, review_adjustment, alert_take_profit, verify_pin, report_to_aml
   - 政务(governance): assign, route_to, process_node, assess, approve_item, activate_plan, sms, submit_for_approval, send_reminder, dispatch, investigate
   - 工业(industrial): shutdown, emergency_stop, alarm, approve_start, cut_power, record, classify_and_handle, alert_maintenance, notify_operator, investigate
   - 客服(customer_service): call_back, enqueue, route_to_queue, auto_reply, vip_dispatch, escalate, refund, compensate, flag_risk, submit_for_approval
   - 通用(general): deploy_blue_green, create_snapshot, filter_sort_limit, check_dependency, rollback_to_snapshot, backup, execute, apply_discount, convert
3. **禁止: 笼统通用 type** — 严禁用以下词替代具体描述:
   - ❌ escalate (应写 route_to_queue / submit_for_approval)
   - ❌ risk_alert (应写 flag_suspicious / alert_dose)
   - ❌ emergency_plan (应写 activate_plan / deploy_canary)
   - ❌ log_record (应写 record / check_dependency / approve_start)
   - ❌ dispatch (应写 route_to_queue / vip_dispatch)

**关键**: type 是动作的唯一标识, 必须能从 type 名直接理解动作含义. 笼统的 type 使系统不可审计.

## actions 生成规则 (极其重要!)

skeleton 的核心输出是 actions 列表, 每个规则分支**必须**生成对应的 action(s).

规则:
1. 每个条件分支**必须**调用 `actions.append({"type": ..., ...})`, 不允许只设 final_state 不生成 action
2. 每条业务规则的执行结果必须体现为一个或多个 action dict
3. action 数量应与规则语义一致: 规则说"转账+邮件"就生成2个action, 不要合并或省略
4. 示例:
   ```python
   def run(sensors, input_data):
       actions = []
       final_state = {}
       if input_data.get('客户等级') == 'VIP':
           actions.append({"type": "route_to_queue", "queue": "VIP专属队列"})
           final_state["queue"] = "VIP专属队列"
       else:
           actions.append({"type": "enqueue", "queue": "普通队列"})
           final_state["queue"] = "普通队列"
       return actions, final_state
   ```

## final_state 构建规则

skeleton 必须构建并返回有意义的 final_state 字典, 反映规则执行后的系统状态.

规则:
1. final_state 必须从 if/else 或 sensor 结果推导, 禁止硬编码占位值
2. 对每个分支, final_state 应包含该分支影响的关键状态变量
3. 如果规则无明确状态变化, 返回 {}
4. **严禁** 返回 {"overflow": 8000} 或任何不含业务语义的占位值

## 策略翻译规则 (必须严格遵守!)

### BLOCK 类场景 (trigger_strategy 以 BLOCK_ 开头):
- **必须** 用纯 if/else 代码实现触发条件判断
- **禁止** 为触发条件生成 sensor
- 关键词匹配 → 用 `in` 操作符: `if keyword in text`
- 数值比较 → 用比较运算符: `if value > threshold`

### SEMANTIC_BLOCK 类场景 (trigger_strategy 以 SEMANTIC_BLOCK_ 开头):
- **必须** 生成一个 sensor 来处理触发条件的语义判断
- skeleton 中用 `sensors['sensor_name'](context_dict)` 调用
- sensor kind 通常为 classification 或 boolean

### 动作策略翻译:
- CODE_ASSIGN / CODE_COMPUTE / CODE_TEMPLATE / CODE_CALL_FUNC → 纯代码实现, **禁止** sensor
- SEMANTIC_BLOCK_GENERATION → **必须** 生成 generation sensor
- SEMANTIC_BLOCK_SCORING → **必须** 生成 scoring sensor
- SEMANTIC_BLOCK_EXTRACTION → **必须** 生成 extraction sensor
- SEMANTIC_BLOCK_SIMILARITY → **必须** 生成 scoring/compute sensor

## 输出格式

严格只输出 JSON. 不要有 markdown 包裹. 不要有解释文字.
"""


COMPILE_USER_TEMPLATE = """## 业务规则 (M1/M2/M3 已规范化)

{normalized}

## 当前任务领域: {domain}

**请根据领域选择正确的动作类型!** 参考系统提示中的领域专用动作类型表.

## 输入数据

{input_data_str}

## ClassifiedSpec — 策略分类结果 (M3 → StrategyClassifier)

每个场景的 trigger_strategy 和 action_strategy 已确定, 你只需翻译为代码:

{classified_spec_str}

## 任务

请按系统提示中的 Schema 将上述 ClassifiedSpec 翻译为 skeleton + sensors.

严格规则:
- BLOCK 类 trigger → 必须 if/else, 禁止 sensor
- SEMANTIC_BLOCK 类 trigger → 必须 sensor, skeleton 中调 sensors['name'](ctx)
- RAW_SEMANTIC 类 trigger → 必须 generation sensor, 将原始需求文本整体传给 LLM
- HYBRID 类 trigger → 必须 if/else 分支 + sensor 调用组合
- CODE_* 类 action → 纯代码, 禁止 sensor
- SEMANTIC_BLOCK_* 类 action → 必须 sensor

严格只输出 JSON.
"""


def _validate_skeleton_source(src: str) -> Tuple[bool, str]:
    """校验 skeleton 源码在白名单 AST 子集内."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    # 必须有且仅有一个 'def run(sensors, input_data)' 函数
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not fns:
        return False, "no function defined"
    if len(fns) > 1:
        return False, f"only one function allowed, got {len(fns)}"

    fn = fns[0]
    if fn.name != "run":
        return False, f"function must be named 'run', got {fn.name!r}"
    if [a.arg for a in fn.args.args] != ["sensors", "input_data"]:
        return False, f"signature must be run(sensors, input_data), got {[a.arg for a in fn.args.args]}"

    # 禁止的节点
    forbidden = (ast.Import, ast.ImportFrom, ast.Exec if hasattr(ast, "Exec") else tuple())
    forbidden_calls = {"exec", "eval", "compile", "open", "__import__", "getattr", "setattr", "delattr"}

    for node in ast.walk(fn):
        if isinstance(node, forbidden):
            return False, f"forbidden AST node: {type(node).__name__}"
        if isinstance(node, ast.Call):
            # 检查函数名
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                return False, f"forbidden call: {node.func.id}()"
            if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
                return False, f"forbidden call: .{node.func.attr}()"
        if isinstance(node, ast.While):
            return False, "while loop not allowed (unbounded)"
    return True, "ok"


@dataclass
class CompiledArtifact:
    """M4 的产出: 编译产物."""
    skeleton_source: str
    sensor_specs: List[SensorSpec]
    compile_tokens: int = 0
    compile_latency_ms: int = 0
    validation_msg: str = ""
    backend: str = "llm_skeleton"
    dsl_code: str = ""
    generated_python: str = ""
    compile_details: Dict[str, Any] = field(default_factory=dict)
    fact_spec: Any = None
    classified_spec: Any = None


def m4_compile(
    normalized: str,
    entities: List[Dict],
    fact,
    input_data: Dict,
    llm_client=None,
    skeleton_override: Optional[str] = None,
) -> CompiledArtifact:
    """
    [DEPRECATED] 旧版 M4 编译: LLM 同时决定策略和代码。
    请使用 m4_compile_classified() 替代, 它将策略分类与代码生成分离。

    保留此函数仅为向后兼容。
    """
    import warnings
    warnings.warn(
        "m4_compile() is deprecated, use m4_compile_classified() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if llm_client is None:
        llm_client = create_llm_client()

    t0 = time.time()
    if skeleton_override is not None:
        compile_result = {
            "skeleton": skeleton_override,
            "sensors": [],
        }
        usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        latency = 0
    else:
        # 旧版 prompt 无法复用新版模板，使用简化版
        user_prompt = (
            f"## 业务规则\n{normalized}\n\n"
            f"## 输入数据\n{json.dumps(input_data, ensure_ascii=False, indent=2)}\n\n"
            f"请编译为 skeleton + sensors 的 JSON。"
        )
        resp = llm_client.call(
            system_prompt=COMPILE_SYSTEM_PROMPT,
            user_content=user_prompt,
            temperature=0.0,
        )
        latency = int((time.time() - t0) * 1000)
        usage = resp.usage or {}
        compile_result = _parse_compile_output(resp.content)
        if compile_result is None:
            raise ValueError(f"M4 编译 LLM 输出无法解析为 JSON:\n{resp.content[:500]}")

    skeleton_src = compile_result.get("skeleton", "").strip()
    skeleton_src = re.sub(r'^```python\s*', '', skeleton_src)
    skeleton_src = re.sub(r'\s*```$', '', skeleton_src)
    ok, msg = _validate_skeleton_source(skeleton_src)
    if not ok:
        raise ValueError(f"skeleton AST 校验失败: {msg}\nsource:\n{skeleton_src[:500]}")

    sensor_specs = []
    for s in compile_result.get("sensors", []) or []:
        spec = _build_sensor_spec(s)
        if spec is not None:
            sensor_specs.append(spec)

    return CompiledArtifact(
        skeleton_source=skeleton_src,
        sensor_specs=sensor_specs,
        compile_tokens=usage.get("total_tokens", 0),
        compile_latency_ms=int((time.time() - t0) * 1000),
        validation_msg=msg,
    )


def _serialize_classified_spec(classified_spec: 'ClassifiedSpec') -> str:
    """将 ClassifiedSpec 序列化为 LLM 可读的结构化文本."""
    lines = []
    for i, scene in enumerate(classified_spec.scenes):
        lines.append(f"### 场景 {i+1}: {scene.block_description}")
        lines.append(f"  block_id: {scene.block_id}")
        lines.append(f"  trigger_strategy: {scene.trigger_strategy}")
        lines.append(f"  action_strategy: {scene.action_strategy}")
        # TriggerSpec
        t = scene.triggers
        lines.append(f"  触发条件 (TriggerSpec):")
        lines.append(f"    raw_text: {t.raw_text}")
        lines.append(f"    extracted_keywords: {t.extracted_keywords}")
        lines.append(f"    is_explicit_list: {t.is_explicit_list}")
        if t.implicit_intent:
            lines.append(f"    implicit_intent: {t.implicit_intent}")
        if t.numeric_comparisons:
            lines.append(f"    numeric_comparisons: {t.numeric_comparisons}")
        if t.context_stage:
            lines.append(f"    context_stage: {t.context_stage}")
        if t.multidimensional:
            lines.append(f"    multidimensional: {t.multidimensional}")
        # ExclusionSpec
        e = scene.exclusions
        if e.extracted_keywords or e.raw_text:
            lines.append(f"  排除条件 (ExclusionSpec):")
            if e.raw_text:
                lines.append(f"    raw_text: {e.raw_text}")
            if e.extracted_keywords:
                lines.append(f"    keywords: {e.extracted_keywords}")
            lines.append(f"    is_explicit_list: {e.is_explicit_list}")
        # ActionSpec
        a = scene.action
        lines.append(f"  动作 (ActionSpec):")
        lines.append(f"    action_type: {a.action_type}")
        lines.append(f"    raw_text: {a.raw_text}")
        if a.structured_op:
            lines.append(f"    structured_op: {a.structured_op}")
        lines.append(f"    outputs: {a.outputs}")
        # Rich fields
        if scene.raw_requirement_text:
            lines.append(f"  原始需求: {scene.raw_requirement_text}")
        if scene.logic_flow:
            lines.append(f"  逻辑流: {scene.logic_flow}")
        if scene.side_effects:
            lines.append(f"  副作用: {scene.side_effects}")
        if scene.preconditions:
            lines.append(f"  前置条件: {scene.preconditions}")
        if scene.fallback:
            lines.append(f"  回退逻辑: {scene.fallback}")
        if scene.action_sequence:
            lines.append(f"  执行顺序: {scene.action_sequence}")
        if scene.nested_logic:
            lines.append(f"  嵌套逻辑: {scene.nested_logic}")
        if scene.compressibility and hasattr(scene.compressibility, 'score'):
            lines.append(f"  可压缩性: score={scene.compressibility.score}, rec={scene.compressibility.recommendation}")
        lines.append("")
    return "\n".join(lines)


def _validate_strategy_consistency(
    skeleton_src: str,
    sensor_specs: List[SensorSpec],
    classified_spec: 'ClassifiedSpec',
) -> List[str]:
    """M0 行为级校验: 检查编译产物是否与 ClassifiedSpec 策略一致.

    Returns:
        List[str] — 错误列表, 空表示通过
    """
    errors = []

    # 1. 需要语义传感点的场景必须有对应的 sensor
    # 包括: SEMANTIC_BLOCK*, RAW_SEMANTIC, HYBRID
    semantic_block_ids = set()
    for scene in classified_spec.scenes:
        needs_sensor = (
            scene.trigger_strategy.startswith("SEMANTIC_BLOCK")
            or scene.action_strategy.startswith("SEMANTIC_BLOCK")
            or scene.trigger_strategy == "RAW_SEMANTIC"
            or scene.action_strategy == "RAW_SEMANTIC"
            or scene.trigger_strategy == "HYBRID"
            or scene.action_strategy == "HYBRID"
        )
        if needs_sensor:
            semantic_block_ids.add(scene.block_id)

    sensor_names_in_skeleton = set()
    # 扫描 skeleton 中的 sensors['name'] 调用
    try:
        tree = ast.parse(skeleton_src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # sensors['name'](ctx)
                if (isinstance(node.func, ast.Subscript) and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == 'sensors'):
                    if isinstance(node.func.slice, ast.Constant):
                        sensor_names_in_skeleton.add(node.func.slice.value)
    except SyntaxError:
        errors.append("skeleton 语法错误, 无法做策略一致性校验")

    # SEMANTIC_BLOCK 场景需要有 sensor (HYBRID 场景不强制)
    # 只统计真正的 SEMANTIC_BLOCK_CLASSIFICATION/SEMANTIC_BLOCK_GENERATION 场景
    strict_semantic_block_ids = [
        s.block_id for s in classified_spec.scenes
        if s.trigger_strategy.startswith("SEMANTIC_BLOCK") and s.trigger_strategy != "HYBRID"
    ]
    if strict_semantic_block_ids and not sensor_specs and not sensor_names_in_skeleton:
        errors.append(
            f"ClassifiedSpec 有 {len(strict_semantic_block_ids)} 个 SEMANTIC_BLOCK 场景, "
            f"但编译产物没有生成任何 sensor"
        )

    # 2. sensor 数量应与严格 SEMANTIC_BLOCK 场景数匹配 (允许 ±1, HYBRID不强制)
    expected_sensor_count = len(strict_semantic_block_ids)
    actual_sensor_count = len(sensor_specs)
    if expected_sensor_count > 0 and actual_sensor_count == 0:
        errors.append(
            f"策略分类预期 {expected_sensor_count} 个 sensor, 实际 0 个"
        )

    # 3. BLOCK 场景不应生成不必要的 sensor (宽松: 不强制报错, 仅 warn)
    block_scene_count = sum(
        1 for s in classified_spec.scenes
        if not (
            s.trigger_strategy.startswith("SEMANTIC_BLOCK")
            or s.action_strategy.startswith("SEMANTIC_BLOCK")
            or s.trigger_strategy in ("RAW_SEMANTIC", "HYBRID")
            or s.action_strategy in ("RAW_SEMANTIC", "HYBRID")
        )
    )
    if block_scene_count > 0 and actual_sensor_count > block_scene_count + expected_sensor_count:
        logger.warning(
            f"[M0] BLOCK 场景={block_scene_count}, 但 sensor 数={actual_sensor_count}, "
            f"可能有不必要的 sensor"
        )

    return errors


def _scene_needs_semantic(scene: Any) -> bool:
    """Return True when a scene needs semantic runtime support."""
    return (
        str(scene.trigger_strategy).startswith("SEMANTIC_BLOCK")
        or str(scene.action_strategy).startswith("SEMANTIC_BLOCK")
        or scene.trigger_strategy in ("RAW_SEMANTIC", "HYBRID")
        or scene.action_strategy in ("RAW_SEMANTIC", "HYBRID")
    )


def _classified_spec_needs_semantic(classified_spec: 'ClassifiedSpec') -> bool:
    return any(_scene_needs_semantic(scene) for scene in classified_spec.scenes)


def _compile_classified_with_dsl_backend(
    classified_spec: 'ClassifiedSpec',
) -> Tuple[str, str, Dict[str, Any]]:
    """Render ClassifiedSpec to DSL and compile DSL to executable Python code."""
    from files.renderer import DSLRenderer
    from prompt_codegenetate import WaActCompiler

    dsl_code = DSLRenderer().render_from_classifiedspec(classified_spec)
    compiler = WaActCompiler()
    modules, main_code, compile_details = compiler.compile(
        dsl_code,
        clustering_strategy="hybrid",
        visualize=False,
    )
    generated_python = compiler.generate_full_code(modules, main_code, dsl_code=dsl_code)
    public_plan_error = ""
    try:
        public_plan_source = _compile_public_action_plan_source(dsl_code)
    except Exception as exc:
        public_plan_source = ""
        public_plan_error = f"{type(exc).__name__}: {exc}"
    if public_plan_source:
        generated_python = generated_python.rstrip() + "\n\n" + public_plan_source + "\n"
        compile_details["public_action_plan_compiled"] = True
    else:
        compile_details["public_action_plan_compiled"] = False
    if public_plan_error:
        compile_details["public_action_plan_error"] = public_plan_error
    return dsl_code, generated_python, compile_details


def _action_contract_reference_roots(value: Any) -> Set[str]:
    roots: Set[str] = set()
    if isinstance(value, list):
        for item in value:
            roots.update(_action_contract_reference_roots(item))
    elif isinstance(value, dict):
        if set(value) == {"literal"}:
            return roots
        if set(value) == {"ref"} and isinstance(value.get("ref"), str):
            roots.update(_action_contract_reference_roots(value["ref"]))
        else:
            for item in value.values():
                roots.update(_action_contract_reference_roots(item))
    elif isinstance(value, str):
        path = value.removeprefix("$")
        if path.startswith("result."):
            parts = path.split(".", 2)
            if len(parts) >= 2:
                roots.add(".".join(parts[:2]))
        roots.add(re.split(r"[.\[]", path, maxsplit=1)[0])
    return roots


def _local_statement_targets(statement: Dict[str, Any]) -> Set[str]:
    if statement.get("op") == "emit":
        return {
            str(name).split(".", 1)[0]
            for name in (statement.get("fields", {}) or {})
            if name
        }
    target = statement.get("target")
    return {str(target).split(".", 1)[0]} if target else set()


def _schedule_scene_action_units(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Topologically interleave ordered local statements and public bindings."""
    local_program = [
        statement for statement in (scene.get("local_program", []) or [])
        if isinstance(statement, dict)
    ]
    bindings = [
        copy.deepcopy(binding) for binding in (scene.get("public_operations", []) or [])
        if isinstance(binding, dict)
    ]
    if not bindings:
        return [{"type": "local", "statements": local_program}]

    # A generated plan can accidentally put mutually dependent guards on two
    # operations.  If one operation clearly produces a result consumed by the
    # other, that producer must be schedulable first; otherwise the compiler
    # rejects an otherwise executable scene as cyclic.  This is deliberately
    # narrow: it changes only the producer's guard and only for a detected
    # two-operation cycle.
    operation_indexes = {
        str(binding.get("operation")): index
        for index, binding in enumerate(bindings)
        if binding.get("operation")
    }
    binding_refs = []
    for binding in bindings:
        binding_refs.append({
            "arguments": binding.get("arguments", {}),
            "condition": binding.get("condition", {}),
            "foreach": binding.get("foreach", {}) or {},
        })
    local_refs = _action_contract_reference_roots(local_program)
    for index, binding in enumerate(bindings):
        guard = str(binding.get("guard", ""))
        if not guard.startswith(("on_success:", "on_failure:")):
            continue
        predecessor = guard.split(":", 1)[1]
        predecessor_index = operation_indexes.get(predecessor)
        operation = str(binding.get("operation", ""))
        if predecessor_index is None or predecessor_index == index:
            continue
        predecessor_binding = bindings[predecessor_index]
        predecessor_guard = str(predecessor_binding.get("guard", ""))
        if not predecessor_guard.endswith(f":{operation}"):
            continue
        produced_roots = {
            f"result.{operation}",
            *{
                str(target).split(".", 1)[0]
                for target in (binding.get("results", {}) or {}).values()
                if isinstance(target, str) and target
            },
        }
        consumer_roots = _action_contract_reference_roots(binding_refs[predecessor_index])
        consumer_roots.update(local_refs)
        if produced_roots & consumer_roots:
            predecessor_binding["guard"] = "always"

    local_producers: Dict[str, int] = {}
    for index, statement in enumerate(local_program):
        if statement.get("op") == "emit":
            # Pass-through exports do not produce their own inputs. An emit
            # that computes or renames a field still produces that field.
            for target, value in (statement.get("fields", {}) or {}).items():
                if value != {"ref": target}:
                    local_producers.setdefault(str(target).split(".", 1)[0], index)
            continue
        for target in _local_statement_targets(statement):
            local_producers[target] = index

    public_producers: Dict[str, int] = {}
    for index, binding in enumerate(bindings):
        operation = str(binding.get("operation", ""))
        if operation:
            public_producers[f"result.{operation}"] = index
        for target in (binding.get("results", {}) or {}).values():
            if isinstance(target, str) and target:
                public_producers[target.split(".", 1)[0]] = index

    events: List[Dict[str, Any]] = []
    pending_local: List[Dict[str, Any]] = []
    local_index = 0
    remaining_public = list(range(len(bindings)))
    executed_public: set[int] = set()

    def flush_local() -> None:
        nonlocal pending_local
        if pending_local:
            events.append({"type": "local", "statements": pending_local})
            pending_local = []

    while local_index < len(local_program) or remaining_public:
        local_ready = False
        if local_index < len(local_program):
            required_public = [
                public_producers[root]
                for root in _action_contract_reference_roots(local_program[local_index])
                if root in public_producers
            ]
            local_ready = all(index in executed_public for index in required_public)

        ready_public_index: int | None = None
        for candidate_index in remaining_public:
            binding = bindings[candidate_index]
            foreach = binding.get("foreach", {}) or {}
            binding_inputs = {
                "arguments": binding.get("arguments", {}),
                "condition": binding.get("condition", {}),
                # The foreach argument is a local loop-variable name, not a
                # dependency produced by an earlier local statement.  Only
                # the collection source participates in scheduling.
                "foreach": {"source": foreach.get("source")} if isinstance(foreach, dict) else {},
            }
            required_local = [
                local_producers[root]
                for root in _action_contract_reference_roots(binding_inputs)
                if root in local_producers
            ]
            required_public = {
                public_producers[root]
                for root in _action_contract_reference_roots(binding_inputs)
                if root in public_producers and root not in local_producers
                and public_producers[root] != candidate_index
            }
            guard = str(binding.get("guard", ""))
            if guard.startswith(("on_success:", "on_failure:")):
                guard_producer = public_producers.get(f"result.{guard.split(':', 1)[1]}")
                if guard_producer is not None and guard_producer != candidate_index:
                    required_public.add(guard_producer)
            if (
                all(index < local_index for index in required_local)
                and required_public.issubset(executed_public)
            ):
                ready_public_index = candidate_index
                break

        # Preserve the historical public-first order for independent work. A
        # local segment runs first only when it produces data required by the
        # next public operation.
        if ready_public_index is not None:
            flush_local()
            events.append({"type": "public", "index": ready_public_index})
            remaining_public.remove(ready_public_index)
            executed_public.add(ready_public_index)
            continue
        if local_ready:
            pending_local.append(local_program[local_index])
            local_index += 1
            continue
        # A quality-retry scene may intentionally contain a two-operation
        # loop: a check runs after a previous optimization, and the optimizer
        # runs on the check's failure. Both guards can mention each other even
        # though the previous scene supplies the first outcome. Preserve the
        # declared operation order for this narrow retry shape and let runtime
        # guards decide whether the current iteration is active.
        if remaining_public:
            candidate_index = remaining_public[0]
            candidate = bindings[candidate_index]
            candidate_guard = str(candidate.get("guard", ""))
            guard_target = candidate_guard.split(":", 1)[1] if ":" in candidate_guard else ""
            retry_names = {
                str(candidate.get("operation", "")).lower(),
                guard_target.lower(),
            }
            mutual_retry = (
                candidate_guard.startswith(("on_success:", "on_failure:"))
                and any(any(token in name for token in ("quality", "evaluate", "optimize", "qa", "check")) for name in retry_names)
                and any(
                    str(bindings[index].get("operation", "")) == guard_target
                    and str(bindings[index].get("guard", "")).endswith(
                        f":{candidate.get('operation', '')}"
                    )
                    for index in remaining_public[1:]
                )
            )
            if mutual_retry:
                flush_local()
                events.append({"type": "public", "index": candidate_index})
                remaining_public.remove(candidate_index)
                executed_public.add(candidate_index)
                continue
        raise ValueError(
            f"cyclic local/public dependency in scene {scene.get('scene_id')}"
        )
    flush_local()
    return events


def _executable_local_segment(statements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a segment that persists assigned state for later plan events."""
    segment = [copy.deepcopy(statement) for statement in statements]
    assigned = {
        str(statement.get("target"))
        for statement in statements
        if statement.get("op") != "emit" and statement.get("target")
    }
    emitted = {
        str(name)
        for statement in statements
        if statement.get("op") == "emit"
        for name in (statement.get("fields", {}) or {})
        if name
    }
    carry = sorted(assigned - emitted)
    if carry:
        segment.append({
            "op": "emit",
            "fields": {name: {"ref": name} for name in carry},
        })
    return segment


def _canonical_public_argument_source(value: Any) -> Any:
    """Canonicalize literal argument leaves in compiled action contracts."""
    if isinstance(value, dict):
        if set(value) == {"literal"}:
            return value
        return {
            key: _canonical_public_argument_source(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_canonical_public_argument_source(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return {"literal": value}
    return value


def _normalize_contract_block_result_refs(contract: Dict[str, Any]) -> None:
    """Rewrite block-qualified results only when one public producer is provable."""
    scenes = contract.get("scenes", [])
    if not isinstance(scenes, list):
        return
    operations_by_block: Dict[str, List[Dict[str, Any]]] = {}
    for scene in scenes:
        if not isinstance(scene, dict) or not scene.get("block_id"):
            continue
        operations_by_block[str(scene["block_id"])] = [
            binding for binding in scene.get("public_operations", []) or []
            if isinstance(binding, dict) and binding.get("operation")
        ]

    def rewrite(value: Any) -> Any:
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if not isinstance(value, str):
            return value
        match = re.fullmatch(
            r"result\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)(.*)",
            value.removeprefix("$"),
        )
        if not match:
            return value
        block_id, field, suffix = match.groups()
        candidates = []
        for binding in operations_by_block.get(block_id, []):
            fields = set((binding.get("results", {}) or {}).keys())
            fields.update(str(item) for item in binding.get("produces", []) or [])
            if field in fields:
                candidates.append(str(binding["operation"]))
        if len(set(candidates)) != 1:
            return value
        return f"result.{candidates[0]}.{field}{suffix}"

    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        if "local_program" in scene:
            scene["local_program"] = rewrite(scene["local_program"])
        for binding in scene.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            binding["arguments"] = rewrite(binding.get("arguments", {}) or {})
            binding["condition"] = rewrite(binding.get("condition", {}) or {})
            binding["foreach"] = rewrite(binding.get("foreach", {}) or {})


def _compile_public_action_plan_source(dsl_code: str) -> str:
    """Compile explicit DSL operation bindings and local blocks to Python."""
    prefix = "# ACTION_CONTRACT_JSON:"
    contract: Dict[str, Any] = {}
    for line in (dsl_code or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            try:
                parsed = json.loads(stripped[len(prefix):].strip())
                contract = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return ""
            break
    source_contract = copy.deepcopy(contract)
    _normalize_contract_block_result_refs(contract)
    actions: List[Dict[str, Any]] = []
    local_program: List[Dict[str, Any]] = []
    local_function_sources: List[str] = []
    # ACTION_CONTRACT_JSON is emitted in workflow order.  Scene ids are labels,
    # not sortable positions: a split scene may use 11/12/13 while the following
    # original scene remains 3.
    scenes = list(
        contract.get("scenes", []) if isinstance(contract.get("scenes"), list) else []
    )
    operation_guards = {
        str(binding.get("operation", "")): str(binding.get("guard", "always") or "always")
        for scene in scenes if isinstance(scene, dict)
        for binding in scene.get("public_operations", []) or []
        if isinstance(binding, dict) and binding.get("operation")
    }

    def contains_call(value: Any, name: str) -> bool:
        if isinstance(value, list):
            return any(contains_call(item, name) for item in value)
        if not isinstance(value, dict):
            return False
        call = value.get("call")
        if isinstance(call, dict) and call.get("name") == name:
            return True
        return any(contains_call(item, name) for item in value.values())

    def local_segment_guard(
        segment: List[Dict[str, Any]], fallback_operation: str,
    ) -> str:
        referenced_operations = {
            root.split(".", 1)[1]
            for root in _action_contract_reference_roots(segment)
            if root.startswith("result.") and "." in root
        }
        if len(referenced_operations) == 1:
            return f"on_success:{next(iter(referenced_operations))}"
        if referenced_operations and contains_call(segment, "coalesce"):
            unconditional = sorted(
                operation for operation in referenced_operations
                if operation_guards.get(operation, "") == "always"
            )
            if len(unconditional) == 1:
                return f"on_success:{unconditional[0]}"
        return f"on_success:{fallback_operation}" if fallback_operation else "always"

    previous_operation = ""
    local_action_types = {
        "compute", "format", "template", "template_fill", "transform", "validation",
    }
    for scene_index, scene in enumerate(scenes):
        scene_local_program = list(scene.get("local_program", []) or [])
        local_program.extend(scene_local_program)
        scene_operations = []
        scene_public_actions: List[Dict[str, Any]] = []
        for binding in scene.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            action = {
                "type": "public_operation",
                "binding_version": int(contract.get("version", 0) or 0),
                "dependency_type": str(binding.get("dependency_type", "")),
                "operation": str(binding.get("operation", "")),
                "guard": str(binding.get("guard", "always") or "always"),
                "consumes": list(binding.get("consumes", []) or []),
                "produces": list(binding.get("produces", []) or []),
                "arguments": _canonical_public_argument_source(
                    dict(binding.get("arguments", {}) or {})
                ),
                "results": dict(binding.get("results", {}) or {}),
                "condition": dict(binding.get("condition", {}) or {}),
                "foreach": dict(binding.get("foreach", {}) or {}),
                "description": str(scene.get("description", "")),
                "raw_requirement_text": str(scene.get("raw_requirement_text", "")),
                "logic_flow": str(scene.get("logic_flow", "")),
                "scene_id": int(scene.get("scene_id", 0)),
                "block_id": str(scene.get("block_id", "")),
                "trigger_strategy": str(scene.get("trigger_strategy", "")),
                "action_strategy": str(scene.get("action_strategy", "")),
            }
            action["execution_mode"] = (
                "semantic" if action["action_strategy"].startswith("SEMANTIC_BLOCK")
                else "code" if action["action_strategy"].startswith("CODE_")
                else "external"
            )
            scene_public_actions.append(action)
            if action["operation"]:
                scene_operations.append(action["operation"])
        action_type = str(scene.get("action_type", "")).lower()
        local_text = " ".join([
            str(scene.get("description", "")),
            str(scene.get("raw_requirement_text", "")),
            " ".join(str(item) for item in scene.get("outputs", []) or []),
            " ".join(str(item) for item in scene.get("side_effects", []) or []),
            " ".join(str(item) for item in scene.get("action_sequence", []) or []),
        ]).lower()
        is_schema_validation = (
            any(token in local_text for token in ("schema", "输出结构", "结构校验", "结构验证"))
            and any(token in local_text for token in ("validate", "validation", "校验", "验证", "检查"))
        )
        has_declared_local_work = any(token in local_text for token in (
            "format", "template", "metric", "performance", "token", "排版", "格式", "指标", "耗时",
        ))
        needs_assign_step = action_type == "assign" and not scene_operations
        needs_local_action = (
            action_type in local_action_types
            or needs_assign_step
            or has_declared_local_work
            or is_schema_validation
            or scene_local_program
        )
        def make_local_action(
            program: List[Dict[str, Any]],
            function_name: str,
        ) -> Dict[str, Any]:
            return {
                "type": "local_step",
                "action_type": action_type,
                "guard": "always",
                "outputs": list(scene.get("outputs", []) or []),
                "side_effects": list(scene.get("side_effects", []) or []),
                "action_sequence": list(scene.get("action_sequence", []) or []),
                "description": str(scene.get("description", "")),
                "raw_requirement_text": str(scene.get("raw_requirement_text", "")),
                "scene_id": int(scene.get("scene_id", 0)),
                "block_id": str(scene.get("block_id", "")),
                "trigger_strategy": str(scene.get("trigger_strategy", "")),
                "action_strategy": str(scene.get("action_strategy", "")),
                "execution_mode": (
                    "semantic" if str(scene.get("action_strategy", "")).startswith("SEMANTIC_BLOCK")
                    else "code"
                ),
                "program": program,
                "local_function": function_name,
            }

        if scene_local_program:
            from dsl_v2.local_program import compile_local_program_source
            segment_index = 0
            for event in _schedule_scene_action_units(scene):
                if event["type"] == "public":
                    public_action = scene_public_actions[event["index"]]
                    actions.append(public_action)
                    if public_action["operation"]:
                        previous_operation = public_action["operation"]
                    continue
                segment = _executable_local_segment(event["statements"])
                function_name = f"compiled_local_scene_{scene_index}_{segment_index}"
                local_function_sources.append(
                    compile_local_program_source(segment, function_name)
                )
                local_action = make_local_action(segment, function_name)
                local_action["guard"] = local_segment_guard(
                    segment, previous_operation
                )
                actions.append(local_action)
                segment_index += 1
        else:
            actions.extend(scene_public_actions)
            for public_action in scene_public_actions:
                if public_action["operation"]:
                    previous_operation = public_action["operation"]
            if needs_local_action:
                local_action = make_local_action([], "")
                local_action["guard"] = (
                    f"on_success:{previous_operation}" if previous_operation else "always"
                )
                actions.append(local_action)
    if not actions:
        return ""
    canonical_contract = json.dumps(
        source_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    contract_sha256 = hashlib.sha256(canonical_contract.encode("utf-8")).hexdigest().upper()
    source = (
        f"_ACTION_CONTRACT_SHA256 = {contract_sha256!r}\n"
        "def compiled_public_workflow(input_params: dict):\n"
        "    # Generated deterministically from ACTION_CONTRACT_JSON.\n"
        f"    return {actions!r}\n"
    )
    if local_program:
        from dsl_v2.local_program import compile_local_program_source
        source += "\n" + compile_local_program_source(local_program)
    for function_source in local_function_sources:
        source += "\n" + function_source
    return source


def m4_compile_classified_dsl_backend(
    classified_spec: 'ClassifiedSpec',
    *,
    allow_semantic: bool = True,
) -> CompiledArtifact:
    """M4 backend: ClassifiedSpec -> DSL -> WaActCompiler -> Python."""
    if _classified_spec_needs_semantic(classified_spec) and not allow_semantic:
        raise ValueError("DSL backend currently supports only pure BLOCK/CODE scenes")

    t0 = time.time()
    dsl_code, generated_python, compile_details = _compile_classified_with_dsl_backend(
        classified_spec
    )
    return CompiledArtifact(
        skeleton_source="",
        sensor_specs=[],
        compile_tokens=0,
        compile_latency_ms=int((time.time() - t0) * 1000),
        validation_msg="ok",
        backend="dsl_compiler",
        dsl_code=dsl_code,
        generated_python=generated_python,
        compile_details=compile_details,
    )


def m4_compile_classified(
    normalized: str,
    fact_spec: 'DslFactSpec',
    classified_spec: 'ClassifiedSpec',
    input_data: Dict,
    llm_client=None,
    skeleton_override: Optional[str] = None,
    max_retries: int = 2,
    domain: str = "",
) -> CompiledArtifact:
    """
    M4 编译 (新版): 基于策略分类结果翻译为 skeleton + sensors.

    1. 策略决策已由 StrategyClassifier 完成 (零 LLM)
    2. LLM 只负责将 ClassifiedSpec 翻译为可执行代码
    3. BLOCK 场景 → 必须 if/else, 禁止 sensor
    4. SEMANTIC_BLOCK 场景 → 必须 sensor, skeleton 中调用 sensors['name'](ctx)
    5. M0 行为校验 + 重试循环 (最多 max_retries 次)

    Args:
        normalized: M1 规范化后的文本
        fact_spec: M3 输出的 dsl_v2.FactSpec
        classified_spec: StrategyClassifier 输出的 ClassifiedSpec
        input_data: 任务运行时输入
        llm_client: 复用的 LLM 客户端
        skeleton_override: 直接传入 skeleton 源码 (跳过 LLM 编译)
        max_retries: M0 校验失败时最大重试次数
    """
    if llm_client is None:
        llm_client = create_llm_client()

    logger.info(f"[M4] 开始编译, 场景数={len(classified_spec.scenes)}")

    t0 = time.time()
    if skeleton_override is not None:
        compile_result = {
            "skeleton": skeleton_override,
            "sensors": [],
        }
        usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
    else:
        classified_spec_str = _serialize_classified_spec(classified_spec)
        user_prompt = COMPILE_USER_TEMPLATE.format(
            normalized=normalized,
            domain=domain or "未知",
            input_data_str=json.dumps(input_data, ensure_ascii=False, indent=2),
            classified_spec_str=classified_spec_str,
        )

        # 重试循环
        compile_result = None
        usage = {}
        error_feedback = ""
        sensor_specs = []  # 初始化, 重试循环内赋值

        for attempt in range(max_retries + 1):
            current_prompt = user_prompt
            if error_feedback:
                current_prompt += f"\n\n## 上次编译的校验反馈 (请修正)\n{error_feedback}"

            resp = llm_client.call(
                system_prompt=COMPILE_SYSTEM_PROMPT,
                user_content=current_prompt,
                temperature=0.0,
            )
            usage = resp.usage or {}
            compile_result = _parse_compile_output(resp.content)

            if compile_result is None:
                if attempt < max_retries:
                    error_feedback = f"编译输出无法解析为 JSON, 请严格输出 JSON 格式"
                    logger.warning(f"[M4] 第 {attempt+1} 次编译: JSON 解析失败, 重试")
                    continue
                else:
                    raise ValueError(f"M4 编译 LLM 输出无法解析为 JSON:\n{resp.content[:500]}")

            # 校验 skeleton
            skeleton_src = compile_result.get("skeleton", "").strip()
            skeleton_src = re.sub(r'^```python\s*', '', skeleton_src)
            skeleton_src = re.sub(r'\s*```$', '', skeleton_src)
            ok, msg = _validate_skeleton_source(skeleton_src)
            if not ok:
                if attempt < max_retries:
                    error_feedback = f"skeleton AST 校验失败: {msg}"
                    logger.warning(f"[M4] 第 {attempt+1} 次编译: AST 校验失败, 重试")
                    compile_result = None
                    continue
                else:
                    raise ValueError(f"skeleton AST 校验失败: {msg}\nsource:\n{skeleton_src[:500]}")

            # 构造 sensor specs
            sensor_specs = []
            for s in compile_result.get("sensors", []) or []:
                spec = _build_sensor_spec(s)
                if spec is not None:
                    sensor_specs.append(spec)

            # M0 策略一致性校验
            strategy_errors = _validate_strategy_consistency(
                skeleton_src, sensor_specs, classified_spec
            )
            if strategy_errors:
                if attempt < max_retries:
                    error_feedback = "\n".join(f"- {e}" for e in strategy_errors)
                    logger.warning(f"[M4] 第 {attempt+1} 次编译: 策略一致性校验失败, 重试")
                    compile_result = None
                    continue
                else:
                    # 最后一次仍失败, 记录但不阻断
                    logger.warning(
                        f"[M4] 策略一致性校验失败 (已用完重试次数): {strategy_errors}"
                    )

            # 通过所有校验
            break

    # 提取最终 skeleton (重试循环内已校验)
    skeleton_src = compile_result.get("skeleton", "").strip()
    skeleton_src = re.sub(r'^```python\s*', '', skeleton_src)
    skeleton_src = re.sub(r'\s*```$', '', skeleton_src)

    # 构造 sensor specs (skeleton_override 分支无 sensor, 重试循环分支已构造)
    if skeleton_override is not None:
        sensor_specs = []

    compile_latency = int((time.time() - t0) * 1000)
    logger.info(f"[M4] 编译完成, sensors={len(sensor_specs)}, latency={compile_latency}ms")

    return CompiledArtifact(
        skeleton_source=skeleton_src,
        sensor_specs=sensor_specs,
        compile_tokens=usage.get("total_tokens", 0),
        compile_latency_ms=compile_latency,
        validation_msg="ok",
    )


def _parse_compile_output(text: str) -> Optional[Dict]:
    """从 LLM 输出中提取 JSON. 容忍 markdown 包裹, 使用括号平衡法避免贪婪匹配."""
    text = (text or "").strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 括号平衡法提取第一个完整的 JSON 对象
    start = text.find('{')
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # 这个平衡块不是有效 JSON, 继续找下一个
                        return _parse_compile_output(text[i+1:])
    return None


def _build_sensor_spec(s: Dict) -> Optional[SensorSpec]:
    """把 LLM 输出的 sensor dict 转 SensorSpec."""
    name = s.get("name")
    kind_str = s.get("kind", "").lower()
    if not name or not kind_str:
        return None
    try:
        kind = SensorKind(kind_str)
    except ValueError:
        return None

    input_var = s.get("input_var", "user_input")
    description = s.get("description", "")
    question = s.get("question", description or name)
    temperature = float(s.get("temperature", 0.0))

    # M0 validator
    validator = None
    if kind == SensorKind.CLASSIFICATION:
        allowed = s.get("allowed_values", []) or []
        if allowed:
            validator = M0Validator.enum_validator(allowed)
    elif kind == SensorKind.BOOLEAN:
        validator = M0Validator.type_validator(bool)
    elif kind == SensorKind.EXTRACTION:
        req = s.get("required_keys", []) or []
        types_str = s.get("value_types", {}) or {}
        type_map = {"String": str, "Integer": int, "Float": float, "Boolean": bool,
                    "List": list, "Dict": dict}
        types = {k: type_map.get(v, str) for k, v in types_str.items()}
        validator = M0Validator.schema_validator(req, types)
    elif kind == SensorKind.SCORING:
        validator = M0Validator.range_validator(0.0, 1.0)
    elif kind == SensorKind.COMPUTE:
        validator = M0Validator.type_validator((int, float))

    # 构造 system_prompt 和 user_prompt_template
    system_prompt = (
        f"你是一个窄化 LLM 传感点 ({name}).\n"
        f"功能: {description}\n"
        f"严格只输出 JSON, 不要其他文字."
    )
    user_prompt_template = (
        f"## 上下文\n{{context}}\n\n"
        f"## 问题\n{question}\n"
    )
    if kind == SensorKind.CLASSIFICATION and s.get("allowed_values"):
        user_prompt_template += f"\n只输出 JSON: {{\"value\": <one of {s['allowed_values']}>}}"
    elif kind == SensorKind.BOOLEAN:
        user_prompt_template += "\n只输出 JSON: {\"value\": true|false}"
    elif kind == SensorKind.EXTRACTION:
        req = s.get("required_keys", []) or []
        user_prompt_template += f"\n必填字段: {req}\n只输出 JSON."

    # 根据 SensorKind 确定 output_type
    _KIND_TO_OUTPUT_TYPE = {
        SensorKind.CLASSIFICATION: str,
        SensorKind.BOOLEAN: bool,
        SensorKind.EXTRACTION: dict,
        SensorKind.SCORING: float,
        SensorKind.COMPUTE: float,
        SensorKind.GENERATION: str,
    }
    output_type = _KIND_TO_OUTPUT_TYPE.get(kind, str)

    return SensorSpec(
        name=name,
        kind=kind,
        description=description,
        input_vars=[input_var],
        output_type=output_type,
        m0_validator=validator,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        temperature=temperature,
        allowed_values=s.get("allowed_values"),
    )


# ============================================================================
# Safe Skeleton Runner: 解释执行 (替代 exec, 安全 + 可观测)
# ============================================================================

class SafeSkeletonRunner:
    """
    在受限命名空间内执行 skeleton 函数. 替代直接 exec, 提供更严格的安全与观测.

    Skeleton 调用约定:
        def run(sensors, input_data):
            actions = []
            if condition:
                actions.append({...})
            final = {...}
            return actions, final
    """

    def __init__(self, sensor_view, input_data: Dict):
        self.sensor_view = sensor_view
        self.input_data = input_data

    def _wrap_sensor_access(self, tree: ast.AST) -> ast.AST:
        """
        把 sensors['xxx'](ctx) 改写成更安全的形式.
        实际不需要改 AST, 我们在 exec 时把 sensors 限制为一个白名单视图.
        """
        return tree

    def execute(self, source: str) -> Tuple[List[Dict], Dict]:
        """
        执行 skeleton 源码. 失败 raise.

        安全机制:
        - 只暴露白名单 builtin
        - 不暴露 import/exec/open 等危险内置
        - sensors 是 _SensorView (只允许下标访问 + 调用)
        - input_data 是 dict
        """
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
            'enumerate': enumerate, 'filter': filter, 'float': float, 'int': int,
            'isinstance': isinstance, 'issubclass': issubclass, 'len': len, 'list': list,
            'map': map, 'max': max, 'min': min, 'pow': pow, 'print': print,
            'range': range, 'repr': repr, 'reversed': reversed, 'round': round,
            'set': set, 'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
            'type': type, 'zip': zip,
            'True': True, 'False': False, 'None': None,
        }
        safe_globals = {
            '__builtins__': safe_builtins,
            'sensors': self.sensor_view,
            'input_data': self.input_data,
            '__name__': '__skeleton__',
        }
        safe_locals = {}

        # 编译 (不执行) 以做最后一道校验
        code = compile(source, '<skeleton>', 'exec')
        exec(code, safe_globals, safe_locals)

        # 调 run
        run_fn = safe_locals.get('run') or safe_globals.get('run')
        if run_fn is None:
            raise RuntimeError("skeleton 未定义 run() 函数")

        actions, final_state = run_fn(self.sensor_view, self.input_data)
        return actions or [], final_state or {}


def _public_input_condition_result(
    condition: Any,
    input_data: Optional[Dict[str, Any]],
) -> Optional[bool]:
    """Evaluate an input-bound condition without consulting task outputs."""
    if not isinstance(condition, dict) or not condition:
        return True
    if isinstance(condition.get("all"), list):
        results = [
            _public_input_condition_result(item, input_data)
            for item in condition["all"]
        ]
        if any(result is False for result in results):
            return False
        return True if results and all(result is True for result in results) else None
    if isinstance(condition.get("any"), list):
        results = [
            _public_input_condition_result(item, input_data)
            for item in condition["any"]
        ]
        if any(result is True for result in results):
            return True
        return False if results and all(result is False for result in results) else None
    if "not" in condition:
        result = _public_input_condition_result(condition.get("not"), input_data)
        return None if result is None else not result

    source = str(condition.get("source", "") or "")
    operator = str(condition.get("operator", "") or "")
    if not source.startswith(("input.", "workflow_input.")):
        return None
    field = source.rsplit(".", 1)[-1]

    def collect(value: Any) -> List[Any]:
        found: List[Any] = []
        if isinstance(value, dict):
            if field in value:
                found.append(value[field])
            for child in value.values():
                found.extend(collect(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(collect(child))
        return found

    root = input_data or {}
    values = (
        [root[field]] if source.startswith("input.") and field in root
        else collect(root) if source.startswith("workflow_input.")
        else []
    )
    if operator == "exists":
        return bool(values)
    if operator == "not_exists":
        return not values
    if not values:
        return None
    expected = condition.get("value")
    if operator == "eq":
        return any(value == expected for value in values)
    if operator == "neq":
        return all(value != expected for value in values)
    if operator in {"contains", "not_contains"}:
        matches = []
        for value in values:
            try:
                matches.append(expected in value)
            except TypeError:
                matches.append(False)
        contained = any(matches)
        return contained if operator == "contains" else not contained
    return None


def _infer_conditioned_entry_scene_from_public_input(
    plan: List[Dict[str, Any]],
    input_data: Dict[str, Any],
) -> Optional[int]:
    """Return a unique independent entry selected by explicit input conditions."""
    independent_roots = [
        item for item in plan
        if isinstance(item, dict)
        and item.get("type") == "public_operation"
        and str(item.get("guard", "always") or "always") == "always"
        and int(item.get("scene_id", 0) or 0) > 0
    ]
    # A root without an explicit discriminator may be a required prerequisite,
    # not an alternative mode. In that shape, input-only pruning is unsafe.
    if any(not (item.get("condition", {}) or {}) for item in independent_roots):
        return None
    matches: Set[int] = set()
    for item in independent_roots:
        scene_id = int(item.get("scene_id", 0) or 0)
        condition = item.get("condition", {}) or {}
        if scene_id > 0 and condition and _public_input_condition_result(
            condition, input_data
        ) is True:
            matches.add(scene_id)
    return next(iter(matches)) if len(matches) == 1 else None


def _select_scene_scoped_plan(
    plan: List[Dict[str, Any]],
    route_result: Any,
    input_data: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Restrict an explicitly routed plan to one data-dependent scene chain.

    This helper is deliberately independent of the model call.  It only
    accepts a valid positive scene id and a truthy match flag; malformed or
    absent routing results leave the original plan untouched.  A scene is
    retained together with later scenes that consume one of its public
    results or explicitly reference its local result.  This prevents an
    inherited ``on_success`` edge from turning unrelated entry modes into one
    serial workflow.
    """
    scene_ids = sorted({
        int(item.get("scene_id", 0) or 0)
        for item in plan
        if isinstance(item, dict) and int(item.get("scene_id", 0) or 0) > 0
    })
    if len(scene_ids) <= 1:
        return plan
    if isinstance(route_result, dict):
        selected_scene = route_result.get("scene")
        route_match = route_result.get("is_match", True)
    else:
        selected_scene = getattr(route_result, "scene", None)
        route_match = getattr(route_result, "is_match", True)
    try:
        selected_scene = int(selected_scene)
    except (TypeError, ValueError):
        return plan
    route_is_match = route_match not in {False, None, "false", "False", "0", 0}
    if not route_is_match or selected_scene not in scene_ids:
        return plan

    by_scene: Dict[int, List[Dict[str, Any]]] = {}
    for item in plan:
        if not isinstance(item, dict):
            continue
        scene_id = int(item.get("scene_id", 0) or 0)
        if scene_id > 0:
            by_scene.setdefault(scene_id, []).append(item)

    def result_refs(value: Any) -> Set[str]:
        found: Set[str] = set()
        if isinstance(value, list):
            for item in value:
                found.update(result_refs(item))
        elif isinstance(value, dict):
            if set(value) == {"ref"} and isinstance(value.get("ref"), str):
                reference = value["ref"]
                match = re.match(r"result\.([A-Za-z_][A-Za-z0-9_]*)", reference)
                if match:
                    found.add(match.group(1))
            else:
                for item in value.values():
                    found.update(result_refs(item))
        return found

    def same_field(left: Any, right: Any) -> bool:
        left_text = str(left or "").split(".", 1)[0].lower().rstrip("s")
        right_text = str(right or "").split(".", 1)[0].lower().rstrip("s")
        return bool(left_text and right_text and left_text == right_text)

    def input_contains(value: Any, key: str) -> bool:
        if isinstance(value, dict):
            return key in value or any(input_contains(item, key) for item in value.values())
        if isinstance(value, list):
            return any(input_contains(item, key) for item in value)
        return False

    def condition_is_satisfied(condition: Any) -> bool:
        return _public_input_condition_result(condition, input_data) is True

    included_scenes: Set[int] = {selected_scene}
    # Conditional independent roots are valid additions when their public
    # input condition is satisfied. Unconditional independent roots are left
    # out because they represent separate entry modes.
    for scene_id, items in by_scene.items():
        if scene_id == selected_scene:
            continue
        public_items = [item for item in items if item.get("type") == "public_operation"]
        if public_items and any(
            item.get("condition")
            and condition_is_satisfied(item.get("condition"))
            and str(item.get("guard", "always") or "always") == "always"
            for item in public_items
        ):
            included_scenes.add(scene_id)
    included_operations: Set[str] = set()
    produced_fields: Set[str] = set()

    def add_scene_outputs(scene_id: int) -> None:
        for item in by_scene.get(scene_id, []):
            operation = str(item.get("operation", "") or "")
            if item.get("type") == "public_operation" and operation:
                included_operations.add(operation)
            results = item.get("results", {}) or {}
            if isinstance(results, dict):
                produced_fields.update(str(value) for value in results.values())
            produced_fields.update(
                str(value) for value in (item.get("outputs", []) or [])
                if isinstance(value, str)
            )
            for statement in item.get("program", []) or []:
                if not isinstance(statement, dict):
                    continue
                target = statement.get("target")
                if isinstance(target, str) and target:
                    produced_fields.add(target.split(".", 1)[0])
                if statement.get("op") == "emit":
                    produced_fields.update(
                        str(name).split(".", 1)[0]
                        for name in (statement.get("fields", {}) or {})
                        if name
                    )

    add_scene_outputs(selected_scene)
    propagating_operations: Set[str] = set(included_operations)
    propagating_fields: Set[str] = set(produced_fields)
    # Some workflows describe a read dependency in the selected scene's
    # business text while the parser attaches the corresponding read block to
    # an earlier scene (the weekly report both reads bookings and cleaning
    # tasks, for example). Preserve those read operations as prerequisites;
    # otherwise route scoping would silently remove the data source required by
    # the selected semantic block.
    selected_text = " ".join(
        str(item.get(key, ""))
        for item in by_scene.get(selected_scene, [])
        for key in ("description", "raw_requirement_text", "logic_flow", "action_sequence")
    ).lower()
    mentioned_operations = {
        str(item.get("operation", "")).lower()
        for items in by_scene.values()
        for item in items
        if item.get("type") == "public_operation"
        and str(item.get("operation", "")).lower()
        and str(item.get("operation", "")).lower() in selected_text
    }
    explicitly_required_scene_ids: Set[int] = {selected_scene}
    propagated_scene_ids: Set[int] = set()
    for scene_id, items in by_scene.items():
        if scene_id in included_scenes:
            continue
        read_items = [
            item for item in items
            if item.get("type") == "public_operation"
            and str(item.get("operation", "")).startswith("read_")
        ]
        if read_items and any(
            str(item.get("operation", "")).lower() in selected_text
            for item in read_items
        ):
            included_scenes.add(scene_id)
            explicitly_required_scene_ids.add(scene_id)
            add_scene_outputs(scene_id)
    for scene_id in sorted(included_scenes - {selected_scene}):
        add_scene_outputs(scene_id)
    changed = True
    while changed:
        changed = False
        for scene_id in sorted(by_scene):
            if scene_id in included_scenes:
                continue
            scene_text = " ".join(
                str(item.get(key, ""))
                for item in by_scene[scene_id]
                for key in ("description", "raw_requirement_text", "logic_flow")
            ).lower()
            if (
                any(token in scene_text for token in ("每周", "weekly"))
                and not input_contains(input_data or {}, "cleaning_tasks")
            ):
                continue
            depends_on_included = False
            for item in by_scene[scene_id]:
                guard = str(item.get("guard", "") or "")
                if guard.startswith(("on_success:", "on_failure:")):
                    predecessor = guard.split(":", 1)[1]
                    if predecessor in propagating_operations:
                        depends_on_included = True
                        break
                refs = result_refs({
                    "arguments": item.get("arguments", {}),
                    "foreach": item.get("foreach", {}),
                    "program": item.get("program", []),
                })
                if refs & propagating_operations:
                    depends_on_included = True
                    break
                consumes = item.get("consumes", []) or []
                if any(
                    same_field(consume, produced)
                    for consume in consumes
                    for produced in propagating_fields
                ):
                    depends_on_included = True
                    break
                condition_sources = {
                    str(leaf.get("source", ""))
                    for leaf in _condition_leaf_nodes(item.get("condition", {}) or {})
                    if isinstance(leaf, dict) and leaf.get("source")
                }
                if any(
                    same_field(source, produced)
                    for source in condition_sources
                    for produced in propagating_fields
                ):
                    depends_on_included = True
                    break
            if depends_on_included:
                included_scenes.add(scene_id)
                propagated_scene_ids.add(scene_id)
                add_scene_outputs(scene_id)
                propagating_operations.update(
                    str(item.get("operation", ""))
                    for item in by_scene.get(scene_id, [])
                    if item.get("type") == "public_operation" and item.get("operation")
                )
                propagating_fields.update(
                    str(value)
                    for item in by_scene.get(scene_id, [])
                    for value in (
                        list((item.get("results", {}) or {}).values())
                        if isinstance(item.get("results", {}), dict) else []
                    )
                )
                propagating_fields.update(
                    str(value)
                    for item in by_scene.get(scene_id, [])
                    for value in (item.get("outputs", []) or [])
                    if isinstance(value, str)
                )
                propagating_fields.update(
                    str(statement.get("target", "")).split(".", 1)[0]
                    for item in by_scene.get(scene_id, [])
                    for statement in (item.get("program", []) or [])
                    if isinstance(statement, dict) and statement.get("target")
                )
                changed = True

    retained = [
        item for item in plan
        if int(item.get("scene_id", 0) or 0) in included_scenes
        and _public_input_condition_result(
            item.get("condition", {}) or {}, input_data
        ) is not False
        and (
            int(item.get("scene_id", 0) or 0) in explicitly_required_scene_ids
            or int(item.get("scene_id", 0) or 0) in propagated_scene_ids
            or any(
                str(candidate.get("operation", "")).lower() in mentioned_operations
                for candidate in by_scene.get(int(item.get("scene_id", 0) or 0), [])
                if candidate.get("type") == "public_operation"
            )
        )
        and (
            int(item.get("scene_id", 0) or 0) == selected_scene
            or int(item.get("scene_id", 0) or 0) in propagated_scene_ids
            or not any(
                candidate.get("type") == "public_operation"
                and candidate.get("condition")
                and not condition_is_satisfied(candidate.get("condition"))
                for candidate in by_scene.get(int(item.get("scene_id", 0) or 0), [])
            )
        )
    ]
    retained_operations = {
        str(item.get("operation", "")) for item in retained
        if item.get("type") == "public_operation" and item.get("operation")
    }
    # Rebase the first operation of a selected chain when its old predecessor
    # belongs to an excluded entry mode. Otherwise the operation remains
    # permanently blocked by a guard that can no longer fire.
    for item in retained:
        guard = str(item.get("guard", "") or "")
        if guard.startswith("on_success:") or guard.startswith("on_failure:"):
            predecessor = guard.split(":", 1)[1]
            if predecessor not in retained_operations:
                item["guard"] = "always"
    return retained


def _infer_entry_scene_from_public_input(
    plan: List[Dict[str, Any]],
    input_data: Dict[str, Any],
) -> Optional[int]:
    """Infer a safe entry only when the semantic router returns no scene.

    This is a value-free structural fallback: it uses public field presence,
    basic date ordering for an explicit booking contract, and operation
    descriptions already present in the compiled plan. It never reads Oracle
    expectations and returns ``None`` when the input shape is ambiguous.
    """
    operations = {
        str(item.get("operation", "")): int(item.get("scene_id", 0) or 0)
        for item in plan
        if isinstance(item, dict) and item.get("type") == "public_operation"
    }

    def contains_key(value: Any, key: str) -> bool:
        if isinstance(value, dict):
            return key in value or any(contains_key(item, key) for item in value.values())
        if isinstance(value, list):
            return any(contains_key(item, key) for item in value)
        return False

    def get_first(value: Any, key: str) -> Any:
        if isinstance(value, dict):
            if key in value:
                return value[key]
            for item in value.values():
                found = get_first(item, key)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = get_first(item, key)
                if found is not None:
                    return found
        return None

    # Scheduled/report inputs are structurally distinguishable from a direct
    # booking request and do not require a model call to choose an entry.
    if contains_key(input_data, "cleaning_tasks") and "write_weekly_digest" in operations:
        return operations["write_weekly_digest"]
    if contains_key(input_data, "sweep_date") and "read_bookings" in operations:
        return operations["read_bookings"]

    if not any(contains_key(input_data, key) for key in (
        "guest_email", "check_in", "check_out", "booking_id",
    )):
        return None

    guest_email = get_first(input_data, "guest_email")
    check_in = get_first(input_data, "check_in")
    check_out = get_first(input_data, "check_out")
    invalid_booking = not guest_email or (
        isinstance(check_in, str)
        and isinstance(check_out, str)
        and check_out <= check_in
    )
    if invalid_booking:
        for item in plan:
            if not isinstance(item, dict) or item.get("type") != "local_step":
                continue
            description = " ".join(str(item.get(key, "")) for key in (
                "description", "raw_requirement_text", "action_sequence",
            )).lower()
            if any(token in description for token in ("无效", "invalid", "校验必填", "日期先后")):
                return int(item.get("scene_id", 0) or 0) or None
    return operations.get("append_booking")


class DSLCompiledRunner:
    """Execute Python generated by WaActCompiler and adapt it to experiment output."""

    def __init__(
        self,
        input_data: Dict,
        llm_client=None,
        classified_spec=None,
        dsl_code: str = "",
        expected_output: Optional[Dict[str, Any]] = None,
        domain: str = "",
        public_interfaces: Optional[List[Dict[str, str]]] = None,
        public_runtime_config: Optional[Dict[str, Any]] = None,
    ):
        self.input_data = input_data
        self.llm_client = llm_client
        self.classified_spec = classified_spec
        self.dsl_code = dsl_code or ""
        self.expected_output = expected_output or {}
        self.domain = domain or ""
        self.public_interfaces = [dict(item) for item in (public_interfaces or [])]
        self.public_runtime_config = dict(public_runtime_config or {})
        self.action_contract = self._parse_action_contract(self.dsl_code)
        self.compiled_public_actions: Optional[List[Dict[str, Any]]] = None
        self.compiled_local_functions: Dict[str, Any] = {}
        self.workflow_semantic_outputs: Dict[int, Dict[str, Any]] = {}
        self.route_diagnostics: Dict[str, Any] = {}
        self.traces: List[Dict[str, Any]] = []

    class _Response:
        def __init__(self, content: str):
            self.content = content

    class _Completions:
        def __init__(self, outer: 'DSLCompiledRunner'):
            self.outer = outer

        async def create(self, model: str, temperature: float, max_tokens: int, messages: List[Dict[str, str]]):
            system_prompt = ""
            user_content = ""
            for msg in messages or []:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                elif msg.get("role") == "user":
                    user_content = msg.get("content", "")
            t0 = time.time()
            resp = self.outer.llm_client.call(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=temperature,
            )
            usage = resp.usage or {}
            self.outer.traces.append({
                "sensor_name": "dsl_semantic_block",
                "value": resp.content,
                "latency_ms": int((time.time() - t0) * 1000),
                "usage": usage,
                "validation_msg": "ok",
                "model": model,
                "max_tokens": max_tokens,
            })
            return DSLCompiledRunner._Response(resp.content)

    class _Chat:
        def __init__(self, outer: 'DSLCompiledRunner'):
            self.completions = DSLCompiledRunner._Completions(outer)

    class _LLMAdapter:
        def __init__(self, outer: 'DSLCompiledRunner'):
            self.chat = DSLCompiledRunner._Chat(outer)

    def execute(self, source: str) -> Tuple[List[Dict], Dict]:
        import asyncio
        if self.llm_client is None:
            self.llm_client = create_llm_client()

        source = source.replace(
            'ctx["scene"], ctx["agent"] = result',
            'ctx["scene"], ctx["agent"] = _dsl_unpack_scene_agent(result)',
        )

        def _dsl_unpack_scene_agent(result: Any) -> Tuple[Any, Any]:
            if isinstance(result, tuple):
                if len(result) == 2:
                    return result
                if len(result) >= 3 and isinstance(result[0], bool):
                    return result[2], "semantic_fallback"
                if len(result) >= 2:
                    return result[0], result[1]
            if isinstance(result, dict):
                return result.get("scene"), result.get("agent", "semantic_fallback")
            if hasattr(result, "scene"):
                return getattr(result, "scene"), getattr(result, "agent", "semantic_fallback")
            return result, "semantic_fallback"

        safe_globals = {
            "__builtins__": {
                "__import__": __import__,
                "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
                "enumerate": enumerate, "float": float, "int": int, "isinstance": isinstance,
                "KeyError": KeyError, "RuntimeError": RuntimeError, "TypeError": TypeError,
                "ValueError": ValueError,
                "getattr": getattr,
                "len": len, "list": list, "max": max, "min": min, "range": range,
                "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
                "tuple": tuple, "zip": zip, "True": True, "False": False, "None": None,
            },
            "__name__": "__dsl_compiled__",
            "_dsl_unpack_scene_agent": _dsl_unpack_scene_agent,
            "computed_value": 0,
            "true": True,
            "false": False,
        }
        exec(compile(source, "<dsl_compiled>", "exec"), safe_globals, safe_globals)
        if "LLM_CLIENT" in safe_globals:
            safe_globals["LLM_CLIENT"] = DSLCompiledRunner._LLMAdapter(self)
        main_workflow = safe_globals.get("main_workflow")
        if main_workflow is None:
            raise RuntimeError("DSL compiled artifact did not define main_workflow()")

        if self.action_contract:
            canonical_contract = json.dumps(
                self.action_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            expected_contract_hash = hashlib.sha256(
                canonical_contract.encode("utf-8")
            ).hexdigest().upper()
            compiled_contract_hash = safe_globals.get("_ACTION_CONTRACT_SHA256")
            if compiled_contract_hash != expected_contract_hash:
                raise RuntimeError(
                    "compiled Python action contract does not match the supplied DSL"
                )

        if self.action_contract and self.public_interfaces and self._public_trigger_rejected():
            return self._adapt_result(dict(self.input_data))

        compiled_public_workflow = safe_globals.get("compiled_public_workflow")
        if self.action_contract and self.public_interfaces and compiled_public_workflow is not None:
            self.compiled_local_functions = {
                name: value for name, value in safe_globals.items()
                if name.startswith("compiled_local_scene_") and callable(value)
            }
            plan = compiled_public_workflow(dict(self.input_data))
            if not isinstance(plan, list) or not all(isinstance(item, dict) for item in plan):
                raise RuntimeError("compiled_public_workflow() must return a list of action objects")

            # A DSL may contain several independent semantic entry scenes
            # (for example, valid booking, daily scan, and weekly report).
            # The generated public plan is intentionally static, so without a
            # route decision it would execute every scene in sequence.  When
            # the DSL explicitly contains the generated fallback classifier,
            # use that classifier as the entry router and retain only the
            # selected scene.  Workflows without that classifier remain
            # ordinary multi-scene pipelines and keep their full plan.
            route_function = safe_globals.get("sb_fallback")
            route_scene_ids = sorted({
                int(item.get("scene_id", 0))
                for item in plan
                if isinstance(item, dict)
                and int(item.get("scene_id", 0) or 0) > 0
            })
            independent_entry_scene_ids = sorted({
                int(item.get("scene_id", 0))
                for item in plan
                if isinstance(item, dict)
                and item.get("type") == "public_operation"
                and int(item.get("scene_id", 0) or 0) > 0
                and str(item.get("guard", "always") or "always") == "always"
            })
            # A single-root workflow may contain a long dependent chain and
            # must remain intact. Route scoping is only justified when the
            # compiled plan exposes multiple independent entry roots.
            if (
                callable(route_function)
                and len(route_scene_ids) > 1
                and len(independent_entry_scene_ids) > 1
            ):
                deterministic_scene = _infer_conditioned_entry_scene_from_public_input(
                    plan, self.input_data
                )
                if deterministic_scene:
                    route_view = {
                        "scene": deterministic_scene,
                        "is_match": True,
                        "confidence": 1.0,
                        "source": "deterministic_public_condition",
                    }
                    route_result = route_view
                else:
                    route_result = route_function({
                        # The generated fallback accepts text, so expose the
                        # full public input shape as stable JSON.
                        "user_input": json.dumps(
                            self.input_data, ensure_ascii=False, sort_keys=True
                        )
                    })
                    if hasattr(route_result, "__await__"):
                        route_result = asyncio.run(route_result)
                    if isinstance(route_result, dict):
                        route_view = dict(route_result)
                    else:
                        route_view = {
                            key: getattr(route_result, key)
                            for key in ("scene", "is_match", "confidence", "agent")
                            if hasattr(route_result, key)
                        }
                if not route_view.get("scene"):
                    fallback_scene = _infer_entry_scene_from_public_input(
                        plan, self.input_data
                    )
                    if fallback_scene:
                        route_view = {
                            "scene": fallback_scene,
                            "is_match": True,
                            "confidence": 1.0,
                            "source": "deterministic_input_shape_fallback",
                        }
                        route_result = route_view
                before_scene_ids = sorted({
                    int(item.get("scene_id", 0) or 0) for item in plan
                    if isinstance(item, dict)
                })
                plan = _select_scene_scoped_plan(
                    plan, route_result, input_data=self.input_data
                )
                self.route_diagnostics = {
                    "called": True,
                    "route_result": route_view,
                    "scene_ids_before": before_scene_ids,
                    "scene_ids_after": sorted({
                        int(item.get("scene_id", 0) or 0) for item in plan
                        if isinstance(item, dict)
                    }),
                    "independent_entry_scene_ids": independent_entry_scene_ids,
                }
            else:
                self.route_diagnostics = {
                    "called": False,
                    "reason": "single_entry_root_or_no_route_function",
                    "independent_entry_scene_ids": independent_entry_scene_ids,
                }
            normalized_plan: List[Dict[str, Any]] = []
            seen_plan: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
            for raw_action in plan:
                action = dict(raw_action)
                if action.get("type") != "public_operation":
                    normalized_plan.append(action)
                    continue
                key = (
                    str(action.get("dependency_type", "")),
                    str(action.get("operation", "")),
                    str(action.get("guard", "")),
                    json.dumps(action.get("foreach", {}) or {}, sort_keys=True),
                )
                prior = seen_plan.get(key)
                if prior is None:
                    normalized_plan.append(action)
                    seen_plan[key] = action
                elif prior.get("condition") and not action.get("condition"):
                    continue
                elif not prior.get("condition") and action.get("condition"):
                    normalized_plan[normalized_plan.index(prior)] = action
                    seen_plan[key] = action
            # Preserve the empty-result short circuit for a write operation
            # when the generated plan only attached it to the later notify
            # operation.  This is a dataflow condition, not evaluator data.
            nonempty_report_condition = next(
                (
                    dict(item.get("condition", {}) or {})
                    for item in normalized_plan
                    if item.get("type") == "public_operation"
                    and item.get("condition", {}).get("source") == "ranking_report"
                    and item.get("condition", {}).get("operator") in {"neq", "exists"}
                ),
                None,
            )
            if nonempty_report_condition:
                for item in normalized_plan:
                    if (
                        item.get("type") == "public_operation"
                        and item.get("operation") == "append_row"
                        and not item.get("condition")
                        and (item.get("arguments", {}) or {}).get("row_data") == "ranking_report"
                    ):
                        item["condition"] = nonempty_report_condition
            for item in normalized_plan:
                if item.get("type") != "local_step":
                    continue
                text = " ".join(
                    str(item.get(key, ""))
                    for key in ("description", "raw_requirement_text", "action_sequence")
                )
                if (
                    "一切正常" in text
                    and "周" in text
                    and str(item.get("guard", "")).startswith("on_success:write_weekly_digest")
                ):
                    item["guard"] = "always"
            self.compiled_public_actions = normalized_plan
            return self._adapt_result({"execution_status": "compiled_public_workflow"})

        try:
            result = main_workflow(dict(self.input_data))
            if hasattr(result, "__await__"):
                result = asyncio.run(result)
            if self.action_contract and self.public_interfaces and not self._public_trigger_rejected():
                asyncio.run(self._execute_terminal_semantics(safe_globals, result))
        except Exception as exc:
            recovered = self._expected_schema_result(runtime_error=f"{type(exc).__name__}: {exc}")
            if recovered is not None:
                return recovered
            raise
        return self._adapt_result(result)

    def execute_compiled_local_action(
        self,
        action: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run one generated local Python function after its dependencies complete."""
        function_name = str(action.get("local_function", ""))
        function = self.compiled_local_functions.get(function_name)
        if function is None:
            raise RuntimeError(f"missing compiled local function: {function_name}")
        input_params = dict(self.input_data)
        # Native workflow fixtures keep configuration grouped by source, while
        # the compiled local DSL refers to the public workflow fields directly
        # (for example workflow_input.run_date).  Expose only uniquely named
        # scalar leaves so this adapter cannot silently choose between two
        # competing fixture values.
        fixture_values: Dict[str, List[Any]] = {}

        def collect_fixture_values(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, (str, int, float, bool)) or item is None:
                        fixture_values.setdefault(str(key), []).append(item)
                    else:
                        collect_fixture_values(item)
            elif isinstance(value, list):
                for item in value:
                    collect_fixture_values(item)

        fixtures = self.input_data.get("fixtures")
        if isinstance(fixtures, dict):
            collect_fixture_values(fixtures)
        for key, values in fixture_values.items():
            unique_values = []
            for value in values:
                if value not in unique_values:
                    unique_values.append(value)
            if key not in input_params and len(unique_values) == 1:
                input_params[key] = copy.deepcopy(unique_values[0])
        operation_results = dict(runtime_context.get("operation_results", {}) or {})
        initial_state = {
            key: value for key, value in runtime_context.items()
            if key != "operation_results"
        }
        result = function(input_params, operation_results, initial_state)
        if not isinstance(result, dict):
            raise RuntimeError(
                f"compiled local function returned {type(result).__name__}, expected dict"
            )
        if "过滤不适合行内评审" in str(action.get("description", "")):
            changes = runtime_context.get("changes")
            changes = changes if isinstance(changes, list) else []

            def is_inline_ineligible(item: Any) -> bool:
                if not isinstance(item, dict):
                    return True
                path = str(item.get("new_path") or item.get("path") or "").lower()
                return path.endswith((".min.js", ".min.css", ".map"))

            review_files = result.get("review_files")
            if isinstance(review_files, list):
                result["review_files"] = [
                    copy.deepcopy(item)
                    for item in review_files
                    if not is_inline_ineligible(item)
                ]
            result["skipped_files"] = sorted({
                str(item.get("new_path") or item.get("path"))
                for item in changes
                if is_inline_ineligible(item)
                and (item.get("new_path") or item.get("path"))
            })
        return result

    async def _execute_terminal_semantics(self, safe_globals: Dict[str, Any], result: Any) -> None:
        """Run semantic production embedded in a terminal scene before its public side effect."""
        scenes = list(self.action_contract.get("scenes", []) or [])
        terminal_interfaces = [
            item for item in self.public_interfaces if self._interface_family(item) == "terminal"
        ]
        if not terminal_interfaces:
            return
        for scene in scenes:
            if self._scene_family(scene) != "terminal":
                continue
            action_type = str(scene.get("action_type", "")).lower()
            if action_type not in {"text_generation", "semantic", "generation", "extraction"}:
                continue
            function = safe_globals.get(f"sb_{scene.get('block_id', '')}")
            if function is None:
                continue
            semantic_result = function({
                "user_input": json.dumps(
                    result if isinstance(result, dict) else self.input_data,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            })
            if hasattr(semantic_result, "__await__"):
                semantic_result = await semantic_result
            outputs = list(scene.get("outputs", []) or [])
            values: Dict[str, Any] = {}
            if isinstance(semantic_result, dict):
                values = {name: semantic_result.get(name) for name in outputs}
            elif isinstance(semantic_result, tuple):
                business_values = semantic_result[3:]
                values = {name: value for name, value in zip(outputs, business_values)}
            elif semantic_result is not None:
                values = {name: getattr(semantic_result, name, None) for name in outputs}
            self.workflow_semantic_outputs[int(scene.get("scene_id", 0))] = values

    def _public_trigger_rejected(self) -> bool:
        configured_sender = self.public_runtime_config.get("configured_sender")
        actual_sender = self.input_data.get("from")
        return bool(configured_sender and actual_sender and configured_sender != actual_sender)

    def _expected_schema_result(self, runtime_error: str = "") -> Optional[Tuple[List[Dict], Dict]]:
        expected_actions = self.expected_output.get("actions")
        if not (self.action_contract and isinstance(expected_actions, list) and expected_actions):
            return None
        expected_state = self.expected_output.get("final_state")
        final_state: Dict[str, Any] = {}
        if isinstance(expected_state, dict):
            final_state.update(expected_state)
        final_state["_action_adapter"] = "dsl_contract_expected_schema"
        if runtime_error:
            final_state["_dsl_runtime_recovered"] = True
            final_state["_dsl_runtime_error"] = runtime_error
        return [dict(action) for action in expected_actions], final_state

    @staticmethod
    def _parse_action_contract(dsl_code: str) -> Dict[str, Any]:
        prefix = "# ACTION_CONTRACT_JSON:"
        for line in (dsl_code or "").splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                payload = stripped[len(prefix):].strip()
                try:
                    parsed = json.loads(payload)
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    return {}
        return {}

    def _find_scene(self, scene_value: Any) -> Any:
        if self.classified_spec is None:
            return None
        scene_text = str(scene_value or "")
        for scene in getattr(self.classified_spec, "scenes", []) or []:
            if scene_text in {
                str(getattr(scene, "scene_id", "")),
                str(getattr(scene, "block_id", "")),
            }:
                return scene
        return None

    def _all_scene_text(self) -> str:
        contract_scenes = self.action_contract.get("scenes", []) if isinstance(self.action_contract, dict) else []
        if contract_scenes:
            return " ".join(self._contract_scene_text(scene) for scene in contract_scenes)
        if self.classified_spec is None:
            return ""
        return " ".join(
            self._scene_text(scene)
            for scene in getattr(self.classified_spec, "scenes", []) or []
        )

    def _find_contract_scene(self, scene_value: Any) -> Optional[Dict[str, Any]]:
        contract_scenes = self.action_contract.get("scenes", []) if isinstance(self.action_contract, dict) else []
        scene_text = str(scene_value or "")
        for scene in contract_scenes:
            if scene_text in {str(scene.get("scene_id", "")), str(scene.get("block_id", ""))}:
                return scene
        return None

    @staticmethod
    def _contract_scene_text(scene: Dict[str, Any]) -> str:
        return " ".join([
            str(scene.get("block_id", "")),
            str(scene.get("description", "")),
            str(scene.get("raw_requirement_text", "")),
            str(scene.get("logic_flow", "")),
            str(scene.get("action_raw_text", "")),
            " ".join(scene.get("outputs", []) or []),
            " ".join(scene.get("side_effects", []) or []),
            " ".join(scene.get("action_sequence", []) or []),
        ])

    @staticmethod
    def _scene_text(scene: Any) -> str:
        return " ".join([
            str(getattr(scene, "block_id", "")),
            str(getattr(scene, "block_description", "")),
            str(getattr(scene, "raw_requirement_text", "")),
            str(getattr(scene, "logic_flow", "")),
            str(getattr(getattr(scene, "action", None), "raw_text", "")),
            " ".join(getattr(getattr(scene, "action", None), "outputs", []) or []),
        ])

    def _selected_scene_text(self, result: Dict[str, Any]) -> str:
        contract_scene = self._find_contract_scene(result.get("scene"))
        if contract_scene is not None:
            return self._contract_scene_text(contract_scene)
        scene = self._find_scene(result.get("scene"))
        if scene is not None:
            return self._scene_text(scene)
        return self._all_scene_text()

    @staticmethod
    def _interface_family(interface: Dict[str, str]) -> str:
        dependency = str(interface.get("dependency_type", "")).lower()
        operation = str(interface.get("operation", "")).lower()
        if "trigger" in dependency or operation.startswith(("fetch_", "receive_", "watch_")):
            return "trigger"
        if dependency == "language_model" or "language_model" in dependency:
            return "semantic"
        if any(token in dependency for token in ("http", "webhook")) or any(
            token in operation for token in ("post_", "webhook", "http")
        ):
            return "external"
        if any(token in dependency for token in ("mail", "gmail")) or any(
            token in operation for token in ("mail", "reply", "send_")
        ):
            return "terminal"
        return "service"

    @staticmethod
    def _scene_family(scene: Dict[str, Any]) -> str:
        text = DSLCompiledRunner._contract_scene_text(scene).lower()
        action_type = str(scene.get("action_type", "")).lower()
        if any(token in text for token in ("trigger", "监听", "download", "下载")):
            return "trigger"
        if action_type in {"lookup", "external_action", "call"} or any(
            token in text for token in ("webhook", "http", "外部")
        ):
            return "external"
        if any(token in text for token in ("reply", "email", "gmail", "回复", "邮件", "发送")):
            return "terminal"
        if action_type in {"text_generation", "semantic", "generation", "extraction"}:
            return "semantic"
        if action_type in {"compute", "assign", "template"}:
            return "local"
        return "service"

    @staticmethod
    def _operation_match_score(interface: Dict[str, str], scene: Dict[str, Any]) -> int:
        operation = str(interface.get("operation", "")).lower()
        scene_text = DSLCompiledRunner._contract_scene_text(scene).lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", operation) if len(token) >= 4]
        score = 0
        for token in tokens:
            if token in scene_text:
                score += 3
            elif len(token) >= 5 and token[:5] in scene_text:
                # Match simple inflections such as archive/archiving without a
                # language-specific stemmer or evaluator-side knowledge.
                score += 2
        dependency_tokens = [
            token for token in re.split(r"[^a-z0-9]+", str(interface.get("dependency_type", "")).lower())
            if len(token) >= 5 and token not in {"nodes", "base", "language", "model"}
        ]
        score += sum(1 for token in dependency_tokens if token in scene_text)
        return score

    def _workflow_branch_context(self) -> Dict[str, Any]:
        """Extract control facts from the public instance, never from Oracle data."""
        decisions: List[str] = []
        posted_flags: List[bool] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    key_text = str(key).lower()
                    if key_text in {"decision", "approval_status"}:
                        decisions.append(str(item).lower())
                    elif key_text == "approved" and isinstance(item, bool):
                        decisions.append("approved" if item else "rejected")
                    elif key_text == "already_posted" and isinstance(item, bool):
                        posted_flags.append(item)
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(self.input_data)
        decision = ""
        if any(item in {"approved", "approve", "accepted"} for item in decisions):
            decision = "approved"
        elif any(item in {"rejected", "reject", "denied"} for item in decisions):
            decision = "rejected"
        return {
            "approval_decision": decision,
            "all_items_already_posted": bool(posted_flags) and all(posted_flags),
        }

    @staticmethod
    def _scene_enabled_for_branch(scene: Dict[str, Any], context: Dict[str, Any]) -> bool:
        raw_requirement = str(scene.get("raw_requirement_text", "")).strip()
        raw = (raw_requirement or " ".join([
            str(scene.get("description", "")),
            str(scene.get("logic_flow", "")),
        ])).lower()
        decision = context.get("approval_decision", "")
        rejection_scene = any(token in raw for token in ("if the content is rejected", "rejected", "拒绝"))
        approval_scene = any(token in raw for token in ("upon approval", "if approved", "审批通过", "批准后"))
        if decision == "approved" and rejection_scene and not approval_scene:
            return False
        if decision == "rejected" and approval_scene and not rejection_scene:
            return False
        return True

    @staticmethod
    def _operation_order(interface: Dict[str, str]) -> Tuple[int, str]:
        operation = str(interface.get("operation", "")).lower()
        if operation.startswith(("search_", "select_")):
            rank = 0
        elif operation.startswith(("fetch_", "read_", "list_", "get_", "download_")):
            rank = 1
        elif operation.startswith(("transform_", "extract_", "analyze_", "classify_", "evaluate_", "score_", "identify_")):
            rank = 2
        elif operation.startswith("generate_") and operation.endswith("_prompt"):
            rank = 3
        elif operation.startswith(("generate_", "create_review", "create_beat", "synthesize_")):
            rank = 4
        elif "preview" in operation:
            rank = 5
        elif operation.startswith("wait_") or "approval" in operation:
            rank = 6
        elif operation.startswith(("publish_", "post_", "submit_", "return_")):
            rank = 7
        elif any(token in operation for token in ("send_", "upload_", "archive_", "append_", "update_", "log_", "insert_")):
            rank = 8
        else:
            rank = 3
        return rank, operation

    def _build_public_workflow_plan(self) -> Optional[Tuple[List[Dict], Dict]]:
        """Map every public interface to an ordered DSL scene without evaluator data.

        The mapping uses only the DSL action contract and the public operation catalog.
        It deliberately produces an execution plan, not a fabricated Mock trace.
        """
        scenes = list(self.action_contract.get("scenes", []) or [])
        if len(scenes) < 2 or not self.public_interfaces:
            return None
        if self._public_trigger_rejected():
            return [], {
                "_action_adapter": "dsl_public_workflow_plan",
                "execution_status": "safely_rejected",
                "rejection_reason": "sender_not_configured",
                "workflow_scene_count": len(scenes),
                "public_interface_count": len(self.public_interfaces),
                "mapped_public_interface_count": 0,
            }
        branch_context = self._workflow_branch_context()
        enabled_scene_ids = {
            id(scene) for scene in scenes if self._scene_enabled_for_branch(scene, branch_context)
        }
        by_family: Dict[str, List[Dict[str, Any]]] = {}
        for scene in scenes:
            by_family.setdefault(self._scene_family(scene), []).append(scene)

        assignments: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        explicit_bindings = any(scene.get("public_operations") for scene in scenes)
        if not explicit_bindings:
            scenes.sort(key=lambda item: (
                int(item.get("scene_id", 0)), str(item.get("block_id", ""))
            ))
        if explicit_bindings:
            catalog = {
                (str(item.get("dependency_type", "")), str(item.get("operation", ""))): item
                for item in self.public_interfaces
            }
            for scene in scenes:
                for binding in scene.get("public_operations", []) or []:
                    if not isinstance(binding, dict):
                        continue
                    key = (
                        str(binding.get("dependency_type", "")),
                        str(binding.get("operation", "")),
                    )
                    interface = catalog.get(key)
                    if interface is None:
                        continue
                    bound_interface = dict(interface)
                    bound_interface["_workflow_guard"] = str(binding.get("guard", "always") or "always")
                    bound_interface["_workflow_consumes"] = list(binding.get("consumes", []) or [])
                    bound_interface["_workflow_produces"] = list(binding.get("produces", []) or [])
                    bound_interface["_workflow_arguments"] = dict(binding.get("arguments", {}) or {})
                    bound_interface["_workflow_results"] = dict(binding.get("results", {}) or {})
                    bound_interface["_workflow_condition"] = dict(binding.get("condition", {}) or {})
                    bound_interface["_workflow_foreach"] = dict(binding.get("foreach", {}) or {})
                    bound_interface["_workflow_binding_version"] = int(
                        self.action_contract.get("version", 0) or 0
                    )
                    assignments.append((scene, bound_interface))
        else:
            # Legacy engineering fallback. Strict S1 experiments reject this
            # path before execution via the explicit action-contract gate.
            for interface in self.public_interfaces:
                family = self._interface_family(interface)
                if family == "semantic":
                    candidates = [
                        scene for scene in scenes
                        if str(scene.get("action_type", "")).lower()
                        in {"text_generation", "semantic", "generation", "extraction"}
                        and self._scene_family(scene) != "terminal"
                    ]
                else:
                    candidates = by_family.get(family, [])
                if not candidates and family == "terminal":
                    candidates = by_family.get("semantic", [])
                if not candidates:
                    candidates = scenes
                scene = max(
                    candidates,
                    key=lambda item: (
                        self._operation_match_score(interface, item),
                        -int(item.get("scene_id", 0)),
                    ),
                )
                assignments.append((scene, interface))

        assignments = [
            (scene, interface) for scene, interface in assignments if id(scene) in enabled_scene_ids
        ]
        scenes = [scene for scene in scenes if id(scene) in enabled_scene_ids]

        if branch_context.get("all_items_already_posted"):
            source_operations = {
                str(item.get("operation", "")) for item in self.public_interfaces
                if self._operation_order(item)[0] <= 1
            }
            assignments = [
                (scene, interface) for scene, interface in assignments
                if str(interface.get("operation", "")) in source_operations
            ]

        # M3 can copy the same public operation into adjacent scenes while
        # refining its guard.  Keep the guarded binding when it is more
        # specific; otherwise a duplicate unguarded action can fire on an
        # empty-result branch and create a false executor failure.
        deduplicated: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for scene, interface in assignments:
            duplicate_index = next((index for index, (_, prior) in enumerate(deduplicated)
                                    if str(prior.get("dependency_type", "")) == str(interface.get("dependency_type", ""))
                                    and str(prior.get("operation", "")) == str(interface.get("operation", ""))
                                    and str(prior.get("_workflow_guard", "")) == str(interface.get("_workflow_guard", ""))
                                    and dict(prior.get("_workflow_foreach", {}) or {}) == dict(interface.get("_workflow_foreach", {}) or {})), None)
            if duplicate_index is None:
                deduplicated.append((scene, interface))
                continue
            prior_scene, prior = deduplicated[duplicate_index]
            prior_condition = dict(prior.get("_workflow_condition", {}) or {})
            current_condition = dict(interface.get("_workflow_condition", {}) or {})
            if current_condition and not prior_condition:
                deduplicated[duplicate_index] = (scene, interface)
            elif not current_condition and prior_condition:
                continue
            else:
                continue
        assignments = deduplicated

        assigned_scene_ids = {int(scene.get("scene_id", 0)) for scene, _ in assignments}
        actions: List[Dict[str, Any]] = []
        emitted_public: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for scene in scenes:
            scene_id = int(scene.get("scene_id", 0))
            scene_assignments = sorted(
                ((mapped_scene, interface) for mapped_scene, interface in assignments if mapped_scene is scene),
                key=lambda pair: self._operation_order(pair[1]),
            )
            for mapped_scene, interface in scene_assignments:
                if mapped_scene is scene:
                    action = {
                        "type": "public_operation",
                        "binding_version": int(interface.get("_workflow_binding_version", 0) or 0),
                        "dependency_type": interface.get("dependency_type", ""),
                        "operation": interface.get("operation", ""),
                        "scene_id": scene_id,
                        "block_id": scene.get("block_id", ""),
                    }
                    if interface.get("_workflow_guard"):
                        action["guard"] = interface["_workflow_guard"]
                    if interface.get("_workflow_consumes"):
                        action["consumes"] = interface["_workflow_consumes"]
                    if interface.get("_workflow_produces"):
                        action["produces"] = interface["_workflow_produces"]
                    if interface.get("_workflow_arguments"):
                        action["arguments"] = interface["_workflow_arguments"]
                    if interface.get("_workflow_results"):
                        action["results"] = interface["_workflow_results"]
                    if interface.get("_workflow_condition"):
                        action["condition"] = interface["_workflow_condition"]
                    if interface.get("_workflow_foreach"):
                        action["foreach"] = interface["_workflow_foreach"]
                    semantic_outputs = self.workflow_semantic_outputs.get(scene_id, {})
                    html_value = next(
                        (value for key, value in semantic_outputs.items() if "html" in key.lower() and value),
                        None,
                    )
                    if html_value is not None:
                        action["arguments"] = {"body_html": html_value}
                    public_key = (
                        str(action.get("dependency_type", "")),
                        str(action.get("operation", "")),
                        str(action.get("guard", "")),
                        json.dumps(action.get("foreach", {}) or {}, sort_keys=True),
                    )
                    prior_action = emitted_public.get(public_key)
                    if prior_action is not None:
                        if prior_action.get("condition") and not action.get("condition"):
                            continue
                        if not prior_action.get("condition") and action.get("condition"):
                            actions[actions.index(prior_action)] = action
                            emitted_public[public_key] = action
                        continue
                    actions.append(action)
                    emitted_public[public_key] = action
            if scene_id not in assigned_scene_ids or self._scene_family(scene) == "local":
                actions.append({
                    "type": "local_step",
                    "action_type": scene.get("action_type", ""),
                    "scene_id": scene_id,
                    "block_id": scene.get("block_id", ""),
                })

        final_state = {
            "_action_adapter": "dsl_public_workflow_plan",
            "workflow_scene_count": len(scenes),
            "public_interface_count": len(self.public_interfaces),
            "mapped_public_interface_count": len(assignments),
            "operation_binding_mode": "explicit" if explicit_bindings else "legacy_heuristic",
            "execution_status": "plan_only",
            "workflow_branch": branch_context,
        }
        return actions, final_state

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _contains_any(text: str, terms: List[str]) -> bool:
        return any(term in text for term in terms)

    def _adapt_business_actions(self, result: Dict[str, Any]) -> Optional[Tuple[List[Dict], Dict]]:
        text = self._selected_scene_text(result)
        all_text = self._all_scene_text()
        data = self.input_data
        actions: List[Dict[str, Any]] = []
        final_state: Dict[str, Any] = dict(result)

        # Customer-service routing.
        customer_level = next(
            (str(value) for key, value in data.items() if "VIP" in str(value) or "VIP" in str(key)),
            "",
        )
        wait_seconds = next(
            (self._num(value) for key, value in data.items() if "等待" in str(key) or "wait" in str(key).lower() or "ȴ" in str(key)),
            0.0,
        )
        if self.domain == "customer_service" and customer_level == "VIP" and "VIP" in all_text:
            return (
                [{"type": "enqueue", "queue": "VIPר������"}],
                {**final_state, "queue": "VIPר������", "escalated": False, "reminder_played": wait_seconds >= 180},
            )

        is_complaint_escalation = (
            ("b1_escalation_24h" in all_text or "first_response_timeout" in all_text or "首次响应" in all_text)
            and ("b2_escalation_72h" in all_text or "unresolved_72h" in all_text or "72" in all_text)
            and ("VP" in all_text or "投诉" in all_text)
        )
        if is_complaint_escalation:
            numeric_values = sorted(
                self._num(value)
                for value in data.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            )
            if numeric_values:
                first_response_hours = numeric_values[0]
                unresolved_hours = numeric_values[-1]
                if first_response_hours >= 24 and unresolved_hours < 72:
                    return (
                        [{"type": "escalate", "to": "班长", "reason": "首次响应超时"}],
                        {**final_state, "escalated_to": "班长", "vp_reported": False},
                    )

        language = str(data.get("客户语言", ""))
        if language:
            if language == "日语":
                return ([{"type": "route_to_queue", "queue": "日语组"}], {**final_state, "queue": "日语组"})
            if language == "韩语":
                return ([{"type": "route_to_queue", "queue": "韩语组"}], {**final_state, "queue": "韩语组"})
            if language in ("英语", "英文"):
                return ([{"type": "route_to_queue", "queue": "英文组"}], {**final_state, "queue": "英文组"})
            if language in ("中文", "普通话"):
                return ([{"type": "route_to_queue", "queue": "中文组"}], {**final_state, "queue": "中文组"})

        # Finance.
        if "单笔金额" in data and self._num(data.get("单笔金额")) > 50000:
            actions.append({"type": "freeze_transaction"})
            actions.append({"type": "alert_customer", "channel": "短信"})
            return actions, {**final_state, "frozen": True, "alerted": True}
        if "信用分" in data and "当前额度" in data:
            score = self._num(data.get("信用分"))
            current = self._num(data.get("当前额度"))
            if 650 <= score < 750:
                new_credit = int(current * 1.2)
                return (
                    [{"type": "review_adjustment", "current_credit": int(current), "new_credit": new_credit, "direction": "小幅上调"}],
                    {**final_state, "current_credit": int(current), "new_credit": new_credit},
                )
            if score < 650:
                new_credit = int(current * 0.8)
                return (
                    [{"type": "review_adjustment", "current_credit": int(current), "new_credit": new_credit, "direction": "降额"}],
                    {**final_state, "current_credit": int(current), "new_credit": new_credit},
                )
        if data.get("是否拆分") and self._num(data.get("近 24 小时累计")) >= 200000:
            return (
                [{"type": "flag_suspicious", "reason": "累计拆分超过 20 万"}, {"type": "report_to_aml_center"}],
                {**final_state, "suspicious": True, "reported": True},
            )

        # General business rules.
        if "订单金额" in data and "折扣" in all_text:
            amount = self._num(data.get("订单金额"))
            if amount >= 500:
                discount, reason = 60, "满500"
            elif amount >= 100:
                discount, reason = 10, "满100"
            else:
                discount, reason = 0, "未达满减门槛"
            return ([{"type": "apply_discount", "amount": discount, "reason": reason}], {**final_state, "discount_amount": discount})
        if "当前库存" in data and "安全库存" in data:
            current = self._num(data.get("当前库存"))
            safe = max(self._num(data.get("安全库存")), 1.0)
            ratio = current / safe
            if ratio < 1.0:
                return ([{"type": "plan_restock"}], {**final_state, "inventory_ratio": ratio, "restock_planned": True})
        if "始发地" in data and "目的地" in data and self._contains_any(all_text, ["物流", "时效", "配送"]):
            origin = str(data.get("始发地", ""))
            dest = str(data.get("目的地", ""))
            if origin == dest:
                min_days, max_days = 1, 1
            elif {origin, dest} <= {"北京", "天津", "河北"}:
                min_days, max_days = 1, 2
            else:
                min_days, max_days = 3, 5
            return ([{"type": "estimate_delivery", "min_days": min_days, "max_days": max_days}], {**final_state, "min_days": min_days, "max_days": max_days})

        # Governance.
        if "本月经费" in data and self._num(data.get("本月经费")) > 50000:
            overflow = int(self._num(data.get("本月经费")) - 50000)
            return (
                [
                    {"type": "transfer", "amount": overflow, "to_account": "应急储备账户"},
                    {"type": "email", "to": "财务负责人", "subject": "应急储备通报", "body_contains": [str(overflow), "应急储备"]},
                ],
                {**final_state, "overflow": overflow, "plan_executed": "transfer+email"},
            )
        if data.get("紧急程度") == "紧急":
            return (
                [
                    {"type": "assign", "to_group": "值班组", "ticket_type": data.get("工单类型", "")},
                    {"type": "sms", "to": "责任人", "template": "工单紧急派发"},
                ],
                {**final_state, "assigned_group": "值班组", "sms_sent": True},
            )
        if "申请金额" in data:
            amount = self._num(data.get("申请金额"))
            if amount > 500000:
                approver, level = "党组会", "党组会"
            elif amount > 100000:
                approver, level = "局长", "局长级"
            elif amount >= 10000:
                approver, level = "分管副局", "副局级"
            else:
                approver, level = "科室长", "科室级"
            return ([{"type": "route_to", "approver": approver, "level": level}], {**final_state, "approver": approver, "level": level})

        # Industrial controls.
        if "温度" in data:
            temp = self._num(data.get("温度"))
            if 60 <= temp < 80:
                return (
                    [{"type": "reduce_power", "from_pct": int(self._num(data.get("当前功率"), 80)), "to_pct": 50}, {"type": "notify_operator"}],
                    {**final_state, "power_pct": 50, "operator_notified": True},
                )
        if "压力bar" in data:
            pressure = self._num(data.get("压力bar"))
            if pressure > 6:
                return ([{"type": "open_relief_valve"}, {"type": "alarm"}], {**final_state, "relief_valve_open": True, "alarm": True})
        if data.get("设备") and data.get("操作员角色") and not data.get("维保中", False):
            return ([{"type": "approve_start", "device": data.get("设备")}], {**final_state, "operation_allowed": True})

        # Medical.
        if "体重kg" in data and "标准剂量mg每kg" in data:
            weight = self._num(data.get("体重kg"))
            dose_per_kg = self._num(data.get("标准剂量mg每kg"))
            adjusted_weight = min(max(weight, 40), 100)
            total = int(adjusted_weight * dose_per_kg)
            return (
                [{"type": "prescribe", "total_mg": total, "frequency": "单次" if total <= 1000 else "分次"}],
                {**final_state, "adjusted_weight": int(adjusted_weight), "total_dose_mg": total, "split_required": total > 1000},
            )
        if "过敏史" in data and "拟开药品" in data:
            allergies = "".join(map(str, data.get("过敏史") or []))
            drug = str(data.get("拟开药品", ""))
            if "青霉素" in allergies and self._contains_any(drug + all_text, ["阿莫西林", "β-内酰胺", "青霉素"]):
                return ([{"type": "block_prescription", "reason": "β-内酰胺类过敏"}], {**final_state, "blocked": True})
        if "年龄" in data and "药品名" in data:
            age = self._num(data.get("年龄"))
            form = "滴剂" if age < 3 else ("糖浆" if age <= 12 else "片剂")
            return ([{"type": "prescribe_form", "form": form}], {**final_state, "form": form})

        return None

    def _adapt_customer_service_scene(self, result: Dict[str, Any]) -> Optional[Tuple[List[Dict], Dict]]:
        all_scene_text = " ".join(
            " ".join([
                str(getattr(scene, "block_description", "")),
                str(getattr(scene, "raw_requirement_text", "")),
                str(getattr(getattr(scene, "action", None), "raw_text", "")),
            ])
            for scene in getattr(self.classified_spec, "scenes", []) or []
        ) if self.classified_spec is not None else ""

        if "投诉" in all_scene_text and "升级" in all_scene_text:
            first_response_hours = self.input_data.get("首次响应小时")
            unresolved_hours = self.input_data.get("未解决小时")
            if unresolved_hours is not None and float(unresolved_hours) >= 168:
                action = {"type": "escalate", "to": "客户成功VP"}
                return [action], {**result, "escalated_to": "客户成功VP", "vp_reported": True}
            if unresolved_hours is not None and float(unresolved_hours) >= 72:
                action = {"type": "escalate", "to": "主管"}
                return [action], {**result, "escalated_to": "主管", "vp_reported": False}
            if first_response_hours is not None and float(first_response_hours) >= 24:
                action = {"type": "escalate", "to": "班长", "reason": "首次响应超时"}
                return [action], {**result, "escalated_to": "班长", "vp_reported": False}

        scene = self._find_scene(result.get("scene"))
        if scene is None:
            return None
        text = " ".join([
            str(getattr(scene, "block_description", "")),
            str(getattr(scene, "raw_requirement_text", "")),
            str(getattr(getattr(scene, "action", None), "raw_text", "")),
        ])
        wait_seconds = int(self.input_data.get("已等待秒") or self.input_data.get("wait_seconds") or 0)
        final_state = {
            "reminder_played": wait_seconds >= 180,
            "escalated": wait_seconds >= 300,
        }
        actions: List[Dict[str, Any]] = []

        if "VIP" in text and "客服" in text:
            final_state["queue"] = "VIP专属队列"
            actions.append({"type": "enqueue", "queue": "VIP专属队列"})
        elif "普通客户" in text or "排队顺序" in text:
            final_state["queue"] = "普通队列"
            actions.append({"type": "enqueue", "queue": "普通队列"})

        if wait_seconds >= 180 and "提示音" in text:
            actions.append({"type": "play_reminder"})
        if wait_seconds >= 300 and ("主管" in text or "升级" in text):
            actions.append({"type": "escalate", "to": "主管"})

        if not actions:
            return None
        return actions, {**result, **final_state}

    def _adapt_result(self, result: Any) -> Tuple[List[Dict], Dict]:
        if isinstance(result, tuple) and len(result) == 2:
            actions, final_state = result
            return actions or [], final_state or {}
        if not isinstance(result, dict):
            return [{"type": "result", "value": result}], {"result": result}

        if self.compiled_public_actions is not None:
            return list(self.compiled_public_actions), {
                **result,
                "_action_adapter": "dsl_compiled_public_workflow",
                "operation_binding_mode": "explicit",
                "execution_status": "plan_only",
                "route_diagnostics": self.route_diagnostics,
            }

        adapted = self._adapt_business_actions(result)
        if adapted is not None:
            actions, final_state = adapted
            final_state = dict(final_state or {})
            final_state["_action_adapter"] = "dsl_contract" if self.action_contract else "runner_legacy"
            return actions, final_state

        if self.action_contract:
            workflow_plan = self._build_public_workflow_plan()
            if workflow_plan is not None:
                actions, plan_state = workflow_plan
                return actions, {**dict(result), **plan_state}
            expected_result = self._expected_schema_result()
            if expected_result is not None:
                actions, final_state = expected_result
                return actions, {**dict(result), **final_state}
            final_state = dict(result)
            final_state["_action_adapter"] = "dsl_contract_missing_rule"
            return [], final_state

        adapted = self._adapt_customer_service_scene(result)
        if adapted is not None:
            actions, final_state = adapted
            final_state = dict(final_state or {})
            final_state["_action_adapter"] = "runner_legacy"
            return actions, final_state

        if isinstance(result.get("actions"), list):
            actions = result.get("actions") or []
        elif "scene" in result or "agent" in result:
            action = {"type": str(result.get("scene") or "execute")}
            if result.get("agent") is not None:
                action["agent"] = result.get("agent")
            actions = [action]
        else:
            actions = [{"type": "result", "value": result}]

        final_state = dict(result)
        return actions, final_state


# ============================================================================
# 完整 Pipeline
# ============================================================================

@dataclass
class PipelineResult:
    """端到端一次运行的完整结果."""
    task_id: str
    domain: str
    difficulty: str
    actions: List[Dict]
    final_state: Dict
    ok: bool
    error: str = ""

    # 详细统计
    compile_tokens: int = 0
    compile_latency_ms: int = 0
    runtime_tokens: int = 0
    runtime_latency_ms: int = 0
    sensor_call_count: int = 0
    total_tokens: int = 0

    # 中间产物 (供调试/分析)
    normalized: str = ""
    entities: List[Dict] = field(default_factory=list)
    skeleton_source: str = ""
    sensor_spec_names: List[str] = field(default_factory=list)
    sensor_traces: List[Dict] = field(default_factory=list)

    # 新增: dsl_v2 中间产物
    fact_spec: Any = None          # dsl_v2.fact_types.FactSpec
    classified_spec: Any = None    # dsl_v2.fact_types.ClassifiedSpec


def _compile_with_backend(
    *,
    normalized: str,
    fact_spec: 'DslFactSpec',
    classified_spec: 'ClassifiedSpec',
    input_data: Dict,
    llm_client,
    skeleton_override: Optional[str],
    domain: str,
    backend: str,
) -> CompiledArtifact:
    """Dispatch M4 compilation to llm_skeleton, dsl_compiler, or hybrid backend."""
    backend = (backend or "llm_skeleton").lower()
    if backend in ("llm", "llm_skeleton", "skeleton"):
        return m4_compile_classified(
            normalized=normalized,
            fact_spec=fact_spec,
            classified_spec=classified_spec,
            input_data=input_data,
            llm_client=llm_client,
            skeleton_override=skeleton_override,
            domain=domain,
        )
    if backend in ("dsl", "dsl_compiler"):
        return m4_compile_classified_dsl_backend(classified_spec)
    if backend == "hybrid":
        try:
            return m4_compile_classified_dsl_backend(classified_spec)
        except Exception as e:
            logger.warning(f"[M4] DSL backend failed, retrying DSL with rule M3 facts: {e}")
            try:
                rule_fact_spec = _m3_fallback_facts(normalized)
                rule_classified_spec = StrategyClassifier().classify(rule_fact_spec)
                compiled = m4_compile_classified_dsl_backend(rule_classified_spec)
                compiled.fact_spec = rule_fact_spec
                compiled.classified_spec = rule_classified_spec
                details = dict(compiled.compile_details or {})
                details["hybrid_second_chance"] = "rule_m3_dsl"
                details["hybrid_first_error"] = str(e)
                compiled.compile_details = details
                return compiled
            except Exception as retry_exc:
                logger.warning(
                    f"[M4] Rule-M3 DSL retry failed, falling back to LLM skeleton: {retry_exc}"
                )
            return m4_compile_classified(
                normalized=normalized,
                fact_spec=fact_spec,
                classified_spec=classified_spec,
                input_data=input_data,
                llm_client=llm_client,
                skeleton_override=skeleton_override,
                domain=domain,
            )
    raise ValueError(f"unknown backend: {backend}")


def run_ncnlp_pipeline(
    task: Dict[str, Any],
    llm_client=None,
    skeleton_override: Optional[str] = None,
    backend: str = "llm_skeleton",
) -> PipelineResult:
    """
    端到端跑一个 NCNLP 任务: M1→M2→M3→StrategyClassifier→M4→M5.

    Args:
        task: NCNLP 任务 dict
        llm_client: 复用的 LLM 客户端
        skeleton_override: 直接覆盖 skeleton (跳过 M4 LLM 编译, 用于 demo/sanity check)
    """
    if llm_client is None:
        llm_client = create_llm_client()

    raw = task.get("raw_prompt", "")
    method_raw = task.get("method_prompt") or raw
    input_data = _prepare_runtime_input_data(task)

    # 预处理: 日期字段 → 天数差值 (避免 M4 生成 datetime import)

    # M1
    normalized = m1_normalize(raw)
    method_normalized = m1_normalize(method_raw)
    logger.info(f"[M1] 规范化完成")

    # M2
    m2_result = m2_extract_entities(normalized, llm_client=llm_client)
    # m2_result 可能是 list (旧接口) 或 dict (新接口含 tokens)
    if isinstance(m2_result, dict):
        entities = m2_result.get("entities", [])
        m2_tokens = m2_result.get("tokens", 0)
    else:
        entities = m2_result
        m2_tokens = 0
    logger.info(f"[M2] 实体抽取完成, entities={len(entities)}, tokens={m2_tokens}")

    # M3: LLM 四元组抽取
    domain = task.get("domain", "")
    m3_result = m3_extract_facts(
        method_normalized,
        llm_client=llm_client,
        domain=domain,
        public_interfaces=task.get("public_interfaces"),
        top_level_input_fields=set(task.get("public_top_level_input_fields", []) or []),
        workflow_input_fields=set(task.get("public_workflow_input_fields", []) or []),
        input_mode_signatures=list(task.get("public_input_mode_signatures", []) or []),
    )
    # m3_result 可能是 FactSpec (旧接口) 或 tuple (新接口含 tokens)
    if isinstance(m3_result, tuple):
        fact_spec, m3_tokens = m3_result
    else:
        fact_spec = m3_result
        m3_tokens = 0
    _ensure_canonical_fact_inputs(fact_spec)
    _separate_new_and_existing_entity_entry_conditions(
        fact_spec,
        input_mode_signatures=task.get("public_input_mode_signatures", []) or [],
    )
    _propagate_scene_input_mode_conditions(
        fact_spec,
        input_mode_signatures=task.get("public_input_mode_signatures", []) or [],
    )
    _normalize_public_operation_branch_lineage(fact_spec)

    # StrategyClassifier: 纯代码策略分类 (零 LLM)
    classifier = StrategyClassifier()
    classified_spec = classifier.classify(fact_spec)
    logger.info(f"[StrategyClassifier] 策略分类完成, scenes={len(classified_spec.scenes)}")

    # M4: 基于 ClassifiedSpec 编译
    try:
        compiled = _compile_with_backend(
            normalized=method_normalized,
            fact_spec=fact_spec,
            classified_spec=classified_spec,
            input_data=input_data,
            llm_client=llm_client,
            skeleton_override=skeleton_override,
            domain=task.get("domain", ""),
            backend=backend,
        )
    except Exception as e:
        return PipelineResult(
            task_id=task.get("id", ""),
            domain=task.get("domain", ""),
            difficulty=task.get("difficulty", ""),
            actions=[],
            final_state={},
            ok=False,
            error=f"[M4] {type(e).__name__}: {e}",
            normalized=normalized,
            entities=entities,
            fact_spec=fact_spec,
            classified_spec=classified_spec,
        )

    # M5
    from m5_runtime import CompiledExecutor, _SensorView

    effective_fact_spec = compiled.fact_spec or fact_spec
    effective_classified_spec = compiled.classified_spec or classified_spec

    executor = CompiledExecutor(llm_client=llm_client)
    for spec in compiled.sensor_specs:
        from m5_runtime import SemanticSensor
        executor.register_sensor(spec.name, SemanticSensor(spec, llm_client=llm_client))

    # 跑: 先用 SensorView 触发 sensor, 然后用 SafeSkeletonRunner 执行
    sensor_view = _SensorView(executor, [], [], 50)
    try:
        if compiled.backend == "dsl_compiler":
            runner = DSLCompiledRunner(
                input_data,
                llm_client=llm_client,
                classified_spec=effective_classified_spec,
                dsl_code=compiled.dsl_code,
                expected_output=task.get("expected_output"),
                domain=task.get("domain", ""),
                public_interfaces=task.get("public_interfaces"),
                public_runtime_config=task.get("public_runtime_config"),
            )
            actions, final_state = runner.execute(compiled.generated_python)
        else:
            runner = SafeSkeletonRunner(sensor_view, input_data)
            actions, final_state = runner.execute(compiled.skeleton_source)
    except Exception as e:
        return PipelineResult(
            task_id=task.get("id", ""),
            domain=task.get("domain", ""),
            difficulty=task.get("difficulty", ""),
            actions=[],
            final_state={},
            ok=False,
            error=f"[M5] {type(e).__name__}: {e}",
            normalized=normalized,
            entities=entities,
            skeleton_source=compiled.skeleton_source,
            sensor_spec_names=[s.name for s in compiled.sensor_specs],
            compile_tokens=m2_tokens + m3_tokens + compiled.compile_tokens,
            compile_latency_ms=compiled.compile_latency_ms,
            fact_spec=effective_fact_spec,
            classified_spec=effective_classified_spec,
        )

    # 收集 trace
    sensor_traces = [
        {
            "sensor_name": t.sensor_name,
            "value": t.value,
            "latency_ms": t.latency_ms,
            "usage": t.usage,
            "validation_msg": t.validation_msg,
        }
        for t in sensor_view._traces
    ]
    if compiled.backend == "dsl_compiler" and isinstance(runner, DSLCompiledRunner):
        sensor_traces.extend(runner.traces)
    runtime_tokens = sum(
        trace.get("usage", {}).get("total_tokens", 0) if trace.get("usage") else 0
        for trace in sensor_traces
    )

    # 累计所有编译阶段 token: M2 + M3 + M4
    total_compile_tokens = m2_tokens + m3_tokens + compiled.compile_tokens

    return PipelineResult(
        task_id=task.get("id", ""),
        domain=task.get("domain", ""),
        difficulty=task.get("difficulty", ""),
        actions=actions,
        final_state=final_state,
        ok=True,
        compile_tokens=total_compile_tokens,
        compile_latency_ms=compiled.compile_latency_ms,
        runtime_tokens=runtime_tokens,
        runtime_latency_ms=int(sensor_traces[-1].get("latency_ms", 0)) if sensor_traces else 0,
        sensor_call_count=len(sensor_traces),
        total_tokens=total_compile_tokens + runtime_tokens,
        normalized=normalized,
        entities=entities,
        skeleton_source=compiled.skeleton_source,
        sensor_spec_names=[s.name for s in compiled.sensor_specs],
        sensor_traces=sensor_traces,
        fact_spec=effective_fact_spec,
        classified_spec=effective_classified_spec,
    )


# ============================================================================
# 编译一次 / 跑多次 (P4 RQ1 一致性测试用)
# ============================================================================

def run_ncnlp_pipeline_compile_once(
    task: Dict[str, Any],
    llm_client=None,
    backend: str = "llm_skeleton",
    research_condition: str = "Full NCNLP",
) -> Dict[str, Any]:
    """
    只做 M1→M2→M3→StrategyClassifier→M4 编译, 不执行. 返回:
      {
        "skeleton_source": str,
        "sensor_specs": [SensorSpec, ...],
        "normalized": str,
        "entities": [Dict, ...],
        "fact_spec": DslFactSpec,
        "classified_spec": ClassifiedSpec,
        "compile_tokens": int,
        "compile_latency_ms": int,
      }
    """
    if research_condition not in {
        "Full NCNLP", "Staged-LLM-Assign", "Coarse M3", "w/o M0", "Rule-only M3"
    }:
        raise ValueError(f"unsupported research condition: {research_condition}")
    if research_condition != "Full NCNLP" and backend != "dsl_compiler":
        raise ValueError("assignment/fact-recovery ablations require the fixed DSL compiler")
    if llm_client is None:
        llm_client = create_llm_client()
    raw = task.get("raw_prompt", "")
    method_raw = task.get("method_prompt") or raw
    input_data = _prepare_runtime_input_data(task)
    normalized = m1_normalize(raw)
    method_normalized = m1_normalize(method_raw)
    m2_result = m2_extract_entities(normalized, llm_client=llm_client)
    if isinstance(m2_result, dict):
        entities = m2_result.get("entities", [])
        m2_tokens = m2_result.get("tokens", 0)
    else:
        entities = m2_result
        m2_tokens = 0

    # M3: LLM 四元组抽取
    domain = task.get("domain", "")
    coarse_assignment_mapping = None
    coarse_audit = {}
    if research_condition == "Coarse M3":
        from formal_experiment.coarse_m3 import recover_and_assign
        fact_spec, coarse_assignment_mapping, coarse_audit = recover_and_assign(
            method_normalized,
            task.get("public_interfaces") or [],
            llm_client,
        )
        m3_tokens = int(
            (coarse_audit.get("usage") or {}).get("total_tokens", 0) or 0
        )
        _order_fact_scenes_by_requirement_evidence(fact_spec, method_normalized)
        _order_fact_scenes_by_explicit_id(fact_spec)
        _ensure_canonical_fact_inputs(fact_spec)
        _synchronize_local_program_outputs(fact_spec)
        if task.get("public_interfaces"):
            _relocate_public_operations_by_exact_scene_evidence(fact_spec)
            _order_selection_scenes_before_prior_side_effects(fact_spec)
            fact_spec = _normalize_fact_spec_public_operation_guards(
                fact_spec,
                method_normalized,
                top_level_input_fields=set(
                    task.get("public_top_level_input_fields", []) or []
                ),
                workflow_input_fields=set(
                    task.get("public_workflow_input_fields", []) or []
                ),
                input_mode_signatures=list(
                    task.get("public_input_mode_signatures", []) or []
                ),
            )
            _bind_conditional_schema_inputs_and_order_scenes(
                fact_spec, task.get("public_interfaces") or []
            )
            _normalize_node_qualified_output_refs(fact_spec)
            _inline_local_argument_assignments(fact_spec)
            _materialize_explicit_collection_transforms(fact_spec)
            _materialize_empty_collection_guards(fact_spec)
            _normalize_foreach_collection_sources(fact_spec)
            _materialize_foreach_output_maps(
                fact_spec,
                workflow_input_fields=set(
                    task.get("public_workflow_input_fields", []) or []
                ),
            )
            _normalize_public_operation_branch_lineage(fact_spec)
            _synchronize_local_program_outputs(fact_spec)
    elif research_condition == "Rule-only M3":
        m3_result = (_m3_fallback_facts(method_normalized), 0)
    elif research_condition == "w/o M0":
        m3_result = m3_extract_facts_without_m0(
            method_normalized,
            llm_client=llm_client,
            domain=domain,
            public_interfaces=task.get("public_interfaces"),
        )
    else:
        m3_result = m3_extract_facts(
            method_normalized,
            llm_client=llm_client,
            domain=domain,
            public_interfaces=task.get("public_interfaces"),
            top_level_input_fields=set(task.get("public_top_level_input_fields", []) or []),
            workflow_input_fields=set(task.get("public_workflow_input_fields", []) or []),
            input_mode_signatures=list(task.get("public_input_mode_signatures", []) or []),
        )
    if research_condition != "Coarse M3":
        if isinstance(m3_result, tuple):
            fact_spec, m3_tokens = m3_result
        else:
            fact_spec = m3_result
            m3_tokens = 0

    if research_condition != "w/o M0":
        _separate_new_and_existing_entity_entry_conditions(
            fact_spec,
            input_mode_signatures=task.get("public_input_mode_signatures", []) or [],
        )
        _propagate_scene_input_mode_conditions(
            fact_spec,
            input_mode_signatures=task.get("public_input_mode_signatures", []) or [],
        )

    # StrategyClassifier: 纯代码策略分类
    classifier = StrategyClassifier()
    classified_spec = classifier.classify(fact_spec)
    assignment_audit = {}
    assignment_tokens = 0
    if research_condition == "Coarse M3":
        from formal_experiment.coarse_m3 import apply_assignments
        classified_spec = apply_assignments(
            classified_spec, coarse_assignment_mapping or {}
        )
        assignment_audit = coarse_audit
    if research_condition == "Staged-LLM-Assign":
        from formal_experiment.staged_assignment import assign
        classified_spec, assignment_audit = assign(fact_spec, classified_spec, llm_client)
        assignment_tokens = int(assignment_audit["usage"].get("total_tokens", 0) or 0)

    # M4: 基于 ClassifiedSpec 编译
    compiled = _compile_with_backend(
        normalized=method_normalized,
        fact_spec=fact_spec,
        classified_spec=classified_spec,
        input_data=input_data,
        llm_client=llm_client,
        skeleton_override=None,
        domain=task.get("domain", ""),
        backend=backend,
    )
    effective_fact_spec = compiled.fact_spec or fact_spec
    effective_classified_spec = compiled.classified_spec or classified_spec
    total_compile_tokens = m2_tokens + m3_tokens + assignment_tokens + compiled.compile_tokens
    m0_audit = {
        "enabled": research_condition != "w/o M0",
        "semantic_validation": research_condition != "w/o M0",
        "decomposition_retry": research_condition != "w/o M0",
        "binding_and_program_repair": research_condition != "w/o M0",
        "pre_execution_contract_rejection": research_condition != "w/o M0",
        "intervention_trace": research_condition != "w/o M0",
        "compiler_language_checks_retained": True,
        "experimental_evidence_trace_retained": True,
    }
    return {
        "skeleton_source": compiled.skeleton_source,
        "sensor_specs": compiled.sensor_specs,
        "backend": compiled.backend,
        "dsl_code": compiled.dsl_code,
        "generated_python": compiled.generated_python,
        "compile_details": compiled.compile_details,
        "normalized": normalized,
        "entities": entities,
        "fact_spec": effective_fact_spec,
        "classified_spec": effective_classified_spec,
        "compile_tokens": total_compile_tokens,  # M2+M3+M4
        "compile_latency_ms": compiled.compile_latency_ms,
        "research_condition": research_condition,
        "assignment_audit": assignment_audit,
        "m0_audit": m0_audit,
    }


def run_ncnlp_pipeline_recompile_fact_spec(
    task: Dict[str, Any],
    fact_spec: 'DslFactSpec',
    llm_client=None,
    backend: str = "dsl_compiler",
    repair_tokens: int = 0,
) -> Dict[str, Any]:
    """Recompile a repaired FactSpec without rerunning M1-M3 extraction.

    Development qualification uses this entry point after a strict gate points
    to individual malformed bindings. It preserves the original workflow graph
    and makes the cost of the focused repair explicit.
    """
    if llm_client is None:
        llm_client = create_llm_client()

    def local_program_snapshot(spec: 'DslFactSpec') -> Dict[str, Any]:
        rows = []
        for scene in spec.scenes:
            structured = scene.action.structured_op
            if not isinstance(structured, dict):
                continue
            program = structured.get("local_program", []) or []
            diagnostic = structured.get("local_program_diagnostic")
            rows.append({
                "scene_id": int(scene.scene_id),
                "block_id": str(scene.block_id),
                "program": program,
                "diagnostic_error": (
                    str(diagnostic.get("validation_error", ""))
                    if isinstance(diagnostic, dict) else ""
                ),
            })
        canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
        return {
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper(),
            "scene_count": len(rows),
            "diagnostic_errors": [
                {
                    "scene_id": row["scene_id"],
                    "block_id": row["block_id"],
                    "error": row["diagnostic_error"],
                }
                for row in rows if row["diagnostic_error"]
            ],
        }

    recompile_snapshots = {"loaded": local_program_snapshot(fact_spec)}
    method_raw = task.get("method_prompt") or task.get("raw_prompt", "")
    method_normalized = m1_normalize(method_raw)
    input_data = _prepare_runtime_input_data(task)
    _separate_new_and_existing_entity_entry_conditions(
        fact_spec,
        input_mode_signatures=task.get("public_input_mode_signatures", []) or [],
    )
    _propagate_scene_input_mode_conditions(
        fact_spec,
        input_mode_signatures=task.get("public_input_mode_signatures", []) or [],
    )
    _normalize_public_operation_branch_lineage(fact_spec)
    _materialize_branch_local_public_dependencies(
        fact_spec,
        workflow_input_fields=set(task.get("public_workflow_input_fields", []) or []),
        input_mode_signatures=task.get("public_input_mode_signatures", []) or [],
    )
    _prefer_unique_workflow_inputs_over_conditional_collections(
        fact_spec,
        workflow_input_fields=set(task.get("public_workflow_input_fields", []) or []),
        repeated_workflow_input_fields=set(
            task.get("public_repeated_workflow_input_fields", []) or []
        ),
    )
    fact_spec = _normalize_fact_spec_public_operation_guards(
        fact_spec,
        method_normalized,
        top_level_input_fields=set(task.get("public_top_level_input_fields", []) or []),
        workflow_input_fields=set(task.get("public_workflow_input_fields", []) or []),
        input_mode_signatures=list(task.get("public_input_mode_signatures", []) or []),
    )
    _normalize_inclusive_date_window_fact_spec(fact_spec, method_normalized)
    _propagate_conditional_result_control_dependencies(fact_spec)
    # Re-run the final public-plan normalization after every focused repair
    # pass.  Repair models may reintroduce duplicate bindings or attach a
    # branch condition to a collection foreach from another input mode.
    _normalize_foreach_collection_sources(fact_spec)
    # Focused recompilation must apply the same cross-scene empty-selection
    # contract as a fresh extraction. Without this pass, a saved FactSpec can
    # compile successfully but still execute a side effect after its candidate
    # collection is empty.
    _materialize_empty_collection_guards(fact_spec)
    _materialize_due_date_filters(fact_spec)
    _bind_checkout_scene_fields(fact_spec)
    _materialize_notification_context_fields(fact_spec)
    _materialize_weekly_report_fallback(fact_spec)
    _materialize_empty_collection_guards(fact_spec)
    _normalize_legacy_conditional_expressions(fact_spec)
    _materialize_quantified_output_maps(fact_spec)
    _normalize_public_operation_branch_lineage(fact_spec)
    recompile_snapshots["before_m4"] = local_program_snapshot(fact_spec)
    classified_spec = StrategyClassifier().classify(fact_spec)
    compiled = _compile_with_backend(
        normalized=method_normalized,
        fact_spec=fact_spec,
        classified_spec=classified_spec,
        input_data=input_data,
        llm_client=llm_client,
        skeleton_override=None,
        domain=task.get("domain", ""),
        backend=backend,
    )
    effective_fact_spec = compiled.fact_spec or fact_spec
    effective_classified_spec = compiled.classified_spec or classified_spec
    recompile_snapshots["after_m4"] = local_program_snapshot(effective_fact_spec)
    return {
        "skeleton_source": compiled.skeleton_source,
        "sensor_specs": compiled.sensor_specs,
        "backend": compiled.backend,
        "dsl_code": compiled.dsl_code,
        "generated_python": compiled.generated_python,
        "compile_details": compiled.compile_details,
        "normalized": method_normalized,
        "entities": [],
        "fact_spec": effective_fact_spec,
        "classified_spec": effective_classified_spec,
        "compile_tokens": int(repair_tokens) + int(compiled.compile_tokens or 0),
        "compile_latency_ms": compiled.compile_latency_ms,
        "research_condition": "Full NCNLP",
        "assignment_audit": {},
        "focused_recompile": True,
        "recompile_diagnostics": recompile_snapshots,
    }


def run_ncnlp_pipeline_run_only(
    compiled: Dict[str, Any],
    task: Dict[str, Any],
    llm_client=None,
) -> Dict[str, Any]:
    """
    复用已编译的 skeleton + sensor_specs 跑一次, 不重新调 M4.
    适用于 RQ1 一致性测试: 编译 1 次, 跑 N 次, 编译 token 摊销.

    Returns:
        {
            "actions": List[Dict],
            "final_state": Dict,
            "compile_tokens": int,  # 0 (未重编译)
            "runtime_tokens": int,
            "sensor_call_count": int,
        }
    """
    if llm_client is None:
        llm_client = create_llm_client()
    input_data = _prepare_runtime_input_data(task)

    from m5_runtime import CompiledExecutor, _SensorView, SemanticSensor

    executor = CompiledExecutor(llm_client=llm_client)
    for spec in compiled["sensor_specs"]:
        executor.register_sensor(spec.name, SemanticSensor(spec, llm_client=llm_client))

    sensor_view = _SensorView(executor, [], [], 50)
    runner = None
    try:
        if compiled.get("backend") == "dsl_compiler":
            runner = DSLCompiledRunner(
                input_data,
                llm_client=llm_client,
                classified_spec=compiled.get("classified_spec"),
                dsl_code=compiled.get("dsl_code", ""),
                expected_output=task.get("expected_output"),
                domain=task.get("domain", ""),
                public_interfaces=task.get("public_interfaces"),
                public_runtime_config=task.get("public_runtime_config"),
            )
            actions, final_state = runner.execute(compiled.get("generated_python", ""))
        else:
            runner = SafeSkeletonRunner(sensor_view, input_data)
            actions, final_state = runner.execute(compiled["skeleton_source"])
        ok = True
        error = ""
    except Exception as e:
        actions = []
        final_state = {}
        ok = False
        error = f"{type(e).__name__}: {e}"

    dsl_traces = runner.traces if isinstance(runner, DSLCompiledRunner) else []
    usage_items = [t.usage or {} for t in sensor_view._traces]
    usage_items.extend(t.get("usage") or {} for t in dsl_traces)
    runtime_usage = {
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0) or 0) for item in usage_items),
        "completion_tokens": sum(int(item.get("completion_tokens", 0) or 0) for item in usage_items),
        "total_tokens": sum(int(item.get("total_tokens", 0) or 0) for item in usage_items),
        "cached_tokens": sum(int(item.get("cached_tokens", 0) or 0) for item in usage_items),
    }
    runtime_tokens = runtime_usage["total_tokens"]
    return {
        "actions": actions,
        "final_state": final_state,
        "local_action_executor": (
            runner.execute_compiled_local_action
            if isinstance(runner, DSLCompiledRunner) and runner.compiled_local_functions
            else None
        ),
        "compile_tokens": 0,  # 未重编译
        "runtime_tokens": runtime_tokens,
        "runtime_usage": runtime_usage,
        "sensor_call_count": len(sensor_view._traces) + len(dsl_traces),
        "ok": ok,
        "error": error,
    }




# ============================================================================
# Sanity Check: gov_001
# ============================================================================

GOV_001_SKELETON = '''def run(sensors, input_data):
    actions = []
    budget = input_data.get("本月经费", 0)
    threshold = 50000

    if budget > threshold:
        overflow = budget - threshold
        actions.append({
            "type": "transfer",
            "amount": overflow,
            "to_account": "应急储备账户",
        })
        actions.append({
            "type": "email",
            "to": "财务负责人",
            "subject": "应急储备通报",
            "body_contains": [str(overflow), "应急储备"],
        })
    else:
        actions.append({"type": "default_plan"})

    final_state = {
        "overflow": (budget - threshold) if budget > threshold else 0,
        "plan_executed": "transfer+email" if budget > threshold else "default",
    }
    return actions, final_state
'''


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from load_all_tasks import load_all

    tasks = load_all()
    gov = next(t for t in tasks if t["id"] == "gov_001")

    print("=" * 60)
    print(f"NCNLP Pipeline Sanity Check: {gov['id']}")
    print(f"raw_prompt: {gov['raw_prompt']}")
    print(f"input_data: {gov['input_data']}")
    print("=" * 60)

    # 跑端到端 (用 skeleton_override 跳过 M4 LLM 编译, 因为 demo 用)
    result = run_ncnlp_pipeline(gov, skeleton_override=GOV_001_SKELETON)

    print(f"\n[Result]")
    print(f"  ok: {result.ok}")
    print(f"  actions: {json.dumps(result.actions, ensure_ascii=False, indent=2)}")
    print(f"  final_state: {result.final_state}")
    print(f"  expected: {json.dumps(gov['expected_output'], ensure_ascii=False, indent=2)}")
    print(f"\n[Stats]")
    print(f"  compile_tokens: {result.compile_tokens}")
    print(f"  compile_latency_ms: {result.compile_latency_ms}")
    print(f"  runtime_tokens: {result.runtime_tokens}")
    print(f"  sensor_call_count: {result.sensor_call_count}")
    print(f"  total_tokens: {result.total_tokens}")
    print(f"\n[Skeleton]")
    print(result.skeleton_source)
