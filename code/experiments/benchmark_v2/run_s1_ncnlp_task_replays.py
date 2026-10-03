#!/usr/bin/env python3
"""Compile one S1 development task once and replay every case three times."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from n8n_h1_mock_executor import MockPlanExecutor
from n8n_h1_oracle_runtime import evaluate_case, replay_is_deterministic
from n8n_h1_unified_tasks import adapt_to_legacy_ncnlp, load_system_view
from workflow_condition import condition_leaves, validate_condition_tree


BENCHMARK = Path(__file__).resolve().parent
EXPERIMENT = BENCHMARK.parent
PROJECT = EXPERIMENT.parent
for search_root in (PROJECT, EXPERIMENT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

S1 = BENCHMARK / "n8n_native_s1" / "source_review_round_v2"
TASKS = S1 / "runtime" / "unified_tasks_v5_candidate_r5" / "development_qualification"
ORACLE = S1 / "d10_gold_oracle_round_v1" / "frozen" / "oracle_spec_v4" / "oracle"
TRACE_ROOT = S1 / "runtime" / "oracle_traces_v3" / "NCNLP"
ARTIFACT_ROOT = S1 / "runtime" / "ncnlp_artifacts_v3"
REPORT_ROOT = S1 / "runtime" / "reports"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def serialize(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {key: serialize(item) for key, item in value.__dict__.items()}
    if isinstance(value, list):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    return value


def load_fact_spec(path: Path) -> Any:
    """Hydrate a saved FactSpec so qualification can resume without re-extraction."""
    from dsl_v2.fact_types import (
        ActionSpec,
        CompressibilityScore,
        ConfigItem,
        ExclusionSpec,
        FactScene,
        FactSpec,
        PolicyItem,
        SemanticVector,
        TriggerSpec,
    )

    payload = read_json(path)
    scenes = []
    for item in payload.get("scenes", []) or []:
        scene = dict(item)
        scene["triggers"] = TriggerSpec(**(scene.get("triggers") or {}))
        scene["exclusions"] = ExclusionSpec(**(scene.get("exclusions") or {}))
        scene["action"] = ActionSpec(**(scene.get("action") or {}))
        scene["compressibility"] = CompressibilityScore(
            **(scene.get("compressibility") or {})
        )
        scene["semantic_vector"] = SemanticVector(
            **(scene.get("semantic_vector") or {})
        )
        scenes.append(FactScene(**scene))
    root = dict(payload)
    root["scenes"] = scenes
    root["configs"] = [ConfigItem(**item) for item in root.get("configs", []) or []]
    root["policies"] = [PolicyItem(**item) for item in root.get("policies", []) or []]
    return FactSpec(**root)


def audit_public_action_contract(
    dsl_code: str,
    public_interfaces: list[dict[str, Any]],
    top_level_input_fields: set[str] | None = None,
    workflow_input_fields: set[str] | None = None,
    repeated_workflow_input_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Require explicit DSL-to-runtime operation bindings before replay.

    The legacy runtime can heuristically associate scene text with operations.
    That behavior is useful for engineering diagnostics, but is not admissible
    for the strict paper experiment because it does not prove that M3/M4
    preserved workflow control flow.
    """
    prefix = "# ACTION_CONTRACT_JSON:"
    contract: dict[str, Any] = {}
    for line in (dsl_code or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        try:
            parsed = json.loads(stripped[len(prefix):].strip())
            contract = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            contract = {}
        break

    catalog = {
        (str(item.get("dependency_type", "")), str(item.get("operation", "")))
        for item in public_interfaces
        if item.get("dependency_type") and item.get("operation")
    }
    interface_by_key = {
        (str(item.get("dependency_type", "")), str(item.get("operation", ""))): item
        for item in public_interfaces
        if item.get("dependency_type") and item.get("operation")
    }
    declared: set[tuple[str, str]] = set()
    malformed_bindings: list[dict[str, Any]] = []
    schema_mismatches: list[dict[str, Any]] = []
    dataflow_mismatches: list[dict[str, Any]] = []
    local_program_errors: list[dict[str, Any]] = []
    cross_scene_guard_diagnostics: list[dict[str, Any]] = []
    local_result_dataflow_diagnostics: list[dict[str, Any]] = []
    top_level_input_fields = set(top_level_input_fields or set())
    workflow_input_fields = set(workflow_input_fields or set())
    repeated_workflow_input_fields = set(repeated_workflow_input_fields or set())

    def required_fields(schema: Any) -> set[str]:
        if not isinstance(schema, dict):
            return set()
        direct = {str(item) for item in schema.get("required", []) or []}
        alternatives = [
            {str(value) for value in item.get("required", []) or []}
            for item in schema.get("anyOf", []) + schema.get("oneOf", [])
            if isinstance(item, dict)
        ]
        return direct | (set.intersection(*alternatives) if alternatives else set())

    def unresolved_argument_source(source: Any, available: set[str]) -> bool:
        if isinstance(source, dict):
            if set(source) == {"literal"}:
                return False
            return not source or any(
                unresolved_argument_source(value, available) for value in source.values()
            )
        if isinstance(source, list):
            return not source or any(unresolved_argument_source(item, available) for item in source)
        if not isinstance(source, str) or not source.strip():
            return True
        path = source.strip().removeprefix("$")
        if path.startswith("result."):
            match = re.fullmatch(
                r"result\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)(?:(?:\.[A-Za-z_][A-Za-z0-9_]*)|(?:\[\d+\]))*",
                path,
            )
            return match is None or f"result.{match.group(1)}" not in available
        if path.startswith("input."):
            match = re.fullmatch(
                r"input\.[A-Za-z_][A-Za-z0-9_]*(?:(?:\.[A-Za-z_][A-Za-z0-9_]*)|(?:\[\d+\]))*",
                path,
            )
            root = path[len("input."):].split(".", 1)[0].split("[", 1)[0]
            return match is None or (bool(top_level_input_fields) and root not in top_level_input_fields)
        if path.startswith("workflow_input."):
            match = re.fullmatch(
                r"workflow_input\.[A-Za-z_][A-Za-z0-9_]*(?:(?:\.[A-Za-z_][A-Za-z0-9_]*)|(?:\[\d+\]))*",
                path,
            )
            root = path[len("workflow_input."):].split(".", 1)[0].split("[", 1)[0]
            return (
                match is None
                or root in repeated_workflow_input_fields
                or (bool(workflow_input_fields) and root not in workflow_input_fields)
            )
        root = re.split(r"[.\[]", path, maxsplit=1)[0]
        if path.startswith("user_input."):
            return True
        if root in {"context", "workflow"}:
            remainder = path.split(".", 1)
            root = re.split(r"[.\[]", remainder[1], maxsplit=1)[0] if len(remainder) == 2 else ""
        # The runtime supports unique recursive lookup for bare fields that
        # are present in the public workflow-input catalog.  Keep that
        # contract distinct from undeclared generic DSL inputs such as
        # ``history``: the former is admissible, the latter is not.
        if root in workflow_input_fields and root not in repeated_workflow_input_fields:
            return False
        return root not in available

    def collect_refs(value: Any) -> list[str]:
        if isinstance(value, list):
            return [ref for item in value for ref in collect_refs(item)]
        if not isinstance(value, dict):
            return []
        refs = []
        if set(value) == {"ref"} and isinstance(value.get("ref"), str):
            refs.append(value["ref"])
        for item in value.values():
            refs.extend(collect_refs(item))
        return refs

    def source_root(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        path = value.strip().removeprefix("$")
        return re.split(r"[.\[]", path, maxsplit=1)[0]

    input_match = re.search(r"(?ms)^INPUTS:\s*(.*?)^ENDINPUTS", dsl_code or "")
    available_variables = {
        match.group(1)
        for match in re.finditer(
            r"^\s*(?:REQUIRED|OPTIONAL)\s+\{\{([^}]+)\}\}",
            input_match.group(1) if input_match else "",
            re.MULTILINE,
        )
    }
    # The DSL may declare generic convenience inputs such as ``history`` or
    # ``user_input``.  They are not automatically part of the benchmark's
    # public input contract.  When the real task input catalog is available,
    # keep only fields proven by that catalog; later scenes may still add
    # their own produced variables through the normal dataflow pass.
    if top_level_input_fields or workflow_input_fields:
        available_variables.intersection_update(top_level_input_fields)
    nested_only_inputs = workflow_input_fields - top_level_input_fields
    produced_variables: set[str] = set()
    contract_version = int(contract.get("version", 0) or 0) if contract else 0
    local_action_types = {
        "compute", "format", "template", "template_fill", "transform", "validation",
    }
    contract_scenes = (
        contract.get("scenes", []) if isinstance(contract.get("scenes"), list) else []
    )

    # Diagnostic only: identify a later scene whose first public operation
    # inherits a cross-scene guard without consuming the predecessor result.
    # Some workflows intentionally sequence side effects this way, so this is
    # reported separately and does not reject the contract yet.
    operation_locations: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for candidate_index, candidate_scene in enumerate(contract_scenes):
        if not isinstance(candidate_scene, dict):
            continue
        for candidate_binding in candidate_scene.get("public_operations", []) or []:
            if not isinstance(candidate_binding, dict):
                continue
            candidate_operation = str(candidate_binding.get("operation", ""))
            if candidate_operation:
                operation_locations.setdefault(candidate_operation, []).append(
                    (candidate_index, candidate_binding)
                )

    def contract_reference_roots(value: Any) -> set[str]:
        roots: set[str] = set()
        if isinstance(value, list):
            for item in value:
                roots.update(contract_reference_roots(item))
        elif isinstance(value, dict):
            if set(value) == {"literal"}:
                return roots
            if set(value) == {"ref"} and isinstance(value.get("ref"), str):
                roots.add(re.split(r"[.\[]", value["ref"].removeprefix("$"), maxsplit=1)[0])
            else:
                for item in value.values():
                    roots.update(contract_reference_roots(item))
        elif isinstance(value, str):
            roots.add(re.split(r"[.\[]", value.removeprefix("$"), maxsplit=1)[0])
        return roots

    def contract_reference_strings(value: Any) -> set[str]:
        refs: set[str] = set()
        if isinstance(value, list):
            for item in value:
                refs.update(contract_reference_strings(item))
        elif isinstance(value, dict):
            if set(value) == {"ref"} and isinstance(value.get("ref"), str):
                refs.add(value["ref"])
            else:
                for item in value.values():
                    refs.update(contract_reference_strings(item))
        return refs

    def operation_phase(operation: str) -> int:
        name = operation.lower()
        tokens = set(re.split(r"[^a-z0-9]+", name))
        if any(token in tokens for token in ("discovery", "discover", "initiate")) or "start" in tokens:
            return 0
        if "poll" in tokens or "status" in tokens:
            return 1
        if "download" in tokens:
            return 2
        if "upload" in tokens or "register" in tokens:
            return 6
        if tokens & {"read", "load", "list", "fetch", "search"}:
            return 0
        if "scrape" in tokens:
            return 1
        if tokens & {"render", "compose", "polish"}:
            return 3
        if "interact" in tokens:
            return 2
        if tokens & {"research", "extract", "generate", "analyze"}:
            return 3
        if "format" in tokens:
            return 4
        if tokens & {"create", "send", "post", "publish", "update", "append"}:
            return 6
        return 2

    for scene_index, candidate_scene in enumerate(contract_scenes):
        if not isinstance(candidate_scene, dict):
            continue
        candidate_bindings = candidate_scene.get("public_operations", []) or []
        if not isinstance(candidate_bindings, list) or not candidate_bindings:
            continue
        candidate_binding = candidate_bindings[0]
        if not isinstance(candidate_binding, dict):
            continue
        guard = str(candidate_binding.get("guard", "") or "")
        if not guard.startswith(("on_success:", "on_failure:")):
            continue
        predecessor = guard.split(":", 1)[1]
        predecessor_locations = operation_locations.get(predecessor, [])
        predecessor_scene_indexes = [index for index, _ in predecessor_locations]
        if not predecessor_scene_indexes or max(predecessor_scene_indexes) >= scene_index:
            continue
        if operation_phase(str(candidate_binding.get("operation", ""))) >= operation_phase(predecessor):
            continue
        candidate_refs = contract_reference_roots({
            "arguments": candidate_binding.get("arguments", {}),
            "condition": candidate_binding.get("condition", {}),
            "foreach": candidate_binding.get("foreach", {}),
        })
        if {f"result.{predecessor}", predecessor} & candidate_refs:
            continue
        cross_scene_guard_diagnostics.append({
            "scene_index": scene_index,
            "scene_id": candidate_scene.get("scene_id"),
            "block_id": candidate_scene.get("block_id"),
            "operation": candidate_binding.get("operation"),
            "guard": guard,
            "predecessor_scene_indexes": predecessor_scene_indexes,
            "reason": "first_operation_in_later_scene_inherits_cross_scene_guard_without_data_dependency",
        })

    # Track collections created by an explicit filter and simple maps derived
    # from them. If a later operation follows the producer's success edge but
    # iterates the unfiltered producer output, the DSL has bypassed a control-
    # flow decision expressed by its own local program.
    filtered_targets_by_producer: dict[str, set[str]] = {}
    filter_scene_by_producer: dict[str, int] = {}
    for candidate_index, candidate_scene in enumerate(contract_scenes):
        if not isinstance(candidate_scene, dict):
            continue
        local_program = candidate_scene.get("local_program", []) or []
        for statement in local_program:
            if not isinstance(statement, dict):
                continue
            target = statement.get("target")
            if not isinstance(target, str) or not target:
                continue
            if statement.get("op") == "filter":
                producer_operations = {
                    match.group(1)
                    for ref in collect_refs(statement)
                    if (match := re.match(
                        r"^result\.([A-Za-z_][A-Za-z0-9_]*)\.", ref
                    ))
                }
                for operation in producer_operations:
                    filtered_targets_by_producer.setdefault(operation, set()).add(target)
                    filter_scene_by_producer.setdefault(operation, candidate_index)
                continue
            if statement.get("op") not in {"assign", "map"}:
                continue
            lineage_input = (
                statement.get("source")
                if statement.get("op") == "map"
                else statement.get("value")
            )
            statement_sources = {
                source_root(ref) for ref in collect_refs(lineage_input)
            }
            for operation, derived_targets in filtered_targets_by_producer.items():
                if statement_sources & derived_targets:
                    derived_targets.add(target)
    workflow_public_targets = {
        str(target)
        for candidate_scene in contract_scenes
        if isinstance(candidate_scene, dict)
        for binding in (candidate_scene.get("public_operations", []) or [])
        if isinstance(binding, dict)
        for target in (binding.get("results", {}) or {}).values()
        if isinstance(target, str) and target
    }

    # A scene may describe an output before the public operation that actually
    # produces it. Treat that as an ordering defect instead of making the
    # declaration itself satisfy downstream data flow.
    public_target_scene_indexes: dict[str, list[int]] = {}
    for candidate_index, candidate_scene in enumerate(contract_scenes):
        if not isinstance(candidate_scene, dict):
            continue
        for binding in candidate_scene.get("public_operations", []) or []:
            if not isinstance(binding, dict):
                continue
            for target in (binding.get("results", {}) or {}).values():
                if isinstance(target, str) and target:
                    public_target_scene_indexes.setdefault(
                        target.split(".", 1)[0], []
                    ).append(candidate_index)

    for scene_index, scene in enumerate(contract_scenes):
        bindings = scene.get("public_operations", [])
        if not isinstance(bindings, list):
            malformed_bindings.append({
                "scene_id": scene.get("scene_id"),
                "reason": "public_operations_must_be_a_list",
            })
            continue
        # Public-operation arguments may use the runtime's unique recursive
        # lookup for bare nested fields. Compiled local programs cannot; only
        # real top-level inputs and variables produced by earlier scenes are
        # valid bare references there.
        scene_available_before = (
            set(available_variables) - nested_only_inputs
        ) | produced_variables
        scene_local_program = scene.get("local_program", []) or []

        # A local program may only read a public operation result produced in
        # the same scene or an earlier scene.  Reading a result from a later
        # scene is a deterministic dataflow error, not a runtime fluctuation.
        for reference in sorted(contract_reference_strings(scene_local_program)):
            match = re.fullmatch(
                r"\$?result\.([A-Za-z_][A-Za-z0-9_]*)\..+", reference
            )
            if not match:
                continue
            operation = match.group(1)
            locations = operation_locations.get(operation, [])
            prior_or_current_indexes = [index for index, _ in locations if index <= scene_index]
            future_indexes = [index for index, _ in locations if index > scene_index]
            if future_indexes and not prior_or_current_indexes:
                local_result_dataflow_diagnostics.append({
                    "scene_id": scene.get("scene_id"),
                    "block_id": scene.get("block_id"),
                    "reference": reference,
                    "operation": operation,
                    "producer_scene_indexes": [index for index, _ in locations],
                    "reason": "local_program_reads_public_result_from_later_scene",
                })

        repeated_global_refs = sorted({
            ref for ref in collect_refs(scene_local_program)
            if ref.startswith("workflow_input.")
            and ref[len("workflow_input."):].split(".", 1)[0].split("[", 1)[0]
            in repeated_workflow_input_fields
        })
        if repeated_global_refs:
            local_program_errors.append({
                "scene_id": scene.get("scene_id"),
                "block_id": scene.get("block_id"),
                "reason": "local_program_reads_repeated_field_without_iteration",
                "references": repeated_global_refs,
            })
        local_roots_before_binding: dict[int, set[str]] = {}
        all_local_roots: set[str] = set()
        schedule_events: list[dict[str, Any]] = []
        if scene_local_program:
            try:
                from pipeline_runner import (
                    _local_statement_targets,
                    _schedule_scene_action_units,
                )
                schedule_events = _schedule_scene_action_units(scene)
                for event in schedule_events:
                    if event["type"] == "local":
                        for statement in event["statements"]:
                            all_local_roots.update(_local_statement_targets(statement))
                    else:
                        # The compiler materializes each executable local
                        # segment with carry fields, so targets produced by a
                        # completed preceding segment are visible here.
                        local_roots_before_binding[event["index"]] = set(all_local_roots)
            except Exception as exc:
                local_program_errors.append({
                    "scene_id": scene.get("scene_id"),
                    "block_id": scene.get("block_id"),
                    "reason": f"local_public_schedule_error:{type(exc).__name__}:{exc}",
                })

        declared_scene_outputs = {
            str(output).split(".", 1)[0]
            for output in scene.get("outputs", []) or []
            if output
        }
        current_public_targets = {
            str(target).split(".", 1)[0]
            for binding in bindings
            if isinstance(binding, dict)
            for target in (binding.get("results", {}) or {}).values()
            if isinstance(target, str) and target
        }
        current_local_targets = {
            str(target).split(".", 1)[0] for target in all_local_roots if target
        }
        for output in sorted(
            declared_scene_outputs - current_public_targets - current_local_targets
        ):
            later_indexes = public_target_scene_indexes.get(output, [])
            if any(index > scene_index for index in later_indexes):
                dataflow_mismatches.append({
                    "scene_id": scene.get("scene_id"),
                    "dependency_type": "",
                    "operation": "",
                    "reason": "scene_output_is_produced_only_by_later_scene",
                    "output": output,
                })

        for binding_index, binding in enumerate(bindings):
            available_variables.update(local_roots_before_binding.get(binding_index, set()))
            if not isinstance(binding, dict):
                malformed_bindings.append({
                    "scene_id": scene.get("scene_id"),
                    "reason": "binding_must_be_an_object",
                    "binding": binding,
                })
                continue
            key = (
                str(binding.get("dependency_type", "")),
                str(binding.get("operation", "")),
            )
            if not all(key):
                malformed_bindings.append({
                    "scene_id": scene.get("scene_id"),
                    "reason": "binding_identity_incomplete",
                    "binding": binding,
                })
                continue
            declared.add(key)
            interface = interface_by_key.get(key)
            if interface is not None:
                request_schema = interface.get("request_schema", {}) or {}
                response_schema = interface.get("response_schema", {}) or {}
                required_request = required_fields(request_schema)
                required_response = required_fields(response_schema)
                consumes = {str(item) for item in binding.get("consumes", []) or []}
                produces = {str(item) for item in binding.get("produces", []) or []}
                missing_request = sorted(required_request - consumes)
                missing_response = sorted(required_response - produces)
                if missing_request or missing_response:
                    schema_mismatches.append({
                        "scene_id": scene.get("scene_id"),
                        "dependency_type": key[0],
                        "operation": key[1],
                        "missing_required_consumes": missing_request,
                        "missing_required_produces": missing_response,
                    })
                if contract_version >= 4:
                    arguments = binding.get("arguments", {})
                    results = binding.get("results", {})
                    argument_keys = set(arguments) if isinstance(arguments, dict) else set()
                    result_keys = set(results) if isinstance(results, dict) else set()
                    missing_argument_maps = sorted(required_request - argument_keys)
                    missing_result_maps = sorted(required_response - result_keys)
                    unresolved_sources: list[dict[str, Any]] = []
                    suspicious_sources: list[dict[str, Any]] = []
                    invalid_targets: list[dict[str, Any]] = []
                    if isinstance(arguments, dict):
                        for field, source in arguments.items():
                            if unresolved_argument_source(source, available_variables):
                                unresolved_sources.append({"field": field, "source": source})
                            if isinstance(source, str):
                                source_leaf = re.split(
                                    r"[.\[]", source.removeprefix("$").rstrip("]")
                                )[-1]
                                field_tokens = {
                                    token.removesuffix("s")
                                    for token in re.split(r"_+", str(field).lower())
                                    if token
                                }
                                source_tokens = {
                                    token.removesuffix("s")
                                    for token in re.split(r"_+", source_leaf.lower())
                                    if token
                                }
                                shared = field_tokens & source_tokens
                                semantic_qualifiers = {
                                    "count", "date", "id", "index", "length",
                                    "name", "number", "time", "total",
                                }
                                if (
                                    field_tokens != source_tokens
                                    and len(shared) >= 2
                                    and bool(
                                        (field_tokens ^ source_tokens)
                                        & semantic_qualifiers
                                    )
                                ):
                                    suspicious_sources.append({
                                        "field": field,
                                        "source": source,
                                        "reason": "argument_source_name_conflict",
                                    })
                    if isinstance(results, dict):
                        for field, target in results.items():
                            if not isinstance(target, str) or not re.fullmatch(
                                r"[A-Za-z_][A-Za-z0-9_.]*", target
                            ):
                                invalid_targets.append({"field": field, "target": target})
                    condition = binding.get("condition", {}) or {}
                    if condition:
                        condition_ok = isinstance(condition, dict)
                        try:
                            validate_condition_tree(condition)
                        except ValueError:
                            condition_ok = False
                        if condition_ok:
                            for leaf in condition_leaves(condition):
                                if unresolved_argument_source(leaf.get("source"), available_variables):
                                    condition_ok = False
                                    break
                                if "value_from" in leaf:
                                    value_from = leaf.get("value_from")
                                    if value_from != "foreach_item" and unresolved_argument_source(
                                        value_from, available_variables
                                    ):
                                        condition_ok = False
                                        break
                        if not condition_ok:
                            malformed_bindings.append({
                                "scene_id": scene.get("scene_id"),
                                "reason": "invalid_condition",
                                "binding": binding,
                            })
                    foreach = binding.get("foreach", {}) or {}
                    if foreach:
                        foreach_ok = (
                            isinstance(foreach, dict)
                            and isinstance(foreach.get("source"), str)
                            and foreach.get("argument") in argument_keys
                            and not unresolved_argument_source(foreach.get("source"), available_variables)
                        )
                        if not foreach_ok:
                            malformed_bindings.append({
                                "scene_id": scene.get("scene_id"),
                                "reason": "invalid_foreach",
                                "binding": binding,
                            })
                        guard = str(binding.get("guard", "") or "")
                        predecessor = (
                            guard.split(":", 1)[1]
                            if guard.startswith("on_success:") else ""
                        )
                        filtered_targets = filtered_targets_by_producer.get(
                            predecessor, set()
                        )
                        foreach_source = foreach.get("source")
                        if (
                            filtered_targets
                            and filter_scene_by_producer.get(predecessor, scene_index)
                            < scene_index
                            and foreach_source not in filtered_targets
                        ):
                            dataflow_mismatches.append({
                                "scene_id": scene.get("scene_id"),
                                "dependency_type": key[0],
                                "operation": key[1],
                                "reason": "filtered_collection_bypassed",
                                "predecessor_operation": predecessor,
                                "foreach_source": foreach.get("source"),
                                "filtered_sources": sorted(filtered_targets),
                            })
                    if (
                        missing_argument_maps or missing_result_maps
                        or unresolved_sources or suspicious_sources or invalid_targets
                    ):
                        dataflow_mismatches.append({
                            "scene_id": scene.get("scene_id"),
                            "dependency_type": key[0],
                            "operation": key[1],
                            "missing_argument_maps": missing_argument_maps,
                            "missing_result_maps": missing_result_maps,
                            "unresolved_sources": unresolved_sources,
                            "suspicious_sources": suspicious_sources,
                            "invalid_targets": invalid_targets,
                        })
                    if isinstance(results, dict):
                        available_variables.add(f"result.{key[1]}")
                        result_targets = {
                            str(target).split(".", 1)[0]
                            for target in results.values()
                            if isinstance(target, str) and target
                        }
                        available_variables.update(result_targets)
                        produced_variables.update(result_targets)
        action_type = str(scene.get("action_type", "")).lower()
        local_text = " ".join([
            str(scene.get("description", "")),
            str(scene.get("raw_requirement_text", "")),
            " ".join(str(item) for item in scene.get("outputs", []) or []),
            " ".join(str(item) for item in scene.get("side_effects", []) or []),
            " ".join(str(item) for item in scene.get("action_sequence", []) or []),
        ]).lower()
        local_step_compiled = (
            action_type in local_action_types
            or (action_type == "assign" and not bindings)
            or bool({
                str(output) for output in scene.get("outputs", []) or [] if output
            } - workflow_public_targets)
            or bool(scene.get("local_program_diagnostic"))
            or any(token in local_text for token in (
                "format", "template", "metric", "performance", "token",
                "排版", "格式", "指标", "耗时",
            ))
            or (
                any(token in local_text for token in (
                    "schema", "输出结构", "结构校验", "结构验证",
                ))
                and any(token in local_text for token in (
                    "validate", "validation", "校验", "验证", "检查",
                ))
            )
        )
        available_variables.update(all_local_roots)
        produced_variables.update(all_local_roots)
        if contract_version >= 6:
            if scene_local_program:
                try:
                    from dsl_v2.local_program import validate_local_program
                    validate_local_program(scene_local_program)
                    from pipeline_runner import _local_program_semantic_error
                    semantic_error = ""
                    ordered_available = set(scene_available_before)
                    for event in schedule_events:
                        if event["type"] == "public":
                            binding = bindings[event["index"]]
                            operation = str(binding.get("operation", ""))
                            if operation:
                                ordered_available.add(f"result.{operation}")
                                ordered_available.update(
                                    f"result.{operation}.{field}"
                                    for field in (binding.get("results", {}) or {})
                                )
                            ordered_available.update(
                                str(target).split(".", 1)[0]
                                for target in (binding.get("results", {}) or {}).values()
                                if isinstance(target, str) and target
                            )
                            continue
                        semantic_error = _local_program_semantic_error(
                            event["statements"],
                            bindings,
                            set(),
                            ordered_available,
                            "\n".join([
                                str(scene.get("raw_requirement_text", "")),
                                str(scene.get("logic_flow", "")),
                            ]),
                        )
                        if semantic_error:
                            break
                        ordered_available.update(
                            str(target).split(".", 1)[0]
                            for statement in event["statements"]
                            for target in _local_statement_targets(statement)
                            if target
                        )
                    if semantic_error:
                        # The scheduler may keep a history-alias assignment
                        # in the same executable segment as the filter, while
                        # the segment-level checker sees only the later
                        # predicate references.  Recheck the complete local
                        # program before rejecting it.  This removes only the
                        # proven history-alias false positive; syntax and
                        # genuine unresolved dataflow errors still reject.
                        if semantic_error == "local_program_history_filter_missing_history_comparison":
                            def has_explicit_history_pool(program: list[dict[str, Any]]) -> bool:
                                history_targets: set[str] = set()

                                def contains_history_token(value: Any) -> bool:
                                    if isinstance(value, list):
                                        return any(contains_history_token(item) for item in value)
                                    if isinstance(value, dict):
                                        if set(value) == {"ref"}:
                                            ref = str(value.get("ref", "")).lower()
                                            return any(token in ref for token in (
                                                "history", "previous", "already", "contacted", "record",
                                            ))
                                        return any(contains_history_token(item) for item in value.values())
                                    return False

                                def refs(value: Any) -> set[str]:
                                    if isinstance(value, list):
                                        return set().union(*(refs(item) for item in value))
                                    if isinstance(value, dict):
                                        if set(value) == {"ref"}:
                                            return {str(value.get("ref", ""))}
                                        return set().union(*(refs(item) for item in value.values()))
                                    return set()

                                for statement in program:
                                    if (
                                        isinstance(statement, dict)
                                        and statement.get("op") == "assign"
                                        and statement.get("target")
                                        and contains_history_token(statement.get("value"))
                                    ):
                                        history_targets.add(str(statement["target"]))
                                return any(
                                    str(target) in history_targets
                                    for statement in program
                                    if isinstance(statement, dict)
                                    and statement.get("op") == "filter"
                                    for target in refs(statement.get("where"))
                                )

                            full_program_error = _local_program_semantic_error(
                                scene_local_program,
                                bindings,
                                set(),
                                set(scene_available_before) | set(available_variables),
                                "\n".join([
                                    str(scene.get("raw_requirement_text", "")),
                                    str(scene.get("logic_flow", "")),
                                ]),
                            )
                            if not full_program_error or has_explicit_history_pool(scene_local_program):
                                semantic_error = ""
                    if semantic_error:
                        raise ValueError(semantic_error)
                except Exception as exc:
                    local_program_errors.append({
                        "scene_id": scene.get("scene_id"),
                        "block_id": scene.get("block_id"),
                        "reason": f"invalid_local_program:{type(exc).__name__}:{exc}",
                    })
                for statement in scene_local_program:
                    if not isinstance(statement, dict) or statement.get("op") != "emit":
                        continue
                    available_variables.update(
                        str(name).split(".", 1)[0]
                        for name in (statement.get("fields", {}) or {})
                        if name
                    )
            elif local_step_compiled:
                diagnostic = scene.get("local_program_diagnostic")
                local_program_errors.append({
                    "scene_id": scene.get("scene_id"),
                    "block_id": scene.get("block_id"),
                    "reason": "deterministic_local_step_has_no_executable_program",
                    "repair_diagnostic": diagnostic,
                })
        if 4 <= contract_version < 6 and local_step_compiled:
            available_variables.update(
                str(output).split(".", 1)[0]
                for output in scene.get("outputs", []) or []
                if output
            )

    missing = sorted(catalog - declared)
    unknown = sorted(declared - catalog)
    errors: list[str] = []
    if not contract:
        errors.append("action_contract_missing_or_invalid")
    if catalog and not declared:
        errors.append("explicit_public_operation_bindings_missing")
    if missing:
        errors.append("public_operation_catalog_not_fully_bound")
    if unknown:
        errors.append("unknown_public_operation_binding")
    if malformed_bindings:
        errors.append("malformed_public_operation_binding")
    if schema_mismatches:
        errors.append("public_operation_schema_binding_mismatch")
    if dataflow_mismatches:
        errors.append("public_operation_dataflow_binding_mismatch")
    if local_program_errors:
        errors.append("local_program_contract_mismatch")
    return {
        "passed": not errors,
        "gate": "strict_explicit_public_operation_binding_v3",
        "catalog_operation_count": len(catalog),
        "declared_operation_count": len(declared),
        "missing_operations": [
            {"dependency_type": dependency, "operation": operation}
            for dependency, operation in missing
        ],
        "unknown_operations": [
            {"dependency_type": dependency, "operation": operation}
            for dependency, operation in unknown
        ],
        "malformed_bindings": malformed_bindings,
        "schema_mismatches": schema_mismatches,
        "dataflow_mismatches": dataflow_mismatches,
        "local_program_errors": local_program_errors,
        "cross_scene_guard_diagnostics": cross_scene_guard_diagnostics,
        "local_result_dataflow_diagnostics": local_result_dataflow_diagnostics,
        "available_variables_after_audit": sorted(available_variables),
        "errors": errors,
    }


def public_input_field_catalog(tasks_root: Path, task_id: str) -> tuple[set[str], set[str]]:
    """Collect names only from public development inputs; never expose values or Oracle data."""
    top_level: set[str] = set()
    recursive: set[str] = set()

    def visit(value: Any, *, root: bool = False) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if root:
                    top_level.add(str(key))
                if key not in {"fixtures"} and not str(key).startswith("F-"):
                    recursive.add(str(key))
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    instance_root = tasks_root / "system_inputs" / task_id / "instances"
    for path in sorted(instance_root.glob("*.json")):
        visit(read_json(path).get("input_data", {}), root=True)
    return top_level, recursive


def public_input_schema_catalog(tasks_root: Path, task_id: str) -> dict[str, dict[str, Any]]:
    """Infer a value-free top-level type contract and reviewed status enums.

    Input types and control-state labels are part of the callable contract, not
    expected outputs.  Only status-like scalar fields expose their finite set
    of observed labels; arbitrary payload text never enters the compiler.
    """
    observed: dict[str, list[Any]] = {}
    instance_root = tasks_root / "system_inputs" / task_id / "instances"
    for path in sorted(instance_root.glob("*.json")):
        payload = read_json(path).get("input_data", {})
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            observed.setdefault(str(key), []).append(value)

    def type_name(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return type(value).__name__

    status_names = {"approval", "approved", "status", "state", "mode"}
    catalog: dict[str, dict[str, Any]] = {}
    for field, values in sorted(observed.items()):
        entry: dict[str, Any] = {
            "types": sorted({type_name(value) for value in values}),
        }
        if field.lower() in status_names and all(isinstance(value, str) for value in values):
            labels = sorted({value for value in values})
            if len(labels) <= 8:
                entry["enum"] = labels
        catalog[field] = entry
    return catalog


def public_input_mode_signatures(tasks_root: Path, task_id: str) -> list[list[str]]:
    """Return distinct public input shapes without values or case labels.

    Nested field names are part of the public request contract and are needed to
    distinguish entry modes that share a container such as ``fixtures`` or
    ``subscriber``.  Fixture identifiers are omitted because they identify the
    test harness rather than workflow fields.
    """
    def field_names(value: Any, *, root: bool = False) -> set[str]:
        names: set[str] = set()
        if isinstance(value, dict):
            for key, nested in value.items():
                key = str(key)
                if root or (key != "fixtures" and not key.startswith("F-")):
                    names.add(key)
                names.update(field_names(nested))
        elif isinstance(value, list):
            for nested in value:
                names.update(field_names(nested))
        return names

    instance_root = tasks_root / "system_inputs" / task_id / "instances"
    signatures = {
        tuple(sorted(field_names(read_json(path).get("input_data", {}), root=True)))
        for path in sorted(instance_root.glob("*.json"))
    }
    return [list(signature) for signature in sorted(signatures, key=lambda item: (len(item), item))]


def public_input_mode_profiles(tasks_root: Path, task_id: str) -> list[dict[str, list[str]]]:
    """Describe each public input mode by top-level and nested field names.

    A flat recursive signature loses the distinction between a direct request
    and a fixture-backed workflow mode.  The distinction is structural and
    contains no values, case labels, or oracle information, so it is safe to
    expose to the compiler as part of the public input contract.
    """
    def collect(value: Any, *, top_level: bool = False) -> tuple[set[str], set[str]]:
        top: set[str] = set()
        nested: set[str] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                key = str(key)
                if top_level:
                    top.add(key)
                elif key != "fixtures" and not key.startswith("F-"):
                    nested.add(key)
                child_top, child_nested = collect(item, top_level=False)
                nested.update(child_top)
                nested.update(child_nested)
        elif isinstance(value, list):
            for item in value:
                child_top, child_nested = collect(item, top_level=False)
                nested.update(child_top)
                nested.update(child_nested)
        return top, nested

    instance_root = tasks_root / "system_inputs" / task_id / "instances"
    profiles = {
        (
            tuple(sorted(collect(read_json(path).get("input_data", {}), top_level=True)[0])),
            tuple(sorted(collect(read_json(path).get("input_data", {}), top_level=True)[1])),
        )
        for path in sorted(instance_root.glob("*.json"))
    }
    return [
        {"top_level": list(top), "nested": list(nested)}
        for top, nested in sorted(profiles, key=lambda item: (len(item[0]) + len(item[1]), item))
    ]


def public_repeated_workflow_input_fields(tasks_root: Path, task_id: str) -> set[str]:
    """Collect public field names occurring inside arrays, without retaining values."""
    repeated: set[str] = set()

    def visit(value: Any, *, inside_array: bool = False) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                key = str(key)
                if inside_array and not key.startswith("F-"):
                    repeated.add(key)
                visit(nested, inside_array=inside_array)
        elif isinstance(value, list):
            for nested in value:
                visit(nested, inside_array=True)

    instance_root = tasks_root / "system_inputs" / task_id / "instances"
    for path in sorted(instance_root.glob("*.json")):
        visit(read_json(path).get("input_data", {}))
    return repeated


def qualify_unique_nested_workflow_refs(
    program: list[dict[str, Any]],
    top_level_inputs: set[str],
    workflow_inputs: set[str],
    repeated_workflow_fields: set[str] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Qualify unique nested input refs, including LLM repair proposals."""
    repeated_workflow_fields = set(repeated_workflow_fields or set())
    assigned: set[str] = set()
    changed = False

    def qualify(value: Any, loop_variables: set[str]) -> Any:
        nonlocal changed
        if isinstance(value, list):
            return [qualify(item, loop_variables) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"ref"} and isinstance(value.get("ref"), str):
            reference = value["ref"]
            root = re.split(r"[.\[]", reference, maxsplit=1)[0]
            if (
                "." not in reference
                and "[" not in reference
                and root in workflow_inputs
                and root not in top_level_inputs
                and root not in repeated_workflow_fields
                and root not in assigned
                and root not in loop_variables
            ):
                changed = True
                return {"ref": f"workflow_input.{reference}"}
            return value
        return {key: qualify(item, loop_variables) for key, item in value.items()}

    normalized_program: list[dict[str, Any]] = []
    for statement in program:
        loop_variables: set[str] = set()
        if isinstance(statement, dict) and statement.get("op") in {"filter", "map", "sort"}:
            for name in (statement.get("item"), statement.get("index")):
                if isinstance(name, str) and name:
                    loop_variables.add(name)
        normalized = qualify(statement, loop_variables)
        normalized_program.append(normalized)
        target = normalized.get("target") if isinstance(normalized, dict) else None
        if isinstance(target, str) and target:
            assigned.add(target)
    return normalized_program, changed


def repair_failed_fact_spec_nodes(
    fact_spec: Any,
    compile_gate: dict[str, Any],
    public_interfaces: list[dict[str, Any]],
    top_level_inputs: set[str],
    workflow_inputs: set[str],
    llm_client: Any,
    repeated_workflow_fields: set[str] | None = None,
) -> tuple[Any, int, dict[str, Any]]:
    """Repair only gate-identified bindings/local programs while preserving the graph."""
    repaired = copy.deepcopy(fact_spec)
    repeated_workflow_fields = set(repeated_workflow_fields or set())
    deterministic_local_updates = 0
    deterministically_updated_scenes: set[str] = set()

    # Qualify bare references that can only denote a unique nested public
    # input. This also repairs local-only scenes, which have no public binding
    # for the focused LLM repair to target.
    for scene in getattr(repaired, "scenes", []) or []:
        structured = getattr(getattr(scene, "action", None), "structured_op", None)
        program = structured.get("local_program", []) if isinstance(structured, dict) else []
        if not isinstance(program, list) or not program:
            continue
        assigned: set[str] = set()
        changed = False

        def qualify_refs(value: Any, loop_variables: set[str]) -> Any:
            nonlocal changed
            if isinstance(value, list):
                return [qualify_refs(item, loop_variables) for item in value]
            if not isinstance(value, dict):
                return value
            if set(value) == {"ref"} and isinstance(value.get("ref"), str):
                reference = value["ref"]
                root = re.split(r"[.\[]", reference, maxsplit=1)[0]
                if (
                    "." not in reference
                    and "[" not in reference
                    and root in workflow_inputs
                    and root not in top_level_inputs
                    and root not in repeated_workflow_fields
                    and root not in assigned
                    and root not in loop_variables
                ):
                    changed = True
                    return {"ref": f"workflow_input.{reference}"}
                return value
            return {
                key: qualify_refs(item, loop_variables)
                for key, item in value.items()
            }

        normalized_program = []
        for statement in program:
            loop_variables = set()
            if isinstance(statement, dict) and statement.get("op") in {"filter", "map", "sort"}:
                for name in (statement.get("item"), statement.get("index")):
                    if isinstance(name, str) and name:
                        loop_variables.add(name)
            normalized = qualify_refs(statement, loop_variables)
            normalized_program.append(normalized)
            target = normalized.get("target") if isinstance(normalized, dict) else None
            if isinstance(target, str) and target:
                assigned.add(target)
        if changed:
            structured["local_program"] = normalized_program
            deterministic_local_updates += 1
            deterministically_updated_scenes.add(str(getattr(scene, "scene_id", "")))
    target_operations: set[tuple[str, str]] = set()
    target_scenes: set[str] = set()
    deterministic_graph_updates = 0
    for item in compile_gate.get("malformed_bindings", []) or []:
        binding = item.get("binding", {}) if isinstance(item, dict) else {}
        operation = str(binding.get("operation", "")) if isinstance(binding, dict) else ""
        scene_id = str(item.get("scene_id", "")) if isinstance(item, dict) else ""
        if operation:
            target_operations.add((scene_id, operation))
    for category in ("schema_mismatches", "dataflow_mismatches"):
        for item in compile_gate.get(category, []) or []:
            if not isinstance(item, dict):
                continue
            operation = str(item.get("operation", ""))
            scene_id = str(item.get("scene_id", ""))
            if operation:
                target_operations.add((scene_id, operation))
            elif scene_id:
                # Ordering diagnostics such as
                # scene_output_is_produced_only_by_later_scene identify the
                # faulty scene but not a single operation. Repair every public
                # binding in that scene while still preserving the graph.
                target_scenes.add(scene_id)
    for item in compile_gate.get("local_program_errors", []) or []:
        if isinstance(item, dict) and item.get("scene_id") is not None:
            target_scenes.add(str(item.get("scene_id")))

    # If a scene declares an output that is uniquely produced by a binding in
    # a later scene, restore that binding to the declaring scene. This repairs
    # an extraction-order defect without inventing operations or consulting
    # expected outputs.
    scenes = list(getattr(repaired, "scenes", []) or [])
    scene_index_by_id = {
        str(getattr(scene, "scene_id", "")): index
        for index, scene in enumerate(scenes)
    }
    for mismatch in compile_gate.get("dataflow_mismatches", []) or []:
        if not isinstance(mismatch, dict) or mismatch.get("reason") != "scene_output_is_produced_only_by_later_scene":
            continue
        target_scene_id = str(mismatch.get("scene_id", ""))
        output = str(mismatch.get("output", ""))
        target_index = scene_index_by_id.get(target_scene_id)
        if target_index is None or not output:
            continue
        matches: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
        for source_index in range(target_index + 1, len(scenes)):
            structured = getattr(
                getattr(scenes[source_index], "action", None), "structured_op", None
            )
            bindings = structured.get("public_operations", []) if isinstance(structured, dict) else []
            for binding in bindings or []:
                if not isinstance(binding, dict):
                    continue
                targets = {
                    str(value).split(".", 1)[0]
                    for value in (binding.get("results", {}) or {}).values()
                    if isinstance(value, str) and value
                }
                if output in targets:
                    matches.append((source_index, binding, bindings))
        if len(matches) != 1:
            continue
        _, binding, source_bindings = matches[0]
        target_structured = getattr(
            getattr(scenes[target_index], "action", None), "structured_op", None
        )
        if not isinstance(target_structured, dict):
            continue
        source_bindings.remove(binding)
        target_structured.setdefault("public_operations", []).append(binding)
        target_scenes.discard(target_scene_id)
        deterministic_graph_updates += 1

    interface_by_operation = {
        str(item.get("operation", "")): item
        for item in public_interfaces if item.get("operation")
    }

    def required_fields(schema: Any) -> set[str]:
        if not isinstance(schema, dict):
            return set()
        direct = {str(item) for item in schema.get("required", []) or []}
        alternatives = [
            {str(value) for value in item.get("required", []) or []}
            for item in schema.get("anyOf", []) + schema.get("oneOf", [])
            if isinstance(item, dict)
        ]
        return direct | (set.intersection(*alternatives) if alternatives else set())
    available = {
        str(item.get("name", "")) if isinstance(item, dict) else str(item)
        for item in (getattr(repaired, "inputs", None) or [])
    }
    available.discard("")
    available.update(f"input.{field}" for field in top_level_inputs)
    available.update(f"workflow_input.{field}" for field in workflow_inputs)
    candidates: list[dict[str, Any]] = []
    locations: dict[tuple[str, str], dict[str, Any]] = {}
    local_only_keys: set[tuple[str, str]] = set()
    scene_structured: dict[str, dict[str, Any]] = {}
    for scene in getattr(repaired, "scenes", []) or []:
        scene_id = str(getattr(scene, "scene_id", ""))
        structured = getattr(getattr(scene, "action", None), "structured_op", None)
        if isinstance(structured, dict):
            scene_structured[scene_id] = structured
        bindings = structured.get("public_operations", []) if isinstance(structured, dict) else []
        for binding in bindings or []:
            if not isinstance(binding, dict):
                continue
            operation = str(binding.get("operation", ""))
            key = (scene_id, operation)
            locations[key] = binding
            if key in target_operations or scene_id in target_scenes:
                candidates.append({
                    "scene_id": getattr(scene, "scene_id", None),
                    "block_id": getattr(scene, "block_id", ""),
                    "raw_requirement_text": getattr(scene, "raw_requirement_text", ""),
                    "logic_flow": getattr(scene, "logic_flow", ""),
                    "operation": operation,
                    "interface": interface_by_operation.get(operation, {}),
                    "binding": copy.deepcopy(binding),
                    "existing_local_program": copy.deepcopy(
                        structured.get("local_program", []) if isinstance(structured, dict) else []
                    ),
                    "available_before_operation": sorted(available),
                    "repeated_workflow_fields_forbidden_as_global_refs": sorted(
                        repeated_workflow_fields or set()
                    ),
                })
            if operation:
                available.add(f"result.{operation}")
            available.update(
                str(target) for target in (binding.get("results", {}) or {}).values()
                if isinstance(target, str) and target
            )
        if (
            scene_id in target_scenes
            and scene_id not in deterministically_updated_scenes
            and not bindings
        ):
            key = (scene_id, "")
            local_only_keys.add(key)
            candidates.append({
                "scene_id": getattr(scene, "scene_id", None),
                "block_id": getattr(scene, "block_id", ""),
                "raw_requirement_text": getattr(scene, "raw_requirement_text", ""),
                "logic_flow": getattr(scene, "logic_flow", ""),
                "operation": "",
                "interface": {},
                "binding": {},
                "existing_local_program": copy.deepcopy(
                    structured.get("local_program", []) if isinstance(structured, dict) else []
                ),
                "available_before_operation": sorted(available),
                "repeated_workflow_fields_forbidden_as_global_refs": sorted(
                    repeated_workflow_fields
                ),
                "local_only_scene": True,
            })
        available.update(
            str(output) for output in (getattr(getattr(scene, "action", None), "outputs", None) or [])
            if output
        )
    if not candidates:
        deterministic_repairs = deterministic_local_updates + deterministic_graph_updates
        return repaired, 0, {
            "requested": 0,
            "accepted": deterministic_repairs,
            "local_programs_updated": deterministic_local_updates,
            "reason": (
                "qualified_nested_workflow_inputs"
                if deterministic_local_updates else "no_target_nodes"
            ),
            "graph_bindings_relocated": deterministic_graph_updates,
            "target_operations": [],
            "target_scenes": sorted(target_scenes),
        }

    response = llm_client.call(
        system_prompt=(
            "Return JSON only. Repair only the supplied public-operation binding and, when required to prepare "
            "its arguments, that scene's local_program. Do not regenerate scenes, rename operations, add public "
            "operations, or change the workflow graph. Preserve scene_id, block_id, dependency_type and operation. "
            "Every binding argument source and condition source must come from available_before_operation or from "
            "a variable produced earlier by the repaired local_program. Wrap constants as {literal: value}; bare "
            "strings in bindings are variable references. Fields listed as forbidden repeated workflow fields must "
            "be reached through a collection record, never workflow_input.<field>. Copy request and response field "
            "names exactly from interface schemas. A binding cannot condition itself on a result it produces. "
            "A local_program is an ordered array using assign/filter/map/sort/emit. Expressions use literal, ref, "
            "object, list, op, or call. Every statement target/item/index must be a plain identifier such as "
            "selected_record or record_payload, never a dotted path, expression, or object. "
            "Useful deterministic calls include find_by(collection, field, expected, "
            "optional_default), get(object, field, optional_default), merge, unique_by, len, coalesce, join, "
            "to_string, and to_markdown_table. Place argument-shaping assignments before the public operation; "
            "the compiler schedules statements from their references. For an object-valued request field, prefer "
            "assigning a named object in local_program and bind the request field to that name. If a selected ID "
            "must be joined back to a collection record, use find_by and then get the required field. Never invent "
            "summary/container variables (for example *_result) that are absent from available_before_operation. "
            "When a public operation consumes a variable prepared by local_program, make that value an explicit "
            "emit field whenever the repaired local program is changed; do not leave a newly introduced public "
            "argument dependent on an unexported intermediate variable. "
            "For a cyclic local/public dependency, remove the cycle inside the supplied scene: a local value "
            "needed by an operation must be computed only from public input or earlier operation results, never "
            "from that operation's own result or from a later operation. Keep result-dependent local statements "
            "after their producer and do not use them to prepare that producer's request. "
            "Allowed "
            "guards: always, first_request, followup, on_success:<operation>, on_failure:<operation>. A condition "
            "may be a leaf {source,operator,value/value_from} or a compound tree using {all:[...]}, {any:[...]}, "
            "or {not:{...}}. Allowed leaf operators: exists, not_exists, eq, neq, contains, not_contains. Preserve "
            "all explicit conjunctions, alternatives, and negations from the requirement. Return "
            "{repairs:[{scene_id,block_id,dependency_type,operation,guard,consumes,produces,arguments,results,"
            "condition,foreach,local_program?}]} . A foreach object has exactly source and argument; never use item. "
            "foreach.argument is the public request field replaced by each current item, not a temporary loop-variable name."
            " For a failed local-only scene, return dependency_type and operation as empty strings, include a valid "
            "local_program, and never add a public operation."
        ),
        user_content=json.dumps({
            "gate_diagnostics": {
                key: compile_gate.get(key, [])
                for key in (
                    "errors", "malformed_bindings", "schema_mismatches",
                    "dataflow_mismatches", "local_program_errors",
                )
            },
            "failed_nodes": candidates,
        }, ensure_ascii=False, sort_keys=True),
        temperature=0.0,
        max_tokens=8192,
        response_format={"type": "json_object"},
    )
    usage = getattr(response, "usage", None) or {}
    tokens = int(usage.get("total_tokens", 0) or 0)
    raw = response.content if hasattr(response, "content") else str(response)
    text = raw.strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    try:
        start = text.find("{")
        parsed, _ = json.JSONDecoder().raw_decode(text[start:]) if start >= 0 else ({}, 0)
    except (json.JSONDecodeError, ValueError):
        parsed = {}
    accepted = deterministic_local_updates + deterministic_graph_updates
    local_programs_updated = deterministic_local_updates
    rejected: list[dict[str, Any]] = []
    guard_pattern = re.compile(
        r"^(always|first_request|followup|on_success:[A-Za-z0-9_]+|on_failure:[A-Za-z0-9_]+)$"
    )
    for proposal in parsed.get("repairs", []) if isinstance(parsed, dict) else []:
        if not isinstance(proposal, dict):
            rejected.append({"reason": "proposal_not_object"})
            continue
        key = (str(proposal.get("scene_id", "")), str(proposal.get("operation", "")))
        if key in local_only_keys:
            proposed_local_program = proposal.get("local_program")
            if proposed_local_program is None:
                rejected.append({"key": key, "reason": "local_program_missing"})
                continue
            try:
                from dsl_v2.local_program import (
                    normalize_local_program_syntax,
                    validate_local_program,
                )
                normalized_local_program = normalize_local_program_syntax(proposed_local_program)
                validate_local_program(normalized_local_program)
            except Exception as exc:
                rejected.append({
                    "key": key,
                    "reason": "invalid_local_program",
                    "detail": f"{type(exc).__name__}: {exc}",
                })
                continue
            structured = scene_structured.get(key[0])
            if structured is None:
                rejected.append({"key": key, "reason": "scene_not_found"})
                continue
            structured["local_program"] = normalized_local_program
            local_programs_updated += 1
            accepted += 1
            continue
        original = locations.get(key)
        interface = interface_by_operation.get(key[1])
        if original is None or not interface:
            rejected.append({"key": key, "reason": "binding_identity_not_targeted"})
            continue
        if str(proposal.get("dependency_type", "")) != str(interface.get("dependency_type", "")):
            rejected.append({"key": key, "reason": "dependency_type_changed"})
            continue
        guard = str(proposal.get("guard", ""))
        if not guard_pattern.fullmatch(guard):
            rejected.append({"key": key, "reason": "invalid_guard"})
            continue
        arguments = proposal.get("arguments")
        results = proposal.get("results")
        condition = proposal.get("condition", {}) or {}
        foreach = proposal.get("foreach", {}) or {}
        if not isinstance(arguments, dict) or not isinstance(results, dict):
            rejected.append({"key": key, "reason": "arguments_or_results_not_object"})
            continue
        if not isinstance(condition, dict) or not isinstance(foreach, dict):
            rejected.append({"key": key, "reason": "condition_or_foreach_not_object"})
            continue
        if condition:
            try:
                validate_condition_tree(condition)
            except ValueError as exc:
                rejected.append({
                    "key": key,
                    "reason": "invalid_condition_tree",
                    "detail": str(exc),
                })
                continue
        proposed_local_program = proposal.get("local_program")
        normalized_local_program = None
        if proposed_local_program is not None:
            try:
                from dsl_v2.local_program import (
                    normalize_local_program_syntax,
                    validate_local_program,
                )
                normalized_local_program = normalize_local_program_syntax(proposed_local_program)
                normalized_local_program, _ = qualify_unique_nested_workflow_refs(
                    normalized_local_program,
                    top_level_inputs,
                    workflow_inputs,
                    repeated_workflow_fields,
                )
                validate_local_program(normalized_local_program)
            except Exception as exc:
                rejected.append({
                    "key": key,
                    "reason": "invalid_local_program",
                    "detail": f"{type(exc).__name__}: {exc}",
                })
                continue
        required_consumes = required_fields(interface.get("request_schema", {}))
        required_produces = required_fields(interface.get("response_schema", {}))
        if foreach:
            if "argument" not in foreach and isinstance(foreach.get("item"), str):
                foreach = {
                    "source": foreach.get("source"),
                    "argument": foreach.get("item"),
                }
            foreach_argument = str(foreach.get("argument", ""))
            if foreach_argument and foreach_argument not in arguments:
                mapped_fields = [
                    str(field)
                    for field, source in arguments.items()
                    if source == foreach_argument
                ]
                if len(mapped_fields) == 1:
                    # Models sometimes use a loop-variable name here even
                    # though the runtime contract expects the request field.
                    foreach = {**foreach, "argument": mapped_fields[0]}
                elif foreach_argument in required_consumes:
                    # The executor replaces this placeholder with the current
                    # item before resolving the request.
                    arguments = {
                        **arguments,
                        foreach_argument: {"literal": None},
                    }
        updated_binding = {
            "dependency_type": str(interface.get("dependency_type", "")),
            "operation": key[1],
            "guard": guard,
            "consumes": sorted(
                required_consumes | {
                    str(item) for item in (proposal.get("consumes", []) or []) if item
                }
            ),
            "produces": sorted(
                required_produces | {
                    str(item) for item in (proposal.get("produces", []) or []) if item
                }
            ),
            "arguments": arguments,
            "results": results,
            "condition": condition,
            "foreach": foreach,
        }
        original.clear()
        original.update(updated_binding)
        if normalized_local_program is not None:
            structured = scene_structured.get(key[0])
            if structured is None:
                continue
            structured["local_program"] = normalized_local_program
            local_programs_updated += 1
        accepted += 1
    return repaired, tokens, {
        "requested": len(candidates),
        "accepted": accepted,
        "local_programs_updated": local_programs_updated,
        "graph_bindings_relocated": deterministic_graph_updates,
        "rejected": rejected,
        "target_operations": sorted(f"{scene}:{operation}" for scene, operation in target_operations),
        "target_scenes": sorted(target_scenes),
    }


def run_task(
    task_id: str,
    repeat_count: int,
    output_root: Path | None = None,
    tasks_root: Path = TASKS,
    oracle_root: Path = ORACLE,
    fact_spec_path: Path | None = None,
    report_filename: str | None = None,
    case_ids: set[str] | None = None,
) -> dict[str, Any]:
    from llm_client import create_llm_client
    from pipeline_runner import (
        run_ncnlp_pipeline_compile_once,
        run_ncnlp_pipeline_recompile_fact_spec,
        run_ncnlp_pipeline_run_only,
    )

    replay_name = {1: "one", 3: "three"}.get(repeat_count, str(repeat_count))
    default_report_name = f"NCNLP_{task_id}_{replay_name}_replay_v3.json"

    tasks_root = tasks_root.resolve()
    oracle_root = oracle_root.resolve()
    task_path = tasks_root / "system_inputs" / task_id / "task.json"
    if not task_path.is_file():
        raise ValueError(f"task is not in the development/qualification bundle: {task_id}")
    if fact_spec_path is not None and not fact_spec_path.is_file():
        raise FileNotFoundError(f"saved FactSpec not found: {fact_spec_path}")
    fact_spec_source_sha256 = (
        hashlib.sha256(fact_spec_path.read_bytes()).hexdigest().upper()
        if fact_spec_path is not None else ""
    )
    manifest_path = tasks_root / "manifest.json"
    bundle_id = read_json(manifest_path).get("bundle_id", tasks_root.parent.name) if manifest_path.is_file() else tasks_root.parent.name
    oracle = read_json(oracle_root / f"{task_id}.oracle.json")
    first_case = oracle["cases"][0]["case_id"]
    compile_view = load_system_view(task_id, first_case, root=tasks_root)
    compile_task = adapt_to_legacy_ncnlp(compile_view)
    top_level_inputs, workflow_inputs = public_input_field_catalog(tasks_root, task_id)
    input_mode_signatures = public_input_mode_signatures(tasks_root, task_id)
    input_mode_profiles = public_input_mode_profiles(tasks_root, task_id)
    input_schema_catalog = public_input_schema_catalog(tasks_root, task_id)
    repeated_workflow_fields = public_repeated_workflow_input_fields(tasks_root, task_id)
    compile_task["public_top_level_input_fields"] = sorted(top_level_inputs)
    compile_task["public_workflow_input_fields"] = sorted(workflow_inputs)
    compile_task["public_input_mode_signatures"] = input_mode_signatures
    compile_task["public_input_schema"] = input_schema_catalog
    compile_task["public_repeated_workflow_input_fields"] = sorted(
        repeated_workflow_fields
    )
    mode_lines = "\n".join(
        f"  - Mode {index}: top-level=[{', '.join(profile['top_level']) or '(none)'}]; "
        f"nested=[{', '.join(profile['nested']) or '(none)'}]"
        for index, profile in enumerate(input_mode_profiles, 1)
    )
    compile_task["method_prompt"] += (
        "\n\nRuntime public-input binding rules:\n"
        "- input.<field> may reference only a real top-level input field.\n"
        "- workflow_input.<field> performs a unique recursive lookup inside public case inputs and is used "
        "for values nested in fixture containers.\n"
        f"- Available top-level fields: {', '.join(sorted(top_level_inputs)) or '(none)'}.\n"
        f"- Available recursive workflow fields: {', '.join(sorted(workflow_inputs)) or '(none)'}.\n"
        "- Public top-level input schema (types and control-state enum labels only): "
        f"{json.dumps(input_schema_catalog, ensure_ascii=False, sort_keys=True)}. "
        "A scalar field has no child properties: reference input.approval, never input.approval.status. "
        "When an enum is supplied, copy its label exactly into conditions rather than translating it.\n"
        "- Fields observed inside repeated collections cannot be read as global workflow_input values; "
        "bind them through foreach/current-item data flow: "
        f"{', '.join(sorted(repeated_workflow_fields)) or '(none)'}.\n"
        "- Observed public input mode profiles (field names and structural location only; no values, labels, "
        "or expected outputs):\n"
        f"{mode_lines}\n"
        "- Distinct input modes are alternative entry paths. Bind mode-specific operations to an identifying "
        "field condition (normally exists/not_exists); never serialize mutually exclusive modes into one "
        "mandatory success chain.\n"
        "- Represent compound business predicates explicitly with all/any/not condition trees; never drop a "
        "conjunct or approximate an OR branch with one arbitrary leaf.\n"
        "- When the requirement permits alternative input sources, preserve every source branch; do not make "
        "an operation from one source an unconditional predecessor of the other source.\n"
        "- A filter, validation, ranking, or selection step changes the data-flow boundary. Every downstream "
        "operation governed by that step must consume or iterate over the filtered/validated/selected output, "
        "not the original unfiltered input. Prefer foreach over the derived collection so an empty collection "
        "naturally suppresses the public operation and its success-dependent descendants.\n"
        "- Public-operation arguments may reference only public input, earlier local outputs, or earlier public "
        "results. Never bind an argument to a value first produced by a later scene or by the same operation.\n"
        "- For approval/rejection workflows, use the public approval or status input as the branch condition for "
        "publish, save, notify, and other approval-gated side effects. Do not infer approval from a notification "
        "API response such as status=posted; that response reports delivery, not the user's approval decision."
    )
    compile_task["input_data"] = {}
    compile_task["source"] = bundle_id
    client = create_llm_client(enable_cache=False)
    compile_attempts = []
    compiled: dict[str, Any] = {}
    compile_gate: dict[str, Any] = {}
    current_fact_spec: Any = None
    compile_reused = fact_spec_path is not None
    for attempt in range(1, 4):
        repair_audit: dict[str, Any] = {}
        if attempt == 1 and fact_spec_path is not None:
            repair_mode = "reused_fact_spec"
        elif current_fact_spec is None:
            repair_mode = "full_extract" if attempt == 1 else "full_extract_retry"
        else:
            repair_mode = "focused_node_repair"
        try:
            if attempt == 1 and fact_spec_path is not None:
                current_fact_spec = load_fact_spec(fact_spec_path.resolve())
                compiled = run_ncnlp_pipeline_recompile_fact_spec(
                    copy.deepcopy(compile_task),
                    current_fact_spec,
                    llm_client=client,
                    backend="dsl_compiler",
                    repair_tokens=0,
                )
            elif current_fact_spec is None:
                compiled = run_ncnlp_pipeline_compile_once(
                    copy.deepcopy(compile_task), llm_client=client, backend="dsl_compiler"
                )
            else:
                current_fact_spec, repair_tokens, repair_audit = repair_failed_fact_spec_nodes(
                    current_fact_spec,
                    compile_gate,
                    compile_view["task"].get("public_interfaces", []),
                    top_level_inputs,
                    workflow_inputs,
                    client,
                    repeated_workflow_fields=repeated_workflow_fields,
                )
                if not repair_audit.get("accepted"):
                    compile_attempts.append({
                        "attempt": attempt,
                        "passed": False,
                        "compile_tokens": repair_tokens,
                        "compile_latency_ms": 0,
                        "gate_errors": compile_gate.get("errors", []),
                        "repair_mode": repair_mode,
                        "repair_audit": repair_audit,
                        "recompile_executed": False,
                    })
                    break
                compiled = run_ncnlp_pipeline_recompile_fact_spec(
                    copy.deepcopy(compile_task),
                    current_fact_spec,
                    llm_client=client,
                    backend="dsl_compiler",
                    repair_tokens=repair_tokens,
                )
        except Exception as exc:
            compile_attempts.append({
                "attempt": attempt,
                "passed": False,
                "compile_tokens": 0,
                "compile_latency_ms": 0,
                "gate_errors": ["compile_structure_exception"],
                "repair_mode": repair_mode,
                "repair_audit": repair_audit,
                "compile_exception": f"{type(exc).__name__}: {exc}",
                "recompile_executed": False,
            })
            if repair_mode.startswith("full_extract"):
                current_fact_spec = None
            if attempt == 3:
                raise
            continue
        current_fact_spec = compiled.get("fact_spec")
        compile_gate = audit_public_action_contract(
            compiled.get("dsl_code", ""),
            compile_view["task"].get("public_interfaces", []),
            top_level_input_fields=top_level_inputs,
            workflow_input_fields=workflow_inputs,
            repeated_workflow_input_fields=repeated_workflow_fields,
        )
        compile_attempts.append({
            "attempt": attempt,
            "passed": compile_gate["passed"],
            "compile_tokens": compiled.get("compile_tokens", 0),
            "compile_latency_ms": compiled.get("compile_latency_ms", 0),
            "gate_errors": compile_gate["errors"],
            "repair_mode": repair_mode,
            "repair_audit": repair_audit,
            "recompile_diagnostics": compiled.get("recompile_diagnostics", {}),
        })
        if compile_gate["passed"]:
            break
    total_compile_tokens = sum(item["compile_tokens"] for item in compile_attempts)
    total_compile_latency_ms = sum(item["compile_latency_ms"] for item in compile_attempts)

    artifact_root = output_root / "artifacts" if output_root else ARTIFACT_ROOT
    trace_root = output_root / "traces" if output_root else TRACE_ROOT
    report_root = output_root / "reports" if output_root else REPORT_ROOT
    artifact = artifact_root / task_id
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "generated.dsl").write_text(compiled.get("dsl_code", ""), encoding="utf-8")
    (artifact / "generated.py").write_text(compiled.get("generated_python", ""), encoding="utf-8")
    (artifact / "fact_spec.json").write_text(
        json.dumps(serialize(current_fact_spec), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact / "compile_metadata.json").write_text(json.dumps({
        "task_id": task_id,
        "backend": compiled.get("backend"),
        "compile_model": getattr(client, "model", ""),
        "compile_tokens": total_compile_tokens,
        "compile_latency_ms": total_compile_latency_ms,
        "compile_attempts": compile_attempts,
        "dsl_sha256": hashlib.sha256(compiled.get("dsl_code", "").encode("utf-8")).hexdigest().upper(),
        "python_sha256": hashlib.sha256(compiled.get("generated_python", "").encode("utf-8")).hexdigest().upper(),
        "strict_compile_gate": compile_gate,
        "compile_reused": compile_reused,
        "source_fact_spec": str(fact_spec_path.resolve()) if fact_spec_path else "",
        "source_fact_spec_sha256": fact_spec_source_sha256,
        "formal_test_executed": False,
        "task_bundle_id": bundle_id,
        "task_bundle_root": str(tasks_root),
        "public_input_mode_signatures": input_mode_signatures,
        "public_input_mode_profiles": input_mode_profiles,
        "public_input_schema": input_schema_catalog,
        "public_repeated_workflow_input_fields": sorted(repeated_workflow_fields),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not compile_gate["passed"]:
        report = {
            "schema_version": "n8n-s1-ncnlp-task-replay-v3",
            "status": "COMPILE_ACTION_CONTRACT_REJECTED",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "task_id": task_id,
            "partition": "development_qualification",
            "formal_test_executed": False,
            "task_bundle_id": bundle_id,
            "task_bundle_root": str(tasks_root),
            "compile_executed": True,
            "compile_reused": compile_reused,
            "source_fact_spec_sha256": fact_spec_source_sha256,
            "compile_model": getattr(client, "model", ""),
            "repeat_count": repeat_count,
            "compile_tokens": total_compile_tokens,
            "compile_latency_ms": total_compile_latency_ms,
            "compile_attempts": compile_attempts,
            "strict_compile_gate": compile_gate,
            "records": [],
        }
        report_root.mkdir(parents=True, exist_ok=True)
        report_name = report_filename or default_report_name
        (report_root / report_name).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report

    selected_cases = [
        case for case in oracle["cases"]
        if case_ids is None or case["case_id"] in case_ids
    ]
    if not selected_cases:
        raise ValueError(f"no oracle cases selected for {task_id}: {sorted(case_ids or [])}")

    records = []
    for case in selected_cases:
        case_id = case["case_id"]
        task = adapt_to_legacy_ncnlp(load_system_view(task_id, case_id, root=tasks_root))
        task["source"] = bundle_id
        runs = []
        for run_index in range(1, repeat_count + 1):
            method = run_ncnlp_pipeline_run_only(compiled, task, llm_client=client)
            executor = MockPlanExecutor(
                task_id,
                case_id,
                tasks_root=tasks_root,
                runtime_contracts=None,
                strict_case_actions=True,
            )
            mock_result = executor.execute(
                method["actions"],
                local_action_executor=method.get("local_action_executor"),
            )
            mock_result["raw_trace"].setdefault("final_state", {}).update({
                "method_execution_status": method["final_state"].get("execution_status"),
                "method_workflow_branch": method["final_state"].get("workflow_branch"),
                "route_diagnostics": method["final_state"].get("route_diagnostics", {}),
            })
            evaluation = evaluate_case(task_id, case, mock_result["raw_trace"])
            envelope = {
                "task_id": task_id,
                "case_id": case_id,
                "run_index": run_index,
                "input_sha256": task["input_sha256"].upper(),
                "raw_trace": mock_result["raw_trace"],
                "method_ok": method["ok"],
                "method_error": method["error"],
                "executor_ok": mock_result["ok"],
                "executor_errors": mock_result["errors"],
                "runtime_tokens": method["runtime_tokens"],
                "sensor_call_count": method["sensor_call_count"],
                "action_plan": method["actions"],
                "action_plan_sha256": canonical_hash(method["actions"]),
            }
            trace_path = trace_root / task_id / case_id / f"run-{run_index}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            runs.append({
                "run_index": run_index,
                "method_ok": method["ok"],
                "executor_ok": mock_result["ok"],
                "oracle_passed": evaluation["passed"],
                "trace_sha256": evaluation["trace_sha256"],
                "runtime_tokens": method["runtime_tokens"],
                "failed_assertion_ids": [
                    item["assertion_id"] for item in evaluation["assertions"] if not item["passed"]
                ],
                "executor_errors": mock_result["errors"],
            })
        records.append({
            "case_id": case_id,
            "accepted": all(item["method_ok"] and item["executor_ok"] and item["oracle_passed"] for item in runs),
            "deterministic": replay_is_deterministic(runs),
            "runs": runs,
        })
    accepted = all(item["accepted"] and item["deterministic"] for item in records)
    report = {
        "schema_version": "n8n-s1-ncnlp-task-replay-v3",
        "status": "TASK_REPLAY_ACCEPTED" if accepted else "TASK_REPLAY_REJECTED",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task_id": task_id,
        "partition": "development_qualification",
        "formal_test_executed": False,
        "task_bundle_id": bundle_id,
        "task_bundle_root": str(tasks_root),
        "compile_executed": True,
        "compile_reused": compile_reused,
        "source_fact_spec_sha256": fact_spec_source_sha256,
        "compile_model": getattr(client, "model", ""),
        "repeat_count": repeat_count,
        "selected_case_ids": [case["case_id"] for case in selected_cases],
        "compile_tokens": total_compile_tokens,
        "compile_latency_ms": total_compile_latency_ms,
        "compile_attempts": compile_attempts,
        "strict_compile_gate": compile_gate,
        "records": records,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    report_name = report_filename or default_report_name
    (report_root / report_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--runs", type=int, default=3, choices=[1, 3, 5, 10])
    parser.add_argument("--out", type=Path, help="New versioned output root; never overwrites prior evidence")
    parser.add_argument("--tasks-root", type=Path, default=TASKS)
    parser.add_argument("--oracle-root", type=Path, default=ORACLE)
    parser.add_argument(
        "--fact-spec", type=Path,
        help="Reuse a saved development FactSpec, then run only strict focused repairs",
    )
    parser.add_argument(
        "--case", dest="case_ids", action="append",
        help="Replay only this case; may be supplied multiple times",
    )
    args = parser.parse_args()
    if args.out and args.out.exists():
        raise FileExistsError("--out must name a new directory")
    report = run_task(
        args.task,
        args.runs,
        output_root=args.out,
        tasks_root=args.tasks_root,
        oracle_root=args.oracle_root,
        fact_spec_path=args.fact_spec,
        case_ids=set(args.case_ids) if args.case_ids else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "TASK_REPLAY_ACCEPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
