#!/usr/bin/env python3
"""Build method-neutral D5 task bundles from the accepted H1 artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


BENCHMARK = Path(__file__).resolve().parent
H1 = BENCHMARK / "n8n_conversion_pilot" / "h1_execution"
REQUIREMENTS = H1 / "submissions" / "requirement_author_a"
CONSENSUS = H1 / "frozen" / "consensus_gold_v1"
ORACLE = H1 / "frozen" / "gold_oracle_v1" / "oracle"
SELECTION = BENCHMARK / "n8n_conversion_pilot" / "selection_manifest.json"
TARGET = H1 / "d5_unified_tasks_v2"
MANIFEST = TARGET / "manifest.json"
TASK_IDS = [
    "N8C-001", "N8C-002", "N8C-003", "N8C-004", "N8C-006", "N8C-007",
    "N8C-008", "N8C-009", "N8C-010", "N8C-011", "N8C-012",
]
FORBIDDEN_SYSTEM_KEYS = {
    "nodes", "cfg_edges", "dfg_edges", "entry_nodes", "terminal_nodes", "assertions",
    "expected", "response_contract", "native_evidence", "requirement_evidence", "final_gold",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest().lower()


def requirement_artifact(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "pilot_id": record["pilot_id"],
        "requirement_text": record["requirement_text"],
        "source_spans": record["source_spans"],
        "transformation_operations": record["transformation_operations"],
        "business_contract": record["business_contract"],
    }


def value_schema(value: Any) -> dict[str, Any]:
    """Expose only a Mock value's shape; never expose its literal value."""
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if value is None:
        return {"type": "null"}
    if isinstance(value, list):
        variants = {json.dumps(value_schema(item), sort_keys=True) for item in value}
        if not variants:
            items: dict[str, Any] = {}
        elif len(variants) == 1:
            items = json.loads(next(iter(variants)))
        else:
            items = {"anyOf": [json.loads(item) for item in sorted(variants)]}
        return {"type": "array", "items": items}
    if isinstance(value, dict):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {key: value_schema(item) for key, item in sorted(value.items())},
            "required": sorted(value),
        }
    return {}


def merge_schemas(values: list[dict[str, Any]]) -> dict[str, Any]:
    variants = {json.dumps(value, sort_keys=True) for value in values}
    if not variants:
        return {}
    if len(variants) == 1:
        return json.loads(next(iter(variants)))
    return {"anyOf": [json.loads(value) for value in sorted(variants)]}


def public_interface_catalog(mocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for mock in mocks:
        grouped.setdefault((mock["dependency_type"], mock["operation"]), []).append(mock)
    return [
        {
            "dependency_type": dependency,
            "operation": operation,
            "request_schema": merge_schemas([value_schema(item.get("request_contract", {})) for item in items]),
            "response_schema": merge_schemas([value_schema(item.get("response_contract", {})) for item in items]),
        }
        for (dependency, operation), items in sorted(grouped.items())
    ]


def schemas() -> dict[str, dict[str, Any]]:
    system_task = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "task_id", "domain", "requirement_sha256", "natural_language_requirement", "public_interfaces"],
        "properties": {
            "schema_version": {"const": "n8n-h1-system-task-v2"},
            "task_id": {"type": "string", "pattern": "^N8C-[0-9]{3}$"},
            "domain": {"type": "string", "minLength": 1},
            "requirement_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "natural_language_requirement": {"type": "string", "minLength": 1},
            "public_interfaces": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["dependency_type", "operation", "request_schema", "response_schema"],
                    "properties": {
                        "dependency_type": {"type": "string"},
                        "operation": {"type": "string"},
                        "request_schema": {"type": "object"},
                        "response_schema": {"type": "object"},
                    },
                },
            },
        },
    }
    instance = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "task_id", "case_id", "kind", "input_data", "input_sha256"],
        "properties": {
            "schema_version": {"const": "n8n-h1-system-instance-v2"},
            "task_id": {"type": "string"}, "case_id": {"type": "string"}, "kind": {"type": "string"},
            "input_data": {"type": "object"},
            "input_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        },
    }
    executor = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "task_id", "side_effects_disabled", "mocks", "case_bindings"],
        "properties": {
            "schema_version": {"const": "n8n-h1-executor-contract-v2"},
            "task_id": {"type": "string"}, "side_effects_disabled": {"const": True},
            "mocks": {"type": "array"}, "case_bindings": {"type": "array"},
        },
    }
    return {"system_task": system_task, "instance": instance, "executor": executor}


def nested_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(key)
            found.update(nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(nested_keys(item))
    return found


def validate_existing() -> list[str]:
    manifest = read_json(MANIFEST)
    errors: list[str] = []
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != hashlib.sha256(canonical_json(unsigned)).hexdigest().upper():
        errors.append("manifest hash mismatch")
    for item in manifest.get("files", []):
        path = TARGET / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            errors.append(f"file mismatch: {item['path']}")
    return errors


def build() -> dict[str, Any]:
    if TARGET.exists():
        if not MANIFEST.is_file():
            raise RuntimeError(f"target exists without manifest: {TARGET}")
        errors = validate_existing()
        return {"status": "PASS" if not errors else "FAIL", "errors": errors}

    schema_values = schemas()
    for name, value in schema_values.items():
        write_json(TARGET / "schemas" / f"{name}.schema.json", value)
    selection = {item["pilot_id"]: item for item in read_json(SELECTION)["items"]}
    task_summary = []

    for task_id in TASK_IDS:
        requirement = read_json(REQUIREMENTS / f"{task_id}.json")
        gold = read_json(CONSENSUS / "gold" / f"{task_id}.gold.json")
        oracle = read_json(ORACLE / f"{task_id}.response.json")
        requirement_hash = sha256_value(requirement_artifact(requirement))
        if requirement_hash != gold["requirement_sha256"] or requirement_hash != oracle["requirement_sha256"]:
            raise RuntimeError(f"requirement/Gold/Oracle hash mismatch: {task_id}")

        public_interfaces = public_interface_catalog(oracle["mocks"])
        system_task = {
            "schema_version": "n8n-h1-system-task-v2",
            "task_id": task_id,
            "domain": selection[task_id]["domain"],
            "requirement_sha256": requirement_hash,
            "natural_language_requirement": requirement["requirement_text"],
            "public_interfaces": public_interfaces,
        }
        schema_errors = list(Draft202012Validator(schema_values["system_task"]).iter_errors(system_task))
        leaked_keys = nested_keys(system_task) & FORBIDDEN_SYSTEM_KEYS
        if schema_errors or leaked_keys:
            raise RuntimeError(f"invalid/leaking system task {task_id}: {schema_errors}; {sorted(leaked_keys)}")
        write_json(TARGET / "system_inputs" / task_id / "task.json", system_task)

        fixtures = {fixture["fixture_id"]: fixture for fixture in oracle["fixtures"]}
        for case in oracle["cases"]:
            selected_fixtures = {fixture_id: fixtures[fixture_id]["input_data"] for fixture_id in case["fixture_ids"]}
            input_data = next(iter(selected_fixtures.values())) if len(selected_fixtures) == 1 else {"fixtures": selected_fixtures}
            instance = {
                "schema_version": "n8n-h1-system-instance-v2",
                "task_id": task_id,
                "case_id": case["case_id"],
                "kind": case["kind"],
                "input_data": input_data,
                "input_sha256": sha256_value(input_data),
            }
            errors = list(Draft202012Validator(schema_values["instance"]).iter_errors(instance))
            if errors or nested_keys(instance) & FORBIDDEN_SYSTEM_KEYS:
                raise RuntimeError(f"invalid/leaking instance {task_id}/{case['case_id']}: {errors}")
            write_json(TARGET / "system_inputs" / task_id / "instances" / f"{case['case_id']}.json", instance)

        executor = {
            "schema_version": "n8n-h1-executor-contract-v2",
            "task_id": task_id,
            "side_effects_disabled": True,
            "mocks": oracle["mocks"],
            "case_bindings": [
                {"case_id": case["case_id"], "fixture_ids": case["fixture_ids"], "mock_ids": case["mock_ids"]}
                for case in oracle["cases"]
            ],
        }
        errors = list(Draft202012Validator(schema_values["executor"]).iter_errors(executor))
        if errors:
            raise RuntimeError(f"invalid executor contract {task_id}: {errors}")
        write_json(TARGET / "executor" / f"{task_id}.json", executor)
        write_json(TARGET / "evaluation" / "gold" / f"{task_id}.json", gold)
        write_json(TARGET / "evaluation" / "oracle" / f"{task_id}.json", oracle)
        task_summary.append({
            "task_id": task_id,
            "domain": selection[task_id]["domain"],
            "case_count": len(oracle["cases"]),
            "fixture_count": len(oracle["fixtures"]),
            "mock_count": len(oracle["mocks"]),
            "public_interface_count": len(public_interfaces),
        })

    (TARGET / "README.md").write_text(
        "# H1 D5 unified task bundle v2\n\n"
        "`system_inputs/` is the only tree visible to NCNLP and RQ2 baselines. "
        "`executor/` is held by the side-effect-free mock runtime. `evaluation/` is held by the evaluator "
        "and must never be mounted into a tested system. The same system task, instance and public interface "
        "catalog are supplied to every method. Public request/response schemas expose field shapes only; "
        "literal Mock values remain executor-private.\n",
        encoding="utf-8",
    )
    tracked = sorted(path for path in TARGET.rglob("*") if path.is_file() and path != MANIFEST)
    consensus_manifest = read_json(CONSENSUS / "freeze_manifest.json")
    source_manifest = read_json(H1 / "frozen" / "gold_oracle_v1" / "freeze_manifest.json")
    unsigned = {
        "bundle_version": "n8n-h1-d5-unified-tasks-v2",
        "status": "SEALED",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task_count": len(TASK_IDS),
        "case_count": sum(item["case_count"] for item in task_summary),
        "visibility_policy": {
            "tested_system": ["system_inputs"],
            "mock_executor": ["system_inputs", "executor"],
            "evaluator": ["system_inputs", "executor", "evaluation"],
        },
        "upstream_consensus_manifest_sha256": consensus_manifest["manifest_sha256"],
        "upstream_gold_oracle_manifest_sha256": source_manifest["manifest_sha256"],
        "tasks": task_summary,
        "files": [
            {"path": path.relative_to(TARGET).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in tracked
        ],
        "invalidation_rule": "Any post-seal file change invalidates this manifest.",
    }
    manifest = {**unsigned, "manifest_sha256": hashlib.sha256(canonical_json(unsigned)).hexdigest().upper()}
    write_json(MANIFEST, manifest)
    return {
        "status": "SEALED",
        "task_count": manifest["task_count"],
        "case_count": manifest["case_count"],
        "file_count": len(manifest["files"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "errors": validate_existing(),
    }


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
