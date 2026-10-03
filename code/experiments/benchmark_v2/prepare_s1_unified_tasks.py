#!/usr/bin/env python3
"""Build leakage-controlled S1 system inputs and private mock contracts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from prepare_n8n_h1_unified_tasks import merge_schemas, public_interface_catalog


BENCHMARK = Path(__file__).resolve().parent
S1 = BENCHMARK / "n8n_native_s1" / "source_review_round_v2"
DEFAULT_REQUIREMENTS = S1 / "d10_requirement_freeze_v1" / "requirements"
DEFAULT_ORACLE = S1 / "d10_gold_oracle_round_v1" / "frozen" / "oracle_spec_v1"
DEFAULT_SPLIT = S1 / "d14_split_d15_double_gold_v2" / "reports" / "dataset_split_v1.json"
DEFAULT_OUT = S1 / "runtime" / "unified_tasks_v1" / "development_qualification"
FORBIDDEN_KEYS = {
    "assertions", "expected", "response_contract", "mocks", "cases", "mock_ids",
    "native_graph_evidence", "cfg_edges", "dfg_edges", "nodes", "final_gold",
}
IDENTIFIER_PROPERTY = re.compile(r"^(?:[A-Z][A-Z0-9_]*-\d{2,}|[A-Z]\d{2,}|\d{3,})$")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def nested_keys(value: Any) -> set[str]:
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            result.add(key)
            result.update(nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            result.update(nested_keys(item))
    return result


def materialize_fixture_shapes(mocks: list[dict[str, Any]], fixtures: dict[str, Any]) -> list[dict[str, Any]]:
    """Replace exact fixture references before deriving public JSON shapes.

    Only the resulting schema is published; fixture values remain private.
    """
    materialized = copy.deepcopy(mocks)

    def visit(value: Any, field: str = "") -> Any:
        if isinstance(value, str):
            match = re.fullmatch(r"see fixture\s+([A-Za-z0-9_-]+)", value.strip(), re.IGNORECASE)
            if not match or match.group(1) not in fixtures:
                return value
            fixture = fixtures[match.group(1)]
            if isinstance(fixture, dict) and field in fixture:
                return copy.deepcopy(fixture[field])
            return copy.deepcopy(fixture)
        if isinstance(value, dict):
            return {key: visit(item, str(key)) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, field) for item in value]
        return value

    for mock in materialized:
        mock["request_contract"] = visit(mock.get("request_contract", {}))
        mock["response_contract"] = visit(mock.get("response_contract", {}))
    return materialized


def generalize_identifier_maps(schema: dict[str, Any]) -> dict[str, Any]:
    """Hide instance-specific object keys while retaining their value shape."""
    generalized = copy.deepcopy(schema)

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        current = {key: visit(item) for key, item in value.items()}
        properties = current.get("properties")
        if (
            current.get("type") == "object"
            and isinstance(properties, dict)
            and len(properties) >= 2
            and all(IDENTIFIER_PROPERTY.fullmatch(str(key)) for key in properties)
        ):
            current.pop("properties", None)
            current.pop("required", None)
            current["additionalProperties"] = merge_schemas(list(properties.values()))
        return current

    return visit(generalized)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["development_qualification"], default="development_qualification")
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--bundle-id",
        default="n8n-s1-development-qualification-unified-tasks-v1",
    )
    parser.add_argument("--status", default="SEALED")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    split_path = args.split.resolve()
    split = read_json(split_path)
    records = [
        item for item in split["records"]
        if item["dataset_partition"] == args.partition
    ]
    if len(records) != 29:
        raise ValueError(f"expected 29 development/qualification tasks, got {len(records)}")

    summary = []
    for split_record in sorted(records, key=lambda item: item["task_id"]):
        task_id = split_record["task_id"]
        requirement = read_json(args.requirements.resolve() / f"{task_id}.json")
        oracle = read_json(args.oracle.resolve() / "oracle" / f"{task_id}.oracle.json")
        requirement_hash = requirement["requirement_sha256"]
        if requirement_hash != oracle["requirement_sha256"] or requirement_hash != split_record["requirement_sha256"]:
            raise ValueError(f"requirement/Oracle/split hash mismatch: {task_id}")
        fixture_values = {item["fixture_id"]: item["input_data"] for item in oracle["fixtures"]}
        public_interfaces = public_interface_catalog(
            materialize_fixture_shapes(oracle["mocks"], fixture_values)
        )
        for interface in public_interfaces:
            interface["request_schema"] = generalize_identifier_maps(interface["request_schema"])
            interface["response_schema"] = generalize_identifier_maps(interface["response_schema"])
        system_task = {
            "schema_version": "n8n-s1-system-task-v1",
            "task_id": task_id,
            "dataset_partition": args.partition,
            "domain": split_record["domain"],
            "requirement_sha256": requirement_hash,
            "natural_language_requirement": requirement["requirement"]["requirement_text"],
            "public_interfaces": public_interfaces,
        }
        leaked = nested_keys(system_task) & FORBIDDEN_KEYS
        if leaked:
            raise ValueError(f"held-out evaluation data leaked into {task_id}: {sorted(leaked)}")
        write_json(out / "system_inputs" / task_id / "task.json", system_task)

        fixtures = fixture_values
        for case in oracle["cases"]:
            selected = {fixture_id: fixtures[fixture_id] for fixture_id in case["fixture_ids"]}
            input_data = next(iter(selected.values())) if len(selected) == 1 else {"fixtures": selected}
            instance = {
                "schema_version": "n8n-s1-system-instance-v1",
                "task_id": task_id,
                "case_id": case["case_id"],
                "kind": case["kind"],
                "input_data": input_data,
                "input_sha256": canonical_hash(input_data),
            }
            if nested_keys(instance) & FORBIDDEN_KEYS:
                raise ValueError(f"evaluation data leaked into instance {task_id}/{case['case_id']}")
            write_json(out / "system_inputs" / task_id / "instances" / f"{case['case_id']}.json", instance)

        executor = {
            "schema_version": "n8n-s1-private-executor-contract-v1",
            "task_id": task_id,
            "side_effects_disabled": True,
            "mocks": oracle["mocks"],
            "case_bindings": [
                {
                    "case_id": case["case_id"],
                    "fixture_ids": case["fixture_ids"],
                    "mock_ids": case["mock_ids"],
                }
                for case in oracle["cases"]
            ],
        }
        write_json(out / "private_executor" / f"{task_id}.json", executor)
        summary.append({
            "task_id": task_id,
            "case_count": len(oracle["cases"]),
            "public_interface_count": len(public_interfaces),
            "mock_count": len(oracle["mocks"]),
        })

    (out / "README.md").write_text(
        "# S1开发/资格任务统一输入\n\n"
        "被测系统只能读取`system_inputs/`。`private_executor/`包含模拟返回值，"
        "只供评价执行器读取，不能交给NCNLP或比较系统。当前目录不包含80条正式测试任务。\n",
        encoding="utf-8",
    )
    tracked = sorted(path for path in out.rglob("*") if path.is_file())
    manifest = {
        "bundle_id": args.bundle_id,
        "status": args.status,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "partition": args.partition,
        "formal_test_included": False,
        "task_count": len(summary),
        "case_count": sum(item["case_count"] for item in summary),
        "visibility_policy": {
            "tested_system": ["system_inputs"],
            "mock_executor": ["system_inputs", "private_executor"],
        },
        "split_sha256": file_hash(split_path),
        "oracle_freeze_manifest_sha256": file_hash(args.oracle.resolve() / "freeze_manifest.json"),
        "tasks": summary,
        "files": [
            {
                "path": path.relative_to(out).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for path in tracked
        ],
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "output": str(out),
        "task_count": manifest["task_count"],
        "case_count": manifest["case_count"],
        "formal_test_included": manifest["formal_test_included"],
        "file_count": len(manifest["files"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
