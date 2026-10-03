#!/usr/bin/env python3
"""Validate returned D15 DFG endpoint adjudication responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


BENCHMARK = Path(__file__).resolve().parent
DEFAULT_PACKAGE = (
    BENCHMARK / "n8n_native_s1" / "source_review_round_v2"
    / "d14_split_d15_double_gold_v2" / "dfg_endpoint_adjudication_v1"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate(package: Path, allow_pending: bool = False) -> dict[str, Any]:
    manifest = read_json(package / "package_manifest.json")
    schema = read_json(package / "dfg_endpoint_adjudication.response.schema.json")
    errors = []
    warnings = []

    for item in manifest["immutable_files"]:
        path = package / item["path"]
        if not path.exists():
            errors.append(f"missing immutable file: {item['path']}")
        elif sha256(path) != item["sha256"]:
            errors.append(f"immutable file hash changed: {item['path']}")

    records = []
    adjudicator_ids = set()
    for batch in manifest["batches"]:
        for task_id in batch["task_ids"]:
            packet_path = package / "packets" / batch["batch_id"] / f"{task_id}.dfg.adjudication.packet.json"
            response_path = package / "responses" / batch["batch_id"] / f"{task_id}.dfg.adjudication.response.json"
            item_errors = []
            if not response_path.exists():
                item_errors.append("response file is missing")
                records.append({"task_id": task_id, "valid": False, "errors": item_errors})
                continue
            packet = read_json(packet_path)
            response = read_json(response_path)
            pending_template = allow_pending and response.get("status") == "pending"
            schema_errors = list(Draft202012Validator(schema).iter_errors(response))
            item_errors.extend(
                error.message
                for error in schema_errors
                if not (pending_template and list(error.path) == ["adjudicator_id"])
            )
            if response.get("task_id") != task_id:
                item_errors.append("task_id mismatch")
            if response.get("status") != "submitted" and not pending_template:
                item_errors.append("status is not submitted")
            adjudicator_id = response.get("adjudicator_id", "").strip()
            if not adjudicator_id:
                if not pending_template:
                    item_errors.append("adjudicator_id is empty")
            else:
                adjudicator_ids.add(adjudicator_id)

            expected = {
                item["dispute_id"]: (item["source_group"], item["target_group"])
                for item in packet["disputed_endpoint_edges"]
            }
            actual_ids = [item.get("dispute_id") for item in response.get("decisions", [])]
            if len(actual_ids) != len(set(actual_ids)):
                item_errors.append("duplicate dispute_id")
            if set(actual_ids) != set(expected):
                item_errors.append("decision set does not match the frozen disputes")
            for decision in response.get("decisions", []):
                dispute_id = decision.get("dispute_id")
                if dispute_id not in expected:
                    continue
                if (decision.get("source_group"), decision.get("target_group")) != expected[dispute_id]:
                    item_errors.append(f"{dispute_id}: endpoint changed")
                verdict = decision.get("decision")
                data_items = decision.get("canonical_data_items", [])
                rationale = decision.get("rationale", "").strip()
                confidence = decision.get("confidence", 0)
                if verdict not in {"include", "exclude"} and not pending_template:
                    item_errors.append(f"{dispute_id}: decision is not final")
                if verdict == "include" and not data_items:
                    item_errors.append(f"{dispute_id}: included edge lacks canonical_data_items")
                if verdict == "exclude" and data_items:
                    item_errors.append(f"{dispute_id}: excluded edge must not contain data items")
                if len(rationale) < 8 and not pending_template:
                    item_errors.append(f"{dispute_id}: rationale is too short")
                if (not isinstance(confidence, (int, float)) or confidence <= 0) and not pending_template:
                    item_errors.append(f"{dispute_id}: confidence must be greater than zero")
            records.append({"task_id": task_id, "valid": not item_errors, "errors": sorted(set(item_errors))})

    if len(adjudicator_ids) > 1:
        warnings.append(
            "multiple adjudicator IDs are present; this is allowed only if allocation was fixed before review"
        )
    valid_count = sum(record["valid"] for record in records)
    complete = not errors and valid_count == manifest["task_count"]
    report = {
        "status": (
            "READY_FOR_ADJUDICATION" if complete and allow_pending
            else "PASS" if complete
            else "FAIL"
        ),
        "task_count": manifest["task_count"],
        "valid_task_count": valid_count,
        "invalid_task_count": manifest["task_count"] - valid_count,
        "adjudicator_ids": sorted(adjudicator_ids),
        "errors": errors,
        "warnings": warnings,
        "records": records,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    report = validate(package, allow_pending=args.allow_pending)
    output = args.output or package / "validation" / "adjudication_return_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "task_count": report["task_count"],
        "valid_task_count": report["valid_task_count"],
        "invalid_task_count": report["invalid_task_count"],
        "adjudicator_ids": report["adjudicator_ids"],
        "errors": report["errors"],
        "warnings": report["warnings"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"PASS", "READY_FOR_ADJUDICATION"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
