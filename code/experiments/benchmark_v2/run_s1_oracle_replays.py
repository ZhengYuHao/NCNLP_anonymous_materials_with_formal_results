#!/usr/bin/env python3
"""Evaluate three system traces per case against the frozen 109-task S1 Oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from n8n_h1_oracle_runtime import evaluate_case, replay_is_deterministic


BENCHMARK = Path(__file__).resolve().parent
S1 = BENCHMARK / "n8n_native_s1" / "source_review_round_v2"
DEFAULT_ORACLE = S1 / "d10_gold_oracle_round_v1" / "frozen" / "oracle_spec_v2"
DEFAULT_SPLIT = S1 / "d14_split_d15_double_gold_v2" / "reports" / "dataset_split_v1.json"
DEFAULT_TRACES = S1 / "runtime" / "oracle_traces_v2"
DEFAULT_REPORTS = S1 / "runtime" / "reports"
DEFAULT_G2_GATE = S1 / "runtime" / "gates" / "g2_gate_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def expected_case_input_hash(oracle: dict[str, Any], case: dict[str, Any]) -> str:
    fixtures = {item["fixture_id"]: item["input_data"] for item in oracle["fixtures"]}
    selected = {fixture_id: fixtures[fixture_id] for fixture_id in case["fixture_ids"]}
    input_data = next(iter(selected.values())) if len(selected) == 1 else {"fixtures": selected}
    return canonical_hash(input_data)


def evaluate_replay_set(
    task_id: str,
    case: dict[str, Any],
    envelopes: list[dict[str, Any]],
    expected_input_hash: str,
) -> dict[str, Any]:
    errors = []
    if [item.get("run_index") for item in envelopes] != [1, 2, 3]:
        errors.append("run_index must be exactly 1, 2, 3")
    if any(item.get("task_id") != task_id for item in envelopes):
        errors.append("task_id mismatch")
    if any(item.get("case_id") != case["case_id"] for item in envelopes):
        errors.append("case_id mismatch")
    input_hashes = {item.get("input_sha256") for item in envelopes}
    if input_hashes != {expected_input_hash}:
        errors.append("run input hash differs from the frozen case fixtures")
    if any(item.get("method_ok") is not True for item in envelopes):
        errors.append("one or more system method executions failed")
    if any(item.get("executor_ok") is not True for item in envelopes):
        errors.append("one or more strict Mock executions failed")

    runs = []
    for envelope in envelopes:
        raw_trace = envelope.get("raw_trace")
        if not isinstance(raw_trace, dict):
            errors.append(f"run {envelope.get('run_index')} raw_trace must be an object")
            continue
        runs.append(evaluate_case(task_id, case, raw_trace))
    deterministic = len(runs) == 3 and replay_is_deterministic(runs)
    if len(runs) == 3 and not deterministic:
        errors.append("trace hashes differ across the three identical-input runs")
    if any(not run["passed"] for run in runs):
        errors.append("one or more Oracle assertions failed")
    return {
        "task_id": task_id,
        "case_id": case["case_id"],
        "passed": not errors and len(runs) == 3,
        "deterministic": deterministic,
        "expected_input_sha256": expected_input_hash,
        "runs": runs,
        "executor_errors": [
            {
                "run_index": envelope.get("run_index"),
                "errors": envelope.get("executor_errors", []),
            }
            for envelope in envelopes
            if envelope.get("executor_errors")
        ],
        "errors": errors,
    }


def task_ids_for_partition(split_path: Path, partition: str) -> list[str]:
    split = read_json(split_path)
    if split.get("selection_reads_system_outputs") is not False:
        raise ValueError("dataset split isolation flag is invalid")
    return sorted(
        item["task_id"] for item in split["records"]
        if item["dataset_partition"] == partition
    )


def require_formal_test_gate(partition: str, gate_path: Path) -> None:
    if partition != "formal_test":
        return
    if not gate_path.is_file():
        raise PermissionError("formal test replay is sealed until the G2 gate file exists")
    gate = read_json(gate_path)
    if gate.get("status") != "PASS" or gate.get("formal_test_authorized") is not True:
        raise PermissionError("formal test replay is sealed because G2 has not authorized it")


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_formal_test_gate(args.partition, args.g2_gate.resolve())
    oracle_root = args.oracle.resolve()
    traces_root = args.traces.resolve()
    task_ids = task_ids_for_partition(args.split.resolve(), args.partition)
    records = []
    missing = []
    expected_trace_count = 0
    for task_id in task_ids:
        oracle = read_json(oracle_root / "oracle" / f"{task_id}.oracle.json")
        for case in oracle["cases"]:
            expected_trace_count += 3
            paths = [
                traces_root / args.system / task_id / case["case_id"] / f"run-{index}.json"
                for index in range(1, 4)
            ]
            absent = [str(path) for path in paths if not path.is_file()]
            if absent:
                missing.extend(absent)
                continue
            envelopes = [read_json(path) for path in paths]
            records.append(evaluate_replay_set(
                task_id,
                case,
                envelopes,
                expected_case_input_hash(oracle, case),
            ))

    if missing:
        status = "WAITING_SYSTEM_TRACES"
    elif all(record["passed"] for record in records):
        status = "ORACLE_REPLAY_ACCEPTED"
    else:
        status = "ORACLE_REPLAY_REJECTED"
    return {
        "schema_version": "n8n-s1-oracle-three-replay-report-v2",
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "system": args.system,
        "partition": args.partition,
        "formal_test_executed": args.partition == "formal_test" and not missing,
        "task_count": len(task_ids),
        "case_count": len(records) + len(missing) // 3,
        "expected_trace_count": expected_trace_count,
        "received_trace_count": expected_trace_count - len(missing),
        "passed_case_count": sum(record["passed"] for record in records),
        "deterministic_case_count": sum(record["deterministic"] for record in records),
        "missing_trace_count": len(missing),
        "missing_traces": missing,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument(
        "--partition",
        choices=["development_qualification", "formal_test"],
        default="development_qualification",
    )
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--g2-gate", type=Path, default=DEFAULT_G2_GATE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args)
    output = args.output or DEFAULT_REPORTS / f"{args.system}_{args.partition}_oracle_replay_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"missing_traces", "records"}}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ORACLE_REPLAY_ACCEPTED" else 2 if report["status"] == "WAITING_SYSTEM_TRACES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
