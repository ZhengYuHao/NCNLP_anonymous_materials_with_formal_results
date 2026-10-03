#!/usr/bin/env python3
"""Run evaluator-separated RQ2 system qualification smoke cases."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path


FORMAL = Path(__file__).resolve().parent
EXPERIMENT = FORMAL.parent
BENCHMARK = EXPERIMENT / "benchmark_v2"
for path in (EXPERIMENT.parent, EXPERIMENT, BENCHMARK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from n8n_h1_mock_executor import MockPlanExecutor
from n8n_h1_oracle_runtime import evaluate_case


H1 = BENCHMARK / "n8n_conversion_pilot" / "h1_execution"
LEGACY_ORACLE = H1 / "frozen" / "gold_oracle_v1" / "oracle"
S1 = BENCHMARK / "n8n_native_s1" / "source_review_round_v2"
S1_ORACLE = S1 / "d10_gold_oracle_round_v1" / "frozen" / "oracle_spec_v4" / "oracle"
S1_TASKS = S1 / "runtime" / "unified_tasks_v5_candidate_r5" / "development_qualification"
REPORTS = FORMAL / "reports" / "smoke"
SYSTEMS = ["direct_llm", "dspy", "lmql", "pal", "ncnlp"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def oracle_case(task_id: str, case_id: str) -> dict:
    path = (
        S1_ORACLE / f"{task_id}.oracle.json"
        if task_id.startswith("N8F-")
        else LEGACY_ORACLE / f"{task_id}.response.json"
    )
    document = read_json(path)
    return next(item for item in document["cases"] if item["case_id"] == case_id)


def run(system_id: str, task_id: str, case_id: str, repeat_index: int) -> dict:
    module = importlib.import_module(f"formal_experiment.systems.{system_id}.system")
    record = module.run_case(task_id, case_id, repeat_index=repeat_index)
    executor = (
        MockPlanExecutor(
            task_id,
            case_id,
            tasks_root=S1_TASKS,
            runtime_contracts=None,
            strict_case_actions=True,
        )
        if task_id.startswith("N8F-")
        else MockPlanExecutor(task_id, case_id)
    )
    local_action_executor = None
    if hasattr(module, "consume_local_action_executor"):
        local_action_executor = module.consume_local_action_executor(record["run_id"])
    executed = executor.execute(
        record["actions"],
        local_action_executor=local_action_executor,
    )
    executed["raw_trace"].setdefault("final_state", {}).update(record["final_state"])
    evaluation = evaluate_case(task_id, oracle_case(task_id, case_id), executed["raw_trace"])
    return {
        "system_record": record,
        "executor": {
            "ok": executed["ok"],
            "errors": executed["errors"],
            "raw_trace": executed["raw_trace"],
        },
        "oracle": {
            "passed": evaluation["passed"],
            "failed_assertion_ids": [
                item["assertion_id"] for item in evaluation["assertions"] if not item["passed"]
            ],
            "trace_sha256": evaluation["trace_sha256"],
        },
        "technical_qualification_passed": record["status"] == "success" and executed["ok"],
        "oracle_passed": evaluation["passed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    parser.add_argument("--task-id", default="N8C-003")
    parser.add_argument("--case-id", default="C-POS-VALIDATED")
    parser.add_argument("--repeat-index", type=int, default=1)
    args = parser.parse_args()

    results = []
    for system_id in args.systems:
        try:
            result = run(system_id, args.task_id, args.case_id, args.repeat_index)
        except Exception as exc:
            result = {
                "technical_qualification_passed": False,
                "oracle_passed": False,
                "runner_error": {"type": type(exc).__name__, "message": str(exc)},
            }
        result["system_id"] = system_id
        results.append(result)
        print(
            f"{system_id}: technical={result['technical_qualification_passed']} "
            f"oracle={result['oracle_passed']} "
            f"status={result.get('system_record', {}).get('status', 'runner_error')}",
            flush=True,
        )

    report = {
        "schema_version": "rq2-system-smoke-v2",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "partition": "development_qualification",
        "formal_test_executed": False,
        "task_id": args.task_id,
        "case_id": args.case_id,
        "repeat_index": args.repeat_index,
        "status": "PASS" if all(item["technical_qualification_passed"] for item in results) else "PARTIAL",
        "passed_count": sum(item["technical_qualification_passed"] for item in results),
        "system_count": len(results),
        "results": results,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{args.task_id}_{args.case_id}_r{args.repeat_index}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
