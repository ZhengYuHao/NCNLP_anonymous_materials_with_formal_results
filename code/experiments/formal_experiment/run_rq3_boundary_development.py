#!/usr/bin/env python3
"""Qualify the two RQ3 boundary conditions on development tasks only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


FORMAL = Path(__file__).resolve().parent
for path in (FORMAL.parent.parent, FORMAL.parent, FORMAL.parent / "benchmark_v2"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from formal_experiment.systems.direct_llm.system import run_case as run_direct_case
from formal_experiment.systems.common import create_formal_llm_client
from n8n_h1_mock_executor import MockPlanExecutor
from n8n_h1_oracle_runtime import evaluate_case
from n8n_h1_unified_tasks import adapt_to_legacy_ncnlp
from n8n_s1_unified_tasks import DEFAULT_ROOT, S1, load_system_view
from pipeline_runner import run_ncnlp_pipeline_compile_once, run_ncnlp_pipeline_run_only


CONDITIONS = ["LLM-generated Python", "All-LLM boundary"]
ORACLE = S1 / "d10_gold_oracle_round_v1" / "frozen" / "oracle_spec_v4" / "oracle"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def task_cases(task_id: str) -> tuple[dict, list[dict]]:
    path = ORACLE / f"{task_id}.oracle.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    return document, document["cases"]


def execute_actions(task_id: str, case: dict, actions: list[dict], local_executor=None) -> dict:
    executed = MockPlanExecutor(
        task_id,
        case["case_id"],
        tasks_root=DEFAULT_ROOT,
        runtime_contracts=None,
        strict_case_actions=True,
    ).execute(actions, local_action_executor=local_executor)
    evaluated = evaluate_case(task_id, case, executed["raw_trace"])
    return {"executor": executed, "evaluation": evaluated}


def qualify_llm_python(task_id: str, cases: list[dict], repeats: int, out: Path) -> dict:
    first_view = load_system_view(task_id, cases[0]["case_id"])
    task = adapt_to_legacy_ncnlp(first_view)
    task.update(input_data={}, source="s1_rq3_development")
    client = create_formal_llm_client()
    report = {
        "schema_version": "rq3-boundary-development-v1",
        "condition": "LLM-generated Python",
        "task_id": task_id,
        "partition": "development_qualification",
        "formal_test_executed": False,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runs": [],
    }
    try:
        compiled = run_ncnlp_pipeline_compile_once(
            task,
            llm_client=client,
            backend="llm_skeleton",
            research_condition="Full NCNLP",
        )
        compile(compiled["generated_python"], "<rq3-llm-python>", "exec")
        (out / "generated.py").write_text(compiled["generated_python"], encoding="utf-8")
        if compiled.get("dsl_code"):
            (out / "generated.dsl").write_text(compiled["dsl_code"], encoding="utf-8")
        report.update(
            compile_status="ACCEPTED",
            backend=compiled["backend"],
            compile_tokens=int(compiled.get("compile_tokens", 0) or 0),
        )
        for case in cases:
            case_task = adapt_to_legacy_ncnlp(load_system_view(task_id, case["case_id"]))
            for repeat in range(1, repeats + 1):
                method = run_ncnlp_pipeline_run_only(compiled, case_task, llm_client=client)
                evidence = execute_actions(
                    task_id,
                    case,
                    method["actions"],
                    method.get("local_action_executor"),
                )
                report["runs"].append({
                    "case_id": case["case_id"],
                    "repeat": repeat,
                    "method_ok": method["ok"],
                    "method_error": method["error"],
                    "actions": method["actions"],
                    **evidence,
                    "passed": method["ok"] and evidence["executor"]["ok"]
                    and evidence["evaluation"]["passed"],
                    "runtime_tokens": int(method.get("runtime_tokens", 0) or 0),
                })
        report["status"] = "REPLAY_ACCEPTED" if all(
            item["passed"] for item in report["runs"]
        ) else "REPLAY_FAILED"
    except Exception as exc:
        report.update(
            status="METHOD_ERROR",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
    return report


def qualify_all_llm(task_id: str, cases: list[dict], repeats: int, out: Path) -> dict:
    client = create_formal_llm_client()
    report = {
        "schema_version": "rq3-boundary-development-v1",
        "condition": "All-LLM boundary",
        "task_id": task_id,
        "partition": "development_qualification",
        "formal_test_executed": False,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "compile_status": "NOT_APPLICABLE",
        "compile_tokens": 0,
        "runs": [],
    }
    for case in cases:
        for repeat in range(1, repeats + 1):
            record = run_direct_case(
                task_id,
                case["case_id"],
                repeat_index=repeat,
                client=client,
            )
            evidence = execute_actions(task_id, case, record["actions"])
            passed = (
                record["status"] == "success"
                and evidence["executor"]["ok"]
                and evidence["evaluation"]["passed"]
            )
            report["runs"].append({
                "case_id": case["case_id"],
                "repeat": repeat,
                "system_record": record,
                **evidence,
                "passed": passed,
            })
    report["status"] = "REPLAY_ACCEPTED" if all(
        item["passed"] for item in report["runs"]
    ) else "REPLAY_FAILED"
    report["total_provider_tokens"] = sum(
        int(item["system_record"]["usage"].get("total_tokens", 0) or 0)
        for item in report["runs"]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, required=True)
    parser.add_argument("--repeats", type=int, choices=(1, 3), default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("output must be new")
    args.out.mkdir(parents=True)
    _, cases = task_cases(args.task)
    summaries = []
    for index, condition in enumerate(args.conditions, start=1):
        condition_out = args.out / f"condition_{index:02d}"
        condition_out.mkdir()
        report = (
            qualify_llm_python(args.task, cases, args.repeats, condition_out)
            if condition == "LLM-generated Python"
            else qualify_all_llm(args.task, cases, args.repeats, condition_out)
        )
        report["scope"] = (
            "Development qualification with fixed inputs, mocks, executor, evaluator, model, "
            "and repetition count; no formal RQ3 effect claim."
        )
        write_json(condition_out / "report.json", report)
        digest = hashlib.sha256((condition_out / "report.json").read_bytes()).hexdigest().upper()
        summaries.append({
            "condition": condition,
            "status": report["status"],
            "report_sha256": digest,
        })
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    write_json(args.out / "summary.json", {
        "formal_test_executed": False,
        "task_id": args.task,
        "results": summaries,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
