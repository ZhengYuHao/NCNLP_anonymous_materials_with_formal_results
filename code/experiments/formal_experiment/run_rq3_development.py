"""Run supported RQ3 conditions with versioned development-only artifacts."""

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

FORMAL = Path(__file__).resolve().parent
for path in (FORMAL.parent.parent, FORMAL.parent, FORMAL.parent / "benchmark_v2"):
    sys.path.insert(0, str(path))

from n8n_s1_unified_tasks import DEFAULT_ROOT, S1, load_system_view
from n8n_h1_unified_tasks import adapt_to_legacy_ncnlp
from n8n_h1_mock_executor import MockPlanExecutor
from n8n_h1_oracle_runtime import evaluate_case
from pipeline_runner import run_ncnlp_pipeline_compile_once, run_ncnlp_pipeline_run_only
from run_s1_ncnlp_task_replays import audit_public_action_contract
from formal_experiment.systems.common import create_formal_llm_client

CONDITIONS = ["Full NCNLP", "Staged-LLM-Assign", "Coarse M3", "w/o M0", "Rule-only M3"]


def serial(value):
    if is_dataclass(value):
        return serial(asdict(value))
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def write(path, value):
    path.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class UsageLedger:
    def __init__(self, client):
        self.client = client
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.client, name)

    def extract_entities(self, *args, **kwargs):
        start = len(self.calls)
        entities, stats = type(self.client).extract_entities(self, *args, **kwargs)
        stats = dict(stats)
        stats["usage"] = {"total_tokens": sum(int(item["usage"].get("total_tokens", 0) or 0)
                                              for item in self.calls[start:])}
        return entities, stats

    def call(self, *args, **kwargs):
        response = self.client.call(*args, **kwargs)
        self.calls.append({"usage": dict(response.usage or {}), "model": getattr(response, "model", None)})
        return response


def run(task_id, condition, out, repeats):
    cases = sorted((DEFAULT_ROOT / "system_inputs" / task_id / "instances").glob("*.json"))
    if not cases:
        raise ValueError("development task required")
    view = load_system_view(task_id, cases[0].stem)
    task = adapt_to_legacy_ncnlp(view)
    task["input_data"] = {}
    task["source"] = "s1_rq3_development"
    out.mkdir(parents=True, exist_ok=False)
    client = UsageLedger(create_formal_llm_client())
    report = {"schema_version": "rq3-development-run-v1", "task_id": task_id, "condition": condition,
              "partition": "development_qualification", "formal_test_executed": False,
              "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "runs": [], "status": "STARTED"}
    try:
        compiled = run_ncnlp_pipeline_compile_once(task, llm_client=client,
                    backend="dsl_compiler", research_condition=condition)
        for key in ("normalized", "entities", "fact_spec", "classified_spec", "assignment_audit", "m0_audit", "compile_details"):
            write(out / f"{key}.json", compiled.get(key))
        for name, key in (("generated.dsl", "dsl_code"), ("generated.py", "generated_python")):
            (out / name).write_text(compiled[key], encoding="utf-8")
        if condition == "w/o M0":
            gate = {
                "passed": True,
                "applied": False,
                "reason": "disabled_by_rq3_without_m0_condition",
            }
        else:
            gate = audit_public_action_contract(compiled["dsl_code"], view["task"]["public_interfaces"])
        measured_compile_tokens = sum(int(c["usage"].get("total_tokens", 0) or 0) for c in client.calls)
        report.update(compile_gate=gate, compile_tokens=measured_compile_tokens,
                      pipeline_reported_compile_tokens=compiled["compile_tokens"],
                      compile_call_count=len(client.calls), backend=compiled["backend"])
        if not gate["passed"]:
            report["status"] = "COMPILE_CONTRACT_REJECTED"
        else:
            oracle = json.loads((S1 / "d10_gold_oracle_round_v1" / "frozen" / "oracle_spec_v4" / "oracle" / f"{task_id}.oracle.json").read_text(encoding="utf-8"))
            for case in oracle["cases"]:
                case_task = adapt_to_legacy_ncnlp(load_system_view(task_id, case["case_id"]))
                for repeat in range(1, repeats + 1):
                    method = run_ncnlp_pipeline_run_only(compiled, case_task, llm_client=client)
                    executed = MockPlanExecutor(
                        task_id,
                        case["case_id"],
                        tasks_root=DEFAULT_ROOT,
                        runtime_contracts=None,
                        strict_case_actions=True,
                    ).execute(
                        method["actions"],
                        local_action_executor=method.get("local_action_executor"),
                    )
                    evaluated = evaluate_case(task_id, case, executed["raw_trace"])
                    report["runs"].append({"case_id": case["case_id"], "repeat": repeat,
                        "method_ok": method["ok"], "method_error": method["error"],
                        "action_plan": method["actions"], "executor": executed, "evaluation": evaluated,
                        "passed": method["ok"] and executed["ok"] and evaluated["passed"],
                        "runtime_tokens": method.get("runtime_tokens", 0)})
            report["status"] = "REPLAY_ACCEPTED" if all(r["passed"] for r in report["runs"]) else "REPLAY_FAILED"
    except Exception as exc:
        report.update(status="METHOD_ERROR", error={"type": type(exc).__name__, "message": str(exc)})
    report["provider_calls"] = client.calls
    report["total_provider_tokens"] = sum(int(c["usage"].get("total_tokens", 0) or 0) for c in client.calls)
    report["scope"] = "Engineering replay with fixed external mocks; no formal RQ3 effect claim."
    report["artifacts"] = [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest().upper()}
                            for p in sorted(out.iterdir()) if p.is_file()]
    write(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, required=True)
    parser.add_argument("--repeats", type=int, choices=[1, 3], default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("output must be new")
    args.out.mkdir(parents=True)
    from logger import default_logger
    default_logger.handlers = [logging.FileHandler(args.out / "runtime.log", encoding="utf-8")]
    results = []
    for index, condition in enumerate(args.conditions):
        report = run(args.task, condition, args.out / f"condition_{index + 1:02d}", args.repeats)
        results.append({k: report.get(k) for k in ("condition", "status", "total_provider_tokens", "error")})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    write(args.out / "summary.json", {"results": results, "formal_test_executed": False})


if __name__ == "__main__":
    main()
