"""Development-only single-fault experiments on saved DSL compilations.

This measures the existing public-operation binding gate. It does not stand in
for the full M0 package or the missing upstream FactSpec/assignment experiments.
"""

import argparse
import copy
import hashlib
import io
import json
import logging
import re
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

FORMAL = Path(__file__).resolve().parent
for path in (FORMAL.parent.parent, FORMAL.parent, FORMAL.parent / "benchmark_v2"):
    sys.path.insert(0, str(path))

from n8n_s1_unified_tasks import DEFAULT_ROOT, S1, load_system_view
from n8n_h1_unified_tasks import adapt_to_legacy_ncnlp
from n8n_h1_mock_executor import MockPlanExecutor
from n8n_h1_oracle_runtime import evaluate_case
from run_s1_ncnlp_task_replays import audit_public_action_contract
from pipeline_runner import _compile_public_action_plan_source, run_ncnlp_pipeline_run_only

PREFIX = "# ACTION_CONTRACT_JSON:"
FAULTS = ["delete_threshold", "invert_comparator", "delete_action", "corrupt_assignment",
          "delete_cfg_or_dfg_edge", "illegal_type", "undefined_variable", "break_dsl_syntax",
          "compilable_logic_error"]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


class NoModelCalls:
    def call(self, *args, **kwargs):
        raise RuntimeError("unexpected model call in fixed-response development replay")


def mutate(dsl, fault):
    lines = dsl.splitlines(keepends=True)
    contract_index = next((i for i, line in enumerate(lines) if line.strip().startswith(PREFIX)), None)
    if fault in {"delete_action", "delete_cfg_or_dfg_edge", "compilable_logic_error"} and contract_index is not None:
        contract = json.loads(lines[contract_index].strip()[len(PREFIX):])
        for si, scene in enumerate(contract.get("scenes", [])):
            for bi, binding in enumerate(scene.get("public_operations", [])):
                target = f"action_contract.scenes[{si}].public_operations[{bi}]"
                before = copy.deepcopy(binding)
                if fault == "delete_action":
                    scene["public_operations"].pop(bi)
                elif fault == "delete_cfg_or_dfg_edge":
                    if not binding.get("consumes"):
                        continue
                    binding["consumes"].pop(0)
                    target += ".consumes[0]"
                else:
                    guard = binding.get("guard", "always")
                    if not guard.startswith(("on_success:", "on_failure:")):
                        continue
                    kind, predecessor = guard.split(":", 1)
                    binding["guard"] = ("on_failure:" if kind == "on_success" else "on_success:") + predecessor
                    target += ".guard"
                lines[contract_index] = PREFIX + " " + json.dumps(contract, ensure_ascii=False) + "\n"
                return "".join(lines), {"target": target, "before": before,
                                         "after": None if fault == "delete_action" else binding}
        return None, {"reason": "no applicable structured operation or dependency"}
    if fault == "corrupt_assignment":
        return None, {"reason": "saved ClassifiedSpec unavailable; DSL text replacement would conflate assignment with syntax"}
    patterns = {
        "delete_threshold": (r"^(\s*IF\s+\{\{\w+\}\}\s*(?:>=|<=|>|<)\s*)(-?\d+(?:\.\d+)?)(\s*)$", lambda m: m[1] + m[3]),
        "invert_comparator": (r"^(\s*IF\s+\{\{\w+\}\}\s*)(>=|<=|>|<)(\s*-?\d+(?:\.\d+)?\s*)$", lambda m: m[1] + {">=": "<", "<=": ">", ">": "<=", "<": ">="}[m[2]] + m[3]),
        "illegal_type": (r"^(DEFINE\s+\{\{\w+\}\}:\s*)(\w+)(.*)$", lambda m: m[1] + "InvalidType" + m[3]),
        "undefined_variable": (r"^(\s*\{\{\w+\}\}\s*=\s*)(-?\d+)(\s*)$", lambda m: m[1] + "{{rq1_undefined_variable}}" + m[3]),
        "break_dsl_syntax": (r"^ENDAGENT\s*$", lambda m: ""),
    }
    if fault in patterns:
        pattern, replace = patterns[fault]
        for i, line in enumerate(lines):
            match = re.match(pattern, line.rstrip("\r\n"))
            if match:
                changed = replace(match)
                lines[i] = changed + "\n"
                return "".join(lines), {"target": f"dsl.line[{i + 1}]", "before": line.rstrip(), "after": changed}
    return None, {"reason": "no applicable executable DSL construct in saved compilation"}


def compile_dsl(dsl):
    from prompt_codegenetate import WaActCompiler
    output = io.StringIO()
    from logger import default_logger
    original_handlers = default_logger.handlers[:]
    default_logger.handlers = [logging.StreamHandler(output)]
    try:
        with redirect_stdout(output):
            compiler = WaActCompiler()
            modules, main, details = compiler.compile(dsl, clustering_strategy="hybrid", visualize=False)
            source = compiler.generate_full_code(modules, main, dsl_code=dsl)
            plan = _compile_public_action_plan_source(dsl)
            if plan:
                source += "\n\n" + plan
        compile(source, "<rq1-generated>", "exec")
        return {"ok": True, "python": source, "log": output.getvalue()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "log": output.getvalue()}
    finally:
        default_logger.handlers = original_handlers


def execute(dsl, compilation, task_id, case):
    if not compilation["ok"]:
        return {"outcome": "explicit_failure", "stage": "compiler", "error": compilation["error"]}
    view = load_system_view(task_id, case["case_id"])
    task = adapt_to_legacy_ncnlp(view)
    compiled = {"backend": "dsl_compiler", "dsl_code": dsl, "generated_python": compilation["python"],
                "sensor_specs": [], "classified_spec": None}
    method = run_ncnlp_pipeline_run_only(compiled, task, llm_client=NoModelCalls())
    if not method["ok"]:
        return {"outcome": "explicit_failure", "stage": "runtime", "error": method["error"]}
    executor = MockPlanExecutor(task_id, case["case_id"], tasks_root=DEFAULT_ROOT,
                                runtime_contracts=None, strict_case_actions=True).execute(method["actions"])
    evaluation = evaluate_case(task_id, case, executor["raw_trace"])
    if not executor["ok"]:
        outcome = "explicit_failure"
    elif not evaluation["passed"]:
        outcome = "silent_harmful"
    else:
        outcome = "no_observed_user_impact"
    return {"outcome": outcome, "stage": "executor" if not executor["ok"] else "evaluation",
            "executor_errors": executor["errors"], "evaluation": evaluation,
            "action_plan": method["actions"], "raw_trace": executor["raw_trace"]}


def run_task(task_id, out):
    task_root = DEFAULT_ROOT / "system_inputs" / task_id
    cases_files = sorted((task_root / "instances").glob("*.json"))
    if not cases_files:
        raise ValueError("task is not in the development bundle")
    view = load_system_view(task_id, cases_files[0].stem)
    artifact = S1 / "runtime" / "ncnlp_artifacts_v3" / task_id
    dsl = (artifact / "generated.dsl").read_text(encoding="utf-8")
    metadata = json.loads((artifact / "compile_metadata.json").read_text(encoding="utf-8"))
    if digest(dsl) != metadata["dsl_sha256"]:
        raise ValueError("source DSL changed since saved compile metadata")
    oracle = json.loads((S1 / "d10_gold_oracle_round_v1" / "frozen" / "oracle_spec_v4" / "oracle" / f"{task_id}.oracle.json").read_text(encoding="utf-8"))
    task_out = out / task_id
    task_out.mkdir(parents=True)
    clean_compile = compile_dsl(dsl)
    clean = [execute(dsl, clean_compile, task_id, case) for case in oracle["cases"]]
    clean_gate = audit_public_action_contract(dsl, view["task"]["public_interfaces"])
    clean_ok = all(item["outcome"] == "no_observed_user_impact" for item in clean)
    records = []
    for fault in FAULTS:
        mutant, edit = mutate(dsl, fault)
        record = {"fault_type": fault, "edit": edit, "applicable": mutant is not None}
        if mutant is not None:
            if mutant == dsl:
                raise ValueError("mutation did not change source")
            fault_out = task_out / fault
            fault_out.mkdir()
            (fault_out / "mutant.dsl").write_text(mutant, encoding="utf-8")
            compilation = compile_dsl(mutant)
            (fault_out / "compile.log").write_text(compilation["log"], encoding="utf-8")
            if compilation["ok"]:
                (fault_out / "generated.py").write_text(compilation["python"], encoding="utf-8")
            gate = audit_public_action_contract(mutant, view["task"]["public_interfaces"])
            runs = []
            for case in oracle["cases"]:
                off = execute(mutant, compilation, task_id, case)
                on = {"outcome": "gate_rejection", "stage": "public_binding_gate"} if not gate["passed"] else copy.deepcopy(off)
                runs.append({"case_id": case["case_id"], "gate_off": off, "gate_on": on})
            record.update(mutant_sha256=digest(mutant), compile_ok=compilation["ok"],
                          binding_gate=gate, runs=runs)
        records.append(record)
    report = {"task_id": task_id, "source_dsl_sha256": digest(dsl), "clean_reference_accepted": clean_ok,
              "clean_compile_ok": clean_compile["ok"], "clean_compile_error": clean_compile.get("error"),
              "clean_binding_gate_passed": clean_gate["passed"], "clean_runs": clean, "faults": records}
    (task_out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("output must be new; previous evidence is preserved")
    reports = [run_task(task, args.out) for task in args.tasks]
    observations = [run for report in reports if report["clean_reference_accepted"]
                    for fault in report["faults"] for run in fault.get("runs", [])]
    harmful = [run for run in observations if run["gate_off"]["outcome"] == "silent_harmful"]
    contained = [run for run in harmful if run["gate_on"]["outcome"] != "silent_harmful"]
    summary = {
        "schema_version": "rq1-development-fault-diagnostic-v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "DIAGNOSTIC_ONLY", "full_rq1_ready": False, "formal_test_executed": False,
        "partition": "development_qualification", "model_calls": 0,
        "task_count": len(reports), "clean_task_count": sum(r["clean_reference_accepted"] for r in reports),
        "excluded_clean_tasks": [{"task_id": r["task_id"], "compile_error": r["clean_compile_error"]}
                                 for r in reports if not r["clean_reference_accepted"]],
        "compile_rejections_by_fault": {
            fault: sum(f.get("compile_ok") is False for r in reports for f in r["faults"]
                       if f["fault_type"] == fault and f["applicable"])
            for fault in FAULTS
        },
        "applicable_mutants": sum(f["applicable"] for r in reports for f in r["faults"]),
        "unavailable_faults": [{"task_id": r["task_id"], "fault_type": f["fault_type"], **f["edit"]}
                               for r in reports for f in r["faults"] if not f["applicable"]],
        "observed_case_count": len(observations), "silent_harmful_count": len(harmful),
        "binding_gate_contained_count": len(contained),
        "binding_gate_hfer": len(contained) / len(harmful) if harmful else None,
        "scope": "Actual compiler/runtime and fixed external mocks; only public-binding gate toggled. No full-M0 HFER/FPR/SLA claim.",
        "task_reports": [str((args.out / r["task_id"] / "report.json").resolve()) for r in reports],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
