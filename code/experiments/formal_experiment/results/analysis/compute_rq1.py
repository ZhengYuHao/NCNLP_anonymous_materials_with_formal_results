"""Compute RQ1 metrics from the reconstructed records.

Outputs outputs/rq1_results.json and the two manuscript tables.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
OUT = HERE / "outputs"

FAULT_LABELS = {
    "delete_threshold": "Remove threshold",
    "invert_comparator": "Invert comparison",
    "delete_action": "Remove action",
    "corrupt_assignment": "Corrupt node assignment",
    "delete_cfg_or_dfg_edge": "Remove control/data edge",
    "illegal_type": "Illegal type",
    "undefined_variable": "Undefined variable",
    "break_dsl_syntax": "Invalid syntax",
    "compilable_logic_error": "Wrong business logic",
}
FAULT_GROUPS = {
    "delete_threshold": "Facts", "invert_comparator": "Facts", "delete_action": "Facts",
    "corrupt_assignment": "Assignment", "delete_cfg_or_dfg_edge": "Relations",
    "illegal_type": "Types", "undefined_variable": "DSL", "break_dsl_syntax": "DSL",
    "compilable_logic_error": "DSL",
}
STAGE_ROWS = [
    ("Recovery", "Constraint retention", "constraint_retention"),
    ("Recovery", "Typed entities (EM)", "typed_entities_em"),
    ("Recovery", "Fact set (EM)", "fact_set_em"),
    ("Assign/compile", "Node assignment (EM)", "node_assignment_em"),
    ("Assign/compile", "Workflow relations (EM)", "workflow_relations_em"),
    ("Assign/compile", "Compilation + behavior", "compilation_behavior_consistent"),
    ("Execution", "Actions + states (round 1)", "actions_states_round1_pass"),
]
FAULT_ORDER = ["delete_threshold", "invert_comparator", "delete_action", "corrupt_assignment",
               "delete_cfg_or_dfg_edge", "illegal_type", "undefined_variable",
               "break_dsl_syntax", "compilable_logic_error"]


def main():
    stage = stats.read_jsonl(RESULTS / "rq1/stage_artifacts.jsonl")
    faults = stats.read_jsonl(RESULTS / "rq1/fault_injection.jsonl")
    clean = stats.read_jsonl(RESULTS / "rq1/clean_checks.jsonl")
    natural = stats.read_jsonl(RESULTS / "rq1/natural_errors.jsonl")

    n_tasks = len(stage)
    stage_rows = []
    for group, label, field in STAGE_ROWS:
        k = sum(1 for r in stage if r[field])
        lo, hi = stats.wilson_pct(k, n_tasks)
        stage_rows.append({"stage": group, "metric": label, "count": k, "total": n_tasks,
                           "rate_pct": round(k / n_tasks * 100, 2), "wilson_ci": [lo, hi]})

    per_fault = {}
    for fault in FAULT_ORDER:
        rows = [r for r in faults if r["fault_type"] == fault]
        silent = [r for r in rows if r["checking_disabled"]["outcome"] == "silent_harmful"]
        outcomes = defaultdict(int)
        for r in silent:
            outcomes[r["checking_enabled"]["outcome"]] += 1
        per_fault[fault] = {
            "group": FAULT_GROUPS[fault], "label": FAULT_LABELS[fault],
            "injected": len(rows), "silent": len(silent),
            "repaired": outcomes["repaired"], "refused": outcomes["refused"],
            "contained": outcomes["contained"], "escaped": outcomes["escaped"],
            "explicit_failure": sum(1 for r in rows
                                    if r["checking_disabled"]["outcome"] == "explicit_failure"),
            "unchanged": sum(1 for r in rows
                             if r["checking_disabled"]["outcome"] == "unchanged"),
        }
    total_silent = sum(v["silent"] for v in per_fault.values())
    total_escaped = sum(v["escaped"] for v in per_fault.values())
    hfer = round(total_escaped / total_silent * 100, 2)
    fpr_count = sum(1 for r in clean if r["false_intervention"])
    fpr = round(fpr_count / len(clean) * 100, 2)

    per_task_silent = defaultdict(int)
    per_task_escaped = defaultdict(int)
    per_task_fi = defaultdict(int)
    for r in faults:
        if r["checking_disabled"]["outcome"] == "silent_harmful":
            per_task_silent[r["task_id"]] += 1
            if r["checking_enabled"]["outcome"] == "escaped":
                per_task_escaped[r["task_id"]] += 1
    for r in clean:
        per_task_fi[r["task_id"]] = 1 if r["false_intervention"] else 0
    hfer_ci = stats.clustered_ratio_bootstrap(per_task_escaped, per_task_silent)
    fpr_ci = stats.clustered_ratio_bootstrap(per_task_fi, {t: 1 for t in per_task_fi})

    replay_consistent = sum(1 for r in stage
                            if r["compilation_behavior_consistent"]
                            and r["replay_semantic_returns_consistent"])

    result = {
        "schema_version": "rq1-results-v1",
        "n_tasks": n_tasks,
        "stage_rows": stage_rows,
        "replay_consistent_among_compiled_behavior": replay_consistent,
        "fault_table": [per_fault[f] for f in FAULT_ORDER],
        "totals": {"silent": total_silent, "repaired": sum(v["repaired"] for v in per_fault.values()),
                   "refused": sum(v["refused"] for v in per_fault.values()),
                   "contained": sum(v["contained"] for v in per_fault.values()),
                   "escaped": total_escaped,
                   "explicit_failure": sum(v["explicit_failure"] for v in per_fault.values()),
                   "unchanged": sum(v["unchanged"] for v in per_fault.values())},
        "hfer_pct": hfer, "hfer_ci": list(hfer_ci),
        "fpr_count": fpr_count, "fpr_total": len(clean), "fpr_pct": fpr, "fpr_ci": list(fpr_ci),
        "natural_errors": {"instances": len(natural),
                           "reached_erroneous_user_result": sum(
                               1 for r in natural if r["reached_erroneous_user_result"])},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rq1_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"stage_counts": [r["count"] for r in stage_rows],
                      "silent_total": total_silent, "escaped": total_escaped,
                      "hfer": hfer, "fpr": fpr}, indent=2))


if __name__ == "__main__":
    main()
