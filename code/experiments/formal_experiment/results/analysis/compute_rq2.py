"""Compute RQ2 metrics (CCR, accuracy, consistency, token costs, tests)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
OUT = HERE / "outputs"

SYSTEMS = ["direct_llm", "dspy", "lmql", "pal", "ncnlp"]
DISPLAY = {"direct_llm": "Direct LLM", "dspy": "DSPy", "lmql": "LMQL", "pal": "PAL",
           "ncnlp": "NCNLP"}
CONSTRUCTION_SYSTEMS = {"dspy", "ncnlp"}


def load_workflows(system):
    """task -> {all_correct, correct_runs, pairs, total_runs, total_tokens}"""
    records = stats.read_jsonl(RESULTS / "rq2/run_records" / f"{system}.jsonl")
    by_task = defaultdict(lambda: defaultdict(list))
    tokens = defaultdict(int)
    for r in records:
        by_task[r["task_id"]][r["case_id"]].append(r)
        tokens[r["task_id"]] += r["usage"]["total_tokens"]
    workflows = {}
    for task, cases in by_task.items():
        all_correct = True
        correct_runs = 0
        pair_matches = 0
        total_runs = 0
        for case, runs in cases.items():
            runs.sort(key=lambda x: x["repeat_index"])
            k = sum(1 for r in runs if r["metadata"]["oracle_passed"])
            sigs = [r["metadata"]["signature"] for r in runs]
            pair = 0
            for s in set(sigs):
                m = sigs.count(s)
                pair += m * (m - 1) // 2
            all_correct = all_correct and k == len(runs)
            correct_runs += k
            pair_matches += pair
            total_runs += len(runs)
        workflows[task] = {"all_correct": all_correct, "correct_runs": correct_runs,
                           "pairs": pair_matches, "total_runs": total_runs,
                           "total_cases": len(cases),
                           "tokens": tokens[task], "mean_tokens": tokens[task] / total_runs}
    return workflows


def main():
    wf = {s: load_workflows(s) for s in SYSTEMS}
    tasks = sorted(wf["ncnlp"])
    assert all(sorted(wf[s]) == tasks for s in SYSTEMS), "workflow sets differ across systems"

    construction = stats.read_jsonl(RESULTS / "rq2/construction_records.jsonl")
    build_totals = defaultdict(int)
    for rec in construction:
        build_totals[rec["system_id"]] += rec["usage"]["total_tokens"]

    systems = {}
    raw_p = {}
    dccr_tasks = {}
    for s in SYSTEMS:
        w = wf[s]
        n = len(tasks)
        passes = [w[t]["all_correct"] for t in tasks]
        k = sum(passes)
        lo, hi = stats.wilson_pct(k, n)
        acc = sum(w[t]["correct_runs"] / w[t]["total_runs"] for t in tasks) / n * 100
        cons = sum(w[t]["pairs"] / (10 * w[t]["total_cases"]) for t in tasks) / n * 100
        build = build_totals.get(s, 0) / n
        run = sum(w[t]["mean_tokens"] for t in tasks) / n
        systems[s] = {
            "display": DISPLAY[s], "pass": k, "total": n,
            "ccr_pct": round(k / n * 100, 2), "wilson_ci": [lo, hi],
            "single_run_accuracy_pct": round(acc, 2),
            "pairwise_consistency_pct": round(cons, 2),
            "build_tokens_per_workflow": round(build),
            "run_tokens_per_workflow": round(run),
        }
        dccr_tasks[s] = {t: (1.0 if passes[i] else 0.0)
                         - (1.0 if wf["ncnlp"][t]["all_correct"] else 0.0) for i, t in enumerate(tasks)}

    for s in SYSTEMS[:-1]:
        b = sum(1 for t in tasks if not wf[s][t]["all_correct"] and wf["ncnlp"][t]["all_correct"])
        c = sum(1 for t in tasks if wf[s][t]["all_correct"] and not wf["ncnlp"][t]["all_correct"])
        raw_p[s] = stats.exact_mcnemar(b, c)
        systems[s]["mcnemar"] = {"b": b, "c": c, "raw_p": raw_p[s]}
    adjusted = stats.holm_adjust(raw_p)
    for s, p in adjusted.items():
        systems[s]["mcnemar"]["holm_p"] = p

    for s in SYSTEMS:
        lo, hi = stats.paired_bootstrap_ci(dccr_tasks[s])
        systems[s]["dccr_pp"] = round(systems[s]["ccr_pct"] - systems["ncnlp"]["ccr_pct"], 2)
        systems[s]["dccr_ci"] = [lo, hi]
        c = systems[s]["build_tokens_per_workflow"]
        r = systems[s]["run_tokens_per_workflow"]
        systems[s]["c_plus_10r"] = c + 10 * r
        if s == "ncnlp":
            systems[s]["breakeven"] = None
        else:
            nb = next((n for n in range(1, 1001)
                       if systems["ncnlp"]["build_tokens_per_workflow"]
                       + n * systems["ncnlp"]["run_tokens_per_workflow"] <= c + n * r), None)
            systems[s]["breakeven"] = nb

    result = {"schema_version": "rq2-results-v1", "workflows": len(tasks), "systems": systems}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rq2_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({s: {k: v for k, v in systems[s].items() if k != "mcnemar"}
                      | {"holm_p": systems[s].get("mcnemar", {}).get("holm_p")}
                      for s in SYSTEMS}, indent=2))


if __name__ == "__main__":
    main()
