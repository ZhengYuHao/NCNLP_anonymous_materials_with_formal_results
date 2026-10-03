"""Compute RQ3 metrics (ablation CCR, contrasts, pre-check quality, costs)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
OUT = HERE / "outputs"

CONFIGS = ["model_assign", "fused", "no_check", "model_code", "rule_recover", "model_exec"]
CONTRASTS = [
    ("Model-assign - Full", "model_assign", "full"),
    ("Fused - Model-assign", "fused", "model_assign"),
    ("Fused - Full", "fused", "full"),
    ("No-check - Full", "no_check", "full"),
    ("Model-code - Full", "model_code", "full"),
    ("Rule-recover - Full", "rule_recover", "full"),
    ("Model-exec - Full", "model_exec", "full"),
    ("Model-exec - Direct LLM", "model_exec", "direct_llm"),
]


def load_workflows(path, id_field):
    records = stats.read_jsonl(path)
    by_task = defaultdict(lambda: defaultdict(list))
    tokens = defaultdict(int)
    for r in records:
        by_task[r["task_id"]][r["case_id"]].append(r)
        tokens[r["task_id"]] += r["usage"]["total_tokens"]
    workflows = {}
    for task, cases in by_task.items():
        all_correct = all(r["metadata"]["oracle_passed"]
                          for case in cases.values() for r in case)
        total_runs = sum(len(c) for c in cases.values())
        workflows[task] = {"all_correct": all_correct,
                           "mean_tokens": tokens[task] / total_runs}
    return workflows


def main():
    wf = {"full": load_workflows(RESULTS / "rq2/run_records/ncnlp.jsonl", "system_id"),
          "direct_llm": load_workflows(RESULTS / "rq2/run_records/direct_llm.jsonl", "system_id")}
    for cfg in CONFIGS:
        wf[cfg] = load_workflows(RESULTS / "rq3/run_records" / f"{cfg}.jsonl", "config_id")
    tasks = sorted(wf["full"])
    assert all(sorted(wf[c]) == tasks for c in wf), "workflow sets differ across configs"

    construction = stats.read_jsonl(RESULTS / "rq3/construction_records.jsonl")
    build_totals = defaultdict(int)
    for rec in construction:
        build_totals[rec["config_id"]] += rec["usage"]["total_tokens"]

    configs = {}
    for name, w in wf.items():
        n = len(tasks)
        passes = [w[t]["all_correct"] for t in tasks]
        k = sum(passes)
        lo, hi = stats.wilson_pct(k, n)
        if name in CONFIGS:
            build = build_totals.get(name, 0) / n
        elif name == "full":
            build = 6000.0  # from RQ2 construction records (480,000 / 80)
        else:
            build = 0.0
        run = sum(w[t]["mean_tokens"] for t in tasks) / n
        configs[name] = {"pass": k, "total": n, "ccr_pct": round(k / n * 100, 2),
                         "wilson_ci": [lo, hi],
                         "build_tokens_per_workflow": round(build),
                         "run_tokens_per_workflow": round(run)}
    for name in CONFIGS:
        configs[name]["dc_tokens"] = configs[name]["build_tokens_per_workflow"] - configs["full"]["build_tokens_per_workflow"]
        configs[name]["dr_tokens"] = configs[name]["run_tokens_per_workflow"] - configs["full"]["run_tokens_per_workflow"]

    raw_p = {}
    contrast_rows = []
    for label, a, b in CONTRASTS:
        pa = {t: wf[a][t]["all_correct"] for t in tasks}
        pb = {t: wf[b][t]["all_correct"] for t in tasks}
        delta = {t: (1.0 if pa[t] else 0.0) - (1.0 if pb[t] else 0.0) for t in tasks}
        b_count = sum(1 for t in tasks if not pa[t] and pb[t])
        c_count = sum(1 for t in tasks if pa[t] and not pb[t])
        raw_p[label] = stats.exact_mcnemar(b_count, c_count)
        lo, hi = stats.paired_bootstrap_ci(delta)
        contrast_rows.append({"comparison": label,
                              "dccr_pp": round(sum(delta.values()) / len(tasks) * 100, 2),
                              "ci": [lo, hi], "b": b_count, "c": c_count,
                              "raw_p": raw_p[label]})
    adjusted = stats.holm_adjust(raw_p)
    for row in contrast_rows:
        row["holm_p"] = adjusted[row["comparison"]]

    precheck_rows = []
    pre = stats.read_jsonl(RESULTS / "rq3/precheck_records.jsonl")
    n = len(tasks)
    for cfg in ("full", "model_assign", "fused"):
        rows = [r for r in pre if r["config_id"] == cfg]
        precheck_rows.append({
            "config": cfg,
            "facts_em_pct": round(sum(r["facts_em"] for r in rows) / n * 100, 2),
            "assignment_em_pct": round(sum(r["assignment_em"] for r in rows) / n * 100, 2),
            "relations_em_pct": round(sum(r["relations_em"] for r in rows) / n * 100, 2),
            "intervention_pct": round(sum(r["intervention"] for r in rows) / n * 100, 2),
            "repairs_per_workflow": round(sum(r["repairs"] for r in rows) / n, 3),
        })

    result = {"schema_version": "rq3-results-v1", "workflows": len(tasks),
              "configs": configs, "contrasts": contrast_rows, "precheck": precheck_rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rq3_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"configs": {c: configs[c]["ccr_pct"] for c in configs},
                      "contrasts": [{k: r[k] for k in ("comparison", "dccr_pp", "holm_p")}
                                    for r in contrast_rows]}, indent=2))


if __name__ == "__main__":
    main()
