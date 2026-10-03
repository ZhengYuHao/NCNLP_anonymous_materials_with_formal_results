"""Verify the recomputed results against the manuscript's reported values.

Deterministic quantities (counts, rates, Wilson intervals, cost means,
break-even points, significance conclusions) must match the manuscript exactly
(within 2-decimal rounding). Bootstrap intervals and adjusted p-values are
recomputed from the reconstructed records and are compared with a documented
tolerance; every deviation is written to outputs/reported_vs_recomputed.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

CI_TOL = 0.011          # Wilson rounding-level tolerance
BOOT_TOL = 3.0          # bootstrap CI tolerance in percentage points
P_TOL_FACTOR = 10.0     # adjusted p tolerance factor (order-of-magnitude)

# ---- manuscript-reported values -------------------------------------------
MS = {
    "rq1_stage": [
        (80, 80, [95.42, 100.00]), (79, 80, [93.25, 99.78]), (79, 80, [93.25, 99.78]),
        (78, 80, [91.34, 99.31]), (77, 80, [89.55, 98.72]), (76, 80, [87.84, 98.04]),
        (74, 80, [84.59, 96.52]),
    ],
    "rq1_faults": {
        "Remove threshold": (66, 50, 10, 5, 1),
        "Invert comparison": (70, 52, 12, 4, 2),
        "Remove action": (64, 48, 10, 5, 1),
        "Corrupt node assignment": (58, 42, 12, 3, 1),
        "Remove control/data edge": (62, 42, 14, 5, 1),
        "Illegal type": (20, 10, 10, 0, 0),
        "Undefined variable": (0, 0, 0, 0, 0),
        "Invalid syntax": (0, 0, 0, 0, 0),
        "Wrong business logic": (72, 50, 12, 7, 3),
    },
    "rq1_totals": {"silent": 412, "repaired": 294, "refused": 80, "contained": 29,
                   "escaped": 9, "explicit_failure": 258, "unchanged": 50},
    "rq1_hfer": (2.18, [0.73, 3.96]),
    "rq1_fpr": (1.25, [0.00, 3.75]),
    "rq1_natural": (20, 3),
    "rq2": {
        "direct_llm": {"pass": 42, "ccr": 52.50, "ci": [41.70, 63.08], "dccr": -37.50,
                       "dccr_ci": [-48.75, -27.50], "acc": 76.75, "build": 0, "run": 3000,
                       "c10r": 30000, "breakeven": 3},
        "dspy": {"pass": 54, "ccr": 67.50, "ci": [56.64, 76.76], "dccr": -22.50,
                 "dccr_ci": [-32.50, -13.75], "acc": 84.25, "build": 4000, "run": 2200,
                 "c10r": 26000, "breakeven": 2},
        "lmql": {"pass": 50, "ccr": 62.50, "ci": [51.55, 72.31], "dccr": -27.50,
                 "dccr_ci": [-37.50, -17.50], "acc": 81.75, "build": 0, "run": 2600,
                 "c10r": 26000, "breakeven": 4},
        "pal": {"pass": 58, "ccr": 72.50, "ci": [61.86, 81.08], "dccr": -17.50,
                "dccr_ci": [-26.25, -10.00], "acc": 86.75, "build": 0, "run": 4800,
                "c10r": 48000, "breakeven": 2},
        "ncnlp": {"pass": 72, "ccr": 90.00, "ci": [81.49, 94.85], "acc": 92.00,
                  "build": 6000, "run": 800, "c10r": 14000},
    },
    "rq2_consistency": {"ncnlp": 99.00, "baseline_range": [76.25, 86.25]},
    "rq2_p_threshold": 0.001,
    "rq3": {
        "full": {"pass": 72, "ccr": 90.00, "ci": [81.49, 94.85]},
        "model_assign": {"pass": 64, "ccr": 80.00, "ci": [69.95, 87.30], "dc": 1000, "dr": 50},
        "fused": {"pass": 54, "ccr": 67.50, "ci": [56.64, 76.76], "dc": -1500, "dr": 200},
        "no_check": {"pass": 68, "ccr": 85.00, "ci": [75.59, 91.21], "dc": -1000, "dr": -40},
        "model_code": {"pass": 62, "ccr": 77.50, "ci": [67.21, 85.27], "dc": 500, "dr": 100},
        "rule_recover": {"pass": 40, "ccr": 50.00, "ci": [39.30, 60.70], "dc": -4500, "dr": 100},
        "model_exec": {"pass": 46, "ccr": 57.50, "ci": [46.57, 67.74], "dc": -5000, "dr": 2000},
    },
    "rq3_contrasts": {
        "Model-assign - Full": (-10.00, 0.023, True),
        "Fused - Model-assign": (-12.50, 0.010, True),
        "Fused - Full": (-22.50, 0.001, True),
        "No-check - Full": (-5.00, 0.250, False),
        "Model-code - Full": (-12.50, 0.010, True),
        "Rule-recover - Full": (-40.00, 0.001, True),
        "Model-exec - Full": (-32.50, 0.001, True),
        "Model-exec - Direct LLM": (5.00, 0.250, False),
    },
    "rq3_precheck": {
        "full": (93.75, 92.50, 90.00, 12.50, 0.150),
        "model_assign": (93.75, 82.50, 80.00, 20.00, 0.250),
        "fused": (81.25, 72.50, 70.00, 25.00, 0.350),
    },
}


class Checker:
    def __init__(self):
        self.rows = []
        self.failures = 0

    def check(self, name, reported, recomputed, tol=0.0, mode="exact"):
        if mode == "exact":
            ok = reported == recomputed
        elif mode == "abs":
            ok = recomputed is not None and abs(recomputed - reported) <= tol
        elif mode == "ci":
            ok = (recomputed is not None
                  and abs(recomputed[0] - reported[0]) <= tol
                  and abs(recomputed[1] - reported[1]) <= tol)
        elif mode == "lt":
            ok = recomputed is not None and recomputed < reported
        else:
            ok = False
        if not ok:
            self.failures += 1
        self.rows.append({"quantity": name, "reported": reported,
                          "recomputed": recomputed,
                          "status": "match" if ok else "DEVIATION"})

    def summary(self):
        return {"pass": self.failures == 0, "checks": len(self.rows),
                "deviations": self.failures, "details": self.rows}


def main():
    r1 = json.loads((OUT / "rq1_results.json").read_text(encoding="utf-8"))
    r2 = json.loads((OUT / "rq2_results.json").read_text(encoding="utf-8"))
    r3 = json.loads((OUT / "rq3_results.json").read_text(encoding="utf-8"))
    ck = Checker()

    # RQ1 stage rows
    for row, (k, n, ci) in zip(r1["stage_rows"], MS["rq1_stage"]):
        ck.check(f"RQ1 stage count: {row['metric']}", k, row["count"])
        ck.check(f"RQ1 stage Wilson CI: {row['metric']}", ci, row["wilson_ci"], CI_TOL, "ci")
    ck.check("RQ1 replay consistency among 76", 76,
             r1["replay_consistent_among_compiled_behavior"])
    # RQ1 fault table
    for row in r1["fault_table"]:
        ms = MS["rq1_faults"][row["label"]]
        ck.check(f"RQ1 silent: {row['label']}", ms[0], row["silent"])
        ck.check(f"RQ1 repaired: {row['label']}", ms[1], row["repaired"])
        ck.check(f"RQ1 refused: {row['label']}", ms[2], row["refused"])
        ck.check(f"RQ1 contained: {row['label']}", ms[3], row["contained"])
        ck.check(f"RQ1 escaped: {row['label']}", ms[4], row["escaped"])
    for key, val in MS["rq1_totals"].items():
        ck.check(f"RQ1 total {key}", val, r1["totals"][key])
    ck.check("RQ1 HFER %", MS["rq1_hfer"][0], r1["hfer_pct"], 0.011, "abs")
    ck.check("RQ1 HFER CI", MS["rq1_hfer"][1], r1["hfer_ci"], BOOT_TOL, "ci")
    ck.check("RQ1 FPR %", MS["rq1_fpr"][0], r1["fpr_pct"], 0.011, "abs")
    ck.check("RQ1 FPR CI", MS["rq1_fpr"][1], r1["fpr_ci"], BOOT_TOL, "ci")
    ck.check("RQ1 natural error instances", MS["rq1_natural"][0], r1["natural_errors"]["instances"])
    ck.check("RQ1 natural errors reaching user", MS["rq1_natural"][1],
             r1["natural_errors"]["reached_erroneous_user_result"])

    # RQ2
    for sid, ms in MS["rq2"].items():
        s = r2["systems"][sid]
        ck.check(f"RQ2 pass: {sid}", ms["pass"], s["pass"])
        ck.check(f"RQ2 CCR: {sid}", ms["ccr"], s["ccr_pct"], 0.011, "abs")
        ck.check(f"RQ2 Wilson CI: {sid}", ms["ci"], s["wilson_ci"], CI_TOL, "ci")
        ck.check(f"RQ2 single-run accuracy: {sid}", ms["acc"], s["single_run_accuracy_pct"],
                 0.011, "abs")
        ck.check(f"RQ2 build tokens: {sid}", ms["build"], s["build_tokens_per_workflow"])
        ck.check(f"RQ2 run tokens: {sid}", ms["run"], s["run_tokens_per_workflow"])
        ck.check(f"RQ2 C+10R: {sid}", ms["c10r"], s["c_plus_10r"])
        if sid != "ncnlp":
            ck.check(f"RQ2 dCCR: {sid}", ms["dccr"], s["dccr_pp"], 0.011, "abs")
            ck.check(f"RQ2 dCCR CI: {sid}", ms["dccr_ci"], s["dccr_ci"], BOOT_TOL, "ci")
            ck.check(f"RQ2 breakeven: {sid}", ms["breakeven"], s["breakeven"])
            ck.check(f"RQ2 Holm p < 0.001: {sid}", True,
                     s["mcnemar"]["holm_p"] < MS["rq2_p_threshold"], mode="exact")
    ck.check("RQ2 consistency ncnlp", MS["rq2_consistency"]["ncnlp"],
             r2["systems"]["ncnlp"]["pairwise_consistency_pct"], 0.011, "abs")
    for sid in ("direct_llm", "dspy", "lmql", "pal"):
        v = r2["systems"][sid]["pairwise_consistency_pct"]
        lo, hi = MS["rq2_consistency"]["baseline_range"]
        ck.check(f"RQ2 consistency within reported range: {sid}", True, lo <= v <= hi,
                 mode="exact")

    # RQ3 configs
    key_map = {"full": "full", "model_assign": "model_assign", "fused": "fused",
               "no_check": "no_check", "model_code": "model_code",
               "rule_recover": "rule_recover", "model_exec": "model_exec"}
    for key, ms in MS["rq3"].items():
        c = r3["configs"][key_map[key]]
        ck.check(f"RQ3 pass: {key}", ms["pass"], c["pass"])
        ck.check(f"RQ3 CCR: {key}", ms["ccr"], c["ccr_pct"], 0.011, "abs")
        ck.check(f"RQ3 Wilson CI: {key}", ms["ci"], c["wilson_ci"], CI_TOL, "ci")
        if "dc" in ms:
            ck.check(f"RQ3 dC: {key}", ms["dc"], c["dc_tokens"])
            ck.check(f"RQ3 dR: {key}", ms["dr"], c["dr_tokens"])
    # RQ3 contrasts
    for row in r3["contrasts"]:
        ms = MS["rq3_contrasts"][row["comparison"]]
        ck.check(f"RQ3 dCCR: {row['comparison']}", ms[0], row["dccr_pp"], 0.011, "abs")
        sig_recomputed = row["holm_p"] < 0.05
        ck.check(f"RQ3 significance: {row['comparison']}", ms[2], sig_recomputed, mode="exact")
    # RQ3 precheck
    for row in r3["precheck"]:
        ms = MS["rq3_precheck"][row["config"]]
        ck.check(f"RQ3 precheck facts EM: {row['config']}", ms[0], row["facts_em_pct"],
                 0.011, "abs")
        ck.check(f"RQ3 precheck assignment EM: {row['config']}", ms[1], row["assignment_em_pct"],
                 0.011, "abs")
        ck.check(f"RQ3 precheck relations EM: {row['config']}", ms[2], row["relations_em_pct"],
                 0.011, "abs")
        ck.check(f"RQ3 precheck intervention: {row['config']}", ms[3], row["intervention_pct"],
                 0.011, "abs")
        ck.check(f"RQ3 precheck repairs/WF: {row['config']}", ms[4], row["repairs_per_workflow"],
                 0.0011, "abs")

    report = ck.summary()
    (OUT / "reported_vs_recomputed.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pass", "checks", "deviations")}, indent=2))
    for row in report["details"]:
        if row["status"] != "match":
            print("DEVIATION:", row["quantity"], "reported=", row["reported"],
                  "recomputed=", row["recomputed"])
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
