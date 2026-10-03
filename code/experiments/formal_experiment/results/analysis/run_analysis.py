"""Run the RQ1-RQ3 analyses and render the manuscript tables from the records."""

from __future__ import annotations

import json
from pathlib import Path

import compute_rq1
import compute_rq2
import compute_rq3

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
TABLES = OUT / "tables"


def fmt(v, nd=2):
    return f"{v:.{nd}f}"


def render_rq1(r):
    rows = []
    for row in r["stage_rows"]:
        lo, hi = row["wilson_ci"]
        rows.append(f"{row['stage']} & {row['metric']} & {row['count']}/{row['total']} & "
                    f"{fmt(row['rate_pct'])} & [{fmt(lo)}, {fmt(hi)}] \\\\")
    rows.append("\\midrule")
    rows.append(f"Checking & HFER & {r['totals']['escaped']}/{r['totals']['silent']} & "
                f"{fmt(r['hfer_pct'])} & [{fmt(r['hfer_ci'][0])}, {fmt(r['hfer_ci'][1])}] \\\\")
    rows.append(f"Checking & FPR & {r['fpr_count']}/{r['fpr_total']} & "
                f"{fmt(r['fpr_pct'])} & [{fmt(r['fpr_ci'][0])}, {fmt(r['fpr_ci'][1])}] \\\\")
    body = "\n".join(rows)
    tex = f"""% RQ1 main table (recomputed from results/rq1 records)
\\begin{{table}}[!htbp]
\\centering
\\caption{{RQ1: stage conversion correctness and error containment after checking (recomputed).}}
\\label{{tab:rq1-main}}
\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}lllrl@{{}}}}
\\toprule
Stage & Metric & Count & Rate (\\%) & 95\\% CI (\\%) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""
    (TABLES / "rq1_main.tex").write_text(tex, encoding="utf-8")

    rows = []
    for row in r["fault_table"]:
        rows.append(f"{row['group']} & {row['label']} & {row['silent']} & {row['repaired']} & "
                    f"{row['refused']} & {row['contained']} & {row['escaped']} \\\\")
    t = r["totals"]
    rows.append("\\midrule")
    rows.append(f"Total & --- & {t['silent']} & {t['repaired']} & {t['refused']} & "
                f"{t['contained']} & {t['escaped']} \\\\")
    body = "\n".join(rows)
    tex = f"""% RQ1 fault containment (recomputed from results/rq1 records)
\\begin{{table}}[!htbp]
\\centering
\\caption{{RQ1: outcomes of the confirmed silent harmful faults after checking is enabled (recomputed).}}
\\label{{tab:fault-containment}}
\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}llrrrrr@{{}}}}
\\toprule
Artifact & Fault & Silent & Repaired & Refused & Contained & Escaped \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""
    (TABLES / "rq1_fault_containment.tex").write_text(tex, encoding="utf-8")


def render_rq2(r):
    systems = r["systems"]
    rows = []
    for sid in ("direct_llm", "dspy", "lmql", "pal", "ncnlp"):
        s = systems[sid]
        lo, hi = s["wilson_ci"]
        dlo, dhi = s["dccr_ci"]
        p = s.get("mcnemar", {}).get("holm_p")
        if sid == "ncnlp":
            delta, ptxt = "Reference", "Reference"
        else:
            delta = f"{fmt(s['dccr_pp'])} [{fmt(dlo)}, {fmt(dhi)}]"
            ptxt = "$<$0.001" if p is not None and p < 0.001 else fmt(p, 3)
        rows.append(f"{s['display']} & {s['pass']}/{s['total']} & {fmt(s['ccr_pct'])}\\% "
                    f"[{fmt(lo)}, {fmt(hi)}] & {delta} & {ptxt} & "
                    f"{fmt(s['single_run_accuracy_pct'])}\\% \\\\")
    body = "\n".join(rows)
    tex = f"""% RQ2 main table (recomputed from results/rq2 records)
\\begin{{table}}[!htbp]
\\centering
\\caption{{RQ2: reliability on the 80 mixed workflows (recomputed).}}
\\label{{tab:rq2-main}}
\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}lllrl@{{}}}}
\\toprule
System & Pass/total & CCR [95\\% CI] & $\\Delta$CCR [95\\% CI] & Corrected $p$ & Single-run acc. \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""
    (TABLES / "rq2_main.tex").write_text(tex, encoding="utf-8")

    rows = []
    for sid in ("direct_llm", "dspy", "lmql", "pal", "ncnlp"):
        s = systems[sid]
        rows.append(f"{s['display']} & {s['build_tokens_per_workflow']} & "
                    f"{s['run_tokens_per_workflow']} & {s['c_plus_10r']} & "
                    f"{'---' if s['breakeven'] is None else s['breakeven']} \\\\")
    body = "\n".join(rows)
    tex = f"""% RQ2 cost table (recomputed from results/rq2 records)
\\begin{{table}}[!htbp]
\\centering
\\caption{{RQ2: mean token costs per workflow and sustained break-even executions (recomputed); --- denotes not applicable.}}
\\label{{tab:rq2-cost}}
\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}lrrrr@{{}}}}
\\toprule
System & Build $C$ & Run $R$ & $C+10R$ & Break-even $N_b^*$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""
    (TABLES / "rq2_cost.tex").write_text(tex, encoding="utf-8")


def render_rq3(r):
    configs = r["configs"]
    order = [("full", "Full"), ("model_assign", "Model-assign"), ("fused", "Fused"),
             ("no_check", "No-check"), ("model_code", "Model-code"),
             ("rule_recover", "Rule-recover"), ("model_exec", "Model-exec")]
    rows = []
    for key, label in order:
        c = configs[key]
        lo, hi = c["wilson_ci"]
        dc = "Reference" if key == "full" else f"{c['dc_tokens']:+d}"
        dr = "Reference" if key == "full" else f"{c['dr_tokens']:+d}"
        rows.append(f"{label} & {c['pass']}/{c['total']} & {fmt(c['ccr_pct'])} & "
                    f"[{fmt(lo)}, {fmt(hi)}] & {dc} & {dr} \\\\")
    body = "\n".join(rows)
    tex = f"""% RQ3 main table (recomputed from results/rq2+rq3 records)
\\begin{{table}}[!htbp]
\\centering
\\caption{{RQ3: reliability and cost changes across all configurations (recomputed).}}
\\label{{tab:rq3-main}}
\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}llrlrr@{{}}}}
\\toprule
Configuration & Pass/total & CCR (\\%) & 95\\% CI & $\\Delta C$ & $\\Delta R$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""
    (TABLES / "rq3_main.tex").write_text(tex, encoding="utf-8")

    rows = []
    for row in r["contrasts"]:
        lo, hi = row["ci"]
        p = row["holm_p"]
        ptxt = "$<$0.001" if p < 0.001 else fmt(p, 3)
        rows.append(f"{row['comparison'].replace('-', '$-$')} & {fmt(row['dccr_pp'])} & "
                    f"[{fmt(lo)}, {fmt(hi)}] & {ptxt} & \\\\")
    body = "\n".join(rows)
    tex = f"""% RQ3 contrasts (recomputed from results/rq2+rq3 records)
\\begin{{table}}[!htbp]
\\centering
\\caption{{RQ3: eight paired contrasts (recomputed). Intervals are unadjusted; $p$-values are Holm-corrected.}}
\\label{{tab:rq3-attribution}}
\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}lrll@{{}}}}
\\toprule
Comparison & $\\Delta$CCR (pp) & 95\\% CI & Holm $p$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""
    (TABLES / "rq3_attribution.tex").write_text(tex, encoding="utf-8")

    rows = []
    for row in r["precheck"]:
        rows.append(f"{row['config']} & {fmt(row['facts_em_pct'])} & {fmt(row['assignment_em_pct'])} & "
                    f"{fmt(row['relations_em_pct'])} & {fmt(row['intervention_pct'])} & "
                    f"{row['repairs_per_workflow']:.3f} \\\\")
    body = "\n".join(rows)
    tex = f"""% RQ3 pre-check table (recomputed from results/rq3 records)
\\begin{{table}}[!htbp]
\\centering
\\caption{{RQ3: pre-check exact-match rates and subsequent intervention (recomputed).}}
\\label{{tab:rq3-precheck}}
\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}lrrrrr@{{}}}}
\\toprule
Configuration & Facts EM (\\%) & Assign. EM (\\%) & Relations EM (\\%) & Intervene (\\%) & Repairs/WF \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""
    (TABLES / "rq3_precheck.tex").write_text(tex, encoding="utf-8")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    compute_rq1.main()
    compute_rq2.main()
    compute_rq3.main()
    r1 = json.loads((OUT / "rq1_results.json").read_text(encoding="utf-8"))
    r2 = json.loads((OUT / "rq2_results.json").read_text(encoding="utf-8"))
    r3 = json.loads((OUT / "rq3_results.json").read_text(encoding="utf-8"))
    render_rq1(r1)
    render_rq2(r2)
    render_rq3(r3)
    print("tables written to", TABLES)


if __name__ == "__main__":
    main()
