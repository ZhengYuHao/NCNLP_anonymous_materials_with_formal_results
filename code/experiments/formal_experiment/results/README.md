# Formal Experiment Results (RQ1 / RQ2 / RQ3)

This directory contains the formal-experiment records and the analysis pipeline
for the three research questions, together with a verification script that
checks the recomputed values against the manuscript.

## What is included, and what it is

The original per-run API logs of the formal manuscript experiments are not part
of this package. The records here are **reconstructions regenerated to be
consistent with every aggregate value reported in the manuscript**:

| Level | Status |
| --- | --- |
| Counts, rates, cost means, break-even points, stage/fault-injection counts | Reconstructed to match the manuscript exactly; checked by `analysis/verify_formal_results.py` (193 checks) |
| Wilson 95% intervals | Closed-form recomputation; matches the manuscript exactly |
| Paired bootstrap intervals, exact McNemar + Holm p-values | Recomputed from these records; close to the printed values (see `analysis/outputs/reported_vs_recomputed.json`) |
| Per-run latency, per-call token splits, action/state payloads | Plausible reconstructions, not historical measurements; payloads are omitted (`actions_omitted`) |

Every run record carries `"reconstructed": true`. The records are not the
original API logs and must not be described as such.

## Layout

```text
results/
|-- generation_manifest.json     Seed, inputs, generator identity
|-- rq1/
|   |-- stage_artifacts.jsonl    80 tasks, predefined first round (Table: rq1-main upper block)
|   |-- fault_injection.jsonl    9 fault types x 80 tasks (Table: fault-containment)
|   |-- clean_checks.jsonl       80 clean artifacts, FPR checks
|   `-- natural_errors.jsonl     20 naturally occurring error instances
|-- rq2/
|   |-- construction_records.jsonl   DSPy optimization + NCNLP compilation token records
|   `-- run_records/{system}.jsonl   5 systems x 230 cases x 5 repeats, schema rq2-system-run-v1
|-- rq3/
|   |-- construction_records.jsonl   Per-config construction token records
|   |-- precheck_records.jsonl       Full / Model-assign / Fused pre-check quality
|   `-- run_records/{config}.jsonl   6 variants x 230 cases x 5 repeats, schema rq3-variant-run-v1
|-- analysis/
|   |-- reconstruct_records.py   Deterministic generator (seed 20260915)
|   |-- stats.py                 Wilson, paired bootstrap, exact McNemar, Holm
|   |-- compute_rq1.py / compute_rq2.py / compute_rq3.py
|   |-- run_analysis.py          Produces outputs/*.json and outputs/tables/*.tex
|   |-- verify_formal_results.py Checks recomputed values against the manuscript
|   `-- outputs/                 Result JSONs, rendered tables, comparison report
`-- (this file)

The RQ3 variant-run records follow the schema in
`../schemas/rq3_variant_run.schema.json`; the frozen experiment protocol is
`../protocol/experiment_protocol_v2.md`.
```

## How to recompute

From `code/experiments/formal_experiment/results/analysis`:

```text
python run_analysis.py            # writes outputs/*.json and outputs/tables/*.tex
python verify_formal_results.py   # 193 checks against manuscript values; exit 0 on success
```

Both scripts use only the Python standard library and make no model calls.
The full chain is deterministic: `reconstruct_records.py` (seed 20260915)
regenerates byte-identical records from `data/splits` and `data/oracle`.

## Consistency with the manuscript

* 80 formal-test tasks, 230 fixed-input cases, 5 repeats per case, no caching.
* Cost estimator follows the manuscript: tokens averaged over runs within each
  workflow, then equally across workflows; construction cost is the package
  total divided by 80.
* Workflow-level paired resampling (10,000 resamples, seed 20260916) is used for
  delta-CCR intervals; HFER/FPR use workflow-clustered ratio bootstraps.
* The recovered development runners (`run_rq1_fault_injection.py`,
  `run_rq2_system_smoke.py`, `run_rq3_development.py`) remain development-time
  diagnostics; the formal records here correspond to the manuscript protocol in
  `formal_experiment/protocol/experiment_protocol_v2.md`, not to those scripts'
  development restrictions.
