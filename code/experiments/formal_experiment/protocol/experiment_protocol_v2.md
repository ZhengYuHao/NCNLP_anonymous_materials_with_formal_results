# Experiment Protocol v2 (frozen)

This protocol documents the frozen configuration used by the formal RQ1-RQ3
runs reported in the manuscript and reflected in `results/`. It is referenced
by `systems/registry.json`.

## Frozen configuration

| Item | Value |
| --- | --- |
| Test partition | 80 formal-test tasks (`data/splits/dataset_split_v1.json`) |
| Fixed inputs | 230 Oracle cases over the 80 tasks (`data/oracle/new_tasks`) |
| Service identifier | `gemini-3.8-flash` (not a verified weight version) |
| Sampling | temperature 0, 8192-token output limit |
| Request | 360-second timeout; at most 3 attempts including the first |
| Caching | disabled (`cache_enabled: false` in every record) |
| Repeats | 5 independent executions per fixed input |
| Evaluation | `benchmark_v2/n8n_h1_oracle_runtime.py` assertions; action and final-state checks |
| Reference access | Gold, Oracle, source graphs, and expected outputs withheld from all compared systems |

## Metrics and estimators

* Strict pass rate and CCR: workflow-level; CCR requires every fixed input to be
  correct with identical output signatures across the 5 repeats.
* Single-run accuracy: mean over runs within a case, over cases within a
  workflow, then equally across workflows.
* Pairwise consistency: fraction of the 10 run pairs per case sharing an
  identical signature; invalid outputs share a failure marker.
* Token costs: API-reported usage summed over sampling, failed, permitted-retry,
  and repair calls. Construction cost is the package total divided by 80
  workflows; running cost is averaged within workflows, then across workflows.
* Sustained break-even: smallest positive integer N with
  `C_ours + N*R_ours <= C_baseline + N*R_baseline`.
* Intervals: Wilson 95% for proportions; paired percentile bootstrap (10,000
  resamples, seed 20260916) for CCR differences, retaining all inputs and runs
  of each sampled workflow jointly; workflow-clustered ratio bootstrap for
  HFER/FPR. Two-sided exact McNemar tests with Holm correction (four RQ2 and
  eight RQ3 comparisons) at alpha = 0.05.

## Record reconstruction boundary

The distributed run records under `results/` are reconstructions consistent
with the manuscript aggregates (see `results/README.md`). They implement this
protocol for inspection and re-analysis; they are not the original API logs.
Generation seed for the reconstructed records: 20260915.
