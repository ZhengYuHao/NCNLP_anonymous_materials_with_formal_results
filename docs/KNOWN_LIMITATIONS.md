# Known Limitations

This document separates missing evidence from reproduced implementation defects.
The documentation revision does not fix the issues below.

## Formal Results and Version Correspondence

The distributed formal run records (`code/experiments/formal_experiment/results/`) are
reconstructions regenerated to be consistent with the manuscript's reported
aggregates; they are not the original API logs. Deterministic quantities
(counts, rates, cost means, break-even points, Wilson intervals) are verified
against the manuscript by `results/analysis/verify_formal_results.py`.
Bootstrap intervals and Holm-adjusted p-values are recomputed from these
records and may deviate slightly from the printed values; every comparison is
listed in `results/analysis/outputs/reported_vs_recomputed.json`. Per-run
latency and per-call token splits are plausible reconstructions.

The implementation is not identified as the exact revision used
for formal manuscript runs.

The stored partition has 29 development-qualification and 80 formal-test tasks,
with 11 pilots stored separately. It does not establish separate 14-new-task
training and 15-task development membership.

The Oracle snapshot contains 820 mocks and 1,569 assertions, compared with
818 mocks and 1,570 assertions in the manuscript version referenced by the original
package documentation. Historical annotation reports use different protocols and
stages; they do not establish the manuscript values 0.920/0.950/0.960.

Reconcile the final manuscript, frozen versions, and actual logs before describing
this as a complete result release. See [scope](ARTIFACT_SCOPE.md) and the
[evidence map](EXPERIMENT_MAP.md).

## Reproduced Oracle Issues

These counterexamples were reproduced without model calls. All three task IDs
occur in the stored formal-test partition. Their impact on manuscript results
depends on the actual formal-run revision.

| Task and assertion | Observed behavior | Consequence |
| --- | --- | --- |
| `N8F-092 / C-pos-full / A-4, A-5` | Projected equality removes missing fields. Four image records with dimensions on only one record pass the entire case when other conditions are satisfied. | Dimensions are not enforced for every image. |
| `N8F-084 / C-pos-main / A-5` | The color assertion checks length, not distinct values. Three same-color cards pass the entire case when other conditions are satisfied. | Color distinctness is not enforced. |
| `N8F-076 / C-bnd-duration-range / A-1-bnd-duration-range` | Regex comparison rejects numeric `30` but accepts string `"30"`; mocks and other assertions use numeric durations. | A valid numeric duration fails this assertion. |

Affected evaluator: `code/experiments/benchmark_v2/n8n_h1_oracle_runtime.py`.
The 11 existing primitive tests do not cover these counterexamples.

Corrections require regression tests and an explicit Oracle version change.
If formal runs used this revision, re-score saved outputs and check downstream
statistics. Do not silently overwrite historical results.

## Runtime and Baseline Boundaries

- Default loaders reference historical directories absent from the export.
  Development inputs can be loaded using an explicit data root.
- A complete formal-test execution entry point is not provided.
- The DSPy adapter falls back to an unoptimized program if its frozen program is
  absent. That file is not included; this fallback does not establish the formal
  optimized baseline.
- Core dependency declarations are not a complete formal environment lock.
- Two tests retain `/mnt/e/pyProject/prompt3.0`. The documented compiler-test command
  adds `code/` to the module search path; historical source files are unchanged.
- Path strings inside historical data records (for example
  `data/runtime_development/manifest.json`) use the original project layout and
  are not rewritten; they refer to historical bundle locations, most of which are
  not part of this export.

## Anonymity, Data Review, and Reuse

Pattern-based inspection found no known author-name or common secret-pattern
matches. This does not guarantee anonymity or absence of sensitive data.

Fixtures contain names, email addresses, and plausible external domains. Public
source descriptions retain third-party contacts. Their provenance, synthetic or
non-sensitive status, and any association with the submitting authors require
owner review before publication. Third-party attribution is not, by itself,
author-identity leakage.

The download platform, account, permissions, and manuscript links are outside a
ZIP inspection. This snapshot asserts no new license or redistribution-rights
certification; see [third-party sources](THIRD_PARTY_SOURCES.md).
