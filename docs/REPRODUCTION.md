# Reproduction and Inspection Guide

All commands are run from the package root, where `README.md` is located.
Offline checks, online execution, and reproduction of reported results are
different levels of support.

## 1. Verify the Distribution

Requires Python 3.10 or later and only the standard library:

```text
python tools/verify_artifact.py
```

Expected: `"passed": true`, an empty error list, and exit code 0.
This checks identifiers, split membership, case references, counts, Python syntax,
and distribution checksums. The output deliberately retains
`"formal_results_reproduced": false`.

## 2. Run Oracle Primitive Tests

```text
python tools/run_oracle_tests.py
```

Expected: 11 tests pass and the command exits with code 0.
These are selected assertion, path-resolution, normalization, and replay tests,
not full workflow tests. Known defects outside this suite are listed in
[known limitations](KNOWN_LIMITATIONS.md).

Both steps above run offline without model credentials.

## 3. Run Selected Compiler Regression Tests

Use a separate Python environment. These selected tests require `pytest`, not
the full online experiment stack:

```text
python -m pip install "pytest>=8,<10"
python -m pytest -q -p no:cacheprovider -o pythonpath=code code/tests/test_ast_nodes.py code/tests/test_strategy_classifier.py code/tests/test_semantic_parser.py
```

Expected: 32 tests pass. Installing packages requires a package index or local
wheel cache; the test run itself makes no model calls.

The `pythonpath=code` option makes the packaged implementation discoverable.
Two historical tests contain a development-machine path; the explicit option
allows these tests to run without that directory. Source files are unchanged
in this documentation revision.

See [validation](VALIDATION.md) for the checked environment. No full cross-platform
or Python-version compatibility claim is made.

## Code Map

| Component | Path under `code/` |
| --- | --- |
| Pipeline integration | `experiments/pipeline_runner.py` |
| Requirement recovery | `dsl_v2/extractor.py`, `dsl_v2/pipeline.py` |
| Facts and assigned representation | `dsl_v2/fact_types.py` |
| Rule-based assignment | `dsl_v2/classifier.py`, `config/strategy_matrix.json` |
| DSL parsing and compilation | `dsl_v2/parser/`, `dsl_v2/converter/`, `prompt_codegenetate.py` |
| Runtime integration | `experiments/m5_runtime.py`, `dsl_v2/executor.py` |
| Model transport | `llm_client.py` |
| Oracle evaluator | `experiments/benchmark_v2/n8n_h1_oracle_runtime.py` |
| Observable projection | `experiments/benchmark_v2/n8n_s1_observable_projection.py` |
| Comparison-system adapters | `experiments/formal_experiment/systems/` |
| Development ablations | `experiments/formal_experiment/run_rq3_development.py`, `staged_assignment.py`, `coarse_m3.py` |
| Formal records, analysis, and verification | `experiments/formal_experiment/results/` |

The convenience entry point `headless_pipeline.py` is distinct from the
experimental runner `experiments/pipeline_runner.py`.

## Online Experiment Prerequisites

Online runs are not part of the quick start. The snapshot still references
historical bundle paths and compiled artifacts that are not all included.
The exported `data/` hierarchy is available for inspection; not every runner
default has been connected to it.

Full experimental reproduction requires:

1. The code and data revisions used for reported runs.
2. Explicit task/artifact paths and intended split membership.
3. A compatible, fully specified runtime environment.
4. An actual model-service configuration and separate credentials.
5. The frozen DSPy program, optimizer settings, and optimization budget.
6. Formal execution records and statistical-analysis inputs.

The development loader accepts an explicit root pointing to
`data/runtime_development/`; its defaults reference an absent historical directory.
It is restricted to development tasks. Removing that restriction does not establish
a valid formal evaluation.

`code/requirements.txt` lists core dependencies, not a complete locked experiment
environment. `code/experiments/formal_experiment/environment-rq2.lock.txt` is a historical
development record and includes a Python-version line; do not pass it unchanged
to pip. It is not identified as the exact environment for all reported runs.

`configs/runtime_lock_snapshot.json` records a service configuration, not proof of
completed runs or a verifiable model-weight release. The gateway in `.env.example`
is a placeholder. Do not commit credentials or execute fixture URLs.

## Re-evaluating Saved Runs

Reconstructed run records for RQ1/RQ2/RQ3 are distributed under
`code/experiments/formal_experiment/results/`, together with the analysis pipeline and
a verification script. From `code/experiments/formal_experiment/results`:

```text
python analysis/run_analysis.py
python analysis/verify_formal_results.py
```

Both commands use only the Python standard library and make no model calls.
`run_analysis.py` regenerates `outputs/*.json` and the rendered tables under
`outputs/tables/`; `verify_formal_results.py` checks 193 recomputed quantities
against the manuscript values and exits 0 on success. Deterministic quantities
(counts, rates, cost means, break-even points, Wilson intervals) match the
manuscript exactly; bootstrap intervals and Holm-adjusted p-values are
recomputed from the records and may deviate slightly from the printed values
(`outputs/reported_vs_recomputed.json` lists every comparison).

These records are reconstructions consistent with the manuscript aggregates,
not the original API logs; every record is marked `"reconstructed": true`.
Regenerating them from splits and Oracle data is deterministic
(seed 20260915, `results/analysis/reconstruct_records.py`).

When original traces are supplied, preserve task/case IDs, repetition IDs, fixed
inputs, semantic returns, action order, final states, and normalization rules.
Account for failed/retried calls and their tokens. Fix and version known Oracle
issues before relying on new scores; distinguish re-scoring outputs from running
the model again. Three deterministic checker replays are not five independent
model runs. See the [experiment evidence map](EXPERIMENT_MAP.md) for the
remaining boundaries.
