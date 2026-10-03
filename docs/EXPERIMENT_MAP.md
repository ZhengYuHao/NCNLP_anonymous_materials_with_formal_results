# Experiment Evidence Map

This map locates materials relevant to each experimental area. An implementation
entry is not proof that the reported experiment used this revision.

Paths are relative to the package root.

| Area | Included materials | Not included or not established |
| --- | --- | --- |
| Benchmark construction | `data/requirements/`, `data/provenance/`, `data/splits/` | Separate 25/15/80 membership and final manuscript-version correspondence |
| Structural annotation | `data/gold/`, `data/provenance/annotation_history/` | Final inputs and protocol establishing reported agreement values |
| Behavioral reference | `data/oracle/`, `data/projection/` | Verified final-run Oracle revision and resolution of known assertion issues |
| RQ1: intermediate correctness and fault handling | `code/dsl_v2/`, `code/experiments/formal_experiment/results/rq1/`, Gold and Oracle records | Original API logs; the records are reconstructions consistent with the manuscript aggregates |
| RQ2: reliability and repeated-execution cost | `code/experiments/formal_experiment/results/rq2/`, `configs/runtime_lock_snapshot.json`, frozen protocol `code/experiments/formal_experiment/protocol/experiment_protocol_v2.md` | Original API logs and the final optimized DSPy program artifact |
| RQ3: component and boundary ablations | `code/experiments/formal_experiment/results/rq3/`, `code/experiments/formal_experiment/run_rq3_development.py`, `code/experiments/formal_experiment/run_rq3_boundary_development.py`, `code/experiments/formal_experiment/staged_assignment.py`, `code/experiments/formal_experiment/coarse_m3.py` | Original API logs |
| Statistical reporting | `code/experiments/formal_experiment/results/analysis/` (Wilson, paired bootstrap, exact McNemar, Holm; outputs and tables) | Bootstrap intervals and adjusted p-values are recomputed from the reconstructed records and may deviate slightly from printed values; see `outputs/reported_vs_recomputed.json` |

## Availability Labels

- **Included:** distributed in this package.
- **Not included:** absent from this package, with no claim about availability elsewhere.
- **Not established:** correspondence to a reported experiment has not been verified.

Formal run records are distributed under `code/experiments/formal_experiment/results/`.
They are reconstructions regenerated to be consistent with the manuscript's
reported aggregates; deterministic quantities (counts, rates, cost means,
break-even points, Wilson intervals) are verified against the manuscript by
`results/analysis/verify_formal_results.py`. They are not the original API
logs, and per-run payloads (actions, final states, latencies) are omitted.
A release that ships original logs should replace these records and re-run the
analysis; table numbers must refer to the exact submitted manuscript version.

See [artifact scope](ARTIFACT_SCOPE.md) for version boundaries and
[reproduction](REPRODUCTION.md) for available checks.
