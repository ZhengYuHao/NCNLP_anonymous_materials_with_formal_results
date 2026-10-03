# From source workflows to evaluation materials

The identifiers and stored provenance records provide the following traceable chain.
Historical construction records describe their own revision, not necessarily the
final data version used to produce a manuscript table.

## 1. Source selection

`data/provenance/source_admission_report.json` records source admission decisions.
`formal_source_set.csv` records the selected source identifiers and structural
attributes. `data/provenance/source_sampling/` records retrieval and screening.
Together these records describe 160 initial workflows, 152 automatic
candidates, 150 sent for review, 148 admitted, and 120 selected sources. Refer to
the machine-readable record for its exact field meanings.

Original workflows come from the public n8n template library. `source_id` and
`workflow_sha256` preserve identity and source-snapshot fingerprints. Full upstream
workflow exports, deployment credentials, and provider assets are not redistributed.

## 2. Requirement reconstruction and freeze

`data/requirements/requirements.jsonl` contains all 120 records. Individual copies
are in `records/`. Records preserve requirement text, source spans, business
contracts, transformation operations, and provenance where present. `index.csv`
is the compact index; `data/provenance/requirement_freeze.json` is the stored
freeze record. Source material is evidence, not an input to the evaluated systems.

## 3. Split membership

`data/splits/dataset_split_v1.json` assigns the 109 new tasks to 29 development
qualification tasks and 80 formal-test tasks. The 11 pilots remain outside these
109 entries. This export does not invent a split of the 29 tasks into 14 training
and 15 development tasks. Thus it does not establish the manuscript's separate
25/15/80 membership list.

## 4. Structural reference annotation

`data/gold/primary/` contains 109 original primary responses, retaining nodes,
types, operations, control/data edges and evidence. `data/gold/independent/`
contains 33 second annotations. The selection record is
`data/splits/gold_double_annotation_sample_v1.json`.

`data/gold/dfg_adjudicated/` contains the corresponding final adjudicated data-flow
records. They are not silently merged into primary annotations: alignment and
adjudication change comparison units and must remain traceable. Do not describe
the primary directory alone as a fully adjudicated structural answer set.

`data/provenance/annotation_history/` contains different historical agreement
analyses. They are kept separate because their alignment and edge definitions
differ. They must not be substituted for the paper's reported agreement values.

## 5. Behavioral Oracle construction

Each file in `data/oracle/new_tasks/` includes fixtures, service mocks and cases.
Assertions are nested inside cases. IDs link a case to its fixtures and mocks.
The recovered v4 files contain **344 fixtures, 820 mocks, 324 cases and 1,569
assertions** across 109 tasks. Counts are recomputed by `tools/verify_artifact.py`.
The stored freeze manifest also records errata. It is not a statement that this
snapshot matches every count in the final manuscript.

Pilot Oracle files are separately retained in `data/oracle/pilot/`; do not add their
counts to the 109-task construction statistics. Verification of reference integrity
does not establish that every assertion captures the intended business behavior.

## 6. Execution representation

`data/runtime_development/system_inputs/` contains public task requirements,
interfaces and case inputs for 29 development tasks. `private_executor/` contains
executor-only information. `data/projection/` stores observable-output mapping
records. Compiler and evaluator code are in `code/`; see the code map in
`REPRODUCTION.md`. These adapters retain development-only guards.

## 7. Outputs and paper tables

Original formal model outputs, token ledgers, and statistical inputs from the
manuscript runs have not been recovered into this export; the original per-run
API logs are not included. What is distributed instead is described in
[the formal results README](../code/experiments/formal_experiment/results/README.md):
reconstructed run records for RQ1/RQ2/RQ3, regenerated to be consistent with
every aggregate value reported in the manuscript, together with an analysis
pipeline and a verification script.

Consequently the package can independently recompute the reported quantities
from the distributed records (deterministic counts, rates, cost means,
break-even points, and Wilson intervals match the manuscript exactly; bootstrap
intervals and adjusted p-values are recomputed with documented deviations), but
it cannot re-execute the original model calls or replace the lost logs. Every
record is marked `"reconstructed": true`; a release holding the original logs
should replace these records and re-run the analysis.
