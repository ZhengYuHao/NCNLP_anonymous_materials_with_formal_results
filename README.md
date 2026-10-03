# NCNLP: Anonymous Review Materials

NCNLP studies the assignment and compilation of code and model responsibilities
in natural-language workflows. This package provides an implementation snapshot
and benchmark materials for inspecting the method, tracing dataset construction,
and running selected offline checks.

**Scope of this release.** Code, requirements, reference annotations, Oracle
specifications, provenance records, and reconstructed formal-experiment records
for RQ1-RQ3 are included. The formal records are reconstructions consistent
with the manuscript's reported aggregates (verified by an included script);
the original per-run API logs are not included. This release therefore supports
inspection, offline validation, and re-analysis of the reported results, but
not re-execution of the original model runs.
See [artifact scope](docs/ARTIFACT_SCOPE.md) and
[known limitations](docs/KNOWN_LIMITATIONS.md).

## Quick Start

Use Python 3.10 or later. From the directory containing this README, run:

```text
python tools/verify_artifact.py
python tools/run_oracle_tests.py
```

These commands use only the Python standard library. They require no credentials,
make no model-service calls, and do not execute fixture URLs.

- The integrity check should report `"passed": true` and exit with code 0.
- The Oracle primitive suite should report 11 passing tests and `OK`.
- Passing checks validate packaging and selected behavior, not empirical claims
  or complete Oracle correctness.

For the 32 selected compiler regression tests and online execution boundaries,
see [reproduction instructions](docs/REPRODUCTION.md).

## Package Organization

```text
NCNLP_anonymous_materials/
|-- README.md
|-- artifact_status.json
|-- MANIFEST.sha256
|-- .env.example
|-- code/                     Implementation and experiment adapters
|-- configs/                  Runtime configuration snapshot
|-- data/
|   |-- requirements/         Frozen requirements and evidence spans
|   |-- gold/                 Primary, independent, and adjudicated references
|   |-- oracle/               Behavioral specifications, fixtures, and mocks
|   |-- runtime_development/  Public inputs and private executor records
|   |-- provenance/           Source selection and annotation history
|   |-- projection/           Observable-output mapping records
|   `-- splits/               Task membership and annotation sampling
|-- docs/                     Scope, evidence map, and reproduction guidance
`-- tools/                    Offline verification entry points
```

Original code and data paths are retained. Raw upstream workflow exports and
model credentials are not included.

## Dataset Overview

| Material | Included in this release |
| --- | --- |
| Frozen requirements | 120 tasks: 11 pilot and 109 new tasks |
| Structural references | 109 primary, 33 independent, 33 adjudicated data-flow records, and 11 pilot references |
| New-task Oracle specifications | 109 tasks; 344 fixtures; 820 mocks; 324 cases; 1,569 assertions |
| Pilot Oracle specifications | 11 tasks, separate from new-task statistics |
| Stored new-task partition | 29 development-qualification and 80 formal-test tasks |
| Development runtime inputs | 29 tasks; public inputs separated from private executor data |

Identifiers connect the records: `N8C-*` denotes pilot tasks and `N8F-*` new tasks.
Original requirement and annotation languages are preserved. The stored partition
does not establish separate training/development membership for a 25/15/80 protocol.

[Data lineage](docs/DATA_LINEAGE.md) explains how these materials are connected.
[Artifact scope](docs/ARTIFACT_SCOPE.md) records version differences and the limits
of comparisons with manuscript statistics.

## Review Guide

| Review objective | Starting point |
| --- | --- |
| Inspect assignment and compilation | [Code map](docs/REPRODUCTION.md#code-map), `code/config/strategy_matrix.json` |
| Trace benchmark construction | [Data lineage](docs/DATA_LINEAGE.md) |
| Locate RQ1, RQ2, and RQ3 materials | [Experiment evidence map](docs/EXPERIMENT_MAP.md) |
| Recompute and check reported results | `code/experiments/formal_experiment/results/README.md` |
| Run available offline checks | [Reproduction instructions](docs/REPRODUCTION.md) |
| Understand tested behavior | [Validation record](docs/VALIDATION.md) |
| Identify missing evidence and known defects | [Known limitations](docs/KNOWN_LIMITATIONS.md) |
| Inspect attribution and reuse boundaries | [Third-party sources](docs/THIRD_PARTY_SOURCES.md) |

## Evaluation Boundaries

Gold annotations, Oracle assertions, expected outputs, source graphs, and private
executor records are evaluation-only. They must not be supplied to evaluated
systems as model inputs. Access to archived test data does not authorize tuning
a new experiment described as held-out on those data.

Online adapters are an implementation snapshot, not a verified one-command
reproduction workflow. Online execution requires a compatible environment,
appropriate data/artifact paths, and separately supplied model-service credentials;
it may incur charges. See the reproduction guide before attempting online runs.

## Integrity, Anonymity, and Reuse

`MANIFEST.sha256` covers distributed files except the manifest itself.
`artifact_status.json` records exported counts and formal-result availability.
Historical hashes inside records retain their original provenance meanings;
they are not interchangeable with distribution-file checksums.

This package is prepared for anonymous review. Git history and private credentials
are not included; the private gateway is a nonfunctional placeholder. Residual
path and data-review limitations are documented in
[known limitations](docs/KNOWN_LIMITATIONS.md). No blanket anonymity or privacy
certification is claimed.

Original workflows and external dependencies remain third-party materials.
This snapshot does not add a software or dataset license. Consult
[third-party sources](docs/THIRD_PARTY_SOURCES.md) for redistribution boundaries.
