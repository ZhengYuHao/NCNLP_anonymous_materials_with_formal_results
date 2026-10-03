# Validation Record

These checks concern the distributed snapshot and its offline entry points.
They do not reproduce formal results, establish all reference answers as correct,
or certify anonymous publication safety.

## Checked Material

| Check | Result |
| --- | --- |
| Requirement membership | 120 unique records; 109 new tasks match the split |
| Requirement copies | JSONL and individual records agree |
| Requirement content hashes | 120 match using the original canonical-JSON definition |
| New-task Oracle membership | 109 files match the requirement IDs |
| New-task Oracle JSON Schema | 109 files pass the supplied Draft 2020-12 schema |
| Case references | Referenced fixtures and mocks exist; all cases have assertions |
| Development input loading | 29 tasks and 94 cases load with an explicit exported root |
| Python source syntax | All 112 implementation source files parse |
| Distribution integrity | All entries in `MANIFEST.sha256` match |

The package verifier does not automatically repeat every independent check above.
Its scope is described in [reproduction](REPRODUCTION.md).

## Documented Test Commands

- Oracle primitive tests: 11 passed without model-service calls.
- Selected AST, assignment-rule, and parser tests: 32 passed with
  `-o pythonpath=code`.
- The historical command without an explicit module path failed during collection
  in a clean exported directory.

The documentation revision is checked on Windows with Python 3.12.14 and
pytest 9.1.1. Schema inspection used jsonschema 4.26.0. The historical Python 3.10
environment and full online dependency stack were not validated.

## Interpretation

Known assertion defects remain; see [known limitations](KNOWN_LIMITATIONS.md).
Passing the primitive suite does not establish correct acceptance or rejection of
every workflow. No model-based comparison, formal fault-injection campaign, cost
reproduction, or statistical re-analysis is claimed.

The legacy `prompt_dslcompiler.py` remains excluded: the original packaging record
identifies a pre-existing syntax error and states that the exported main pipeline
does not import it. Active DSL v2 compiler modules remain included.
No pre-existing implementation file was modified in this documentation revision;
the revision only adds the reconstruction and analysis scripts under
`code/experiments/formal_experiment/results/analysis/` and updates documentation.
