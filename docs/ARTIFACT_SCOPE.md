# Scope and version boundaries

This is an available-materials snapshot, not a verified formal-run release.

| Item | Available evidence | Boundary |
| --- | --- | --- |
| Requirements | 120 frozen requirement records | Pilot and new-task versions are distinguished |
| New-task split | 29 development qualification + 80 formal test | Separate training/development membership is not established |
| Reference annotations | 109 primary, 33 independent, 33 adjudicated DFG records | Primary and adjudicated records are not automatically conflated |
| Oracle v4 | 109 tasks, 344 fixtures, 820 mocks, 324 cases, 1,569 assertions | Differs from the manuscript's 818 mocks and 1,570 assertions |
| Annotation agreement | Historical raw/alignment-normalized/reannotation reports | Not verified as the source of manuscript values 0.920/0.950/0.960 |
| Runtime configuration | gemini-3.8-flash service identifier, temperature 0, 8192 tokens, 360 seconds, three attempts, cache off | Configuration file is a historical lock; it explicitly does not attest completion of formal runs |
| Implementation | Recovered compiler, runtime and development comparison adapters | Not identified as the exact code revision of the final reported runs |
| DSPy | Development adapter code | Final formal program, optimizer and optimization budget are unconfirmed; no development artifact is passed off as the final program |
| Formal result records | Reconstructed run records, analysis pipeline and verification under `code/experiments/formal_experiment/results/` | Consistent with manuscript aggregates (checked by `verify_formal_results.py`); not the original API logs; bootstrap intervals and adjusted p-values are recomputed from these records |

The historical annotation reports include distinct definitions and revision stages.
For example, an alignment-normalized diagnostic is not the same measurement as
an original pre-adjudication comparison. Keep report labels when using them.

Credentials were removed from the exported copy only. Replacements of a private
gateway use `https://api.example.com/v1`, which is a nonfunctional placeholder.
Record hashes inside data retain their original provenance meanings;
`MANIFEST.sha256` checks the bytes distributed in this package. Source-to-local
packaging mappings are not included. Two historical tests retain a development
path; the documented test command supplies the exported module path explicitly.

Mock payloads contain example people, organizations, URLs and email strings. Many
use reserved example domains, while some use plausible external domains. These
are retained as test data rather than rewritten, because changing them can alter
domain checks and expected-output matching. Their non-sensitive/synthetic status
requires owner review before public release. Do not contact any address or execute
any external service URL from the test fixtures.

See [known limitations](KNOWN_LIMITATIONS.md) for reproduced Oracle defects and
runtime boundaries. A successful packaging or unit-test check does not establish
that these issues have been resolved.
