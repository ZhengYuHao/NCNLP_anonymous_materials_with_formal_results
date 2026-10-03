"""Offline integrity checks, not reproduction of reported experiment outcomes."""
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def main():
    errors = []
    requirements = [json.loads(s) for s in (ROOT / "data/requirements/requirements.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    ids = [r["record_id"] for r in requirements]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate requirement IDs")
    split = read(ROOT / "data/splits/dataset_split_v1.json")["records"]
    new_ids = {i for i in ids if i.startswith("N8F-")}
    if {r["task_id"] for r in split} != new_ids or len(split) != len(new_ids):
        errors.append("Split membership does not cover the new tasks exactly once")
    oracles = [read(p) for p in (ROOT / "data/oracle/new_tasks").glob("*.json")]
    if {o["task_id"] for o in oracles} != new_ids:
        errors.append("Oracle IDs do not match new-task requirements")
    for o in oracles:
        fixtures = {f["fixture_id"] for f in o["fixtures"]}
        mocks = {m["mock_id"] for m in o["mocks"]}
        for case in o["cases"]:
            if not set(case["fixture_ids"]) <= fixtures:
                errors.append(o["task_id"] + ": missing fixture")
            if not set(case["mock_ids"]) <= mocks:
                errors.append(o["task_id"] + ": missing mock")
            if not case.get("assertions"):
                errors.append(o["task_id"] + ": empty assertions")
    counts = {"requirements": len(ids), "oracle_tasks": len(oracles),
              "fixtures": sum(len(o["fixtures"]) for o in oracles),
              "mocks": sum(len(o["mocks"]) for o in oracles),
              "cases": sum(len(o["cases"]) for o in oracles),
              "assertions": sum(len(c["assertions"]) for o in oracles for c in o["cases"]),
              "split": dict(Counter(r["dataset_partition"] for r in split))}
    expected = read(ROOT / "artifact_status.json")["counts"]
    for key, value in counts.items():
        if expected[key] != value:
            errors.append("Count mismatch: " + key)
    for folder in ("primary", "independent", "dfg_adjudicated"):
        docs = [read(p) for p in (ROOT / "data/gold" / folder).glob("*.json")]
        gold_ids = [d["task_id"] for d in docs]
        if len(gold_ids) != len(set(gold_ids)) or not set(gold_ids) <= new_ids:
            errors.append("Invalid gold membership: " + folder)
        if folder == "primary" and set(gold_ids) != new_ids:
            errors.append("Missing primary annotations")
    for p in (ROOT / "code").rglob("*.py"):
        try:
            ast.parse(p.read_text(encoding="utf-8-sig"))
        except SyntaxError as exc:
            errors.append(str(p.relative_to(ROOT)) + ": " + str(exc))
    manifest = ROOT / "MANIFEST.sha256"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected_hash, rel = line.split("  ", 1)
            p = ROOT / rel
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected_hash:
                errors.append("Checksum mismatch: " + rel)
    else:
        errors.append("Distribution checksum manifest missing")
    print(json.dumps({"passed": not errors, "counts": counts, "errors": errors,
                      "formal_results_reproduced": False}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
