#!/usr/bin/env python3
"""Validate the immutable S1 120-requirement freeze snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
TARGET = BASE / "n8n_native_s1" / "source_review_round_v2" / "d10_requirement_freeze_v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def digest_value(value) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))


def main() -> int:
    manifest = read_json(TARGET / "freeze_manifest.json")
    errors = []
    manifest_copy = dict(manifest)
    recorded_manifest_hash = manifest_copy.pop("manifest_sha256", None)
    if digest_value(manifest_copy) != recorded_manifest_hash:
        errors.append("freeze manifest hash mismatch")
    if digest_file(TARGET / "canonical_requirement_set.jsonl") != manifest["canonical_requirement_set_sha256"]:
        errors.append("canonical JSONL hash mismatch")
    if digest_file(TARGET / "canonical_requirement_index.csv") != manifest["canonical_requirement_index_sha256"]:
        errors.append("canonical CSV hash mismatch")
    ids = set()
    sources = set()
    roles = {"train_dev_only": 0, "formal_evaluation_pool": 0}
    origins = {}
    for item in manifest["records"]:
        path = TARGET / "requirements" / f"{item['record_id']}.json"
        if not path.exists():
            errors.append(f"{item['record_id']}: missing frozen record")
            continue
        if digest_file(path) != item["record_sha256"]:
            errors.append(f"{item['record_id']}: record hash mismatch")
        record = read_json(path)
        artifact = record["requirement"]
        if digest_value(artifact) != record["requirement_sha256"] or record["requirement_sha256"] != item["requirement_sha256"]:
            errors.append(f"{item['record_id']}: requirement hash mismatch")
        if record["record_id"] in ids:
            errors.append(f"{item['record_id']}: duplicate record ID")
        if record["source_id"] in sources:
            errors.append(f"{item['record_id']}: duplicate source ID")
        ids.add(record["record_id"])
        sources.add(record["source_id"])
        roles[record["dataset_role"]] = roles.get(record["dataset_role"], 0) + 1
        origins[record["version_origin"]] = origins.get(record["version_origin"], 0) + 1
    if len(ids) != 120 or manifest.get("total_requirement_count") != 120:
        errors.append("freeze must contain 120 requirements")
    if roles != {"train_dev_only": 11, "formal_evaluation_pool": 109}:
        errors.append(f"dataset role counts mismatch: {roles}")
    if origins != manifest["version_origin_counts"]:
        errors.append(f"version origin counts mismatch: {origins}")
    if manifest.get("unresolved_count") != 0 or manifest.get("excluded_count") != 0:
        errors.append("freeze contains unresolved or excluded requirements")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "checked_requirements": len(ids),
        "dataset_roles": roles,
        "version_origins": origins,
        "errors": errors,
    }
    reports = TARGET / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "freeze_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
