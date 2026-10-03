#!/usr/bin/env python3
"""Build and freeze the final 120-requirement S1 set after D9 closure."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parent
ROUND = BASE / "n8n_native_s1" / "source_review_round_v2"
D8 = ROUND / "d8_requirement_round_v1"
D9 = ROUND / "d9_requirement_verification_v1"
TARGET = ROUND / "d10_requirement_freeze_v1"

PILOT_FINAL_GATE = (
    BASE / "n8n_conversion_pilot" / "h1_execution" / "runtime" / "reports"
    / "requirement_final_gate_result_v1_2.json"
)
REVISED = {
    "N8F-006": D9 / "revisions" / "batch_01_revision_01" / "revision_records" / "N8F-006.revision.json",
    "N8F-008": D9 / "revisions" / "batch_01_revision_01" / "revision_records" / "N8F-008.revision.json",
    **{
        task_id: D9 / "revisions" / "remaining_revision_01" / "revision_records" / f"{task_id}.revision.json"
        for task_id in ("N8F-018", "N8F-031", "N8F-047", "N8F-060", "N8F-092", "N8F-100")
    },
}
RECONSIDERED = {"N8F-027", "N8F-030"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def artifact_hash(artifact: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(artifact).encode("utf-8"))


def d9_artifact(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "requirement_text": value["requirement_text"],
        "source_spans": value["source_spans"],
        "transformation_operations": value["transformation_operations"],
        "business_contract": value["business_contract"],
    }


def find_batch(task_id: str, manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for batch in manifest["batches"]:
        for task in batch["tasks"]:
            if task["task_id"] == task_id:
                return batch["batch_id"], task
    raise KeyError(task_id)


def verifier_response(batch: str, task_id: str) -> Path:
    return D9 / "verifier_B" / batch / "responses" / f"{task_id}.response.json"


def revision_confirmation(task_id: str) -> Path:
    if task_id in {"N8F-006", "N8F-008"}:
        return (
            D9 / "revisions" / "batch_01_revision_01" / "verifier_confirmation"
            / "responses" / f"{task_id}.response.json"
        )
    record = read_json(REVISED[task_id])
    verifier = {
        "N8F-018": "RV-03", "N8F-031": "RV-04", "N8F-047": "RV-05",
        "N8F-060": "RV-05", "N8F-092": "RV-08", "N8F-100": "RV-08",
    }[task_id]
    del record
    return (
        D9 / "revisions" / "remaining_revision_01" / "confirmations" / verifier
        / "responses" / f"{task_id}.response.json"
    )


def build_pilot_records() -> list[dict[str, Any]]:
    d8_manifest = read_json(D8 / "round_manifest.json")
    final_gate = read_json(PILOT_FINAL_GATE)
    if final_gate.get("status") != "REQUIREMENTS_ACCEPTED" or final_gate.get("unresolved_ids"):
        raise ValueError("pilot requirement final gate is not closed")
    expected = {
        row["pilot_id"]: row["revised_requirement_sha256"] for row in final_gate["records"]
    }
    for pilot_id in final_gate["accepted_round_1_unchanged"]:
        freeze = read_json(
            BASE / "n8n_conversion_pilot" / "h1_execution" / "runtime" / "reports"
            / "requirement_revision_freeze_gate_result_v1_1.json"
        )
        expected[pilot_id] = next(
            row["requirement_sha256"] for row in freeze["frozen_records"] if row["pilot_id"] == pilot_id
        )
    records = []
    for inherited in sorted(d8_manifest["inherited_requirements"], key=lambda row: row["pilot_id"]):
        task_id = inherited["pilot_id"]
        artifact = inherited["frozen_requirement"]
        req_hash = artifact_hash(artifact)
        if expected.get(task_id) != req_hash or inherited["requirement_sha256"] != req_hash:
            raise ValueError(f"{task_id}: inherited pilot hash does not match final gate")
        records.append({
            "schema_version": "n8n-s1-frozen-requirement-record-v1",
            "record_id": task_id,
            "source_id": inherited["source_id"],
            "workflow_sha256": inherited["workflow_sha256"],
            "dataset_role": "train_dev_only",
            "version_origin": "accepted_pilot_requirement",
            "requirement_sha256": req_hash,
            "requirement": artifact,
            "decision_provenance": {
                "final_gate_path": PILOT_FINAL_GATE.relative_to(BASE).as_posix(),
                "final_gate_sha256": sha256_file(PILOT_FINAL_GATE),
                "decision": "accept",
            },
        })
    return records


def build_formal_records() -> list[dict[str, Any]]:
    d9_manifest = read_json(D9 / "round_manifest.json")
    closure = read_json(D9 / "revisions" / "remaining_revision_01" / "closure_manifest.json")
    recon_gate_path = D9 / "reconsiderations" / "full_description_01" / "reports" / "returned_gate.json"
    recon_gate = read_json(recon_gate_path)
    if closure.get("status") != "pass" or closure["current_d9_counts"]["closed"] != 107:
        raise ValueError("six-revision closure manifest is invalid")
    if recon_gate.get("status") != "PASS" or recon_gate.get("accepted") != 2:
        raise ValueError("reconsideration gate is not closed")
    records = []
    for index in range(1, 110):
        task_id = f"N8F-{index:03d}"
        batch, task_manifest = find_batch(task_id, d9_manifest)
        packet_path = D9 / "verifier_B" / batch / "packets" / f"{task_id}.packet.json"
        packet = read_json(packet_path)
        if task_id in REVISED:
            revision_path = REVISED[task_id]
            revision = read_json(revision_path)
            artifact = revision["revised_requirement"]
            req_hash = revision["revised_requirement_sha256"]
            confirmation_path = revision_confirmation(task_id)
            confirmation = read_json(confirmation_path)
            if confirmation.get("decision") != "accept" or confirmation.get("revised_requirement_sha256") != req_hash:
                raise ValueError(f"{task_id}: revised requirement lacks accepted confirmation")
            origin = "revised_and_confirmed"
            provenance = {
                "original_verification_path": verifier_response(batch, task_id).relative_to(BASE).as_posix(),
                "revision_record_path": revision_path.relative_to(BASE).as_posix(),
                "revision_record_sha256": sha256_file(revision_path),
                "final_confirmation_path": confirmation_path.relative_to(BASE).as_posix(),
                "final_confirmation_sha256": sha256_file(confirmation_path),
                "decision": "accept",
            }
        elif task_id in RECONSIDERED:
            artifact = packet["candidate_requirement"]
            req_hash = packet["requirement_sha256"]
            response_path = (
                D9 / "reconsiderations" / "full_description_01" / "RV-03" / "responses"
                / f"{task_id}.response.json"
            )
            response = read_json(response_path)
            if response.get("decision") != "accept" or response.get("requirement_sha256") != req_hash:
                raise ValueError(f"{task_id}: reconsideration does not accept unchanged requirement")
            origin = "accepted_after_full_description_reconsideration"
            provenance = {
                "original_verification_path": verifier_response(batch, task_id).relative_to(BASE).as_posix(),
                "reconsideration_response_path": response_path.relative_to(BASE).as_posix(),
                "reconsideration_response_sha256": sha256_file(response_path),
                "decision": "accept",
            }
        else:
            artifact = packet["candidate_requirement"]
            req_hash = packet["requirement_sha256"]
            response_path = verifier_response(batch, task_id)
            response = read_json(response_path)
            if response.get("decision") != "accept" or response.get("requirement_sha256") != req_hash:
                raise ValueError(f"{task_id}: original requirement is not accepted")
            origin = "accepted_unchanged"
            provenance = {
                "verification_response_path": response_path.relative_to(BASE).as_posix(),
                "verification_response_sha256": sha256_file(response_path),
                "decision": "accept",
            }
        artifact = d9_artifact(artifact)
        if artifact_hash(artifact) != req_hash:
            raise ValueError(f"{task_id}: final requirement hash mismatch")
        if task_manifest["source_id"] != packet["source_id"]:
            raise ValueError(f"{task_id}: source identity mismatch")
        records.append({
            "schema_version": "n8n-s1-frozen-requirement-record-v1",
            "record_id": task_id,
            "source_id": packet["source_id"],
            "workflow_sha256": next(
                row["workflow_sha256"] for row in read_json(D8 / "round_manifest.json")["authoring_assignments"]
                if row["task_id"] == task_id
            ),
            "dataset_role": "formal_evaluation_pool",
            "version_origin": origin,
            "requirement_sha256": req_hash,
            "requirement": artifact,
            "source_evidence": {
                "packet_path": packet_path.relative_to(BASE).as_posix(),
                "packet_sha256": sha256_file(packet_path),
                "public_description_sha256": packet["public_description_sha256"],
            },
            "decision_provenance": provenance,
        })
    return records


def main() -> int:
    if (TARGET / "freeze_manifest.json").exists():
        raise FileExistsError("freeze already exists; validate it instead of rewriting the snapshot")
    records = build_pilot_records() + build_formal_records()
    if len(records) != 120 or len({row["record_id"] for row in records}) != 120:
        raise ValueError("final set must contain 120 unique record IDs")
    if len({row["source_id"] for row in records}) != 120:
        raise ValueError("final set contains duplicate source IDs")
    requirements = TARGET / "requirements"
    requirements.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    for formal_index, record in enumerate(records, start=1):
        record["formal_index"] = formal_index
        path = requirements / f"{record['record_id']}.json"
        write_json(path, record)
        manifest_records.append({
            "formal_index": formal_index,
            "record_id": record["record_id"],
            "source_id": record["source_id"],
            "dataset_role": record["dataset_role"],
            "version_origin": record["version_origin"],
            "requirement_sha256": record["requirement_sha256"],
            "record_sha256": sha256_file(path),
        })
    jsonl = TARGET / "canonical_requirement_set.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    csv_path = TARGET / "canonical_requirement_index.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_records[0]))
        writer.writeheader()
        writer.writerows(manifest_records)
    manifest = {
        "schema_version": "n8n-s1-requirement-freeze-v1",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "FROZEN",
        "total_requirement_count": 120,
        "train_dev_only_count": 11,
        "formal_evaluation_pool_count": 109,
        "version_origin_counts": {
            "accepted_pilot_requirement": 11,
            "accepted_unchanged": 99,
            "revised_and_confirmed": 8,
            "accepted_after_full_description_reconsideration": 2,
        },
        "unresolved_count": 0,
        "excluded_count": 0,
        "canonical_requirement_set_sha256": sha256_file(jsonl),
        "canonical_requirement_index_sha256": sha256_file(csv_path),
        "records": manifest_records,
        "invalidation_rule": "Any post-freeze change to a requirement record, canonical set, or index invalidates this snapshot.",
    }
    manifest["manifest_sha256"] = artifact_hash(manifest)
    write_json(TARGET / "freeze_manifest.json", manifest)
    (TARGET / "README.md").write_text(
        """# S1最终需求冻结集

本目录冻结120条自然语言需求：11条方法开发试点仅限训练/开发，109条构成正式评估池。冻结只确定需求边界和版本，不表示109条Gold、独立正确性规则或系统运行结果已经完成。

版本来源：99条首轮直接接受，8条返修后由原复核者确认，2条基于完整公开说明重新判断后接受。任何冻结后修改都会使freeze_manifest.json失效。
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "frozen": 120,
        "train_dev_only": 11,
        "formal_evaluation_pool": 109,
        "manifest_sha256": manifest["manifest_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
