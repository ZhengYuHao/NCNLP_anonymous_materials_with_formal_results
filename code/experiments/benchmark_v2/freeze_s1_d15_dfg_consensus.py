#!/usr/bin/env python3
"""Merge accepted shared edges and adjudicated disputes into frozen D15 DFG consensus."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from validate_s1_d15_dfg_endpoint_adjudication import validate


BENCHMARK = Path(__file__).resolve().parent
DEFAULT_ROOT = (
    BENCHMARK / "n8n_native_s1" / "source_review_round_v2"
    / "d14_split_d15_double_gold_v2"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def response_index(package: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in package.glob("responses/batch_*/*.response.json"):
        response = read_json(path)
        result[response["task_id"]] = response
    return result


def unique_items(items: list[str]) -> list[str]:
    result = []
    for item in items:
        value = item.strip()
        if value and value not in result:
            result.append(value)
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    package = root / "dfg_endpoint_adjudication_v1"
    out = (args.out or root / "dfg_consensus_v1_frozen").resolve()
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")

    gate = validate(package)
    if gate["status"] != "PASS":
        raise ValueError("adjudication return gate has not passed")
    agreement_path = root / "reports" / "dfg_reannotation_agreement_v1.json"
    agreement = read_json(agreement_path)
    if agreement.get("formal_test_opened") is not False:
        raise ValueError("formal test isolation flag is not intact")
    responses = response_index(package)

    consensus_dir = out / "consensus"
    records = []
    shared_count = 0
    included_count = 0
    excluded_count = 0
    for task in agreement["per_task"]:
        task_id = task["task_id"]
        edges = []
        for edge in task["shared_edges"]:
            data_items = unique_items(edge["lane_a_data_items"] + edge["lane_b_data_items"])
            edges.append({
                "source_group": edge["source_group"],
                "target_group": edge["target_group"],
                "data_items": data_items,
                "decision_source": "independent_annotation_agreement",
                "decision_rationale": "A/B两路独立标注均确认该端点存在数据消费关系。",
            })
            shared_count += 1

        decisions = []
        if task["requires_endpoint_adjudication"]:
            response = responses[task_id]
            decisions = response["decisions"]
            for decision in decisions:
                if decision["decision"] == "include":
                    edges.append({
                        "source_group": decision["source_group"],
                        "target_group": decision["target_group"],
                        "data_items": unique_items(decision["canonical_data_items"]),
                        "decision_source": "third_party_adjudication",
                        "dispute_id": decision["dispute_id"],
                        "adjudicator_id": response["adjudicator_id"],
                        "decision_rationale": decision["rationale"],
                        "confidence": decision["confidence"],
                    })
                    included_count += 1
                else:
                    excluded_count += 1

        endpoint_keys = [(edge["source_group"], edge["target_group"]) for edge in edges]
        if len(endpoint_keys) != len(set(endpoint_keys)):
            raise ValueError(f"duplicate consensus endpoint in {task_id}")
        edges.sort(key=lambda edge: (edge["source_group"], edge["target_group"]))
        for index, edge in enumerate(edges, start=1):
            edge["edge_id"] = f"d{index}"

        record = {
            "schema_version": "n8n-s1-d15-dfg-consensus-v1",
            "task_id": task_id,
            "status": "frozen",
            "metric_unit": "directed endpoint between frozen shared business-step groups",
            "dfg_edges": edges,
            "provenance": {
                "pre_adjudication_f1": task["endpoint_metrics"]["f1"],
                "shared_edge_count": len(task["shared_edges"]),
                "adjudicated_dispute_count": len(decisions),
                "adjudicator_id": responses[task_id]["adjudicator_id"] if decisions else None,
                "system_outputs_used": False,
            },
        }
        output_path = consensus_dir / f"{task_id}.dfg.consensus.json"
        write_json(output_path, record)
        records.append({
            "task_id": task_id,
            "edge_count": len(edges),
            "shared_edge_count": len(task["shared_edges"]),
            "adjudicated_dispute_count": len(decisions),
            "sha256": sha256(output_path),
        })

    summary = {
        "schema_version": "n8n-s1-d15-dfg-consensus-summary-v1",
        "status": "FROZEN",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "formal_test_opened": False,
        "task_count": len(records),
        "pre_adjudication_endpoint_metrics": agreement["aggregate_endpoint_metrics"],
        "pre_adjudication_threshold": agreement["threshold"],
        "pre_adjudication_threshold_passed": agreement["threshold_passed"],
        "shared_edge_count": shared_count,
        "adjudicated_include_count": included_count,
        "adjudicated_exclude_count": excluded_count,
        "final_edge_count": shared_count + included_count,
        "records": records,
    }
    write_json(out / "consensus_summary.json", summary)
    manifest_files = sorted(consensus_dir.glob("*.json")) + [out / "consensus_summary.json"]
    manifest = {
        "freeze_id": "n8n-s1-d15-dfg-consensus-v1",
        "status": "FROZEN",
        "frozen_at": summary["frozen_at"],
        "formal_test_opened": False,
        "source_agreement_report": {
            "path": agreement_path.relative_to(root).as_posix(),
            "sha256": sha256(agreement_path),
        },
        "source_adjudication_manifest": {
            "path": (package / "package_manifest.json").relative_to(root).as_posix(),
            "sha256": sha256(package / "package_manifest.json"),
        },
        "task_count": len(records),
        "files": [
            {
                "path": path.relative_to(out).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in manifest_files
        ],
    }
    write_json(out / "freeze_manifest.json", manifest)
    (out / "README.md").write_text(
        "# D15 DFG最终共识数据\n\n"
        "本目录由冻结的A/B独立重标结果和第三方局部分歧处理结果确定性生成。"
        "仲裁前一致性指标保留在`consensus_summary.json`中，不能用仲裁后的结果替代。"
        "生成过程中未使用NCNLP或任何比较系统输出。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"],
        "output": str(out),
        "task_count": summary["task_count"],
        "shared_edge_count": summary["shared_edge_count"],
        "adjudicated_include_count": summary["adjudicated_include_count"],
        "adjudicated_exclude_count": summary["adjudicated_exclude_count"],
        "final_edge_count": summary["final_edge_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
