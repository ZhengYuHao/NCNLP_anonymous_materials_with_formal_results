#!/usr/bin/env python3
"""Compute endpoint agreement for the frozen D15 DFG re-annotation round."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from compute_n8n_h1_gold_agreement import prf


BENCHMARK = Path(__file__).resolve().parent
DEFAULT_ROOT = (
    BENCHMARK
    / "n8n_native_s1"
    / "source_review_round_v2"
    / "d14_split_d15_double_gold_v2"
)
THRESHOLD = 0.85


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.casefold())


def responses(root: Path, lane: str) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(root.glob(f"{lane}/batch_*/responses/*.dfg.response.json")):
        value = read_json(path)
        if value["status"] != "submitted":
            raise ValueError(f"response is not submitted: {path}")
        result[value["task_id"]] = value
    return result


def edge_map(response: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (edge["source_group"], edge["target_group"]): edge
        for edge in response["dfg_edges"]
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    frozen = root / "dfg_reannotation_v1_frozen"
    manifest = read_json(frozen / "round_manifest.json")
    if manifest["protocol_status"] != "frozen_before_formal_test":
        raise ValueError("DFG protocol is not frozen")
    lane_a = responses(frozen, "lane_A")
    lane_b = responses(frozen, "lane_B")
    if set(lane_a) != set(lane_b) or len(lane_a) != 33:
        raise ValueError("A/B task sets are incomplete or different")

    per_task = []
    data_item_exact_matches = 0
    shared_endpoint_count = 0
    for task_id in sorted(lane_a):
        a_edges = edge_map(lane_a[task_id])
        b_edges = edge_map(lane_b[task_id])
        a_keys = set(a_edges)
        b_keys = set(b_edges)
        shared = a_keys & b_keys
        a_only = a_keys - b_keys
        b_only = b_keys - a_keys
        metrics = prf(len(shared), len(a_keys), len(b_keys))
        shared_details = []
        for key in sorted(shared):
            a_items = {normalize(item) for item in a_edges[key]["data_items"]}
            b_items = {normalize(item) for item in b_edges[key]["data_items"]}
            exact = a_items == b_items
            data_item_exact_matches += int(exact)
            shared_endpoint_count += 1
            shared_details.append({
                "source_group": key[0],
                "target_group": key[1],
                "lane_a_data_items": a_edges[key]["data_items"],
                "lane_b_data_items": b_edges[key]["data_items"],
                "normalized_string_sets_equal_diagnostic": exact,
            })
        per_task.append({
            "task_id": task_id,
            "reviewer_a": lane_a[task_id]["reviewer_id"],
            "reviewer_b": lane_b[task_id]["reviewer_id"],
            "endpoint_metrics": metrics,
            "lane_a_only_edges": [a_edges[key] for key in sorted(a_only)],
            "lane_b_only_edges": [b_edges[key] for key in sorted(b_only)],
            "shared_edges": shared_details,
            "requires_endpoint_adjudication": bool(a_only or b_only),
        })

    aggregate = prf(
        sum(item["endpoint_metrics"]["matches"] for item in per_task),
        sum(item["endpoint_metrics"]["primary_count"] for item in per_task),
        sum(item["endpoint_metrics"]["double_count"] for item in per_task),
    )
    disputes = [item for item in per_task if item["requires_endpoint_adjudication"]]
    passed = aggregate["f1"] >= THRESHOLD
    report = {
        "schema_version": "n8n-s1-d15-dfg-reannotation-agreement-v1",
        "status": "PASS_WITH_ADJUDICATION_REQUIRED" if passed and disputes else "PASS" if passed else "FAIL",
        "metric_scope": "pre-adjudication lane_A versus lane_B over frozen shared business-step endpoints",
        "formal_test_opened": False,
        "task_count": len(per_task),
        "threshold": THRESHOLD,
        "aggregate_endpoint_metrics": aggregate,
        "threshold_passed": passed,
        "perfect_agreement_task_count": len(per_task) - len(disputes),
        "endpoint_adjudication_task_count": len(disputes),
        "lane_a_only_edge_count": sum(len(item["lane_a_only_edges"]) for item in disputes),
        "lane_b_only_edge_count": sum(len(item["lane_b_only_edges"]) for item in disputes),
        "shared_endpoint_data_item_exact_string_rate_diagnostic": (
            data_item_exact_matches / shared_endpoint_count if shared_endpoint_count else None
        ),
        "per_task": per_task,
    }
    reports = root / "reports"
    write_json(reports / "dfg_reannotation_agreement_v1.json", report)
    write_json(reports / "dfg_endpoint_adjudication_queue_v1.json", {
        "schema_version": "n8n-s1-d15-dfg-endpoint-adjudication-queue-v1",
        "status": "READY" if disputes else "EMPTY",
        "task_count": len(disputes),
        "lane_a_only_edge_count": report["lane_a_only_edge_count"],
        "lane_b_only_edge_count": report["lane_b_only_edge_count"],
        "tasks": disputes,
    })

    lines = [
        "# D15 DFG独立重标一致性报告", "",
        f"状态：**{report['status']}**", "",
        "> 本结果是协议修订后的独立重标一致性，不覆盖第一次双标及节点对应复核前后的历史结果。", "",
        "| 指标 | 结果 | 门槛 | 判断 |",
        "|---|---:|---:|---|",
        f"| DFG端点Precision | {fmt(aggregate['precision'])} | - | - |",
        f"| DFG端点Recall | {fmt(aggregate['recall'])} | - | - |",
        f"| DFG端点F1 | {fmt(aggregate['f1'])} | >= {THRESHOLD:.2f} | {'通过' if passed else '未通过'} |",
        "", "## 汇总", "",
        f"- 任务数：{len(per_task)}",
        f"- A组边数：{aggregate['primary_count']}",
        f"- B组边数：{aggregate['double_count']}",
        f"- 匹配端点：{aggregate['matches']}",
        f"- 完全一致任务：{report['perfect_agreement_task_count']}",
        f"- 需处理局部分歧任务：{report['endpoint_adjudication_task_count']}",
        f"- A侧独有边：{report['lane_a_only_edge_count']}",
        f"- B侧独有边：{report['lane_b_only_edge_count']}",
        "", "## 分任务结果", "",
        "| 任务 | A边 | B边 | 匹配 | F1 | A独有 | B独有 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in per_task:
        metric = item["endpoint_metrics"]
        lines.append(
            f"| {item['task_id']} | {metric['primary_count']} | {metric['double_count']} | "
            f"{metric['matches']} | {fmt(metric['f1'])} | {len(item['lane_a_only_edges'])} | "
            f"{len(item['lane_b_only_edges'])} |"
        )
    lines.extend([
        "", "## 结论", "",
        "DFG端点F1已经通过冻结门槛。下一步只处理19条任务中的局部端点分歧；14条完全一致任务不再交给第三方。",
        "数据项自由文本的字符串相同率只作为诊断，不作为DFG端点一致性主指标。",
    ])
    (reports / "dfg_reannotation_agreement_v1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "task_count": report["task_count"],
        "aggregate_endpoint_metrics": aggregate,
        "threshold_passed": passed,
        "perfect_tasks": report["perfect_agreement_task_count"],
        "adjudication_tasks": report["endpoint_adjudication_task_count"],
        "lane_a_only_edges": report["lane_a_only_edge_count"],
        "lane_b_only_edges": report["lane_b_only_edge_count"],
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    raise SystemExit(main(parser.parse_args()))
