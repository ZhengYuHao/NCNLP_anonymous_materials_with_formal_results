#!/usr/bin/env python3
"""Compute pre-adjudication agreement for the S1 D15 secondary-Gold sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compute_n8n_h1_gold_agreement import (
    aggregate,
    align_nodes,
    cohen_kappa,
    edge_metrics,
    prf,
)


BENCHMARK = Path(__file__).resolve().parent
ROUND_ROOT = BENCHMARK / "n8n_native_s1" / "source_review_round_v2"
DEFAULT_ROOT = ROUND_ROOT / "d14_split_d15_double_gold_v2"
PRIMARY_ROOT = ROUND_ROOT / "d10_gold_oracle_round_v1" / "gold_primary"
THRESHOLDS = {
    "node_type_cohen_kappa": 0.80,
    "cfg_f1": 0.85,
    "dfg_f1": 0.85,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_primary(task_id: str) -> Path:
    matches = list(PRIMARY_ROOT.glob(f"batch_*/responses/{task_id}.response.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one primary response for {task_id}, found {len(matches)}")
    return matches[0]


def task_metrics(response_path: Path, packet_path: Path) -> dict[str, Any]:
    secondary = read_json(response_path)
    task_id = secondary["task_id"]
    primary = read_json(find_primary(task_id))
    packet = read_json(packet_path)
    requirement = packet["frozen_requirement"]["requirement_text"]
    native_names = [item["name"] for item in packet["native_graph_evidence"]["native_nodes"]]
    alignment = align_nodes(primary["nodes"], secondary["nodes"], requirement, native_names)
    mapping = {item["primary_node"]: item["double_node"] for item in alignment}
    primary_nodes = {item["node_id"]: item for item in primary["nodes"]}
    secondary_nodes = {item["node_id"]: item for item in secondary["nodes"]}
    type_pairs = [
        (primary_nodes[left]["node_type"], secondary_nodes[right]["node_type"])
        for left, right in mapping.items()
    ]
    family_pairs = [
        (primary_nodes[left]["operation_family"], secondary_nodes[right]["operation_family"])
        for left, right in mapping.items()
    ]
    matched_primary = set(mapping)
    matched_secondary = set(mapping.values())
    return {
        "task_id": task_id,
        "batch": response_path.parents[1].name,
        "primary_annotator_id": primary["annotator_id"],
        "secondary_annotator_id": secondary["annotator_id"],
        "node": prf(len(alignment), len(primary["nodes"]), len(secondary["nodes"])),
        "node_type_pairs": type_pairs,
        "node_type_raw_agreement": (
            sum(left == right for left, right in type_pairs) / len(type_pairs)
            if type_pairs else None
        ),
        "node_type_cohen_kappa": cohen_kappa(type_pairs),
        "operation_family_raw_agreement": (
            sum(left == right for left, right in family_pairs) / len(family_pairs)
            if family_pairs else None
        ),
        "cfg": edge_metrics(primary["cfg_edges"], secondary["cfg_edges"], mapping, False),
        "dfg_endpoint_only_diagnostic": edge_metrics(
            primary["dfg_edges"], secondary["dfg_edges"], mapping, False
        ),
        "dfg": edge_metrics(primary["dfg_edges"], secondary["dfg_edges"], mapping, True),
        "alignment": alignment,
        "unmatched_primary_nodes": sorted(set(primary_nodes) - matched_primary),
        "unmatched_secondary_nodes": sorted(set(secondary_nodes) - matched_secondary),
        "low_margin_alignments": [
            item for item in alignment if item["primary_margin_over_next_best"] < 0.15
        ],
        "primary_nodes": primary["nodes"],
        "secondary_nodes": secondary["nodes"],
    }


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    metrics = report["aggregate"]
    checks = report["threshold_checks"]
    lines = [
        "# D15 Gold第二人独立标注一致性报告（仲裁前）", "",
        f"状态：**{report['status']}**", "",
        "| 指标 | 结果 | 当前门槛 | 通过 |",
        "|---|---:|---:|---|",
        f"| 节点边界F1（诊断） | {fmt(metrics['node']['f1'])} | 不作主门槛 | - |",
        f"| 节点类型原始一致率 | {fmt(metrics['node_type_raw_agreement'])} | 诊断 | - |",
        f"| 节点类型Cohen's Kappa | {fmt(metrics['node_type_cohen_kappa'])} | >= 0.80 | {'是' if checks['node_type_cohen_kappa'] else '否'} |",
        f"| CFG端点F1 | {fmt(metrics['cfg']['f1'])} | >= 0.85 | {'是' if checks['cfg_f1'] else '否'} |",
        f"| DFG端点F1（诊断） | {fmt(metrics['dfg_endpoint_only_diagnostic']['f1'])} | 不作主门槛 | - |",
        f"| DFG端点+变量F1 | {fmt(metrics['dfg']['f1'])} | >= 0.85 | {'是' if checks['dfg_f1'] else '否'} |",
        "", "## 对齐工作量", "",
        f"- 任务数：{report['task_count']}",
        f"- 自动候选匹配：{report['matched_node_pairs']} 对",
        f"- 未匹配主标节点：{report['unmatched_primary_nodes']}",
        f"- 未匹配第二标节点：{report['unmatched_secondary_nodes']}",
        f"- 低边际候选匹配：{report['low_margin_alignment_count']} 对",
        f"- 需要人工查看节点对应关系的任务：{report['tasks_requiring_alignment_review']}",
        "", "## 分任务结果", "",
        "| 任务 | 节点F1 | 类型一致率 | CFG F1 | DFG端点F1 | DFG严格F1 | 未匹配主/次 | 低边际 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in report["per_task"]:
        lines.append(
            f"| {task['task_id']} | {fmt(task['node']['f1'])} | "
            f"{fmt(task['node_type_raw_agreement'])} | {fmt(task['cfg']['f1'])} | "
            f"{fmt(task['dfg_endpoint_only_diagnostic']['f1'])} | {fmt(task['dfg']['f1'])} | "
            f"{len(task['unmatched_primary_nodes'])}/{len(task['unmatched_secondary_nodes'])} | "
            f"{len(task['low_margin_alignments'])} |"
        )
    lines.extend([
        "", "## 计算边界", "",
        "节点候选只依据冻结需求证据区间和原生节点证据重合度建立；节点类型不参与匹配。",
        "标签文本只以极小权重打破完全相同的分数，不改变匹配门槛。",
        "未匹配节点和全部原始边保留在指标分母中，因此拆分、合并和漏标不会被忽略。",
        "低边际匹配及未匹配节点必须经方法中立的人工对齐审核后，才能形成最终仲裁前一致性。",
        "本报告衡量人工标注一致性，不是NCNLP系统效果。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    submission_audit = read_json(root / "reports" / "secondary_gold_submission_audit_v1.json")
    if submission_audit["status"] != "PASS":
        raise ValueError("secondary submission audit must pass before agreement analysis")

    per_task: list[dict[str, Any]] = []
    for response_path in sorted((root / "gold_secondary").glob("batch_*/responses/*.response.json")):
        task_id = response_path.name.removesuffix(".response.json")
        packet_path = response_path.parent.parent / "packets" / f"{task_id}.packet.json"
        per_task.append(task_metrics(response_path, packet_path))

    aggregate_metrics = aggregate(per_task)
    checks = {
        "node_type_cohen_kappa": (
            aggregate_metrics["node_type_cohen_kappa"] is not None
            and aggregate_metrics["node_type_cohen_kappa"] >= THRESHOLDS["node_type_cohen_kappa"]
        ),
        "cfg_f1": aggregate_metrics["cfg"]["f1"] >= THRESHOLDS["cfg_f1"],
        "dfg_f1": aggregate_metrics["dfg"]["f1"] >= THRESHOLDS["dfg_f1"],
    }
    review_tasks = [
        task["task_id"] for task in per_task
        if task["unmatched_primary_nodes"]
        or task["unmatched_secondary_nodes"]
        or task["low_margin_alignments"]
    ]
    report = {
        "schema_version": "n8n-s1-d15-pre-adjudication-agreement-v1",
        "status": "PASS" if all(checks.values()) and not review_tasks else "NEEDS_ALIGNMENT_REVIEW",
        "scope": "109-workflow population; frozen 33-workflow secondary annotation sample",
        "alignment_rule": (
            "maximum-weight one-to-one alignment using max(requirement span IoU, native evidence "
            "Jaccard) >= 0.50; node type and operation family do not affect matching"
        ),
        "edge_rule": (
            "all primary and secondary edges remain in denominators; a CFG match requires aligned "
            "endpoints and a DFG match additionally requires normalized variable equality"
        ),
        "thresholds": THRESHOLDS,
        "task_count": len(per_task),
        "matched_node_pairs": sum(task["node"]["matches"] for task in per_task),
        "unmatched_primary_nodes": sum(len(task["unmatched_primary_nodes"]) for task in per_task),
        "unmatched_secondary_nodes": sum(len(task["unmatched_secondary_nodes"]) for task in per_task),
        "low_margin_alignment_count": sum(len(task["low_margin_alignments"]) for task in per_task),
        "tasks_requiring_alignment_review": len(review_tasks),
        "alignment_review_task_ids": review_tasks,
        "aggregate": aggregate_metrics,
        "threshold_checks": checks,
        "per_task": per_task,
    }
    reports = root / "reports"
    json_path = reports / "gold_agreement_pre_adjudication_v1.json"
    md_path = reports / "gold_agreement_pre_adjudication_v1.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, report)

    queue = {
        "schema_version": "n8n-s1-d15-node-alignment-review-queue-v1",
        "instruction": (
            "Confirm node correspondences using frozen requirement and native evidence only. "
            "Do not resolve node-type or edge disagreements during alignment."
        ),
        "tasks": [task for task in per_task if task["task_id"] in review_tasks],
    }
    (reports / "node_alignment_review_queue_v1.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "task_count": report["task_count"],
        "matched_node_pairs": report["matched_node_pairs"],
        "unmatched_primary_nodes": report["unmatched_primary_nodes"],
        "unmatched_secondary_nodes": report["unmatched_secondary_nodes"],
        "low_margin_alignment_count": report["low_margin_alignment_count"],
        "tasks_requiring_alignment_review": report["tasks_requiring_alignment_review"],
        "aggregate": aggregate_metrics,
        "threshold_checks": checks,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    raise SystemExit(main(parser.parse_args()))
