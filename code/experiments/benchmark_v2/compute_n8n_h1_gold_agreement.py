#!/usr/bin/env python3
"""Compute pre-adjudication agreement for frozen H1 Gold annotations."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment


BENCHMARK = Path(__file__).resolve().parent
FROZEN = BENCHMARK / "n8n_conversion_pilot" / "h1_execution" / "frozen" / "gold_oracle_v1"
REPORTS = BENCHMARK / "n8n_conversion_pilot" / "h1_execution" / "runtime" / "reports"
TASK_IDS = ["N8C-003", "N8C-006", "N8C-010", "N8C-011"]
THRESHOLDS = {
    "node_f1": 0.80,
    "node_type_raw_agreement": 0.85,
    "node_type_cohen_kappa": 0.70,
    "cfg_f1": 0.80,
    "dfg_f1": 0.80,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").casefold())


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union else 0.0


def evidence_intervals(node: dict[str, Any], requirement: str) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for evidence in node.get("requirement_evidence", []):
        start = requirement.find(evidence)
        if start >= 0:
            intervals.append((start, start + len(evidence)))
    return intervals


def native_references(node: dict[str, Any], native_names: list[str]) -> set[str]:
    references: set[str] = set()
    normalized_names = [(name, normalize(name)) for name in native_names]
    for evidence in node.get("native_evidence", []):
        normalized_evidence = normalize(evidence)
        for name, normalized_name in normalized_names:
            if normalized_name and normalized_name in normalized_evidence:
                references.add(name)
    return references


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def node_similarity(
    node_a: dict[str, Any],
    node_b: dict[str, Any],
    requirement: str,
    native_names: list[str],
) -> dict[str, float]:
    intervals_a = evidence_intervals(node_a, requirement)
    intervals_b = evidence_intervals(node_b, requirement)
    requirement_iou = max(
        (interval_iou(left, right) for left in intervals_a for right in intervals_b),
        default=0.0,
    )
    native_jaccard = jaccard(
        native_references(node_a, native_names),
        native_references(node_b, native_names),
    )
    label_similarity = SequenceMatcher(
        None,
        normalize(node_a.get("label")),
        normalize(node_b.get("label")),
    ).ratio()
    return {
        "score": max(requirement_iou, native_jaccard),
        "requirement_iou": requirement_iou,
        "native_jaccard": native_jaccard,
        "label_similarity_diagnostic_only": label_similarity,
    }


def align_nodes(
    nodes_a: list[dict[str, Any]],
    nodes_b: list[dict[str, Any]],
    requirement: str,
    native_names: list[str],
    threshold: float = 0.50,
) -> list[dict[str, Any]]:
    details = [
        [node_similarity(node_a, node_b, requirement, native_names) for node_b in nodes_b]
        for node_a in nodes_a
    ]
    matrix = [
        [item["score"] + item["label_similarity_diagnostic_only"] * 1e-6 for item in row]
        for row in details
    ]
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    matches: list[dict[str, Any]] = []
    for row, column in zip(rows, columns):
        detail = details[row][column]
        if detail["score"] < threshold:
            continue
        alternatives = sorted(
            (item["score"] for index, item in enumerate(details[row]) if index != column),
            reverse=True,
        )
        next_best = alternatives[0] if alternatives else 0.0
        matches.append(
            {
                "primary_node": nodes_a[row]["node_id"],
                "double_node": nodes_b[column]["node_id"],
                **detail,
                "primary_margin_over_next_best": detail["score"] - next_best,
            }
        )
    return matches


def prf(matches: int, primary_count: int, double_count: int) -> dict[str, float | int]:
    precision = matches / primary_count if primary_count else float(double_count == 0)
    recall = matches / double_count if double_count else float(primary_count == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matches": matches,
        "primary_count": primary_count,
        "double_count": double_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    counts_a = Counter(left for left, _ in pairs)
    counts_b = Counter(right for _, right in pairs)
    labels = set(counts_a) | set(counts_b)
    expected = sum(
        (counts_a[label] / len(pairs)) * (counts_b[label] / len(pairs))
        for label in labels
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def edge_metrics(
    edges_a: list[dict[str, Any]],
    edges_b: list[dict[str, Any]],
    mapping: dict[str, str],
    include_variable: bool,
) -> dict[str, float | int]:
    mapped_a: Counter[tuple[str, ...]] = Counter()
    for edge in edges_a:
        if edge["source"] not in mapping or edge["target"] not in mapping:
            continue
        key = (mapping[edge["source"]], mapping[edge["target"]])
        if include_variable:
            key += (normalize(edge.get("variable")),)
        mapped_a[key] += 1
    canonical_b: Counter[tuple[str, ...]] = Counter()
    for edge in edges_b:
        key = (edge["source"], edge["target"])
        if include_variable:
            key += (normalize(edge.get("variable")),)
        canonical_b[key] += 1
    matches = sum((mapped_a & canonical_b).values())
    return prf(matches, len(edges_a), len(edges_b))


def task_metrics(task_id: str) -> dict[str, Any]:
    primary = read_json(FROZEN / "gold_primary" / f"{task_id}.response.json")
    double = read_json(FROZEN / "gold_double" / f"{task_id}.response.json")
    packet = read_json(FROZEN / "gold_primary" / f"{task_id}.packet.json")
    requirement = packet["frozen_requirement"]["requirement_text"]
    native_names = [item["name"] for item in packet["native_graph_evidence"]["native_nodes"]]
    alignment = align_nodes(primary["nodes"], double["nodes"], requirement, native_names)
    mapping = {item["primary_node"]: item["double_node"] for item in alignment}
    primary_nodes = {item["node_id"]: item for item in primary["nodes"]}
    double_nodes = {item["node_id"]: item for item in double["nodes"]}
    type_pairs = [
        (primary_nodes[left]["node_type"], double_nodes[right]["node_type"])
        for left, right in mapping.items()
    ]
    family_pairs = [
        (primary_nodes[left]["operation_family"], double_nodes[right]["operation_family"])
        for left, right in mapping.items()
    ]
    matched_primary = set(mapping)
    matched_double = set(mapping.values())
    return {
        "task_id": task_id,
        "node": prf(len(alignment), len(primary["nodes"]), len(double["nodes"])),
        "node_type_pairs": type_pairs,
        "node_type_raw_agreement": (
            sum(left == right for left, right in type_pairs) / len(type_pairs) if type_pairs else None
        ),
        "node_type_cohen_kappa": cohen_kappa(type_pairs),
        "operation_family_raw_agreement": (
            sum(left == right for left, right in family_pairs) / len(family_pairs) if family_pairs else None
        ),
        "cfg": edge_metrics(primary["cfg_edges"], double["cfg_edges"], mapping, False),
        "dfg_endpoint_only_diagnostic": edge_metrics(
            primary["dfg_edges"], double["dfg_edges"], mapping, False
        ),
        "dfg": edge_metrics(primary["dfg_edges"], double["dfg_edges"], mapping, True),
        "alignment": alignment,
        "unmatched_primary_nodes": sorted(set(primary_nodes) - matched_primary),
        "unmatched_double_nodes": sorted(set(double_nodes) - matched_double),
        "low_margin_alignments": [
            item
            for item in alignment
            if item["primary_margin_over_next_best"] < 0.15
        ],
    }


def aggregate(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    type_pairs = [pair for task in per_task for pair in task["node_type_pairs"]]
    node = prf(
        sum(task["node"]["matches"] for task in per_task),
        sum(task["node"]["primary_count"] for task in per_task),
        sum(task["node"]["double_count"] for task in per_task),
    )
    cfg = prf(
        sum(task["cfg"]["matches"] for task in per_task),
        sum(task["cfg"]["primary_count"] for task in per_task),
        sum(task["cfg"]["double_count"] for task in per_task),
    )
    dfg = prf(
        sum(task["dfg"]["matches"] for task in per_task),
        sum(task["dfg"]["primary_count"] for task in per_task),
        sum(task["dfg"]["double_count"] for task in per_task),
    )
    dfg_endpoint_only = prf(
        sum(task["dfg_endpoint_only_diagnostic"]["matches"] for task in per_task),
        sum(task["dfg_endpoint_only_diagnostic"]["primary_count"] for task in per_task),
        sum(task["dfg_endpoint_only_diagnostic"]["double_count"] for task in per_task),
    )
    return {
        "node": node,
        "node_type_raw_agreement": (
            sum(left == right for left, right in type_pairs) / len(type_pairs) if type_pairs else None
        ),
        "node_type_cohen_kappa": cohen_kappa(type_pairs),
        "node_type_pair_count": len(type_pairs),
        "cfg": cfg,
        "dfg_endpoint_only_diagnostic": dfg_endpoint_only,
        "dfg": dfg,
    }


def threshold_checks(metrics: dict[str, Any]) -> dict[str, bool]:
    kappa = metrics["node_type_cohen_kappa"]
    return {
        "node_f1": metrics["node"]["f1"] >= THRESHOLDS["node_f1"],
        "node_type_raw_agreement": (
            metrics["node_type_raw_agreement"] >= THRESHOLDS["node_type_raw_agreement"]
        ),
        "node_type_cohen_kappa": kappa is not None and kappa >= THRESHOLDS["node_type_cohen_kappa"],
        "cfg_f1": metrics["cfg"]["f1"] >= THRESHOLDS["cfg_f1"],
        "dfg_f1": metrics["dfg"]["f1"] >= THRESHOLDS["dfg_f1"],
    }


def markdown(report: dict[str, Any]) -> str:
    aggregate_metrics = report["aggregate"]
    checks = report["threshold_checks"]
    fmt = lambda value: "NA" if value is None else f"{value:.3f}"
    lines = [
        "# H1 Gold 双标一致性报告（裁决前）",
        "",
        f"状态：**{report['status']}**",
        "",
        "| 指标 | 结果 | 预设门槛 | 通过 |",
        "|---|---:|---:|---|",
        f"| Node F1 | {fmt(aggregate_metrics['node']['f1'])} | >= 0.80 | {'是' if checks['node_f1'] else '否'} |",
        f"| 节点类型原始一致率 | {fmt(aggregate_metrics['node_type_raw_agreement'])} | >= 0.85 | {'是' if checks['node_type_raw_agreement'] else '否'} |",
        f"| 节点类型 Cohen's kappa | {fmt(aggregate_metrics['node_type_cohen_kappa'])} | >= 0.70 | {'是' if checks['node_type_cohen_kappa'] else '否'} |",
        f"| CFG Endpoint F1 | {fmt(aggregate_metrics['cfg']['f1'])} | >= 0.80 | {'是' if checks['cfg_f1'] else '否'} |",
        f"| DFG Endpoint-only F1（诊断） | {fmt(aggregate_metrics['dfg_endpoint_only_diagnostic']['f1'])} | 不作门槛 | - |",
        f"| DFG Endpoint+Variable F1 | {fmt(aggregate_metrics['dfg']['f1'])} | >= 0.80 | {'是' if checks['dfg_f1'] else '否'} |",
        "",
        "## 分任务结果",
        "",
        "| 任务 | Node F1 | 类型一致率 | CFG F1 | DFG端点F1 | DFG严格F1 | 未匹配A/B | 低边际匹配 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in report["per_task"]:
        lines.append(
            f"| {task['task_id']} | {fmt(task['node']['f1'])} | "
            f"{fmt(task['node_type_raw_agreement'])} | {fmt(task['cfg']['f1'])} | "
            f"{fmt(task['dfg_endpoint_only_diagnostic']['f1'])} | {fmt(task['dfg']['f1'])} | "
            f"{len(task['unmatched_primary_nodes'])}/"
            f"{len(task['unmatched_double_nodes'])} | {len(task['low_margin_alignments'])} |"
        )
    lines.extend(
        [
            "",
            "节点只依据冻结需求原文区间与原生证据重合度进行一对一最大权匹配；节点类型和操作族不参与匹配。",
            "该报告是裁决前一致性，不是系统效果，也不是最终共识Gold。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute frozen H1 pre-adjudication Gold agreement")
    parser.add_argument("--json-out", type=Path, default=REPORTS / "h1_gold_agreement_pre_adjudication_v1.json")
    parser.add_argument("--md-out", type=Path, default=REPORTS / "h1_gold_agreement_pre_adjudication_v1.md")
    args = parser.parse_args()
    per_task = [task_metrics(task_id) for task_id in TASK_IDS]
    aggregate_metrics = aggregate(per_task)
    checks = threshold_checks(aggregate_metrics)
    report = {
        "status": "PASS" if all(checks.values()) else "NEEDS_ADJUDICATION",
        "scope": "GM-A primary versus GM-B independent double annotation; pre-adjudication",
        "alignment_rule": (
            "maximum-weight one-to-one alignment using max(requirement span IoU, native evidence Jaccard) "
            ">= 0.50; label similarity is used only as a 1e-6 deterministic tie-breaker; node type and "
            "operation family do not affect matching"
        ),
        "edge_rule": "all primary/double edges remain in denominators; matches require aligned endpoints and normalized DFG variable",
        "thresholds": THRESHOLDS,
        "aggregate": aggregate_metrics,
        "threshold_checks": checks,
        "per_task": per_task,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "aggregate": aggregate_metrics, "checks": checks}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
