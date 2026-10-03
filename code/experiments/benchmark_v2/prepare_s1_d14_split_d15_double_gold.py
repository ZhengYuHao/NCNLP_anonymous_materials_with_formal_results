#!/usr/bin/env python3
"""Create a result-blind formal split and D15 secondary Gold annotation packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARK = Path(__file__).resolve().parent
ROUND_ROOT = BENCHMARK / "n8n_native_s1" / "source_review_round_v2"
REQUIREMENT_FREEZE = ROUND_ROOT / "d10_requirement_freeze_v1"
SOURCE_SET = ROUND_ROOT / "frozen" / "source_admission_v1" / "formal_source_set.jsonl"
INVENTORY = BENCHMARK / "n8n_native_s1" / "s1_candidate_inventory.jsonl"
GOLD_ROUND = ROUND_ROOT / "d10_gold_oracle_round_v1"
DEFAULT_OUT = ROUND_ROOT / "d14_split_d15_double_gold_v2"

FEATURES = (
    "domain",
    "size",
    "topology",
    "cross_type_direction",
    "has_control_operation",
    "has_explicit_code",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(namespace: str, task_id: str) -> str:
    return hashlib.sha256(f"{namespace}:{task_id}".encode("utf-8")).hexdigest()


def feature_values(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[field]).lower() for field in FEATURES)


def obligations(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple((field, str(row[field]).lower()) for field in FEATURES)


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]).lower() for row in rows).items()))


def select_stratified(
    rows: list[dict[str, Any]],
    target_size: int,
    namespace: str,
    forced: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Greedily approach marginal quotas, with deterministic hash tie-breaking."""
    forced = forced or set()
    by_id = {row["task_id"]: row for row in rows}
    if not forced <= set(by_id):
        raise ValueError(f"unknown forced task IDs: {sorted(forced - set(by_id))}")
    if len(forced) > target_size:
        raise ValueError("forced set is larger than target")

    totals = Counter(obligation for row in rows for obligation in obligations(row))
    targets = {key: value * target_size / len(rows) for key, value in totals.items()}
    selected = [by_id[task_id] for task_id in sorted(forced)]
    selected_ids = set(forced)
    counts = Counter(obligation for row in selected for obligation in obligations(row))
    seen_combos = Counter(feature_values(row) for row in selected)

    while len(selected) < target_size:
        candidates = [row for row in rows if row["task_id"] not in selected_ids]

        def score(row: dict[str, Any]) -> tuple[float, float, str]:
            quota_gain = 0.0
            rare_gain = 0.0
            for key in obligations(row):
                deficit = max(targets[key] - counts[key], 0.0)
                quota_gain += deficit / max(targets[key], 1.0)
                if counts[key] == 0:
                    rare_gain += 1.0 / totals[key]
            combo_gain = 1.0 if seen_combos[feature_values(row)] == 0 else 0.0
            return (
                quota_gain + 0.5 * rare_gain + 0.15 * combo_gain,
                combo_gain,
                stable_hash(namespace, row["task_id"]),
            )

        chosen = max(candidates, key=score)
        selected.append(chosen)
        selected_ids.add(chosen["task_id"])
        counts.update(obligations(chosen))
        seen_combos.update([feature_values(chosen)])
    return sorted(selected, key=lambda row: row["task_id"])


def high_risk(row: dict[str, Any]) -> bool:
    return row["size"] == "native_41_plus" or (
        row["size"] == "native_21_40"
        and row["topology"] in {"iteration_or_cycle", "multi_branch_join"}
        and row["cross_type_direction"] == "bidirectional"
    )


def find_gold_file(task_id: str, kind: str) -> Path:
    matches = list((GOLD_ROUND / "gold_primary").glob(f"batch_*/{kind}/{task_id}.*.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one primary Gold {kind} file for {task_id}, found {len(matches)}")
    return matches[0]


def blank_gold(task_id: str, requirement_sha256: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "annotator_id": None,
        "status": "unsubmitted",
        "requirement_sha256": requirement_sha256,
        "guideline_version": "K-v2.1+n8n-conversion-v1",
        "nodes": [],
        "cfg_edges": [],
        "dfg_edges": [],
        "entry_nodes": [],
        "terminal_nodes": [],
        "confidence": None,
        "notes": "",
    }


SECONDARY_GUIDE = """
# Gold 第二人独立标注说明

## 任务

请依据冻结需求与脱敏原生图，独立建立方法中立的任务节点、控制依赖（CFG）和数据依赖（DFG）。
节点只分为 `semantic`、`code`、`terminal`。

## 独立性边界

1. 不得查看第一名标注者的 Gold、NCNLP 输出、比较系统输出、DSL 或 Python 输出。
2. 冻结需求决定业务行为边界；原生图只能核验和定位证据，不能增加需求未声明的行为。
3. 原生 n8n 节点不等于任务节点，禁止机械一对一复制。
4. 每个节点均需提供需求证据、原生图证据和分类理由。
5. CFG 表达执行顺序、条件和分支；DFG 表达数据产生与使用关系。
6. 完成后填写个人标注者编号，将 `status` 改为 `submitted`。证据确实不足时使用 `insufficient_evidence` 并说明原因。

请只编辑 `responses/` 中的文件，不修改 `packets/`、Schema 或批次清单。
"""


def build_rows() -> list[dict[str, Any]]:
    freeze = read_json(REQUIREMENT_FREEZE / "freeze_manifest.json")
    source_set = {item["source_id"]: item for item in read_jsonl(SOURCE_SET)}
    inventory = {item["source_id"]: item for item in read_jsonl(INVENTORY)}
    rows = []
    for record in freeze["records"]:
        if record["dataset_role"] != "formal_evaluation_pool":
            continue
        source_id = record["source_id"]
        source = source_set[source_id]
        native = inventory[source_id]
        if record.get("workflow_sha256", source["workflow_sha256"]) != source["workflow_sha256"]:
            raise ValueError(f"workflow identity mismatch for {record['record_id']}")
        rows.append({
            "task_id": record["record_id"],
            "source_id": source_id,
            "workflow_sha256": source["workflow_sha256"],
            "lineage_group": native["deduplication"]["near_duplicate_group"],
            "domain": source["domain_heuristic"],
            "size": source["native_node_count_band"],
            "topology": source["native_topology"],
            "cross_type_direction": source["cross_type_direction"],
            "has_control_operation": bool(source["has_control_operation"]),
            "has_explicit_code": bool(source["has_explicit_code"]),
            "native_node_count": native["node_count_execution"],
            "native_main_edge_count": native["main_edge_count"],
            "native_dependency_edge_count": native["dependency_edge_count"],
            "requirement_sha256": record["requirement_sha256"],
        })
    if len(rows) != 109 or len({row["task_id"] for row in rows}) != 109:
        raise ValueError("formal pool must contain 109 unique tasks")
    return sorted(rows, key=lambda row: row["task_id"])


def split_report(all_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    test_ids = {row["task_id"] for row in test_rows}
    dev_rows = [row for row in all_rows if row["task_id"] not in test_ids]
    fields = {}
    for field in FEATURES:
        overall = distribution(all_rows, field)
        test = distribution(test_rows, field)
        dev = distribution(dev_rows, field)
        values = sorted(set(overall) | set(test) | set(dev))
        fields[field] = {
            value: {
                "all": overall.get(value, 0),
                "test": test.get(value, 0),
                "dev": dev.get(value, 0),
                "test_share": round(test.get(value, 0) / overall[value], 4),
            }
            for value in values
        }
    test_lineages = {row["lineage_group"] for row in test_rows}
    dev_lineages = {row["lineage_group"] for row in dev_rows}
    return {
        "all_tasks": len(all_rows),
        "test_tasks": len(test_rows),
        "development_and_qualification_tasks": len(dev_rows),
        "unique_workflow_hashes": len({row["workflow_sha256"] for row in all_rows}),
        "unique_lineage_groups": len({row["lineage_group"] for row in all_rows}),
        "cross_split_lineage_overlap": sorted(test_lineages & dev_lineages),
        "feature_distributions": fields,
    }


def write_csv_file(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_secondary_packages(out: Path, sample: list[dict[str, Any]], batch_count: int) -> list[dict[str, Any]]:
    batches = [sample[index::batch_count] for index in range(batch_count)]
    manifests = []
    schema_source = GOLD_ROUND / "schemas" / "gold.schema.json"
    for index, rows in enumerate(batches, 1):
        batch_id = f"batch_{index:02d}"
        directory = out / "gold_secondary" / batch_id
        write_text(directory / "README.md", SECONDARY_GUIDE)
        shutil.copy2(schema_source, directory / "gold.schema.json")
        task_ids = []
        for row in rows:
            task_id = row["task_id"]
            (directory / "packets").mkdir(parents=True, exist_ok=True)
            shutil.copy2(find_gold_file(task_id, "packets"), directory / "packets" / f"{task_id}.packet.json")
            write_json(directory / "responses" / f"{task_id}.response.json", blank_gold(task_id, row["requirement_sha256"]))
            task_ids.append(task_id)
        message = (
            f"你好，请完成 Gold 第二人独立标注。你收到的文件夹是 "
            f"`d14_split_d15_double_gold_v2/gold_secondary/{batch_id}`，共 {len(rows)} 条任务。\n\n"
            "请先阅读 `README.md`，逐条阅读 `packets/`，只填写同名 `responses/`。"
            "不得查看第一名标注者答案、NCNLP或比较系统输出。完成后填写个人标注者编号，"
            "将状态改为 `submitted`，并直接返回整个原文件夹。"
        )
        write_text(directory / "交付留言.txt", message)
        manifest = {
            "round_id": "n8n-s1-d15-secondary-gold-v2",
            "role": "independent_secondary_gold_annotator",
            "batch_id": batch_id,
            "task_count": len(rows),
            "task_ids": task_ids,
            "primary_gold_included": False,
            "system_outputs_included": False,
        }
        write_json(directory / "batch_manifest.json", manifest)
        manifests.append(manifest)
    return manifests


def main(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    all_rows = build_rows()
    if len({row["lineage_group"] for row in all_rows}) != len(all_rows):
        raise ValueError("formal pool contains shared near-duplicate lineage groups")

    dev_size = len(all_rows) - args.test_size
    domain_counts = Counter(row["domain"] for row in all_rows)
    dev_candidates = [row for row in all_rows if domain_counts[row["domain"]] > 1]
    dev_rows = select_stratified(
        dev_candidates,
        dev_size,
        "n8n-s1-development-qualification-v2",
    )
    dev_ids = {row["task_id"] for row in dev_rows}
    test_rows = [row for row in all_rows if row["task_id"] not in dev_ids]
    test_ids = {row["task_id"] for row in test_rows}
    split_rows = [
        {**row, "dataset_partition": "formal_test" if row["task_id"] in test_ids else "development_qualification"}
        for row in all_rows
    ]

    forced_high_risk = {row["task_id"] for row in all_rows if high_risk(row)}
    double_rows = select_stratified(
        all_rows,
        args.double_annotation_size,
        "n8n-s1-secondary-gold-v1",
        forced=forced_high_risk,
    )
    double_ids = {row["task_id"] for row in double_rows}
    sample_rows = [
        {
            **row,
            "sample_reason": "mandatory_high_risk" if row["task_id"] in forced_high_risk else "stratified_fill",
            "dataset_partition": "formal_test" if row["task_id"] in test_ids else "development_qualification",
        }
        for row in double_rows
    ]

    reports = out / "reports"
    write_json(reports / "dataset_split_v1.json", {
        "schema_version": "n8n-s1-dataset-split-v2",
        "selection_reads_system_outputs": False,
        "selection_features": list(FEATURES),
        "selection_strategy": (
            "stratify the 29-task development/qualification set first; reserve singleton-domain "
            "tasks for the formal test; assign the remaining 80 tasks to the formal test"
        ),
        "tie_breaker": "sha256(namespace:task_id)",
        "records": split_rows,
    })
    write_csv_file(reports / "dataset_split_v1.csv", split_rows)
    coverage = split_report(all_rows, test_rows)
    write_json(reports / "dataset_split_coverage_v1.json", coverage)
    write_json(reports / "gold_double_annotation_sample_v1.json", {
        "schema_version": "n8n-s1-gold-double-annotation-sample-v1",
        "population": len(all_rows),
        "sample_size": len(sample_rows),
        "sample_fraction": round(len(sample_rows) / len(all_rows), 4),
        "mandatory_high_risk_count": len(forced_high_risk),
        "high_risk_rule": (
            "native_41_plus OR (native_21_40 AND topology in {iteration_or_cycle, "
            "multi_branch_join} AND bidirectional)"
        ),
        "selection_reads_system_outputs": False,
        "records": sample_rows,
    })
    write_csv_file(reports / "gold_double_annotation_sample_v1.csv", sample_rows)
    batch_manifests = build_secondary_packages(out, sample_rows, args.batches)

    lines = [
        "# D14数据划分与D15双标抽样报告", "",
        f"- 正式任务总体：{len(all_rows)}", f"- 正式测试候选：{len(test_rows)}",
        f"- 开发与系统资格检查：{len(all_rows) - len(test_rows)}",
        f"- Gold独立双标：{len(sample_rows)}（{len(sample_rows) / len(all_rows):.1%}）",
        f"- 强制高风险双标：{len(forced_high_risk)}", "- 系统输出参与选择：否",
        f"- 跨集合近重复谱系重叠：{len(coverage['cross_split_lineage_overlap'])}", "",
        "## 划分原则", "",
        "1. 只使用正式运行前冻结的来源结构特征，不读取任何系统输出或效果。",
        "2. 以人工近重复谱系组为最小隔离单位；本总体109条恰好对应109个不同谱系组。",
        "3. 先按领域、节点规模、拓扑、跨类型方向、控制结构和显式代码分层选择29条开发/资格任务，其余80条封存为正式测试；仅有1条的领域样本保留在正式测试中。",
        "4. 双标样本强制覆盖预定义高风险结构，再用相同分层规则补足至33条。", "",
        "## 门禁", "",
        f"- 109条唯一任务：{'PASS' if len(all_rows) == 109 else 'FAIL'}",
        f"- 109个唯一工作流哈希：{'PASS' if coverage['unique_workflow_hashes'] == 109 else 'FAIL'}",
        f"- 跨集合谱系泄漏为0：{'PASS' if not coverage['cross_split_lineage_overlap'] else 'FAIL'}",
        f"- 双标比例不少于30%：{'PASS' if len(sample_rows) / len(all_rows) >= 0.30 else 'FAIL'}",
        f"- 所有高风险任务进入双标：{'PASS' if forced_high_risk <= double_ids else 'FAIL'}",
    ]
    write_text(reports / "D14_D15_summary_v1.md", "\n".join(lines))

    artifact_paths = sorted(
        path
        for path in out.rglob("*")
        if path.is_file() and path.name not in {"freeze_manifest.json", "validation_v1.json"}
    )
    manifest = {
        "schema_version": "n8n-s1-d14-d15-preparation-freeze-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not coverage["cross_split_lineage_overlap"] and len(sample_rows) >= 33 else "FAIL",
        "supersedes": "../d14_split_d15_double_gold_v1/freeze_manifest.json",
        "revision_reason": (
            "Before inspecting any system output, the formal test allocation was increased "
            "from 65 to 80 tasks to improve statistical power; the remaining 29 tasks plus "
            "the existing 11-task pilot pool are reserved for development and qualification."
        ),
        "upstream_requirement_freeze_sha256": sha256_file(REQUIREMENT_FREEZE / "freeze_manifest.json"),
        "upstream_formal_source_set_sha256": sha256_file(SOURCE_SET),
        "formal_tasks": len(all_rows),
        "formal_test_tasks": len(test_rows),
        "development_qualification_tasks": len(all_rows) - len(test_rows),
        "secondary_gold_tasks": len(sample_rows),
        "secondary_gold_batches": batch_manifests,
        "artifacts": {str(path.relative_to(out)).replace("\\", "/"): sha256_file(path) for path in artifact_paths},
        "invalidation_rule": "Any change to an artifact or upstream freeze invalidates this manifest.",
    }
    write_json(out / "freeze_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "formal_test": len(test_rows),
        "development_qualification": len(all_rows) - len(test_rows),
        "secondary_gold": len(sample_rows),
        "mandatory_high_risk": len(forced_high_risk),
        "lineage_overlap": len(coverage["cross_split_lineage_overlap"]),
        "out": str(out),
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--test-size", type=int, default=80)
    parser.add_argument("--double-annotation-size", type=int, default=33)
    parser.add_argument("--batches", type=int, default=4)
    raise SystemExit(main(parser.parse_args()))
