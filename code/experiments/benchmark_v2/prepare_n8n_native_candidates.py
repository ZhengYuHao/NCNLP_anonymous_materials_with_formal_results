#!/usr/bin/env python3
"""Build an auditable n8n native-Hybrid candidate pool.

The local Hugging Face mirror is discovery evidence only. Formal pilot candidates
must be resolved through the n8n official API and then independently annotated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = ROOT / "n8n_node_classification.v1.json"
DEFAULT_MIRROR = ROOT / "candidate_sources" / "HF_n8n_templates" / "0000.parquet"
DEFAULT_OUT = ROOT / "n8n_native"
API_ROOT = "https://api.n8n.io/templates"

NAMED_REFERENCE_PATTERNS = (
    re.compile(r"\$node\[['\"]([^'\"]+)['\"]\]"),
    re.compile(r"\$items\(['\"]([^'\"]+)['\"]"),
    re.compile(r"\$\(['\"]([^'\"]+)['\"]\)"),
)

DOMAIN_KEYWORDS = {
    "customer_service": ("support", "customer", "ticket", "complaint", "zendesk", "helpdesk"),
    "finance": ("invoice", "payment", "finance", "accounting", "bank", "expense"),
    "governance": ("policy", "compliance", "legal", "contract", "audit", "risk"),
    "industrial": ("manufacturing", "sensor", "maintenance", "inspection", "industrial", "iot"),
    "medical": ("medical", "health", "patient", "clinical", "diagnosis", "hospital"),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_node_type(node_type: str, registry: dict[str, Any]) -> str:
    classes = registry["classes"]
    for class_name in registry["priority"]:
        rule = classes[class_name]
        if node_type in rule.get("exact", []):
            return class_name
        lowered = node_type.lower()
        if any(fragment.lower() in lowered for fragment in rule.get("contains", [])):
            return class_name
    return "external_action_or_io"


def iter_parameter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from iter_parameter_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_parameter_strings(nested)


def extract_named_references(parameters: Any) -> set[str]:
    references: set[str] = set()
    for text in iter_parameter_strings(parameters):
        for pattern in NAMED_REFERENCE_PATTERNS:
            references.update(pattern.findall(text))
    return references


def parse_connections(workflow: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_edges: list[dict[str, Any]] = []
    dependency_edges: list[dict[str, Any]] = []
    connections = workflow.get("connections") or {}
    if not isinstance(connections, dict):
        return main_edges, dependency_edges
    for source, channels in connections.items():
        if not isinstance(channels, dict):
            continue
        for channel, outputs in channels.items():
            if not isinstance(outputs, list):
                continue
            for source_index, output_group in enumerate(outputs):
                if not isinstance(output_group, list):
                    continue
                for edge in output_group:
                    if not isinstance(edge, dict) or not edge.get("node"):
                        continue
                    record = {
                        "source": source,
                        "target": edge["node"],
                        "channel": channel,
                        "source_index": source_index,
                        "target_index": int(edge.get("index", 0)),
                    }
                    if channel == "main" or edge.get("type") == "main":
                        main_edges.append(record)
                    else:
                        dependency_edges.append(record)
    return main_edges, dependency_edges


def has_cycle(nodes: set[str], edges: list[dict[str, Any]]) -> bool:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge["source"] in nodes and edge["target"] in nodes:
            adjacency[edge["source"]].append(edge["target"])
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(target) for target in adjacency[node]):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in nodes if node not in visited)


def weak_component_count(nodes: set[str], edges: list[dict[str, Any]]) -> int:
    if not nodes:
        return 0
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source in nodes and target in nodes:
            adjacency[source].add(target)
            adjacency[target].add(source)
    remaining = set(nodes)
    count = 0
    while remaining:
        count += 1
        queue = deque([remaining.pop()])
        while queue:
            for target in adjacency[queue.popleft()]:
                if target in remaining:
                    remaining.remove(target)
                    queue.append(target)
    return count


def size_band(count: int) -> str:
    if count <= 10:
        return "native_1_10"
    if count <= 20:
        return "native_11_20"
    if count <= 40:
        return "native_21_40"
    return "native_41_plus"


def infer_domain(name: str, description: str) -> str:
    text = f"{name} {description}".lower()
    scores = {
        domain: sum(text.count(keyword) for keyword in keywords)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=lambda key: (scores[key], key))
    return best if scores[best] else "general"


def topology_label(branch_count: int, join_count: int, cyclic: bool) -> str:
    if cyclic:
        return "iteration_or_cycle"
    if branch_count == 0 and join_count == 0:
        return "sequence_or_parallel_sequences"
    if branch_count == 1 and join_count == 0:
        return "single_branch"
    if branch_count == 1:
        return "branch_join"
    if join_count:
        return "multi_branch_join"
    return "multi_branch"


def analyze_workflow(
    workflow: dict[str, Any],
    registry: dict[str, Any],
    *,
    source_kind: str,
    source_id: str,
    name: str = "",
    description: str = "",
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes = workflow.get("nodes") or []
    if not isinstance(nodes, list):
        nodes = []
    valid_nodes = [node for node in nodes if isinstance(node, dict)]
    names = [str(node.get("name", "")) for node in valid_nodes]
    name_counts = Counter(names)
    duplicate_names = sorted(name for name, count in name_counts.items() if name and count > 1)
    node_classes: dict[str, str] = {}
    type_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    for node in valid_nodes:
        node_name = str(node.get("name", ""))
        node_type = str(node.get("type", ""))
        node_class = classify_node_type(node_type, registry)
        node_classes[node_name] = node_class
        type_counts[node_type] += 1
        class_counts[node_class] += 1

    main_edges, dependency_edges = parse_connections(workflow)
    known_names = set(names)
    resolved_main = [
        edge for edge in main_edges
        if edge["source"] in known_names and edge["target"] in known_names
    ]
    unresolved_main = [edge for edge in main_edges if edge not in resolved_main]
    main_resolution = len(resolved_main) / len(main_edges) if main_edges else 0.0

    cross_main_edges = [
        {
            **edge,
            "source_class": node_classes.get(edge["source"]),
            "target_class": node_classes.get(edge["target"]),
            "direction": (
                "semantic_to_deterministic"
                if node_classes.get(edge["source"]) == "semantic_operation"
                else "deterministic_to_semantic"
            ),
        }
        for edge in resolved_main
        if {node_classes.get(edge["source"]), node_classes.get(edge["target"])}
        & {"semantic_operation"}
        and (
            node_classes.get(edge["source"])
            in {"code_operation", "control_operation", "deterministic_transform"}
            or node_classes.get(edge["target"])
            in {"code_operation", "control_operation", "deterministic_transform"}
        )
    ]

    named_data_edges: set[tuple[str, str]] = set()
    for node in valid_nodes:
        target = str(node.get("name", ""))
        for source in extract_named_references(node.get("parameters", {})):
            if source in known_names:
                named_data_edges.add((source, target))
    cross_named_data_edges = [
        {
            "source": source,
            "target": target,
            "source_class": node_classes.get(source),
            "target_class": node_classes.get(target),
            "direction": (
                "semantic_to_deterministic"
                if node_classes.get(source) == "semantic_operation"
                else "deterministic_to_semantic"
            ),
        }
        for source, target in sorted(named_data_edges)
        if {node_classes.get(source), node_classes.get(target)} & {"semantic_operation"}
        and (
            node_classes.get(source)
            in {"code_operation", "control_operation", "deterministic_transform"}
            or node_classes.get(target)
            in {"code_operation", "control_operation", "deterministic_transform"}
        )
    ]

    execution_names = {
        node_name for node_name, node_class in node_classes.items()
        if node_class not in {"ignored_annotation", "model_dependency", "tool_dependency"}
    }
    out_degree: Counter[str] = Counter()
    in_degree: Counter[str] = Counter()
    for edge in resolved_main:
        if edge["source"] in execution_names and edge["target"] in execution_names:
            out_degree[edge["source"]] += 1
            in_degree[edge["target"]] += 1
    branch_count = sum(degree > 1 for degree in out_degree.values())
    join_count = sum(degree > 1 for degree in in_degree.values())
    cyclic = has_cycle(execution_names, resolved_main)
    topology = topology_label(branch_count, join_count, cyclic)
    deterministic_count = sum(
        class_counts[class_name]
        for class_name in ("code_operation", "control_operation", "deterministic_transform")
    )
    semantic_count = class_counts["semantic_operation"]
    coupled = bool(cross_main_edges or cross_named_data_edges)
    formal_description = len(description.strip()) >= 120
    official = source_kind == "n8n_official"
    strict_candidate = bool(
        official
        and semantic_count >= 1
        and deterministic_count >= 1
        and coupled
        and main_edges
        and not unresolved_main
        and not duplicate_names
        and formal_description
        and 4 <= len(execution_names) <= 80
    )
    mirror_discovery_candidate = bool(
        source_kind == "n8n_hf_mirror"
        and semantic_count >= 1
        and deterministic_count >= 1
        and coupled
        and main_edges
        and main_resolution >= 0.98
    )

    return {
        "source_kind": source_kind,
        "source_id": source_id,
        "name": name or str(workflow.get("name", "")),
        "description_length": len(description),
        "description_sha256": sha256_text(description) if description else None,
        "workflow_sha256": sha256_text(canonical_json(workflow)),
        "provenance": provenance or {},
        "node_count_raw": len(valid_nodes),
        "node_count_execution": len(execution_names),
        "native_node_count_band": size_band(len(execution_names)),
        "node_class_counts": dict(sorted(class_counts.items())),
        "node_type_counts": dict(sorted(type_counts.items())),
        "semantic_operation_count": semantic_count,
        "code_operation_count": class_counts["code_operation"],
        "control_operation_count": class_counts["control_operation"],
        "deterministic_transform_count": class_counts["deterministic_transform"],
        "deterministic_task_count": deterministic_count,
        "model_dependency_count": class_counts["model_dependency"],
        "tool_dependency_count": class_counts["tool_dependency"],
        "main_edge_count": len(main_edges),
        "main_edge_resolution_rate": round(main_resolution, 6),
        "unresolved_main_edge_count": len(unresolved_main),
        "dependency_edge_count": len(dependency_edges),
        "direct_semantic_deterministic_main_edges": len(cross_main_edges),
        "direct_cross_main_edge_examples": cross_main_edges[:20],
        "semantic_to_deterministic_main_edges": sum(
            edge["direction"] == "semantic_to_deterministic" for edge in cross_main_edges
        ),
        "deterministic_to_semantic_main_edges": sum(
            edge["direction"] == "deterministic_to_semantic" for edge in cross_main_edges
        ),
        "named_data_reference_edges": len(named_data_edges),
        "cross_type_named_data_edges": cross_named_data_edges,
        "branch_count": branch_count,
        "join_count": join_count,
        "has_cycle": cyclic,
        "weak_component_count": weak_component_count(execution_names, resolved_main),
        "native_topology": topology,
        "domain_heuristic": infer_domain(name or str(workflow.get("name", "")), description),
        "duplicate_node_names": duplicate_names,
        "coupled_semantic_deterministic": coupled,
        "mirror_discovery_candidate": mirror_discovery_candidate,
        "strict_official_pilot_candidate": strict_candidate,
        "oracle_feasibility": "requires_manual_review",
        "gold_status": "not_gold",
    }


def shortlist(candidates: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    """Deterministic greedy coverage selection without reading system outputs."""
    if target <= 0:
        return []
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    covered: set[tuple[str, str]] = set()

    def obligations(item: dict[str, Any]) -> set[tuple[str, str]]:
        values = {
            "domain": item["domain_heuristic"],
            "size": item["native_node_count_band"],
            "topology": item["native_topology"],
            "explicit_code": "yes" if item["code_operation_count"] else "no",
            "control": "yes" if item["control_operation_count"] else "no",
            "cross_main": "two_plus" if item["direct_semantic_deterministic_main_edges"] >= 2 else "one_or_named",
            "cross_direction": (
                "bidirectional"
                if item["semantic_to_deterministic_main_edges"] and item["deterministic_to_semantic_main_edges"]
                else "semantic_to_deterministic"
                if item["semantic_to_deterministic_main_edges"]
                else "deterministic_to_semantic"
            ),
        }
        return set(values.items())

    while remaining and len(selected) < target:
        ranked = sorted(
            remaining,
            key=lambda item: (
                -len(obligations(item) - covered),
                -min(item["direct_semantic_deterministic_main_edges"], 3),
                -int(item["code_operation_count"] > 0),
                sha256_text(f"n8n-native-pilot-v1:{item['source_id']}"),
            ),
        )
        chosen = ranked[0]
        remaining.remove(chosen)
        selected.append(chosen)
        covered.update(obligations(chosen))
    return selected


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "workflow_count": len(records),
        "semantic_workflows": sum(item["semantic_operation_count"] > 0 for item in records),
        "deterministic_workflows": sum(item["deterministic_task_count"] > 0 for item in records),
        "broad_hybrid_workflows": sum(
            item["semantic_operation_count"] > 0 and item["deterministic_task_count"] > 0
            for item in records
        ),
        "coupled_hybrid_workflows": sum(item["coupled_semantic_deterministic"] for item in records),
        "direct_cross_main_workflows": sum(
            item["direct_semantic_deterministic_main_edges"] > 0 for item in records
        ),
        "explicit_code_hybrid_workflows": sum(
            item["semantic_operation_count"] > 0 and item["code_operation_count"] > 0
            for item in records
        ),
        "strict_official_pilot_candidates": sum(item["strict_official_pilot_candidate"] for item in records),
        "main_edges": sum(item["main_edge_count"] for item in records),
        "dependency_edges": sum(item["dependency_edge_count"] for item in records),
        "resolved_main_edges": sum(
            item["main_edge_count"] - item["unresolved_main_edge_count"] for item in records
        ),
        "node_count_execution": {
            "min": min((item["node_count_execution"] for item in records), default=0),
            "median": statistics.median(item["node_count_execution"] for item in records) if records else 0,
            "max": max((item["node_count_execution"] for item in records), default=0),
        },
        "domain_heuristic_counts": dict(sorted(Counter(item["domain_heuristic"] for item in records).items())),
        "native_size_band_counts": dict(sorted(Counter(item["native_node_count_band"] for item in records).items())),
        "native_topology_counts": dict(sorted(Counter(item["native_topology"] for item in records).items())),
    }


def render_report(title: str, source_note: str, summary: dict[str, Any], shortlist_count: int) -> str:
    edges = summary["main_edges"]
    resolved = summary["resolved_main_edges"]
    rate = resolved / edges if edges else 0.0
    return f"""# {title}

> 生成时间：{datetime.now(timezone.utc).isoformat()}  
> 节点分类版本：n8n-node-classification-v1

## 数据边界

{source_note}

自动分类仅用于候选发现，不是Gold。正式节点类型、CFG和DFG仍需双人独立复核与仲裁。

## 汇总

| 项目 | 数量 |
|---|---:|
| 去重工作流 | {summary['workflow_count']} |
| 含实际语义操作 | {summary['semantic_workflows']} |
| 含确定性任务 | {summary['deterministic_workflows']} |
| 语义与确定性任务共现 | {summary['broad_hybrid_workflows']} |
| 两类任务存在主边或命名变量耦合 | {summary['coupled_hybrid_workflows']} |
| 存在直接跨类型主执行边 | {summary['direct_cross_main_workflows']} |
| 语义操作与显式Code共现 | {summary['explicit_code_hybrid_workflows']} |
| 通过官方试标候选自动门禁 | {summary['strict_official_pilot_candidates']} |
| 主执行边端点解析率 | {rate:.2%} ({resolved}/{edges}) |
| 当前短名单 | {shortlist_count} |

## 解释限制

- `broad_hybrid`只表示节点类型共现，不自动证明两类节点形成完整逻辑链；
- `coupled_hybrid`要求主执行边或显式命名变量引用，但变量类型仍需人工确认；
- 本地镜像中的通用提示语不是原始用户需求，不能作为正式公共输入；
- `domain_heuristic`和结构标签只用于分层抽样，不进入论文Gold指标；
- 候选筛选完全不读取NCNLP或baseline输出。
"""


def render_shortlist_report(selected: list[dict[str, Any]], eligible_count: int) -> str:
    summary = summarize(selected)
    sem_to_det = sum(item["semantic_to_deterministic_main_edges"] > 0 for item in selected)
    det_to_sem = sum(item["deterministic_to_semantic_main_edges"] > 0 for item in selected)
    bidirectional = sum(
        item["semantic_to_deterministic_main_edges"] > 0
        and item["deterministic_to_semantic_main_edges"] > 0
        for item in selected
    )

    def rows(values: dict[str, int]) -> str:
        return "\n".join(f"| {key} | {count} |" for key, count in values.items())

    expected_domains = set(DOMAIN_KEYWORDS) | {"general"}
    missing_domains = sorted(expected_domains - set(summary["domain_heuristic_counts"]))
    return f"""# n8n 原生工作流试标短名单报告

> 状态：人工来源复核前的自动短名单，不是 Gold。

## 形成过程

80 条完整官方图中有 {eligible_count} 条通过严格自动准入。短名单使用固定哈希打破平局，并以领域启发式、原生节点数、拓扑、显式 Code、控制节点和跨类型方向做贪心覆盖，确定性选出 {len(selected)} 条。选择过程没有读取 NCNLP 或 baseline 输出。

## 核心结构

| 项目 | 工作流数 |
|---|---:|
| 语义输出进入确定性节点 | {sem_to_det} |
| 确定性结果进入语义节点 | {det_to_sem} |
| 两个方向均存在 | {bidirectional} |
| 含显式 Code 节点 | {sum(item['code_operation_count'] > 0 for item in selected)} |
| 含分支、汇合或循环候选 | {sum(item['native_topology'] != 'sequence_or_parallel_sequences' for item in selected)} |

## 领域启发式分布

| 自动领域线索 | 数量 |
|---|---:|
{rows(summary['domain_heuristic_counts'])}

## 原生节点规模

| 原生执行节点带 | 数量 |
|---|---:|
{rows(summary['native_size_band_counts'])}

## 当前缺口

- 自动领域线索未覆盖：{', '.join(missing_domains) if missing_domains else '无'}；自动领域只用于候选分层，最终领域由人工复核。
- 5 条候选超过 40 个原生执行节点，试标时需要判断能否映射为不超过冻结范围的方法中立任务节点；不能为了入选任意合并。
- 官方说明普遍包含搭建和凭据细节，公共输入必须逐字选择行为段落，不能由研究者改写。
- Oracle、许可边界和原始 JSON 再分发状态尚未通过人工门禁。

## 下一门禁

两名复核者独立完成 `manual_review_queue.jsonl`。只有公共行为输入、真实混合操作、跨类型业务耦合和独立 Oracle 均有证据的记录，才可进入 12 至 20 条 Gold 试点。
"""


def load_parquet_rows(path: Path) -> Iterable[dict[str, str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise SystemExit(
            "analyze-mirror requires pyarrow; use the bundled workspace Python runtime"
        ) from exc
    table = parquet.read_table(path, columns=["key", "value"])
    yield from table.to_pylist()


def analyze_mirror(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    by_hash: dict[str, dict[str, Any]] = {}
    all_value_hashes: set[str] = set()
    malformed_hashes: set[str] = set()
    malformed_rows = 0
    global_prompt_keys: set[str] = set()
    for row in load_parquet_rows(args.mirror):
        raw_value = row.get("value") or ""
        raw_digest = sha256_text(raw_value)
        all_value_hashes.add(raw_digest)
        if row.get("key"):
            global_prompt_keys.add(row["key"])
        try:
            workflow = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            malformed_rows += 1
            malformed_hashes.add(raw_digest)
            continue
        if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
            malformed_rows += 1
            malformed_hashes.add(raw_digest)
            continue
        digest = sha256_text(canonical_json(workflow))
        entry = by_hash.setdefault(digest, {"workflow": workflow, "prompts": set()})
        if row.get("key"):
            entry["prompts"].add(row["key"])

    records: list[dict[str, Any]] = []
    for digest, entry in sorted(by_hash.items()):
        workflow = entry["workflow"]
        meta = workflow.get("meta") or {}
        template_id = meta.get("templateId") if isinstance(meta, dict) else None
        record = analyze_workflow(
            workflow,
            registry,
            source_kind="n8n_hf_mirror",
            source_id=f"mirror:{digest[:16]}",
            name=str(workflow.get("name", "")),
            provenance={
                "mirror_workflow_sha256": digest,
                "prompt_variant_count": len(entry["prompts"]),
                "official_template_id_clue": template_id,
                "formal_requirement_text_available": False,
            },
        )
        records.append(record)

    candidates = [item for item in records if item["mirror_discovery_candidate"]]
    selected = shortlist(candidates, args.pilot_size)
    out = args.out
    write_jsonl(out / "mirror_inventory.jsonl", records)
    write_jsonl(out / "mirror_discovery_candidates.jsonl", candidates)
    write_json(
        out / "mirror_discovery_shortlist.json",
        {
            "status": "discovery_only_not_gold",
            "selection_algorithm": "deterministic greedy structural coverage; no system outputs",
            "requested_size": args.pilot_size,
            "selected_size": len(selected),
            "items": selected,
        },
    )
    summary = summarize(records)
    summary["unique_json_values_total"] = len(all_value_hashes)
    summary["valid_unique_workflows"] = len(by_hash)
    summary["malformed_or_non_workflow_unique_values"] = len(malformed_hashes)
    summary["malformed_or_non_workflow_rows"] = malformed_rows
    summary["unique_prompt_keys"] = len(global_prompt_keys)
    write_json(out / "reports" / "mirror_audit_report.json", summary)
    (out / "reports" / "mirror_audit_report.md").write_text(
        render_report(
            "n8n本地镜像候选发现审计",
            "该镜像用于结构发现和分类器校准。只有能够回溯到n8n官方模板ID、说明和版本的工作流，才可进入正式试标。",
            summary,
            len(selected),
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def request_json(url: str, *, attempts: int, timeout: int, delay: float) -> dict[str, Any]:
    headers = {"User-Agent": "NCNLP-FSE-research-audit/1.0"}
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # urllib errors vary across Windows TLS stacks
            errors.append(f"urllib attempt {attempt}: {type(exc).__name__}: {exc}")
        try:
            result = subprocess.run(
                [
                    "curl.exe", "--fail", "--silent", "--show-error", "--location",
                    "--max-time", str(timeout), "--user-agent", headers["User-Agent"], url,
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
            errors.append(f"curl attempt {attempt}: {result.stderr.strip()}")
        except Exception as exc:
            errors.append(f"curl attempt {attempt}: {type(exc).__name__}: {exc}")
        if attempt < attempts:
            time.sleep(delay * (2 ** (attempt - 1)))
    raise RuntimeError(f"failed to fetch {url}: " + " | ".join(errors))


def compact_metadata(item: dict[str, Any], retrieved_at: str) -> dict[str, Any]:
    node_types = sorted({
        str(node.get("name")) for node in item.get("nodes", [])
        if isinstance(node, dict) and node.get("name")
    })
    categories = []
    for category in item.get("categories", []) or []:
        if isinstance(category, dict):
            categories.append(str(category.get("name") or category.get("displayName") or category.get("id")))
        else:
            categories.append(str(category))
    user = item.get("user") if isinstance(item.get("user"), dict) else {}
    description = str(item.get("description") or "")
    return {
        "template_id": int(item["id"]),
        "name": str(item.get("name") or ""),
        "created_at": item.get("createdAt"),
        "total_views": item.get("totalViews"),
        "author": {
            "user_id": user.get("id"),
            "username": user.get("username"),
            "verified": user.get("verified"),
        },
        "categories": sorted(categories),
        "description": description,
        "description_sha256": sha256_text(description),
        "node_catalog_types": node_types,
        "official_api_url": f"{API_ROOT}/workflows/{item['id']}",
        "retrieved_at": retrieved_at,
    }


def build_manual_review_record(
    item: dict[str, Any], metadata_by_id: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    template_id = int(item["source_id"].split(":", 1)[1])
    metadata = metadata_by_id[template_id]
    return {
        "review_id": f"N8N-SOURCE-{template_id}",
        "source_id": item["source_id"],
        "official_api_url": metadata["official_api_url"],
        "name": item["name"],
        "author": metadata["author"],
        "created_at": metadata["created_at"],
        "retrieved_at": metadata["retrieved_at"],
        "public_description_verbatim": metadata["description"],
        "public_description_sha256": metadata["description_sha256"],
        "workflow_sha256": item["workflow_sha256"],
        "automatic_discovery_summary": {
            "warning": "Candidate discovery only; not Gold.",
            "node_count_execution": item["node_count_execution"],
            "semantic_operation_count": item["semantic_operation_count"],
            "code_operation_count": item["code_operation_count"],
            "control_operation_count": item["control_operation_count"],
            "deterministic_transform_count": item["deterministic_transform_count"],
            "native_topology": item["native_topology"],
            "direct_cross_main_edge_examples": item["direct_cross_main_edge_examples"],
            "cross_type_named_data_edges": item["cross_type_named_data_edges"],
        },
        "review": {
            "description_contains_complete_behavioral_goal": None,
            "behavioral_input_can_be_extracted_verbatim_without_paraphrase": None,
            "implementation_detail_leakage": None,
            "semantic_and_deterministic_operations_are_genuine": None,
            "cross_type_coupling_is_behaviorally_meaningful": None,
            "external_dependencies_are_mockable": None,
            "independent_oracle_is_feasible": None,
            "license_and_redistribution_status_checked": None,
            "provisional_decision": None,
            "exclusion_reason_codes": [],
            "reviewer_id": None,
            "reviewed_at": None,
            "notes": ""
        }
    }


def metadata_prefilter(item: dict[str, Any], registry: dict[str, Any]) -> bool:
    classes = Counter(classify_node_type(node_type, registry) for node_type in item["node_catalog_types"])
    deterministic = sum(classes[name] for name in ("code_operation", "control_operation", "deterministic_transform"))
    item["catalog_node_class_counts"] = dict(sorted(classes.items()))
    item["catalog_prefilter_pass"] = classes["semantic_operation"] > 0 and deterministic > 0
    return item["catalog_prefilter_pass"]


def evenly_spaced_pages(total_pages: int, count: int) -> list[int]:
    if total_pages <= 0 or count <= 0:
        return []
    if count >= total_pages:
        return list(range(1, total_pages + 1))
    if count == 1:
        return [1]
    return sorted({1 + round(index * (total_pages - 1) / (count - 1)) for index in range(count)})


def fetch_official(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    out = args.out
    cache = out / "raw_official"
    cache.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    count_url = f"{API_ROOT}/search?{urllib.parse.urlencode({'rows': 1, 'page': 1, 'category': args.category})}"
    first = request_json(count_url, attempts=args.attempts, timeout=args.timeout, delay=args.delay)
    total = int(first["totalWorkflows"])
    total_pages = math.ceil(total / args.rows)
    pages = evenly_spaced_pages(total_pages, args.sample_pages)
    metadata_by_id: dict[int, dict[str, Any]] = {}
    page_log: list[dict[str, Any]] = []
    for page in pages:
        compact_path = cache / f"search_{args.category.lower()}_{args.rows}_{page}.compact.json"
        if compact_path.exists() and not args.force:
            compact = json.loads(compact_path.read_text(encoding="utf-8"))
        else:
            query = urllib.parse.urlencode({"rows": args.rows, "page": page, "category": args.category})
            response = request_json(
                f"{API_ROOT}/search?{query}", attempts=args.attempts,
                timeout=args.timeout, delay=args.delay,
            )
            compact = [compact_metadata(item, retrieved_at) for item in response.get("workflows", [])]
            write_json(compact_path, compact)
            time.sleep(args.delay)
        for item in compact:
            metadata_by_id[item["template_id"]] = item
        page_log.append({"page": page, "record_count": len(compact), "cache": str(compact_path.relative_to(ROOT))})

    metadata = sorted(metadata_by_id.values(), key=lambda item: item["template_id"])
    prefiltered = [item for item in metadata if metadata_prefilter(item, registry)]
    ordered = sorted(
        prefiltered,
        key=lambda item: sha256_text(f"n8n-official-detail-v1:{item['template_id']}"),
    )
    analyzed: list[dict[str, Any]] = []
    detail_errors: list[dict[str, Any]] = []
    for item in ordered[: args.detail_limit]:
        template_id = item["template_id"]
        raw_path = cache / f"workflow_{template_id}.json"
        try:
            if raw_path.exists() and not args.force:
                wrapper = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                wrapper = request_json(
                    item["official_api_url"], attempts=args.attempts,
                    timeout=args.timeout, delay=args.delay,
                )
                write_json(raw_path, wrapper)
                time.sleep(args.delay)
            detail = wrapper.get("workflow") or {}
            workflow = detail.get("workflow") or {}
            description = str(detail.get("description") or item["description"])
            record = analyze_workflow(
                workflow,
                registry,
                source_kind="n8n_official",
                source_id=f"n8n:{template_id}",
                name=str(detail.get("name") or item["name"]),
                description=description,
                provenance={
                    "template_id": template_id,
                    "author": item["author"],
                    "created_at": detail.get("createdAt") or item["created_at"],
                    "retrieved_at": retrieved_at,
                    "official_api_url": item["official_api_url"],
                    "raw_local_path": str(raw_path.relative_to(ROOT)),
                    "raw_response_sha256": sha256_text(canonical_json(wrapper)),
                    "redistribution_status": "metadata_and_derived_annotations_only_pending_terms_review",
                },
            )
            analyzed.append(record)
        except Exception as exc:
            detail_errors.append({"template_id": template_id, "error": f"{type(exc).__name__}: {exc}"})

    eligible = [item for item in analyzed if item["strict_official_pilot_candidate"]]
    selected = shortlist(eligible, args.pilot_size)
    write_jsonl(out / "official_sampling_frame.jsonl", metadata)
    write_jsonl(out / "official_catalog_prefilter.jsonl", prefiltered)
    write_jsonl(out / "official_analyzed_candidates.jsonl", analyzed)
    write_json(
        out / "pilot_shortlist.json",
        {
            "status": "candidate_for_double_annotation_not_gold",
            "selection_algorithm": "deterministic greedy structural coverage over strict official candidates; no system outputs",
            "requested_size": args.pilot_size,
            "selected_size": len(selected),
            "items": selected,
        },
    )
    review_records = [build_manual_review_record(item, metadata_by_id) for item in selected]
    write_jsonl(out / "manual_review_queue.jsonl", review_records)
    write_json(
        out / "manual_review_queue_manifest.json",
        {
            "status": "awaiting_independent_source_review",
            "record_count": len(review_records),
            "selection_input": "pilot_shortlist.json",
            "review_guideline": "n8n_manual_source_review.md",
            "selection_does_not_use_system_outputs": True,
            "queue_sha256": sha256_text("\n".join(canonical_json(item) for item in review_records) + "\n"),
        },
    )
    selected_summary = summarize(selected)
    selected_summary.update({
        "eligible_input_count": len(eligible),
        "selected_count": len(selected),
        "semantic_to_deterministic_workflows": sum(
            item["semantic_to_deterministic_main_edges"] > 0 for item in selected
        ),
        "deterministic_to_semantic_workflows": sum(
            item["deterministic_to_semantic_main_edges"] > 0 for item in selected
        ),
        "bidirectional_workflows": sum(
            item["semantic_to_deterministic_main_edges"] > 0
            and item["deterministic_to_semantic_main_edges"] > 0
            for item in selected
        ),
    })
    write_json(out / "reports" / "pilot_shortlist_report.json", selected_summary)
    (out / "reports" / "pilot_shortlist_report.md").write_text(
        render_shortlist_report(selected, len(eligible)), encoding="utf-8"
    )
    summary = summarize(analyzed)
    summary.update({
        "official_total_in_category_at_start": total,
        "category": args.category,
        "rows_per_page": args.rows,
        "total_pages": total_pages,
        "sampled_pages": pages,
        "sampling_frame_count": len(metadata),
        "catalog_prefilter_count": len(prefiltered),
        "detail_target": args.detail_limit,
        "detail_success_count": len(analyzed),
        "detail_errors": detail_errors,
        "pilot_requested": args.pilot_size,
        "pilot_selected": len(selected),
        "gate_status": "PASS" if len(selected) >= args.pilot_size and not detail_errors else "NOT_READY",
        "page_log": page_log,
    })
    write_json(out / "reports" / "official_candidate_report.json", summary)
    (out / "reports" / "official_candidate_report.md").write_text(
        render_report(
            "n8n官方原生混合工作流候选审计",
            f"官方AI分类在开始时报告{total}条模板。本报告使用预注册的等距页抽样建立候选发现框，不用于估计平台总体分布。正式试标短名单只来自已下载完整官方图且通过严格自动门禁的记录。",
            summary,
            len(selected),
        )
        + f"\n## 门禁\n\n当前状态：**{summary['gate_status']}**。详情下载错误：{len(detail_errors)}。\n",
        encoding="utf-8",
    )
    write_json(
        out / "official_retrieval_manifest.json",
        {
            "retrieved_at": retrieved_at,
            "api_root": API_ROOT,
            "registry_version": registry["version"],
            "sampling_rule": "equally spaced pages across the official AI category; candidate discovery only",
            "parameters": {
                "category": args.category,
                "rows": args.rows,
                "sample_pages": args.sample_pages,
                "detail_limit": args.detail_limit,
                "pilot_size": args.pilot_size,
            },
            "page_log": page_log,
            "raw_detail_files": sorted(
                {
                    item["provenance"]["raw_local_path"]: item["provenance"]["raw_response_sha256"]
                    for item in analyzed
                }.items()
            ),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["gate_status"] == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    mirror = subparsers.add_parser("analyze-mirror", help="Audit the local HF mirror for discovery only")
    mirror.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR)
    mirror.add_argument("--pilot-size", type=int, default=40)
    mirror.set_defaults(func=analyze_mirror)

    official = subparsers.add_parser("fetch-official", help="Build and analyze an official API sampling frame")
    official.add_argument("--category", default="AI")
    official.add_argument("--rows", type=int, default=50)
    official.add_argument("--sample-pages", type=int, default=12)
    official.add_argument("--detail-limit", type=int, default=80)
    official.add_argument("--pilot-size", type=int, default=40)
    official.add_argument("--attempts", type=int, default=4)
    official.add_argument("--timeout", type=int, default=60)
    official.add_argument("--delay", type=float, default=1.0)
    official.add_argument("--force", action="store_true")
    official.set_defaults(func=fetch_official)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out = args.out.resolve()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
