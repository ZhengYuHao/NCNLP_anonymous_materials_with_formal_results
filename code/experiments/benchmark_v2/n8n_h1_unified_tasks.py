#!/usr/bin/env python3
"""Load the tested-system view of an H1 D5 task without evaluator leakage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BENCHMARK = Path(__file__).resolve().parent
DEFAULT_ROOT = BENCHMARK / "n8n_conversion_pilot" / "h1_execution" / "d5_unified_tasks_v2_1"
RUNTIME_CONTRACTS = BENCHMARK / "n8n_conversion_pilot" / "h1_execution" / "d5_runtime_contracts_v3"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_system_view(task_id: str, case_id: str, root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    task = read_json(root / "system_inputs" / task_id / "task.json")
    instance = read_json(root / "system_inputs" / task_id / "instances" / f"{case_id}.json")
    if task["task_id"] != task_id or instance["task_id"] != task_id or instance["case_id"] != case_id:
        raise ValueError("task/instance identity mismatch")
    config_path = RUNTIME_CONTRACTS / "system_inputs" / f"{task_id}.json"
    runtime_config = read_json(config_path).get("values", {}) if config_path.is_file() else {}
    return {"task": task, "instance": instance, "runtime_config": runtime_config}


def compose_method_prompt(system_view: dict[str, Any]) -> str:
    task = system_view["task"]
    interfaces = "\n".join(
        f"- {item['dependency_type']}::{item['operation']}\n"
        f"  request_schema={json.dumps(item.get('request_schema', {}), ensure_ascii=False, sort_keys=True)}\n"
        f"  response_schema={json.dumps(item.get('response_schema', {}), ensure_ascii=False, sort_keys=True)}"
        for item in task["public_interfaces"]
    )
    output_contract = task.get("public_output_contract")
    output_section = (
        "\n\nRequired public output contract:\n"
        + json.dumps(output_contract, ensure_ascii=False, sort_keys=True)
        if isinstance(output_contract, dict) else ""
    )
    return (
        f"{task['natural_language_requirement']}\n\n"
        "Available public operations (the same catalog is supplied to every method):\n"
        f"{interfaces}{output_section}"
    )


def adapt_to_legacy_ncnlp(system_view: dict[str, Any]) -> dict[str, Any]:
    """Use only the public system view; never attach Gold, Oracle, or expected output."""
    task = system_view["task"]
    instance = system_view["instance"]
    return {
        "id": task["task_id"],
        "domain": task["domain"],
        "difficulty": "h1_train_dev",
        "raw_prompt": task["natural_language_requirement"],
        "method_prompt": compose_method_prompt(system_view),
        "input_data": instance["input_data"],
        "public_interfaces": task["public_interfaces"],
        "public_runtime_config": dict(system_view.get("runtime_config", {})),
        "case_id": instance["case_id"],
        "input_sha256": instance["input_sha256"],
        "source": "n8n_h1_d5_unified_v2",
    }
