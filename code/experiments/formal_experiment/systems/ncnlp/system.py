#!/usr/bin/env python3
"""Formal NCNLP run adapter over the sealed D5 compiled artifacts."""

from __future__ import annotations

import time
from pathlib import Path

from ..common import create_formal_llm_client, load_public_case, make_record
from pipeline_runner import run_ncnlp_pipeline_run_only


LEGACY_FROZEN_ARTIFACTS = (
    Path(__file__).resolve().parents[3]
    / "benchmark_v2" / "n8n_conversion_pilot" / "h1_execution" / "runtime"
    / "frozen" / "d5_runtime_evidence_v1" / "artifacts"
)
DEVELOPMENT_FROZEN_ARTIFACTS = (
    Path(__file__).resolve().parents[2]
    / "reports" / "d33_development_29_final_engineering_regression_v1" / "tasks"
)
_LOCAL_ACTION_EXECUTORS = {}


def frozen_task_root(task_id: str) -> Path:
    if task_id.startswith("N8F-"):
        return DEVELOPMENT_FROZEN_ARTIFACTS / task_id / "artifacts" / task_id
    return LEGACY_FROZEN_ARTIFACTS / task_id


def consume_local_action_executor(run_id: str):
    """Return the in-memory DSL local executor without adding it to the JSON record."""
    return _LOCAL_ACTION_EXECUTORS.pop(run_id, None)


def load_frozen_compilation(task_id: str) -> dict:
    task_root = frozen_task_root(task_id)
    if not task_root.is_dir():
        raise ValueError(f"No frozen compilation directory for {task_id}: {task_root}")
    if task_id.startswith("N8F-"):
        candidates = [task_root] if (task_root / "generated.py").is_file() else []
    else:
        candidates = [
            path for path in task_root.iterdir()
            if path.is_dir() and (path / "generated.py").is_file()
        ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one frozen compilation for {task_id}, found {len(candidates)}")
    artifact = candidates[0]
    return {
        "backend": "dsl_compiler",
        "dsl_code": (artifact / "generated.dsl").read_text(encoding="utf-8"),
        "generated_python": (artifact / "generated.py").read_text(encoding="utf-8"),
        "sensor_specs": [],
        "classified_spec": None,
        "compile_tokens": 0,
        "compile_latency_ms": 0,
        "artifact_path": str(artifact),
    }


def run_case(task_id: str, case_id: str, repeat_index: int = 1, client=None):
    started = time.perf_counter_ns()
    _, task = load_public_case(task_id, case_id)
    client = client or create_formal_llm_client()
    try:
        compiled = load_frozen_compilation(task_id)
        result = run_ncnlp_pipeline_run_only(compiled, task, llm_client=client)
        status = "success" if result["ok"] else "execution_error"
        record = make_record(
            system_id="ncnlp", task_id=task_id, case_id=case_id,
            repeat_index=repeat_index, status=status, actions=result["actions"],
            final_state=result["final_state"], usage=result.get("runtime_usage"),
            started_ns=started, error_stage="runtime" if result["error"] else "",
            error_type="NCNLPRuntimeError" if result["error"] else "",
            error_message=result["error"],
            metadata={
                "compile_reused": True,
                "compile_tokens_charged_in_record": 0,
                "sensor_call_count": result["sensor_call_count"],
                "frozen_artifact_path": compiled["artifact_path"],
            },
        )
        _LOCAL_ACTION_EXECUTORS[record["run_id"]] = result.get("local_action_executor")
        return record
    except Exception as exc:
        return make_record(
            system_id="ncnlp", task_id=task_id, case_id=case_id,
            repeat_index=repeat_index, status="method_error", started_ns=started,
            error_stage="adapter", error_type=type(exc).__name__, error_message=str(exc),
        )
