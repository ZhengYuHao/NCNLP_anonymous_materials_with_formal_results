#!/usr/bin/env python3
"""Shared, evaluator-blind utilities for formal RQ2 system adapters."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


FORMAL = Path(__file__).resolve().parents[1]
EXPERIMENT = FORMAL.parent
BENCHMARK = EXPERIMENT / "benchmark_v2"
PROJECT = EXPERIMENT.parent
for path in (PROJECT, EXPERIMENT, BENCHMARK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from n8n_h1_unified_tasks import adapt_to_legacy_ncnlp, load_system_view
from n8n_s1_unified_tasks import load_system_view as load_s1_system_view
from llm_client import create_llm_client


FORMAL_MODEL_ID = "gemini-3.8-flash"
FORMAL_TEMPERATURE = 0.0
FORMAL_MAX_OUTPUT_TOKENS = 8192
FORMAL_TIMEOUT_SECONDS = 360
FORMAL_MAX_ATTEMPTS = 3


def create_formal_llm_client():
    """Create the single uncached client configuration used by formal runs."""
    configured_model = os.getenv("LLM_MODEL", FORMAL_MODEL_ID)
    if configured_model != FORMAL_MODEL_ID:
        raise RuntimeError(
            f"formal model mismatch: expected {FORMAL_MODEL_ID}, got {configured_model}"
        )
    return create_llm_client(
        model=FORMAL_MODEL_ID,
        temperature=FORMAL_TEMPERATURE,
        max_tokens=FORMAL_MAX_OUTPUT_TOKENS,
        timeout=FORMAL_TIMEOUT_SECONDS,
        max_retries=FORMAL_MAX_ATTEMPTS,
        enable_cache=False,
    )


def load_public_case(task_id: str, case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    view = (load_s1_system_view if task_id.startswith("N8F-") else load_system_view)(task_id, case_id)
    adapted = adapt_to_legacy_ncnlp(view)
    if task_id.startswith("N8F-"):
        adapted.update(source="n8n_s1_unified_v3", difficulty="development_qualification")
    return view, adapted


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = usage.get("cached_tokens", prompt_details.get("cached_tokens", 0))
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
        "completion_tokens": int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "cached_tokens": int(cached_tokens or 0),
    }


def public_prompt(view: dict[str, Any]) -> str:
    task = view["task"]
    instance = view["instance"]
    return json.dumps(
        {
            "natural_language_requirement": task["natural_language_requirement"],
            "input_data": instance["input_data"],
            "public_interfaces": task["public_interfaces"],
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def make_record(
    *,
    system_id: str,
    task_id: str,
    case_id: str,
    repeat_index: int,
    status: str,
    actions: list[dict[str, Any]] | None = None,
    final_state: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    started_ns: int | None = None,
    attempt_count: int = 1,
    error_stage: str = "",
    error_type: str = "",
    error_message: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latency_ms = 0 if started_ns is None else max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
    return {
        "schema_version": "rq2-system-run-v1",
        "run_id": f"{system_id}:{task_id}:{case_id}:r{repeat_index}",
        "system_id": system_id,
        "task_id": task_id,
        "case_id": case_id,
        "repeat_index": repeat_index,
        "status": status,
        "actions": actions or [],
        "final_state": final_state or {},
        "usage": normalize_usage(usage),
        "latency_ms": int(latency_ms),
        "attempt_count": max(1, int(attempt_count)),
        "cache_enabled": False,
        "model_config": {
            "model_id": FORMAL_MODEL_ID,
            "temperature": FORMAL_TEMPERATURE,
            "max_output_tokens": FORMAL_MAX_OUTPUT_TOKENS,
            "timeout_seconds": FORMAL_TIMEOUT_SECONDS,
            "max_attempts": FORMAL_MAX_ATTEMPTS,
        },
        "error": {"stage": error_stage, "type": error_type, "message": error_message},
        "metadata": metadata or {},
    }
