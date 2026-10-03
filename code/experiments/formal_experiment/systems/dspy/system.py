#!/usr/bin/env python3
"""Formal DSPy adapter with evaluator-blind public inputs and frozen program loading."""

from __future__ import annotations

import json
import os
import time
import typing
from pathlib import Path
from types import SimpleNamespace

from typing_extensions import NotRequired, Required

# LiteLLM 1.98 contains a delayed Python-3.11-style import on one provider path.
# Supplying the standard typing_extensions aliases keeps the Python 3.10 lock usable.
if not hasattr(typing, "NotRequired"):
    typing.NotRequired = NotRequired
if not hasattr(typing, "Required"):
    typing.Required = Required

import dspy
from dotenv import load_dotenv

from ..common import FORMAL_MODEL_ID, PROJECT, create_formal_llm_client, load_public_case, make_record, normalize_usage, public_prompt


FROZEN_PROGRAM = Path(__file__).with_name("frozen_program.json")


class WorkflowSignature(dspy.Signature):
    """Execute a workflow using only the public requirement, input, and interface catalog."""

    public_task_json: str = dspy.InputField(
        desc="JSON containing natural_language_requirement, input_data, and public_interfaces"
    )
    actions_json: str = dspy.OutputField(
        desc=(
            "JSON array of ordered public_operation actions; every action has type, "
            "dependency_type, operation, and arguments"
        )
    )
    final_state_json: str = dspy.OutputField(desc="JSON object containing only user-visible final state")


class DSPyWorkflow(dspy.Module):
    def __init__(self):
        super().__init__()
        self.execute = dspy.Predict(WorkflowSignature)

    def forward(self, public_task_json: str):
        return self.execute(public_task_json=public_task_json)


class UnifiedClientDSPyLM(dspy.BaseLM):
    """DSPy transport backed by the experiment's uncached OpenAI-compatible client."""

    forward_contract = "legacy"

    def __init__(self, model: str, client=None):
        super().__init__(model=model, temperature=0.0, max_tokens=8192, cache=False, num_retries=2)
        self.client = client or create_formal_llm_client()

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt or ""}]
        system_parts = [str(item.get("content", "")) for item in messages if item.get("role") == "system"]
        conversation_parts = [
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in messages if item.get("role") != "system"
        ]
        response = self.client.call(
            system_prompt="\n".join(system_parts) or "Follow the DSPy signature exactly.",
            user_content="\n\n".join(conversation_parts),
            temperature=0.0,
            max_tokens=8192,
        )
        message = SimpleNamespace(content=response.content, reasoning_content=None, tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
        return SimpleNamespace(
            choices=[choice], usage=response.usage or {}, model=response.model,
            _hidden_params={}, cache_hit=False,
        )


def build_lm():
    load_dotenv(PROJECT / ".env")
    model = os.getenv("LLM_MODEL", FORMAL_MODEL_ID)
    if model != FORMAL_MODEL_ID:
        raise RuntimeError(
            f"formal model mismatch: expected {FORMAL_MODEL_ID}, got {model}"
        )
    return UnifiedClientDSPyLM(model=f"openai/{model}")


def load_program() -> tuple[DSPyWorkflow, bool]:
    program = DSPyWorkflow()
    optimized = FROZEN_PROGRAM.is_file()
    if optimized:
        program.load(str(FROZEN_PROGRAM))
    return program, optimized


def aggregate_history_usage(history: list) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    for item in history:
        usage = item.get("usage", {}) if isinstance(item, dict) else getattr(item, "usage", {})
        normalized = normalize_usage(dict(usage or {}))
        for key in total:
            total[key] += normalized[key]
    return total


def run_case(task_id: str, case_id: str, repeat_index: int = 1, lm=None, program=None):
    started = time.perf_counter_ns()
    view, _ = load_public_case(task_id, case_id)
    lm = lm or build_lm()
    if program is None:
        program, optimized = load_program()
    else:
        optimized = FROZEN_PROGRAM.is_file()
    history_start = len(lm.history)
    try:
        with dspy.context(lm=lm):
            prediction = program(public_task_json=public_prompt(view))
        actions = json.loads(prediction.actions_json)
        final_state = json.loads(prediction.final_state_json)
        if not isinstance(actions, list) or not isinstance(final_state, dict):
            raise ValueError("DSPy outputs must decode to an actions array and final_state object")
        usage = aggregate_history_usage(lm.history[history_start:])
        return make_record(
            system_id="dspy", task_id=task_id, case_id=case_id,
            repeat_index=repeat_index, status="success", actions=actions,
            final_state=final_state, usage=usage, started_ns=started,
            metadata={"framework": f"dspy-{dspy.__version__}", "optimized_program_loaded": optimized},
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return make_record(
            system_id="dspy", task_id=task_id, case_id=case_id,
            repeat_index=repeat_index, status="parse_error",
            usage=aggregate_history_usage(lm.history[history_start:]), started_ns=started,
            error_stage="parse", error_type=type(exc).__name__, error_message=str(exc),
            metadata={"framework": f"dspy-{dspy.__version__}", "optimized_program_loaded": optimized},
        )
    except Exception as exc:
        return make_record(
            system_id="dspy", task_id=task_id, case_id=case_id,
            repeat_index=repeat_index, status="method_error",
            usage=aggregate_history_usage(lm.history[history_start:]), started_ns=started,
            error_stage="dspy_runtime", error_type=type(exc).__name__, error_message=str(exc),
            metadata={"framework": f"dspy-{dspy.__version__}", "optimized_program_loaded": optimized},
        )
