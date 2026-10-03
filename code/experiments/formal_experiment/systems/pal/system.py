#!/usr/bin/env python3
"""Formal PAL adapter: uncached code generation followed by restricted execution."""

from __future__ import annotations

import time

from ..common import create_formal_llm_client, load_public_case, make_record, public_prompt
from baselines.b3_pal import extract_python_code, safe_exec


SYSTEM_PROMPT = """Solve the supplied workflow by writing executable Python.
Use only json, math, decimal, re, datetime, time, or collections.
The code must print exactly one JSON object with actions and final_state.
Each action must have this shape:
{"type":"public_operation","dependency_type":"...","operation":"...","arguments":{...}}
Use only the listed public interfaces and preserve their required order.
Respond with one ```python code block and no placeholders."""


def run_case(task_id: str, case_id: str, repeat_index: int = 1, client=None, timeout_sec: int = 10):
    started = time.perf_counter_ns()
    view, _ = load_public_case(task_id, case_id)
    client = client or create_formal_llm_client()
    try:
        response = client.call(
            system_prompt=SYSTEM_PROMPT,
            user_content=public_prompt(view),
            temperature=0.0,
            max_tokens=8192,
        )
        code = extract_python_code(response.content)
        executed = safe_exec(code, timeout_sec=timeout_sec)
        if not executed.get("success"):
            message = executed.get("error", "PAL execution failed")
            status = "timeout" if "超时" in message else "execution_error"
            return make_record(
                system_id="pal", task_id=task_id, case_id=case_id,
                repeat_index=repeat_index, status=status, usage=response.usage,
                started_ns=started, error_stage="python_execution",
                error_type="PALExecutionError", error_message=message,
                metadata={"generated_code": code, "stdout": executed.get("stdout", "")},
            )
        parsed = executed.get("result")
        if not isinstance(parsed, dict) or not isinstance(parsed.get("actions"), list):
            return make_record(
                system_id="pal", task_id=task_id, case_id=case_id,
                repeat_index=repeat_index, status="parse_error", usage=response.usage,
                started_ns=started, error_stage="parse", error_type="InvalidPALResult",
                error_message="Executed program must print an object containing actions.",
                metadata={"generated_code": code, "stdout": executed.get("stdout", "")},
            )
        return make_record(
            system_id="pal", task_id=task_id, case_id=case_id,
            repeat_index=repeat_index, status="success", actions=parsed["actions"],
            final_state=parsed.get("final_state", {}), usage=response.usage,
            started_ns=started, attempt_count=(response.usage or {}).get("attempt_count", 1),
            metadata={"generated_code": code, "stdout": executed.get("stdout", "")},
        )
    except Exception as exc:
        return make_record(
            system_id="pal", task_id=task_id, case_id=case_id,
            repeat_index=repeat_index, status="method_error", started_ns=started,
            error_stage="llm_call", error_type=type(exc).__name__, error_message=str(exc),
        )
