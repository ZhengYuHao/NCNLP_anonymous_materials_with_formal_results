#!/usr/bin/env python3
"""Formal Direct-LLM adapter: one uncached call over the public task view."""

from __future__ import annotations

import time

from ..common import create_formal_llm_client, load_public_case, make_record, normalize_usage, public_prompt
from baselines.b0_direct import parse_json_output


SYSTEM_PROMPT = """Execute the supplied natural-language workflow for the supplied input.
Return one JSON object with exactly these top-level fields: actions and final_state.
Each action must call one listed public interface and have this shape:
{"type":"public_operation","dependency_type":"...","operation":"...","arguments":{...}}
Preserve required action order. Do not invent interfaces. Return JSON only."""


def run_case(task_id: str, case_id: str, repeat_index: int = 1, client=None):
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
        parsed = parse_json_output(response.content)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("actions"), list):
            return make_record(
                system_id="direct_llm", task_id=task_id, case_id=case_id,
                repeat_index=repeat_index, status="parse_error", usage=response.usage,
                started_ns=started, error_stage="parse", error_type="InvalidJSONEnvelope",
                error_message="Response must contain an actions array.",
                metadata={"raw_response": response.content},
            )
        return make_record(
            system_id="direct_llm", task_id=task_id, case_id=case_id,
            repeat_index=repeat_index, status="success", actions=parsed["actions"],
            final_state=parsed.get("final_state", {}), usage=normalize_usage(response.usage),
            started_ns=started, attempt_count=(response.usage or {}).get("attempt_count", 1),
            metadata={"raw_response": response.content},
        )
    except Exception as exc:
        return make_record(
            system_id="direct_llm", task_id=task_id, case_id=case_id,
            repeat_index=repeat_index, status="method_error", started_ns=started,
            error_stage="llm_call", error_type=type(exc).__name__, error_message=str(exc),
        )
