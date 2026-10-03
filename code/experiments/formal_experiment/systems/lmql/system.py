#!/usr/bin/env python3
"""Formal LMQL adapter using the LMQL compiler and constrained decoder."""

from __future__ import annotations

import json
import os
import time
import asyncio
from typing import Any

from dotenv import load_dotenv

from ..common import FORMAL_MODEL_ID, PROJECT, create_formal_llm_client, load_public_case, make_record, normalize_usage, public_prompt


QUERY = r'''
argmax(max_len=12000)
    "<lmql:system/>Execute the public workflow. Output one JSON object with actions and final_state. Each action must be a public_operation with dependency_type, operation, and arguments. Use only listed interfaces and preserve order. After the complete JSON, write <END>.<lmql:user/>"
    "Public task:\n{task_payload}\nJSON result:[RESULT]" where len(TOKENS(RESULT)) < 8192 and STOPS_BEFORE(RESULT, "<END>")
    return RESULT
from model
'''


_PROVIDER_USAGE_LEDGER: list[dict[str, int]] = []
_PROVIDER_RESPONSE_LEDGER: list[str] = []


def provider_usage_since(start: int) -> dict[str, int]:
    usage = {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")}
    for item in _PROVIDER_USAGE_LEDGER[start:]:
        for key in usage:
            usage[key] += item[key]
    return usage


def token_stream_chunks(tokenizer, text: str) -> list[tuple[str, str]]:
    """Split complete text into the one-token chunks expected by LMQL's chat adapter."""
    token_ids = tokenizer(text)["input_ids"]
    token_bytes = tokenizer.decode_bytes(token_ids)
    encoded_tokens = []
    for item in token_bytes:
        raw = str(item)[2:-1]
        encoded_tokens.append(
            raw.encode("utf-8").decode("unicode_escape") if "\\x" not in raw else "bytes:" + raw
        )
    chunks: list[tuple[str, str]] = []
    pending = b""
    for index, item in enumerate(token_bytes):
        pending += item
        rendered = ""
        try:
            rendered = pending.decode("utf-8")
            pending = b""
        except UnicodeDecodeError:
            pass
        chunks.append((rendered, encoded_tokens[index]))
    if pending and chunks:
        rendered, encoded = chunks[-1]
        chunks[-1] = (rendered + pending.decode("utf-8", errors="replace"), encoded)
    return chunks


def configure_provider_endpoint():
    """Configure authentication for an OpenAI-compatible endpoint without changing LMQL semantics."""
    load_dotenv(PROJECT / ".env")
    api_key = os.getenv("LLM_API_KEY", "")
    api_base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    os.environ["OPENAI_API_KEY"] = api_key

    import lmql.runtime.caching as lmql_caching
    lmql_cache = PROJECT / "experiments" / "formal_experiment" / "reports" / ".lmql_runtime_cache"
    lmql_cache.mkdir(parents=True, exist_ok=True)
    lmql_caching.CACHE_DIR = lmql_cache

    import lmql.language.fragment_parser as fragment_parser
    import lmql.runtime.bopenai.openai_api as openai_api

    def compatible_remove_indentation(source: str, oneline: bool = False) -> str:
        """LMQL 0.7.3 fix: ignore indented continuation-only lines on Python 3.10.21."""
        source_lines = []
        for raw_line in source.split("\n"):
            line = raw_line.rstrip()
            if line.endswith("\\"):
                line = line[:-1].rstrip()
            if line.strip():
                source_lines.append(line)
        if not source_lines:
            return ""
        indent = min(len(line) - len(line.lstrip()) for line in source_lines)
        stripped = [line[indent:] for line in source_lines]
        return " \\\n".join(stripped) if oneline else "\n".join(stripped)

    fragment_parser.remove_indentation = compatible_remove_indentation
    original_ast_parse = fragment_parser.ast_parse

    def compatible_ast_parse(source, unindent=False, oneline=False, loc=None):
        if unindent and oneline:
            tokens = [fragment_parser.double_escape(token) for token in source]
            text = fragment_parser.untokenize_without_comments(tokens)
            text = compatible_remove_indentation(text, oneline=True)
            text = text.replace("\\\n", " ").strip().rstrip("\\").strip()
            import ast
            return ast.parse(text)
        return original_ast_parse(source, unindent=unindent, oneline=oneline, loc=loc)

    fragment_parser.ast_parse = compatible_ast_parse

    def endpoint_and_headers(kwargs: dict[str, Any]):
        kwargs.pop("api_config", None)
        model_name = kwargs.get("model", "")
        is_chat = openai_api.model_info(model_name).is_chat_model
        suffix = "chat/completions" if is_chat else "completions"
        return f"{api_base}/{suffix}", {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    openai_api.get_endpoint_and_headers = endpoint_and_headers


def configure_nonstream_transport_bridge():
    """Bridge a non-streaming compatible endpoint into LMQL's token stream contract."""
    import lmql.runtime.bopenai.openai_api as openai_api

    async def bridged_chat_api(**kwargs):
        api_config = kwargs.get("api_config", {})
        tokenizer = api_config.get("tokenizer")
        if tokenizer is None:
            raise RuntimeError("LMQL transport bridge requires its configured tokenizer")

        prompt = kwargs["prompt"][0]
        prompt_tokens = openai_api.tokenize(prompt, tokenizer=tokenizer, openai_byte_encoding=True)
        if kwargs.get("echo", False):
            yield {
                "choices": [{
                    "text": prompt,
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": {
                        "text_offset": [0 for _ in prompt_tokens],
                        "token_logprobs": [0.0 for _ in prompt_tokens],
                        "tokens": [str(token) for token in prompt_tokens],
                        "top_logprobs": [{str(token): 0.0} for token in prompt_tokens],
                    },
                }]
            }

        max_tokens = int(kwargs.get("max_tokens", 0))
        if max_tokens == 0:
            return
        if len(kwargs["prompt"]) != 1:
            raise RuntimeError("LMQL chat transport bridge does not support batched prompts")
        if "logit_bias" in kwargs:
            raise RuntimeError("Endpoint cannot preserve LMQL logit-bias constraints")

        segments = openai_api.tagged_segments(prompt)
        system_parts = [segment["text"] for segment in segments if segment["tag"] == "system"]
        conversation_parts = []
        for segment in segments:
            if segment["tag"] == "system" or not segment["text"]:
                continue
            role = segment["tag"] if segment["tag"] in {"user", "assistant"} else "user"
            conversation_parts.append(f"{role}: {segment['text']}")

        client = create_formal_llm_client()
        request = asyncio.create_task(asyncio.to_thread(
            client.call,
            system_prompt="\n".join(system_parts) or "Follow the LMQL query constraints exactly.",
            user_content="\n".join(conversation_parts),
            temperature=float(kwargs.get("temperature", 0.0)),
            max_tokens=max_tokens,
        ))
        while not request.done():
            done, _ = await asyncio.wait({request}, timeout=0.5)
            if not done:
                # LMQL's stream watchdog treats any received chunk as progress.
                yield {
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "finish_reason": None,
                        "logprobs": {
                            "text_offset": [],
                            "token_logprobs": [],
                            "tokens": [],
                            "top_logprobs": [],
                        },
                    }]
                }
        response = await request
        _PROVIDER_USAGE_LEDGER.append(normalize_usage(response.usage or {}))
        _PROVIDER_RESPONSE_LEDGER.append(response.content)

        content = response.content
        output_chunks = token_stream_chunks(tokenizer, content)
        for index, (chunk_text, token) in enumerate(output_chunks):
            yield {
                "choices": [{
                    "text": chunk_text,
                    "index": 0,
                    "finish_reason": "stop" if index == len(output_chunks) - 1 else None,
                    "logprobs": {
                        "text_offset": [0],
                        "token_logprobs": [0.0],
                        "tokens": [token],
                        "top_logprobs": [{token: 0.0}],
                    },
                }]
            }

    openai_api.chat_api = bridged_chat_api


def _result_text(result: Any) -> str:
    if isinstance(result, list):
        if not result:
            return ""
        result = result[0]
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("RESULT", ""))
    variables = getattr(result, "variables", {})
    return variables.get("RESULT", "") if isinstance(variables, dict) else str(result)


def parse_result_json(raw: str) -> tuple[dict[str, Any], dict[str, int]]:
    """Remove only LMQL query-delimiter residue, never repair JSON content."""
    candidate = raw.strip()
    removed_query_colon = 0
    if candidate.startswith(":"):
        candidate = candidate[1:].lstrip()
        removed_query_colon = 1
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("LMQL result must be a JSON object")
    return parsed, {"removed_query_delimiter_colon": removed_query_colon}


def normalize_actions(actions: Any) -> tuple[list[dict[str, Any]], int]:
    """Map LMQL's native action objects into the frozen public-operation envelope."""
    if not isinstance(actions, list):
        raise ValueError("LMQL result must contain an actions array")
    normalized = []
    inserted_types = 0
    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("Every LMQL action must be an object")
        if not all(key in action for key in ("dependency_type", "operation", "arguments")):
            raise ValueError("LMQL action is missing a required public-operation field")
        item = dict(action)
        if "type" not in item:
            item["type"] = "public_operation"
            inserted_types += 1
        if item["type"] != "public_operation":
            raise ValueError("LMQL action type must be public_operation")
        normalized.append(item)
    return normalized, inserted_types


def run_case(task_id: str, case_id: str, repeat_index: int = 1):
    started = time.perf_counter_ns()
    usage_start = len(_PROVIDER_USAGE_LEDGER)
    raw = ""
    view, _ = load_public_case(task_id, case_id)
    configure_provider_endpoint()
    configure_nonstream_transport_bridge()
    import lmql
    from lmql.runtime.bopenai import get_stats

    model_name = os.getenv("LLM_MODEL", FORMAL_MODEL_ID)
    if model_name != FORMAL_MODEL_ID:
        raise RuntimeError(
            f"formal model mismatch: expected {FORMAL_MODEL_ID}, got {model_name}"
        )
    model = lmql.model(f"openai/{model_name}", tokenizer="gpt-4o", chat_model=True)
    try:
        async def execute_query():
            stats = get_stats()
            from lmql.runtime.bopenai import _api
            if _api is not None:
                _api.maximum_retries = 1
            before_requests = int(stats.requests)
            query_result = await lmql.run(QUERY, task_payload=public_prompt(view), model=model)
            return (
                query_result,
                provider_usage_since(usage_start),
                max(1, int(stats.requests) - before_requests),
            )

        result, usage, requests = asyncio.run(execute_query())
        raw = _result_text(result)
        parsed, query_syntax_normalization = parse_result_json(raw)
        actions, inserted_types = normalize_actions(parsed.get("actions"))
        return make_record(
            system_id="lmql", task_id=task_id, case_id=case_id,
            repeat_index=repeat_index, status="success", actions=actions,
            final_state=parsed.get("final_state", {}),
            usage=usage, started_ns=started,
            attempt_count=requests,
            metadata={
                "framework": f"lmql-{getattr(lmql, '__version__', '0.7.3')}",
                "usage_scope": "Exact provider usage through the disclosed non-stream transport bridge",
                "provider_endpoint_adapter": True,
                "nonstream_transport_bridge": True,
                "syntax_normalization": {
                    "inserted_constant_public_operation_type": inserted_types,
                    **query_syntax_normalization,
                },
                "raw_response": raw,
            },
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return make_record(
            system_id="lmql", task_id=task_id, case_id=case_id,
            repeat_index=repeat_index, status="parse_error",
            usage=provider_usage_since(usage_start), started_ns=started,
            error_stage="parse", error_type=type(exc).__name__, error_message=str(exc),
            metadata={
                "framework": "lmql-0.7.3", "provider_endpoint_adapter": True,
                "nonstream_transport_bridge": True,
                "raw_response": raw,
                "provider_raw_response": (
                    _PROVIDER_RESPONSE_LEDGER[-1] if _PROVIDER_RESPONSE_LEDGER else ""
                ),
            },
        )
    except Exception as exc:
        return make_record(
            system_id="lmql", task_id=task_id, case_id=case_id,
            repeat_index=repeat_index, status="method_error",
            usage=provider_usage_since(usage_start), started_ns=started,
            error_stage="lmql_runtime", error_type=type(exc).__name__, error_message=str(exc),
            metadata={
                "framework": "lmql-0.7.3", "provider_endpoint_adapter": True,
                "nonstream_transport_bridge": True,
            },
        )
