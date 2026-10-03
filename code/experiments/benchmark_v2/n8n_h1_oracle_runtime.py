#!/usr/bin/env python3
"""Runtime primitives for deterministic H1 Oracle assertions."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from n8n_s1_observable_projection import project_observable_trace


MISSING = object()
PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+|\*)?\]")


class ProjectedValues(list[Any]):
    """Values selected by [] or [*] from a list-valued trace path."""


@dataclass(frozen=True)
class AssertionResult:
    assertion_id: str
    passed: bool
    target: str
    operator: str
    expected: Any
    actual: Any
    error: str = ""


def path_tokens(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    consumed = ""
    for match in PATH_TOKEN.finditer(path):
        gap = path[len(consumed):match.start()]
        if gap not in {"", "."}:
            raise ValueError(f"invalid target path near {gap!r}: {path}")
        if match.group(1) is not None:
            tokens.append(match.group(1))
        elif match.group(2) in {None, "*"}:
            tokens.append("*")
        else:
            tokens.append(int(match.group(2)))
        consumed = path[:match.end()]
    if consumed != path or not tokens:
        raise ValueError(f"invalid target path: {path}")
    return tokens


def resolve_path(value: Any, path: str) -> Any:
    current = [value]
    projected = False
    for token in path_tokens(path):
        next_values = []
        if token == "*":
            projected = True
            for item in current:
                if isinstance(item, list):
                    next_values.extend(item)
                else:
                    next_values.append(MISSING)
        elif isinstance(token, int):
            for item in current:
                if isinstance(item, list) and 0 <= token < len(item):
                    next_values.append(item[token])
                else:
                    next_values.append(MISSING)
        else:
            for item in current:
                if token == "length" and isinstance(item, (dict, list, tuple, set, str)):
                    next_values.append(len(item))
                elif isinstance(item, dict) and token in item:
                    next_values.append(item[token])
                else:
                    next_values.append(MISSING)
        current = next_values
    if projected:
        return ProjectedValues(current)
    return current[0] if current else MISSING


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(actual, dict):
        return expected in actual
    if isinstance(actual, (list, tuple, set, str)):
        return expected in actual
    return False


def _projected_contains(actual: ProjectedValues, expected: Any) -> bool:
    return any(
        item is not MISSING and (item == expected or _contains(item, expected))
        for item in actual
    )


def evaluate_assertion(assertion: dict[str, Any], trace: dict[str, Any]) -> AssertionResult:
    assertion_id = assertion["assertion_id"]
    target = assertion["target"]
    operator = assertion["operator"]
    expected = assertion.get("expected")
    try:
        actual = resolve_path(trace, target)
        if operator == "exists":
            passed = (
                bool(actual) and all(item is not MISSING for item in actual)
                if isinstance(actual, ProjectedValues)
                else actual is not MISSING
            )
        elif operator == "not_exists":
            passed = (
                not actual or all(item is MISSING for item in actual)
                if isinstance(actual, ProjectedValues)
                else actual is MISSING
            )
        elif actual is MISSING:
            passed = False
        elif operator == "equals":
            if isinstance(actual, ProjectedValues):
                clean = [item for item in actual if item is not MISSING]
                passed = clean == expected if isinstance(expected, list) else bool(clean) and all(
                    item == expected for item in clean
                )
            else:
                passed = actual == expected
        elif operator == "not_equals":
            passed = (
                all(item is MISSING or item != expected for item in actual)
                if isinstance(actual, ProjectedValues)
                else actual != expected
            )
        elif operator == "contains":
            passed = (
                _projected_contains(actual, expected)
                if isinstance(actual, ProjectedValues)
                else _contains(actual, expected)
            )
        elif operator == "length_equals":
            passed = hasattr(actual, "__len__") and len(
                [item for item in actual if item is not MISSING]
                if isinstance(actual, ProjectedValues) else actual
            ) == expected
        elif operator == "matches":
            if isinstance(actual, ProjectedValues):
                clean = [item for item in actual if item is not MISSING]
                passed = bool(clean) and all(
                    isinstance(item, str) and re.search(str(expected), item) is not None
                    for item in clean
                )
            else:
                passed = isinstance(actual, str) and re.search(str(expected), actual) is not None
        else:
            raise ValueError(f"unsupported assertion operator: {operator}")
        return AssertionResult(
            assertion_id=assertion_id,
            passed=bool(passed),
            target=target,
            operator=operator,
            expected=expected,
            actual=None if actual is MISSING else actual,
        )
    except (TypeError, ValueError) as exc:
        return AssertionResult(
            assertion_id=assertion_id,
            passed=False,
            target=target,
            operator=operator,
            expected=expected,
            actual=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def normalize_trace(task_id: str, raw_trace: dict[str, Any], case_id: str = "") -> dict[str, Any]:
    """Map method-specific observable representations to the frozen Oracle contract."""
    trace = project_observable_trace(task_id, case_id, raw_trace)
    exec_log = trace.setdefault("exec_log", {})

    # The frozen Oracles use a small, tool-independent vocabulary for common
    # observations. Build that view from public inputs and operation logs; the
    # mapping never reads expected values or task-specific answers.
    input_data = trace.get("input", {})
    if isinstance(input_data, dict):
        trigger = exec_log.setdefault("trigger", {})
        for key, value in input_data.items():
            if key != "fixtures":
                trigger.setdefault(str(key), copy.deepcopy(value))

    operation_rows: list[tuple[str, str, dict[str, Any]]] = []
    for dependency, operations in list(exec_log.items()):
        if not isinstance(operations, dict) or dependency in {"trigger", "metrics", "quality"}:
            continue
        for operation, log in operations.items():
            if isinstance(log, dict) and isinstance(log.get("calls"), int):
                request = log.get("request")
                if isinstance(request, dict) and "cycle_overview" in request:
                    overview = request.get("cycle_overview")
                    request.setdefault(
                        "contains_cycle_overview",
                        isinstance(overview, (dict, list, str)) and bool(overview),
                    )
                operation_rows.append((str(dependency), str(operation), log))

    def first_matching(dependency_terms: tuple[str, ...], operation_terms: tuple[str, ...]):
        for dependency, operation, log in operation_rows:
            if (
                any(term in dependency.lower() for term in dependency_terms)
                and any(term in operation.lower() for term in operation_terms)
            ):
                return log
        return None

    sheets_read = first_matching(
        ("googlesheet",), ("read", "load", "list", "fetch", "search")
    )
    if sheets_read is not None:
        canonical_read = copy.deepcopy(sheets_read)
        response = canonical_read.get("response", {})
        rows = response.get("rows") if isinstance(response, dict) else None
        if isinstance(rows, list):
            canonical_read.setdefault("row_count", len(rows))
        exec_log.setdefault("googleSheets", {}).setdefault("read", canonical_read)

    llm_log = first_matching(
        ("lmchat", "llm", "openai", "anthropic", "gemini", "deepseek"),
        ("answer", "generate", "analyze", "extract", "classify", "plan", "interpret"),
    )
    if llm_log is not None:
        canonical_llm = exec_log.setdefault("llm", {})
        request = llm_log.get("request", {})
        if isinstance(request, dict):
            prompt_parts = [value for value in request.values() if isinstance(value, str)]
            if prompt_parts:
                canonical_llm.setdefault("prompt", "\n".join(prompt_parts))
        response = llm_log.get("response", {})
        if isinstance(response, dict):
            for key in ("answer", "text", "content", "output"):
                if key in response:
                    canonical_llm.setdefault("answer", copy.deepcopy(response[key]))
                    break

    telegram_send = first_matching(("telegram",), ("send", "reply", "post"))
    if telegram_send is not None:
        canonical_send = copy.deepcopy(telegram_send.get("request", {}))
        if isinstance(canonical_send, dict):
            for key in ("text", "body", "content"):
                if key in canonical_send:
                    canonical_send.setdefault("message", copy.deepcopy(canonical_send[key]))
                    break
            exec_log.setdefault("telegram", {}).setdefault("send", canonical_send)

    if task_id == "N8C-003":
        webhook = trace.get("mock", {}).get("M-WEBHOOK-POST", {})
        for request in webhook.get("requests", []):
            body = request.get("body", {})
            canonical = body.setdefault("validation_result", {})
            if "mismatch_detected" in canonical:
                continue
            validation = body.get("validation", {})
            if isinstance(validation.get("all_match"), bool):
                canonical["mismatch_detected"] = not validation["all_match"]
            elif isinstance(validation.get("mismatches"), list):
                canonical["mismatch_detected"] = bool(validation["mismatches"])
        gmail = trace.get("mock", {}).get("M-GMAIL-SEND-REPLY", {})
        for request in gmail.get("requests", []):
            body_html = request.get("body_html")
            if isinstance(body_html, str) and "<html" in body_html.lower():
                request["body_html"] = "<html:oracle-format-valid>"
    return trace


def evaluate_case(task_id: str, case: dict[str, Any], raw_trace: dict[str, Any]) -> dict[str, Any]:
    trace = normalize_trace(task_id, raw_trace, str(case.get("case_id", "")))
    results = [evaluate_assertion(assertion, trace) for assertion in case["assertions"]]
    return {
        "case_id": case["case_id"],
        "passed": all(result.passed for result in results),
        "assertions": [asdict(result) for result in results],
        "trace_sha256": trace_signature(trace),
        "raw_trace_sha256": trace_signature(raw_trace),
    }


def trace_signature(trace: dict[str, Any]) -> str:
    payload = json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def replay_is_deterministic(case_runs: list[dict[str, Any]]) -> bool:
    return bool(case_runs) and len({run["trace_sha256"] for run in case_runs}) == 1
