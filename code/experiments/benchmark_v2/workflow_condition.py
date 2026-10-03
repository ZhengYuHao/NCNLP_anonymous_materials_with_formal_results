"""Shared validation and evaluation for workflow condition trees."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any


LEAF_OPERATORS = {
    "exists",
    "not_exists",
    "eq",
    "neq",
    "contains",
    "not_contains",
}


def condition_leaves(condition: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield primitive predicates from a validated or candidate condition tree."""
    if not condition:
        return
    compound_keys = [key for key in ("all", "any", "not") if key in condition]
    if compound_keys:
        key = compound_keys[0]
        children = condition[key] if key in {"all", "any"} else [condition[key]]
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    yield from condition_leaves(child)
        return
    yield condition


def validate_condition_tree(condition: dict[str, Any]) -> None:
    """Validate the backward-compatible condition AST shape."""
    if not isinstance(condition, dict) or not condition:
        raise ValueError("condition must be a non-empty object")
    compound_keys = [key for key in ("all", "any", "not") if key in condition]
    if compound_keys:
        if len(compound_keys) != 1 or len(condition) != 1:
            raise ValueError("compound condition must contain exactly one of all, any, or not")
        key = compound_keys[0]
        children = condition[key]
        if key in {"all", "any"}:
            if not isinstance(children, list) or len(children) < 2:
                raise ValueError(f"{key} condition requires at least two children")
            for child in children:
                validate_condition_tree(child)
        else:
            if not isinstance(children, dict) or not children:
                raise ValueError("not condition requires one child object")
            validate_condition_tree(children)
        return

    operator = condition.get("operator")
    source = condition.get("source")
    if operator not in LEAF_OPERATORS:
        raise ValueError(f"unsupported leaf operator: {operator}")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("leaf condition requires a source")
    if operator not in {"exists", "not_exists"}:
        has_value = "value" in condition
        has_value_from = "value_from" in condition
        if has_value == has_value_from:
            raise ValueError("comparison condition requires exactly one of value or value_from")
        if has_value_from and not isinstance(condition.get("value_from"), str):
            raise ValueError("value_from must be a source string")


def evaluate_condition_tree(
    condition: dict[str, Any],
    resolve: Callable[[str], tuple[bool, Any]],
    contains: Callable[[Any, Any], bool],
) -> bool:
    """Evaluate a validated condition tree with short-circuit semantics."""
    validate_condition_tree(condition)
    if "all" in condition:
        return all(evaluate_condition_tree(child, resolve, contains) for child in condition["all"])
    if "any" in condition:
        return any(evaluate_condition_tree(child, resolve, contains) for child in condition["any"])
    if "not" in condition:
        return not evaluate_condition_tree(condition["not"], resolve, contains)

    found, source = resolve(condition["source"])
    operator = condition["operator"]
    if operator == "exists":
        return found and source is not None
    if operator == "not_exists":
        return not found or source is None
    if "value_from" in condition:
        value_found, value = resolve(condition["value_from"])
        if not value_found:
            return False
    else:
        value = condition.get("value")
    if not found:
        return False
    if operator == "eq":
        return source == value
    if operator == "neq":
        # In workflow contracts, comparison with an empty sentinel expresses
        # a non-empty gate. A missing/null derived value must not pass that
        # gate merely because Python considers None different from [] or "".
        if value in (None, "", [], {}):
            if source is None:
                return False
            # Generated contracts may use an empty collection sentinel for a
            # positive gate over a boolean validation result. Python treats
            # ``False != []`` as true, but the failed validation must close
            # the positive workflow branch.
            if isinstance(source, bool):
                return source
        return source != value
    if operator == "contains":
        return contains(source, value)
    return not contains(source, value)
