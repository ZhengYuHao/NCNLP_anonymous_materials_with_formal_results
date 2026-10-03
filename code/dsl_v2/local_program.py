"""Validation and Python source generation for deterministic local DSL steps."""

from __future__ import annotations

import json
import re
from typing import Any


NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
STATEMENT_OPS = {"assign", "filter", "map", "sort", "emit"}
EXPRESSION_OPS = {
    "eq", "neq", "lt", "lte", "gt", "gte", "and", "or", "not",
    "in", "contains", "add", "sub", "mul", "div",
}
CALLS = {
    "len", "days_between", "date_add_days", "if_else", "count_true", "coalesce", "get", "join", "to_string",
    "to_markdown_table", "merge", "unique_by", "find_by",
}


class LocalProgramError(ValueError):
    pass


def normalize_local_program_syntax(program: Any) -> Any:
    """Canonicalize equivalent surface forms without adding semantics."""
    if not isinstance(program, list):
        return program

    def temporal_arithmetic(name: str, args: list[Any]) -> Any:
        if name not in {"add", "sub"} or len(args) != 2:
            return {"op": {"name": name, "args": args}}
        left, right = args
        left_ref = left.get("ref", "") if isinstance(left, dict) else ""
        right_ref = right.get("ref", "") if isinstance(right, dict) else ""
        date_like = any(token in str(left_ref).lower() for token in (
            "date", "period_start", "window_start", "window_end",
        ))
        days_like = (
            isinstance(right, (int, float)) and not isinstance(right, bool)
        ) or any(token in str(right_ref).lower() for token in (
            "day", "length", "interval",
        ))
        if not date_like or not days_like:
            return {"op": {"name": name, "args": args}}
        if name == "sub":
            right = {"op": {"name": "mul", "args": [right, -1]}}
        return {"call": {"name": "date_add_days", "args": [left, right]}}

    def expression(value: Any) -> Any:
        if isinstance(value, list):
            return [expression(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"op", "args"} and isinstance(value.get("op"), str):
            name = value["op"]
            args = value["args"]
            if isinstance(args, list) and name in EXPRESSION_OPS:
                return temporal_arithmetic(name, [expression(item) for item in args])
            if isinstance(args, list) and name in CALLS:
                return {"call": {"name": name, "args": [expression(item) for item in args]}}
        if set(value) == {"name", "args"} and value.get("name") in CALLS:
            args = value.get("args")
            if isinstance(args, list):
                return {"call": {"name": value["name"], "args": [expression(item) for item in args]}}
        if len(value) == 1:
            form, item = next(iter(value.items()))
            if form in {"literal", "ref"}:
                return {form: item}
            if form in {"op", "call"} and isinstance(item, dict):
                normalized = dict(item)
                if isinstance(normalized.get("args"), list):
                    normalized["args"] = [expression(arg) for arg in normalized["args"]]
                if form == "op" and normalized.get("name") in EXPRESSION_OPS:
                    return temporal_arithmetic(
                        str(normalized["name"]), normalized.get("args", []),
                    )
                return {form: normalized}
            if form in {"object"} and isinstance(item, dict):
                call_name = item.get("call")
                call_args = item.get("args", item.get("operands"))
                if (
                    call_name in CALLS
                    and isinstance(call_args, list)
                    and set(item).issubset({"call", "args", "operands"})
                ):
                    return {"call": {
                        "name": call_name,
                        "args": [expression(nested) for nested in call_args],
                    }}
                if len(item) == 1:
                    logical_name, logical_value = next(iter(item.items()))
                    if logical_name in {"and", "or"} and isinstance(logical_value, list):
                        return {"op": {
                            "name": logical_name,
                            "args": [expression(nested) for nested in logical_value],
                        }}
                    if logical_name == "not":
                        logical_args = (
                            logical_value if isinstance(logical_value, list)
                            else [logical_value]
                        )
                        return {"op": {
                            "name": "not",
                            "args": [expression(nested) for nested in logical_args],
                        }}
                return {form: {key: expression(nested) for key, nested in item.items()}}
            if form in {"list"} and isinstance(item, list):
                return {form: [expression(nested) for nested in item]}
            if form in {"any", "all"} and isinstance(item, dict):
                normalized = dict(item)
                if "source" in normalized:
                    normalized["source"] = expression(normalized["source"])
                if "where" in normalized:
                    normalized["where"] = expression(normalized["where"])
                return {form: normalized}
        # Models commonly emit JSON object literals directly inside lists.
        # Canonical IR makes the object form explicit so it cannot be confused
        # with an expression operator.
        return {"object": {key: expression(item) for key, item in value.items()}}

    normalized_program = []
    for raw in program:
        if not isinstance(raw, dict):
            normalized_program.append(raw)
            continue
        statement = dict(raw)
        operation = statement.get("op")
        # Some generated FactSpecs encode a collection operation as an object
        # expression assigned to a variable, for example:
        # {"op":"assign", "target":"urls", "value":{"object":
        # {"op":"map", "over":..., "item":"row", "value":...}}}.
        # This is equivalent to a native top-level map/filter statement.  The
        # expansion is syntax-only and preserves the source, loop variable and
        # predicate/value exactly.
        if operation == "assign":
            value = statement.get("value")
            nested = value.get("object") if isinstance(value, dict) else None
            if isinstance(nested, dict) and nested.get("op") in {"map", "filter"}:
                nested_operation = nested["op"]
                source = nested.get("source", nested.get("over"))
                item = nested.get("item", "loop_item")
                expanded = {
                    "op": nested_operation,
                    "target": statement.get("target"),
                    "source": source,
                    "item": item,
                }
                if nested_operation == "map":
                    expanded["value"] = nested.get("value", nested.get("expression"))
                    if "index" in nested:
                        expanded["index"] = nested["index"]
                else:
                    expanded["where"] = nested.get(
                        "where", nested.get("condition", nested.get("predicate"))
                    )
                normalized_program.append(expanded)
                continue
        # Older FactSpec versions used ``over`` for the collection consumed by
        # map/filter.  It is an unambiguous alias for the canonical ``source``
        # field and can be normalized without changing program semantics.
        if operation in {"filter", "map", "sort"} and "source" not in statement and "over" in statement:
            statement["source"] = statement.pop("over")
        if operation == "filter" and "where" not in statement and "condition" in statement:
            statement["where"] = statement.pop("condition")
        if operation == "assign" and "value" not in statement and "source" in statement:
            statement["value"] = statement.pop("source")
        if operation == "map" and "value" not in statement and "expression" in statement:
            statement["value"] = statement.pop("expression")
        if operation == "map":
            def normalize_map_index(value: Any) -> Any:
                if isinstance(value, list):
                    return [normalize_map_index(item) for item in value]
                if not isinstance(value, dict):
                    return value
                if set(value) == {"ref"} and value.get("ref") == "map.index":
                    return {"ref": "map_index"}
                return {key: normalize_map_index(item) for key, item in value.items()}

            rewritten_value = normalize_map_index(statement.get("value"))
            if rewritten_value != statement.get("value"):
                statement["value"] = rewritten_value
                statement.setdefault("index", "map_index")
        if operation in {"filter", "map", "sort"} and "item" not in statement:
            statement["item"] = "loop_item"
        for key in ("value", "source", "where", "key"):
            if key in statement:
                statement[key] = expression(statement[key])
        if operation == "emit" and isinstance(statement.get("fields"), dict):
            statement["fields"] = {
                key: expression(value) for key, value in statement["fields"].items()
            }
        normalized_program.append(statement)
    return normalized_program


def _require_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise LocalProgramError(f"{label} must be a simple identifier")
    return value


def validate_expression(expression: Any, path: str = "expression") -> None:
    if expression is None or isinstance(expression, (bool, int, float, str)):
        return
    if isinstance(expression, list):
        for index, item in enumerate(expression):
            validate_expression(item, f"{path}[{index}]")
        return
    if not isinstance(expression, dict) or len(expression) != 1:
        preview = repr(expression)
        raise LocalProgramError(
            f"{path} must contain exactly one expression form; got {preview[:500]}"
        )
    form, value = next(iter(expression.items()))
    if form == "literal":
        return
    if form == "ref":
        if not isinstance(value, str) or not value.strip():
            raise LocalProgramError(f"{path}.ref must be a non-empty path")
        return
    if form == "object":
        if not isinstance(value, dict):
            raise LocalProgramError(f"{path}.object must be an object")
        for key, item in value.items():
            _require_name(key, f"{path}.object key")
            validate_expression(item, f"{path}.object.{key}")
        return
    if form == "list":
        if not isinstance(value, list):
            raise LocalProgramError(f"{path}.list must be an array")
        for index, item in enumerate(value):
            validate_expression(item, f"{path}.list[{index}]")
        return
    if form in {"any", "all"}:
        if not isinstance(value, dict):
            raise LocalProgramError(f"{path}.{form} must be an object")
        _require_name(value.get("item"), f"{path}.{form}.item")
        if set(value) != {"source", "item", "where"}:
            raise LocalProgramError(f"{path}.{form} requires source, item and where")
        validate_expression(value["source"], f"{path}.{form}.source")
        validate_expression(value["where"], f"{path}.{form}.where")
        return
    if form == "op":
        if not isinstance(value, dict) or value.get("name") not in EXPRESSION_OPS:
            raise LocalProgramError(f"{path}.op uses an unsupported operator")
        args = value.get("args")
        if not isinstance(args, list) or not args:
            raise LocalProgramError(f"{path}.op.args must be a non-empty array")
        for index, item in enumerate(args):
            validate_expression(item, f"{path}.op.args[{index}]")
        return
    if form == "call":
        if not isinstance(value, dict) or value.get("name") not in CALLS:
            raise LocalProgramError(f"{path}.call uses an unsupported function")
        args = value.get("args", [])
        if not isinstance(args, list):
            raise LocalProgramError(f"{path}.call.args must be an array")
        if value.get("name") == "if_else" and len(args) != 3:
            raise LocalProgramError(f"{path}.call.if_else requires exactly three arguments")
        for index, item in enumerate(args):
            validate_expression(item, f"{path}.call.args[{index}]")
        return
    raise LocalProgramError(f"{path} uses unsupported form {form!r}")


def validate_local_program(program: Any) -> list[dict[str, Any]]:
    if not isinstance(program, list) or not program:
        raise LocalProgramError("local_program must be a non-empty array")
    for index, statement in enumerate(program):
        path = f"local_program[{index}]"
        if not isinstance(statement, dict) or statement.get("op") not in STATEMENT_OPS:
            raise LocalProgramError(f"{path} uses an unsupported statement")
        operation = statement["op"]
        if operation == "assign":
            _require_name(statement.get("target"), f"{path}.target")
            validate_expression(statement.get("value"), f"{path}.value")
        elif operation in {"filter", "map"}:
            _require_name(statement.get("target"), f"{path}.target")
            _require_name(statement.get("item"), f"{path}.item")
            if operation == "map" and "index" in statement:
                _require_name(statement.get("index"), f"{path}.index")
            validate_expression(statement.get("source"), f"{path}.source")
            validate_expression(
                statement.get("where") if operation == "filter" else statement.get("value"),
                f"{path}.{'where' if operation == 'filter' else 'value'}",
            )
        elif operation == "sort":
            _require_name(statement.get("target"), f"{path}.target")
            _require_name(statement.get("item"), f"{path}.item")
            validate_expression(statement.get("source"), f"{path}.source")
            validate_expression(statement.get("key"), f"{path}.key")
            if "reverse" in statement and not isinstance(statement["reverse"], bool):
                raise LocalProgramError(f"{path}.reverse must be boolean")
        elif operation == "emit":
            fields = statement.get("fields")
            if not isinstance(fields, dict) or not fields:
                raise LocalProgramError(f"{path}.fields must be a non-empty object")
            for key, value in fields.items():
                _require_name(key, f"{path}.fields key")
                validate_expression(value, f"{path}.fields.{key}")
        allowed = {
            "assign": {"op", "target", "value"},
            "filter": {"op", "target", "source", "item", "where"},
            "map": {"op", "target", "source", "item", "index", "value"},
            "sort": {"op", "target", "source", "item", "key", "reverse"},
            "emit": {"op", "fields"},
        }[operation]
        unknown = set(statement) - allowed
        if unknown:
            raise LocalProgramError(f"{path} contains unsupported fields: {sorted(unknown)}")
    return program


def _expression_source(expression: Any, scope_name: str = "scope") -> str:
    if expression is None or isinstance(expression, (bool, int, float, str)):
        return repr(expression)
    if isinstance(expression, list):
        return "[" + ", ".join(_expression_source(item, scope_name) for item in expression) + "]"
    form, value = next(iter(expression.items()))
    if form == "literal":
        return repr(value)
    if form == "ref":
        return f"_lp_ref({value!r}, input_params, operation_results, state, {scope_name})"
    if form == "object":
        return "{" + ", ".join(
            f"{key!r}: {_expression_source(item, scope_name)}" for key, item in value.items()
        ) + "}"
    if form == "list":
        return "[" + ", ".join(_expression_source(item, scope_name) for item in value) + "]"
    if form in {"any", "all"}:
        source = _expression_source(value["source"], scope_name)
        predicate = _expression_source(value["where"], "nested_scope")
        return (
            f"_lp_quantifier({form!r}, {source}, {value['item']!r}, "
            f"lambda nested_scope: {predicate}, {scope_name})"
        )
    args = ", ".join(_expression_source(item, scope_name) for item in value.get("args", []))
    if form == "op":
        return f"_lp_op({value['name']!r}, [{args}])"
    if form == "call" and value.get("name") == "if_else":
        call_args = value.get("args", [])
        condition = _expression_source(call_args[0], scope_name)
        when_true = _expression_source(call_args[1], scope_name)
        when_false = _expression_source(call_args[2], scope_name)
        return f"({when_true} if {condition} else {when_false})"
    return f"_lp_call({value['name']!r}, [{args}])"


def compile_local_program_source(
    program: list[dict[str, Any]],
    function_name: str = "compiled_local_program",
) -> str:
    validate_local_program(program)
    _require_name(function_name, "function_name")
    lines = [
        "def _lp_read(value, path):",
        "    current = value",
        "    for part in path.split('.') if path else []:",
        "        if isinstance(current, dict) and part in current:",
        "            current = current[part]",
        "        elif isinstance(current, (list, tuple)) and part.isdigit():",
        "            index = int(part)",
        "            if index >= len(current): raise KeyError(path)",
        "            current = current[index]",
        "        else:",
        "            raise KeyError(path)",
        "    return current",
        "def _lp_find(value, key):",
        "    found = []",
        "    if isinstance(value, dict):",
        "        for name, item in value.items():",
        "            if name == key: found.append(item)",
        "            found.extend(_lp_find(item, key))",
        "    elif isinstance(value, list):",
        "        for item in value: found.extend(_lp_find(item, key))",
        "    return found",
        "def _lp_ref(path, input_params, operation_results, state, scope):",
        "    if path.startswith('result.'):",
        "        return _lp_read(operation_results, path[7:])",
        "    if path.startswith('input.'):",
        "        return _lp_read(input_params, path[6:])",
        "    if path.startswith('workflow_input.'):",
        "        tail = path[15:]",
        "        root, _, suffix = tail.partition('.')",
        "        values = _lp_find(input_params, root)",
        "        if not values: return None",
        "        # A promoted fixture field can also remain under fixtures.  Identical",
        "        # copies are one logical binding; only conflicting values are ambiguous.",
        "        unique_values = []",
        "        for value in values:",
        "            if not any(value == existing for existing in unique_values):",
        "                unique_values.append(value)",
        "        if len(unique_values) != 1: raise KeyError(path)",
        "        return _lp_read(unique_values[0], suffix) if suffix else unique_values[0]",
        "    cut = len(path)",
        "    for marker in ('.', '['):",
        "        position = path.find(marker)",
        "        if position >= 0: cut = min(cut, position)",
        "    root, suffix = path[:cut], path[cut:]",
        "    if root in scope: current = scope[root]",
        "    elif root in state: current = state[root]",
        "    elif root in input_params: current = input_params[root]",
        "    else:",
        "        # Optional history/log collections are empty when the source",
        "        # has not been created yet.  Required business inputs still",
        "        # raise so missing data cannot be silently accepted.",
        "        optional_collection_tokens = ('history', 'previous', 'already', 'record', 'log')",
        "        if any(token in root.lower() for token in optional_collection_tokens): current = []",
        "        else: raise KeyError(path)",
        "    dynamic_key = False",
        "    while suffix:",
        "        if suffix.startswith('.'):",
        "            suffix = suffix[1:]",
        "            stops = [p for p in (suffix.find('.'), suffix.find('[')) if p >= 0]",
        "            stop = min(stops) if stops else len(suffix)",
        "            key, suffix = suffix[:stop], suffix[stop:]",
        "        elif suffix.startswith('['):",
        "            stop = suffix.find(']')",
        "            if stop < 0: raise KeyError(path)",
        "            token, suffix = suffix[1:stop], suffix[stop + 1:]",
        "            dynamic_key = not token.isdigit()",
        "            key = _lp_ref(token, input_params, operation_results, state, scope) if dynamic_key else int(token)",
        "        else: raise KeyError(path)",
        "        if isinstance(current, dict) and key in current: current = current[key]",
        "        elif isinstance(current, (list, tuple)) and isinstance(key, int) and 0 <= key < len(current): current = current[key]",
        "        elif dynamic_key: return None",
        "        else: raise KeyError(path)",
        "        dynamic_key = False",
        "    return current",
        "def _lp_op(name, args):",
        "    if name == 'eq': return args[0] == args[1]",
        "    if name == 'neq': return args[0] != args[1]",
        "    if name == 'lt': return args[0] < args[1]",
        "    if name == 'lte': return args[0] <= args[1]",
        "    if name == 'gt': return args[0] > args[1]",
        "    if name == 'gte': return args[0] >= args[1]",
        "    if name == 'and': return all(args)",
        "    if name == 'or': return any(args)",
        "    if name == 'not': return not args[0]",
        "    if name == 'in': return args[0] in args[1]",
        "    if name == 'contains': return args[1] in args[0]",
        "    if name == 'add': return args[0] + args[1]",
        "    if name == 'sub': return args[0] - args[1]",
        "    if name == 'mul': return args[0] * args[1]",
        "    if name == 'div': return args[0] / args[1]",
        "    raise ValueError(name)",
        "def _lp_quantifier(kind, values, item_name, predicate, outer_scope):",
        "    checks = (predicate({**outer_scope, item_name: item}) for item in values)",
        "    return any(checks) if kind == 'any' else all(checks)",
        "def _lp_call(name, args):",
        "    if name == 'len': return len(args[0])",
        "    if name == 'count_true':",
        "        values = args if len(args) > 1 else args[0]",
        "        if isinstance(values, bool): return int(values)",
        "        if isinstance(values, dict): values = values.values()",
        "        return sum(1 for value in values if value is True)",
        "    if name == 'coalesce':",
        "        for value in args:",
        "            if value is not None: return value",
        "        return None",
        "    if name == 'if_else': return args[1] if args[0] else args[2]",
        "    if name == 'get':",
        "        container, key = args[0], args[1]",
        "        default = args[2] if len(args) > 2 else None",
        "        if isinstance(container, dict): return container.get(key, default)",
        "        if isinstance(container, (list, tuple)):",
        "            if isinstance(key, int) and -len(container) <= key < len(container): return container[key]",
        "            return default",
        "        return getattr(container, key, default) if isinstance(key, str) else default",
        "    if name == 'join': return str(args[1]).join(str(value) for value in args[0])",
        "    if name == 'to_string': return str(args[0])",
        "    if name == 'to_markdown_table':",
        "        rows = args[0]",
        "        if not isinstance(rows, list) or not rows: return ''",
        "        if not all(isinstance(row, dict) for row in rows): raise TypeError('rows')",
        "        columns = list(rows[0])",
        "        def cell(value): return str(value).replace('|', r'\\|').replace('\\n', '<br>')",
        "        header = '| ' + ' | '.join(cell(name) for name in columns) + ' |'",
        "        separator = '| ' + ' | '.join('---' for _ in columns) + ' |'",
        "        body = ['| ' + ' | '.join(cell(row.get(name, '')) for name in columns) + ' |' for row in rows]",
        "        return '\\n'.join([header, separator, *body])",
        "    if name == 'merge':",
        "        result = {}",
        "        for value in args:",
        "            if not isinstance(value, dict): raise TypeError('merge')",
        "            result.update(value)",
        "        return result",
        "    if name == 'unique_by':",
        "        values, key = args[0], args[1]",
        "        seen, result = set(), []",
        "        for value in values:",
        "            marker = value.get(key) if isinstance(value, dict) else getattr(value, key, None)",
        "            if marker in seen: continue",
        "            seen.add(marker)",
        "            result.append(value)",
        "        return result",
        "    if name == 'find_by':",
        "        values, key, expected = args[0], args[1], args[2]",
        "        default = args[3] if len(args) > 3 else None",
        "        for value in values:",
        "            actual = value.get(key) if isinstance(value, dict) else getattr(value, key, None)",
        "            if actual == expected: return value",
        "        return default",
        "    if name == 'days_between':",
        "        from datetime import date, datetime",
        "        def as_date(value):",
        "            # A missing contact date means the record is not recently contacted.",
        "            # Use a large elapsed-time sentinel so eager boolean evaluation",
        "            # remains deterministic instead of parsing the string 'None'.",
        "            if value is None: return None",
        "            if isinstance(value, datetime): return value.date()",
        "            if isinstance(value, date): return value",
        "            text = str(value).strip()",
        "            try: return datetime.fromisoformat(text.replace('Z', '+00:00')).date()",
        "            except ValueError: return date.fromisoformat(text[:10])",
        "        left_date, right_date = as_date(args[0]), as_date(args[1])",
        "        if left_date is None or right_date is None: return 10**9",
        "        return abs((left_date - right_date).days)",
        "    if name == 'date_add_days':",
        "        from datetime import date, datetime, timedelta",
        "        value = args[0]",
        "        if isinstance(value, datetime): parsed = value.date()",
        "        elif isinstance(value, date): parsed = value",
        "        else:",
        "            text = str(value).strip()",
        "            try: parsed = datetime.fromisoformat(text.replace('Z', '+00:00')).date()",
        "            except ValueError: parsed = date.fromisoformat(text[:10])",
        "        return (parsed + timedelta(days=int(args[1]))).isoformat()",
        "    raise ValueError(name)",
        f"def {function_name}(input_params, operation_results, initial_state=None):",
        "    state = dict(initial_state or {})",
        "    output = {}",
        "    scope = {}",
    ]
    for index, statement in enumerate(program):
        operation = statement["op"]
        if operation == "assign":
            lines.append(f"    state[{statement['target']!r}] = {_expression_source(statement['value'])}")
        elif operation in {"filter", "map"}:
            source = _expression_source(statement["source"])
            value_key = "where" if operation == "filter" else "value"
            expression = _expression_source(statement[value_key], "item_scope")
            lines.extend([
                f"    _items_{index} = {source}",
                f"    state[{statement['target']!r}] = []",
            ])
            if operation == "map" and statement.get("index"):
                lines.extend([
                    f"    for _index_{index}, _item_{index} in enumerate(_items_{index}, start=1):",
                    f"        item_scope = {{{statement['item']!r}: _item_{index}, {statement['index']!r}: _index_{index}}}",
                ])
            else:
                lines.extend([
                    f"    for _item_{index} in _items_{index}:",
                    f"        item_scope = {{{statement['item']!r}: _item_{index}}}",
                ])
            if operation == "filter":
                lines.append(f"        if {expression}: state[{statement['target']!r}].append(_item_{index})")
            else:
                lines.append(f"        state[{statement['target']!r}].append({expression})")
        elif operation == "sort":
            source = _expression_source(statement["source"])
            key = _expression_source(statement["key"], "item_scope")
            lines.extend([
                f"    _items_{index} = list({source})",
                f"    def _key_{index}(_item_{index}):",
                f"        item_scope = {{{statement['item']!r}: _item_{index}}}",
                f"        return {key}",
                f"    state[{statement['target']!r}] = sorted(_items_{index}, key=_key_{index}, reverse={bool(statement.get('reverse', False))!r})",
            ])
        elif operation == "emit":
            for key, value in statement["fields"].items():
                lines.append(f"    output[{key!r}] = {_expression_source(value)}")
    lines.append("    return output")
    return "\n".join(lines) + "\n"


def canonical_local_program(program: list[dict[str, Any]]) -> str:
    validate_local_program(program)
    return json.dumps(program, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
