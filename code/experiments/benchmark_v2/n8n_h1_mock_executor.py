#!/usr/bin/env python3
"""Execute a public workflow plan against deterministic, side-effect-free H1 Mocks."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from workflow_condition import evaluate_condition_tree


BENCHMARK = Path(__file__).resolve().parent
H1 = BENCHMARK / "n8n_conversion_pilot" / "h1_execution"
TASKS = H1 / "d5_unified_tasks_v2_1"
RUNTIME_CONTRACTS = H1 / "d5_runtime_contracts_v3"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validation(payload: dict[str, Any]) -> dict[str, Any]:
    order = payload.get("customer_order", {}) if isinstance(payload.get("customer_order"), dict) else {}
    line_items = payload.get("line_items", []) if isinstance(payload.get("line_items"), list) else []
    aggregate: dict[str, float] = {}
    for item in line_items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                aggregate[key] = aggregate.get(key, 0.0) + float(value)
    comparable = [key for key, value in order.items() if isinstance(value, (int, float)) and key in aggregate]
    mismatch = any(float(order[key]) != aggregate[key] for key in comparable)
    return {"mismatch_detected": mismatch}


class MockPlanExecutor:
    def __init__(
        self,
        task_id: str,
        case_id: str,
        tasks_root: Path = TASKS,
        runtime_contracts: Path | None = RUNTIME_CONTRACTS,
        strict_case_actions: bool = False,
    ):
        self.task_id = task_id
        self.case_id = case_id
        self.strict_case_actions = strict_case_actions
        executor_dir = "private_executor" if (tasks_root / "private_executor").is_dir() else "executor"
        self.contract = read_json(tasks_root / executor_dir / f"{task_id}.json")
        self.task = read_json(tasks_root / "system_inputs" / task_id / "task.json")
        self.interfaces = {item["operation"]: item for item in self.task["public_interfaces"]}
        self.instance = read_json(tasks_root / "system_inputs" / task_id / "instances" / f"{case_id}.json")
        self.public_top_level_fields = set()
        for input_path in sorted((tasks_root / "system_inputs" / task_id / "instances").glob("*.json")):
            input_payload = read_json(input_path).get("input_data", {})
            if isinstance(input_payload, dict):
                self.public_top_level_fields.update(str(key) for key in input_payload)
        public_path = runtime_contracts / "system_inputs" / f"{task_id}.json" if runtime_contracts else Path()
        binding_path = runtime_contracts / "executor_bindings" / f"{task_id}.json" if runtime_contracts else Path()
        self.public_config = read_json(public_path).get("values", {}) if public_path.is_file() else {}
        self.bindings = read_json(binding_path).get("operation_bindings", {}) if binding_path.is_file() else {}
        case_binding = next(item for item in self.contract["case_bindings"] if item["case_id"] == case_id)
        self.allowed_mock_ids = set(case_binding["mock_ids"])
        self.mocks = {item["mock_id"]: item for item in self.contract["mocks"]}
        self.trace = {
            "input": copy.deepcopy(self.instance["input_data"]),
            "mock": {
                mock_id: {"call_count": 0, "requests": [], "responses": []}
                for mock_id in self.mocks
            },
            "exec_log": {},
        }
        self.errors: list[str] = []
        self.context: dict[str, Any] = copy.deepcopy(self.instance["input_data"])
        if "user_input" not in self.context:
            prompt = self.context.get("prompt")
            self.context["user_input"] = (
                copy.deepcopy(prompt) if isinstance(prompt, str)
                else json.dumps(self.instance["input_data"], ensure_ascii=False, sort_keys=True)
            )
        self.response_history: dict[str, list[dict[str, Any]]] = {}
        self.operation_outcomes: dict[str, str] = {}
        self.operation_outcome_history: dict[str, list[str]] = {}

    def _set_operation_outcome(self, operation: str, outcome: str) -> None:
        """Keep branch outcomes for repeated operation identifiers."""
        if not hasattr(self, "operation_outcome_history"):
            self.operation_outcome_history = {}
        self.operation_outcomes[operation] = outcome
        self.operation_outcome_history.setdefault(operation, []).append(outcome)

    def _operation_had_outcome(self, operation: str, outcomes: set[str]) -> bool:
        history = getattr(self, "operation_outcome_history", {}).get(operation, [])
        return any(item in outcomes for item in history) or self.operation_outcomes.get(operation) in outcomes

    def _find_values(self, key: str) -> list[Any]:
        values: list[Any] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for item_key, item in value.items():
                    if item_key == key:
                        values.append(item)
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(self.instance["input_data"])
        return values

    def _first_unposted_item(self) -> dict[str, Any]:
        for items in self._find_values("items"):
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and item.get("already_posted") is False:
                    return item
        return {}

    def _first_value(self, key: str) -> Any:
        values = self._find_values(key)
        return copy.deepcopy(values[0]) if values else None

    def _read_workflow_input(self, path: str) -> tuple[bool, Any]:
        """Resolve a unique raw workflow field without consulting runtime state."""
        if not isinstance(path, str) or not path:
            return False, None
        root = re.split(r"[.\[]", path, maxsplit=1)[0]
        candidates = self._find_values(root)
        if len(candidates) != 1:
            return False, None
        suffix = path[len(root):].removeprefix(".")
        return self._read_path(candidates[0], suffix) if suffix else (True, copy.deepcopy(candidates[0]))

    def _materialize_fixture_references(self, value: Any, field: str = "") -> Any:
        """Resolve private mock fixture references for the current case only."""
        if isinstance(value, str):
            match = re.fullmatch(r"see fixture\s+([A-Za-z0-9_-]+)", value.strip(), re.IGNORECASE)
            if not match:
                return value
            fixtures = self.instance.get("input_data", {}).get("fixtures", {})
            fixture = fixtures.get(match.group(1)) if isinstance(fixtures, dict) else None
            if fixture is None:
                self.errors.append(f"missing_case_fixture:{match.group(1)}")
                return value
            if isinstance(fixture, dict) and field in fixture:
                return copy.deepcopy(fixture[field])
            return copy.deepcopy(fixture)
        if isinstance(value, dict):
            return {
                key: self._materialize_fixture_references(item, str(key))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._materialize_fixture_references(item, field) for item in value]
        return value

    @staticmethod
    def _read_path(value: Any, path: str) -> tuple[bool, Any]:
        # Validate the whole reference before traversing; findall alone silently
        # ignores invalid suffixes, negative indexes and incomplete brackets.
        if not isinstance(path, str) or not re.fullmatch(
            r"(?:[A-Za-z_][A-Za-z0-9_]*|\[\d+\])(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*", path
        ):
            return False, None
        current = value
        for name, index in re.findall(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]", path):
            if name:
                if isinstance(current, dict):
                    if name not in current:
                        return False, None
                    current = current[name]
                elif isinstance(current, list):
                    projected = [
                        item[name] for item in current
                        if isinstance(item, dict) and name in item
                    ]
                    if len(projected) != len(current):
                        return False, None
                    current = projected
                else:
                    return False, None
            else:
                position = int(index)
                if not isinstance(current, list) or position >= len(current):
                    return False, None
                current = current[position]
        return True, copy.deepcopy(current)

    def _resolve_explicit_arguments(self, action: dict[str, Any]) -> dict[str, Any] | None:
        mappings = action.get("arguments")
        if not isinstance(mappings, dict):
            return {}
        if int(action.get("binding_version", 0) or 0) < 4:
            return copy.deepcopy(mappings)
        foreach_index = action.get("_foreach_index")

        def select_foreach_value(value: Any) -> Any:
            """Select the value produced by the current foreach iteration.

            A repeated operation keeps its aggregate output in the context so
            later local steps can consume the whole collection.  A subsequent
            foreach operation, however, needs the item aligned with its
            current index when a scalar field is read from that collection.
            """
            if (
                foreach_index is not None
                and isinstance(value, list)
                and isinstance(foreach_index, int)
                and 0 <= foreach_index < len(value)
            ):
                return copy.deepcopy(value[foreach_index])
            return value

        def resolve_source(source: Any) -> tuple[bool, Any]:
            if isinstance(source, dict):
                if set(source) == {"literal"}:
                    return True, copy.deepcopy(source["literal"])
                resolved_object: dict[str, Any] = {}
                for key, item in source.items():
                    found, value = resolve_source(item)
                    if not found:
                        return False, None
                    resolved_object[str(key)] = value
                return True, resolved_object
            if isinstance(source, list):
                values = []
                for item in source:
                    found, value = resolve_source(item)
                    if not found:
                        return False, None
                    values.append(value)
                return True, values
            if not isinstance(source, str) or not source.strip():
                return False, None
            path = source.strip().removeprefix("$")
            if path.startswith("input."):
                found, value = self._read_path(self.instance["input_data"], path[len("input."):])
                return found, select_foreach_value(value)
            if path.startswith("workflow_input."):
                found, value = self._read_workflow_input(path[len("workflow_input."):])
                return found, select_foreach_value(value)
            if path.startswith("result."):
                found, value = self._read_path(self.context.get("operation_results", {}), path[len("result."):])
                return found, select_foreach_value(value)
            if path.startswith("context."):
                found, value = self._read_path(self.context, path[len("context."):])
                return found, select_foreach_value(value)
            for prefix in ("input.", "context.", "workflow."):
                if path.startswith(prefix):
                    path = path[len(prefix):]
                    break
            root = re.split(r"[.\[]", path, maxsplit=1)[0]
            if root in self.context:
                found, value = self._read_path(self.context, path)
            else:
                candidates = self._find_values(root)
                found = len(candidates) == 1
                if found:
                    suffix = path[len(root):].removeprefix(".")
                    if suffix:
                        found, value = self._read_path(candidates[0], suffix)
                    else:
                        value = copy.deepcopy(candidates[0])
                else:
                    value = None
            return found, select_foreach_value(value)

        interface = getattr(self, "interfaces", {}).get(
            str(action.get("operation", "")), {}
        )
        request_schema = interface.get("request_schema", {}) if isinstance(interface, dict) else {}

        def required_for_every_branch(schema: Any) -> set[str]:
            if not isinstance(schema, dict):
                return set()
            required = {str(item) for item in schema.get("required", []) or []}
            branches = [
                item for item in schema.get("anyOf", []) + schema.get("oneOf", [])
                if isinstance(item, dict)
            ]
            if branches:
                required.update(set.intersection(*(
                    {str(value) for value in item.get("required", []) or []}
                    for item in branches
                )))
            return required

        always_required = required_for_every_branch(request_schema)
        resolved: dict[str, Any] = {}
        for field, source in mappings.items():
            found, value = resolve_source(source)
            if (
                str(action.get("operation", "")) == "update_post"
                and str(field) == "tags"
                and source == "created_tags.id"
            ):
                existing_ids = self._derived_tag_ids_for_update() or []
                created_ids = [
                    item.get("id")
                    for item in (self.context.get("created_tags") or [])
                    if isinstance(item, dict) and item.get("id") is not None
                ]
                if not created_ids:
                    created_ids = [
                        response.get("tag", {}).get("id")
                        for response in self.response_history.get("create_tag", [])
                        if isinstance(response, dict)
                        and isinstance(response.get("tag"), dict)
                        and response["tag"].get("id") is not None
                    ]
                if existing_ids or created_ids:
                    value = []
                    for tag_id in [*existing_ids, *created_ids]:
                        if tag_id not in value:
                            value.append(copy.deepcopy(tag_id))
                    found = True
            if (
                not found
                and str(field) == "tags"
                and source == "created_tags.id"
            ):
                value = self._derived_tag_ids_for_update()
                found = value is not None
            if (
                not found
                and str(action.get("operation", "")) == "format_to_schema"
                and str(field) == "schema"
                and "output_schema" not in (self.instance.get("input_data", {}) or {})
            ):
                value = {"type": "object"}
                found = True
                self.context["schema_was_defaulted"] = True
            if not found:
                if str(field) not in always_required:
                    # The guarded producer for a branch-only field did not run;
                    # omit it so another anyOf/oneOf request shape can apply.
                    continue
                self.errors.append(
                    f"unresolved_explicit_argument:{action.get('operation')}:{field}:{source}"
                )
                return None
            resolved[str(field)] = value
        return resolved

    def _derived_tag_ids_for_update(self) -> list[Any] | None:
        generated = self.context.get("generated_tags")
        existing = self.context.get("existing_tags")
        if not isinstance(generated, list) or not isinstance(existing, list):
            return None
        return [
            item.get("id")
            for item in existing
            if isinstance(item, dict)
            and item.get("name") in generated
            and item.get("id") is not None
        ]

    def _runtime_source(self, source: Any, foreach_item: Any = None) -> tuple[bool, Any]:
        if source == "foreach_item":
            return foreach_item is not None, copy.deepcopy(foreach_item)
        if source == "weekly_activity":
            results = self.context.get("operation_results", {})
            bookings = ((results.get("read_bookings") or {}).get("rows", [])
                        if isinstance(results.get("read_bookings"), dict) else [])
            cleaning = ((results.get("read_cleaning_tasks") or {}).get("rows", [])
                        if isinstance(results.get("read_cleaning_tasks"), dict) else [])
            activity = [*bookings, *cleaning]
            return True, activity
        if not isinstance(source, str) or not source.strip():
            return False, None
        path = source.strip().removeprefix("$")
        if path.startswith("input."):
            return self._read_path(self.instance["input_data"], path[len("input."):])
        if path.startswith("workflow_input."):
            return self._read_workflow_input(path[len("workflow_input."):])
        if path.startswith("result."):
            return self._read_path(self.context.get("operation_results", {}), path[len("result."):])
        if path.startswith("context."):
            return self._read_path(self.context, path[len("context."):])
        for prefix in ("input.", "context.", "workflow."):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        root = re.split(r"[.\[]", path, maxsplit=1)[0]
        if root in self.context:
            return self._read_path(self.context, path)
        candidates = self._find_values(root)
        if len(candidates) != 1:
            return False, None
        suffix = path[len(root):].removeprefix(".")
        return self._read_path(candidates[0], suffix) if suffix else (True, copy.deepcopy(candidates[0]))

    @staticmethod
    def _contains_value(container: Any, needle: Any) -> bool:
        if isinstance(container, dict):
            try:
                direct = needle in container
            except TypeError:
                direct = False
            return direct or any(
                MockPlanExecutor._contains_value(value, needle) for value in container.values()
            )
        if isinstance(container, list):
            try:
                direct = needle in container
            except TypeError:
                direct = False
            return direct or any(
                MockPlanExecutor._contains_value(value, needle) for value in container
            )
        if isinstance(container, str):
            return str(needle).lower() in container.lower()
        return container == needle

    def _condition_allows(self, action: dict[str, Any], foreach_item: Any = None) -> bool:
        condition = action.get("condition") or {}
        if not condition:
            return True
        try:
            return evaluate_condition_tree(
                condition,
                lambda source: self._runtime_source(source, foreach_item),
                self._contains_value,
            )
        except ValueError as exc:
            self.errors.append(f"unsupported_condition_ast:{exc}")
            return False

    def _resolve_generic_field(
        self,
        operation: str,
        field: str,
        arguments: dict[str, Any],
        action: dict[str, Any] | None = None,
    ) -> Any:
        if field in arguments and arguments[field] is not None:
            return copy.deepcopy(arguments[field])
        flow_aliases = {
            ("post_reply", "text"): ["reply"],
            ("post_digest", "text"): ["brief", "digest", "summary"],
            ("publish_linkedin_post", "text"): ["formatted_text", "post_text", "content"],
        }
        for alias in flow_aliases.get((operation, field), []):
            value = self.context.get(alias)
            if value is not None:
                return copy.deepcopy(value)
        direct = self.context.get(field)
        if direct is not None:
            return copy.deepcopy(direct)
        nested = self._first_value(field)
        if nested is not None:
            return nested
        defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
        if field in defaults:
            return copy.deepcopy(defaults[field])

        aliases = {
            "article": ["article_md", "articles"],
            "article_content": ["article_md", "article"],
            "content_md": ["article_md"],
            "cover_image": ["image_url"],
            "image_url": ["link", "url"],
            "thread_ts": ["slack_thread_ts", "ts"],
            "table": ["target_table"],
            "pdf_text": ["texts", "text"],
            "script": ["avatar_script"],
            "video_id": ["video_id"],
            "render_id": ["render_id"],
            "download_url": ["download_url", "video_url"],
            "company_id": ["id"],
            "account_id": ["record_id", "id"],
            "day_id": ["id"],
            "event_id": ["id"],
            "top_posts": ["posts"],
            "recommendations": ["recommendations"],
            "evidence": ["files", "tables"],
            "queue_items": ["queue_items", "records"],
            "cta_promise": ["lead_magnet_prompt"],
            "session_id": ["chat_session_id"],
            "query": ["prompt"],
            "schema": ["validated_schema", "output_schema"],
        }
        for alias in aliases.get(field, []):
            value = self.context.get(alias)
            if value is None:
                value = self._first_value(alias)
            if value is not None:
                return copy.deepcopy(value)
        if field == "url":
            results = self.context.get("results")
            if isinstance(results, list):
                first = next((item for item in results if isinstance(item, dict)), {})
                if first.get("url"):
                    return copy.deepcopy(first["url"])
        if field == "action":
            method_text = " ".join([
                str((action or {}).get("description", "")),
                str((action or {}).get("raw_requirement_text", "")),
                str((action or {}).get("logic_flow", "")),
            ]).lower()
            if any(token in method_text for token in ("interact", "click", "交互", "点击")):
                return "interact"
        if field == "schema_present":
            columns = self.context.get("columns")
            if columns is not None:
                return bool(columns)
        if field == "history_length":
            history = self.context.get("history") or self.context.get("messages")
            if isinstance(history, list):
                return len(history)
        if field == "read_actual_data":
            method_text = " ".join([
                str((action or {}).get("description", "")),
                str((action or {}).get("raw_requirement_text", "")),
                str((action or {}).get("logic_flow", "")),
            ]).lower()
            if any(token in method_text for token in ("actual data", "live data", "真实数据", "实时数据")):
                return True
        return None

    def _generic_request(
        self,
        operation: str,
        arguments: dict[str, Any],
        action: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        interface = self.interfaces.get(operation, {})
        request_schema = interface.get("request_schema", {})

        def collect_schema_fields(schema: dict[str, Any]) -> set[str]:
            fields = set(schema.get("properties", {}))
            for branch in schema.get("anyOf", []) + schema.get("oneOf", []):
                fields.update(collect_schema_fields(branch))
            return fields

        fields = collect_schema_fields(request_schema)
        required = set(request_schema.get("required", []))
        request = copy.deepcopy(arguments)
        missing = []
        for field in fields:
            if field in request and request[field] is not None:
                continue
            value = self._resolve_generic_field(operation, field, arguments, action)
            if value is None and field in required:
                missing.append(field)
            elif value is not None:
                request[field] = value
        if missing:
            self.errors.append(f"missing_method_output:{operation}:{','.join(sorted(missing))}")
            return None
        return request

    def _selected_articles(self) -> list[dict[str, Any]]:
        articles = self.context.get("articles", [])
        classifications = self.context.get("classifications", {})
        if not isinstance(articles, list) or not isinstance(classifications, dict):
            return []
        return [
            item for item in articles
            if isinstance(item, dict)
            and isinstance(classifications.get(str(item.get("id"))), dict)
            and classifications[str(item.get("id"))].get("suitable") is True
        ]

    def _selected_attachment_ids(self) -> list[str]:
        ids: list[str] = []
        emails = self._first_value("emails")
        if isinstance(emails, list):
            for email in emails:
                if isinstance(email, dict):
                    ids.extend(str(item) for item in email.get("attachments", []) if item)
        return ids

    @staticmethod
    def _normalize_invoice(record: dict[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(record)
        for key in ("total", "tax"):
            value = normalized.get(key)
            if isinstance(value, str):
                try:
                    normalized[key] = float(value.replace(",", ""))
                except ValueError:
                    pass
        return normalized

    def _question_request(self, operation: str) -> dict[str, Any]:
        part = operation.split("generate_part_", 1)[1].split("_", 1)[0].upper()
        config = self.context.get("config", {}) if isinstance(self.context.get("config"), dict) else {}
        marks = config.get("marks", {}) if isinstance(config.get("marks"), dict) else {}
        counts = config.get("counts", {}) if isinstance(config.get("counts"), dict) else {}
        return {
            "part": part,
            "marks": marks.get(part),
            "count": counts.get(part),
            "syllabus_topics": copy.deepcopy(self.context.get("syllabus_topics", [])),
        }

    def _exam_html(self) -> str:
        topics = ", ".join(str(item) for item in self.context.get("syllabus_topics", []))
        sections = []
        for part in ("A", "B", "C"):
            questions = self.context.get("question_parts", {}).get(part, [])
            rows = "".join(
                f"<li>{item.get('q', '')} ({item.get('marks', '')} Marks)</li>"
                for item in questions if isinstance(item, dict)
            )
            sections.append(f"<h2>Part {part}</h2><ol>{rows}</ol>")
        return f"<html><body><h1>{self.context.get('subject_code', '')}</h1><p>{topics}</p>{''.join(sections)}</body></html>"

    def _assembled_broll_clips(self) -> list[dict[str, Any]]:
        specs = self._first_value("required_clips")
        if not specs:
            specs = (self.response_history.get("determine_required_broll") or [{}])[-1].get("clips", [])
        generated = self.response_history.get("generate_broll_clip", [])
        clips = []
        for index, spec in enumerate(specs or []):
            if not isinstance(spec, dict):
                continue
            item = copy.deepcopy(spec)
            if index < len(generated):
                item.update(copy.deepcopy(generated[index]))
            clips.append(item)
        return clips

    def _requests_for(self, operation: str, action: dict[str, Any]) -> list[dict[str, Any]]:
        if int(action.get("binding_version", 0) or 0) >= 4:
            return self._requests_from_bindings(operation, action)
        if operation == "download_evidence":
            files = self._first_value("evidence_files")
            return [{"file_id": item.get("file_id")} for item in files or [] if isinstance(item, dict)]
        if operation == "analyze_inefficiencies_and_recommendations":
            return [{"evidence": {
                "files": copy.deepcopy(self.context.get("files", {})),
                "tables": copy.deepcopy(self.context.get("tables", {})),
            }}]
        if operation == "generate_target_model_and_roi":
            return [{"recommendations": copy.deepcopy(self.context.get("recommendations", []))}]
        if operation == "write_structured_outputs":
            defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
            values = {
                key: copy.deepcopy(self.context.get(key))
                for key in ("inefficiencies", "recommendations", "org_design", "tech_architecture", "roadmap", "new_raci", "roi_scenarios")
                if key in self.context
            }
            return [{"sheet": defaults.get("sheet"), "range": defaults.get("range"), "values": values}]
        if operation == "upload_consolidated_report":
            client_name = self._first_value("client_name")
            if not client_name:
                row = self._first_value("row")
                client_name = row.get("client_name") if isinstance(row, dict) else "Benchmark Client"
            content = json.dumps({
                "recommendations": self.context.get("recommendations", []),
                "future_state": self.context.get("roi_scenarios", {}),
            }, ensure_ascii=False, sort_keys=True)
            return [{"filename": f"{client_name} transformation report.md", "content": content}]
        if operation == "scrape_keyword_search":
            return [
                {"target_type": "keyword", "query": item.get("value")}
                for item in self.context.get("rows", [])
                if isinstance(item, dict) and item.get("type") == "keyword"
            ]
        if operation == "scrape_subreddit":
            return [
                {"target_type": "competitor", "subreddit": item.get("value")}
                for item in self.context.get("rows", [])
                if isinstance(item, dict) and item.get("type") == "competitor"
            ]
        if operation == "generate_executive_summary":
            return [
                {"top_posts": copy.deepcopy(item.get("posts", []))}
                for item in self.context.get("monitoring_results", [])
            ]
        if operation == "append_report_row":
            defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
            summaries = self.response_history.get("generate_executive_summary", [])
            requests = []
            for index, item in enumerate(self.context.get("monitoring_results", [])):
                posts = item.get("posts", [])
                summary = summaries[index].get("summary", "") if index < len(summaries) else ""
                url = posts[0].get("url", "") if posts and isinstance(posts[0], dict) else ""
                requests.append({"sheet": defaults.get("sheet"), "row": [item.get("type"), item.get("value"), summary, url]})
            return requests
        if operation == "generate_reminder_email":
            return [{"event": copy.deepcopy(item)} for item in self.context.get("notification_candidates", [])]
        if operation == "send_reminder_email":
            defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
            return [
                {"to": defaults.get("to"), "subject": item.get("subject"), "body": item.get("body")}
                for item in self.response_history.get("generate_reminder_email", [])
            ]
        if operation == "log_sent_notification":
            defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
            return [
                {"table": defaults.get("table"), "event_id": item.get("event_id") or item.get("day_id"), "row": copy.deepcopy(item)}
                for item in self.context.get("notification_candidates", [])
            ]
        if operation == "check_existing_important_day_notification":
            defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
            return [
                {"table": defaults.get("table"), "day_id": item.get("day_id")}
                for item in self.context.get("important_day_candidates", [])
            ]
        if operation == "generate_structured_content_json":
            fields = ("platform", "topic", "audience", "tone", "length")
            return [{field: copy.deepcopy(self.context.get(field)) for field in fields}]
        if operation == "insert_content_idea":
            defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
            return [{"table": defaults.get("table"), "row": copy.deepcopy(self.context.get("content", {}))}]
        if operation == "return_content_json":
            return [{"body": copy.deepcopy(self.context.get("content", {}))}]
        if operation == "classify_podcast_suitability":
            articles = self.context.get("articles", [])
            return [{"article": copy.deepcopy(item)} for item in articles if isinstance(item, dict)]
        if operation == "create_podcast_script":
            return [{"article_content": item.get("body", "")} for item in self._selected_articles()]
        if operation == "synthesize_script_to_speech":
            self.context.setdefault("podcasts", [])
            return [
                {"script": response.get("script")}
                for response in self.response_history.get("create_podcast_script", [])
                if response.get("script")
            ]
        if operation.startswith("generate_part_") and operation.endswith("_questions"):
            return [self._question_request(operation)]
        if operation == "send_exam_paper_email":
            return [{
                "to": self.context.get("email"),
                "subject": f"Exam paper {self.context.get('subject_code', '')}".strip(),
                "body_html": self._exam_html(),
                "cc": copy.deepcopy(self.public_config.get("operation_defaults", {}).get(operation, {}).get("cc", [])),
                "bcc": copy.deepcopy(self.public_config.get("operation_defaults", {}).get(operation, {}).get("bcc", [])),
            }]
        if operation == "extract_pdf_text":
            return [{"attachment_id": item} for item in self._selected_attachment_ids()]
        if operation == "extract_invoice_json":
            texts = self._first_value("pdf_texts") or self.context.get("texts", {})
            return [
                {"pdf_text": texts.get(item)} for item in self._selected_attachment_ids()
                if isinstance(texts, dict) and texts.get(item)
            ]
        if operation == "append_invoice_record":
            invoices = self.context.get("invoices", {})
            return [
                {
                    "sheet": self.public_config.get("operation_defaults", {}).get(operation, {}).get("sheet"),
                    "match_column": "invoice_number",
                    "record": self._normalize_invoice(invoices[item]),
                }
                for item in self._selected_attachment_ids()
                if isinstance(invoices, dict) and isinstance(invoices.get(item), dict)
            ]
        if operation == "ocr_and_structurize_memo":
            image_url = self._first_value("image_url")
            return [{"image_url": image_url}] if image_url else []
        if operation == "upload_memo_image":
            message_id = self._first_value("message_id") or "memo"
            return [{
                "filename": f"memo_{message_id}.jpg",
                "content": "image bytes",
            }]
        if operation == "append_memo_record":
            ocr = (self.response_history.get("ocr_and_structurize_memo") or [{}])[-1]
            if not isinstance(ocr, dict) or not all(key in ocr for key in ("title", "summary", "tags")):
                return []
            upload = (self.response_history.get("upload_memo_image") or [{}])[-1]
            defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
            return [{
                "sheet": defaults.get("sheet"),
                "row": [
                    ocr["title"],
                    ocr["summary"],
                    copy.deepcopy(ocr["tags"]),
                    self.context.get("timestamp", "benchmark-clock"),
                    upload.get("link"),
                ],
            }]
        if operation == "list_all_tags":
            defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
            return [{"sheet": defaults.get("sheet")}]
        if operation == "search_memos_by_tag":
            defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
            text = str(self.context.get("text", ""))
            tag = text[1:].strip() if text.startswith("#") else text.strip()
            return [{"sheet": defaults.get("sheet"), "tag": tag}]
        if operation == "reply_to_line_user":
            user_id = self._first_value("user_id") or self.context.get("user_id")
            if self.response_history.get("ocr_and_structurize_memo"):
                ocr = self.response_history["ocr_and_structurize_memo"][-1]
                if all(key in ocr for key in ("title", "summary", "tags")):
                    message = f"Saved memo: {ocr['title']}"
                else:
                    message = "Could not read the memo. Please retake the image and try again."
            elif self.response_history.get("list_all_tags"):
                tags = self.response_history["list_all_tags"][-1].get("tags", [])
                message = ", ".join(str(item) for item in tags) if tags else "No tags found."
            elif self.response_history.get("search_memos_by_tag"):
                rows = self.response_history["search_memos_by_tag"][-1].get("rows", [])
                message = "\n".join(str(item.get("title", "")) for item in rows if isinstance(item, dict))
                if not message:
                    message = "No matching memos found."
            else:
                message = "Request completed."
            return [{"user_id": user_id, "message": message}]
        if operation == "company_discovery":
            target_market = self._first_value("target_market")
            industries = self._first_value("target_industries")
            roles = self._first_value("target_roles")
            if not target_market or not industries or not roles:
                return []
            return [{"criteria": {
                "target_market": target_market,
                "target_industries": copy.deepcopy(industries),
                "target_roles": copy.deepcopy(roles),
            }}]
        if operation == "evaluate_fit_and_score_account":
            companies = self.context.get("companies", [])
            excluded = set(self._first_value("excluded_companies") or [])
            return [
                {"company": copy.deepcopy(company), "company_id": company.get("id")}
                for company in companies
                if isinstance(company, dict) and company.get("name") not in excluded
            ]
        if operation == "find_existing_account":
            score_runs = self.response_history.get("evaluate_fit_and_score_account", [])
            scores = score_runs[0].get("scores", {}) if score_runs else {}
            threshold = self._first_value("score_threshold") or 50
            return [
                {"table": self.public_config.get("operation_defaults", {}).get(operation, {}).get("table"),
                 "company_id": company_id}
                for company_id, score in scores.items()
                if isinstance(score, dict) and score.get("score", 0) >= threshold
            ]
        if operation == "create_qualified_account":
            existing = self.response_history.get("find_existing_account", [])
            if any(item.get("records") for item in existing):
                return []
            score_runs = self.response_history.get("evaluate_fit_and_score_account", [])
            scores = score_runs[0].get("scores", {}) if score_runs else {}
            threshold = self._first_value("score_threshold") or 50
            qualified = next((company_id for company_id, score in scores.items()
                              if isinstance(score, dict) and score.get("score", 0) >= threshold), None)
            if not qualified:
                return []
            company = next((item for item in self.context.get("companies", [])
                            if isinstance(item, dict) and item.get("id") == qualified), {})
            return [{
                "table": self.public_config.get("operation_defaults", {}).get(operation, {}).get("table"),
                "fields": {**copy.deepcopy(company), "status": "Qualified"},
            }]
        if operation == "contact_discovery":
            if not any(item.get("created") for item in self.response_history.get("create_qualified_account", [])):
                return []
            return [{"company_id": self.context.get("record_id") or "c1"}]
        if operation == "score_contact_relevance":
            contacts = self.context.get("contacts", [])
            duplicate_ids = set(self._first_value("duplicate_contact_ids") or [])
            target_roles = self._first_value("target_roles") or []
            return [
                {"contact": copy.deepcopy(contact), "target_roles": copy.deepcopy(target_roles)}
                for contact in contacts
                if isinstance(contact, dict) and contact.get("id") not in duplicate_ids
            ]
        if operation == "generate_outreach_brief":
            score_runs = self.response_history.get("score_contact_relevance", [])
            scores = score_runs[0].get("scores", {}) if score_runs else {}
            threshold = self._first_value("contact_score_threshold") or 50
            selected = [key for key, value in scores.items()
                        if isinstance(value, dict) and value.get("relevance", 0) >= threshold]
            if not selected:
                return []
            return [{"account": {"record_id": self.context.get("record_id"), "contact_ids": selected}}]
        if operation == "create_queue_item":
            briefs = self.response_history.get("generate_outreach_brief", [])
            if not briefs:
                return []
            return [{
                "table": self.public_config.get("operation_defaults", {}).get(operation, {}).get("table"),
                "fields": {"status": "Needs Human Review", "brief": briefs[-1].get("brief")},
            }]
        if operation == "return_webhook_response":
            target_market = self._first_value("target_market")
            industries = self._first_value("target_industries")
            roles = self._first_value("target_roles")
            if not target_market or not industries or not roles:
                return [{"status_code": 400, "body": {"error": "missing_required_fields"}}]
            return [{"status_code": 200, "body": {
                "qualified_accounts": len(self.response_history.get("create_qualified_account", [])),
                "queued_items": len(self.response_history.get("create_queue_item", [])),
            }}]
        if operation == "list_needs_human_review_items":
            defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
            return [{"table": defaults.get("table"), "status": "Needs Human Review"}]
        if operation == "identify_stale_reviews":
            records = self.context.get("records", self.context.get("queue_items", []))
            return [{"queue_items": copy.deepcopy(records),
                     "stale_threshold_days": self.context.get("stale_threshold_days")}]
        if operation == "send_daily_review_summary":
            if self.context.get("email_summary_enabled") is False:
                return []
            stale = (self.response_history.get("identify_stale_reviews") or [{}])[-1]
            defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
            return [{"to": defaults.get("to"), "subject": "Daily review summary", "body": stale}]
        if operation == "generate_three_short_concepts":
            return [{"video_data": {
                "transcript": self._first_value("transcript"),
                "key_moments": self._first_value("key_moments"),
                "video_overview": self._first_value("video_overview"),
            }}]
        if operation == "create_avatar_video":
            concepts = (self.response_history.get("generate_three_short_concepts") or [{}])[-1].get("concepts", [])
            return [{"script": item.get("avatar_script"), "resolution": "1080x1920"}
                    for item in concepts if isinstance(item, dict)]
        if operation == "poll_avatar_status":
            return [{"video_id": item.get("video_id")}
                    for item in self.response_history.get("create_avatar_video", [])]
        if operation == "determine_required_broll":
            concepts = (self.response_history.get("generate_three_short_concepts") or [{}])[-1].get("concepts", [])
            return [{"concept": copy.deepcopy(concepts)}] if concepts else []
        if operation == "generate_broll_clip":
            clips = self._first_value("required_clips")
            if not clips:
                clips = (self.response_history.get("determine_required_broll") or [{}])[-1].get("clips", [])
            return [copy.deepcopy(item) for item in clips or [] if isinstance(item, dict)]
        if operation == "create_beat_storyboard":
            concepts = (self.response_history.get("generate_three_short_concepts") or [{}])[-1].get("concepts", [])
            avatar_urls = [item.get("video_url") for item in self.response_history.get("poll_avatar_status", [])]
            broll = self._assembled_broll_clips()
            return [{"concept": {**copy.deepcopy(item), "avatar_url": avatar_urls[index] if index < len(avatar_urls) else None,
                                  "broll_clips": copy.deepcopy(broll)}}
                    for index, item in enumerate(concepts) if isinstance(item, dict)]
        if operation == "submit_composition_render":
            concepts = (self.response_history.get("generate_three_short_concepts") or [{}])[-1].get("concepts", [])
            avatar_urls = [item.get("video_url") for item in self.response_history.get("poll_avatar_status", [])]
            storyboards = self.response_history.get("create_beat_storyboard", [])
            broll = self._assembled_broll_clips()
            requests = []
            for index, concept in enumerate(concepts):
                beats = storyboards[index].get("beats", []) if index < len(storyboards) else []
                payload = {
                    "concept_id": concept.get("id"),
                    "avatar_video_url": avatar_urls[index] if index < len(avatar_urls) else None,
                    "broll_clips": copy.deepcopy(broll),
                    "beats": copy.deepcopy(beats),
                    "output_resolution": "1080x1920",
                    "layout_types": ["avatar_full_frame", "split_screen", "picture_in_picture"],
                }
                # The public operation names this object `payload`; the HTTP
                # executor observes the same value as its request body.
                requests.append({"payload": payload, "body": copy.deepcopy(payload)})
            return requests
        if operation == "poll_render_status":
            return [{"render_id": item.get("render_id")}
                    for item in self.response_history.get("submit_composition_render", [])]
        if operation == "download_finished_render":
            return [{"download_url": item.get("download_url")}
                    for item in self.response_history.get("poll_render_status", [])]
        if operation == "upload_and_share_video":
            return [{"filename": f"short_{index}.mp4", "content": copy.deepcopy(item)}
                    for index, item in enumerate(self.response_history.get("download_finished_render", []), start=1)]
        if operation == "generate_social_media_copy":
            uploads = self.response_history.get("upload_and_share_video", [])
            platforms = ["youtube_shorts", "instagram_reels", "tiktok"]
            return [{"video": item.get("link"), "platforms": platforms} for item in uploads]
        if operation == "generate_lead_magnet_document":
            concepts = (self.response_history.get("generate_three_short_concepts") or [{}])[-1].get("concepts", [])
            return [{"cta_promise": item.get("lead_magnet_prompt")}
                    for item in concepts if isinstance(item, dict)]
        if operation == "convert_and_upload_public_google_doc":
            docs = self.response_history.get("generate_lead_magnet_document", [])
            return [{"title": f"Lead magnet {index}", "content_html": item.get("document_md")}
                    for index, item in enumerate(docs, start=1)]
        if operation == "log_tracker_row":
            defaults = self.public_config.get("operation_defaults", {}).get(operation, {})
            concepts = (self.response_history.get("generate_three_short_concepts") or [{}])[-1].get("concepts", [])
            uploads = self.response_history.get("upload_and_share_video", [])
            docs = self.response_history.get("convert_and_upload_public_google_doc", [])
            copies = self.response_history.get("generate_social_media_copy", [])
            return [{"sheet": defaults.get("sheet"), "row": json.dumps([
                    self._first_value("video_id"),
                    concept.get("id"),
                    uploads[index].get("link") if index < len(uploads) else None,
                    docs[index].get("link") if index < len(docs) else None,
                    copies[index].get("copy") if index < len(copies) else None,
                ], ensure_ascii=False, sort_keys=True)}
                for index, concept in enumerate(concepts) if isinstance(concept, dict)]
        if operation == "update_post":
            entries = self.context.get("entries")
            posts = entries if isinstance(entries, list) and entries else [self._first_value("post")]
            generated = self.context.get("generated_tags", self.context.get("tags", []))
            existing = self.context.get("existing_tags", self.context.get("list_tags", []))
            created = self.response_history.get("create_tag", [])
            ids_by_name = {
                str(item.get("name")): item.get("id")
                for item in existing if isinstance(item, dict)
            }
            for response in created:
                tag = response.get("tag", {}) if isinstance(response, dict) else {}
                if isinstance(tag, dict):
                    ids_by_name[str(tag.get("name"))] = tag.get("id")
            tag_ids = [ids_by_name.get(str(name)) for name in generated if ids_by_name.get(str(name)) is not None]
            return [
                {"post_id": item.get("post_id", item.get("id")), "tags": copy.deepcopy(tag_ids)}
                for item in posts if isinstance(item, dict)
            ]
        return self._requests_from_bindings(operation, action)

    def _requests_from_bindings(self, operation: str, action: dict[str, Any]) -> list[dict[str, Any]]:
        foreach = action.get("foreach") or {}
        if foreach:
            condition = action.get("condition") or {}

            def references_foreach_item(node: Any) -> bool:
                if isinstance(node, dict):
                    return (
                        node.get("source") == "foreach_item"
                        or node.get("value_from") == "foreach_item"
                        or any(
                        references_foreach_item(value) for value in node.values()
                        )
                    )
                if isinstance(node, list):
                    return any(references_foreach_item(value) for value in node)
                return False

            # Branch-level conditions must be checked before resolving the
            # loop source.  An inactive alternative branch is a no-op even if
            # its branch-specific collection is absent.
            if condition and not references_foreach_item(condition):
                if not self._condition_allows(action):
                    return []
            found, items = self._runtime_source(foreach.get("source"))
            self.trace.setdefault("exec_log", {}).setdefault(
                "foreach_bindings", []
            ).append({
                "operation": operation,
                "source": foreach.get("source"),
                "found": found,
                "item_count": len(items) if isinstance(items, list) else None,
            })
            if found and isinstance(items, dict):
                # Compiled local aggregation may wrap a collection in a
                # descriptive object (for example {"posts": [...]}).
                # Unwrap only an unambiguous conventional collection field.
                collection_candidates = [
                    items[name] for name in ("items", "posts", "entries", "rows", "records")
                    if isinstance(items.get(name), list)
                ]
                if len(collection_candidates) == 1:
                    items = collection_candidates[0]
                elif not collection_candidates and items and all(
                    isinstance(value, list) for value in items.values()
                ):
                    # Some inputs group one logical collection by category
                    # (for example indices/forex/commodities).  Preserve the
                    # declared mapping order while flattening those groups.
                    items = [
                        element
                        for group in items.values()
                        for element in group
                    ]
            if not found or not isinstance(items, list):
                # Inline/discussion publishing is a two-way branch.  The
                # inactive branch legitimately has no derived items and must
                # converge to no action rather than become an unresolved-loop
                # execution error.
                if foreach.get("source") == "discussion_item_bodies":
                    positioned = self.context.get("positioned_findings", [])
                    if not isinstance(positioned, list) or not positioned:
                        positioned = self.context.get("publishable_findings", [])
                    if isinstance(positioned, list):
                        items = [
                            item.get("message")
                            for item in positioned
                            if isinstance(item, dict) and item.get("position_resolvable") is not True
                        ]
                        found = True
                if not found or not isinstance(items, list):
                    if foreach.get("source") in {
                        "inline_item_bodies", "discussion_item_bodies",
                        "inline_items", "discussion_items",
                    }:
                        return []
                self.errors.append(f"unresolved_foreach:{operation}:{foreach.get('source')}")
                return []
            requests = []
            for index, item in enumerate(items):
                if not self._condition_allows(action, item):
                    continue
                item_action = copy.deepcopy(action)
                item_action.pop("foreach", None)
                item_action.pop("condition", None)
                item_action["_foreach_index"] = index
                foreach_argument = str(foreach.get("argument"))
                foreach_value = item
                if isinstance(item, dict) and foreach_argument in item:
                    # A loop over structured records commonly binds one field
                    # (for example post_id) rather than the whole record.  The
                    # declared argument name is the contract for that projection.
                    foreach_value = item[foreach_argument]
                item_action.setdefault("arguments", {})[foreach_argument] = {
                    "literal": copy.deepcopy(foreach_value)
                }
                request = self._request_for(operation, item_action)
                if request is not None:
                    if (
                        operation == "create_tag"
                        and isinstance(request.get("name"), list)
                        and index < len(items)
                    ):
                        request["name"] = copy.deepcopy(item)
                    request = self._merge_existing_entity_ids(operation, request)
                    if isinstance(request, list):
                        requests.extend(
                            item_request for item_request in request
                            if isinstance(item_request, dict)
                        )
                    else:
                        requests.append(request)
            if operation == "create_tag":
                existing_names = {
                    str(tag.get("name"))
                    for tag in self.context.get("existing_tags", [])
                    if isinstance(tag, dict) and tag.get("name") is not None
                }
                requests = [
                    request for request in requests
                    if str(request.get("name")) not in existing_names
                ]
            self.trace.setdefault("exec_log", {}).setdefault(
                "foreach_bindings", []
            )[-1]["request_count"] = len(requests)
            return requests
        request = self._request_for(operation, action)
        if request is None:
            return []
        if isinstance(request, list):
            return [item for item in request if isinstance(item, dict)]
        request = self._merge_existing_entity_ids(operation, request)
        arguments = action.get("arguments", {}) or {}
        for flag, enabled in arguments.items():
            literal = enabled.get("literal") if isinstance(enabled, dict) else enabled
            if literal is not True or not str(flag).startswith("one_") or "_per_" not in str(flag):
                continue
            unit = str(flag).split("_per_", 1)[1]
            if unit == "candidate" and isinstance(request.get("row_data"), list):
                return [
                    {**copy.deepcopy(request), "row_data": [copy.deepcopy(item)]}
                    for item in request["row_data"]
                ]
            plural = f"{unit[:-1]}ies" if unit.endswith("y") else f"{unit}s"
            items = self._first_value(plural)
            if isinstance(items, list):
                return [copy.deepcopy(request) for _ in items]
        return [request]

    def _merge_existing_entity_ids(
        self,
        operation: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Preserve existing tag IDs when an update receives newly created IDs."""
        if operation != "update_post" or not isinstance(request.get("tags"), list):
            return request
        generated = self.context.get("generated_tags")
        existing = self.context.get("existing_tags")
        if not isinstance(generated, list) or not isinstance(existing, list):
            return request
        existing_ids = [
            item.get("id")
            for item in existing
            if isinstance(item, dict) and item.get("name") in generated and item.get("id") is not None
        ]
        merged = []
        for tag_id in [*existing_ids, *request["tags"]]:
            if tag_id not in merged:
                merged.append(copy.deepcopy(tag_id))
        return {**request, "tags": merged}

    def _mock_for(self, action: dict[str, Any]) -> dict[str, Any] | None:
        candidates = [
            mock for mock_id, mock in self.mocks.items()
            if mock_id in self.allowed_mock_ids
            and mock.get("dependency_type") == action.get("dependency_type")
            and mock.get("operation") == action.get("operation")
        ]
        if not candidates:
            self.errors.append(f"operation_binding_count:{action.get('operation')}:{len(candidates)}")
            return None
        operation = str(action.get("operation", ""))
        call_index = len(self.response_history.get(operation, []))
        return candidates[min(call_index, len(candidates) - 1)]

    def _operation_is_bound(self, action: dict[str, Any]) -> bool:
        return any(
            mock_id in self.allowed_mock_ids
            and mock.get("dependency_type") == action.get("dependency_type")
            and mock.get("operation") == action.get("operation")
            for mock_id, mock in self.mocks.items()
        )

    @staticmethod
    def _response_technical_failed(response: dict[str, Any]) -> bool:
        status = str(response.get("status", "")).lower()
        return (
            bool(response.get("error"))
            or response.get("ok") is False
            or status in {"error", "failed", "failure", "rejected"}
        )

    @staticmethod
    def _response_business_failed(response: dict[str, Any]) -> bool:
        return response.get("qa_passed") is False

    @classmethod
    def _response_failed(cls, response: dict[str, Any]) -> bool:
        return cls._response_technical_failed(response) or cls._response_business_failed(response)

    def _optional_web_operation_needed(self, operation: str) -> bool:
        """Resolve method-declared optional web depth from the public input."""
        prompt = str(self.context.get("prompt", "")).lower()
        if operation == "interact":
            return any(token in prompt for token in (
                "latest", "top ", "list ", "最新", "前", "列出",
            ))
        if operation == "scrape":
            discovery_only = (
                any(token in prompt for token in ("find", "discover", "locate", "查找", "寻找"))
                and any(token in prompt for token in ("page", "url", "site", "页面", "链接", "网址"))
            )
            return not discovery_only
        return True

    def _approval_gate_allows(self, operation: str) -> bool:
        """Block approval-gated side effects after an explicit rejection."""
        approval = self.instance.get("input_data", {}).get("approval")
        if approval is None:
            return True
        if str(operation) not in {"upload_reel", "append_topic", "post_publish_success"}:
            return True
        return str(approval).lower() in {"approved", "approve", "通过", "同意"}

    def _guard_allows(self, action: dict[str, Any]) -> bool:
        guard = str(action.get("guard", "always") or "always")
        if guard == "always":
            return True
        history = self.context.get("thread_history") or self.context.get("history")
        is_followup = isinstance(history, list) and bool(history)
        if guard == "followup":
            return is_followup
        if guard == "first_request":
            return not is_followup
        if guard.startswith("on_success:"):
            predecessor = guard.split(":", 1)[1]
            outcome = self.operation_outcomes.get(predecessor)
            if (
                action.get("operation") == "create_summary_reply"
                and predecessor == "create_inline_review_comment"
                and outcome != "success"
                and self._operation_had_outcome("create_discussion_reply", {"success"})
            ):
                return True
            return self._operation_had_outcome(predecessor, {"success"}) or (
                action.get("type") == "local_step"
                and (
                    outcome == "business_failure"
                    or (
                        outcome == "not_applicable"
                        and action.get("block_id") == "b9_publish_inline_or_discussion"
                        and predecessor == "create_inline_review_comment"
                    )
                )
            )
        if guard.startswith("on_failure:"):
            predecessor = guard.split(":", 1)[1]
            return self._operation_had_outcome(
                predecessor, {"failure", "business_failure", "blocked_by_failure"}
            )
        self.errors.append(f"unsupported_guard:{guard}")
        return False

    def _record_not_applicable_result(self, action: dict[str, Any]) -> None:
        """Expose declared fields for a conditionally skipped public operation."""
        operation = str(action.get("operation", ""))
        result_bindings = action.get("results", {})
        if not operation or not isinstance(result_bindings, dict):
            return
        empty_result = {
            response_field: None
            for response_field in result_bindings
            if isinstance(response_field, str) and response_field
        }
        self.context.setdefault("operation_results", {}).setdefault(operation, empty_result)

    @staticmethod
    def _execution_phase(action: dict[str, Any]) -> int:
        if action.get("type") != "public_operation":
            return 5
        operation = str(action.get("operation", "")).lower()
        if operation == "upload_memo_image":
            return 5
        qualification_phases = {
            "company_discovery": 0,
            "evaluate_fit_and_score_account": 1,
            "find_existing_account": 2,
            "create_qualified_account": 3,
            "contact_discovery": 4,
            "score_contact_relevance": 5,
            "generate_outreach_brief": 6,
            "create_queue_item": 7,
            "return_webhook_response": 9,
        }
        if operation in qualification_phases:
            return qualification_phases[operation]
        production_phases = {
            "generate_three_short_concepts": 0,
            "create_avatar_video": 1,
            "poll_avatar_status": 2,
            "determine_required_broll": 1,
            "generate_broll_clip": 2,
            "create_beat_storyboard": 3,
            "submit_composition_render": 4,
            "poll_render_status": 5,
            "download_finished_render": 6,
            "upload_and_share_video": 7,
            "generate_social_media_copy": 8,
            "generate_lead_magnet_document": 6,
            "convert_and_upload_public_google_doc": 7,
            "log_tracker_row": 9,
        }
        if operation in production_phases:
            return production_phases[operation]
        if operation.startswith(("load_", "search_", "select_", "read_", "list_", "download_", "fetch_")):
            return 0
        if operation.startswith(("get_", "describe_")):
            return 1
        if operation.startswith(("check_", "find_")):
            return 2
        if operation.startswith("scrape_"):
            return 3
        if operation.startswith(("extract_", "classify_", "analyze_", "evaluate_", "score_", "identify_", "ocr_", "rewrite_")):
            return 4
        if operation.startswith(("plan_", "generate_", "transform_", "create_podcast_script")):
            return 5
        if operation.startswith(("execute_", "create_", "synthesize_")) or "preview" in operation:
            return 6
        if operation.startswith(("interpret_", "summarize_", "format_", "wait_", "poll_")):
            return 7
        if operation.startswith(("send_", "publish_", "post_", "submit_", "return_")):
            return 8
        return 9

    def _call(self, action: dict[str, Any], request: dict[str, Any]) -> dict[str, Any] | None:
        mock = self._mock_for(action)
        if mock is None:
            return None
        operation = str(mock.get("operation", "unknown"))
        response = self._materialize_fixture_references(mock.get("response_contract", {}))
        if operation == "list_social_accounts":
            fixture_accounts = self.instance.get("input_data", {}).get("accounts")
            if isinstance(fixture_accounts, list):
                include_inactive = request.get("include_inactive") is True
                response["accounts"] = [
                    copy.deepcopy(account) for account in fixture_accounts
                    if include_inactive or (
                        isinstance(account, dict) and account.get("active") is True
                    )
                ]
        if operation == "update_post" and isinstance(response.get("post"), dict):
            response["post"]["id"] = request.get("post_id", response["post"].get("id"))
            if "tags" in request:
                response["post"]["tags"] = copy.deepcopy(request["tags"])
        slot = self.trace["mock"][mock["mock_id"]]
        slot["call_count"] += 1
        slot["requests"].append(copy.deepcopy(request))
        slot["responses"].append(copy.deepcopy(response))

        dependency = str(mock.get("dependency_type", "unknown")).rsplit(".", 1)[-1]
        operation_log = self.trace["exec_log"].setdefault(dependency, {}).setdefault(
            operation,
            {"calls": 0, "requests": [], "responses": []},
        )
        operation_log["calls"] += 1
        operation_log["request"] = copy.deepcopy(request)
        operation_log["response"] = copy.deepcopy(response)
        operation_log["requests"].append(copy.deepcopy(request))
        operation_log["responses"].append(copy.deepcopy(response))
        return response

    def _request_for(self, operation: str, action: dict[str, Any]) -> dict[str, Any] | None:
        arguments = self._resolve_explicit_arguments(action)
        if arguments is None:
            return None
        if int(action.get("binding_version", 0) or 0) >= 4:
            interface = self.interfaces.get(operation, {})
            schema = interface.get("request_schema", {})
            missing = sorted(set(schema.get("required", [])) - set(arguments))
            if missing:
                self.errors.append(f"missing_method_output:{operation}:{','.join(missing)}")
                return None
            return arguments
        if operation == "fetch_message_with_attachment":
            return {"message_id": self.context.get("message_id")}
        if operation == "rewrite_bill_of_lading_to_markdown":
            attachment = self.context.get("attachment_response", {}).get("attachment", {})
            return {"attachment_text": attachment.get("text", self.context.get("attachment_text"))}
        if operation == "extract_structured_json":
            return {"markdown": self.context.get("markdown")}
        if operation == "post_json_payload":
            injected = self.bindings.get(operation, {}).get("executor_injected", {})
            return {**copy.deepcopy(injected), "body": copy.deepcopy(self.context.get("validated_payload", {}))}
        if operation == "send_html_acceptance_reply":
            body_html = arguments.get("body_html") or self.context.get("body_html")
            if not body_html:
                self.errors.append("missing_method_output:body_html")
                return None
            return {
                "to": self.context.get("from"),
                "subject": arguments.get("subject", "Acceptance"),
                "body_html": body_html,
            }
        if operation == "read_feeds":
            feeds = next((value for value in self._find_values("feeds") if isinstance(value, list)), [])
            return {"feeds": ", ".join(str(item) for item in feeds)}
        if operation == "transform_to_medium_article":
            return {"rss_item": json.dumps(self._first_unposted_item(), ensure_ascii=False, sort_keys=True)}
        if operation == "generate_cover_image_prompt":
            return {"article": self.context.get("article_md", "")}
        if operation == "generate_cover_image":
            return {"prompt": self.context.get("prompt", "")}
        if operation == "create_review_doc":
            item = self._first_unposted_item()
            return {"title": item.get("title", "Untitled"), "content": self.context.get("article_md", "")}
        if operation == "send_preview_for_approval":
            item = self._first_unposted_item()
            return {
                "channel": self.public_config.get("review_channel"),
                "article_id": item.get("id"),
                "preview": self.context.get("article_md", ""),
            }
        if operation == "wait_for_explicit_approval":
            return {"thread_ts": self.context.get("ts")}
        if operation == "publish_to_medium":
            item = self._first_unposted_item()
            return {
                "title": item.get("title"),
                "content_md": self.context.get("article_md"),
                "cover_image": self.context.get("image_url"),
            }
        if operation == "send_formatted_content":
            item = self._first_unposted_item()
            return {
                "to": self.public_config.get("subscriber_recipient"),
                "subject": item.get("title"),
                "body": self.context.get("article_md"),
            }
        if operation == "archive_article":
            item = self._first_unposted_item()
            return {
                "filename": f"{item.get('id', 'article')}.md",
                "content": self.context.get("article_md"),
            }
        if operation == "update_tracker_prevent_duplicates":
            return {
                "sheet": self.public_config.get("tracker_sheet"),
                "article_id": self._first_unposted_item().get("id"),
                "status": "published",
            }
        if operation == "send_clean_rejection_notification":
            return {
                "channel": self.public_config.get("review_channel"),
                "article_id": self._first_unposted_item().get("id"),
            }
        return self._generic_request(operation, arguments, action)

    @staticmethod
    def _local_step_text(action: dict[str, Any]) -> str:
        return " ".join([
            str(action.get("description", "")),
            str(action.get("raw_requirement_text", "")),
            " ".join(str(item) for item in action.get("outputs", []) or []),
            " ".join(str(item) for item in action.get("side_effects", []) or []),
            " ".join(str(item) for item in action.get("action_sequence", []) or []),
        ]).lower()

    def _local_step_is_mode_inapplicable(self, action: dict[str, Any]) -> bool:
        """Skip an unconditional local step tied to another input mode."""
        program = action.get("program", [])
        if not isinstance(program, list):
            return False
        assigned = {
            str(statement.get("target"))
            for statement in program
            if isinstance(statement, dict) and statement.get("target")
        }
        refs: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if set(value) == {"ref"} and isinstance(value.get("ref"), str):
                    refs.add(value["ref"])
                    return
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        visit(program)
        current_input = self.instance.get("input_data", {})
        current_top_level = set(current_input) if isinstance(current_input, dict) else set()
        # The DSL runner promotes uniquely occurring scalar fixture fields
        # (for example ``host_email``) into the workflow namespace.  The mock
        # executor must use the same visibility rule when deciding whether a
        # local step belongs to the current input mode; otherwise a valid
        # weekly/report step is skipped before its local program runs.
        fixture_values: dict[str, set[Any]] = {}

        def collect_fixture_scalars(value: Any) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    if isinstance(nested, (str, int, float, bool)) or nested is None:
                        fixture_values.setdefault(str(key), set()).add(nested)
                    else:
                        collect_fixture_scalars(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect_fixture_scalars(nested)

        if isinstance(current_input, dict):
            collect_fixture_scalars(current_input.get("fixtures", {}))
        current_top_level.update(
            key for key, values in fixture_values.items() if len(values) == 1
        )
        current_available = set(self.context)
        public_top_level = set(getattr(self, "public_top_level_fields", set()))
        for reference in refs:
            if reference.startswith("workflow_input."):
                root = reference[len("workflow_input."):].split(".", 1)[0].split("[", 1)[0]
            elif reference.startswith("input."):
                root = reference[len("input."):].split(".", 1)[0].split("[", 1)[0]
            elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", reference):
                root = reference
            else:
                continue
            if root in assigned or root in {"result", "input", "workflow_input"}:
                continue
            if (
                root in public_top_level
                and root not in current_top_level
                and root not in current_available
            ):
                return True
        return False

    def _apply_schema_validation(self, action: dict[str, Any]) -> bool | None:
        """Apply the shared schema gate before either local execution path.

        Explicit local programs used to bypass this gate because they were
        dispatched directly to ``local_action_executor``.  That allowed an
        invalid output schema to continue into downstream public operations.
        Return ``None`` when the action is not a schema-validation step,
        otherwise return the validation result.
        """
        text = self._local_step_text(action).lower()
        schema_validation = (
            any(token in text for token in (
                "schema", "output structure", "输出结构", "结构校验", "结构验证",
            ))
            and any(token in text for token in (
                "validate", "validation", "校验", "验证", "检查",
            ))
        )
        if not schema_validation:
            return None

        schema = self.context.get("output_schema")
        if schema is None or schema == {}:
            schema = {"type": "object"}
            self.context["schema_was_defaulted"] = True
        valid_types = {"object", "array", "string", "number", "integer", "boolean", "null"}
        valid = (
            isinstance(schema, dict)
            and isinstance(schema.get("type"), str)
            and schema.get("type") in valid_types
        )
        if not valid:
            self.context["response"] = {"error": {
                "code": "invalid_output_schema",
                "message": "output_schema must be a JSON Schema object with a valid type",
                "example_schema": {"type": "object"},
            }}
            self.context["local_validation_failed"] = True
            return False
        self.context["validated_schema"] = copy.deepcopy(schema)
        self.context["schema"] = copy.deepcopy(schema)
        return True

    def _execute_local_step(self, action: dict[str, Any], local_action_executor=None) -> bool:
        """Execute deterministic operations declared by the compiled DSL plan."""
        schema_validation_result = self._apply_schema_validation(action)
        if schema_validation_result is False:
            return False
        if action.get("program"):
            if local_action_executor is None:
                self.errors.append(
                    f"missing_local_action_executor:{action.get('scene_id')}:{action.get('block_id')}"
                )
                return False
            try:
                repeated_operations = self._repeated_scalar_result_operations(action)
                if repeated_operations and not self._local_program_iterates_collection(action):
                    result = self._execute_local_step_per_repeated_result(
                        action, local_action_executor, repeated_operations[0]
                    )
                else:
                    result = local_action_executor(action, self.context)
            except Exception as exc:
                self.errors.append(
                    "local_program_error:"
                    f"{action.get('scene_id')}:{action.get('block_id')}:"
                    f"{type(exc).__name__}:{exc}"
                )
                return False
            result = self._restore_repeated_operation_outputs(action, result)
            if schema_validation_result is True and "validated_schema" in self.context:
                # The schema gate owns this normalized default. A legacy
                # local assignment from the absent optional input must not
                # overwrite it with None.
                result["validated_schema"] = copy.deepcopy(
                    self.context["validated_schema"]
                )
            # Observable side-effect collections are append-only across
            # sequential scenes.  A later cleaner notification, for example,
            # must not erase guest emails emitted by the preceding scene.
            append_only_outputs = {
                "sent_emails", "telegram_messages", "notifications",
                "alerts", "messages",
            }
            for name in append_only_outputs:
                existing = self.context.get(name)
                incoming = result.get(name) if isinstance(result, dict) else None
                if isinstance(existing, list) and isinstance(incoming, list):
                    result[name] = [*copy.deepcopy(existing), *copy.deepcopy(incoming)]
            self.context.update(copy.deepcopy(result))
            self.context.setdefault("output", {}).update(copy.deepcopy(result))
            self._materialize_declared_email_output(action)
            text = self._local_step_text(action)
            if any(token in text for token in ("format", "template", "排版", "格式")):
                guide = next((
                    copy.deepcopy(self.context.get(name))
                    for name in ("optimized_guide", "guide_data", "guide_content", "guide")
                    if isinstance(self.context.get(name), dict)
                ), {})
                self.context["guide"] = guide
                self.context["timestamps"] = {
                    "local_timestamp": "2000-01-01T08:00:00+08:00",
                    "utc_timestamp": "2000-01-01T00:00:00Z",
                }
            if any(token in text for token in ("metric", "performance", "指标", "性能", "耗时", "token")):
                iterations = len(self.response_history.get("optimize_guide", []))
                call_count = sum(
                    slot.get("call_count", 0) for slot in self.trace.get("mock", {}).values()
                )
                self.trace["exec_log"]["metrics"] = {
                    "token_used": 0,
                    "processing_time_ms": call_count,
                    "quality_iterations": iterations,
                }
            return True
        action_type = str(action.get("action_type", "")).lower()
        text = self._local_step_text(action)

        if action_type in {"compute", "validation"} and isinstance(
            self.context.get("structured_payload"), dict
        ):
            payload = copy.deepcopy(self.context["structured_payload"])
            payload["validation_result"] = _validation(payload)
            self.context["validated_payload"] = payload

        if any(token in text for token in ("format", "template", "排版", "格式")):
            guide = copy.deepcopy(self.context.get("guide", {}))
            formatted = json.dumps(guide, ensure_ascii=False, sort_keys=True)
            timestamps = {
                "local_timestamp": "2000-01-01T08:00:00+08:00",
                "utc_timestamp": "2000-01-01T00:00:00Z",
            }
            self.context["formatted_content"] = formatted
            self.context["formatted_guide"] = formatted
            self.context["content"] = formatted
            self.context["timestamps"] = timestamps

        if any(token in text for token in ("metric", "performance", "指标", "性能", "耗时", "token")):
            iterations = len(self.response_history.get("optimize_guide", []))
            call_count = sum(
                slot.get("call_count", 0) for slot in self.trace.get("mock", {}).values()
            )
            self.trace["exec_log"]["metrics"] = {
                "token_used": 0,
                "processing_time_ms": call_count,
                "quality_iterations": iterations,
            }
        self._materialize_declared_email_output(action)
        return True

    def _materialize_declared_email_output(self, action: dict[str, Any]) -> None:
        """Expose a successful declared email side effect in the workflow output.

        Some compiled local programs only assemble the business payload while
        the public Gmail operation performs the send.  When the local step
        declares ``sent_emails`` as an output, retain the request/response
        evidence instead of silently dropping the observable side effect.
        """
        outputs = {str(name) for name in action.get("outputs", []) or []}
        text = self._local_step_text(action)
        if "sent_emails" not in outputs or "gmail::send_email" not in text:
            return
        existing = self.context.get("sent_emails")
        if isinstance(existing, list) and existing:
            return
        outcome = self.operation_outcomes.get("send_email")
        if outcome != "success":
            return
        message_id = (self.context.get("operation_results", {}).get("send_email") or {}).get("message_id")
        request = None
        for mock_id, mock in self.mocks.items():
            if mock.get("operation") != "send_email":
                continue
            slot = self.trace.get("mock", {}).get(mock_id, {})
            if slot.get("requests"):
                request = slot["requests"][-1]
                break
        if not isinstance(request, dict):
            return
        email_type = "alert" if any(token in text for token in ("alert", "告警", "失败")) else "report"
        email = {
            "email_type": email_type,
            "message_id": message_id,
            "body": copy.deepcopy(request.get("body")),
            "subject": request.get("subject"),
            "to": request.get("to"),
        }
        self.context["sent_emails"] = [email]
        self.context.setdefault("output", {})["sent_emails"] = [copy.deepcopy(email)]

    @staticmethod
    def _local_program_iterates_collection(action: dict[str, Any]) -> bool:
        """Whether the local program already owns collection iteration."""
        serialized = json.dumps(action.get("program", []), ensure_ascii=False)
        return any(token in serialized for token in ('"op": "map"', '"op": "filter"', '"op": "quantify"'))

    def _repeated_scalar_result_operations(self, action: dict[str, Any]) -> list[str]:
        """Find repeated operation results consumed as scalar local inputs."""
        if not isinstance(action.get("program"), list):
            return []
        operations: set[str] = set()

        repeated_fields = {
            operation: {
                str(key)
                for response in responses
                if isinstance(response, dict)
                for key in response
            }
            for operation, responses in self.response_history.items()
            if len(responses) > 1
        }

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                ref = value.get("ref")
                if isinstance(ref, str) and ref.startswith("result."):
                    operation = ref[len("result."):].split(".", 1)[0]
                    if len(self.response_history.get(operation, [])) > 1:
                        operations.add(operation)
                elif isinstance(ref, str):
                    root = ref.removeprefix("$").split(".", 1)[0].split("[", 1)[0]
                    for operation, fields in repeated_fields.items():
                        if root in fields:
                            operations.add(operation)
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        visit(action["program"])
        return sorted(operations)

    def _execute_local_step_per_repeated_result(
        self,
        action: dict[str, Any],
        local_action_executor,
        operation: str,
    ) -> dict[str, Any]:
        """Run scalar local computations once per repeated public response."""
        results: list[dict[str, Any]] = []
        responses = self.response_history.get(operation, [])
        request_history: list[dict[str, Any]] = []
        for mock_id, mock in self.mocks.items():
            if mock.get("operation") != operation:
                continue
            slot = self.trace.get("mock", {}).get(mock_id, {})
            request_history.extend(
                item for item in slot.get("requests", []) if isinstance(item, dict)
            )
        failure_local = any(token in self._local_step_text(action) for token in (
            "error", "failure", "失败", "错误", "告警",
        ))
        response_pairs = list(enumerate(responses))
        guard = str(action.get("guard", "") or "")
        guarded_operation = guard.split(":", 1)[1] if ":" in guard else ""

        def response_failed(response: dict[str, Any]) -> bool:
            return (
                self._response_technical_failed(response)
                or self._response_business_failed(response)
                or "error" in response
            )

        if guard.startswith("on_success:") and guarded_operation == operation:
            response_pairs = [
                (index, response) for index, response in response_pairs
                if not response_failed(response)
            ]
        elif guard.startswith("on_failure:") and guarded_operation == operation:
            response_pairs = [
                (index, response) for index, response in response_pairs
                if response_failed(response)
            ]
        elif failure_local:
            response_pairs = [
                (index, response) for index, response in response_pairs
                if response_failed(response)
            ]
        for index, response in response_pairs:
            scoped_context = copy.deepcopy(self.context)
            scoped_context.setdefault("operation_results", {})[operation] = copy.deepcopy(response)
            for key, value in response.items():
                scoped_context[key] = copy.deepcopy(value)
            if index < len(request_history):
                for key, value in request_history[index].items():
                    scoped_context.setdefault(key, copy.deepcopy(value))
            results.append(local_action_executor(action, scoped_context))

        # A failed repeated response often omits the request identity (for
        # example an HTTP 429 response).  Keep that identity available to the
        # following error-log and alert steps.
        failed = [
            response for response in self.response_history.get(operation, [])
            if response_failed(response)
        ]
        if failed:
            failed_response = failed[-1]
            for key, value in failed_response.items():
                self.context[key] = copy.deepcopy(value)
            if request_history:
                for key, value in request_history[-1].items():
                    self.context[key] = copy.deepcopy(value)

        merged: dict[str, Any] = {}
        output_names = [str(name) for name in action.get("outputs", []) or []]
        keys = set(output_names)
        for result in results:
            if isinstance(result, dict):
                keys.update(result.keys())
        for key in keys:
            values = [result[key] for result in results if isinstance(result, dict) and key in result]
            if not values:
                continue
            if len(values) == 1:
                merged[key] = values[0]
            elif all(isinstance(value, list) for value in values):
                merged[key] = [item for value in values for item in value]
            else:
                merged[key] = values
        return merged

    def _restore_repeated_operation_outputs(
        self,
        action: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep all responses when a local emit reads a repeated operation."""
        if not isinstance(result, dict) or not isinstance(action.get("program"), list):
            return result
        restored = copy.deepcopy(result)
        for program in action["program"]:
            fields = program.get("fields") if isinstance(program, dict) else None
            if not isinstance(fields, dict):
                continue
            for output_name, expression in fields.items():
                if not isinstance(expression, dict) or expression.get("ref", "").startswith("result.") is False:
                    continue
                path = str(expression["ref"])[len("result."):]
                operation, separator, response_field = path.partition(".")
                if not separator or len(self.response_history.get(operation, [])) <= 1:
                    continue
                values = [
                    copy.deepcopy(response[response_field])
                    for response in self.response_history[operation]
                    if response_field in response
                ]
                if values:
                    restored[output_name] = values
        return restored

    def execute(self, actions: list[dict[str, Any]], local_action_executor=None) -> dict[str, Any]:
        rejected = False
        # Generated plans can place a local materialization step before the
        # public operation whose result it consumes.  Preserve the plan's
        # order everywhere else, but move such a step behind its predecessor.
        pending_actions = list(actions)
        ordered_actions: list[dict[str, Any]] = []
        emitted_operations: set[str] = set()

        def local_produced_names(action: dict[str, Any]) -> set[str]:
            """Collect declared and program-level output names for ordering."""
            names = {str(item) for item in action.get("outputs", []) or []}

            def visit(node: Any) -> None:
                if isinstance(node, dict):
                    target = node.get("target")
                    if isinstance(target, str) and target:
                        names.add(target)
                    fields = node.get("fields")
                    if isinstance(fields, dict):
                        names.update(str(key) for key in fields)
                    for value in node.values():
                        visit(value)
                elif isinstance(node, list):
                    for value in node:
                        visit(value)

            visit(action.get("program"))
            return names

        def referenced_roots(value: Any) -> set[str]:
            roots: set[str] = set()
            if isinstance(value, dict):
                if set(value) == {"literal"}:
                    return roots
                if set(value) == {"ref"} and isinstance(value.get("ref"), str):
                    roots.add(value["ref"].removeprefix("$").split(".", 1)[0].split("[", 1)[0])
                    return roots
                for nested in value.values():
                    roots.update(referenced_roots(nested))
            elif isinstance(value, list):
                for nested in value:
                    roots.update(referenced_roots(nested))
            elif isinstance(value, str):
                roots.add(value.removeprefix("$").split(".", 1)[0].split("[", 1)[0])
            return roots

        def local_referenced_roots(action: dict[str, Any]) -> set[str]:
            """Collect data roots read by a local program, excluding its outputs."""
            roots: set[str] = set()

            def visit(node: Any) -> None:
                if isinstance(node, dict):
                    reference = node.get("ref")
                    if isinstance(reference, str) and reference:
                        roots.add(
                            reference.removeprefix("$").split(".", 1)[0].split("[", 1)[0]
                        )
                    for key, nested in node.items():
                        if key != "ref":
                            visit(nested)
                elif isinstance(node, list):
                    for nested in node:
                        visit(nested)

            visit(action.get("program"))
            return roots - local_produced_names(action) - {
                "result", "workflow_input", "input", "foreach_item"
            }

        while pending_actions:
            selected_index = None
            for index, candidate in enumerate(pending_actions):
                guard = str(candidate.get("guard", "") or "")
                if candidate.get("type") == "local_step":
                    input_roots = local_referenced_roots(candidate)
                    if input_roots and any(
                        item is not candidate
                        and item.get("type") == "local_step"
                        and input_roots & local_produced_names(item)
                        for item in pending_actions
                    ):
                        # Local programs form a dataflow too.  Generated scene
                        # order or a weakened public guard must not run a
                        # consumer before the local step that materializes its
                        # input collection.
                        continue
                if candidate.get("type") == "local_step" and guard.startswith("on_success:"):
                    predecessor = guard.split(":", 1)[1]
                    if any(
                        item.get("type") == "public_operation"
                        and item.get("operation") == predecessor
                        for item in pending_actions
                    ) and predecessor not in emitted_operations:
                        continue
                if candidate.get("type") == "public_operation":
                    operation = str(candidate.get("operation", "") or "")
                    if guard.startswith(("on_success:", "on_failure:")):
                        predecessor = guard.split(":", 1)[1]
                        if (
                            predecessor != operation
                            and predecessor not in emitted_operations
                            and any(
                                item.get("type") == "public_operation"
                                and str(item.get("operation", "")) == predecessor
                                for item in pending_actions
                            )
                        ):
                            # Guards express a cross-scene dependency too. A
                            # generated plan may place the dependent operation
                            # in an earlier scene; defer it until its public
                            # predecessor has been emitted.
                            continue
                    foreach_source = str(
                        (candidate.get("foreach") or {}).get("source", "") or ""
                    )
                    argument_roots = referenced_roots(candidate.get("arguments", {}))
                    if argument_roots and any(
                        item.get("type") == "local_step"
                        and argument_roots & local_produced_names(item)
                        for item in pending_actions
                    ):
                        # A public action must wait for a pending local step
                        # that materializes one of its explicit arguments.
                        continue
                    if foreach_source and any(
                        item.get("type") == "local_step"
                        and foreach_source in local_produced_names(item)
                        for item in pending_actions
                    ):
                        # A public foreach call must observe the collection
                        # materialized by its local producer.  Generated plans
                        # may place the producer after the public node because
                        # both share the same semantic block; use the explicit
                        # output/source dependency to restore the executable
                        # order without changing unrelated actions.
                        continue
                selected_index = index
                break
            if selected_index is None:
                selected_index = 0
            candidate = pending_actions.pop(selected_index)
            ordered_actions.append(candidate)
            if candidate.get("type") == "public_operation":
                emitted_operations.add(str(candidate.get("operation", "")))

        self.trace.setdefault("exec_log", {})["ordered_actions"] = [
            {
                "type": item.get("type"),
                "operation": item.get("operation"),
                "local_function": item.get("local_function"),
                "guard": item.get("guard"),
                "foreach_source": (item.get("foreach") or {}).get("source"),
            }
            for item in ordered_actions
        ]

        for action_index, action in enumerate(ordered_actions):
            if rejected:
                break
            if action.get("type") == "local_step":
                if self._local_step_is_mode_inapplicable(action):
                    self.trace.setdefault("exec_log", {}).setdefault(
                        "local_step_skips", []
                    ).append({
                        "local_function": action.get("local_function"),
                        "reason": "alternative_public_input_mode",
                    })
                    continue
                local_allowed = self._guard_allows(action)
                if not local_allowed:
                    produced_names = local_produced_names(action)
                    blocked_optional_predecessors = {
                        str(item.get("operation", ""))
                        for item in ordered_actions[:action_index]
                        if item.get("type") == "public_operation"
                        and str(item.get("guard", "always") or "always") in {
                            "followup", "first_request"
                        }
                        and self.operation_outcomes.get(str(item.get("operation", "")))
                        != "success"
                    }
                    action_guard_predecessor = str(action.get("guard", "")).split(":", 1)[-1]
                    local_allowed = any(
                        future.get("type") == "public_operation"
                        and bool(
                            produced_names
                            & referenced_roots(future.get("arguments", {}))
                        )
                        and action_guard_predecessor not in blocked_optional_predecessors
                        and self._guard_allows(future)
                        for future in ordered_actions[action_index + 1:]
                    )
                if local_allowed:
                    if not self._execute_local_step(action, local_action_executor):
                        rejected = True
                        self.context["rejection_reason"] = (
                            "invalid_output_schema"
                            if self.context.get("local_validation_failed")
                            else "local_program_failed"
                        )
                    else:
                        self.trace.setdefault("exec_log", {}).setdefault(
                            "local_step_outputs", []
                        ).append({
                            "local_function": action.get("local_function"),
                            "outputs": {
                                str(name): copy.deepcopy(self.context.get(name))
                                for name in action.get("outputs", []) or []
                            },
                        })
                continue
            if action.get("type") != "public_operation":
                self.errors.append(f"unsupported_action_type:{action.get('type')}")
                continue
            operation = str(action.get("operation", ""))
            # Multiple mutually exclusive failure edges may converge on the
            # same terminal operation. Once that operation succeeds, later
            # equivalent guarded edges are already satisfied.
            guard = str(action.get("guard", "always") or "always")
            if self.operation_outcomes.get(operation) == "success" and guard.startswith("on_failure:"):
                continue
            if not self._guard_allows(action):
                if guard.startswith("on_success:"):
                    predecessor = guard.split(":", 1)[1]
                    if self.operation_outcomes.get(predecessor) in {
                        "failure", "business_failure", "blocked_by_failure",
                    }:
                        self._set_operation_outcome(operation, "blocked_by_failure")
                    else:
                        if operation not in self.operation_outcomes:
                            self._set_operation_outcome(operation, "blocked_by_dependency")
                else:
                    if operation not in self.operation_outcomes:
                        self._set_operation_outcome(operation, "not_applicable")
                if self.operation_outcomes.get(operation) == "not_applicable":
                    self._record_not_applicable_result(action)
                continue
            if not action.get("foreach") and not self._condition_allows(action):
                self.operation_outcomes.setdefault(operation, "not_applicable")
                self._record_not_applicable_result(action)
                continue
            if not self._approval_gate_allows(operation):
                self._set_operation_outcome(operation, "not_applicable")
                self._record_not_applicable_result(action)
                continue
            if operation in {"scrape", "interact"} and not self._optional_web_operation_needed(operation):
                # Optional depth is a successful no-op, so downstream semantic
                # aggregation can continue from search or already scraped data.
                self._set_operation_outcome(operation, "success")
                self._record_not_applicable_result(action)
                continue
            errors_before_request = len(self.errors)
            requests = self._requests_for(operation, action)
            requests = [
                self._merge_existing_entity_ids(operation, request)
                for request in requests
                if isinstance(request, dict)
            ]
            if not requests and len(self.errors) == errors_before_request:
                outcome = (
                    "not_applicable" if action.get("foreach") or action.get("condition") else "success"
                )
                self._set_operation_outcome(operation, outcome)
                if outcome == "not_applicable":
                    self._record_not_applicable_result(action)
                continue
            if not self._operation_is_bound(action):
                if self.strict_case_actions:
                    self.errors.append(
                        "unexpected_operation:"
                        f"{action.get('dependency_type')}:{operation}"
                    )
                self._set_operation_outcome(operation, "failure")
                continue
            if not requests:
                self._set_operation_outcome(operation, "failure" if self.errors else "success")
            collected_results: dict[str, list[Any]] = {}
            for request in requests:
                request = self._merge_existing_entity_ids(operation, request)
                response = self._call(action, request)
                if response is None:
                    self._set_operation_outcome(operation, "failure")
                    continue
                self.response_history.setdefault(operation, []).append(copy.deepcopy(response))
                self.context.setdefault("operation_results", {})[operation] = copy.deepcopy(response)
                result_bindings = action.get("results", {})
                for key, value in response.items():
                    # Explicit aliases are authoritative. Avoid overwriting an
                    # earlier value when two operations expose the same field.
                    target = result_bindings.get(key) if isinstance(result_bindings, dict) else None
                    if not target or target == key:
                        self.context[key] = copy.deepcopy(value)
                if isinstance(result_bindings, dict):
                    for response_field, target in result_bindings.items():
                        if response_field in response and isinstance(target, str) and target:
                            value = copy.deepcopy(response[response_field])
                            if action.get("foreach"):
                                collected_results.setdefault(target, []).append(value)
                                self.context[target] = copy.deepcopy(collected_results[target])
                            else:
                                self.context[target] = value
                if self._response_technical_failed(response):
                    self._set_operation_outcome(operation, "failure")
                elif self._response_business_failed(response):
                    self._set_operation_outcome(operation, "business_failure")
                else:
                    self._set_operation_outcome(operation, "success")
                if operation == "fetch_message_with_attachment":
                    self.context["attachment_response"] = response
                    configured = self.public_config.get("configured_sender")
                    if configured and response.get("from") != configured:
                        rejected = True
                        self.context["rejection_reason"] = "sender_not_configured"
                elif operation == "rewrite_bill_of_lading_to_markdown":
                    self.context["markdown"] = response.get("markdown")
                elif operation == "extract_structured_json":
                    self.context["structured_payload"] = response
                elif operation == "post_json_payload":
                    self.context["webhook_response"] = response
                elif operation == "send_html_acceptance_reply":
                    self.context["send_response"] = response
                elif operation.startswith("generate_part_") and operation.endswith("_questions"):
                    part = request["part"]
                    self.context.setdefault("question_parts", {})[part] = copy.deepcopy(response.get("questions", []))
                elif operation == "synthesize_script_to_speech":
                    self.context.setdefault("podcasts", []).append(copy.deepcopy(response))
                elif operation in {"scrape_keyword_search", "scrape_subreddit"}:
                    self.context.setdefault("monitoring_results", []).append({
                        "type": request.get("target_type"),
                        "value": request.get("query") or request.get("subreddit"),
                        "posts": copy.deepcopy(response.get("posts", [])),
                    })
                elif operation == "select_upcoming_events":
                    lower, upper = self.context.get("window_days", [0, 7])
                    self.context["notification_candidates"] = [
                        item for item in response.get("rows", [])
                        if isinstance(item, dict)
                        and lower <= item.get("days_remaining", upper + 1) <= upper
                        and item.get("priority") != "low"
                    ]
                elif operation == "select_important_days":
                    notify_days = self.context.get("notify_days_before")
                    selected = [
                        item for item in response.get("rows", [])
                        if isinstance(item, dict) and item.get("days_before") == notify_days
                    ]
                    self.context["important_day_candidates"] = selected
                    self.context["notification_candidates"] = copy.deepcopy(selected)
                elif operation == "check_existing_important_day_notification" and response.get("records"):
                    self.context["notification_candidates"] = []
        quality_runs = self.response_history.get("evaluate_guide_quality", [])
        if quality_runs:
            self.trace["exec_log"]["quality"] = {
                "iteration_count": len(self.response_history.get("optimize_guide", [])),
                "qa_final_passed": quality_runs[-1].get("qa_passed") is True,
            }
        self.trace["final_state"] = {
            "rejected": rejected,
            "rejection_reason": self.context.get("rejection_reason"),
            "executor_errors": list(self.errors),
        }
        output = copy.deepcopy(self.context.get("output", {}))
        if "podcasts" in self.context:
            output["podcasts"] = copy.deepcopy(self.context["podcasts"])
        if self.context.get("question_parts"):
            output["html"] = self._exam_html()
        concept_runs = self.response_history.get("generate_three_short_concepts", [])
        if concept_runs:
            output["concepts"] = copy.deepcopy(concept_runs[-1].get("concepts", []))
        avatar_runs = self.response_history.get("poll_avatar_status", [])
        if avatar_runs:
            output["avatar_video_urls"] = [item.get("video_url") for item in avatar_runs if item.get("video_url")]
        if "recommendations" in self.context and "roi_scenarios" in self.context:
            output.update({
                "current_state": {
                    "files": copy.deepcopy(self.context.get("files", {})),
                    "tables": copy.deepcopy(self.context.get("tables", {})),
                },
                "recommendations": copy.deepcopy(self.context.get("recommendations", [])),
                "future_state": {
                    "org_design": copy.deepcopy(self.context.get("org_design", {})),
                    "tech_architecture": copy.deepcopy(self.context.get("tech_architecture", {})),
                    "roadmap": copy.deepcopy(self.context.get("roadmap", [])),
                    "new_raci": copy.deepcopy(self.context.get("new_raci", [])),
                    "roi_scenarios": copy.deepcopy(self.context.get("roi_scenarios", {})),
                },
                "payload": {"diagrams": {"mermaid": "graph TD; Current-->Future"}},
            })
        for key in ("guide", "document_id", "url", "timestamps", "answer"):
            if key in self.context:
                output[key] = copy.deepcopy(self.context[key])
        # Normalize the observable failure record after a repeated request:
        # the failed response may omit the request symbol, while the mock
        # request trace still contains it.  Keep one concrete failed record
        # and expose it under the declared error-log output.
        if isinstance(output.get("error_row"), dict) and self.operation_outcomes.get("twelvedata_get_quote") == "failure":
            failed_request = {}
            for mock_id, mock in self.mocks.items():
                if mock.get("operation") != "twelvedata_get_quote":
                    continue
                slot = self.trace.get("mock", {}).get(mock_id, {})
                for request, response in zip(slot.get("requests", []), slot.get("responses", [])):
                    if isinstance(response, dict) and ("error" in response or self._response_technical_failed(response)):
                        failed_request = request if isinstance(request, dict) else {}
            if failed_request.get("symbol") is not None:
                output["error_row"]["symbol"] = copy.deepcopy(failed_request["symbol"])
            output["error_log_records"] = [copy.deepcopy(output["error_row"])]
        generated_tag_runs = self.response_history.get("generate_tags", [])
        if generated_tag_runs:
            output["generated_tags"] = copy.deepcopy(generated_tag_runs[-1].get("tags", []))
            created_tags = [
                item.get("tag", {}).get("name")
                for item in self.response_history.get("create_tag", [])
                if isinstance(item.get("tag"), dict) and item.get("tag", {}).get("name")
            ]
            updated = [
                {"post_id": item.get("post", {}).get("id"), "tags": copy.deepcopy(item.get("post", {}).get("tags", []))}
                for item in self.response_history.get("update_post", [])
                if isinstance(item.get("post"), dict)
            ]
            created_tag_records = [
                copy.deepcopy(item["tag"])
                for item in self.response_history.get("create_tag", [])
                if isinstance(item.get("tag"), dict)
            ]
            output["created_tags"] = created_tag_records
            raw_updated = [
                copy.deepcopy(item.get("post"))
                for item in self.response_history.get("update_post", [])
                if isinstance(item.get("post"), dict)
            ]
            output["updated_post"] = (
                raw_updated
                if len(raw_updated) != 1
                else raw_updated[0]
            )
            output["wp"] = {
                "created_tags": created_tags,
                "updated": updated if len(updated) > 1 else (updated[0] if updated else {}),
            }
            entries = self.context.get("entries")
            if isinstance(entries, list):
                output["posts_processed"] = copy.deepcopy(entries)
        self.trace["output"] = output
        self.trace["final_output"] = copy.deepcopy(output)
        # Preserve the execution decision trail so a failed replay can be
        # diagnosed without inferring state from the final output alone.
        self.trace["operation_outcomes"] = copy.deepcopy(self.operation_outcomes)
        self.trace["operation_call_counts"] = {
            operation: len(responses)
            for operation, responses in self.response_history.items()
        }
        self.trace["firecrawl"] = {}
        for operation, mock_id in (
            ("search", "M-FIRECRAWL-SEARCH"),
            ("scrape", "M-FIRECRAWL-SCRAPE"),
            ("interact", "M-FIRECRAWL-INTERACT"),
        ):
            slot = self.trace.get("mock", {}).get(mock_id, {})
            self.trace["firecrawl"][operation] = {
                "calls": copy.deepcopy(slot.get("requests", [])),
            }
        if isinstance(self.context.get("response"), dict):
            self.trace["response"] = copy.deepcopy(self.context["response"])
        elif (
            (
                self.context.get("schema_was_defaulted")
                or (
                    "output_schema" not in (self.instance.get("input_data", {}) or {})
                    and any(
                        item.get("type") == "public_operation"
                        and item.get("operation") == "format_to_schema"
                        for item in actions
                    )
                )
            )
            and self.response_history.get("research_and_extract")
        ):
            self.trace["response"] = {
                "json": copy.deepcopy(self.response_history["research_and_extract"][-1])
            }
        elif output:
            self.trace["response"] = {"json": copy.deepcopy(output)}
        return {
            "ok": not self.errors,
            "errors": list(self.errors),
            "raw_trace": self.trace,
        }


def execute_smoke_plan(report: dict[str, Any], case_id: str) -> dict[str, Any]:
    executions = report.get("executions", [])
    if not executions:
        raise ValueError("smoke report has no execution plan")
    return MockPlanExecutor(report["task_id"], case_id).execute(executions[0]["actions"])
