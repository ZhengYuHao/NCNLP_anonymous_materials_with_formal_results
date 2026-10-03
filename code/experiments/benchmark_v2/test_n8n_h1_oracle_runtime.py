#!/usr/bin/env python3
"""Tests for the H1 Oracle runtime primitives."""

import unittest

from n8n_h1_oracle_runtime import (
    evaluate_assertion,
    evaluate_case,
    normalize_trace,
    replay_is_deterministic,
    resolve_path,
)


class OracleRuntimeTests(unittest.TestCase):
    def test_virtual_length_path(self) -> None:
        trace = {"response": {"messages": [{"id": 1}, {"id": 2}]}}
        self.assertEqual(resolve_path(trace, "response.messages.length"), 2)

    def test_resolves_dict_and_list_paths(self) -> None:
        trace = {"mock": {"M-X": {"requests": [{"body": {"items": [3, 4]}}]}}}
        self.assertEqual(resolve_path(trace, "mock.M-X.requests[0].body.items[1]"), 4)

    def test_resolves_wildcard_and_empty_bracket_projections(self) -> None:
        trace = {"output": {"items": [{"id": "a"}, {"id": "b"}]}}
        self.assertEqual(resolve_path(trace, "output.items[*].id"), ["a", "b"])
        self.assertEqual(resolve_path(trace, "output.items[].id"), ["a", "b"])

    def test_projection_operator_semantics(self) -> None:
        trace = {
            "output": {
                "items": [
                    {"id": "a", "status": "ready", "tags": ["x", "y"]},
                    {"id": "b", "status": "ready", "tags": ["z"]},
                ]
            }
        }
        assertions = [
            {"assertion_id": "p1", "target": "output.items[*].id", "operator": "contains", "expected": "b"},
            {"assertion_id": "p2", "target": "output.items[].status", "operator": "equals", "expected": "ready"},
            {"assertion_id": "p3", "target": "output.items[*].id", "operator": "not_equals", "expected": "c"},
            {"assertion_id": "p4", "target": "output.items[*].id", "operator": "matches", "expected": "^[ab]$"},
            {"assertion_id": "p5", "target": "output.items[*].tags", "operator": "contains", "expected": "z"},
            {"assertion_id": "p6", "target": "output.items[*].id", "operator": "length_equals", "expected": 2},
            {"assertion_id": "p7", "target": "output.items[*].id", "operator": "exists", "expected": True},
        ]
        self.assertTrue(all(evaluate_assertion(item, trace).passed for item in assertions))

    def test_projection_equals_can_compare_the_full_list(self) -> None:
        trace = {"output": {"items": [{"date": "2026-09-03"}, {"date": "2026-09-04"}]}}
        assertion = {
            "assertion_id": "p1",
            "target": "output.items[].date",
            "operator": "equals",
            "expected": ["2026-09-03", "2026-09-04"],
        }
        self.assertTrue(evaluate_assertion(assertion, trace).passed)

    def test_supported_operators(self) -> None:
        trace = {"value": 3, "text": "hello", "items": ["a", "b"]}
        assertions = [
            {"assertion_id": "a1", "target": "value", "operator": "equals", "expected": 3},
            {"assertion_id": "a2", "target": "value", "operator": "not_equals", "expected": 4},
            {"assertion_id": "a3", "target": "text", "operator": "contains", "expected": "ell"},
            {"assertion_id": "a4", "target": "items", "operator": "length_equals", "expected": 2},
            {"assertion_id": "a5", "target": "text", "operator": "matches", "expected": "^he"},
            {"assertion_id": "a6", "target": "value", "operator": "exists", "expected": True},
            {"assertion_id": "a7", "target": "missing", "operator": "not_exists", "expected": True},
        ]
        self.assertTrue(all(evaluate_assertion(item, trace).passed for item in assertions))

    def test_normalizes_native_validation_shape(self) -> None:
        raw = {
            "mock": {
                "M-WEBHOOK-POST": {
                    "requests": [{"body": {"validation": {"all_match": False}}}]
                }
            }
        }
        normalized = normalize_trace("N8C-003", raw)
        self.assertTrue(
            normalized["mock"]["M-WEBHOOK-POST"]["requests"][0]["body"]
            ["validation_result"]["mismatch_detected"]
        )

    def test_normalizes_common_operation_logs_without_task_specific_values(self) -> None:
        raw = {
            "input": {"chat_id": "chat-1", "message_text": "question"},
            "exec_log": {
                "googleSheets": {"read_event_schedule": {
                    "calls": 1, "response": {"rows": [{"id": 1}, {"id": 2}]},
                }},
                "lmChatOpenRouter": {"answer_schedule_question": {
                    "calls": 1,
                    "request": {"question": "question", "schedule_table": "| id |"},
                    "response": {"answer": "answer"},
                }},
                "telegram": {"send_message": {
                    "calls": 1, "request": {"chat_id": "chat-1", "text": "answer"},
                }},
            },
        }
        normalized = normalize_trace("N8F-ANY", raw)
        self.assertEqual(normalized["exec_log"]["trigger"]["chat_id"], "chat-1")
        self.assertEqual(normalized["exec_log"]["googleSheets"]["read"]["row_count"], 2)
        self.assertIn("| id |", normalized["exec_log"]["llm"]["prompt"])
        self.assertEqual(normalized["exec_log"]["llm"]["answer"], "answer")
        self.assertEqual(normalized["exec_log"]["telegram"]["send"]["message"], "answer")

    def test_projects_real_cycle_overview_to_legacy_descriptor(self) -> None:
        raw = {
            "exec_log": {
                "gmail": {
                    "send_welcome_email": {
                        "calls": 1,
                        "request": {
                            "to": "person@example.com",
                            "cycle_overview": {"next_period_start": "2026-08-29"},
                        },
                    },
                },
            },
        }
        normalized = normalize_trace("N8F-ANY", raw)
        self.assertTrue(
            normalized["exec_log"]["gmail"]["send_welcome_email"]
            ["request"]["contains_cycle_overview"]
        )

    def test_case_result_and_replay_signature(self) -> None:
        case = {
            "case_id": "C-1",
            "assertions": [
                {"assertion_id": "A-1", "target": "output.ok", "operator": "equals", "expected": True}
            ],
        }
        run = evaluate_case("N8C-001", case, {"output": {"ok": True}})
        self.assertTrue(run["passed"])
        self.assertTrue(replay_is_deterministic([run, run, run]))

    def test_normalizes_valid_html_wording_but_preserves_raw_hash(self) -> None:
        case = {"case_id": "C", "assertions": [{
            "assertion_id": "A",
            "target": "mock.M-GMAIL-SEND-REPLY.requests[0].body_html",
            "operator": "contains",
            "expected": "<html",
        }]}
        first = {"mock": {"M-GMAIL-SEND-REPLY": {"requests": [{"body_html": "<html>first</html>"}]}}}
        second = {"mock": {"M-GMAIL-SEND-REPLY": {"requests": [{"body_html": "<html>second</html>"}]}}}
        result_a = evaluate_case("N8C-003", case, first)
        result_b = evaluate_case("N8C-003", case, second)
        self.assertEqual(result_a["trace_sha256"], result_b["trace_sha256"])
        self.assertNotEqual(result_a["raw_trace_sha256"], result_b["raw_trace_sha256"])


if __name__ == "__main__":
    unittest.main()
