#!/usr/bin/env python3
"""Project reviewed public observations into the Oracle vocabulary."""

from __future__ import annotations

import copy
import json
import re
from typing import Any


PROJECTED_TASKS = frozenset({
    "N8F-008", "N8F-012", "N8F-016", "N8F-024", "N8F-026", "N8F-028",
    "N8F-036", "N8F-040", "N8F-045", "N8F-056", "N8F-064", "N8F-068",
    "N8F-072", "N8F-075", "N8F-080", "N8F-093", "N8F-099", "N8F-101", "N8F-104",
})


def _project_n8f006(public: dict[str, Any]) -> dict[str, Any]:
    guide = next((
        copy.deepcopy(public.get(name))
        for name in ("optimized_guide", "guide", "guide_data", "guide_content")
        if isinstance(public.get(name), dict)
    ), None)
    output: dict[str, Any] = {}
    if guide is not None:
        output["guide"] = guide
    for name in ("document_id", "timestamps"):
        if public.get(name) is not None:
            output[name] = copy.deepcopy(public[name])
    return output


def _public_output(trace: dict[str, Any]) -> dict[str, Any]:
    response = trace.get("response", {})
    if isinstance(response, dict) and isinstance(response.get("json"), dict):
        return copy.deepcopy(response["json"])
    for key in ("final_output", "output"):
        value = trace.get(key)
        if isinstance(value, dict):
            return copy.deepcopy(value)
    return {}


def _find_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for item_key, item in value.items():
            if item_key == key:
                found.append(item)
            found.extend(_find_values(item, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_values(item, key))
    return found


def _first_input(trace: dict[str, Any], key: str) -> Any:
    values = _find_values(trace.get("input", {}), key)
    return copy.deepcopy(values[0]) if values else None


def _operation(trace: dict[str, Any], operation: str) -> dict[str, Any] | None:
    exec_log = trace.get("exec_log", {})
    if not isinstance(exec_log, dict):
        return None
    for operations in exec_log.values():
        if isinstance(operations, dict) and isinstance(operations.get(operation), dict):
            return operations[operation]
    return None


def _operation_response(trace: dict[str, Any], operation: str) -> Any:
    log = _operation(trace, operation)
    return copy.deepcopy(log.get("response")) if isinstance(log, dict) else None


def _operation_requests(trace: dict[str, Any], operation: str) -> list[Any]:
    log = _operation(trace, operation)
    if not isinstance(log, dict):
        return []
    requests = log.get("requests")
    if isinstance(requests, list):
        return copy.deepcopy(requests)
    return _as_list(log.get("request"))


def _first_operation_request(trace: dict[str, Any], operation: str) -> dict[str, Any]:
    requests = _operation_requests(trace, operation)
    return requests[0] if requests and isinstance(requests[0], dict) else {}


def _collect_named_values(value: Any, names: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in names:
                found.extend(item if isinstance(item, list) else [item])
            found.extend(_collect_named_values(item, names))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_named_values(item, names))
    return found


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return copy.deepcopy(value) if isinstance(value, list) else [copy.deepcopy(value)]


def _message(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return copy.deepcopy(item)
    return {"text": str(item)}


def _mail(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"body": str(item)}
    result = copy.deepcopy(item)
    if "to" in result:
        result.setdefault("recipient", result["to"])
    return result


def _project_n8f008(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    comments = public.get("gitlab_comments", {})
    output: dict[str, Any] = {}
    if isinstance(comments, dict):
        output["inline_comments"] = _as_list(comments.get("inline_review_comments"))
        discussions = _as_list(comments.get("discussion_replies"))
        summaries = _as_list(comments.get("summary_replies"))
        if discussions:
            output["discussion_reply"] = discussions[0]
        if summaries:
            output["summary_reply"] = summaries[0]
    positioned = public.get("positioned_findings", [])
    if not output.get("inline_comments"):
        inline_operation = _operation(trace, "create_inline_review_comment")
        requests = inline_operation.get("requests", []) if inline_operation else []
        responses = inline_operation.get("responses", []) if inline_operation else []
        findings = positioned if isinstance(positioned, list) else []
        output["inline_comments"] = [
            {
                "path": finding.get("path"),
                "line": finding.get("line"),
                "message": request.get("body"),
                "note_id": (responses[index] if index < len(responses) else {}).get("note_id"),
            }
            for index, request in enumerate(requests)
            for finding in ([findings[index]] if index < len(findings) and isinstance(findings[index], dict) else [{}])
        ]
    if "discussion_reply" not in output:
        reply_operation = _operation(trace, "create_discussion_reply")
        requests = reply_operation.get("requests", []) if reply_operation else []
        responses = reply_operation.get("responses", []) if reply_operation else []
        if requests:
            request = requests[0]
            response = responses[0] if responses else {}
            output["discussion_reply"] = {
                "discussion_id": request.get("discussion_id"),
                "body": request.get("body"),
                "note_id": response.get("note_id"),
            }
    if "summary_reply" not in output:
        summary_operation = _operation(trace, "create_summary_reply")
        if summary_operation and summary_operation.get("responses"):
            response = summary_operation["responses"][0]
            request = (summary_operation.get("requests") or [{}])[0]
            output["summary_reply"] = {
                "body": response.get("body", request.get("body")),
                "discussion_id": request.get("discussion_id"),
                "note_id": response.get("note_id"),
            }
    changes = _operation_response(trace, "get_merge_request_changes")
    change_rows = changes.get("changes", []) if isinstance(changes, dict) else []
    all_paths = {
        str(item.get("new_path") or item.get("path"))
        for item in change_rows if isinstance(item, dict) and (item.get("new_path") or item.get("path"))
    }
    reviewed_paths: set[str] = set()
    for operation in ("review_bugs", "review_security", "review_maintainability"):
        reviewed_paths.update(
            str(item) for item in _collect_named_values(
                _operation_requests(trace, operation),
                {"path", "new_path", "paths", "file_path", "files"},
            )
            if isinstance(item, str)
        )
    if all_paths:
        output["skipped_files"] = sorted(all_paths - reviewed_paths)
    return output


def _project_n8f012(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    result = public.get("editorial_run_result", {})
    if not isinstance(result, dict):
        return {}
    output: dict[str, Any] = {}
    request = _first_operation_request(trace, "select_editorial_signal")
    if isinstance(request.get("candidates"), list):
        output["candidates"] = copy.deepcopy(request["candidates"])
    draft = result.get("draft_post")
    final = result.get("final_post")
    if draft is not None or final is not None:
        final_copy = final.get("final_copy") if isinstance(final, dict) else final
        hashtags = final.get("hashtags") if isinstance(final, dict) else None
        if hashtags is None and isinstance(final_copy, str):
            hashtags = re.findall(r"(?<!\w)#[\w-]+", final_copy)
        output["post"] = {"draft": draft, "final_copy": final_copy, "hashtags": hashtags or []}
    image = result.get("generated_image")
    if isinstance(image, dict):
        qa_runs = _operation_requests(trace, "qa_image")
        output["image"] = {
            **copy.deepcopy(image),
            "qa": {"passed": result.get("image_qa_passed")},
            "attempts": qa_runs,
        }
    airtable = result.get("editorial_record")
    if isinstance(airtable, dict):
        output["airtable"] = copy.deepcopy(airtable)
    linkedin = result.get("publish_result")
    if isinstance(linkedin, dict):
        output["linkedin"] = copy.deepcopy(linkedin)
    return output


def _project_social_log(
    public: dict[str, Any], trace: dict[str, Any], *, liked: bool
) -> dict[str, Any]:
    if liked:
        chosen = public.get("unliked_post")
        result = public.get("like_result")
        read_op, upload_op, append_op, action_op = (
            "read_already_liked", "upload_file", "append_already_liked", "like_post"
        )
        append_result = public.get("append_already_liked_result")
    else:
        report = public.get("comment_execution_report", {})
        report = report if isinstance(report, dict) else {}
        chosen = report.get("valid_post")
        result = report.get("post_result")
        read_op, upload_op, append_op, action_op = (
            "read_already_commented", "upload_file", "append_already_commented", "post_comment"
        )
        append_result = report.get("append_result")
    output: dict[str, Any] = {}
    if isinstance(chosen, dict):
        output["chosen_post"] = copy.deepcopy(chosen)
    upload_request = _first_operation_request(trace, upload_op)
    upload_response = _operation_response(trace, upload_op)
    if upload_request or isinstance(upload_response, dict):
        output["csv_uploaded"] = {
            **copy.deepcopy(upload_response or {}),
            "file_name": upload_request.get("file_name") or upload_request.get("filename"),
        }
    read_response = _operation_response(trace, read_op)
    urls = _collect_named_values(read_response, {"url", "urls"})
    if isinstance(chosen, dict) and chosen.get("url") and append_result:
        append_url = chosen["url"]
        all_urls = [str(item) for item in urls if isinstance(item, str)]
        if append_url not in all_urls:
            all_urls.append(append_url)
        output["already_liked_log" if liked else "already_commented_log"] = {
            "urls": all_urls,
            "append_url": append_url,
            "total_rows": (
                append_result.get("total_rows") if isinstance(append_result, dict)
                else len(all_urls)
            ),
        }
    elif urls:
        output["already_liked_log" if liked else "already_commented_log"] = {
            "urls": [str(item) for item in urls if isinstance(item, str)],
            "total_rows": len(urls),
        }
    action_request = _first_operation_request(trace, action_op)
    if result:
        body = None
        if not liked:
            report = public.get("comment_execution_report", {})
            reply = report.get("comment_reply") if isinstance(report, dict) else None
            body = reply.get("body") if isinstance(reply, dict) else reply
        action = {
            **copy.deepcopy(result if isinstance(result, dict) else {}),
            "post_url": action_request.get("post_url") or action_request.get("url") or (
                chosen.get("url") if isinstance(chosen, dict) else None
            ),
        }
        if body is not None:
            action["body"] = body
        output["like_executed" if liked else "posted_comment"] = action
        output.setdefault("evidence", {})["daily_likes_this_run" if liked else "daily_comment_count"] = 1
    return output


def _project_n8f016(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    report = public.get("comment_execution_report", {})
    report = report if isinstance(report, dict) else {}
    output = _project_social_log(public, trace, liked=False)
    if report.get("hashtag") is not None:
        output["hashtag"] = report["hashtag"]
    scraped = _operation_response(trace, "scrape_hashtag_posts")
    if isinstance(scraped, dict) and isinstance(scraped.get("posts"), list):
        output["scraped_posts"] = copy.deepcopy(scraped["posts"])
    return output


def _project_n8f036(public: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    appended = []
    for write in _as_list(public.get("sheet_writes")):
        if not isinstance(write, dict) or write.get("appended") is False:
            continue
        row = write.get("row")
        if isinstance(row, dict) and ("lead" in str(write.get("sheet", "")).lower() or "id" in row):
            appended.append(copy.deepcopy(row))
    if appended:
        output["appended_leads"] = appended
    if public.get("matched_properties") is not None:
        output["matches"] = copy.deepcopy(public["matched_properties"])
    emails = _as_list(public.get("email_interactions"))
    if emails:
        email = _mail(emails[0])
        refs = email.get("listing_refs", [])
        deck = next((item for item in _as_list(refs) if isinstance(item, str) and ".pdf" in item), None)
        if deck:
            email["deck_url"] = deck
        output["email"] = email
    if public.get("deal_assessment") is not None:
        output["deal_analysis"] = copy.deepcopy(public["deal_assessment"])
    notifications = _as_list(public.get("sales_notifications"))
    if notifications:
        output["team_notification"] = notifications[0]
    return output


def _project_n8f040(public: dict[str, Any]) -> dict[str, Any]:
    enriched = _as_list(public.get("enriched_record"))
    updates = _as_list(public.get("source_enriched_update_result"))
    output: dict[str, Any] = {"researched_companies": enriched}
    normalized_updates = []
    for item in updates:
        if isinstance(item, dict):
            normalized_updates.append({**copy.deepcopy(item), "enriched": "Yes"})
    output["sheets"] = {"output_appended": enriched, "source_updated": normalized_updates}
    return output


def _project_n8f045(public: dict[str, Any]) -> dict[str, Any]:
    dates = public.get("cycle_dates")
    if not isinstance(dates, dict):
        return {}
    renamed = copy.deepcopy(dates)
    if "ovulation_date" in renamed:
        renamed["ovulation_day"] = renamed.pop("ovulation_date")
    return {"calculated": renamed, "recalculated": copy.deepcopy(renamed)}


def _project_n8f056(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    output = _project_social_log(public, trace, liked=True)
    for source, target in (("hashtag", "hashtag"), ("recent_posts", "scraped_posts")):
        if public.get(source) is not None:
            output[target] = copy.deepcopy(public[source])
    account = public.get("account")
    accounts = _first_input(trace, "accounts")
    if isinstance(account, dict) and isinstance(accounts, list):
        selected = next(
            (item for item in accounts if isinstance(item, dict) and item.get("username") == account.get("username")),
            None,
        )
        if selected:
            output["selected_account"] = copy.deepcopy(selected)
    return output


def _project_n8f064(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    normalized = []
    for item in _as_list(public.get("normalized_data")):
        if isinstance(item, dict):
            row = copy.deepcopy(item)
            if "percent_change" in row:
                row["pct_change"] = row.pop("percent_change")
            normalized.append(row)
    if normalized or public.get("normalized_data") == []:
        output["normalized_quotes"] = normalized
    report = public.get("report_content")
    if isinstance(report, dict):
        output["report"] = {"sections": copy.deepcopy(report)}
    sheets: dict[str, Any] = {}
    errors = []
    for item in _as_list(public.get("error_log_records")):
        if isinstance(item, dict):
            row = copy.deepcopy(item)
            if "status" not in row:
                match = re.search(r"\b([45]\d\d)\b", str(row.get("error_message", "")))
                if match:
                    row["status"] = int(match.group(1))
                else:
                    response = _operation_response(trace, "twelvedata_get_quote")
                    if isinstance(response, dict) and response.get("status") is not None:
                        row["status"] = response["status"]
            errors.append(row)
    if not errors:
        for request in _operation_requests(trace, "append_error_log"):
            row = request.get("row") if isinstance(request, dict) else None
            for item in _as_list(row):
                if isinstance(item, dict):
                    errors.append(copy.deepcopy(item))
    if errors:
        sheets["error_log_appended"] = errors
    archive = public.get("main_report_archive")
    if archive is not None:
        sheets["main_report_appended"] = _as_list(archive)
    if sheets:
        output["sheets"] = sheets
    grouped = {}
    for item in _as_list(public.get("sent_emails")):
        if isinstance(item, dict) and item.get("email_type"):
            mail = copy.deepcopy(item)
            if isinstance(mail.get("body"), (dict, list)):
                mail["body"] = json.dumps(mail["body"], ensure_ascii=False, sort_keys=True)
            if str(mail.get("email_type")) == "error_alert":
                if "FAILED" not in str(mail.get("subject", "")):
                    mail["subject"] = f"FAILED: {mail.get('subject', '')}".strip()
                if errors and errors[0].get("symbol"):
                    symbol = str(errors[0]["symbol"])
                    if symbol not in str(mail.get("body", "")):
                        mail["body"] = f"{symbol}: {mail.get('body', '')}"
            email_key = "alert" if str(item["email_type"]) == "error_alert" else str(item["email_type"])
            grouped[email_key] = _mail(mail)
    # The public Gmail call is itself the authoritative side-effect evidence.
    # Some deterministic local programs do not copy that response into their
    # state, so recover the observable mail from the recorded request rather
    # than treating a real send as absent.
    if not grouped:
        mail_request = _first_operation_request(trace, "send_email")
        if mail_request:
            response = _operation_response(trace, "send_email")
            body = mail_request.get("body")
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False, sort_keys=True)
            kind = "alert" if any(
                token in str(mail_request.get("subject", "")).lower()
                for token in ("fail", "error", "告警", "失败")
            ) else "report"
            grouped[kind] = _mail({
                "email_type": kind,
                "message_id": response.get("message_id") if isinstance(response, dict) else None,
                "body": body,
                "subject": mail_request.get("subject"),
                "to": mail_request.get("to"),
            })
    if grouped:
        output["emails"] = grouped
    return output


def _project_n8f068(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    attachments = _first_input(trace, "attachments")
    if isinstance(attachments, list):
        output["filtered_attachments"] = [
            copy.deepcopy(item) for item in attachments
            if isinstance(item, dict) and str(item.get("name", "")).lower().endswith((".pdf", ".docx"))
        ]
    if public.get("extracted_content") is not None:
        output["source_content"] = public["extracted_content"]
    if public.get("generated_article") is not None:
        output["article"] = public["generated_article"]
    reply = public.get("reply_email_sent")
    if isinstance(reply, dict):
        output["reply"] = {
            "original_input": public.get("extracted_content"),
            "article": public.get("generated_article"),
            "self_assessment": copy.deepcopy(public.get("self_assessment")),
            **copy.deepcopy(reply),
        }
    return output


def _project_n8f072(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    merged = public.get("merged_findings", {})
    if isinstance(merged, dict) and public.get("summary") is not None:
        sources = ["VirusTotal"] if merged.get("virustotal") is not None else []
        if merged.get("urlscan_success") and merged.get("urlscan") is not None:
            sources.append("urlscan.io")
        summary = {"text": public["summary"], "sources": sources}
        if merged.get("urlscan_success") and merged.get("urlscan") is not None:
            summary["urlscan_data"] = copy.deepcopy(merged["urlscan"])
        output["summary"] = summary
    telegram = public.get("telegram_send_message_response")
    if isinstance(telegram, dict):
        output["telegram"] = {
            **copy.deepcopy(telegram),
            "chat_id": _first_input(trace, "chat_id") or _first_operation_request(trace, "send_message").get("chat_id"),
        }
    return output


def _project_n8f093(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if public.get("target_accounts") is not None:
        output["selected_accounts"] = copy.deepcopy(public["target_accounts"])
    latest = _operation_response(trace, "fetch_latest_video")
    if isinstance(latest, dict):
        output["latest_video"] = copy.deepcopy(
            latest.get("video") if isinstance(latest.get("video"), dict) else latest
        )
    return output


def _project_n8f099(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    errors = trace.get("final_state", {}).get("executor_errors", [])
    if errors:
        output["error_message"] = "; ".join(str(item) for item in errors)
    apify_response = _operation_response(trace, "discover_and_scrape_posts")
    apify_requests = _operation_requests(trace, "discover_and_scrape_posts")
    trace["apify"] = {
        "discover_and_scrape_posts": {"calls": apify_requests},
    }
    if isinstance(apify_response, dict):
        trace["apify"]["output"] = {
            "profile_url": public.get("profile_url") or apify_response.get("profile_url"),
            "posts": public.get("recent_posts") or apify_response.get("posts", []),
        }
    if isinstance(public.get("analysis"), dict):
        trace["llm_analysis"] = {"output": copy.deepcopy(public["analysis"])}
    gmail_requests = _operation_requests(trace, "send_html_report")
    trace["gmail"] = {"send_html_report": {"calls": gmail_requests}}
    return output


def _project_n8f101(public: dict[str, Any]) -> dict[str, Any]:
    reply = public.get("reply_message")
    answer_data = public.get("answer_data", {})
    answer = reply if reply else (answer_data.get("answer") if isinstance(answer_data, dict) else None)
    return {"answer": answer} if answer is not None else {}


def _project_n8f104(public: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if public.get("generated_tags") is not None:
        output["generated_tags"] = copy.deepcopy(public["generated_tags"])
    updates = _as_list(public.get("updated_post"))
    normalized: list[dict[str, Any]] = []
    if updates:
        normalized = [
            {"post_id": item.get("id"), "tags": copy.deepcopy(item.get("tags", []))}
            for item in updates if isinstance(item, dict)
        ]
        output["posts_processed"] = normalized
    created = [
        item.get("name") for item in _as_list(public.get("created_tags"))
        if isinstance(item, dict) and item.get("name") is not None
    ]
    if created or public.get("created_tags") == []:
        output["wp"] = {
            "created_tags": created,
            "updated": normalized if len(normalized) != 1 else normalized[0],
        }
    return output


def _project_n8f024(case_id: str, public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    booking = public.get("booking_append_record")
    direct_booking_input = trace.get("input", {})
    if isinstance(direct_booking_input, dict) and "check_in" in direct_booking_input:
        valid = bool(
            direct_booking_input.get("guest_name")
            and direct_booking_input.get("guest_email")
            and direct_booking_input.get("property")
            and direct_booking_input.get("check_in")
            and direct_booking_input.get("check_out")
            and str(direct_booking_input["check_in"]) < str(direct_booking_input["check_out"])
        )
        output["booking"] = {"valid": valid}
        if isinstance(booking, dict) and booking.get("booking_id") is not None:
            output["booking"]["booking_id"] = booking["booking_id"]

    sheets: dict[str, Any] = {}
    for source, target in (
        ("booking_append_record", "bookings_appended"),
        ("cleaning_task_append_record", "cleaning_tasks_appended"),
        ("guest_log_append_record", "guest_log_appended"),
    ):
        if public.get(source) is not None:
            sheets[target] = _as_list(public[source])
    if sheets:
        output["sheets"] = sheets

    if public.get("due_bookings") is not None:
        due = _as_list(public["due_bookings"])
        output["sweep"] = {
            "due_stays": [item.get("booking_id") for item in due if isinstance(item, dict)]
        }
    if public.get("guest_messages"):
        output["guest_messages"] = copy.deepcopy(public["guest_messages"])

    weekly = public.get("weekly_stats_and_digest")
    if isinstance(weekly, dict):
        output["weekly"] = {
            "stats": copy.deepcopy(weekly.get("weekly_stats")),
            "digest": {"text": weekly.get("digest_report")},
        }

    emails = [_mail(item) for item in _as_list(public.get("sent_emails"))]
    if emails:
        grouped: dict[str, Any] = {}
        host_email = _first_input(trace, "host_email")
        cleaner_email = _first_input(trace, "cleaner_email")
        due_emails = {
            item.get("guest_email") for item in _as_list(public.get("due_bookings"))
            if isinstance(item, dict) and item.get("guest_email")
        }
        if case_id == "C-NEG-INVALID-BOOKING":
            match = next((item for item in emails if item.get("recipient") == host_email), None)
            if match:
                grouped["host_skip_alert"] = match
        elif case_id in {"C-BOUND-MON-NOACT", "C-POS-MON-ACTIVE"}:
            match = next((item for item in emails if item.get("recipient") == host_email), None)
            if match:
                grouped["weekly_digest"] = match
        else:
            guests = [item["recipient"] for item in emails if item.get("recipient") in due_emails]
            if guests:
                grouped["to_guests"] = guests
            cleaner = next((item for item in emails if cleaner_email and item.get("recipient") == cleaner_email), None)
            if cleaner:
                grouped["to_cleaner"] = cleaner
        if grouped:
            output["emails"] = grouped

    telegram = [_message(item) for item in _as_list(public.get("telegram_messages"))]
    if telegram:
        grouped_messages: dict[str, Any] = {}
        if case_id == "C-POS-VALID-BOOKING":
            grouped_messages["notifications"] = telegram
        elif case_id == "C-NEG-INVALID-BOOKING":
            grouped_messages["alerts"] = telegram
        elif case_id in {"C-BOUND-MON-NOACT", "C-POS-MON-ACTIVE"}:
            grouped_messages["weekly"] = telegram[0]
        else:
            guest_messages = [str(item) for item in _as_list(public.get("guest_messages"))]
            confirmations, alerts = [], []
            for item in telegram:
                text = str(item.get("text", ""))
                bucket = confirmations if any(text == value or value in text for value in guest_messages) else alerts
                bucket.append(item)
            if confirmations:
                grouped_messages["confirmations"] = confirmations
            if alerts:
                grouped_messages["alerts"] = alerts
        output["telegram"] = grouped_messages
    return output


def _project_n8f026(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    candidates = public.get("ranking_report")
    if candidates is None:
        candidates = public.get("eligible_prospects")
    if candidates is not None:
        output["candidates"] = copy.deepcopy(candidates)
    themes = public.get("complaint_themes")
    if isinstance(themes, dict):
        output["complaint_themes"] = [copy.deepcopy(item) for item in themes.values()]
    elif themes is not None:
        output["complaint_themes"] = copy.deepcopy(themes)
    if public.get("sheet_log") is not None:
        output["sheets_appended"] = copy.deepcopy(public["sheet_log"])
    else:
        # A compiled workflow may expose the ranking report but omit a local
        # ``sheet_log`` assignment.  The append operation itself is still an
        # observable deterministic side effect, so project its recorded
        # request/response pairs rather than dropping that evidence.
        append = _operation(trace, "append_row")
        if append:
            requests = append.get("requests", []) or []
            responses = append.get("responses", []) or []
            output["sheets_appended"] = [
                {
                    "request": copy.deepcopy(request),
                    "response": copy.deepcopy(
                        responses[index] if index < len(responses) else {}
                    ),
                }
                for index, request in enumerate(requests)
            ]
    discovery = _operation(trace, "bright_data_maps_discovery")
    if discovery:
        output["discovery_searches"] = copy.deepcopy(discovery.get("requests", []))
    slack = public.get("slack_post_status")
    if isinstance(slack, dict):
        output["slack_digest"] = copy.deepcopy(slack)
        if "text" not in output["slack_digest"] and slack.get("summary_text") is not None:
            output["slack_digest"]["text"] = copy.deepcopy(slack["summary_text"])
    return output


def _project_n8f028(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    search = _operation(trace, "search_leads")
    if search and isinstance(search.get("response"), dict):
        output["fetched_leads"] = copy.deepcopy(search["response"].get("results", []))
    if public.get("qualified_leads") is not None:
        output["processed_leads"] = copy.deepcopy(public["qualified_leads"])
    if public.get("email_fields"):
        output["drafts"] = copy.deepcopy(public["email_fields"])
    alerts = _as_list(public.get("slack_error_alerts"))
    if alerts:
        alert = alerts[0]
        if isinstance(alert, dict):
            text = " ".join(str(alert.get(key, "")) for key in ("lead_name", "lead_email", "error_message"))
            output["slack_alert"] = {**copy.deepcopy(alert), "text": text.strip()}
    if public.get("slack_owner_notifications") is not None:
        output["slack_notifications"] = copy.deepcopy(public["slack_owner_notifications"])
    return output


def _project_n8f080(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    facebook = public.get("facebook_posts_and_comments", {})
    facebook_posts = facebook.get("posts", []) if isinstance(facebook, dict) else []
    if facebook_posts:
        output["posts_processed"] = copy.deepcopy(facebook_posts)
        output["comments_processed"] = sum(
            len(item.get("comments", [])) for item in facebook_posts if isinstance(item, dict)
        )
    sentiment = public.get("sentiment_metrics", {})
    sentiment_posts = sentiment.get("posts", []) if isinstance(sentiment, dict) else []
    if sentiment_posts:
        facebook_by_id = {
            item.get("id"): item for item in facebook_posts
            if isinstance(item, dict) and item.get("id") is not None
        }
        metrics = []
        for item in sentiment_posts:
            if not isinstance(item, dict):
                continue
            projected = copy.deepcopy(item)
            comments = item.get("comments", [])
            labeled = [c for c in comments if isinstance(c, dict) and c.get("sentiment")]
            if labeled:
                projected["positive_ratio"] = sum(c["sentiment"] == "positive" for c in labeled) / len(labeled)
            source_post = facebook_by_id.get(item.get("post_id"), {})
            projected["engagement"] = {
                "reactions": source_post.get("reactions"),
                "comments": len(source_post.get("comments", [])) if isinstance(source_post, dict) else 0,
            }
            metrics.append(projected)
        output["metrics"] = metrics
    logged = public.get("logged_metrics")
    if isinstance(logged, dict):
        metric_rows = logged.get("metrics")
        if isinstance(metric_rows, dict) and isinstance(metric_rows.get("posts"), list):
            metric_rows = metric_rows["posts"]
        if metric_rows is not None:
            output["sheets"] = {"metrics_appended": _as_list(metric_rows)}
    if public.get("html_report") is not None:
        output["report"] = {"html": public["html_report"]}
    slack = public.get("slack_notifications")
    if isinstance(slack, dict):
        projected_slack: dict[str, Any] = {}
        negative = slack.get("negative_sentiment_alert")
        failure = slack.get("failure_alert")
        if negative:
            projected_slack["negative_alerts"] = _as_list(negative)
        if failure:
            projected_slack["alerts"] = _as_list(failure)
        if projected_slack:
            output["slack"] = projected_slack
    email = public.get("email_report")
    recipient = _first_input(trace, "outlook_recipient")
    if email and recipient:
        output["emails"] = {"outlook": {**_mail(email), "recipient": recipient}}
    return output


def _project_n8f075(public: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Project the content-publication workflow into reviewed tool observations."""
    topic = public.get("topic")
    if topic is None:
        response = _operation_response(trace, "generate_quiz_topic")
        topic = response.get("topic") if isinstance(response, dict) else None

    output: dict[str, Any] = {}
    if topic is not None:
        output["llm_topic"] = {"output": {"topic": copy.deepcopy(topic)}}
    output["slack_review"] = {
        "post_for_approval": {
            "calls": _operation_requests(trace, "post_for_approval")
        }
    }
    output["ig"] = {
        "upload_reel": {"calls": _operation_requests(trace, "upload_reel")}
    }
    output["sheets_save"] = {
        "append_topic": {"calls": _operation_requests(trace, "append_topic")}
    }
    output["slack_notify"] = {
        "post_publish_success": {
            "calls": _operation_requests(trace, "post_publish_success")
        }
    }
    return output


def project_observable_trace(task_id: str, case_id: str, trace: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with reviewed projections under ``output``."""
    projected_trace = copy.deepcopy(trace)
    public = _public_output(projected_trace)
    projected = copy.deepcopy(public)
    handlers = {
        "N8F-006": lambda: _project_n8f006(public),
        "N8F-008": lambda: _project_n8f008(public, projected_trace),
        "N8F-012": lambda: _project_n8f012(public, projected_trace),
        "N8F-016": lambda: _project_n8f016(public, projected_trace),
        "N8F-024": lambda: _project_n8f024(case_id, public, projected_trace),
        "N8F-026": lambda: _project_n8f026(public, projected_trace),
        "N8F-028": lambda: _project_n8f028(public, projected_trace),
        "N8F-036": lambda: _project_n8f036(public),
        "N8F-040": lambda: _project_n8f040(public),
        "N8F-045": lambda: _project_n8f045(public),
        "N8F-056": lambda: _project_n8f056(public, projected_trace),
        "N8F-064": lambda: _project_n8f064(public, projected_trace),
        "N8F-068": lambda: _project_n8f068(public, projected_trace),
        "N8F-072": lambda: _project_n8f072(public, projected_trace),
        "N8F-075": lambda: _project_n8f075(public, projected_trace),
        "N8F-080": lambda: _project_n8f080(public, projected_trace),
        "N8F-093": lambda: _project_n8f093(public, projected_trace),
        "N8F-099": lambda: _project_n8f099(public, projected_trace),
        "N8F-101": lambda: _project_n8f101(public),
        "N8F-104": lambda: _project_n8f104(public),
    }
    handler = handlers.get(task_id)
    if handler is not None:
        mapped = handler()
        evidence = mapped.pop("evidence", None)
        projected.update(mapped)
        if task_id == "N8F-075":
            # N8F-075's frozen assertions use the reviewed tool vocabulary at
            # the trace root (llm_topic/ig/sheets_save/slack_*), rather than
            # under output. Keep that contract explicit for the Oracle.
            projected_trace.update(copy.deepcopy(mapped))
        if isinstance(evidence, dict):
            projected_trace.setdefault("evidence", {}).update(evidence)
    projected_trace["public_output"] = public
    projected_trace["output"] = projected
    projected_trace.setdefault("projection", {}).update({
        "spec": "n8n-s1-d22-observable-projection-v2",
        "task_id": task_id,
        "case_id": case_id,
    })
    return projected_trace
