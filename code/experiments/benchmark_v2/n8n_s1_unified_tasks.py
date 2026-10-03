"""Evaluator-blind loader for the S1 development qualification partition."""

from pathlib import Path

from prepare_s1_unified_tasks import canonical_hash, read_json


S1 = Path(__file__).resolve().parent / "n8n_native_s1" / "source_review_round_v2"
DEFAULT_ROOT = S1 / "runtime" / "unified_tasks_v5_candidate_r5" / "development_qualification"


def load_system_view(task_id: str, case_id: str, root: Path = DEFAULT_ROOT) -> dict:
    for identifier in (task_id, case_id):
        if not identifier or any(char in identifier for char in ("/", "\\", ":")) or identifier in {".", ".."}:
            raise ValueError("invalid task/case identifier")
    manifest = read_json(root / "manifest.json")
    if manifest.get("partition") != "development_qualification" or manifest.get("formal_test_included") is not False:
        raise ValueError("only development qualification is authorized")
    if task_id not in {item["task_id"] for item in manifest["tasks"]}:
        raise ValueError("task is not in the authorized development bundle")
    task = read_json(root / "system_inputs" / task_id / "task.json")
    instance = read_json(root / "system_inputs" / task_id / "instances" / f"{case_id}.json")
    if task["task_id"] != task_id or instance["task_id"] != task_id or instance["case_id"] != case_id:
        raise ValueError("task/instance identity mismatch")
    if task.get("dataset_partition") != "development_qualification":
        raise ValueError("formal test is sealed")
    if instance["input_sha256"] != canonical_hash(instance["input_data"]):
        raise ValueError("instance input hash mismatch")
    # Only public files are loaded. In particular, no H1 runtime defaults apply.
    return {"task": task, "instance": instance, "runtime_config": {}}
