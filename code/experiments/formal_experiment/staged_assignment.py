"""RQ3 staged LLM assignment with unchanged recovered facts and renderer input."""

import copy
import json
from dataclasses import asdict

TRIGGER_STRATEGIES = {"BLOCK_CONTAINS", "BLOCK_COMPARE", "SEMANTIC_BLOCK_CLASSIFICATION",
                      "SEMANTIC_BLOCK_EXTRACTION", "RAW_SEMANTIC", "HYBRID"}
ACTION_STRATEGIES = {"CODE_ASSIGN", "CODE_TEMPLATE", "CODE_CALL_FUNC", "CODE_COMPUTE",
                     "SEMANTIC_BLOCK_GENERATION", "SEMANTIC_BLOCK_SCORING",
                     "SEMANTIC_BLOCK_EXTRACTION", "SEMANTIC_BLOCK_SIMILARITY", "RAW_SEMANTIC", "HYBRID"}


def assign(fact_spec, classified_spec, client):
    prompt = (
        "Assign implementation strategies to the recovered workflow facts. "
        "Use deterministic code for explicit rules and calculations; use semantic blocks "
        "for open-ended interpretation and generation. Do not modify facts or omit scenes. "
        "Return JSON only: {\"assignments\": [{\"scene_id\": 1, "
        "\"trigger_strategy\": \"...\", \"action_strategy\": \"...\"}]}.\n"
        "Allowed trigger strategies: " + ", ".join(sorted(TRIGGER_STRATEGIES)) + "\n"
        "Allowed action strategies: " + ", ".join(sorted(ACTION_STRATEGIES))
    )
    response = client.call(system_prompt=prompt,
        user_content=json.dumps(asdict(fact_spec), ensure_ascii=False, sort_keys=True),
        temperature=0.0, max_tokens=8192)
    result = json.loads(response.content)
    assignments = result.get("assignments") if isinstance(result, dict) else None
    if not isinstance(assignments, list):
        raise ValueError("LLM assignment must contain an assignments array")
    mapping = {}
    for item in assignments:
        if not isinstance(item, dict) or set(item) != {"scene_id", "trigger_strategy", "action_strategy"}:
            raise ValueError("assignment may only change strategy fields")
        scene_id = item["scene_id"]
        if type(scene_id) is not int or scene_id in mapping:
            raise ValueError("duplicate or invalid scene identity")
        if item["trigger_strategy"] not in TRIGGER_STRATEGIES or item["action_strategy"] not in ACTION_STRATEGIES:
            raise ValueError("unknown assignment strategy")
        mapping[scene_id] = item
    expected = [scene.scene_id for scene in classified_spec.scenes]
    if len(set(expected)) != len(expected) or set(mapping) != set(expected):
        raise ValueError("assignment scene set differs from recovered facts")
    assigned = copy.deepcopy(classified_spec)
    for scene in assigned.scenes:
        scene.trigger_strategy = mapping[scene.scene_id]["trigger_strategy"]
        scene.action_strategy = mapping[scene.scene_id]["action_strategy"]
        scene.meta = {**scene.meta, "rq3_assignment_source": "llm_after_fact_recovery"}
    return assigned, {"usage": dict(response.usage or {}), "raw_response": response.content,
                      "system_prompt": prompt, "assignment_count": len(assignments)}
