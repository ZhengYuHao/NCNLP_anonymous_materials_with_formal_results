"""RQ3 Coarse M3: recover facts and assign strategies in one model call."""

from __future__ import annotations

import copy
import json
from dataclasses import fields

from dsl_v2.fact_types import (
    ActionSpec,
    CompressibilityScore,
    ConfigItem,
    ExclusionSpec,
    FactScene,
    FactSpec,
    PolicyItem,
    SemanticVector,
    TriggerSpec,
)
from formal_experiment.staged_assignment import ACTION_STRATEGIES, TRIGGER_STRATEGIES
from baselines.b0_direct import parse_json_output


def _strict_dataclass(cls, payload):
    if not isinstance(payload, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    allowed = {item.name for item in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{cls.__name__} contains unknown fields: {sorted(unknown)}")
    return cls(**payload)


def hydrate_fact_spec(payload) -> FactSpec:
    if not isinstance(payload, dict):
        raise ValueError("coarse M3 fact_spec must be an object")
    root = copy.deepcopy(payload)
    scenes = []
    for item in root.get("scenes", []) or []:
        if not isinstance(item, dict):
            raise ValueError("coarse M3 scenes must contain objects")
        scene = copy.deepcopy(item)
        scene["triggers"] = _strict_dataclass(TriggerSpec, scene.get("triggers") or {})
        scene["exclusions"] = _strict_dataclass(ExclusionSpec, scene.get("exclusions") or {})
        scene["action"] = _strict_dataclass(ActionSpec, scene.get("action") or {})
        scene["compressibility"] = _strict_dataclass(
            CompressibilityScore, scene.get("compressibility") or {}
        )
        scene["semantic_vector"] = _strict_dataclass(
            SemanticVector, scene.get("semantic_vector") or {}
        )
        scenes.append(_strict_dataclass(FactScene, scene))
    root["scenes"] = scenes
    root["configs"] = [
        _strict_dataclass(ConfigItem, item) for item in root.get("configs", []) or []
    ]
    root["policies"] = [
        _strict_dataclass(PolicyItem, item) for item in root.get("policies", []) or []
    ]
    fact_spec = _strict_dataclass(FactSpec, root)
    if not fact_spec.scenes:
        raise ValueError("coarse M3 must recover at least one scene")
    scene_ids = [scene.scene_id for scene in fact_spec.scenes]
    if any(type(scene_id) is not int for scene_id in scene_ids) or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("coarse M3 scene identities must be unique integers")
    return fact_spec


def validate_assignments(assignments, scene_ids: list[int]) -> dict[int, dict]:
    if not isinstance(assignments, list):
        raise ValueError("coarse M3 must contain an assignments array")
    mapping = {}
    for item in assignments:
        if not isinstance(item, dict) or set(item) != {
            "scene_id", "trigger_strategy", "action_strategy"
        }:
            raise ValueError("coarse M3 assignment may only contain the three strategy fields")
        scene_id = item["scene_id"]
        if type(scene_id) is not int or scene_id in mapping:
            raise ValueError("duplicate or invalid coarse M3 scene identity")
        if item["trigger_strategy"] not in TRIGGER_STRATEGIES:
            raise ValueError("unknown coarse M3 trigger strategy")
        if item["action_strategy"] not in ACTION_STRATEGIES:
            raise ValueError("unknown coarse M3 action strategy")
        mapping[scene_id] = copy.deepcopy(item)
    if set(mapping) != set(scene_ids):
        raise ValueError("coarse M3 assignment scene set differs from recovered facts")
    return mapping


def recover_and_assign(normalized: str, public_interfaces: list[dict], client):
    system_prompt = (
        "In one model call, recover the complete workflow facts and assign an implementation "
        "strategy to every recovered scene. Return JSON only with exactly two top-level fields: "
        "fact_spec and assignments. fact_spec must follow the supplied FactSpec template. Preserve "
        "the requirement without adding behavior. Each externally observable operation and each "
        "deterministic transform between operations must be a separate ordered scene. Copy public "
        "dependency_type and operation names exactly. Store operation bindings under "
        "action.structured_op.public_operations and deterministic local statements under "
        "action.structured_op.local_program. assignments must contain exactly scene_id, "
        "trigger_strategy, and action_strategy for every scene.\n"
        "Allowed trigger strategies: " + ", ".join(sorted(TRIGGER_STRATEGIES)) + "\n"
        "Allowed action strategies: " + ", ".join(sorted(ACTION_STRATEGIES))
    )
    template = {
        "agent_name": "",
        "agent_description": "",
        "persona_role": "",
        "persona_capabilities": [],
        "constraints": [],
        "inputs": [],
        "outputs": [],
        "scenes": [{
            "scene_id": 1,
            "block_id": "",
            "source_ref": "",
            "block_description": "",
            "agent_expression": "",
            "triggers": {},
            "exclusions": {},
            "action": {
                "raw_text": "",
                "action_type": "assign",
                "structured_op": {"public_operations": [], "local_program": []},
                "outputs": [],
            },
            "raw_requirement_text": "",
            "logic_flow": "",
            "side_effects": [],
            "raw_context": "",
            "preconditions": [],
            "fallback": "",
            "action_sequence": [],
            "nested_logic": None,
            "meta_rules": [],
            "compressibility": {},
            "semantic_vector": {},
        }],
        "fallback_scene_id": 0,
        "fallback_agent_expression": "\"unknown\"",
        "examples": [],
        "configs": [],
        "policies": [],
    }
    user_content = json.dumps({
        "natural_language_requirement": normalized,
        "public_interfaces": public_interfaces,
        "fact_spec_template": template,
    }, ensure_ascii=False, sort_keys=True)
    response = client.call(
        system_prompt=system_prompt,
        user_content=user_content,
        temperature=0.0,
        max_tokens=16000,
    )
    parsed = parse_json_output(response.content)
    if not isinstance(parsed, dict) or set(parsed) != {"fact_spec", "assignments"}:
        raise ValueError("coarse M3 response must contain exactly fact_spec and assignments")
    fact_spec = hydrate_fact_spec(parsed["fact_spec"])
    mapping = validate_assignments(
        parsed["assignments"], [scene.scene_id for scene in fact_spec.scenes]
    )
    return fact_spec, mapping, {
        "usage": dict(response.usage or {}),
        "raw_response": response.content,
        "system_prompt": system_prompt,
        "model_call_count": 1,
    }


def apply_assignments(classified_spec, mapping: dict[int, dict]):
    assigned = copy.deepcopy(classified_spec)
    scene_ids = [scene.scene_id for scene in assigned.scenes]
    if set(scene_ids) != set(mapping):
        raise ValueError("coarse M3 classified scene set differs from joint response")
    for scene in assigned.scenes:
        item = mapping[scene.scene_id]
        scene.trigger_strategy = item["trigger_strategy"]
        scene.action_strategy = item["action_strategy"]
        scene.meta = {**scene.meta, "rq3_assignment_source": "joint_fact_and_assignment_call"}
    return assigned
