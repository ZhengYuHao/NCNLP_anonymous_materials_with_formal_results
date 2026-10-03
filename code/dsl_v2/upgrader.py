"""
Upgrader — BLOCK → SEMANTIC_BLOCK 单向升级器 + 场景簇升级。

职责：
1. 接收失败的 BLOCK 场景（由 ExamplesGate 定位）
2. 调用 LLM 生成 SEMANTIC_BLOCK 的 prompt
3. 将场景的 trigger_strategy 升级为 SEMANTIC_BLOCK_CLASSIFICATION
4. 记录完整的升级溯源元数据
5. 支持多场景合并为 RAW_SEMANTIC（场景簇升级）

升级是单向的——BLOCK 升级到 SEMANTIC_BLOCK 后不可降级。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .fact_types import SceneClassification


# ============================================================================
# 默认 LLM prompt 生成器
# ============================================================================

UPGRADE_PROMPT_TEMPLATE = """你是一个 DSL 编译器。一个 BLOCK 因为关键词覆盖不全而升级为 SEMANTIC_BLOCK。

## 失败的 BLOCK 信息
- block_id: {block_id}
- 描述: {block_description}
- 触发条件原文: {trigger_raw_text}
- 当前关键词: {trigger_keywords}
- 排除条件原文: {exclusion_raw_text}
- 排除关键词: {exclusion_keywords}
- 动作类型: {action_type}
- 失败原因: {failure_reason}

## 任务
生成一个 SEMANTIC_BLOCK(classification) 的 prompt。

## 要求
1. 任务类型: classification
2. 输入: user_input (String) — 用户的原始输入
3. 输出: scene (Integer) — 场景编号, confidence (Float) — 置信度
4. prompt 中必须包含具体的判断规则，参考上面的关键词
5. 排除条件也要写入 prompt 规则中
6. 输出格式为严格的 JSON，包含 task, input_vars, output_vars, prompt_template

## 输出格式
```json
{{
    "task": "classification",
    "input_vars": [{{"name": "user_input", "type": "String"}}],
    "output_vars": [
        {{"name": "scene", "type": "Integer"}},
        {{"name": "confidence", "type": "Float"}}
    ],
    "prompt_template": "你是场景分类器。..."
}}
```
"""


# ============================================================================
# 默认 LLM 回调——生产环境使用真实 LLM
# ============================================================================

def _default_llm_call(prompt: str) -> str:
    """默认 LLM 回调。没有配置时使用基于模板的生成。"""
    return json.dumps({
        "task": "classification",
        "input_vars": [{"name": "user_input", "type": "String"}],
        "output_vars": [
            {"name": "scene", "type": "Integer"},
            {"name": "confidence", "type": "Float"},
        ],
        "prompt_template": prompt,
    })


# ============================================================================
# Upgrader
# ============================================================================

class Upgrader:
    """BLOCK → SEMANTIC_BLOCK 单向升级器。

    Args:
        llm_call: LLM 调用回调。接收 prompt 字符串，返回 JSON 字符串。
                  默认使用模板生成（不调用真实 LLM）。
    """

    def __init__(self, llm_call: Optional[Callable[[str], str]] = None):
        self.llm_call = llm_call or _default_llm_call

    def upgrade_scene(
        self,
        scene: SceneClassification,
        failure_reason: str = "",
    ) -> SceneClassification:
        """升级一个场景的触发策略为 SEMANTIC_BLOCK。

        如果已经是 SEMANTIC_BLOCK，不做改变（单向性保证）。
        """
        if scene.trigger_strategy.startswith("SEMANTIC_BLOCK"):
            return scene  # 单向性：不降级

        # 1. 构建 LLM prompt
        llm_prompt = UPGRADE_PROMPT_TEMPLATE.format(
            block_id=scene.block_id,
            block_description=scene.block_description or "",
            trigger_raw_text=scene.triggers.raw_text or "",
            trigger_keywords=", ".join(scene.triggers.extracted_keywords or []),
            exclusion_raw_text=scene.exclusions.raw_text or "",
            exclusion_keywords=", ".join(scene.exclusions.extracted_keywords or []),
            action_type=scene.action.action_type or "",
            failure_reason=failure_reason or "EXAMPLES 未通过",
        )

        # 2. 调用 LLM 生成 prompt
        response = self.llm_call(llm_prompt)
        try:
            llm_data = json.loads(response)
        except json.JSONDecodeError:
            llm_data = {
                "task": "classification",
                "input_vars": [{"name": "user_input", "type": "String"}],
                "output_vars": [
                    {"name": "scene", "type": "Integer"},
                    {"name": "confidence", "type": "Float"},
                ],
                "prompt_template": llm_prompt,
            }

        # 3. 构建升级后的 scene
        meta = dict(scene.meta)
        meta["upgraded_from"] = scene.trigger_strategy
        meta["upgrade_reason"] = failure_reason or "EXAMPLES 未通过"
        meta["upgrade_llm_prompt_generated"] = True
        meta["upgrade_semantic_task"] = llm_data.get("task", "classification")
        meta["upgrade_prompt_template"] = llm_data.get("prompt_template", "")
        meta["exclusions_preserved"] = json.dumps(
            scene.exclusions.extracted_keywords, ensure_ascii=False
        )

        return SceneClassification(
            scene_id=scene.scene_id,
            block_id=scene.block_id,
            block_description=scene.block_description,
            agent_expression=scene.agent_expression,
            source_ref=scene.source_ref,
            triggers=scene.triggers,
            exclusions=scene.exclusions,
            action=scene.action,
            trigger_strategy="SEMANTIC_BLOCK_CLASSIFICATION",
            action_strategy=scene.action_strategy,
            meta=meta,
        )

    def batch_upgrade(
        self,
        failed_scenes: list[tuple[SceneClassification, str]],
    ) -> list[SceneClassification]:
        """批量升级多个失败的 BLOCK。

        Args:
            failed_scenes: [(scene, failure_reason), ...]

        Returns:
            升级后的 SceneClassification 列表
        """
        return [
            self.upgrade_scene(scene, reason)
            for scene, reason in failed_scenes
        ]

    def upgrade_cluster(
        self,
        scenes: list[SceneClassification],
        failure_reason: str = "",
    ) -> list[SceneClassification]:
        """升级一组场景为合并的 RAW_SEMANTIC。

        将多个协作失败的场景合并为一个 RAW_SEMANTIC，
        PROMPT 包含所有相关场景的 raw_requirement_text。

        Args:
            scenes: 待合并的场景列表
            failure_reason: 失败原因

        Returns:
            [合并后的场景, ...被吸收的场景(ABSORBED)]
        """
        if len(scenes) <= 1:
            return [self.upgrade_scene(scenes[0], failure_reason)]

        # 合并所有场景的原文
        merged_raw = []
        merged_context = []
        for s in scenes:
            if s.raw_requirement_text:
                merged_raw.append(f"### 场景{s.scene_id}: {s.block_description}\n{s.raw_requirement_text}")
            if s.raw_context:
                merged_context.append(s.raw_context)

        # 取第一个场景作为主场景，合并信息
        primary = scenes[0]
        meta = dict(primary.meta)
        meta["upgraded_from"] = f"cluster({', '.join(s.block_id for s in scenes)})"
        meta["upgrade_reason"] = failure_reason or "跨BLOCK协作失败，合并为RAW_SEMANTIC"
        meta["cluster_members"] = [s.block_id for s in scenes]

        # 创建合并场景
        merged_scene = SceneClassification(
            scene_id=primary.scene_id,
            block_id=primary.block_id + "_cluster",
            block_description=" + ".join(s.block_description for s in scenes[:3]),
            agent_expression=primary.agent_expression,
            source_ref="; ".join(s.source_ref for s in scenes if s.source_ref),
            triggers=primary.triggers,
            exclusions=primary.exclusions,
            action=primary.action,
            trigger_strategy="RAW_SEMANTIC",
            action_strategy="SEMANTIC_BLOCK_GENERATION",
            meta=meta,
            raw_requirement_text="\n\n".join(merged_raw),
            raw_context="\n".join(merged_context),
            logic_flow="; ".join(s.logic_flow for s in scenes if s.logic_flow),
            side_effects=[e for s in scenes for e in (s.side_effects or [])],
        )

        # 其余场景标记为被合并
        absorbed = []
        for s in scenes[1:]:
            absorbed_meta = dict(s.meta)
            absorbed_meta["absorbed_into"] = merged_scene.block_id
            absorbed.append(SceneClassification(
                scene_id=s.scene_id,
                block_id=s.block_id,
                block_description=s.block_description,
                agent_expression=s.agent_expression,
                source_ref=s.source_ref,
                triggers=s.triggers,
                exclusions=s.exclusions,
                action=s.action,
                trigger_strategy="ABSORBED",
                action_strategy="ABSORBED",
                meta=absorbed_meta,
            ))

        return [merged_scene] + absorbed
