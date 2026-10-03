"""
统一 LLM 接口的抽取器 - 使用 llm_client.py

把原始提示词文本 → AgentSpec 结构化对象
"""
from __future__ import annotations

import json
from typing import Optional

try:
    from llm_client import UnifiedLLMClient, create_llm_client
    LLM_CLIENT_AVAILABLE = True
except ImportError:
    LLM_CLIENT_AVAILABLE = False
    UnifiedLLMClient = None

from files.extractor import ExtractorBase
from files.schema import AgentSpec, SubScenario
from files.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_messages


def _fix_extraction_data(data: dict) -> dict:
    """修复 LLM 返回的不完整数据"""
    type_aliases = {
        "string": "String",
        "str": "String",
        "integer": "Integer",
        "int": "Integer",
        "float": "Float",
        "number": "Float",
        "boolean": "Boolean",
        "bool": "Boolean",
        "list": "List",
        "array": "List",
        "dict": "Dict",
        "object": "Dict",
        "map": "Dict",
        "json": "Dict",
        "any": "Any",
    }

    def normalize_typed_items(items: object) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict) or "type" not in item:
                continue
            normalized = type_aliases.get(str(item["type"]).strip().lower())
            if normalized:
                item["type"] = normalized

    normalize_typed_items(data.get("inputs"))
    normalize_typed_items(data.get("outputs"))
    for scene in data.get("scenes", []) or []:
        for sub in scene.get("sub_scenarios", []) or []:
            semantic = sub.get("semantic")
            if isinstance(semantic, dict):
                normalize_typed_items(semantic.get("input_vars"))
                normalize_typed_items(semantic.get("output_vars"))

    if "scenes" not in data or not data["scenes"]:
        if "scenes" not in data:
            data["scenes"] = []
        if not data["scenes"]:
            data["scenes"].append({
                "scene_id": 0,
                "block_id": "b0_default",
                "block_description": "默认兜底场景",
                "name": "默认场景",
                "priority": 0,
                "agent_expression": '"未知智能体"',
                "trigger_expression": "",
                "exclusion_expression": "",
                "sub_scenarios": [
                    {
                        "name": "默认子场景",
                        "trigger_keywords": ["default"],
                        "exclusion_keywords": [],
                        "semantic": None
                    }
                ]
            })
        return data

    for scene in data["scenes"]:
        if not scene.get("sub_scenarios"):
            scene["sub_scenarios"] = [
                {
                    "name": "默认子场景",
                    "trigger_keywords": ["default"],
                    "exclusion_keywords": [],
                    "semantic": None
                }
            ]
        for sub in scene.get("sub_scenarios", []):
            if not sub.get("trigger_keywords"):
                sub["trigger_keywords"] = ["default"]

    return data


class UnifiedExtractor(ExtractorBase):
    """使用统一 llm_client.py 做抽取。

    用法:
        extractor = UnifiedExtractor()
        spec = extractor.extract(prompt_text)

        # 或指定模型
        extractor = UnifiedExtractor(model="gpt-5.4", temperature=0.1)
    """

    def __init__(
        self,
        llm_client: Optional[UnifiedLLMClient] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 32760,
    ):
        if llm_client is None:
            if not LLM_CLIENT_AVAILABLE:
                raise RuntimeError("llm_client.py 不可用")
            llm_client = create_llm_client()
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_usage: Optional[Dict[str, int]] = None

    def extract(
        self,
        document_text: str,
        public_interfaces: Optional[list[dict]] = None,
    ) -> AgentSpec:
        messages = build_extraction_messages(document_text, public_interfaces=public_interfaces)
        user_content = messages[0]["content"]

        response = self.llm_client.call(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_content=user_content,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if hasattr(response, 'usage') and response.usage:
            self.last_usage = response.usage

        return self._parse(response.content if hasattr(response, 'content') else str(response))

    @staticmethod
    def _parse(raw_json_text: str) -> AgentSpec:
        text = raw_json_text.strip()
        if text.startswith("```"):
            text = text.lstrip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip().rstrip("`").strip()
        data = json.loads(text)
        data = _fix_extraction_data(data)
        for input_spec in data.get("inputs", []):
            if "default" in input_spec and input_spec["default"] is not None:
                input_spec["default"] = str(input_spec["default"])
        return AgentSpec.model_validate(data).sort_scenes()


class MockExtractor(ExtractorBase):
    """从预存 JSON 文件加载抽取结果。"""

    def __init__(self, fixture_path: str):
        self.fixture_path = fixture_path

    def extract(self, document_text: str) -> AgentSpec:
        with open(self.fixture_path, encoding="utf-8") as f:
            return self._parse(f.read())

    @staticmethod
    def _parse(raw_json_text: str) -> AgentSpec:
        text = raw_json_text.strip()
        if text.startswith("```"):
            text = text.lstrip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip().rstrip("`").strip()
        data = json.loads(text)
        data = _fix_extraction_data(data)
        for input_spec in data.get("inputs", []):
            if "default" in input_spec and input_spec["default"] is not None:
                input_spec["default"] = str(input_spec["default"])
        return AgentSpec.model_validate(data).sort_scenes()
