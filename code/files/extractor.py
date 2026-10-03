"""LLM 抽取器。

负责:把原始提示词文本 → AgentSpec 结构化对象。

提供两种实现:
- ClaudeExtractor: 调用 Anthropic Claude API
- MockExtractor: 从预存 JSON 文件直接加载,便于离线测试/CI
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_messages
from .schema import AgentSpec


class ExtractorBase(ABC):
    @abstractmethod
    def extract(self, document_text: str) -> AgentSpec: ...

    @staticmethod
    def _parse(raw_json_text: str) -> AgentSpec:
        """容忍 LLM 输出偶尔带 ``` 围栏的情况。"""
        text = raw_json_text.strip()
        if text.startswith("```"):
            # 剥掉可能的 ```json ... ```
            text = text.lstrip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip().rstrip("`").strip()
        data = json.loads(text)
        return AgentSpec.model_validate(data).sort_scenes()


class ClaudeExtractor(ExtractorBase):
    """使用 Anthropic Claude 做抽取。

    用法:
        extractor = ClaudeExtractor(api_key=os.environ["ANTHROPIC_API_KEY"])
        spec = extractor.extract(prompt_text)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-7",
        max_tokens: int = 8192,
    ):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "需要安装 anthropic 包: pip install anthropic"
            ) from e

        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def extract(self, document_text: str) -> AgentSpec:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=build_extraction_messages(document_text),
        )
        # 把所有 text block 拼起来作为 raw JSON
        raw_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        raw = "".join(raw_parts)
        return self._parse(raw)


class MockExtractor(ExtractorBase):
    """从预存 JSON 文件加载抽取结果。

    用于:
    - 单元测试
    - 离线环境(无 API key)
    - 人工微调抽取结果后,跳过 LLM 调用直接渲染
    """

    def __init__(self, fixture_path: str | Path):
        self.fixture_path = Path(fixture_path)

    def extract(self, document_text: str) -> AgentSpec:  # noqa: ARG002
        with self.fixture_path.open("r", encoding="utf-8") as f:
            return self._parse(f.read())
