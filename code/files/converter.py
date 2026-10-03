"""主转换器 — 五层编译管线: extract → classify → render → validate → gate。

使用示例:
    from scene_dsl_converter import SceneDSLConverter, ClaudeExtractor

    conv = SceneDSLConverter(extractor=ClaudeExtractor())
    dsl = conv.convert_text(prompt_text)

迁移计划:
    v2.0 引入五层管线：extract(LLM) → classify(StrategyClassifier)
    → render(DSLRendererV2) → validate(DSLValidator + ANTI_HOLLOW) → gate(ExamplesGate)
    此版本直接用 PipelineV2 替换旧的三层管线。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .extractor import ClaudeExtractor, ExtractorBase
from .renderer import DSLRenderer
from .schema import AgentSpec
from .validator import DSLValidator

# PipelineV2 在方法内延迟导入，避免 files/__init__ → converter → dsl_v2 → compiler → converter 的循环依赖

logger = logging.getLogger(__name__)


class SceneDSLConverter:
    """五层编译管线编排器。

    直接替换旧的三层 (extract → render → validate) 管线。
    """

    def __init__(
        self,
        extractor: Optional[ExtractorBase] = None,
        renderer: Optional[DSLRenderer] = None,
        validator: Optional[DSLValidator] = None,
        strict: bool = True,
    ):
        self.extractor = extractor or ClaudeExtractor()
        self.renderer = renderer or DSLRenderer()  # 保留向后兼容
        self.validator = validator or DSLValidator()
        self.strict = strict
        # PipelineV2 延迟初始化（使用时才导入）
        self._pipeline = None
        # 保留最后一次 PipelineV2Result，供上层获取中间数据
        self._last_pipeline_result = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from dsl_v2.pipeline import PipelineV2
            self._pipeline = PipelineV2()
        return self._pipeline

    def convert_text(self, document_text: str) -> str:
        """五层管线编译。

        extract(LLM) → classify(StrategyClassifier)
        → render(DSLRendererV2) → validate(含 ANTI_HOLLOW + 旧语法检查) → gate(ExamplesGate)
        """
        spec = self.extractor.extract(document_text)
        result = self.pipeline.compile(spec)
        self._last_pipeline_result = result  # 保留中间数据
        if not result.success:
            raise ValueError(
                f"DSL 编译失败: {result.error_message}"
            )
        # 第4层验证：PipelineV2 内部 ANTI_HOLLOW + 旧 DSLValidator 语法检查
        validation_result = self.validator.validate(result.dsl_code)
        if self.strict and not validation_result.ok:
            raise ValueError(
                f"DSL 验证失败: {validation_result.errors}"
            )
        return result.dsl_code

    def convert_from_spec(self, spec: AgentSpec) -> str:
        """从 AgentSpec 直接编译（跳过 LLM 抽取）。"""
        result = self.pipeline.compile(spec)
        self._last_pipeline_result = result  # 保留中间数据
        if not result.success:
            raise ValueError(
                f"DSL 编译失败: {result.error_message}"
            )
        # 旧 DSLValidator 语法检查（去重后 block_id 唯一性通过）
        validation_result = self.validator.validate(result.dsl_code)
        if self.strict and not validation_result.ok:
            raise ValueError(
                f"DSL 验证失败: {validation_result.errors}"
            )
        return result.dsl_code

    def convert_file(self, input_path: str | Path, output_path: str | Path) -> str:
        with Path(input_path).open("r", encoding="utf-8") as f:
            text = f.read()
        dsl = self.convert_text(text)
        with Path(output_path).open("w", encoding="utf-8") as f:
            f.write(dsl)
        return dsl

    def convert_file_with_trace(
        self,
        input_path: str | Path,
        output_dsl_path: str | Path,
        output_json_path: str | Path,
    ) -> str:
        with Path(input_path).open("r", encoding="utf-8") as f:
            text = f.read()
        spec = self.extractor.extract(text)
        with Path(output_json_path).open("w", encoding="utf-8") as f:
            f.write(spec.model_dump_json(indent=2))
        dsl = self.convert_from_spec(spec)
        with Path(output_dsl_path).open("w", encoding="utf-8") as f:
            f.write(dsl)
        return dsl
