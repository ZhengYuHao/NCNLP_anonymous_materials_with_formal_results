"""DSL 验证与增强管道 - 整合 DSL 静态验证、语义检测和增强"""
from dataclasses import dataclass, field
from typing import List, Optional, Any
from .dsl_validator import DSLValidator, ValidationResult, ValidationError, ValidationWarning
from .semantic_gap_detector import SemanticGap, SemanticGapDetector
from .dsl_enricher import DSLEnricher
import logging

logger = logging.getLogger(__name__)


@dataclass
class DSLEnrichmentResult:
    """DSL 增强处理结果"""
    status: str = "pending"  # pending | success | enriched | failed
    static_validation: Optional[ValidationResult] = None
    detected_gaps: List[SemanticGap] = field(default_factory=list)
    enriched_dsl: str = ""
    enriched_validation: Optional[ValidationResult] = None
    errors: List[Any] = field(default_factory=list)
    warnings: List[Any] = field(default_factory=list)

    def __str__(self):
        return (f"DSLEnrichmentResult(status={self.status}, "
                f"gaps={len(self.detected_gaps)}, "
                f"static_errors={len(self.static_validation.errors) if self.static_validation else 0})")


class DSLEnrichmentPipeline:
    """DSL 验证与增强管道"""

    def __init__(self, llm_client):
        """
        初始化 DSL 增强管道

        Args:
            llm_client: LLM 客户端实例
        """
        self.validator = DSLValidator()
        self.detector = SemanticGapDetector(llm_client)
        self.enricher = DSLEnricher(llm_client)
        self.llm = llm_client

    async def process(
        self,
        dsl_code: str,
        original_prompt: str = "",
        skip_llm_if_valid: bool = True
    ) -> DSLEnrichmentResult:
        """
        处理 DSL 代码：验证 → 检测缺失 → 增强

        Args:
            dsl_code: 待处理的 DSL 代码
            original_prompt: 原始自然语言需求
            skip_llm_if_valid: 如果静态验证通过且无高严重性问题，是否跳过 LLM 增强

        Returns:
            处理结果
        """
        result = DSLEnrichmentResult()
        original_dsl = dsl_code

        # Step 1: 静态验证
        logger.info("[DSL增强] Step 1: 执行静态验证...")
        validation_result = self.validator.validate(dsl_code)
        result.static_validation = validation_result

        if not validation_result.is_valid:
            # 有静态错误，标记失败但仍然尝试增强
            logger.warning(f"[DSL增强] 静态验证失败: {validation_result.errors}")

        # 记录静态警告
        result.warnings.extend(validation_result.warnings)

        # Step 2: 语义缺失检测（LLM）
        logger.info("[DSL增强] Step 2: 检测语义缺失...")
        gaps = await self.detector.detect(dsl_code, original_prompt)
        result.detected_gaps = gaps

        # 按严重性分类
        high_gaps = [g for g in gaps if g.severity == 'high']
        medium_gaps = [g for g in gaps if g.severity == 'medium']
        low_gaps = [g for g in gaps if g.severity == 'low']

        logger.info(f"[DSL增强] 检测到 {len(gaps)} 个语义缺失: "
                    f"高={len(high_gaps)}, 中={len(medium_gaps)}, 低={len(low_gaps)}")

        # 打印高严重性缺失详情
        if high_gaps:
            for gap in high_gaps:
                logger.warning(f"[DSL增强] 高严重性问题 [{gap.gap_type}]: {gap.description}")

        # 检查是否需要增强
        need_enrichment = False

        # 如果有高严重性缺失，必须增强
        if high_gaps:
            need_enrichment = True

        # 如果设置了 skip_llm_if_valid 且静态验证通过，可以跳过
        if skip_llm_if_valid and validation_result.is_valid and not high_gaps:
            logger.info("[DSL增强] 静态验证通过且无高严重性问题，跳过 LLM 增强")
            result.status = "success"
            result.enriched_dsl = dsl_code
            return result

        # Step 3: LLM 增强 DSL
        if need_enrichment and gaps:
            logger.info("[DSL增强] Step 3: 执行 LLM 增强...")
            enriched_dsl = await self.enricher.enrich_iterative(
                dsl_code,
                original_prompt,
                max_iterations=3
            )

            # enrich_iterative 返回的是元组 (dsl, gaps)
            if isinstance(enriched_dsl, tuple):
                result.enriched_dsl, remaining_gaps = enriched_dsl
            else:
                result.enriched_dsl = enriched_dsl

            # 验证增强后的 DSL 是否有效
            logger.info("[DSL增强] Step 4: 验证增强后的 DSL...")
            enriched_validation = self.validator.validate(result.enriched_dsl)
            result.enriched_validation = enriched_validation

            if enriched_validation.is_valid:
                logger.info("[DSL增强] 增强后 DSL 静态验证通过")
                result.status = "enriched"

                # 对比增强前后的差距
                high_reduction = len(high_gaps)
                if high_reduction > 0:
                    logger.info(f"[DSL增强] 成功修复 {high_reduction} 个高严重性问题")
            else:
                # 增强后仍然有错误
                logger.warning(f"[DSL增强] 增强后 DSL 仍有 {len(enriched_validation.errors)} 个错误")
                result.errors.extend(enriched_validation.errors)

                # 检查增强是否真的改进了
                enriched_high_gaps = [g for g in result.detected_gaps if g.severity == 'high']
                if len(enriched_high_gaps) < len(high_gaps):
                    # 有改进，使用增强后的版本
                    result.status = "partial_enriched"
                else:
                    # 没有改进或更糟，使用原始版本
                    logger.warning("[DSL增强] 增强可能没有改善，使用原始 DSL")
                    result.enriched_dsl = original_dsl
                    result.status = "enrichment_failed"
        else:
            # 没有高严重性问题，不需要增强
            result.status = "success"
            result.enriched_dsl = dsl_code

        return result

    def process_sync(
        self,
        dsl_code: str,
        original_prompt: str = "",
        skip_llm_if_valid: bool = True
    ) -> DSLEnrichmentResult:
        """
        同步版本的处理

        Args:
            dsl_code: 待处理的 DSL 代码
            original_prompt: 原始自然语言需求
            skip_llm_if_valid: 如果静态验证通过且无高严重性问题，是否跳过 LLM 增强

        Returns:
            处理结果
        """
        result = DSLEnrichmentResult()
        original_dsl = dsl_code

        # Step 1: 静态验证
        logger.info("[DSL增强] Step 1: 执行静态验证...")
        validation_result = self.validator.validate(dsl_code)
        result.static_validation = validation_result

        if not validation_result.is_valid:
            logger.warning(f"[DSL增强] 静态验证失败: {validation_result.errors}")

        result.warnings.extend(validation_result.warnings)

        # Step 2: 语义缺失检测（同步调用 LLM）
        logger.info("[DSL增强] Step 2: 检测语义缺失...")
        gaps = self.detector.detect_sync(dsl_code, original_prompt)
        result.detected_gaps = gaps

        high_gaps = [g for g in gaps if g.severity == 'high']
        medium_gaps = [g for g in gaps if g.severity == 'medium']
        low_gaps = [g for g in gaps if g.severity == 'low']

        logger.info(f"[DSL增强] 检测到 {len(gaps)} 个语义缺失: "
                    f"高={len(high_gaps)}, 中={len(medium_gaps)}, 低={len(low_gaps)}")

        if skip_llm_if_valid and validation_result.is_valid and not high_gaps:
            result.status = "success"
            result.enriched_dsl = dsl_code
            return result

        # Step 3: LLM 增强 DSL（同步）
        if high_gaps and gaps:
            logger.info("[DSL增强] Step 3: 执行 LLM 增强...")
            enriched_dsl = self.enricher.enrich_sync(dsl_code, original_prompt, high_gaps)
            result.enriched_dsl = enriched_dsl

            # 验证增强后的 DSL
            enriched_validation = self.validator.validate(enriched_dsl)
            result.enriched_validation = enriched_validation

            if enriched_validation.is_valid:
                result.status = "enriched"
            else:
                result.errors.extend(enriched_validation.errors)
                result.status = "enrichment_failed"
        else:
            result.status = "success"
            result.enriched_dsl = dsl_code

        return result

    async def validate_only(self, dsl_code: str) -> ValidationResult:
        """
        仅执行静态验证，不进行增强

        Args:
            dsl_code: 待验证的 DSL 代码

        Returns:
            验证结果
        """
        return self.validator.validate(dsl_code)

    def get_gap_summary(self, gaps: List[SemanticGap]) -> str:
        """
        获取语义缺失摘要

        Args:
            gaps: 语义缺失列表

        Returns:
            摘要字符串
        """
        if not gaps:
            return "未检测到语义缺失"

        high = len([g for g in gaps if g.severity == 'high'])
        medium = len([g for g in gaps if g.severity == 'medium'])
        low = len([g for g in gaps if g.severity == 'low'])

        by_type = {}
        for gap in gaps:
            by_type[gap.gap_type] = by_type.get(gap.gap_type, 0) + 1

        type_summary = ", ".join([f"{k}={v}" for k, v in by_type.items()])

        return (f"共检测到 {len(gaps)} 个语义缺失: "
                f"高={high}, 中={medium}, 低={low} | "
                f"类型分布: {type_summary}")