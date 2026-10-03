"""
Prompt 处理流水线 (Pipeline)
串联 Prompt 1.0 和 Prompt 2.0 的完整处理流程
"""

import time
from datetime import datetime
from typing import Optional, Dict, Any

from logger import info, warning, error, debug
from llm_client import UnifiedLLMClient, create_llm_client
from data_models import (
    ProcessingMode, ProcessingStatus, Prompt10Result, Prompt20Result,
    FullPipelineResult, generate_id, get_timestamp
)
from history_manager import HistoryManager, PipelineHistory

# 导入处理模块
from prompt_preprocessor import PromptPreprocessor
from prompt_structurizer import PromptStructurizer


class PromptPipeline:
    """
    Prompt 处理流水线
    
    流程:
    原始输入 → [Prompt 1.0 预处理] → 标准化文本 → [Prompt 2.0 结构化] → 模板 + 变量表
    """
    
    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.DICTIONARY,
        term_mapping: Optional[Dict[str, str]] = None,
        ambiguity_blacklist: Optional[list] = None,
        use_mock_llm: bool = False,
        enable_deep_check: bool = True,
        enable_cache: bool = False
    ):
        """
        初始化流水线

        Args:
            mode: 处理模式
            term_mapping: 术语映射表
            ambiguity_blacklist: 歧义词黑名单
            use_mock_llm: 是否使用模拟LLM
            enable_deep_check: 是否启用深度歧义检测
            enable_cache: 是否启用缓存
        """
        # 创建共享的 LLM 客户端（支持缓存）
        self.llm_client = create_llm_client(use_mock=use_mock_llm, enable_cache=enable_cache)

        # 初始化阶段1处理器
        self.preprocessor = PromptPreprocessor(
            mode=mode,
            term_mapping=term_mapping,
            ambiguity_blacklist=ambiguity_blacklist,
            llm_client=self.llm_client,
            enable_deep_check=enable_deep_check
        )

        # 初始化阶段2处理器
        self.structurizer = PromptStructurizer(
            llm_client=self.llm_client,
            use_mock=use_mock_llm
        )

        self.use_mock = use_mock_llm
        self.enable_cache = enable_cache
        self.history_manager = HistoryManager()
    
    def run(
        self,
        raw_input: str,
        save_history: bool = True,
        stop_on_ambiguity: bool = False
    ) -> FullPipelineResult:
        """
        执行完整流水线
        
        Args:
            raw_input: 用户原始输入
            save_history: 是否保存处理历史
            stop_on_ambiguity: 是否在检测到歧义时停止
            
        Returns:
            FullPipelineResult: 完整处理结果
        """
        pipeline_id = generate_id()
        timestamp = get_timestamp()
        start_time = time.time()
        
        info(f"\n{'='*70}")
        info(f"【流水线启动】 ID: {pipeline_id}")
        info(f"{'='*70}")
        info(f"输入: {raw_input}")
        info(f"{'='*70}\n")
        
        result = FullPipelineResult(
            pipeline_id=pipeline_id,
            timestamp=timestamp,
            raw_input=raw_input
        )
        
        try:
            # ===== 阶段 1: Prompt 1.0 预处理 =====
            info("\n>>> 阶段 1: Prompt 1.0 预处理")
            info("-" * 50)
            
            prompt10_result = self.preprocessor.process(
                raw_input,
                save_history=save_history,
                show_comparison=False
            )
            result.prompt10_result = prompt10_result
            
            # 检查是否需要在歧义时停止
            if stop_on_ambiguity and prompt10_result.ambiguity_detected:
                warning(f"检测到歧义，流水线停止: {prompt10_result.ambiguity_details}")
                result.overall_status = ProcessingStatus.AMBIGUITY.value
                result.error_message = prompt10_result.ambiguity_details
                result.total_time_ms = int((time.time() - start_time) * 1000)
                return result
            
            info(f"阶段1完成 - 状态: {prompt10_result.status}")
            
            # ===== 阶段 2: Prompt 2.0 结构化 =====
            info("\n>>> 阶段 2: Prompt 2.0 结构化")
            info("-" * 50)
            
            prompt20_result = self.structurizer.process_from_prompt10(
                prompt10_result,
                save_history=save_history
            )
            result.prompt20_result = prompt20_result
            
            # 设置最终输出
            result.final_template = prompt20_result.template_text
            result.final_variables = prompt20_result.variable_registry
            result.overall_status = ProcessingStatus.SUCCESS.value
            
            info(f"阶段2完成 - 变量数量: {len(prompt20_result.variables)}")
            
        except Exception as e:
            error(f"流水线执行失败: {e}")
            result.overall_status = ProcessingStatus.ERROR.value
            result.error_message = str(e)
        
        result.total_time_ms = int((time.time() - start_time) * 1000)
        
        # 保存完整流水线历史记录
        if save_history:
            self._save_pipeline_history(result)
        
        # 打印最终结果
        self._print_result(result)
        
        return result
    
    def _save_pipeline_history(self, result: FullPipelineResult):
        """保存完整流水线历史记录"""
        # 统计变量类型
        type_stats = {}
        if result.prompt20_result:
            for var in result.prompt20_result.variables:
                dtype = var.data_type
                type_stats[dtype] = type_stats.get(dtype, 0) + 1

        # 收集缓存统计
        cache_stats = self.llm_client.get_cache_stats() if self.enable_cache else {}

        # 收集优化统计
        prompt10_rule_stats = {}
        if result.prompt10_result and hasattr(result.prompt10_result, 'rule_engine_stats'):
            prompt10_rule_stats = result.prompt10_result.rule_engine_stats

        history = PipelineHistory(
            pipeline_id=result.pipeline_id,
            timestamp=result.timestamp,
            raw_input=result.raw_input,

            # 阶段1结果
            prompt10_id=result.prompt10_result.id if result.prompt10_result else "",
            prompt10_original=result.prompt10_result.original_text if result.prompt10_result else "",
            prompt10_processed=result.prompt10_result.processed_text if result.prompt10_result else "",
            prompt10_mode=result.prompt10_result.mode if result.prompt10_result else "",
            prompt10_steps=[s.to_dict() for s in result.prompt10_result.steps] if result.prompt10_result else [],
            prompt10_terminology_changes=result.prompt10_result.terminology_changes if result.prompt10_result else {},
            prompt10_ambiguity_detected=result.prompt10_result.ambiguity_detected if result.prompt10_result else False,
            prompt10_status=result.prompt10_result.status if result.prompt10_result else "",
            prompt10_time_ms=result.prompt10_result.processing_time_ms if result.prompt10_result else 0,

            # 阶段2结果
            prompt20_id=result.prompt20_result.id if result.prompt20_result else "",
            prompt20_template=result.prompt20_result.template_text if result.prompt20_result else "",
            prompt20_variables=result.prompt20_result.variable_registry if result.prompt20_result else [],
            prompt20_variable_count=len(result.prompt20_result.variables) if result.prompt20_result else 0,
            prompt20_type_stats=type_stats,
            prompt20_extraction_log=result.prompt20_result.extraction_log if result.prompt20_result else [],
            prompt20_time_ms=result.prompt20_result.processing_time_ms if result.prompt20_result else 0,

            # 整体状态
            overall_status=result.overall_status,
            total_time_ms=result.total_time_ms,
            error_message=result.error_message,

            # 优化统计（极窄化LLM）
            prompt10_rule_stats=prompt10_rule_stats,
            prompt20_optimization_stats={},  # 从 prompt20_result 中提取
            prompt30_optimization_stats={},
            total_cache_hits=cache_stats.get('hits', 0),
            total_cache_misses=cache_stats.get('misses', 0),
            cache_hit_rate=cache_stats.get('hit_rate', 0.0)
        )

        try:
            self.history_manager.save_pipeline_history(history)
            info(f"流水线历史记录已保存: {result.pipeline_id}")
        except Exception as e:
            warning(f"保存流水线历史记录失败: {e}")
    
    def _print_result(self, result: FullPipelineResult):
        """打印流水线结果"""
        info(f"\n{'='*70}")
        info(f"【流水线完成】 ID: {result.pipeline_id}")
        info(f"{'='*70}")
        
        info(f"\n状态: {result.overall_status}")
        info(f"总耗时: {result.total_time_ms}ms")
        
        if result.prompt10_result:
            info(f"\n--- Prompt 1.0 结果 ---")
            info(f"  原始: {result.prompt10_result.original_text}")
            info(f"  处理后: {result.prompt10_result.processed_text}")
            if result.prompt10_result.terminology_changes:
                info(f"  术语替换: {result.prompt10_result.terminology_changes}")
        
        if result.prompt20_result:
            info(f"\n--- Prompt 2.0 结果 ---")
            info(f"  模板: {result.prompt20_result.template_text}")
            info(f"  变量数: {len(result.prompt20_result.variables)}")
            for var in result.prompt20_result.variables:
                info(f"    - {var.name}: {var.value} ({var.data_type})")
        
        if result.error_message:
            error(f"\n错误信息: {result.error_message}")
        
        info(f"\n{'='*70}\n")


# ============================================================================
# 便捷函数
# ============================================================================

def process_prompt(
    raw_input: str,
    mode: str = "dictionary",
    term_mapping: Optional[Dict[str, str]] = None,
    use_mock: bool = False
) -> FullPipelineResult:
    """
    便捷函数：一键处理 prompt
    
    Args:
        raw_input: 原始输入
        mode: 处理模式 ("dictionary", "smart", "hybrid")
        term_mapping: 术语映射表
        use_mock: 是否使用模拟LLM
        
    Returns:
        完整处理结果
    """
    processing_mode = ProcessingMode(mode)
    
    pipeline = PromptPipeline(
        mode=processing_mode,
        term_mapping=term_mapping,
        use_mock_llm=use_mock
    )
    
    return pipeline.run(raw_input)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 定义术语映射
    TERM_MAPPING = {
        "大模型": "大型语言模型(LLM)",
        "套壳": "基于API的封装应用",
        "RAG": "检索增强生成(RAG)",
        "chain": "处理链(Chain)",
    }
    
    # 定义歧义词黑名单
    AMBIGUITY_BLACKLIST = ["这个", "那个", "它", "他们"]
    
    info("=" * 70)
    info("Prompt 处理流水线演示")
    info("=" * 70)
    
    # 场景1: 正常流程
    info("\n\n>>> 场景1: 正常处理流程")
    pipeline = PromptPipeline(
        mode=ProcessingMode.DICTIONARY,
        term_mapping=TERM_MAPPING,
        ambiguity_blacklist=AMBIGUITY_BLACKLIST,
        use_mock_llm=False  # 使用真实LLM
    )
    
    result = pipeline.run(
        "请为一位有3年经验的Java程序员生成一个为期2周的Python学习计划"
    )
    
    # 场景2: 使用便捷函数
    info("\n\n>>> 场景2: 使用便捷函数")
    result2 = process_prompt(
        "帮我搞一个RAG的大模型应用",
        mode="dictionary",
        term_mapping=TERM_MAPPING,
        use_mock=False
    )
    
    # 打印 JSON 结果
    info("\n>>> 完整 JSON 输出:")
    info(result2.to_json())
