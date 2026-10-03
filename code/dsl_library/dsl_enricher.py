"""
DSL语义增强器
在DSL生成后，自动为函数调用填充实现代码

核心功能：
1. 检测DSL中的函数调用（type="call"）
2. 查询函数实现库获取实现代码
3. 填充到Operation.implementation字段
4. 收集需要的导入语句
"""

from typing import List, Dict, Optional, Any
import logging

from data_models import Operation, Prompt25Result
from dsl_library import FUNCTION_REGISTRY, render_implementation, has_implementation, get_function_impl

logger = logging.getLogger(__name__)


class DSLEnricher:
    """
    DSL语义增强器
    为DSL中的函数调用填充实现代码
    """
    
    def __init__(self):
        self._enrich_stats = {
            'total_operations': 0,
            'enriched_operations': 0,
            'fallback_operations': 0,
            'missing_functions': []
        }
    
    def enrich(self, prompt_result: Prompt25Result) -> Prompt25Result:
        """
        增强Prompt25Result中的操作
        
        Args:
            prompt_result: Prompt 2.5 结构化语义结果
            
        Returns:
            增强后的Prompt25Result
        """
        logger.info(f"[DSLEnricher] 开始增强 DSL，共 {len(prompt_result.operations)} 个操作")
        
        self._reset_stats()
        self._enrich_stats['total_operations'] = len(prompt_result.operations)
        
        for i, op in enumerate(prompt_result.operations):
            if op.type == 'call' and op.function:
                self._enrich_operation(op)
        
        for branch in prompt_result.branches:
            for i, op in enumerate(branch.operations):
                if op.type == 'call' and op.function:
                    self._enrich_operation(op)
        
        logger.info(f"[DSLEnricher] 增强完成: "
                   f"已填充={self._enrich_stats['enriched_operations']}, "
                   f"兜底={self._enrich_stats['fallback_operations']}, "
                   f"缺失={len(self._enrich_stats['missing_functions'])}")
        
        return prompt_result
    
    def _enrich_operation(self, op: Operation):
        """增强单个操作"""
        func_name = op.function
        
        if has_implementation(func_name):
            rendered_code = render_implementation(func_name, op.params or {})
            if rendered_code:
                op.implementation = rendered_code
                imports = FUNCTION_REGISTRY.get_imports(func_name)
                op.impl_required_imports = imports
                self._enrich_stats['enriched_operations'] += 1
                logger.debug(f"[DSLEnricher] 已填充实现: {func_name} -> {rendered_code[:50]}...")
            else:
                self._fallback_to_llm(op, func_name)
        else:
            self._fallback_to_llm(op, func_name)
    
    def _fallback_to_llm(self, op: Operation, func_name: str):
        """无法找到实现时的兜底处理"""
        op.implementation = None
        self._enrich_stats['fallback_operations'] += 1
        if func_name not in self._enrich_stats['missing_functions']:
            self._enrich_stats['missing_functions'].append(func_name)
        logger.warning(f"[DSLEnricher] 函数无实现，将使用LLM兜底: {func_name}")
    
    def _reset_stats(self):
        """重置统计信息"""
        self._enrich_stats = {
            'total_operations': 0,
            'enriched_operations': 0,
            'fallback_operations': 0,
            'missing_functions': []
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取增强统计"""
        return self._enrich_stats.copy()
    
    def get_missing_functions(self) -> List[str]:
        """获取缺失实现的函数列表"""
        return self._enrich_stats['missing_functions'].copy()
    
    def get_coverage_rate(self) -> float:
        """获取覆盖率"""
        total = self._enrich_stats['total_operations']
        if total == 0:
            return 1.0
        enriched = self._enrich_stats['enriched_operations']
        return enriched / total


def enrich_prompt_result(prompt_result: Prompt25Result) -> Prompt25Result:
    """
    快捷函数：为Prompt25Result填充实现代码
    
    Args:
        prompt_result: Prompt 2.5 结构化语义结果
        
    Returns:
        增强后的Prompt25Result
    """
    enricher = DSLEnricher()
    return enricher.enrich(prompt_result)


def enrich_operation(op: Operation) -> Operation:
    """
    快捷函数：增强单个操作
    
    Args:
        op: 操作单元
        
    Returns:
        增强后的操作
    """
    if op.type == 'call' and op.function:
        enricher = DSLEnricher()
        enricher._enrich_operation(op)
    return op


def get_coverage_report(prompt_result: Prompt25Result) -> Dict[str, Any]:
    """
    获取覆盖率报告
    
    Args:
        prompt_result: Prompt 2.5 结构化语义结果
        
    Returns:
        覆盖率报告
    """
    enricher = DSLEnricher()
    enricher.enrich(prompt_result)
    
    return {
        'total_operations': enricher._enrich_stats['total_operations'],
        'enriched_count': enricher._enrich_stats['enriched_operations'],
        'fallback_count': enricher._enrich_stats['fallback_operations'],
        'coverage_rate': enricher.get_coverage_rate(),
        'missing_functions': enricher.get_missing_functions(),
        'enriched_functions': [
            op.function for op in prompt_result.operations
            if op.type == 'call' and op.function and op.implementation
        ]
    }