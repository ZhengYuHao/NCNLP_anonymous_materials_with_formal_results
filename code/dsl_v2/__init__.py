"""
DSL v2 编译器模块
基于 files 模块架构和 llm_client.py 统一接口的新一代 DSL 编译器
"""

from .compiler import PromptCompiler, CompilerResult
from .extractor import UnifiedExtractor, MockExtractor

__all__ = ['PromptCompiler', 'CompilerResult', 'UnifiedExtractor', 'MockExtractor']
__version__ = '2.0.0'
