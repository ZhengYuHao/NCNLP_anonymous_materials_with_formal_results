"""
DSL函数实现库
用于解决DSL生成代码的"空心化"问题

核心模块：
- function_impl_registry.py: 函数实现注册表
- module_factory.py: 懒加载工厂类
- dsl_enricher.py: DSL语义增强器
- enhanced_compiler.py: 增强版编译器
- helper_functions.py: 辅助函数
- mock_clients.py: 模拟客户端
"""

from dsl_library.function_impl_registry import (
    FunctionImplRegistry,
    FunctionImplementation,
    FUNCTION_REGISTRY,
    get_function_impl,
    has_implementation,
    render_implementation,
    get_imports,
)

from dsl_library.module_factory import ModuleFactory

from dsl_library.dsl_enricher import (
    DSLEnricher,
    enrich_prompt_result,
    enrich_operation,
    get_coverage_report,
)

from dsl_library.enhanced_compiler import (
    EnhancedWaActCompiler,
    compile_with_implementation,
    compile_with_fallback,
)

from dsl_library.helper_functions import (
    _get_top_n,
    _limit_results,
    _deduplicate_results,
    _sort_results,
    _compare_version,
    _detect_query_type,
    _filter_sensitive_content,
    _validate_input,
    _check_user_permission,
    _check_rate_limit,
    _switch_retrieval_mode,
    _escalate_to_human_queue,
    _trigger_stronger_model_regeneration,
    _check_user_dissatisfaction_count,
)

__all__ = [
    # 函数实现注册表
    'FunctionImplRegistry',
    'FunctionImplementation',
    'FUNCTION_REGISTRY',
    'get_function_impl',
    'has_implementation',
    'render_implementation',
    # 懒加载工厂
    'ModuleFactory',
    # DSL增强器
    'DSLEnricher',
    'enrich_prompt_result',
    'enrich_operation',
    'get_coverage_report',
    # 增强编译器
    'EnhancedWaActCompiler',
    'compile_with_implementation',
    'compile_with_fallback',
    # 辅助函数
    '_get_top_n',
    '_limit_results',
    '_deduplicate_results',
    '_sort_results',
    '_compare_version',
    '_detect_query_type',
    '_filter_sensitive_content',
    '_validate_input',
    '_check_user_permission',
    '_check_rate_limit',
    '_switch_retrieval_mode',
    '_escalate_to_human_queue',
    '_trigger_stronger_model_regeneration',
    '_check_user_dissatisfaction_count',
]

__version__ = '1.0.0'