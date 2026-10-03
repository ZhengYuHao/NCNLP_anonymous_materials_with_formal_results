"""
函数实现注册表
用于存储和管理DSL函数的实现代码
"""

from typing import Dict, Optional, List, Any
from dataclasses import dataclass


@dataclass
class FunctionImplementation:
    """函数实现"""
    name: str
    code: str
    params: List[str]
    description: str
    required_imports: List[str] = None
    
    def __post_init__(self):
        if self.required_imports is None:
            self.required_imports = []


class FunctionImplRegistry:
    """函数实现注册表"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._register_defaults()
            self._initialized = True
    
    def _register_defaults(self):
        """注册默认函数实现"""
        
        self.register(FunctionImplementation(
            name="vector_retrieval",
            code="await vector_db.search(query, top_k=top_k)",
            params=["query", "top_k"],
            description="向量检索",
            required_imports=["from dsl_library import ModuleFactory"]
        ))
        
        self.register(FunctionImplementation(
            name="graph_retrieval",
            code="await kg_engine.search(query, hops=hops)",
            params=["query", "hops"],
            description="图检索",
            required_imports=["from dsl_library import ModuleFactory"]
        ))
        
        self.register(FunctionImplementation(
            name="hybrid_retrieval",
            code="await hybrid_searcher.search(query, alpha=alpha)",
            params=["query", "alpha"],
            description="混合检索",
            required_imports=["from dsl_library import ModuleFactory"]
        ))
        
        self.register(FunctionImplementation(
            name="get_top_results",
            code="await _get_top_n(results, n)",
            params=["results", "n"],
            description="获取Top N结果",
            required_imports=["from dsl_library import _get_top_n"]
        ))
        
        self.register(FunctionImplementation(
            name="filter_by_threshold",
            code="await _filter_by_threshold(results, threshold)",
            params=["results", "threshold"],
            description="按阈值过滤",
            required_imports=["from dsl_library import _filter_by_threshold"]
        ))
        
        self.register(FunctionImplementation(
            name="cache_result",
            code="await cache.set(cache_key, result, ttl=ttl)",
            params=["cache_key", "result", "ttl"],
            description="缓存结果",
            required_imports=["from dsl_library import ModuleFactory"]
        ))
        
        self.register(FunctionImplementation(
            name="rerank_results",
            code="await reranker.rerank(results, query)",
            params=["results", "query"],
            description="重排序结果",
            required_imports=["from dsl_library import ModuleFactory"]
        ))
        
        self.register(FunctionImplementation(
            name="deduplicate_results",
            code="await _deduplicate_results(results)",
            params=["results"],
            description="去重结果",
            required_imports=["from dsl_library import _deduplicate_results"]
        ))
        
        self.register(FunctionImplementation(
            name="count_results",
            code="len(results)",
            params=["results"],
            description="统计结果数量"
        ))
        
        self.register(FunctionImplementation(
            name="validate_input",
            code="await _validate_input(user_input)",
            params=["user_input"],
            description="验证输入",
            required_imports=["from dsl_library import _validate_input"]
        ))
        
        self.register(FunctionImplementation(
            name="check_permission",
            code="await _check_user_permission(user_id)",
            params=["user_id"],
            description="检查权限",
            required_imports=["from dsl_library import _check_user_permission"]
        ))
        
        self.register(FunctionImplementation(
            name="check_rate_limit",
            code="await _check_rate_limit(user_id)",
            params=["user_id"],
            description="检查频率限制",
            required_imports=["from dsl_library import _check_rate_limit"]
        ))
        
        self.register(FunctionImplementation(
            name="switch_retrieval_mode",
            code="await _switch_retrieval_mode(mode)",
            params=["mode"],
            description="切换检索模式",
            required_imports=["from dsl_library import _switch_retrieval_mode"]
        ))
        
        self.register(FunctionImplementation(
            name="escalate_to_human",
            code="await _escalate_to_human_queue(query)",
            params=["query"],
            description="转人工处理",
            required_imports=["from dsl_library import _escalate_to_human_queue"]
        ))
        
        self.register(FunctionImplementation(
            name="trigger_regeneration",
            code="await _trigger_stronger_model_regeneration(query)",
            params=["query"],
            description="触发更强模型重新生成",
            required_imports=["from dsl_library import _trigger_stronger_model_regeneration"]
        ))
        
        self.register(FunctionImplementation(
            name="classify_intent",
            code="await intent_classifier.classify(query)",
            params=["query"],
            description="意图分类",
            required_imports=["from dsl_library import ModuleFactory"]
        ))
        
        self.register(FunctionImplementation(
            name="decompose_intent",
            code="await intent_analyzer.decompose(query)",
            params=["query"],
            description="意图分解",
            required_imports=["from dsl_library import ModuleFactory"]
        ))
        
        self.register(FunctionImplementation(
            name="semantic_search",
            code="await semantic_engine.search(query, top_k=top_k)",
            params=["query", "top_k"],
            description="语义搜索",
            required_imports=["from dsl_library import ModuleFactory"]
        ))
        
        self.register(FunctionImplementation(
            name="filter_sensitive",
            code="await _filter_sensitive_content(content)",
            params=["content"],
            description="内容过滤",
            required_imports=["from dsl_library import _filter_sensitive_content"]
        ))
        
        self.register(FunctionImplementation(
            name="check_dissatisfaction",
            code="await _check_user_dissatisfaction_count(user_id)",
            params=["user_id"],
            description="检查用户不满次数",
            required_imports=["from dsl_library import _check_user_dissatisfaction_count"]
        ))
        
        self.register(FunctionImplementation(
            name="acquire_lock",
            code="await concurrency_controller.acquire_lock(key)",
            params=["key"],
            description="获取并发锁",
            required_imports=["from dsl_library import ModuleFactory"]
        ))
        
        self.register(FunctionImplementation(
            name="release_lock",
            code="await concurrency_controller.release_lock(key)",
            params=["key"],
            description="释放并发锁",
            required_imports=["from dsl_library import ModuleFactory"]
        ))
        
        self.register(FunctionImplementation(
            name="record_feedback",
            code="await feedback_handler.record(user_id, query, result, rating)",
            params=["user_id", "query", "result", "rating"],
            description="记录反馈",
            required_imports=["from dsl_library import ModuleFactory"]
        ))
    
    def register(self, impl: FunctionImplementation):
        """注册函数实现"""
        self._registry[impl.name] = impl
    
    def get(self, name: str) -> Optional[FunctionImplementation]:
        """获取函数实现"""
        return self._registry.get(name)
    
    def has(self, name: str) -> bool:
        """检查函数是否存在"""
        return name in self._registry
    
    def list_all(self) -> List[str]:
        """列出所有函数名"""
        return list(self._registry.keys())
    
    def count(self) -> int:
        """获取函数数量"""
        return len(self._registry)
    
    def get_imports(self, name: str) -> List[str]:
        """获取函数需要的导入语句"""
        impl = self._registry.get(name)
        if impl:
            return impl.required_imports or []
        return []


FUNCTION_REGISTRY = FunctionImplRegistry()


def get_function_impl(name: str) -> Optional[FunctionImplementation]:
    """获取函数实现"""
    return FUNCTION_REGISTRY.get(name)


def has_implementation(name: str) -> bool:
    """检查是否有实现"""
    return FUNCTION_REGISTRY.has(name)


def render_implementation(name: str, params: Dict[str, Any]) -> Optional[str]:
    """渲染函数调用代码"""
    impl = get_function_impl(name)
    if not impl:
        return None
    
    code = impl.code
    for param_name in impl.params:
        if param_name in params:
            value = params[param_name]
            if isinstance(value, str):
                code = code.replace(param_name, f'"{value}"')
            else:
                code = code.replace(param_name, str(value))
    
    return code


def get_imports(name: str) -> List[str]:
    """获取函数需要的导入语句"""
    impl = get_function_impl(name)
    if impl:
        return impl.required_imports or []
    return []