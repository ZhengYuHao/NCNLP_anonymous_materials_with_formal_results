"""
模块懒加载工厂
用于延迟初始化各种客户端和服务

使用方式：
    from dsl_library.module_factory import ModuleFactory
    
    # 在需要的地方
    vector_db = ModuleFactory.get_vector_db()
    results = await vector_db.search(query, top_k=10)
"""

from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)


class ModuleFactory:
    """
    懒加载工厂类
    所有模块实例通过此类统一管理和延迟初始化
    """
    
    _vector_db: Optional[Any] = None
    _kg_engine: Optional[Any] = None
    _cache: Optional[Any] = None
    _reranker: Optional[Any] = None
    _semantic_engine: Optional[Any] = None
    _hybrid_searcher: Optional[Any] = None
    _code_analyzer: Optional[Any] = None
    _intent_classifier: Optional[Any] = None
    _intent_analyzer: Optional[Any] = None
    _fallback_handler: Optional[Any] = None
    _concurrency_controller: Optional[Any] = None
    _feedback_handler: Optional[Any] = None
    
    _initialized: bool = False
    _config: Dict[str, Any] = {}
    
    @classmethod
    def initialize(cls, config: Optional[Dict[str, Any]] = None):
        """
        初始化工厂配置
        
        Args:
            config: 配置字典，格式如：
                {
                    'vector_db': {'type': 'milvus', 'host': 'localhost', 'port': 19530},
                    'kg_engine': {'type': 'neo4j', 'uri': 'bolt://localhost:7687'},
                    ...
                }
        """
        if config:
            cls._config = config
        cls._initialized = True
        logger.info("ModuleFactory 已初始化")
    
    @classmethod
    def get_vector_db(cls) -> Any:
        """
        获取向量数据库客户端（懒加载）
        
        使用模拟客户端，如果需要真实客户端，请替换此方法
        """
        if cls._vector_db is None:
            from dsl_library.mock_clients import MockVectorDBClient
            cls._vector_db = MockVectorDBClient()
            logger.info("向量数据库客户端已初始化")
        return cls._vector_db
    
    @classmethod
    def get_kg_engine(cls) -> Any:
        """获取知识图谱引擎（懒加载）"""
        if cls._kg_engine is None:
            from dsl_library.mock_clients import MockKGEngine
            cls._kg_engine = MockKGEngine()
            logger.info("知识图谱引擎已初始化")
        return cls._kg_engine
    
    @classmethod
    def get_cache(cls) -> Any:
        """获取缓存客户端（懒加载）"""
        if cls._cache is None:
            from dsl_library.mock_clients import MockCacheClient
            cls._cache = MockCacheClient()
            logger.info("缓存客户端已初始化")
        return cls._cache
    
    @classmethod
    def get_reranker(cls) -> Any:
        """获取重排序器（懒加载）"""
        if cls._reranker is None:
            from dsl_library.mock_clients import MockReranker
            cls._reranker = MockReranker()
            logger.info("重排序器已初始化")
        return cls._reranker
    
    @classmethod
    def get_semantic_engine(cls) -> Any:
        """获取语义引擎（懒加载）"""
        if cls._semantic_engine is None:
            from dsl_library.mock_clients import MockSemanticEngine
            cls._semantic_engine = MockSemanticEngine()
            logger.info("语义引擎已初始化")
        return cls._semantic_engine
    
    @classmethod
    def get_hybrid_searcher(cls) -> Any:
        """获取混合搜索器（懒加载）"""
        if cls._hybrid_searcher is None:
            from dsl_library.mock_clients import MockHybridSearcher
            cls._hybrid_searcher = MockHybridSearcher()
            logger.info("混合搜索器已初始化")
        return cls._hybrid_searcher
    
    @classmethod
    def get_code_analyzer(cls) -> Any:
        """获取代码分析器（懒加载）"""
        if cls._code_analyzer is None:
            from dsl_library.mock_clients import MockCodeAnalyzer
            cls._code_analyzer = MockCodeAnalyzer()
            logger.info("代码分析器已初始化")
        return cls._code_analyzer
    
    @classmethod
    def get_intent_classifier(cls) -> Any:
        """获取意图分类器（懒加载）"""
        if cls._intent_classifier is None:
            from dsl_library.mock_clients import MockIntentClassifier
            cls._intent_classifier = MockIntentClassifier()
            logger.info("意图分类器已初始化")
        return cls._intent_classifier
    
    @classmethod
    def get_intent_analyzer(cls) -> Any:
        """获取意图分析器（懒加载）"""
        if cls._intent_analyzer is None:
            from dsl_library.mock_clients import MockIntentAnalyzer
            cls._intent_analyzer = MockIntentAnalyzer()
            logger.info("意图分析器已初始化")
        return cls._intent_analyzer
    
    @classmethod
    def get_fallback_handler(cls) -> Any:
        """获取回退处理器（懒加载）"""
        if cls._fallback_handler is None:
            from dsl_library.mock_clients import MockFallbackHandler
            cls._fallback_handler = MockFallbackHandler()
            logger.info("回退处理器已初始化")
        return cls._fallback_handler
    
    @classmethod
    def get_concurrency_controller(cls) -> Any:
        """获取并发控制器（懒加载）"""
        if cls._concurrency_controller is None:
            from dsl_library.mock_clients import MockConcurrencyController
            cls._concurrency_controller = MockConcurrencyController()
            logger.info("并发控制器已初始化")
        return cls._concurrency_controller
    
    @classmethod
    def get_feedback_handler(cls) -> Any:
        """获取反馈处理器（懒加载）"""
        if cls._feedback_handler is None:
            from dsl_library.mock_clients import MockFeedbackHandler
            cls._feedback_handler = MockFeedbackHandler()
            logger.info("反馈处理器已初始化")
        return cls._feedback_handler
    
    @classmethod
    def reset(cls):
        """重置所有模块实例（用于测试）"""
        cls._vector_db = None
        cls._kg_engine = None
        cls._cache = None
        cls._reranker = None
        cls._semantic_engine = None
        cls._hybrid_searcher = None
        cls._code_analyzer = None
        cls._intent_classifier = None
        cls._intent_analyzer = None
        cls._fallback_handler = None
        cls._concurrency_controller = None
        cls._feedback_handler = None
        cls._initialized = False
        cls._config = {}
        logger.info("ModuleFactory 已重置")