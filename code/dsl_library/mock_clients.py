"""
模拟客户端实现
用于在没有真实服务时进行测试

所有模拟客户端都实现了真实接口，返回合理的模拟数据
"""

from typing import List, Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class MockVectorDBClient:
    """模拟向量数据库客户端"""
    
    async def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """模拟向量检索"""
        logger.info(f"[MockVectorDB] 搜索: query={query}, top_k={top_k}")
        await asyncio.sleep(0.01)
        return [
            {'id': f'doc_{i}', 'content': f'模拟文档{i}', 'score': 0.9 - i * 0.05}
            for i in range(min(top_k, 5))
        ]


class MockKGEngine:
    """模拟知识图谱引擎"""
    
    async def search(self, query: str, hops: int = 2) -> List[Dict]:
        """模拟图检索"""
        logger.info(f"[MockKG] 搜索: query={query}, hops={hops}")
        await asyncio.sleep(0.01)
        return [
            {'node_id': f'node_{i}', 'name': f'节点{i}', 'type': 'entity', 'score': 0.85 - i * 0.1}
            for i in range(3)
        ]
    
    async def traverse(self, node_id: str, depth: int) -> List[Dict]:
        """模拟图遍历"""
        logger.info(f"[MockKG] 遍历: node_id={node_id}, depth={depth}")
        return [{'node_id': node_id, 'children': []}]
    
    async def get_neighbors(self, node_id: str, limit: int) -> List[Dict]:
        """模拟获取邻居"""
        logger.info(f"[MockKG] 邻居: node_id={node_id}, limit={limit}")
        return [{'node_id': f'neighbor_{i}', 'relation': 'related'} for i in range(min(limit, 3))]
    
    async def find_path(self, start: str, end: str, max_hops: int) -> List[Dict]:
        """模拟路径查找"""
        logger.info(f"[MockKG] 路径: {start} -> {end}, max_hops={max_hops}")
        return [{'path': [start, 'middle', end], 'length': 2}]


class MockCacheClient:
    """模拟缓存客户端"""
    
    def __init__(self):
        self._store: Dict[str, Any] = {}
    
    async def set(self, key: str, value: Any, ttl: int = 1800):
        """模拟设置缓存"""
        logger.info(f"[MockCache] 设置: key={key}, ttl={ttl}")
        self._store[key] = value
    
    async def get(self, key: str) -> Optional[Any]:
        """模拟获取缓存"""
        logger.info(f"[MockCache] 获取: key={key}")
        return self._store.get(key)
    
    async def delete(self, key: str):
        """模拟删除缓存"""
        if key in self._store:
            del self._store[key]


class MockReranker:
    """模拟重排序器"""
    
    async def rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """模拟重排序"""
        logger.info(f"[MockReranker] 重排序: {len(results)} 条结果")
        await asyncio.sleep(0.01)
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)


class MockSemanticEngine:
    """模拟语义搜索引擎"""
    
    async def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """模拟语义搜索"""
        logger.info(f"[MockSemantic] 搜索: query={query}, top_k={top_k}")
        await asyncio.sleep(0.01)
        return [
            {'id': f'semantic_doc_{i}', 'content': f'语义文档{i}', 'score': 0.88 - i * 0.06}
            for i in range(min(top_k, 5))
        ]


class MockHybridSearcher:
    """模拟混合搜索器"""
    
    async def search(self, query: str, alpha: float = 0.5) -> List[Dict]:
        """模拟混合搜索"""
        logger.info(f"[MockHybrid] 搜索: query={query}, alpha={alpha}")
        await asyncio.sleep(0.02)
        return [
            {'id': f'hybrid_doc_{i}', 'content': f'混合文档{i}', 'score': 0.9 - i * 0.05, 'source': 'hybrid'}
            for i in range(5)
        ]


class MockCodeAnalyzer:
    """模拟代码分析器"""
    
    async def analyze(self, code: str, language: str = 'python') -> Dict:
        """模拟代码分析"""
        logger.info(f"[MockCodeAnalyzer] 分析: language={language}, 代码长度={len(code)}")
        await asyncio.sleep(0.01)
        return {
            'language': language,
            'functions': ['main', 'helper'],
            'classes': ['Processor'],
            'lines': len(code.split('\n')),
            'quality_score': 0.85
        }
    
    async def static_analyze(self, code: str) -> Dict:
        """模拟静态分析"""
        logger.info(f"[MockCodeAnalyzer] 静态分析")
        return {
            'issues': [],
            'complexity': 'medium',
            'maintainability': 0.8
        }
    
    async def semantic_analyze(self, code: str, generate_docs: bool = True) -> Dict:
        """模拟语义分析"""
        logger.info(f"[MockCodeAnalyzer] 语义分析: generate_docs={generate_docs}")
        return {
            'ast': {},
            'semantic_info': {'variables': 5, 'functions': 2},
            'documentation': 'Generated documentation here' if generate_docs else None
        }


class MockIntentClassifier:
    """模拟意图分类器"""
    
    async def classify(self, query: str) -> Dict:
        """模拟意图分类"""
        logger.info(f"[MockIntentClassifier] 分类: query={query[:50]}...")
        
        if any(kw in query for kw in ['代码', 'python', 'java', '函数']):
            intent = 'code'
        elif any(kw in query for kw in ['比较', '分析', '为什么']):
            intent = 'complex'
        else:
            intent = 'simple'
        
        return {
            'intent': intent,
            'confidence': 0.85,
            'categories': [intent]
        }


class MockIntentAnalyzer:
    """模拟意图分析器"""
    
    async def decompose(self, query: str) -> List[Dict]:
        """模拟意图分解"""
        logger.info(f"[MockIntentAnalyzer] 分解: query={query[:50]}...")
        return [
            {'sub_intent': 'search', 'params': {'query': query}},
            {'sub_intent': 'filter', 'params': {'threshold': 0.8}},
            {'sub_intent': 'rank', 'params': {'top_k': 5}}
        ]


class MockFallbackHandler:
    """模拟回退处理器"""
    
    async def execute_retrieval(self, query: str) -> List[Dict]:
        """模拟回退检索"""
        logger.info(f"[MockFallback] 回退检索: query={query}")
        return [{'id': 'fallback_doc', 'content': '回退结果', 'score': 0.7}]


class MockConcurrencyController:
    """模拟并发控制器"""
    
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
    
    async def acquire_lock(self, key: str, timeout: int = 30) -> bool:
        """模拟获取锁"""
        logger.info(f"[MockConcurrency] 获取锁: key={key}")
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return True
    
    async def release_lock(self, key: str):
        """模拟释放锁"""
        logger.info(f"[MockConcurrency] 释放锁: key={key}")
        if key in self._locks:
            self._locks[key].release()


class MockFeedbackHandler:
    """模拟反馈处理器"""
    
    def __init__(self):
        self._feedback_count: Dict[str, int] = {}
    
    async def record(self, user_id: str, query: str, result: Any, rating: Optional[int] = None):
        """模拟记录反馈"""
        logger.info(f"[MockFeedback] 记录: user={user_id}, rating={rating}")
        if user_id not in self._feedback_count:
            self._feedback_count[user_id] = 0
        if rating and rating < 3:
            self._feedback_count[user_id] += 1
    
    def get_dissatisfaction_count(self, user_id: str) -> int:
        """获取不满次数"""
        return self._feedback_count.get(user_id, 0)