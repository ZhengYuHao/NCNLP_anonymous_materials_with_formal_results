"""
带缓存的 LLM 客户端
用于避免重复 LLM 调用
"""

import os
import json
import hashlib
from functools import lru_cache
from typing import Dict, Any


class CachedLLMClient:
    """带缓存的 LLM 客户端"""

    def __init__(self, base_client, cache_size: int = 1000):
        """
        初始化缓存客户端

        Args:
            base_client: 底层 LLM 客户端
            cache_size: 缓存大小（未使用，保留用于 LRU）
        """
        self.base_client = base_client
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0

    def _make_cache_key(self, system_prompt: str, user_content: str) -> str:
        """
        生成缓存键

        Args:
            system_prompt: 系统 Prompt
            user_content: 用户内容

        Returns:
            缓存键（SHA256 哈希）
        """
        combined = f"{system_prompt}|||{user_content}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def call(self, system_prompt: str, user_content: str, **kwargs) -> Any:
        """
        调用 LLM（带缓存）

        Args:
            system_prompt: 系统 Prompt
            user_content: 用户内容
            **kwargs: 其他参数（temperature, model等，但不影响缓存键）

        Returns:
            LLM 响应对象
        """
        cache_key = self._make_cache_key(system_prompt, user_content)

        # 检查缓存
        cache_file = f".cache/llm_{cache_key}.json"
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    self.hits += 1
                    # 标记响应来自缓存
                    cached['from_cache'] = True
                    from llm_client import LLMResponse
                    return LLMResponse(**cached)
        except Exception as e:
            # 缓存读取失败，继续调用 LLM
            pass

        # 调用 LLM
        self.misses += 1
        response = self.base_client._call_without_cache(system_prompt, user_content, **kwargs)

        # 保存到缓存
        try:
            os.makedirs('.cache', exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump({
                    'content': response.content,
                    'model': response.model,
                    'usage': response.usage
                }, f)
        except Exception:
            # 缓存保存失败，不影响主流程
            pass

        return response

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计

        Returns:
            统计信息字典
        """
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": self.hits / total if total > 0 else 0.0
        }

    def clear_cache(self):
        """清空缓存目录"""
        import shutil
        cache_dir = ".cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
        self.hits = 0
        self.misses = 0
