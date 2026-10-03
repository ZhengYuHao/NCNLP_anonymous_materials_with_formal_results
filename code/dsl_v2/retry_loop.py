"""
DSL v2 重试循环模块
管理编译重试逻辑和历史记录
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field


@dataclass
class RetryAttempt:
    """重试尝试记录"""
    attempt: int
    dsl_code: str = ""
    errors: List[Any] = field(default_factory=list)
    error_count: int = 0
    auto_fix_attempted: bool = False
    auto_fix_count: int = 0
    error: Optional[str] = None


@dataclass
class RetryHistory:
    """重试历史记录"""
    attempts: List[RetryAttempt] = field(default_factory=list)
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    final_decision: str = ""
    auto_fix_applied: bool = False

    def add_attempt(self, attempt: RetryAttempt):
        """添加尝试记录"""
        self.attempts.append(attempt)

    def get_last_attempt(self) -> Optional[RetryAttempt]:
        """获取最后一次尝试"""
        return self.attempts[-1] if self.attempts else None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'attempts': [
                {
                    'attempt': a.attempt,
                    'dsl_length': len(a.dsl_code),
                    'error_count': a.error_count,
                    'auto_fix_attempted': a.auto_fix_attempted,
                    'auto_fix_count': a.auto_fix_count,
                    'error': a.error
                }
                for a in self.attempts
            ],
            'total_tokens': self.total_tokens,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'final_decision': self.final_decision,
            'auto_fix_applied': self.auto_fix_applied
        }


class RetryLoop:
    """
    重试循环管理器

    负责：
    - 管理编译重试次数
    - 收集和记录每次尝试的历史
    - 决定何时停止重试
    """

    def __init__(self, max_retries: int = 3):
        """
        初始化重试循环

        Args:
            max_retries: 最大重试次数
        """
        self.max_retries = max_retries
        self.history = RetryHistory()

    def should_retry(self, attempt: int) -> bool:
        """
        判断是否应该继续重试

        Args:
            attempt: 当前尝试次数（从0开始）

        Returns:
            是否继续重试
        """
        return attempt < self.max_retries - 1

    def create_attempt(self, attempt: int) -> RetryAttempt:
        """
        创建新的尝试记录

        Args:
            attempt: 尝试次数（从0开始）

        Returns:
            新的 RetryAttempt 对象
        """
        return RetryAttempt(attempt=attempt + 1)

    def record_success(self, final_decision: str = "success"):
        """记录成功"""
        self.history.final_decision = final_decision

    def record_failure(self, final_decision: str = "failed"):
        """记录失败"""
        self.history.final_decision = final_decision

    def record_auto_fix(self, applied: bool, count: int = 0):
        """记录自动修复"""
        self.history.auto_fix_applied = applied
        if self.history.get_last_attempt():
            self.history.get_last_attempt().auto_fix_attempted = applied
            self.history.get_last_attempt().auto_fix_count = count

    def add_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        """添加 token 统计"""
        self.history.total_prompt_tokens += prompt_tokens
        self.history.total_completion_tokens += completion_tokens
        self.history.total_tokens = (
            self.history.total_prompt_tokens + self.history.total_completion_tokens
        )
