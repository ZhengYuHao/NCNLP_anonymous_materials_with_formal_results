"""
基于规则的文本规范化器（替代 LLM）
用于 Prompt 1.0 阶段的文本标准化
"""

import re
from typing import List, Tuple


class RuleBasedTextNormalizer:
    """基于规则的文本规范化器（替代 LLM）"""

    # 口语气词映射表
    COLLOQUIAL_FILLERS = {
        "那个": "", "嗯": "", "吧": "", "呢": "", "嘛": "",
        "搞一下": "处理", "弄一下": "操作", "整一下": "调整"
    }

    # 常见口语替换
    COLLOQUIAL_REPLACEMENTS = {
        "没啥意思": "价值较低",
        "挺不错的": "良好",
        "搞得怎么样了": "进展如何",
        "随便弄一下": "快速处理"
    }

    # 标点符号规范化
    PUNCTUATION_NORMALIZATIONS = {
        "。。。": "...",
        "..": ".",
        "。。": "...",
        " .": ".",
        " . ": ".",
    }

    @classmethod
    def normalize(cls, text: str) -> Tuple[str, List[str]]:
        """
        基于规则的文本规范化

        Args:
            text: 原始文本

        Returns:
            (规范化文本, 变更日志列表)
        """
        changes = []
        result = text

        # 1. 移除语气词
        for filler, replacement in cls.COLLOQUIAL_FILLERS.items():
            if filler in result:
                old = result
                result = result.replace(filler, replacement)
                if old != result:
                    changes.append(f"移除语气词: '{filler}'")

        # 2. 替换口语表达
        for colloquial, formal in cls.COLLOQUIAL_REPLACEMENTS.items():
            if colloquial in result:
                old = result
                result = result.replace(colloquial, formal)
                if old != result:
                    changes.append(f"口语规范化: '{colloquial}' → '{formal}'")

        # 3. 标点符号规范化（连续句号→省略号，中文句号→句号）
        result = re.sub(r'。{3,}', '...', result)
        result = re.sub(r'\.{3,}', '...', result)

        # 4. 移除多余空格
        result = re.sub(r'\s+', ' ', result).strip()

        return result, changes


class SyntacticAmbiguityDetector:
    """基于句法分析的歧义检测器（替代 LLM）"""

    # 经典歧义模式
    AMBIGUOUS_PATTERNS = [
        (r'(.+?)的(.+?)被(.+?)', "主语歧义: 无法确定是被{3}的对象"),
        (r'(.+?)和(.+?)的(.+?)', "修饰歧义: '和'的范围不明确"),
        (r'(.+?)中(.+?)最(.+?)', "范围歧义: '中'的范围不明确"),
    ]

    @classmethod
    def detect(cls, text: str) -> str:
        """
        检测句法歧义

        Args:
            text: 输入文本

        Returns:
            如果存在歧义则返回描述,否则返回空字符串
        """
        for pattern, message_template in cls.AMBIGUOUS_PATTERNS:
            match = re.search(pattern, text)
            if match:
                # 提取组并填充模板
                groups = match.groups()
                try:
                    message = message_template.format(*groups)
                    return message
                except Exception:
                    continue

        return ""
