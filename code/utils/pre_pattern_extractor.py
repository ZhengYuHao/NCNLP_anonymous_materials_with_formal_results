"""
基于预定义模式的实体提取器（代码层预处理）
用于 Prompt 2.0 阶段的实体提取优化
"""

import re
from typing import List, Dict, Any


class PrePatternExtractor:
    """基于预定义模式的实体提取器（代码层）"""

    # 常见数字+单位模式
    NUMERIC_PATTERNS = [
        (r'(\d+)个(人|开发者|工程师|成员)', 'team_size', 'Integer'),
        (r'(\d+)年', 'duration_years', 'Integer'),
        (r'(\d+)周', 'duration_weeks', 'Integer'),
        (r'(\d+)月', 'duration_months', 'Integer'),
        (r'(\d+)天', 'duration_days', 'Integer'),
        (r'(\d+)万', 'budget_wan', 'Integer'),
        (r'(\d+)轮', 'context_rounds', 'Integer'),
    ]

    # 技术栈模式
    TECH_PATTERNS = [
        (r'(Python|Java|JavaScript|TypeScript|Go|Rust|C\+\+|Node\.js|React|Vue|Angular|Spring|Django|Flask)',
         'tech_stack', 'String'),
    ]

    @classmethod
    def extract(cls, text: str) -> List[Dict]:
        """
        使用正则表达式提取常见模式

        Args:
            text: 输入文本

        Returns:
            提取的实体列表
        """
        extracted = []

        # 1. 提取数字+单位模式
        for pattern, var_name, var_type in cls.NUMERIC_PATTERNS:
            for match in re.finditer(pattern, text):
                value = int(match.group(1))
                original_text = match.group(0)
                extracted.append({
                    "name": var_name,
                    "original_text": original_text,
                    "start_index": match.start(),
                    "end_index": match.end(),
                    "type": var_type,
                    "value": value,
                    "source": "regex_pattern"  # 标记来源
                })

        # 2. 提取技术栈模式
        tech_stack_values = []
        tech_start = None
        tech_end = None

        for pattern, var_name, var_type in cls.TECH_PATTERNS:
            for match in re.finditer(pattern, text):
                tech_value = match.group(1)
                tech_stack_values.append(tech_value)

                if tech_start is None or match.start() < tech_start:
                    tech_start = match.start()
                if tech_end is None or match.end() > tech_end:
                    tech_end = match.end()

        # 如果找到了技术栈，合并为一个实体
        if tech_stack_values and tech_start is not None and tech_end is not None:
            extracted.append({
                "name": "tech_stack",
                "original_text": text[tech_start:tech_end],
                "start_index": tech_start,
                "end_index": tech_end,
                "type": "List",
                "value": tech_stack_values,
                "source": "regex_pattern"
            })

        # 按起始位置排序
        extracted.sort(key=lambda x: x['start_index'])

        return extracted

    @classmethod
    def merge_with_llm(cls, regex_entities: List[Dict], llm_entities: List[Dict]) -> List[Dict]:
        """
        合并正则提取和 LLM 提取的结果

        策略：
        - 正则提取的优先级更高（更准确）
        - LLM 补充正则无法识别的模式

        Args:
            regex_entities: 正则提取的实体
            llm_entities: LLM 提取的实体

        Returns:
            合并后的实体列表
        """
        # 标记已被正则覆盖的位置
        covered_ranges = []
        for entity in regex_entities:
            covered_ranges.append((entity['start_index'], entity['end_index']))

        # 过滤掉与正则结果重叠的 LLM 实体
        merged = regex_entities.copy()
        for llm_entity in llm_entities:
            start = llm_entity['start_index']
            end = llm_entity['end_index']

            # 检查是否与任何正则结果重叠
            is_overlapping = False
            for r_start, r_end in covered_ranges:
                if not (end <= r_start or start >= r_end):
                    is_overlapping = True
                    break

            if not is_overlapping:
                llm_entity['source'] = 'llm_extraction'
                merged.append(llm_entity)

        # 按起始位置排序
        merged.sort(key=lambda x: x['start_index'])

        return merged
