"""DSL 语义缺失检测器 - 用 LLM 检测 DSL 中语义不完整的地方"""
from dataclasses import dataclass, field
from typing import List, Optional
import json
import re


@dataclass
class SemanticGap:
    """语义缺失项"""
    gap_type: str  # variable_undefined | return_missing | error_handling | type_inference | logic_error
    location: str  # 行号或代码片段
    description: str  # 问题描述
    suggestion: str  # 修复建议
    severity: str  # high | medium | low


class SemanticGapDetector:
    """检测 DSL 中的语义缺失"""

    def __init__(self, llm_client):
        """
        初始化语义缺失检测器

        Args:
            llm_client: LLM 客户端实例
        """
        self.llm = llm_client

    async def detect(self, dsl_code: str, original_prompt: str = "") -> List[SemanticGap]:
        """
        检测 DSL 中的语义缺失

        Args:
            dsl_code: 待检测的 DSL 代码
            original_prompt: 原始自然语言需求（可选）

        Returns:
            语义缺失列表
        """
        # 构建 prompt
        prompt = self._build_detection_prompt(dsl_code, original_prompt)

        # 调用 LLM（需要 system_prompt 和 user_content 两个参数）
        response = self.llm.call("", prompt)

        # 解析响应
        try:
            result = json.loads(response.content)
            gaps = [
                SemanticGap(
                    gap_type=g.get('type', 'unknown'),
                    location=g.get('location', ''),
                    description=g.get('description', ''),
                    suggestion=g.get('suggestion', ''),
                    severity=g.get('severity', 'medium')
                )
                for g in result.get('gaps', [])
            ]
            return gaps
        except json.JSONDecodeError:
            # 如果 JSON 解析失败，尝试从文本中提取
            return self._parse_text_response(response.content, dsl_code)

    def _build_detection_prompt(self, dsl_code: str, original_prompt: str) -> str:
        """构建检测 prompt"""
        original_section = f"""
## 原始需求
{original_prompt}
""" if original_prompt else ""

        prompt = f"""你是一个 DSL 代码审查专家。请分析以下 DSL 代码，找出语义不完整的地方。

{original_section}
## DSL 代码
```
{dsl_code}
```

## 检测项

请仔细检查以下几类语义缺失：

### 1. 变量定义缺失 (variable_undefined)
- 是否有变量被使用但从未赋值？
- 是否有条件分支内定义的变量在分支外使用？
- 是否有函数调用返回的结果未被捕获？

### 2. 返回值缺失 (return_missing)
- 是否有 IF/ELIF 分支缺少 RETURN 语句？
- 是否有所有分支都有 RETURN，但外层缺少 RETURN？
- 函数的实际返回值是否与 OUTPUTS 声明匹配？

### 3. 错误处理缺失 (error_handling)
- 是否有 CALL 语句缺少对 None/错误的处理？
- 是否有未处理的边界情况（如空结果、异常等）？
- 是否有条件检查但没有对应的 ELSE 分支？

### 4. 类型推断缺失 (type_inference)
- 是否有变量的类型无法从上下文推断？
- 是否有类型转换可能出错？
- 是否有数值操作但类型不匹配？

### 5. 逻辑错误 (logic_error)
- 是否有永远执行不到的代码分支（死代码）？
- 是否有永远为真/为假的条件？
- 是否有变量在赋值前被使用？

## 输出格式

请以严格的 JSON 格式输出检测结果：
{{
    "gaps": [
        {{
            "type": "variable_undefined | return_missing | error_handling | type_inference | logic_error",
            "location": "行号或具体代码片段",
            "description": "问题描述，用简洁的语言说明问题",
            "suggestion": "具体的修复建议",
            "severity": "high | medium | low"
        }}
    ]
}}

注意：
- severity 为 high 表示必须修复，否则生成的代码无法正确运行
- severity 为 medium 表示建议修复，可能导致潜在问题
- severity 为 low 表示可选修复，代码可以运行但可能有隐患
- 如果没有发现问题，返回空的 gaps 数组
- 请只输出 JSON，不要有其他解释文字
"""
        return prompt

    def _parse_text_response(self, content: str, dsl_code: str) -> List[SemanticGap]:
        """从文本响应中解析语义缺失"""
        gaps = []

        # 尝试用正则表达式匹配常见的 gap 格式
        patterns = [
            # 匹配 [type] location: description
            r'\[([^\]]+)\]\s*([^:]+):\s*([^\n]+)',
            # 匹配 - type: description at location
            r'-\s*(\w+):\s*([^\n]+?)\s+(?:at|in|位置|行)\s*(\d+)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 3:
                    gap_type = groups[0].strip().lower()
                    location = groups[1].strip() if len(groups) > 1 else ""
                    description = groups[2].strip() if len(groups) > 2 else ""

                    # 验证类型是否有效
                    valid_types = ['variable_undefined', 'return_missing', 'error_handling', 'type_inference', 'logic_error']
                    if gap_type not in valid_types:
                        continue

                    gaps.append(SemanticGap(
                        gap_type=gap_type,
                        location=location,
                        description=description,
                        suggestion="请检查并修复",
                        severity="medium"
                    ))

        return gaps

    def detect_sync(self, dsl_code: str, original_prompt: str = "") -> List[SemanticGap]:
        """
        同步版本的检测（用于不需要 async 的场景）

        Args:
            dsl_code: 待检测的 DSL 代码
            original_prompt: 原始自然语言需求（可选）

        Returns:
            语义缺失列表
        """
        # 构建 prompt
        prompt = self._build_detection_prompt(dsl_code, original_prompt)

        # 同步调用 LLM
        response = self.llm.call(prompt)

        # 解析响应
        try:
            result = json.loads(response.content)
            gaps = [
                SemanticGap(
                    gap_type=g.get('type', 'unknown'),
                    location=g.get('location', ''),
                    description=g.get('description', ''),
                    suggestion=g.get('suggestion', ''),
                    severity=g.get('severity', 'medium')
                )
                for g in result.get('gaps', [])
            ]
            return gaps
        except json.JSONDecodeError:
            return self._parse_text_response(response.content, dsl_code)