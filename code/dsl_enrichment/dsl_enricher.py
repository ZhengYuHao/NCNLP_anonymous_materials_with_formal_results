"""DSL 语义增强器 - 根据检测到的缺失生成完整的 DSL"""
from typing import List, Tuple, Optional
from .semantic_gap_detector import SemanticGap, SemanticGapDetector


class DSLEnricher:
    """DSL 语义增强器 - 根据检测到的缺失生成完整的 DSL"""

    def __init__(self, llm_client):
        """
        初始化 DSL 增强器

        Args:
            llm_client: LLM 客户端实例
        """
        self.llm = llm_client
        self.detector = SemanticGapDetector(llm_client)

    async def enrich(
        self,
        dsl_code: str,
        original_prompt: str,
        gaps: List[SemanticGap]
    ) -> str:
        """
        增强 DSL，补充缺失的语义信息

        Args:
            dsl_code: 原始 DSL 代码
            original_prompt: 原始自然语言需求
            gaps: 检测到的语义缺失列表

        Returns:
            增强后的 DSL 代码
        """
        if not gaps:
            return dsl_code

        gaps_text = self._format_gaps(gaps)

        prompt = f"""你是一个 DSL 代码生成专家。请根据以下信息，生成一个语义完整的 DSL 代码。

## 原始需求
{original_prompt}

## 当前 DSL 代码（有问题）
```
{dsl_code}
```

## 检测到的语义缺失
{gaps_text}

## 任务
1. 修复上述所有语义缺失
2. 保持原有的业务逻辑不变
3. 确保生成的 DSL 满足以下要求：
   - 所有使用的变量都有定义
   - 所有分支都有明确的返回值
   - 所有 CALL 都有适当的错误处理
   - 类型推断清晰明确

## DSL 语法参考
```
# 模块定义
MODULE module_name
  INPUTS: {{var1}}, {{var2}}
  OUTPUTS: {{result}}

  # 变量定义
  DEFINE {{variable_name}}: Type = value

  # 条件分支
  IF {{condition}}:
    {{result}} = CALL function1(...)
    RETURN {{result}}
  ELIF {{condition}}:
    {{result}} = CALL function2(...)
    RETURN {{result}}
  ELSE:
    RETURN None
  ENDIF

  # 循环
  FOR {{item}} IN {{collection}}:
    {{result}} = CALL process({{item}})
  ENDFOR

  # 函数调用
  {{result}} = CALL function_name(arg1={{arg1}}, arg2={{arg2}})
ENDMODULE
```

## 输出格式
只输出修复后的完整 DSL 代码，不要有其他解释文字。
不要在代码前后添加任何说明。
"""

        response = self.llm.call(prompt)
        return response.content.strip()

    async def enrich_iterative(
        self,
        dsl_code: str,
        original_prompt: str,
        max_iterations: int = 3
    ) -> Tuple[str, List[SemanticGap]]:
        """
        迭代增强 DSL，直到没有高严重性的语义缺失

        Args:
            dsl_code: 原始 DSL 代码
            original_prompt: 原始自然语言需求
            max_iterations: 最大迭代次数

        Returns:
            (增强后的 DSL, 剩余语义缺失列表)
        """
        current_dsl = dsl_code

        for i in range(max_iterations):
            # 检测语义缺失
            gaps = await self.detector.detect(current_dsl, original_prompt)

            # 检查是否还有高严重性缺失
            high_severity_gaps = [g for g in gaps if g.severity == 'high']

            if not high_severity_gaps:
                # 没有高严重性缺失，可以结束
                break

            # 增强 DSL
            current_dsl = await self.enrich(current_dsl, original_prompt, high_severity_gaps)

        return current_dsl, gaps

    def _format_gaps(self, gaps: List[SemanticGap]) -> str:
        """格式化语义缺失列表为文本"""
        if not gaps:
            return "未检测到语义缺失"

        lines = []
        for i, gap in enumerate(gaps, 1):
            lines.append(f"""
### 缺失 {i}
- 类型: {gap.gap_type}
- 位置: {gap.location}
- 问题: {gap.description}
- 建议: {gap.suggestion}
- 严重性: {gap.severity}
""")
        return "\n".join(lines)

    async def quick_fix(
        self,
        dsl_code: str,
        original_prompt: str,
        fix_type: str
    ) -> str:
        """
        快速修复特定类型的语义缺失

        Args:
            dsl_code: 原始 DSL 代码
            original_prompt: 原始自然语言需求
            fix_type: 修复类型 (variable_undefined | return_missing | error_handling)

        Returns:
            修复后的 DSL 代码
        """
        prompt = f"""你是一个 DSL 代码修复专家。请修复以下 DSL 代码中的 {fix_type} 问题。

## 原始需求
{original_prompt}

## 当前 DSL 代码
```
{dsl_code}
```

## 修复类型
{fix_type}

## 任务
修复上述类型的语义缺失，保持其他部分不变。

## 输出格式
只输出修复后的 DSL 代码，不要有其他解释文字。
"""

        response = self.llm.call(prompt)
        return response.content.strip()

    def enrich_sync(
        self,
        dsl_code: str,
        original_prompt: str,
        gaps: List[SemanticGap]
    ) -> str:
        """
        同步版本的增强

        Args:
            dsl_code: 原始 DSL 代码
            original_prompt: 原始自然语言需求
            gaps: 检测到的语义缺失列表

        Returns:
            增强后的 DSL 代码
        """
        if not gaps:
            return dsl_code

        gaps_text = self._format_gaps(gaps)

        prompt = f"""你是一个 DSL 代码生成专家。请根据以下信息，生成一个语义完整的 DSL 代码。

## 原始需求
{original_prompt}

## 当前 DSL 代码（有问题）
```
{dsl_code}
```

## 检测到的语义缺失
{gaps_text}

## 任务
1. 修复上述所有语义缺失
2. 保持原有的业务逻辑不变
3. 确保生成的 DSL 满足以下要求：
   - 所有使用的变量都有定义
   - 所有分支都有明确的返回值
   - 所有 CALL 都有适当的错误处理

## 输出格式
只输出修复后的完整 DSL 代码，不要有其他解释文字。
"""

        response = self.llm.call(prompt)
        return response.content.strip()