"""
统一的 LLM 客户端模块
提供 OpenAI 兼容的 API 调用接口，供所有模块共用
"""

import os
import json
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from logger import info, warning, error, debug
from dotenv import load_dotenv

# 优化模块：正则提取器
from utils.pre_pattern_extractor import PrePatternExtractor

# 优化模块：缓存客户端
from utils.cached_llm_client import CachedLLMClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


# ============================================================================
# 配置常量
# ============================================================================

# 加载环境变量
load_dotenv()

# 从环境变量读取配置，提供默认值
DEFAULT_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://api.example.com/v1"
)
DEFAULT_API_KEY = os.getenv(
    "LLM_API_KEY",
    ""  # 默认为空，强制从环境变量配置
)


class LLMModel(Enum):
    """支持的模型枚举"""
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_4o = "gpt-4o"
    GPT_4o_MINI = "gpt-4.1-mini-2025-04-14"
    GPT_4_TURBO = "gpt-4-turbo"
    QWEN3_32B = "qwen3-32b-lb-pv"
    MINIMAX_M2_5 = "MiniMax-M2.5"
    GPT_5_TURBO = "gpt-5-nano-2025-08-07"
    DEEPSEEK_V32 = "deepseek-v3.2"


# ============================================================================
# 预定义模型配置
# ============================================================================

MODEL_CONFIGS = {
    "default": {
        "model": LLMModel.GPT_4o_MINI.value,
        "base_url": os.getenv("LLM_BASE_URL", "https://api.example.com/v1"),
        "api_key": os.getenv("LLM_API_KEY", ""),
        "timeout": 360
    },
    "gpt-4o-mini": {
        "model": LLMModel.GPT_4o_MINI.value,
        "base_url": os.getenv("LLM_BASE_URL", "https://api.example.com/v1"),
        "api_key": os.getenv("LLM_API_KEY", ""),
        "timeout": 360
    },
    "gpt-3.5": {
        "model": LLMModel.GPT_35_TURBO.value,
        "base_url": os.getenv("LLM_BASE_URL", "https://api.example.com/v1"),
        "api_key": os.getenv("LLM_API_KEY", ""),
        "timeout": 360
    },
    "gpt-4o": {
        "model": LLMModel.GPT_4o.value,
        "base_url": os.getenv("LLM_BASE_URL", "https://api.example.com/v1"),
        "api_key": os.getenv("LLM_API_KEY", ""),
        "timeout": 360
    },
    "qwen3-32b": {
        "model": LLMModel.QWEN3_32B.value,
        "base_url": os.getenv("QWEN_BASE_URL", "https://api.example.com/v1"),
        "api_key": os.getenv("QWEN_API_KEY", ""),
    },
    "qwen3-custom": {
        "model": LLMModel.QWEN3_32B.value,
        "base_url": os.getenv("QWEN_BASE_URL", "https://api.example.com/v1"),
        "api_key": os.getenv("QWEN_API_KEY", ""),
    },
    "gpt-5": {
        "model": LLMModel.GPT_5_TURBO.value,
        "base_url": os.getenv("LLM_BASE_URL", "https://api.example.com/v1"),
        "api_key": os.getenv("LLM_API_KEY", ""),
        "timeout": 360
    },
    "minimax-m2.5": {
        "model": LLMModel.MINIMAX_M2_5.value,
        "base_url": os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "disable_thinking": True,
        "timeout": 360
    },
    "deepseek-v3.2": {
        "model": LLMModel.DEEPSEEK_V32.value,
        "base_url": os.getenv("LLM_BASE_URL", "https://api.example.com/v1"),
        "api_key": os.getenv("LLM_API_KEY", ""),
        "timeout": 360
    }
}


@dataclass
class LLMResponse:
    """LLM 响应数据类"""
    content: str  # 响应内容
    model: str  # 使用的模型
    usage: Optional[Dict[str, int]] = None  # token 使用情况
    raw_response: Optional[Any] = None  # 原始响应对象


# ============================================================================
# 统一 LLM 客户端
# ============================================================================

class UnifiedLLMClient:
    """
    统一的 LLM 客户端
    
    功能：
    - 提供标准化的 API 调用接口
    - 支持多种调用模式（普通对话、结构化输出、实体抽取等）
    - 统一的错误处理和日志记录
    """
    
    def __init__(
        self,
        model: str = LLMModel.GPT_4o_MINI.value,
        temperature: float = 0.1,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 360,
        enable_cache: bool = False,
        max_tokens: int = 32760,
    ):
        """
        初始化 LLM 客户端

        Args:
            model: 模型名称
            temperature: 温度参数（0-2，越低越确定性）
            base_url: API 基础 URL
            api_key: API 密钥
            max_retries: 最大重试次数
            timeout: 超时时间（秒）
            enable_cache: 是否启用缓存
            max_tokens: 最大输出 token 数（默认 32760）
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("LLM_API_KEY", DEFAULT_API_KEY)
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._client: Optional["OpenAI"] = None
        self.enable_cache = enable_cache
        self._cache_client: Optional[CachedLLMClient] = None
        self.disable_thinking: bool = False
    
    def _get_client(self) -> "OpenAI":
        """获取或创建 OpenAI 客户端实例"""
        if OpenAI is None:
            raise RuntimeError("请先安装 openai: pip install openai")
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout
            )
        return self._client

    def _get_cache_client(self) -> CachedLLMClient:
        """获取或创建缓存客户端实例"""
        if self._cache_client is None:
            self._cache_client = CachedLLMClient(self)
        return self._cache_client

    def _call_without_cache(
        self,
        system_prompt: str,
        user_content: str,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        """
        不使用缓存的调用方法

        Args:
            system_prompt: 系统提示词
            user_content: 用户输入
            temperature: 可选的温度覆盖
            model: 可选的模型覆盖
            max_tokens: 可选的最大输出 token 数覆盖
            response_format: 可选的结构化输出格式

        Returns:
            LLMResponse 对象
        """
        _model = model or self.model
        _temp = temperature if temperature is not None else self.temperature
        _max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # 如果是千问模型，在系统提示词中添加强抑制指令
        if "qwen" in _model.lower():
            suppress_thought = """

【禁止输出思考过程】
- 直接输出最终结果
- 严禁输出推理步骤、思考过程或中间想法
- 严禁使用"好的"、"我现在需要"、"接下来"等引导词
- 只输出符合要求格式的内容
- 输出必须简洁精确
"""
            system_prompt = system_prompt + suppress_thought

        info(f"[LLM调用] 模型: {_model}, Temperature: {_temp}")
        debug(f"[System] {system_prompt[:100]}...")
        debug(f"[User] {user_content[:100]}...")

        client = self._get_client()

        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=_model,
                    temperature=_temp,
                    max_tokens=_max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                if _model.lower().startswith("gemini-3"):
                    # The OpenAI-compatible Gemini endpoint counts hidden
                    # reasoning against max_tokens. Its default reasoning level
                    # can truncate long JSON responses before the payload ends.
                    kwargs["extra_body"] = {
                        "reasoning_effort": os.getenv("LLM_REASONING_EFFORT", "low")
                    }
                if response_format is not None:
                    kwargs["response_format"] = response_format
                resp = client.chat.completions.create(**kwargs)

                content = (resp.choices[0].message.content or "").strip()

                # 清理响应中的格式标签和脏数据
                content = self._clean_llm_response(content)

                usage = None
                if hasattr(resp, 'usage') and resp.usage:
                    prompt_details = getattr(resp.usage, "prompt_tokens_details", None)
                    cached_tokens = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                        "cached_tokens": cached_tokens or 0,
                        "attempt_count": attempt + 1,
                    }

                debug(f"[LLM响应] 长度: {len(content)} 字符")

                return LLMResponse(
                    content=content,
                    model=_model,
                    usage=usage,
                    raw_response=resp
                )

            except Exception as e:
                warning(f"LLM 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    error(f"LLM 调用最终失败: {e}")
                    raise

        raise RuntimeError("LLM 调用失败，已超过最大重试次数")

    def _call_anthropic_compatible(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float,
        model: str,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Anthropic API 兼容调用 (用于 MiniMax M2.5)"""
        import requests
        
        _max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        info(f"[Anthropic兼容] 模型: {model}, Temperature: {temperature}, MaxTokens: {_max_tokens}")
        
        url = f"{self.base_url}/v1/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": model,
            "temperature": temperature,
            "max_tokens": _max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_content}
            ]
        }
        
        # 关闭思考模式（如果模型支持）
        if getattr(self, 'disable_thinking', False):
            data["disable_thinking"] = True
        
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=data, timeout=self.timeout)
                resp.raise_for_status()
                
                result = resp.json()
                
                # 提取文本内容 (处理 thinking 和 text 两种块)
                content_parts = []
                for block in result.get("content", []):
                    if block.get("type") == "text":
                        content_parts.append(block.get("text", ""))
                    elif block.get("type") == "thinking":
                        # 可以选择是否包含思考过程
                        # content_parts.append(block.get("thinking", ""))
                        pass
                
                content = "\n".join(content_parts).strip()
                content = self._clean_llm_response(content)
                
                usage = result.get("usage", {})
                
                return LLMResponse(
                    content=content,
                    model=model,
                    usage={
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    },
                    raw_response=result
                )
                
            except Exception as e:
                warning(f"Anthropic兼容调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    error(f"Anthropic兼容调用最终失败: {e}")
                    raise
        
        raise RuntimeError("Anthropic兼容调用失败，已超过最大重试次数")
    
    def _clean_llm_response(self, content: str) -> str:
        """
        清理 LLM 响应中的格式标签和脏数据

        Args:
            content: 原始 LLM 响应内容

        Returns:
            清理后的内容
        """
        import re

        # 移除完整的格式标签（包括角色名）
        # 匹配格式：<|im_start|>role 或 <|im_end|>
        patterns_to_remove = [
            r'<\|im_start\|>\s*\w+\s*',  # <|im_start|>role 格式
            r'<\|im_end\|>',              # <|im_end|> 格式
            r'<\|.*?\|>',                 # 其他 <|xxx|> 格式的标签
            r'<think>.*?</think>',        # Qwen 思考过程标签
            r'<思考>.*?</思考>',          # 中文思考过程标签
        ]

        cleaned = content
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)

        # 清理多余的空白字符
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # 多个空行变成两个

        # 清理开头的空白字符（包括换行符、空格、制表符、软回车等）
        cleaned = re.sub(r'^[\s\r\n\t\u2028\u2029]+', '', cleaned)

        # 清理结尾的空白字符
        cleaned = re.sub(r'[\s\r\n\t\u2028\u2029]+$', '', cleaned)

        return cleaned

    def _suppress_thought_process(self, user_content: str) -> str:
        """
        在用户提示词末尾添加指令，抑制模型的思考过程输出

        Args:
            user_content: 原始用户提示词

        Returns:
            添加了抑制指令的提示词
        """
        suppress_instruction = """

【关键要求】
1. 直接输出最终结果，不展示思考过程
2. 禁止使用以下任何标记：|im_start|>、|im_end|>、<思考>、</思考> 等
3. 禁止使用引导词：好的、我现在需要、接下来、首先、然后等
4. 输出内容必须符合要求的格式
5. 保持输出简洁精确，不要多余解释
"""

        return user_content + suppress_instruction

    def call(
        self,
        system_prompt: str,
        user_content: str,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        """
        基础对话调用（支持缓存）

        Args:
            system_prompt: 系统提示词
            user_content: 用户输入
            temperature: 可选的温度覆盖
            model: 可选的模型覆盖
            max_tokens: 可选的最大输出 token 数覆盖
            response_format: 可选的结构化输出格式, 例 {"type": "json_object"}

        Returns:
            LLMResponse 对象
        """
        # 如果启用缓存，使用缓存客户端
        if self.enable_cache:
            cache_client = self._get_cache_client()
            response = cache_client.call(system_prompt, user_content, temperature=temperature, model=model,
                                         response_format=response_format)

            # 标记是否来自缓存
            if hasattr(response, 'from_cache') and response.from_cache:
                info("[缓存命中] 使用缓存结果")
            else:
                info("[缓存未命中] 调用 LLM")

            return response
        else:
            # 直接调用，不使用缓存
            return self._call_without_cache(system_prompt, user_content, temperature, model, max_tokens, response_format)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            缓存统计字典
        """
        if self.enable_cache and self._cache_client:
            return self._cache_client.get_stats()
        return {
            "hits": 0,
            "misses": 0,
            "total": 0,
            "hit_rate": 0.0
        }
    
    def call_simple(
        self,
        system_prompt: str,
        user_content: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        简单调用，直接返回字符串
        
        Args:
            system_prompt: 系统提示词
            user_content: 用户输入
            temperature: 可选的温度覆盖
            
        Returns:
            响应文本
        """
        response = self.call(system_prompt, user_content, temperature)
        return response.content
    
    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        enable_optimization: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        优化的实体抽取（极窄化LLM）

        策略：
        1. 正则预处理：提取常见模式（数字+单位、技术栈等）
        2. 如果正则提取足够（3+个实体），直接返回
        3. 否则调用LLM补充
        4. 合并结果（正则优先）

        Args:
            text: 待抽取文本
            entity_types: 期望的实体类型列表
            enable_optimization: 是否启用优化（正则预处理）

        Returns:
            (实体列表, 优化统计信息)
        """
        types_hint = ""
        if entity_types:
            types_hint = "实体类型限定: " + ", ".join(entity_types) + "\n"

        # 步骤1：正则预处理（如果启用优化）
        regex_entities = []
        if enable_optimization:
            debug(f"[实体提取] 尝试正则预处理...")
            regex_entities = PrePatternExtractor.extract(text)

            # 如果正则提取足够（3+个实体），直接返回
            if len(regex_entities) >= 3:
                info(f"[实体提取] 正则提取成功，提取 {len(regex_entities)} 个实体，跳过 LLM")
                stats = {
                    "regex_count": len(regex_entities),
                    "llm_count": 0,
                    "merged_count": len(regex_entities),
                    "llm_called": False,
                    "optimization_enabled": True
                }
                return regex_entities, stats
            elif len(regex_entities) > 0:
                info(f"[实体提取] 正则提取 {len(regex_entities)} 个实体，继续使用 LLM 补充")
            else:
                debug(f"[实体提取] 正则提取未找到实体，使用 LLM")
        else:
            debug(f"[实体提取] 优化未启用，直接使用 LLM")
        
        system_prompt = """你是一个实体抽取专家。你的任务是从文本中识别需要动态调整的"变量"。

""" + types_hint + """

## 一、必须提取的变量类型

✅ **类型1：Mustache模板变量**
- 格式：`{{variable_name}}` 或 `{{{variable_name}}}`
- 示例：`{{conversations}}`、`{{user_input}}`、`{{{content}}}`
- 提取方式：name=去掉{{}}后的变量名，type=String，value=原样

✅ **类型2：数值型变量**
- 具体数字：3年、5人、50万、20轮、2秒
- 时间相关：2周、8周、3天
- 提取方式：识别数字+单位组合

✅ **类型3：枚举/状态码**
- 路由编码："1"、"2"、"3"等
- 状态值："pending"、"active"、"completed"
- 布尔值："true"、"false"、"是"、"否"
- 提取方式：识别数字或固定枚举值

✅ **类型4：JSON/结构化数据引用**
- JSON字段引用：`obj.field`、`data.name`
- 数组下标：`items[0]`、`list[1]`
- 示例：`role`、`content`（在JSON上下文中）

✅ **类型5：DSL/代码片段中的占位符**
- 条件占位：`{condition}`、`{action}`
- 变量引用：`{{variable}}`（已在类型1覆盖）
- 示例：`{c2(X)}`、`{L1(Y)}`

✅ **类型6：多智能体状态变量**
- 跨智能体状态：`previous_agent`、`process`
- 上下文变量：`history`、`context`
- 流程变量：`scene`、`agent`、`type`

✅ **类型7：路由关键词模式**
- 判断条件："没有brief"、"无其他补充"
- 触发话术："可以"、"好的"、"行"
- 提取方式：识别为keyword类型

## 二、不提取的常量类型

❌ 功能需求描述："需要支持多轮对话"、"用大模型做底座"
❌ 固定配置描述："上下文窗口存20轮"
❌ 通用动词（单独出现）："需要"、"要"、"支持"、"实现"
❌ 架构描述："微服务架构"、"分布式系统"
❌ 具体数值已作为变量提取后，不再重复提取其组成部分

## 三、识别规则

1. **Mustache变量优先**：所有 `{{...}}` 必须提取，无论上下文
2. **数值+单位组合**：如"5个人"提取为一个变量team_size=5
3. **JSON字段上下文**：如果文本中出现JSON示例，提取其字段名
4. **枚举值识别**：数字1、2、3等在路由上下文中视为枚举
5. **状态变量识别**：previous_agent、process等在多智能体上下文中提取

## 四、输出格式

```json
[
  {{
    "name": "变量英文名(snake_case)",
    "original_text": "原文精确片段",
    "start_index": 起始位置,
    "end_index": 结束位置,
    "type": "String|Integer|Boolean|List|Enum|Keyword|Mustache",
    "value": "当前值",
    "category": "numerical|template|enum|keyword|dsl|state|json"
  }}
]
```

## 五、完整示例

示例1（Mustache变量）：
原文: "聊天历史记录：{{conversations}}，当前输入：{{user_input}}"
提取:
- {{"name": "conversations", "original_text": "{{conversations}}", "type": "Mustache", "category": "template"}}
- {{"name": "user_input", "original_text": "{{user_input}}", "type": "Mustache", "category": "template"}}

示例2（数值+关键词组合）：
原文: "需要5个Java程序员，预算大约30万以内，没有brief"
提取:
- {{"name": "team_size", "original_text": "5个", "value": 5, "type": "Integer", "category": "numerical"}}
- {{"name": "budget", "original_text": "30万", "value": 300000, "type": "Integer", "category": "numerical"}}
- {{"name": "no_brief", "original_text": "没有brief", "type": "Keyword", "category": "keyword"}}

示例3（路由枚举）：
原文: "1表示表单填写，2表示选号阶段"
提取:
- {{"name": "route_form_fill", "original_text": "1", "value": "1", "type": "Enum", "category": "enum"}}
- {{"name": "route_selection", "original_text": "2", "value": "2", "type": "Enum", "category": "enum"}}

示例4（多智能体状态）：
原文: "根据previous_agent和当前用户输入判断"
提取:
- {{"name": "previous_agent", "original_text": "previous_agent", "type": "String", "category": "state"}}

示例5（JSON字段）：
原文: '{{"role": "user", "content": "无其他补充"}}'
提取:
- {{"name": "role", "original_text": "role", "type": "String", "category": "json"}}
- {{"name": "content", "original_text": "content", "type": "String", "category": "json"}}

## 六、严格规则

1. 只输出 JSON 数组格式，不要有任何其他文字
2. Mustache变量 `{{...}}` 必须原样保留在 original_text 中
3. 必须返回精确的 start_index 和 end_index
4. type 必须是：String|Integer|Boolean|List|Enum|Keyword|Mustache
5. category 必须是：numerical|template|enum|keyword|dsl|state|json
6. 如果同一文本符合多种类型，优先选择更具体的类型"""

        # 步骤2：调用LLM补充（如果需要）
        response = self.call(system_prompt, text)

        # 打印原始响应用于调试
        debug(f"[实体抽取] LLM 原始响应:\n{response.content}\n")

        try:
            # 先清理可能的 markdown 包裹
            cleaned = response.content.strip()
            # 去除 ```json ... ``` 包裹
            json_block_match = __import__('re').search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned)
            if json_block_match:
                cleaned = json_block_match.group(1).strip()
            llm_entities = json.loads(cleaned)
            if not isinstance(llm_entities, list):
                warning(f"[实体抽取] 响应格式不是数组，实际类型: {type(llm_entities)}")
                # 如果返回的是 {"entities": [...]} 格式, 尝试提取
                if isinstance(llm_entities, dict) and "entities" in llm_entities:
                    llm_entities = llm_entities["entities"]
                else:
                    llm_entities = []
        except json.JSONDecodeError as e:
            warning(f"[实体抽取] JSON 解析失败: {e}")
            # 兜底: 尝试从文本中提取第一个 JSON 数组
            try:
                import re as _re
                # 括号平衡法提取 JSON 数组
                start = cleaned.find('[')
                if start != -1:
                    depth = 0
                    in_str = False
                    esc = False
                    for idx in range(start, len(cleaned)):
                        ch = cleaned[idx]
                        if esc:
                            esc = False
                            continue
                        if ch == '\\':
                            esc = True
                            continue
                        if ch == '"':
                            in_str = not in_str
                            continue
                        if in_str:
                            continue
                        if ch == '[':
                            depth += 1
                        elif ch == ']':
                            depth -= 1
                            if depth == 0:
                                candidate = cleaned[start:idx+1]
                                llm_entities = json.loads(candidate)
                                if isinstance(llm_entities, list):
                                    break
                    else:
                        llm_entities = []
                else:
                    llm_entities = []
            except (json.JSONDecodeError, ValueError, UnboundLocalError):
                llm_entities = []

        # 步骤3：合并结果（正则优先）
        if enable_optimization and regex_entities:
            merged_entities = PrePatternExtractor.merge_with_llm(regex_entities, llm_entities)
            info(f"[实体抽取] 合并结果: 正则{len(regex_entities)} + LLM{len(llm_entities)} → {len(merged_entities)}")
            stats = {
                "regex_count": len(regex_entities),
                "llm_count": len(llm_entities),
                "merged_count": len(merged_entities),
                "llm_called": True,
                "optimization_enabled": True
            }
            return merged_entities, stats
        else:
            # 如果没有启用优化或正则未提取到实体，直接返回LLM结果
            info(f"[实体抽取] 识别到 {len(llm_entities)} 个实体（纯LLM）")
            stats = {
                "regex_count": len(regex_entities) if enable_optimization else 0,
                "llm_count": len(llm_entities),
                "merged_count": len(llm_entities),
                "llm_called": True,
                "optimization_enabled": enable_optimization
            }
            return llm_entities, stats
    
    def detect_ambiguity(self, text: str) -> Optional[str]:
        """
        检测文本歧义
        
        Args:
            text: 待检测文本
            
        Returns:
            如果存在歧义则返回描述，否则返回 None
        """
        system_prompt = """检测以下文本是否存在严重的语义歧义或多重解读可能。

判断标准:
- 句法结构是否存在歧义(如定语从句归属不明)
- 指代对象是否清晰
- 逻辑关系是否唯一

如果文本清晰无歧义,请仅回复 "PASS"
如果存在歧义,请简短描述歧义点(不超过30字)

示例:
输入: "咬死了猎人的狗"
输出: "无法确定主语,可能是狗咬死了猎人,也可能是猎人的狗被咬死"

输入: "请优化RAG的检索流程"
输出: "PASS"
"""
        
        response = self.call(system_prompt, text)
        raw = response.content.strip()
        
        # 判断是否通过
        if "PASS" in raw.upper():
            return None
        if len(raw) <= 80 and any(kw in raw for kw in ("无歧义", "通过", "无严重歧义", "清晰无歧义")):
            if not any(bad in raw for bad in ("存在歧义", "有歧义", "存在严重歧义")):
                return None
        
        return raw
    
    def standardize_text(self, text: str) -> str:
        """
        文本标准化（口语转书面语）
        
        Args:
            text: 原始文本
            
        Returns:
            标准化后的文本
        """
        system_prompt = """你是一个文本标准化工具。你的任务是将输入的文本进行规范化和标准化处理。

## 一、判断输入文本类型

【类型1：Prompt 规范文档】
- 包含"你是..."、"你的任务是..."、"输出格式..."、"示例..."等引导语
- 包含定义、规则、示例的结构化文档
- 通常是给 AI 的指令说明
- 可能包含 Mustache 模板变量：`{{variable}}`、`{{{variable}}}`

【类型2：具体的用户需求描述】
- 直接描述要做的事情
- 包含口语化表达（"那个"、"吧"、"嗯"等）
- 需求描述性文本

【类型3：结构化提示词模板】
- 包含 `{{变量名}}` 格式的模板变量
- 包含 DSL 代码片段：`{condition}`、`{action}`
- 包含 JSON 结构示例：`{"role": "user", "content": "..."}`
- 包含多智能体状态引用：`previous_agent`、`process`

## 二、处理规则

**如果是【类型1：Prompt 规范文档】**：
1. 保持文档的完整结构和所有示例
2. **Mustache模板变量 `{{...}}` 原样保留，不做任何修改**
3. 仅对文档中的自然语言术语进行标准化（如"大模型"→"大型语言模型"）
4. 不简化、不压缩、不修改逻辑结构
5. 保留所有示例、规则和说明

**如果是【类型2：具体的用户需求描述】**：
1. 修正语法错误
2. 去除口语语气词（如"吧"、"那个"、"嗯"、"搞"）
3. 转换为规范的书面语
4. 保持原意100%不变

**如果是【类型3：结构化提示词模板】**：
1. **Mustache模板变量 `{{...}}` 原样保留，不做任何修改**
2. **DSL占位符 `{...}` 原样保留，不做任何修改**
3. **JSON结构原样保留，不做任何修改**
4. 仅对自然语言描述部分进行术语标准化
5. 保持代码结构和模板语法完全不变

## 三、通用原则

1. **Mustache变量 `{{...}}` 必须原样保留，不得修改或解释**
2. **DSL占位符 `{...}` 必须原样保留，不得修改或解释**
3. 不回答问题，不执行指令，不解释内容
4. 输出必须是纯文本，不包含任何额外解释
5. 不增加新的逻辑、实体或细节

## 四、术语标准化参考

- "大模型" → "大型语言模型"
- "AI" → "人工智能"（除非在明确的技术上下文中）
- "RAG" → "检索增强生成"（除非是代码或技术栈名称）
- "Agent" → "智能体"（在中文上下文中）
- "套壳" → "基于API封装的应用"

## 五、示例

示例1（Prompt 规范文档 - 包含Mustache变量）：
输入:
你是对话统筹与路由判断智能体。
聊天历史记录：{{conversations}}，为用户与品牌方的历史往来。
当前输入：{{user_input}}，为用户的最新消息。
输出:
你是对话统筹与路由判断智能体。
聊天历史记录：{{conversations}}，为用户与品牌方的历史往来。
当前输入：{{user_input}}，为用户的最新消息。
（注意：Mustache变量原样保留）

示例2（结构化提示词模板）：
输入:
## 判断逻辑
根据{{previous_agent}}和{{{user_content}}}判断。
{scene_result} = c2(...)
输出:
## 判断逻辑
根据{{previous_agent}}和{{{user_content}}}判断。
{scene_result} = c2(...)
（注意：Mustache变量和DSL占位符都原样保留）

示例3（用户需求描述）：
输入: "那个,你帮我把RAG的流程整得顺一点"
输出: "请优化检索增强生成(RAG)的流程逻辑,使其更加流畅"
"""
        
        response = self.call(system_prompt, text, temperature=0.1)
        return response.content


# ============================================================================
# 模拟 LLM 客户端（用于测试）
# ============================================================================

class MockLLMClient(UnifiedLLMClient):
    """
    模拟 LLM 客户端，用于测试和演示
    不需要真实的 API 连接
    """
    
    def __init__(self, **kwargs):
        # 不调用父类初始化，避免 API 连接
        self.model = kwargs.get('model', LLMModel.GPT_35_TURBO.value)
        self.temperature = kwargs.get('temperature', 0.1)
    
    def call(
        self,
        system_prompt: str,
        user_content: str,
        temperature: Optional[float] = None,
        model: Optional[str] = None
    ) -> LLMResponse:
        """模拟调用"""
        info(f"[MockLLM] 模拟调用")
        debug(f"[System] {system_prompt[:50]}...")
        debug(f"[User] {user_content[:50]}...")
        
        # 根据系统提示词类型返回不同的模拟响应
        if "DSL" in system_prompt or "逻辑重构" in system_prompt:
            content = self._mock_dsl_transpilation(user_content)
        elif "实体抽取" in system_prompt or "变量" in system_prompt:
            content = self._mock_entity_extraction(user_content)
        elif "歧义" in system_prompt:
            content = self._mock_ambiguity_detection(user_content)
        elif "标准化" in system_prompt or "书面语" in system_prompt:
            content = self._mock_standardization(user_content)
        else:
            content = f"[模拟响应] {user_content[:50]}..."
        
        return LLMResponse(
            content=content,
            model=self.model,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
    
    def _mock_entity_extraction(self, text: str) -> str:
        """模拟实体抽取（修复重复定义问题）"""
        import re
        entities = []

        # 用于跟踪相同单位的计数（修复重复定义）
        unit_counter = {}

        # 识别数字+单位模式
        for match in re.finditer(r'(\d+)(年|周|月|天|小时|分钟)', text):
            unit = match.group(2)
            value = int(match.group(1))

            # 对相同单位进行计数，创建唯一名称
            count = unit_counter.get(unit, 0) + 1
            unit_counter[unit] = count

            # 如果是第一个该单位，使用标准名称
            if count == 1:
                var_name = f"duration_{unit}"
            else:
                # 如果是重复的单位，添加后缀
                var_name = f"duration_{unit}_{count}"

            entities.append({
                "name": var_name,
                "original_text": match.group(0),
                "start_index": match.start(),
                "end_index": match.end(),
                "type": "Integer",
                "value": value
            })
        
        # 识别专业术语
        tech_terms = ["Java程序员", "Python", "数据分析", "机器学习", "前端开发"]
        for term in tech_terms:
            if term in text:
                idx = text.find(term)
                entities.append({
                    "name": f"tech_term_{len(entities)}",
                    "original_text": term,
                    "start_index": idx,
                    "end_index": idx + len(term),
                    "type": "String",
                    "value": term
                })
        
        return json.dumps(entities, ensure_ascii=False)
    
    def _mock_ambiguity_detection(self, text: str) -> str:
        """模拟歧义检测"""
        ambiguous_patterns = ["意思", "那个", "随便", "看着办", "差不多"]
        for pattern in ambiguous_patterns:
            if pattern in text:
                return f"检测到歧义表达: '{pattern}' 含义不明确"
        return "PASS"
    
    def _mock_standardization(self, text: str) -> str:
        """模拟文本标准化"""
        # 简单的替换规则
        result = text
        replacements = {
            "搞一个": "开发一个",
            "弄一下": "处理",
            "那个": "",
            "吧": "",
            "嗯": "",
        }
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result.strip()

    def _mock_dsl_transpilation(self, user_content: str) -> str:
        """模拟DSL转译：解析用户输入并生成DSL伪代码"""
        import re
        
        # 解析变量定义
        variables = []
        in_variable_section = False
        lines = user_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('【变量定义】'):
                in_variable_section = True
                continue
            elif line.startswith('【'):
                # 进入其他部分
                in_variable_section = False
                continue
            
            if in_variable_section and line.startswith('-'):
                # 格式: - var_name: Type [= value]
                match = re.match(r'-\s*(\w+):\s*(\w+)(?:\s*=\s*(.+))?', line)
                if match:
                    var_name, var_type, initial_value = match.groups()
                    variables.append({
                        'name': var_name,
                        'type': var_type,
                        'initial_value': initial_value
                    })
        
        # 解析逻辑描述（简单关键词检测）
        logic = ""
        in_logic_section = False
        for line in lines:
            line = line.strip()
            if line.startswith('【逻辑描述】'):
                in_logic_section = True
                continue
            elif line.startswith('【'):
                in_logic_section = False
                continue
            
            if in_logic_section:
                logic = line
                break
        
        # 构建DSL代码
        dsl_lines = []
        
        # 1. 变量定义
        for var in variables:
            if var['initial_value']:
                dsl_lines.append(f"DEFINE {{{{{var['name']}}}}}: {var['type']} = {var['initial_value']}")
            else:
                dsl_lines.append(f"DEFINE {{{{{var['name']}}}}}: {var['type']}")
        
        # 确保 result 变量被定义（如果尚未定义）
        result_var_defined = any(var['name'] == 'result' for var in variables)
        if not result_var_defined:
            dsl_lines.append("DEFINE {{result}}: String")
        
        # 2. 根据逻辑描述生成控制流
        if '如果' in logic or '若' in logic:
            # 简单条件语句
            dsl_lines.append("")
            dsl_lines.append("# 条件判断")
            dsl_lines.append("IF {{condition}} == true")
            dsl_lines.append("    {{result}} = CALL process_success()")
            dsl_lines.append("ELSE")
            dsl_lines.append("    {{result}} = CALL process_failure()")
            dsl_lines.append("ENDIF")
        
        if '对于每个' in logic or '遍历' in logic:
            # 简单循环
            dsl_lines.append("")
            dsl_lines.append("# 循环处理")
            dsl_lines.append("FOR {{item}} IN {{collection}}")
            dsl_lines.append("    {{result}} = CALL process_item({{item}})")
            dsl_lines.append("ENDFOR")
        
        # 如果没有生成任何控制流，添加一个默认的CALL
        if len(dsl_lines) <= len(variables) + (1 if not result_var_defined else 0):
            dsl_lines.append("")
            dsl_lines.append("# 默认处理")
            dsl_lines.append("{{result}} = CALL generate_output()")
        
        # 3. 添加返回语句
        dsl_lines.append("")
        dsl_lines.append("RETURN {{result}}")
        
        return '\n'.join(dsl_lines)


# ============================================================================
# 工厂函数
# ============================================================================

def create_llm_client(
    use_mock: bool = False,
    config_name: Optional[str] = "default",
    **kwargs
) -> UnifiedLLMClient:
    """
    创建 LLM 客户端的工厂函数

    Args:
        use_mock: 是否使用模拟客户端
        config_name: 预定义配置名称，可选值: "default", "gpt-3.5", "gpt-4", "qwen3-32b"
        **kwargs: 传递给客户端的其他参数（可覆盖配置中的值）
            - enable_cache: 是否启用缓存（默认False）

    Returns:
        LLM 客户端实例

    Examples:
        # 使用默认配置
        client = create_llm_client()

        # 使用预定义的 Qwen3 配置
        client = create_llm_client(config_name="qwen3-32b")

        # 使用配置但覆盖部分参数
        client = create_llm_client(config_name="qwen3-32b", temperature=0.5)

        # 启用缓存
        client = create_llm_client(enable_cache=True)
    """
    if use_mock:
        info("[LLM] 使用模拟客户端")
        return MockLLMClient(**kwargs)

    # 如果指定了配置名称，则应用配置
    if config_name:
        if config_name not in MODEL_CONFIGS:
            available = ", ".join(MODEL_CONFIGS.keys())
            raise ValueError(f"未知的配置名称: {config_name}, 可用配置: {available}")

        config = dict(MODEL_CONFIGS[config_name])  # 浅拷贝，避免污染原始配置

        # 🔧 关键修复：始终从 os.environ 读取最新的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
        # （MODEL_CONFIGS 在模块导入时已固化了 .env 的值，运行时更新 os.environ 不会自动同步）
        _env_base_url = os.environ.get("LLM_BASE_URL")
        _env_api_key = os.environ.get("LLM_API_KEY")
        _env_model = os.environ.get("LLM_MODEL")
        if _env_base_url:
            config["base_url"] = _env_base_url
        if _env_api_key:
            config["api_key"] = _env_api_key
        if _env_model:
            config["model"] = _env_model

        info(f"[LLM] 使用预定义配置: {config_name}")
        info(f"[LLM]  模型: {config['model']}")
        info(f"[LLM]  地址: {config['base_url']}")
        if kwargs.get('enable_cache', False):
            info(f"[LLM]  缓存: 已启用")
        if config.get('disable_thinking'):
            info(f"[LLM]  思考模式: 已关闭")

        # 合并配置：预定义配置 + kwargs（kwargs优先级更高）
        merged_kwargs = {**config, **kwargs}
        
        # 提取 disable_thinking（不在 __init__ 参数中）
        disable_thinking = merged_kwargs.pop('disable_thinking', False)
        
        client = UnifiedLLMClient(**merged_kwargs)
        client.disable_thinking = disable_thinking
        return client
    else:
        if kwargs.get('enable_cache', False):
            info("[LLM] 使用真实客户端（自定义配置，缓存已启用）")
        else:
            info("[LLM] 使用真实客户端（自定义配置）")
        return UnifiedLLMClient(**kwargs)


# ============================================================================
# 异步调用支持（可选）
# ============================================================================

async def invoke_function(func_name: str, *args, **kwargs) -> Any:
    """
    异步函数调用接口（供 generated_workflow.py 使用）

    Args:
        func_name: 函数名称
        *args: 位置参数（将被转换为列表传递）
        **kwargs: 函数参数

    Returns:
        函数执行结果
    """
    client = create_llm_client()

    # 构建调用提示词
    system_prompt = f"执行函数: {func_name}"

    # 将位置参数和关键字参数合并到一个字典中传递
    call_params = {"args": args, **kwargs}
    user_content = json.dumps(call_params, ensure_ascii=False)

    response = client.call(system_prompt, user_content)
    return response.content


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    # 方式1: 使用预定义配置（推荐）
    print("=" * 60)
    print("方式1: 使用预定义配置")
    print("=" * 60)
    
    # 使用 Qwen3-32B 配置
    qwen_client = create_llm_client(config_name="qwen3-32b")
    response = qwen_client.call_simple("你是一个助手", "你好，你是谁？")
    info(f"Qwen3 响应: {response[:100]}...")
    
    # 方式2: 使用默认配置
    print("\n" + "=" * 60)
    print("方式2: 使用默认配置")
    print("=" * 60)
    
    default_client = create_llm_client()
    
    # 测试实体抽取
    entities = default_client.extract_entities(
        "请为一位有3年经验的Java程序员生成一个为期2周的Python学习计划"
    )
    info(f"抽取到的实体: {json.dumps(entities, ensure_ascii=False, indent=2)}")
    
    # 测试歧义检测
    ambiguity = default_client.detect_ambiguity("这个需求没啥意思,你看着办")
    info(f"歧义检测结果: {ambiguity}")
    
    # 测试文本标准化
    standardized = default_client.standardize_text("那个,帮我搞一个RAG的应用吧")
    info(f"标准化结果: {standardized}")
    
    # 方式3: 使用配置并覆盖参数
    print("\n" + "=" * 60)
    print("方式3: 使用配置并覆盖参数")
    print("=" * 60)
    
    custom_client = create_llm_client(
        config_name="qwen3-32b",
        temperature=0.8,
        timeout=120
    )
    info("已创建自定义 Qwen3 客户端（temperature=0.8, timeout=120）")
    
    # 显示可用的配置列表
    print("\n" + "=" * 60)
    print("可用的预定义配置:")
    print("=" * 60)
    for name, config in MODEL_CONFIGS.items():
        print(f"- {name}: {config['model']} @ {config['base_url']}")
