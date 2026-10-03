"""
DSL v2 主编译器
整合 llm_client.py 统一接口和 files 模块的 DSL v1.2 转换流程
"""

import os
import sys
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from llm_client import UnifiedLLMClient, LLMModel, create_llm_client
    from logger import info, warning, error, debug
except ImportError:
    UnifiedLLMClient = None
    LLMModel = None
    create_llm_client = None

    def info(msg): print(f"[INFO] {msg}")
    def warning(msg): print(f"[WARNING] {msg}")
    def error(msg): print(f"[ERROR] {msg}")
    def debug(msg): print(f"[DEBUG] {msg}")

from files.converter import SceneDSLConverter
from files.validator import DSLValidator, ValidationResult as FilesValidationResult, ValidationError
from files.renderer import DSLRenderer
from files.schema import AgentSpec, SubScenario

from .extractor import UnifiedExtractor, MockExtractor
from .autofix import AutoFixEngine
from .retry_loop import RetryLoop, RetryHistory, RetryAttempt


@dataclass
class CompilerResult:
    """编译结果"""
    success: bool
    dsl_code: str
    validation_result: Optional[FilesValidationResult] = None
    history: Optional[RetryHistory] = None
    total_tokens: int = 0
    error_message: Optional[str] = None
    spec: Optional[AgentSpec] = None
    # 中间数据透传——供前端阶段3展示
    fact_spec: Optional[Any] = None
    classified_spec: Optional[Any] = None


class PromptCompiler:
    """
    DSL v2 主编译器

    正确集成 files 模块的 SceneDSLConverter 流程:
    1. extract: 使用 UnifiedExtractor (llm_client.py) 从自然语言提取 AgentSpec
    2. render: 使用 DSLRenderer 将 AgentSpec 渲染为 DSL v1.2 代码
    3. validate: 使用 DSLValidator 验证 DSL 代码

    支持自动修复和重试机制。
    """

    def __init__(
        self,
        llm_client: Optional[UnifiedLLMClient] = None,
        max_retries: int = 3,
        auto_fix_threshold: int = 3,
        use_mock: bool = False,
        strict: bool = True
    ):
        if llm_client is None and not use_mock:
            if create_llm_client is not None:
                # 使用 create_llm_client 以获取最新的环境变量配置
                llm_client = create_llm_client(config_name="default", temperature=0.1)
            elif UnifiedLLMClient is not None:
                llm_client = UnifiedLLMClient(
                    model=LLMModel.GPT_4o_MINI.value if LLMModel else "gpt-4o-mini-2024-07-18",
                    temperature=0.1,
                    enable_cache=False,
                    max_tokens=32760,
                    timeout=360,
                )
            else:
                raise RuntimeError("llm_client.py 不可用且 use_mock=False")

        self.use_mock = use_mock
        self.max_retries = max_retries
        self.auto_fix_threshold = auto_fix_threshold
        self.strict = strict

        if use_mock:
            extractor = MockExtractor("fixture.json")
        else:
            extractor = UnifiedExtractor(llm_client=llm_client)

        self.converter = SceneDSLConverter(
            extractor=extractor,
            renderer=DSLRenderer(),
            validator=DSLValidator(),
            strict=strict
        )

        self.autofix = AutoFixEngine()
        self.retry_loop = RetryLoop(max_retries=max_retries)

    def compile(self, prompt_text: str) -> CompilerResult:
        return self.compile_with_retry({"prompt_text": prompt_text})

    def compile_with_retry(self, prompt_dict: Dict[str, Any]) -> CompilerResult:
        prompt_text = prompt_dict.get("prompt_text", "")
        error_feedback = prompt_dict.get("error_feedback", "")

        history = RetryHistory()
        last_error: Optional[str] = None
        last_validation: Optional[FilesValidationResult] = None
        last_spec: Optional[AgentSpec] = None
        last_dsl: Optional[str] = None

        for attempt in range(self.max_retries):
            info(f"\n🔄 第 {attempt + 1} 次编译尝试...")

            spec = None
            try:
                if error_feedback and attempt > 0:
                    full_prompt = f"{prompt_text}\n\n# 上一次尝试的错误:\n{error_feedback}\n\n请根据错误反馈修正 DSL 代码。"
                else:
                    full_prompt = prompt_text

                if self.use_mock:
                    spec_dict = self._mock_extract(full_prompt)
                    spec = AgentSpec.model_validate(spec_dict)
                else:
                    spec = self.converter.extractor.extract(full_prompt)
                    usage = getattr(self.converter.extractor, 'last_usage', None)
                    if usage:
                        self.retry_loop.add_tokens(
                            usage.get('prompt_tokens', 0),
                            usage.get('completion_tokens', 0)
                        )
                        history.total_tokens = self.retry_loop.history.total_tokens

            except Exception as e:
                error(f"❌ 提取异常: {e}")
                history.final_decision = 'error'
                return CompilerResult(
                    success=False,
                    dsl_code="",
                    validation_result=FilesValidationResult(
                        ok=False,
                        errors=[ValidationError(line=0, message=f"提取失败: {e}")]
                    ),
                    history=history,
                    total_tokens=history.total_tokens,
                )

            spec = self._fix_spec(spec)
            last_spec = spec

            # 使用新管线（PipelineV2: classify → render → validate → gate）
            try:
                dsl_code = self.converter.convert_from_spec(spec)
                last_dsl = dsl_code
                # 旧 DSLValidator 语法检查（去重后 block_id 唯一性已通过）
                validation = self.converter.validator.validate(dsl_code)
                last_validation = validation

                # 提取中间数据
                pipeline_result = self.converter._last_pipeline_result
                fact_spec = pipeline_result.fact_spec if pipeline_result else None
                classified_spec = pipeline_result.classified_spec if pipeline_result else None

                info(f"✅ 编译成功（新管线）！")
                history.final_decision = 'success'
                return CompilerResult(
                    success=True,
                    dsl_code=dsl_code,
                    validation_result=validation,
                    history=history,
                    total_tokens=history.total_tokens,
                    spec=spec,
                    fact_spec=fact_spec,
                    classified_spec=classified_spec,
                )
            except ValueError as e:
                error(f"❌ 新管线编译失败: {e}")
                # 回退到旧管道渲染
                dsl_code = self.converter.renderer.render(spec)
                last_dsl = dsl_code
                validation = self.converter.validator.validate(dsl_code)
                last_validation = validation

                if validation.ok or (not self.strict and len(validation.errors) == 0):
                    info(f"✅ 编译成功（旧管线回退）！")
                    history.final_decision = 'fallback'
                    return CompilerResult(
                        success=True,
                        dsl_code=dsl_code,
                        validation_result=validation,
                        history=history,
                        total_tokens=history.total_tokens,
                        spec=spec
                    )

            retry_attempt = RetryAttempt(attempt=attempt + 1)
            retry_attempt.dsl_code = last_dsl or ""
            retry_attempt.errors = last_validation.errors if last_validation else []
            retry_attempt.error_count = len(retry_attempt.errors)
            history.add_attempt(retry_attempt)

            error(f"❌ 验证失败，发现 {len(last_validation.errors) if last_validation else 0} 个错误")

            if attempt < self.max_retries - 1:
                error_feedback = self._generate_error_feedback(last_validation)

                fixed_dsl, fix_count = self.autofix.fix(last_dsl or "", last_validation.errors if last_validation else [])

                if fix_count > 0:
                    info(f"  🔧 自动修复了 {fix_count} 个语法错误")
                    fixed_validation = self.converter.validator.validate(fixed_dsl)

                    if fixed_validation.ok:
                        info(f"  ✅ 自动修复成功！")
                        history.final_decision = 'auto_fixed'
                        history.auto_fix_applied = True
                        return CompilerResult(
                            success=True,
                            dsl_code=fixed_dsl,
                            validation_result=fixed_validation,
                            history=history,
                            total_tokens=history.total_tokens,
                            spec=spec
                        )
                    else:
                        info(f"  ⚠️  自动修复不完整，继续...")
                        last_error = f"验证错误: {len(fixed_validation.errors)} 个"
                        last_validation = fixed_validation
                        last_dsl = fixed_dsl
                else:
                    last_error = f"验证错误: {len(validation.errors)} 个"
            else:
                last_error = f"验证错误: {len(validation.errors)} 个"

        history.final_decision = 'failed'
        return CompilerResult(
            success=False,
            dsl_code=last_dsl or "",
            validation_result=last_validation,
            history=history,
            total_tokens=history.total_tokens,
            error_message=last_error,
            spec=last_spec
        )

    def _mock_extract(self, prompt_text: str) -> Dict[str, Any]:
        """Mock 模式下的提取"""
        return {
            "agent_name": "MockAgent",
            "agent_description": f"从 '{prompt_text[:50]}...' 生成的 Mock Agent",
            "persona": {
                "role": "助理",
                "capabilities": ["智能分类"]
            },
            "inputs": [
                {"name": "user_input", "type": "String", "required": True}
            ],
            "outputs": [
                {"name": "scene", "type": "String"},
                {"name": "agent", "type": "String"}
            ],
            "scenes": [
                {
                    "priority": 1,
                    "block_id": "b1_default",
                    "block_description": "默认场景",
                    "scene_id": 0,
                    "agent_expression": "\"默认助理\"",
                    "sub_scenarios": [
                        {
                            "name": "默认分类",
                            "trigger_keywords": ["其他"],
                            "exclusion_keywords": [],
                            "semantic": None
                        }
                    ]
                }
            ],
            "fallback_scene_id": 0,
            "fallback_agent_expression": "\"默认\""
        }

    def _fix_spec(self, spec: AgentSpec) -> AgentSpec:
        """修复 LLM 返回的不完整数据"""
        fixed_scenes = []
        for scene in spec.scenes:
            if not scene.sub_scenarios:
                scene.sub_scenarios = [
                    SubScenario(
                        name="默认子场景",
                        trigger_keywords=["default"],
                        exclusion_keywords=[],
                        semantic=None
                    )
                ]
            fixed_scenes.append(scene)
        spec.scenes = fixed_scenes
        return spec

    def _generate_error_feedback(self, validation: FilesValidationResult) -> str:
        """生成错误反馈信息"""
        if not validation.errors:
            return ""

        error_lines = ["验证错误:"]
        for i, err in enumerate(validation.errors[:5], 1):
            error_lines.append(f"  {i}. {err}")

        if len(validation.errors) > 5:
            error_lines.append(f"  ... 还有 {len(validation.errors) - 5} 个错误")

        return "\n".join(error_lines)
