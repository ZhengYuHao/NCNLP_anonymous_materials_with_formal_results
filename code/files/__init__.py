"""scene_dsl_converter — 场景检测提示词 → DSL v1.2 转换器。

核心 API:
    from scene_dsl_converter import (
        SceneDSLConverter,
        ClaudeExtractor,
        MockExtractor,
        AgentSpec,
    )
"""
from .converter import SceneDSLConverter
from .executor import SpecExecutor, RegressionReport, run_regression
from .extractor import ClaudeExtractor, MockExtractor, ExtractorBase
from .renderer import DSLRenderer
from .validator import DSLValidator, ValidationResult
from .schema import (
    AgentSpec,
    PersonaSpec,
    ConstraintSpec,
    InputSpec,
    OutputSpec,
    Scene,
    SubScenario,
    SemanticHook,
    ExampleCase,
)

__all__ = [
    "SceneDSLConverter",
    "SpecExecutor",
    "RegressionReport",
    "run_regression",
    "ClaudeExtractor",
    "MockExtractor",
    "ExtractorBase",
    "DSLRenderer",
    "DSLValidator",
    "ValidationResult",
    "AgentSpec",
    "PersonaSpec",
    "ConstraintSpec",
    "InputSpec",
    "OutputSpec",
    "Scene",
    "SubScenario",
    "SemanticHook",
    "ExampleCase",
]

__version__ = "1.2.0"
