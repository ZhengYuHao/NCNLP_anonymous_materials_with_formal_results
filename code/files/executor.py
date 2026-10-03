"""DSL v1.2 语义执行器(用于回归验证)。

这是一个轻量级解释器:给定一个 AgentSpec 和运行时输入,
按 v1.2 规范中定义的语义执行,返回 (scene, agent)。

最重要的用途:
- 对 EXAMPLES 里每个 CASE 跑一遍
- 验证 expected 结果能被规则真实命中
- 这就是"防止 Python 空心化"的最后一道保险

若任何 CASE 失败,说明抽取出的关键词列表或优先级有问题,
需要回到 LLM 抽取阶段微调 prompt 或人工修正 JSON。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .schema import AgentSpec, Scene, SubScenario


@dataclass
class ExecutionResult:
    scene: int
    agent: str
    matched_block_id: str
    semantic_calls: list[str] = field(default_factory=list)


def default_semantic_stub(function_name: str, **kwargs) -> str:  # noqa: ARG001
    """默认语义函数桩: 返回 'positive',让 gate 多半通过。
    生产环境需替换为真实 LLM 调用。
    """
    return "positive"


class SpecExecutor:
    """把 AgentSpec 当成可执行规则集来跑。"""

    def __init__(
        self,
        semantic_fn: Optional[Callable[..., str]] = None,
    ):
        self.semantic_fn = semantic_fn or default_semantic_stub

    def execute(self, spec: AgentSpec, runtime: dict[str, Any]) -> ExecutionResult:
        user_input = str(runtime.get("user_input", ""))
        previous_agent = str(runtime.get("previous_agent", ""))

        semantic_calls: list[str] = []

        for scene in spec.scenes:  # 已按 priority 排序
            match = self._try_match_scene(
                scene, user_input, runtime, semantic_calls,
            )
            if match is not None:
                agent_val = self._resolve_agent_expr(scene.agent_expression, previous_agent)
                return ExecutionResult(
                    scene=scene.scene_id,
                    agent=agent_val,
                    matched_block_id=scene.block_id,
                    semantic_calls=semantic_calls,
                )

        # fallback
        agent_val = self._resolve_agent_expr(
            spec.fallback_agent_expression, previous_agent,
        )
        return ExecutionResult(
            scene=spec.fallback_scene_id,
            agent=agent_val,
            matched_block_id="b_fallback",
            semantic_calls=semantic_calls,
        )

    # --- 匹配逻辑 ---

    def _try_match_scene(
        self,
        scene: Scene,
        user_input: str,
        runtime: dict[str, Any],
        semantic_calls: list[str],
    ) -> Optional[SubScenario]:
        for sub in scene.sub_scenarios:
            if not self._contains_any(user_input, sub.trigger_keywords):
                continue
            if sub.exclusion_keywords and self._contains_any(
                user_input, sub.exclusion_keywords,
            ):
                continue
            # 到这里就命中了关键词条件
            if sub.semantic is None:
                return sub
            # 需要语义 gate
            args = {a: runtime.get(a, "") for a in sub.semantic.arguments}
            result = self.semantic_fn(
                sub.semantic.function_name, **args,
            )
            semantic_calls.append(f"{sub.semantic.function_name}({user_input!r}) -> {result!r}")
            if self._eval_gate(result, sub.semantic.gate_condition):
                return sub
            # gate 未通过,继续下一 sub
        return None

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        return any(kw in text for kw in keywords)

    @staticmethod
    def _eval_gate(value: str, gate_condition: str) -> bool:
        """支持 '== "x"' 和 '!= "x"' 两种简单形式。"""
        cond = gate_condition.strip()
        if cond.startswith("!="):
            rhs = cond[2:].strip().strip('"')
            return value != rhs
        if cond.startswith("=="):
            rhs = cond[2:].strip().strip('"')
            return value == rhs
        raise ValueError(f"不支持的 gate 条件: {gate_condition}")

    @staticmethod
    def _resolve_agent_expr(expr: str, previous_agent: str) -> str:
        """{{previous_agent}} → 运行时值;"xxx" → 去引号。"""
        e = expr.strip()
        if e.startswith("{{") and e.endswith("}}"):
            var_name = e[2:-2].strip()
            # 目前只识别 previous_agent
            if var_name == "previous_agent":
                return previous_agent
            return ""  # 未知变量兜底
        if e.startswith('"') and e.endswith('"'):
            return e[1:-1]
        return e


@dataclass
class RegressionReport:
    passed: int
    failed: int
    cases: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0


def run_regression(
    spec: AgentSpec,
    semantic_fn: Optional[Callable[..., str]] = None,
) -> RegressionReport:
    """把 spec.examples 当测试用例跑一遍,检查 expected 是否全部被命中。"""
    exe = SpecExecutor(semantic_fn=semantic_fn)
    report = RegressionReport(passed=0, failed=0)

    for idx, case in enumerate(spec.examples, 1):
        result = exe.execute(spec, case.input)
        expected_scene = case.expected.get("scene")
        expected_agent = case.expected.get("agent")
        expected_path = case.execution_path[0] if case.execution_path else None

        ok = (
            result.scene == expected_scene
            and result.agent == expected_agent
            and (expected_path is None or result.matched_block_id == expected_path)
        )
        if ok:
            report.passed += 1
        else:
            report.failed += 1
        report.cases.append({
            "index": idx,
            "input": case.input,
            "expected": case.expected,
            "actual": {"scene": result.scene, "agent": result.agent},
            "expected_path": expected_path,
            "actual_path": result.matched_block_id,
            "ok": ok,
        })
    return report
