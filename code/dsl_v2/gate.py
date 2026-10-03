"""
ExamplesGate — 执行验证门禁。

职责：
1. 在 DSL 编译期对所有 EXAMPLES 进行模拟执行
2. 验证每个 case 的 EXECUTION_PATH 是否匹配
3. 收集失败的 BLOCK ID 供 Upgrader 升级
4. 无 EXAMPLES 时自动跳过

与 Upgrader 的关系：
- Gate 只负责"发现问题"，不负责"解决问题"
- 升级逻辑在 upgrader.py 中
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .executor import DSLExecutor, CaseResult


@dataclass
class GateResult:
    """ExamplesGate 的执行结果。"""
    all_passed: bool = True
    failed_blocks: set[str] = field(default_factory=set)
    case_results: list[CaseResult] = field(default_factory=list)
    skipped: bool = False
    message: str = ""


class ExamplesGate:
    """执行验证门禁。"""

    def __init__(self):
        self.executor = DSLExecutor()

    def run(
        self,
        dsl_code: str,
        examples: list[dict],
    ) -> GateResult:
        """对 DSL 代码执行 EXAMPLES 验证。

        Args:
            dsl_code: 待验证的 DSL 1.2 代码
            examples: 示例列表，每项含 input, expected, execution_path

        Returns:
            GateResult 包含验证结果和失败的 BLOCK ID
        """
        if not examples:
            return GateResult(all_passed=True, skipped=True, message="无 EXAMPLES，跳过门禁")

        exec_results = self.executor.run_examples(dsl_code, examples)

        gate_result = GateResult(
            all_passed=exec_results.all_passed,
            failed_blocks=exec_results.failed_blocks,
            case_results=exec_results.results,
        )

        if gate_result.all_passed:
            gate_result.message = f"全部 {len(examples)} 个 EXAMPLES 通过"
        else:
            failed_count = len([r for r in exec_results.results if not r.passed])
            gate_result.message = (
                f"{failed_count}/{len(examples)} 个 EXAMPLES 失败: "
                f"涉及 BLOCK {gate_result.failed_blocks}"
            )

        return gate_result
