"""
ANTI_HOLLOW lint — BLOCK 内 CALL sb_xxx 输出空心化检查。

规则：对 BLOCK 内每个 CALL sb_xxx 的输出变量 V：
- V 必须通过 V.field 的形式在后续 IF 中被分流
- IF V ==/!= literal 算空心化（值比较而非字段比较）
- RETURN V 算空心化
- 递归扫描所有 IF/ELIF/ELSE 嵌套层级

与 files/validator.py 的关系：
- 这是一条独立的 lint 规则
- 集成到管线时在 DSLValidator.validate() 中调用
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class HollowViolation:
    """一个空心化违规记录。"""
    block_id: str
    line: int
    variable: str
    violation_type: str  # "hollow_call" | "direct_return" | "value_compare"
    description: str
    suggestion: str


def check_anti_hollow(dsl_code: str) -> list[HollowViolation]:
    """对 DSL 代码执行 ANTI_HOLLOW lint 检查。

    Args:
        dsl_code: DSL 1.2 代码文本

    Returns:
        发现的违规列表。空列表表示无违规。
    """
    if not dsl_code or not dsl_code.strip():
        return []

    lines = dsl_code.split("\n")
    violations: list[HollowViolation] = []

    # 1. 提取所有 BLOCK
    blocks = _extract_blocks(lines)
    if not blocks:
        return []

    # 2. 对每个 BLOCK 执行空心化检查
    for block_start, block_end, block_id in blocks:
        block_violations = _check_block(lines, block_start, block_end, block_id)
        violations.extend(block_violations)

    return violations


# ============================================================================
# BLOCK 提取
# ============================================================================


def _extract_blocks(lines: list[str]) -> list[tuple[int, int, str]]:
    """从行列表中提取 BLOCK 边界。

    Returns:
        [(start_line, end_line, block_id), ...]  所有行号是 0-based
    """
    blocks = []
    in_block = False
    block_start = 0
    block_id = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        m = re.match(r'^BLOCK\s+(\w+)\s+"', stripped)
        if m:
            in_block = True
            block_start = i
            block_id = m.group(1)

        if in_block and stripped == "ENDBLOCK":
            blocks.append((block_start, i, block_id))
            in_block = False

    return blocks


# ============================================================================
# BLOCK 级空心化检查
# ============================================================================


def _check_block(
    lines: list[str],
    start: int,
    end: int,
    block_id: str,
) -> list[HollowViolation]:
    """检查单个 BLOCK 的空心化违规。"""
    violations: list[HollowViolation] = []
    block_lines = lines[start:end+1]

    # 找到所有 CALL sb_xxx 的赋值变量
    call_vars: dict[str, int] = {}
    for i, line in enumerate(block_lines):
        m = re.match(r'\{\{(\w+)\}\}\s*=\s*CALL\s+sb_\w+', line.strip())
        if m:
            call_vars[m.group(1)] = start + i

    if not call_vars:
        return []

    # 对每个 CALL 输出变量，扫描后续行
    for var_name, call_line in call_vars.items():
        violation = _trace_variable(
            lines, start, end, call_line, var_name, block_id
        )
        if violation:
            violations.append(violation)

    return violations


def _trace_variable(
    lines: list[str],
    block_start: int,
    block_end: int,
    call_line: int,
    var_name: str,
    block_id: str,
) -> HollowViolation | None:
    """追踪一个 CALL 输出变量在 BLOCK 内的使用路径。

    逐分支独立检测：
    1. 找到 BLOCK 内引用 V 的所有 IF/ELIF 条件行
    2. 对每个条件行独立检查: .field 访问 vs 值比较
    3. 如果存在引用 V 且只用值比较/直接 RETURN 的分支 → VIOLATION
    """
    # 收集所有引用 var_name 的 IF/ELIF 条件行
    var_refs: list[dict] = []  # [{line, kind: "field"|"value"|"return"}, ...]

    for i in range(call_line + 1, block_end + 1):
        stripped = lines[i].strip()

        # 检测 IF/ELIF 条件中引用 V
        if re.match(r'(IF|ELIF)\s+', stripped):
            has_field = bool(re.search(
                rf'\{{{{\s*{var_name}\s*}}\}}\.\w+', stripped
            ))
            has_value = bool(re.search(
                rf'\{{{{\s*{var_name}\s*}}\}}\s*(!=|==)\s*"', stripped
            ))
            if has_field or has_value:
                var_refs.append({
                    "line": i,
                    "kind": "field" if has_field else "value",
                    "stripped": stripped,
                })

        # 检测 RETURN 中引用 V
        if re.search(rf'RETURN\s+.*\{{{{\s*{re.escape(var_name)}\s*}}}}', stripped):
            # 检查这个 RETURN 是否在 IF 分支内
            # 如果是，且该分支没有 .field 访问 → violation
            var_refs.append({
                "line": i,
                "kind": "return",
                "stripped": stripped,
            })

    if not var_refs:
        return None  # V 从未被使用（可能只是被赋值后没用到）

    # 检查所有引用：如果有任一条是 value 或 return 类型，且没有其他 field 访问
    has_field = any(r["kind"] == "field" for r in var_refs)
    has_value_or_return = any(r["kind"] in ("value", "return") for r in var_refs)

    if has_field and not has_value_or_return:
        return None  # 全部通过 .field 分流

    if has_value_or_return:
        return HollowViolation(
            block_id=block_id,
            line=call_line,
            variable=var_name,
            violation_type="value_compare",
            description=(
                f"SEMANTIC_BLOCK 输出 {{{{{var_name}}}}} 未经 .field 分流"
                f"（存在值比较或直接 RETURN），违反 §9"
            ),
            suggestion=(
                f"改写 SEMANTIC_BLOCK 输出为结构化对象，"
                f"用 {{{{{var_name}}}}}.field 分流"
            ),
        )

    return None
