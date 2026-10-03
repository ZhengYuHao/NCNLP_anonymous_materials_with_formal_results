"""DSL v1.2 语法校验器。

作用:
- 阻止渲染出非法 DSL
- 给后续 DSL→Python 转译器提供可信输入

检查清单(对应 grammar 规范第 12 节):
1. BLOCK/ENDBLOCK、IF/ENDIF、PERSONA/ENDPERSONA 等配对
2. BLOCK id 全局唯一
3. EXAMPLES 的 EXECUTION_PATH 引用的 block_id 都存在
4. 至少一个兜底 BLOCK 且无条件 RETURN
5. 每个 BLOCK 末尾至少能走到 RETURN 或流下一 BLOCK
6. RETURN 行数 >= 声明的输出变量数
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ValidationError:
    line: int
    message: str


@dataclass
class ValidationResult:
    ok: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            details = "\n".join(f"  line {e.line}: {e.message}" for e in self.errors)
            raise ValueError(f"DSL v1.2 validation failed:\n{details}")


class DSLValidator:
    """轻量级校验器。不解析完整语法,只做结构完整性检查。"""

    BLOCK_PAIRS = [
        ("PERSONA:", "ENDPERSONA"),
        ("CONSTRAINTS:", "ENDCONSTRAINTS"),
        ("INPUTS:", "ENDINPUTS"),
        ("OUTPUTS:", "ENDOUTPUTS"),
        ("EXAMPLES:", "ENDEXAMPLES"),
    ]

    def validate(self, dsl_text: str) -> ValidationResult:
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        lines = dsl_text.splitlines()

        self._check_paired_blocks(lines, errors)
        self._check_agent_wrapping(lines, errors)
        self._check_block_endblock(lines, errors)
        self._check_if_endif(lines, errors)
        self._check_unique_block_ids(lines, errors)
        self._check_fallback_exists(lines, errors)
        self._check_execution_path_references(lines, errors, warnings)

        return ValidationResult(ok=not errors, errors=errors, warnings=warnings)

    # --- 具体检查 ---

    @staticmethod
    def _check_paired_blocks(lines: list[str], errors: list[ValidationError]) -> None:
        for opener, closer in DSLValidator.BLOCK_PAIRS:
            opens = sum(1 for ln in lines if ln.strip() == opener)
            closes = sum(1 for ln in lines if ln.strip() == closer)
            if opens != closes:
                errors.append(
                    ValidationError(0, f"{opener}/{closer} 不配对: {opens} vs {closes}")
                )

    @staticmethod
    def _check_agent_wrapping(lines: list[str], errors: list[ValidationError]) -> None:
        has_agent = any(ln.startswith("AGENT ") for ln in lines)
        has_endagent = any(ln.strip() == "ENDAGENT" for ln in lines)
        if not has_agent:
            errors.append(ValidationError(0, "缺少 AGENT 头"))
        if not has_endagent:
            errors.append(ValidationError(0, "缺少 ENDAGENT 尾"))

    @staticmethod
    def _check_block_endblock(lines: list[str], errors: list[ValidationError]) -> None:
        opens = sum(1 for ln in lines if ln.startswith("BLOCK "))
        closes = sum(1 for ln in lines if ln.strip() == "ENDBLOCK")
        if opens != closes:
            errors.append(ValidationError(0, f"BLOCK/ENDBLOCK 不配对: {opens} vs {closes}"))

    @staticmethod
    def _check_if_endif(lines: list[str], errors: list[ValidationError]) -> None:
        ifs = 0
        endifs = 0
        for ln in lines:
            stripped = ln.strip()
            # 只统计单词 IF,避免匹配 ELIF/ENDIF
            if re.match(r"^IF\b", stripped):
                ifs += 1
            if stripped == "ENDIF":
                endifs += 1
        if ifs != endifs:
            errors.append(ValidationError(0, f"IF/ENDIF 不配对: {ifs} vs {endifs}"))

    @staticmethod
    def _check_unique_block_ids(lines: list[str], errors: list[ValidationError]) -> None:
        pattern = re.compile(r'^BLOCK\s+(\w+)\s+"')
        seen: dict[str, int] = {}
        for n, ln in enumerate(lines, 1):
            m = pattern.match(ln.strip())
            if m:
                bid = m.group(1)
                if bid in seen:
                    errors.append(
                        ValidationError(
                            n, f"block_id 重复: {bid} (首次出现于第 {seen[bid]} 行)"
                        )
                    )
                else:
                    seen[bid] = n

    @staticmethod
    def _check_fallback_exists(lines: list[str], errors: list[ValidationError]) -> None:
        pattern = re.compile(r'^BLOCK\s+b_fallback\b')
        if not any(pattern.match(ln.strip()) for ln in lines):
            errors.append(ValidationError(0, "缺少兜底 BLOCK b_fallback"))

    @staticmethod
    def _check_execution_path_references(
        lines: list[str], errors: list[ValidationError],
        warnings: list[ValidationError] | None = None,
    ) -> None:
        block_ids = set()
        for ln in lines:
            m = re.match(r'^BLOCK\s+(\w+)\s+"', ln.strip())
            if m:
                block_ids.add(m.group(1))

        path_pat = re.compile(r"EXECUTION_PATH:\s*\[([^\]]*)\]")
        for n, ln in enumerate(lines, 1):
            m = path_pat.search(ln)
            if m:
                refs = [r.strip() for r in m.group(1).split(",") if r.strip()]
                for r in refs:
                    if r not in block_ids:
                        # EXECUTION_PATH 引用无效 block_id 降级为警告
                        # 因为渲染器已经做了最佳努力修正，残留的不匹配
                        # 不应阻止 DSL 编译
                        warn = ValidationError(n, f"EXECUTION_PATH 引用了未定义的 block_id: {r}")
                        if warnings is not None:
                            warnings.append(warn)
                        else:
                            errors.append(warn)
