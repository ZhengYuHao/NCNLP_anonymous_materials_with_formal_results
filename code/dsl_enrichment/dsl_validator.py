"""DSL 静态验证器 - 规则驱动的DSL基础验证"""
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional
import re


@dataclass
class ValidationError:
    line: int
    message: str
    code: str = "DSL_ERROR"


@dataclass
class ValidationWarning:
    line: int
    message: str
    code: str = "DSL_WARNING"


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)


class DSLValidator:
    """DSL 静态验证器 - 规则驱动的DSL基础验证"""

    # 内置变量（不需要定义即可使用）
    BUILTIN_VARS = {
        'None', 'True', 'False', 'null', 'true', 'false',
        'query', 'result', 'results', 'error', 'ctx'
    }

    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationWarning] = []

    def validate(self, dsl_code: str) -> ValidationResult:
        """
        验证 DSL 代码的基础正确性

        检测项：
        1. IF/ELIF/ELSE/ENDIF 配对完整性
        2. FOR/ENDFOR 配对完整性
        3. 变量使用前是否定义
        4. 函数调用参数是否完整
        5. RETURN 语句位置是否合理
        """
        self.errors = []
        self.warnings = []

        if not dsl_code or not dsl_code.strip():
            self.errors.append(ValidationError(
                line=0,
                message="DSL 代码为空"
            ))
            return ValidationResult(is_valid=False, errors=self.errors, warnings=self.warnings)

        lines = dsl_code.split('\n')

        # 1. 控制流配对检查（栈式验证）
        self._validate_control_flow_pairing(lines)

        # 2. 变量定义-使用链检查
        self._validate_variable_lineage(lines)

        # 3. CALL 语句格式检查
        self._validate_call_statements(lines)

        # 4. RETURN 语句位置检查
        self._validate_return_statements(lines)

        # 5. DEFINE 语句格式检查
        self._validate_define_statements(lines)

        return ValidationResult(
            is_valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings
        )

    def _validate_control_flow_pairing(self, lines: List[str]):
        """检查 IF/ELIF/ELSE/ENDIF 配对"""
        stack = []  # 栈用于跟踪嵌套的控制流
        line_mapping = []  # 记录每个控制流的行号

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            upper_stripped = stripped.upper()

            # IF 语句
            if upper_stripped.startswith('IF ') or upper_stripped.startswith('ELIF '):
                stack.append(upper_stripped.split()[0])
                line_mapping.append(line_num)

            # ELSE 语句
            elif upper_stripped == 'ELSE' or upper_stripped.startswith('ELSE:'):
                # 检查 ELSE 前是否有未闭合的 ELIF
                if stack and stack[-1] == 'ELIF':
                    stack.pop()
                    line_mapping.pop()
                stack.append('ELSE')
                line_mapping.append(line_num)

            # ENDIF / ENDFOR 等
            elif any(upper_stripped.startswith(end) for end in ('ENDIF', 'ENDFOR', 'ENDWHILE', 'ENDMODULE')):
                end_type = None
                for end in ('ENDIF', 'ENDFOR', 'ENDWHILE', 'ENDMODULE'):
                    if upper_stripped.startswith(end):
                        end_type = end
                        break

                if end_type:
                    expected_open = end_type.replace('END', '')

                    # 尝试匹配 IF 或 FOR
                    if expected_open == 'IF':
                        # IF 可能嵌套 ELIF
                        while stack and stack[-1] in ('ELIF', 'ELSE'):
                            stack.pop()
                            line_mapping.pop()
                        if stack and stack[-1] == 'IF':
                            stack.pop()
                            line_mapping.pop()
                        elif not stack:
                            self.errors.append(ValidationError(
                                line=line_num,
                                message=f"多余的 ENDIF，没有对应的 IF"
                            ))
                    elif expected_open in ('FOR', 'WHILE'):
                        if stack and stack[-1] == expected_open:
                            stack.pop()
                            line_mapping.pop()
                        elif not stack:
                            self.errors.append(ValidationError(
                                line=line_num,
                                message=f"多余的 {end_type}，没有对应的 {expected_open}"
                            ))

        # 检查是否有未闭合的控制流
        if stack:
            for i, control_type in enumerate(stack):
                self.errors.append(ValidationError(
                    line=line_mapping[i],
                    message=f"{control_type} 控制流未闭合"
                ))

    def _validate_variable_lineage(self, lines: List[str]):
        """检查变量定义-使用链"""
        defined_vars: Set[str] = set()  # 已定义的变量
        all_used_vars: Set[Tuple[int, str]] = set()  # 所有使用的变量 (行号, 变量名)
        assigned_vars: Set[Tuple[int, str]] = set()  # 在赋值左侧定义的变量

        # 首先标记所有 DEFINE 定义的变量
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # DEFINE 语句定义变量
            if stripped.upper().startswith('DEFINE'):
                match = re.match(r'DEFINE\s+\{\{(\w+)\}\}', stripped, re.IGNORECASE)
                if match:
                    defined_vars.add(match.group(1))
                    assigned_vars.add((line_num, match.group(1)))

            # OUTPUTS 声明的输出变量
            if stripped.upper().startswith('OUTPUTS:'):
                output_match = re.findall(r'\{\{(\w+)\}\}', stripped)
                defined_vars.update(output_match)

        # 再次遍历检查变量使用
        undefined_usage: List[Tuple[int, str]] = []

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            upper_stripped = stripped.upper()

            # 跳过控制流关键字行
            if upper_stripped.startswith(('IF ', 'ELIF ', 'ELSE', 'ENDIF', 'FOR ', 'ENDFOR', 'DEFINE ', 'MODULE ', 'ENDMODULE', 'CALL ')):
                # IF/ELIF 条件中使用的变量也需要追踪
                if upper_stripped.startswith('IF ') or upper_stripped.startswith('ELIF '):
                    # 条件表达式中的变量
                    condition = stripped.split(':', 1)[1] if ':' in stripped else ''
                    vars_in_condition = re.findall(r'\{\{(\w+)\}\}', condition)
                    for var in vars_in_condition:
                        all_used_vars.add((line_num, var))
                        if var not in defined_vars and var not in self.BUILTIN_VARS:
                            # 可能是后续定义，先记录为可能未定义
                            if (line_num, var) not in assigned_vars:
                                undefined_usage.append((line_num, var))
                continue

            # 跳过注释和空行
            if stripped.startswith('#') or not stripped:
                continue

            # 提取变量引用
            vars_in_line = re.findall(r'\{\{(\w+)\}\}', line)
            all_used_vars.update((line_num, v) for v in vars_in_line)

            # 检查是否在赋值左侧（定义）
            if '=' in line and not upper_stripped.startswith('CALL '):
                # 处理 result = CALL xxx 或 results = CALL xxx 格式
                if 'CALL' in upper_stripped:
                    # CALL 语句的赋值
                    left_side = line.split('=')[0].strip()
                    defined_in_call = re.findall(r'\{\{(\w+)\}\}', left_side)
                    for var in defined_in_call:
                        defined_vars.add(var)
                        assigned_vars.add((line_num, var))
                else:
                    # 普通赋值语句
                    left_side = line.split('=')[0].strip()
                    # 处理 ctx["var"] = 格式
                    if 'ctx["' in left_side or "ctx['" in left_side:
                        match = re.search(r'ctx\[["\'](\w+)["\']\]', left_side)
                        if match:
                            defined_vars.add(match.group(1))
                            assigned_vars.add((line_num, match.group(1)))
                    else:
                        defined = re.findall(r'\{\{(\w+)\}\}', left_side)
                        for var in defined:
                            defined_vars.add(var)
                            assigned_vars.add((line_num, var))

            # 检查右侧是否使用了未定义的变量
            if '=' in line:
                right_side = line.split('=', 1)[1]

                # 跳过 CALL 语句的右侧（CALL 函数调用不存在未定义变量问题）
                if 'CALL' in right_side.upper():
                    continue

                used_in_right = re.findall(r'\{\{(\w+)\}\}', right_side)
                for var in used_in_right:
                    if var not in defined_vars and var not in self.BUILTIN_VARS:
                        # 检查是否是 INPUTS 中声明的
                        is_input = False
                        for iline in range(line_num - 1, -1, -1):
                            if lines[iline].strip().upper().startswith('INPUTS:'):
                                if f'{{{{{var}}}}}' in lines[iline]:
                                    is_input = True
                                    break
                            if lines[iline].strip().upper().startswith('MODULE '):
                                break

                        if not is_input:
                            undefined_usage.append((line_num, var))

        # 报告未定义的变量（作为警告，因为可能是后续分支定义）
        for line_num, var in undefined_usage:
            self.warnings.append(ValidationWarning(
                line=line_num,
                message=f"变量 {{{{{var}}}}} 使用前可能未定义"
            ))

    def _validate_call_statements(self, lines: List[str]):
        """检查 CALL 语句格式"""
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            if 'CALL' in stripped.upper():
                # 检查括号是否配对
                open_count = stripped.count('(')
                close_count = stripped.count(')')
                if open_count != close_count:
                    self.errors.append(ValidationError(
                        line=line_num,
                        message=f"CALL 语句括号不配对: {stripped[:50]}..."
                    ))

                # 检查是否有 CALL 但无函数名
                if re.match(r'^\s*CALL\s*$', stripped, re.IGNORECASE):
                    self.errors.append(ValidationError(
                        line=line_num,
                        message="CALL 语句缺少函数名"
                    ))

                # 检查 CALL 后面是否跟着函数调用
                call_match = re.search(r'CALL\s+(\w+)\s*\(', stripped, re.IGNORECASE)
                if not call_match and 'CALL' in stripped.upper():
                    # 可能是赋值形式的 CALL: {{result}} = CALL func()
                    call_match = re.search(r'CALL\s+(\w+)\s*\(', stripped, re.IGNORECASE)

    def _validate_return_statements(self, lines: List[str]):
        """检查 RETURN 语句位置"""
        in_module = False
        in_branch = False  # 是否在某个分支内
        has_return = False
        module_name = None

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            upper_stripped = stripped.upper()

            if upper_stripped.startswith('MODULE '):
                in_module = True
                # 提取模块名
                module_match = re.match(r'MODULE\s+(\w+)', stripped, re.IGNORECASE)
                if module_match:
                    module_name = module_match.group(1)

            elif upper_stripped.startswith('ENDMODULE'):
                in_module = False
                module_name = None

            elif upper_stripped.startswith('IF ') or upper_stripped.startswith('ELIF '):
                in_branch = True

            elif upper_stripped.startswith('ENDIF') or upper_stripped.startswith('ENDFOR'):
                in_branch = False

            elif upper_stripped.startswith('RETURN'):
                has_return = True
                # RETURN 不在模块级别也不是分支级别，可能是警告
                if in_module and not in_branch:
                    # 检查是否有实际返回值
                    return_part = upper_stripped[6:].strip()
                    if not return_part:
                        self.warnings.append(ValidationWarning(
                            line=line_num,
                            message="RETURN 语句没有返回值"
                        ))

        # 如果模块没有 RETURN 语句，给出警告
        if in_module and not has_return and module_name:
            # 检查是否是 IF 分支内部有 RETURN
            pass  # 这需要在更细粒度的控制流分析中处理

    def _validate_define_statements(self, lines: List[str]):
        """检查 DEFINE 语句格式"""
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            if stripped.upper().startswith('DEFINE'):
                # DEFINE 应该有完整的格式: DEFINE {{var}}: Type = value
                if not re.search(r'DEFINE\s+\{\{\w+\}\}\s*:\s*\w+\s*=', stripped, re.IGNORECASE):
                    # 检查是否是简单的 DEFINE（无赋值）
                    if re.search(r'DEFINE\s+\{\{\w+\}\}\s*:\s*\w+$', stripped, re.IGNORECASE):
                        self.warnings.append(ValidationWarning(
                            line=line_num,
                            message=f"DEFINE 语句可能缺少赋值: {stripped[:50]}"
                        ))
                    else:
                        self.errors.append(ValidationError(
                            line=line_num,
                            message=f"DEFINE 语句格式不正确: {stripped[:50]}"
                        ))