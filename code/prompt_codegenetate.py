#代码实现
"""
WaAct S.E.D.E Framework - Step 4: 依赖分析、模块分解与代码生成
功能：将 Prompt 3.0 DSL 编译为可执行的模块化 Python 代码
作者：WaAct Compiler Team
版本：2.0
"""

import re
import json
import os
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from logger import info, warning, error, debug
from dsl_v2.converter.semantic_converter import SemanticBlockConverter

def _check_and_get_impl(func_name):
    """运行时检查并获取函数实现"""
    try:
        from dsl_library import has_implementation, render_implementation
        if has_implementation(func_name):
            return render_implementation(func_name, {})
    except ImportError:
        pass
    return None


# ============================================================================
# 数据结构定义
# ============================================================================

class BlockType(Enum):
    """代码块类型枚举"""
    ASSIGN = "ASSIGN"          # 普通赋值
    CALL = "CALL"              # LLM 调用
    IF = "IF"                  # 条件判断
    ELIF = "ELIF"              # 否则如果分支
    ELSE = "ELSE"              # 否则分支
    ENDIF = "ENDIF"            # 条件结束
    FOR = "FOR"                # FOR循环
    ENDFOR = "ENDFOR"          # FOR循环结束
    WHILE = "WHILE"            # 循环（扩展）
    ENDWHILE = "ENDWHILE"      # 循环结束


@dataclass
class CodeBlock:
    """原子代码块"""
    id: str
    type: BlockType
    code_lines: List[str]
    inputs: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    line_number: int = 0
    is_async: bool = False  # 是否包含异步操作（LLM 调用）
    
    def __repr__(self):
        return f"Block({self.id}, {self.type.value}, ins={self.inputs}, outs={self.outputs})"


@dataclass
class ModuleDefinition:
    """模块定义（编译后的函数）"""
    name: str
    inputs: List[str]
    outputs: List[str]
    body_code: str
    is_async: bool = False
    original_blocks: List[CodeBlock] = field(default_factory=list)
    
    def to_python(self) -> str:
        """生成完整的 Python 函数代码"""
        async_prefix = "async " if self.is_async else ""
        return self.body_code


@dataclass
class AgentSpec:
    """DSL v1.2 Agent 规格（解析后的顶层结构）"""
    agent_name: str = ""
    agent_description: str = ""
    persona: Dict[str, str] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    inputs: List[Dict[str, str]] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    semantic_blocks: List[Dict[str, Any]] = field(default_factory=list)
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    configs: Dict[str, str] = field(default_factory=dict)
    policies: List[Dict[str, str]] = field(default_factory=list)


class DSL12Parser:
    """
    DSL v1.2 解析器 - 解析 DSL v1.2 顶层结构
    提取 AGENT、PERSONA、BLOCK 等顶层语法单元
    """

    KEYWORD_BLOCKS = {
        'PERSONA:', 'CONSTRAINTS:', 'INPUTS:', 'OUTPUTS:', 'EXAMPLES:',
        'CONFIG:', 'POLICY:',
        'ENDPERSONA', 'ENDCONSTRAINTS', 'ENDINPUTS', 'ENDOUTPUTS', 'ENDEXAMPLES',
        'ENDCONFIG', 'ENDPOLICY'
    }

    def __init__(self):
        self.spec = AgentSpec()
        self._configs: Dict[str, str] = {}
        self._policies: List[Dict[str, str]] = []

    def parse(self, dsl_code: str) -> AgentSpec:
        """解析完整的 DSL v1.2 代码"""
        lines = dsl_code.split('\n')
        self.spec = AgentSpec()
        self._configs = {}
        self._policies = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith('AGENT '):
                self._parse_agent_header(line)
                i += 1
                continue

            if line == 'PERSONA:':
                i = self._parse_persona(lines, i)
                continue

            if line == 'CONSTRAINTS:':
                i = self._parse_constraints(lines, i)
                continue

            if line == 'CONFIG:':
                i = self._parse_config(lines, i)
                continue

            if line == 'POLICY:':
                i = self._parse_policy(lines, i)
                continue

            if line == 'INPUTS:':
                i = self._parse_inputs(lines, i)
                continue

            if line == 'OUTPUTS:':
                i = self._parse_outputs(lines, i)
                continue

            if line.startswith('BLOCK '):
                block_data, i = self._parse_block(lines, i)
                self.spec.blocks.append(block_data)
                continue

            if line == 'EXAMPLES:':
                i = self._parse_examples(lines, i)
                continue

            if line.startswith('SEMANTIC_BLOCK '):
                new_i, semantic_data = self._parse_semantic_block(lines, i)
                i = new_i
                self.spec.semantic_blocks.append(semantic_data)
                continue

            i += 1

        # 将 CONFIG/POLICY 存入 spec（路径C：供 generate_full_code 使用）
        self.spec.configs = self._configs
        self.spec.policies = self._policies

        return self.spec

    def _parse_agent_header(self, line: str):
        """解析 AGENT 头部：AGENT {{Name}} description"""
        match = re.match(r'AGENT\s+\{\{(\w+)\}\}\s+"([^"]*)"', line)
        if match:
            self.spec.agent_name = match.group(1)
            self.spec.agent_description = match.group(2)

    def _parse_persona(self, lines: List[str], start: int) -> int:
        """解析 PERSONA 块"""
        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == 'ENDPERSONA':
                return i + 1
            if line.startswith('ROLE:'):
                self.spec.persona['role'] = line[5:].strip()
            elif ':' in line:
                key, val = line.split(':', 1)
                self.spec.persona[key.strip()] = val.strip()
            i += 1
        return i

    def _parse_constraints(self, lines: List[str], start: int) -> int:
        """解析 CONSTRAINTS 块"""
        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == 'ENDCONSTRAINTS':
                return i + 1
            if line and ':' in line:
                self.spec.constraints.append(line)
            i += 1
        return i

    def _parse_config(self, lines: List[str], start: int) -> int:
        """解析 CONFIG 块"""
        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == 'ENDCONFIG':
                return i + 1
            if line and ':' in line:
                key, _, value = line.partition(':')
                self._configs[key.strip()] = value.strip()
            i += 1
        return i

    def _parse_policy(self, lines: List[str], start: int) -> int:
        """解析 POLICY 块"""
        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == 'ENDPOLICY':
                return i + 1
            if line and ':' in line:
                key, _, desc = line.partition(':')
                self._policies.append({
                    'key': key.strip(),
                    'description': desc.strip(),
                    'trigger_condition': '',
                    'action': ''
                })
            i += 1
        return i

    def _parse_inputs(self, lines: List[str], start: int) -> int:
        """解析 INPUTS 块"""
        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == 'ENDINPUTS':
                return i + 1
            if 'REQUIRED' in line or 'OPTIONAL' in line or '{{' in line:
                self.spec.inputs.append(line)
            i += 1
        return i

    def _parse_outputs(self, lines: List[str], start: int) -> int:
        """解析 OUTPUTS 块"""
        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == 'ENDOUTPUTS':
                return i + 1
            if '{{' in line:
                self.spec.outputs.append(line)
            i += 1
        return i

    def _parse_block(self, lines: List[str], start: int) -> Tuple[Dict[str, Any], int]:
        """解析 BLOCK 块，返回 (block_data, next_index)"""
        block_line = lines[start].strip()
        match = re.match(r'BLOCK\s+(\w+)\s+"([^"]*)"', block_line)
        if not match:
            return {'id': '', 'description': '', 'statements': []}, start + 1

        block_id = match.group(1)
        block_desc = match.group(2)
        statements = []
        i = start + 1

        while i < len(lines):
            line = lines[i].strip()
            if line == 'ENDBLOCK':
                return {
                    'id': block_id,
                    'description': block_desc,
                    'statements': statements
                }, i + 1
            if line.startswith('BLOCK ') or re.match(r'^BLOCK\s+\w+', line):
                warning(f"⚠️ BLOCK {block_id} 缺少 ENDBLOCK，提前结束")
                return {
                    'id': block_id,
                    'description': block_desc,
                    'statements': statements
                }, i
            if self._is_valid_dsl_statement(line):
                statements.append(line)
            i += 1

        return {
            'id': block_id,
            'description': block_desc,
            'statements': statements
        }, i

    def _is_valid_dsl_statement(self, line: str) -> bool:
        """检查是否为有效的 DSL 语句（过滤 Python 异常代码和 DSL 标记）"""
        line = line.strip()
        if not line:
            return False
        if line.startswith('self.') or line.startswith('class '):
            return False
        if ':' in line and 'Optional' in line:
            return False
        if line.startswith('return ') or line.startswith('def '):
            return False
        if '=' in line and line.split('=')[0].strip().startswith('self'):
            return False
        if line in ('ENDBLOCK', 'BLOCK', 'ENDAGENT', 'ENDPERSONA', 'ENDCONSTRAINTS',
                    'ENDINPUTS', 'ENDOUTPUTS', 'ENDEXAMPLES', 'ENDCASE',
                    'EXECUTION_PATH', 'INPUT:', 'EXPECTED:'):
            return False
        if line.startswith('BLOCK ') or re.match(r'^BLOCK\s+\w+', line):
            return False
        if line.startswith('ENDBLOCK') or line.startswith('ENDAGENT') or line.startswith('ENDPERSONA'):
            return False
        if line.startswith('ENDCONSTRAINTS') or line.startswith('ENDINPUTS') or line.startswith('ENDOUTPUTS'):
            return False
        if line.startswith('ENDEXAMPLES') or line.startswith('ENDCASE'):
            return False
        if line.startswith('ENDSEMANTIC_BLOCK'):
            return False
        return True

    def _parse_semantic_block(self, lines: List[str], start: int) -> Tuple[int, Dict[str, Any]]:
        """解析 SEMANTIC_BLOCK 块"""
        semantic_data = {
            'block_id': '',
            'description': '',
            'name': '',
            'model': 'gpt-4o',
            'temperature': 0.1,
            'max_tokens': 2048,
            'task': 'classification',
            'input_vars': [],
            'output_vars': [],
            'prompt': ''
        }

        header_line = lines[start].strip()
        header_match = re.match(r'SEMANTIC_BLOCK\s+(\w+)(?:\s+"([^"]+)")?', header_line)
        if header_match:
            semantic_data['block_id'] = header_match.group(1)
            semantic_data['description'] = header_match.group(2) or ''
            semantic_data['name'] = semantic_data['block_id']

        i = start + 1
        in_input = False
        in_output = False
        in_prompt = False
        prompt_lines = []

        while i < len(lines):
            line = lines[i].strip()

            if line == 'ENDSEMANTIC_BLOCK':
                if prompt_lines:
                    semantic_data['prompt'] = ' '.join(prompt_lines)
                return i + 1, semantic_data

            if line.startswith('NAME '):
                semantic_data['name'] = line[5:].strip().strip('"')

            elif line.startswith('MODEL '):
                semantic_data['model'] = line[6:].strip()

            elif line.startswith('TASK '):
                semantic_data['task'] = line[5:].strip()

            elif line.startswith('TEMPERATURE '):
                try:
                    semantic_data['temperature'] = float(line[12:].strip())
                except ValueError:
                    pass

            elif line.startswith('MAX_TOKENS '):
                try:
                    semantic_data['max_tokens'] = int(line[11:].strip())
                except ValueError:
                    pass

            elif line == 'INPUT:':
                in_input = True
                in_output = False
                in_prompt = False

            elif line == 'OUTPUT:':
                in_input = False
                in_output = True
                in_prompt = False

            elif line.startswith('PROMPT'):
                in_input = False
                in_output = False
                in_prompt = True
                prompt_content = line[6:].strip()
                if prompt_content.startswith('"'):
                    semantic_data['prompt'] = prompt_content.strip('"')
                else:
                    prompt_lines = [prompt_content]

            elif in_input and ':' in line:
                var_match = re.match(r'(\w+):\s*(\w+)', line)
                if var_match:
                    semantic_data['input_vars'].append({
                        'name': var_match.group(1),
                        'type': var_match.group(2)
                    })

            elif in_output and ':' in line:
                var_match = re.match(r'(\w+):\s*(\w+)', line)
                if var_match:
                    semantic_data['output_vars'].append({
                        'name': var_match.group(1),
                        'type': var_match.group(2)
                    })

            elif in_prompt and line and not line.startswith('ENDSEMANTIC'):
                if line.startswith('"""'):
                    pass
                elif line.endswith('"""'):
                    semantic_data['prompt'] = ' '.join(prompt_lines)
                    in_prompt = False
                else:
                    prompt_lines.append(line)

            i += 1

        return i, semantic_data

    def _parse_examples(self, lines: List[str], start: int) -> int:
        """解析 EXAMPLES 块"""
        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == 'ENDEXAMPLES':
                return i + 1
            self.spec.examples.append(line)
            i += 1
        return i


class DSL12ToIntermediateConverter:
    """
    DSL v1.2 → 中间格式转换器
    将 DSL v1.2 结构转换为 WaActCompiler 能处理的旧 DSL 格式
    """

    def __init__(self):
        self.parser = DSL12Parser()

    def convert(self, dsl_code: str) -> Tuple[str, List[Dict[str, Any]]]:
        """执行完整的 DSL v1.2 → 中间格式转换

        Returns:
            (转换后的 DSL 代码, semantic_blocks 列表)
        """
        spec = self.parser.parse(dsl_code)
        lines = []

        lines.append(f"# Agent: {spec.agent_name} - {spec.agent_description}")
        lines.append("")

        if spec.persona:
            role = spec.persona.get('role', '')
            lines.append(f"# Persona: {role}")
            lines.append("")

        if spec.inputs:
            lines.append("# Inputs:")
            for inp in spec.inputs:
                lines.append(f"#   {inp}")
            lines.append("")

        if spec.outputs:
            lines.append(f"# Outputs: {', '.join(spec.outputs)}")
            lines.append("")

        for block in spec.blocks:
            for stmt in block['statements']:
                lines.append(f"    {self._convert_elif_contains(stmt)}")
            lines.append("")

        return '\n'.join(lines), spec.semantic_blocks

    def _convert_elif_contains(self, stmt: str) -> str:
        """将 ELIF CONTAINS 格式转换为 IF CONTAINS 格式"""
        stmt = stmt.strip()
        match = re.match(r'^ELIF\s+(\S+)\s+CONTAINS\s+\[([^\]]+)\]$', stmt, re.IGNORECASE)
        if match:
            var = match.group(1).strip().strip('{}')
            items = [item.strip().strip('"\'') for item in match.group(2).split(',')]
            quoted_items = ", ".join('"' + item + '"' for item in items)
            return f"IF {var} CONTAINS [{quoted_items}]"
        return stmt

    def is_dsl_v12(self, dsl_code: str) -> bool:
        """检测是否为 DSL v1.2 格式"""
        dsl_code = dsl_code.strip()
        return (
            dsl_code.startswith('AGENT ') or
            'PERSONA:' in dsl_code or
            'BLOCK ' in dsl_code and 'ENDBLOCK' in dsl_code or
            'CONSTRAINTS:' in dsl_code or
            'ENDAGENT' in dsl_code
        )


# ============================================================================
# 0. 三阶段代码修复系统
# ============================================================================

class DSLNormalizer:
    """阶段1: DSL 预处理器 - 规范化语法"""
    
    PLACEHOLDER_PATTERNS = {
        r'_\d+': lambda m: m.group(0)[1:],  # _30 -> 30
        r'_placeholder_': 'None',
        r'__(\w+)__': r'\1',  # __Python__ -> Python
    }
    
    def normalize(self, dsl_code: str) -> str:
        """规范化 DSL 代码"""
        info("[Phase 1] DSL 语法规范化...")

        converter = DSL12ToIntermediateConverter()
        if converter.is_dsl_v12(dsl_code):
            info("  📋 检测到 DSL v1.2 格式，启动语法转换器...")
            dsl_code, _ = converter.convert(dsl_code)
            info("  ✅ DSL v1.2 → 中间格式转换完成")

        original_lines = dsl_code.split('\n')
        normalized_lines = []

        for line in original_lines:
            line = self._fix_placeholders(line)
            line = self._fix_variable_names(line)
            normalized_lines.append(line)

        normalized = '\n'.join(normalized_lines)
        info(f"  ✅ 占位符修复: {dsl_code.count('_') - normalized.count('_')} 处")
        return normalized
    
    def _fix_placeholders(self, line: str) -> str:
        """修复占位符"""
        # 使用负向前后查找确保只匹配独立的占位符如 _30，而不是 count_1 中的 _1
        # (?<![a-zA-Z0-9_])_(\d+)(?![a-zA-Z0-9_]) 匹配前后都不是字母数字下划线的 _数字
        line = re.sub(r'(?<![a-zA-Z0-9_])_(\d+)(?![a-zA-Z0-9_])', r'\1', line)  # _30 -> 30
        line = re.sub(r'_placeholder_', 'None', line)
        line = re.sub(r'__Python__', 'Python', line)
        line = re.sub(r'__Java__', 'Java', line)
        return line
    
    def _fix_variable_names(self, line: str) -> str:
        """修复变量命名（下划线开头/结尾）"""
        def replace_var(match):
            var = match.group(1)
            if var.startswith('_') or var.endswith('_'):
                return match.group(0)
            return match.group(0)
        
        line = re.sub(r'\{\{(\w+)\}\}', replace_var, line)
        return line


class VariableLineageAnalyzer:
    """阶段2: 变量血缘分析器 - 追踪变量从定义到使用的全过程"""
    
    def __init__(self, blocks: List['CodeBlock']):
        self.blocks = blocks
        self.graph = nx.DiGraph()
        self.var_definitions = defaultdict(list)  # 变量定义位置
        self.var_usages = defaultdict(list)       # 变量使用位置
    
    def analyze(self) -> Dict[str, Any]:
        """执行变量血缘分析"""
        info("[Phase 2] 变量血缘分析...")
        
        self._build_def_usage_map()
        lineage = self._trace_lineage()
        
        info(f"  ✅ 追踪到 {len(lineage['variables'])} 个变量")
        return lineage
    
    def _build_def_usage_map(self):
        """构建变量定义-使用映射"""
        for block in self.blocks:
            for var in block.outputs:
                self.var_definitions[var].append(block.id)
            for var in block.inputs:
                self.var_usages[var].append(block.id)
    
    def _trace_lineage(self) -> Dict[str, Any]:
        """追踪变量血缘"""
        all_vars = set(self.var_definitions.keys()) | set(self.var_usages.keys())
        
        lineage = {
            'variables': {},
            'missing_inputs': [],
            'missing_outputs': []
        }
        
        for var in all_vars:
            defined_in = self.var_definitions.get(var, [])
            used_in = self.var_usages.get(var, [])
            
            lineage['variables'][var] = {
                'defined_in': defined_in,
                'used_in': used_in,
                'is_external': len(defined_in) == 0 and len(used_in) > 0
            }
            
            if len(defined_in) == 0 and len(used_in) > 0:
                lineage['missing_inputs'].append(var)
        
        for block in self.blocks:
            for var in block.inputs:
                if var not in self.var_definitions:
                    lineage['missing_inputs'].append(var)
        
        lineage['missing_inputs'] = list(set(lineage['missing_inputs']))
        
        if lineage['missing_inputs']:
            info(f"  ⚠️ 发现 {len(lineage['missing_inputs'])} 个未声明的输入变量")
        
        return lineage
    
    def get_block_inputs(self, block_id: str) -> Set[str]:
        """获取模块的实际输入（基于变量使用）"""
        for block in self.blocks:
            if block.id == block_id:
                used_vars = set()
                for line in block.code_lines:
                    vars_in_line = re.findall(r'\{\{(\w+)\}\}', line)
                    used_vars.update(vars_in_line)
                return used_vars - block.outputs
        return set()
    
    def get_block_outputs(self, block_id: str) -> Set[str]:
        """获取模块的实际输出（基于变量定义）"""
        for block in self.blocks:
            if block.id == block_id:
                defined_vars = set()
                for line in block.code_lines:
                    assign_match = re.findall(r'\{\{(\w+)\}\}\s*=', line)
                    defined_vars.update(assign_match)
                return defined_vars
        return set()


class PostProcessor:
    """阶段3: 后处理器 - AST 语法验证和兜底修复"""
    
    def __init__(self, modules: List['ModuleDefinition']):
        self.modules = modules
        self.errors = []
        self.fixes_applied = []
    
    def process(self, file_path: str = None) -> bool:
        """执行后处理验证和修复"""
        info("[Phase 3] 后处理验证...")
        
        import ast
        
        all_code = '\n\n'.join([m.body_code for m in self.modules])
        
        try:
            ast.parse(all_code)
            info("  ✅ AST 语法验证通过")
            return True
        except SyntaxError as e:
            info(f"  ⚠️ 发现语法错误: {e.msg} (行 {e.lineno})")
            self.errors.append(str(e))
            return self._apply_fixes(file_path)
    
    def _apply_fixes(self, file_path: str = None) -> bool:
        """应用自动修复"""
        info("  🔧 尝试自动修复...")
        
        fixed = False
        
        for module in self.modules:
            original = module.body_code
            
            module.body_code = self._fix_undefined_vars(module.body_code)
            module.body_code = self._fix_type_errors(module.body_code)
            
            if module.body_code != original:
                self.fixes_applied.append(module.name)
                fixed = True
        
        if fixed:
            info(f"  ✅ 自动修复了 {len(self.fixes_applied)} 个模块")
        
        return fixed
    
    def _fix_undefined_vars(self, code: str) -> str:
        """修复未定义变量引用"""
        common_undefined = ['count_1', 'cache_key', 'current_results', 
                           'feedback', 'negative_feedback_count', 'queue_status']
        
        for var in common_undefined:
            pattern = r'\b' + var + r'\b'
            if re.search(pattern, code) and f'def {var}' not in code:
                code = re.sub(r'\{\{' + var + r'\}\}', var, code)
        
        return code
    
    def _fix_type_errors(self, code: str) -> str:
        """修复类型错误"""
        code = re.sub(r'\.length\b', 'len()', code)
        code = re.sub(r'\.results\.length', 'len(results)', code)
        
        return code


class WorkflowFixer:
    """工作流修复器 - 整合三个阶段"""
    
    def __init__(self):
        self.normalizer = DSLNormalizer()
        self.lineage_analyzer = None
        self.post_processor = None

    @staticmethod
    def _extend_unique(values: List[str], additions: Set[str]) -> List[str]:
        """Append newly discovered variables without changing signature order."""
        merged = list(values)
        merged.extend(sorted(value for value in additions if value not in merged))
        return merged
    
    def fix(self, dsl_code: str, blocks: List[CodeBlock], 
            modules: List['ModuleDefinition']) -> Tuple[str, List['ModuleDefinition']]:
        """执行完整的三阶段修复"""
        
        dsl_code = self.normalizer.normalize(dsl_code)
        
        self.lineage_analyzer = VariableLineageAnalyzer(blocks)
        lineage = self.lineage_analyzer.analyze()
        
        for module in modules:
            actual_inputs = self.lineage_analyzer.get_block_inputs(module.name)
            actual_outputs = self.lineage_analyzer.get_block_outputs(module.name)
            
            module.inputs = self._extend_unique(module.inputs, actual_inputs)
            module.outputs = self._extend_unique(module.outputs, actual_outputs)
        
        self.post_processor = PostProcessor(modules)
        self.post_processor.process()
        
        return dsl_code, modules


# ============================================================================
# 1. 增强型伪代码解析器
# ============================================================================

class PseudoCodeParser:
    """DSL 解析器 - 支持完整的控制流语法"""
    
    def __init__(self):
        # 变量模式：{{variable_name}}
        self.var_pattern = re.compile(r'\{\{(\w+)\}\}')
        # CALL 模式：CALL function_name(args)
        self.call_pattern = re.compile(r'CALL\s+(\w+)\s*\((.*?)\)')
        
    def _extract_vars(self, line: str) -> Set[str]:
        """提取行中的所有变量"""
        return set(self.var_pattern.findall(line))
    
    def _classify_block_type(self, line: str) -> BlockType:
        """识别代码块类型"""
        line_upper = line.upper().strip()
        if line_upper.startswith("IF "):
            return BlockType.IF
        elif line_upper.startswith("ELIF "):
            return BlockType.ELIF
        elif line_upper == "ELSE" or line_upper == "ELSE:":
            return BlockType.ELSE
        elif line_upper == "ENDIF":
            return BlockType.ENDIF
        elif line_upper.startswith("FOR "):
            return BlockType.FOR
        elif line_upper == "ENDFOR":
            return BlockType.ENDFOR
        elif line_upper.startswith("WHILE "):
            return BlockType.WHILE
        elif line_upper == "ENDWHILE":
            return BlockType.ENDWHILE
        elif "CALL" in line:
            return BlockType.CALL
        else:
            return BlockType.ASSIGN
    
    def parse(self, dsl_code: str) -> List[CodeBlock]:
        """
        解析 DSL 代码为原子块列表（支持多行CALL语句）
        
        Args:
            dsl_code: Prompt 3.0 伪代码
            
        Returns:
            解析后的代码块列表
        """
        blocks = []
        lines = dsl_code.strip().split('\n')
        block_counter = 0
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()

            # 跳过空行和注释
            if not line or line.startswith("#"):
                i += 1
                continue

            # 处理多行 DEFINE 语句（字典、列表等）
            # DSL 语法: DEFINE {{variable}}: Type = value
            # 检查是否包含字典开始符 { 但没有对应的结束符 }（不包括在 {{}} 中的 }）
            if line.startswith("DEFINE") and ":" in line and "=" in line:
                # 检查是否有字典定义且不在同一行结束
                # 需要找到不在 {{}} 中的 { 和 }
                line_for_check = line
                # 移除 {{...}} 部分
                import re
                line_without_vars = re.sub(r'\{\{[^}]*\}\}', '', line_for_check)

                if '{' in line_without_vars and '}' not in line_without_vars:
                    # 收集多行 DEFINE 的所有行
                    define_lines = [line]
                    i += 1
                    while i < len(lines):
                        next_line = lines[i].strip()
                        if not next_line:
                            i += 1
                            continue
                        define_lines.append(next_line)
                        # 检查这一行是否有未匹配的 }
                        next_line_without_vars = re.sub(r'\{\{[^}]*\}\}', '', next_line)
                        if '}' in next_line_without_vars:
                            i += 1
                            break
                        i += 1

                    # 合并为单个 block
                    full_define = '\n'.join(define_lines)
                    block_type = BlockType.ASSIGN  # DEFINE 也是赋值类型

                    # 提取变量
                    vars_in_define = set()
                    for dl in define_lines:
                        vars_in_define.update(self._extract_vars(dl))

                    # 提取输出（从第一行的变量名）
                    outputs = set()
                    inputs = vars_in_define
                    if "=" in define_lines[0]:
                        left, right = define_lines[0].split("=", 1)
                        outputs = self._extract_vars(left)

                    blocks.append(CodeBlock(
                        id=f"OP_{block_counter}",
                        type=block_type,
                        code_lines=define_lines,
                        inputs=inputs,
                        outputs=outputs,
                        line_number=i - len(define_lines) + 1,
                        is_async=False
                    ))
                    block_counter += 1
                    continue

            # 处理单行 DEFINE 语句
            if line.startswith("DEFINE") and ":" in line and "=" in line:
                block_type = BlockType.ASSIGN
                vars_in_line = self._extract_vars(line)

                # 提取输出（从变量名）
                outputs = set()
                inputs = vars_in_line
                if "=" in line:
                    left, right = line.split("=", 1)
                    outputs = self._extract_vars(left)

                blocks.append(CodeBlock(
                    id=f"OP_{block_counter}",
                    type=block_type,
                    code_lines=[line],
                    inputs=inputs,
                    outputs=outputs,
                    line_number=i + 1,
                    is_async=False
                ))
                block_counter += 1
                i += 1
                continue

            # 检查是否是多行CALL语句
            if "CALL" in line and "(" in line and ")" not in line:
                # 收集多行CALL的所有行
                call_lines = [line]
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line:
                        i += 1
                        continue
                    call_lines.append(next_line)
                    if ")" in next_line:
                        i += 1
                        break
                    i += 1
                
                # 合并为单个block
                full_call = '\n'.join(call_lines)
                block_type = BlockType.CALL
                
                # 提取变量
                vars_in_call = set()
                for cl in call_lines:
                    vars_in_call.update(self._extract_vars(cl))
                
                # 提取输出（从第一行）
                outputs = set()
                inputs = set()
                if "=" in call_lines[0]:
                    left, right = call_lines[0].split("=", 1)
                    outputs = self._extract_vars(left)
                
                # 从所有行提取输入
                for cl in call_lines:
                    if "=" in cl:
                        _, right = cl.split("=", 1)
                        inputs.update(self._extract_vars(right))
                
                blocks.append(CodeBlock(
                    id=f"OP_{block_counter}",
                    type=block_type,
                    code_lines=call_lines,
                    inputs=inputs,
                    outputs=outputs,
                    line_number=i - len(call_lines) + 1,
                    is_async=True
                ))
                block_counter += 1
                continue
            
            # 单行处理
            block_type = self._classify_block_type(line)
            vars_in_line = self._extract_vars(line)
            
            # 控制流语句
            if block_type in [BlockType.IF, BlockType.ELIF, BlockType.ELSE, BlockType.ENDIF,
                             BlockType.FOR, BlockType.ENDFOR,
                             BlockType.WHILE, BlockType.ENDWHILE]:
                blocks.append(CodeBlock(
                    id=f"CTRL_{block_counter}",
                    type=block_type,
                    code_lines=[line],
                    inputs=vars_in_line,
                    outputs=set(),
                    line_number=i + 1
                ))
                block_counter += 1
                i += 1
                continue
            
            # 赋值或函数调用
            if "=" in line:
                left, right = line.split("=", 1)
                outputs = self._extract_vars(left)
                inputs = self._extract_vars(right)

                is_async = "CALL" in right

                blocks.append(CodeBlock(
                    id=f"OP_{block_counter}",
                    type=block_type,
                    code_lines=[line],
                    inputs=inputs,
                    outputs=outputs,
                    line_number=i + 1,
                    is_async=is_async
                ))
                block_counter += 1
            # 修复 P0-1: 处理 RETURN CALL 语句（没有等号的情况）
            elif line.upper().startswith("RETURN") and "CALL" in line:
                # RETURN CALL 语句，提取变量
                inputs = vars_in_line
                outputs = set()  # RETURN 语句没有本地输出，直接返回

                blocks.append(CodeBlock(
                    id=f"OP_{block_counter}",
                    type=block_type,
                    code_lines=[line],
                    inputs=inputs,
                    outputs=outputs,
                    line_number=i + 1,
                    is_async=True  # RETURN CALL 也是异步的
                ))
                block_counter += 1
            # 修复 P1-2: 处理单纯的 CALL 语句（没有等号，也不是 RETURN CALL）
            elif line.upper().startswith("CALL") or ("CALL" in line and not line.upper().startswith("RETURN")):
                # 纯 CALL 语句，提取变量
                inputs = vars_in_line
                outputs = set()  # 无输出的调用

                blocks.append(CodeBlock(
                    id=f"OP_{block_counter}",
                    type=block_type,
                    code_lines=[line],
                    inputs=inputs,
                    outputs=outputs,
                    line_number=i + 1,
                    is_async=True
                ))
                block_counter += 1
            # 处理单纯 RETURN 语句（没有 CALL）
            elif line.upper().startswith("RETURN ") and "CALL" not in line:
                # RETURN var 语句，提取返回的变量作为输出
                inputs = vars_in_line
                # 手动提取 RETURN 后的变量名
                return_part = line.strip()[7:].strip()  # 去掉 "RETURN "
                outputs = set()
                if return_part and not return_part.startswith('await') and not return_part.startswith('invoke'):
                    # 简单变量名（不是表达式）
                    if ' ' not in return_part and not return_part.endswith(')'):
                        outputs.add(return_part)

                blocks.append(CodeBlock(
                    id=f"OP_{block_counter}",
                    type=block_type,
                    code_lines=[line],
                    inputs=inputs,
                    outputs=outputs,
                    line_number=i + 1,
                    is_async=False
                ))
                block_counter += 1
            # 处理其他普通语句（没有CALL，没有等号，比如条件表达式）
            elif block_type == BlockType.ASSIGN:
                inputs = vars_in_line
                outputs = set()

                blocks.append(CodeBlock(
                    id=f"OP_{block_counter}",
                    type=block_type,
                    code_lines=[line],
                    inputs=inputs,
                    outputs=outputs,
                    line_number=i + 1,
                    is_async=False
                ))
                block_counter += 1

            i += 1
        
        return blocks


# ============================================================================
# 2. 依赖图分析器（核心算法）
# ============================================================================

class DependencyAnalyzer:
    """依赖关系图构建与分析"""
    
    def __init__(self, blocks: List[CodeBlock], external_inputs: Optional[Set[str]] = None):
        self.blocks = blocks
        self.graph = nx.DiGraph()
        self.var_producer: Dict[str, str] = {}  # 变量定义表
        self.external_inputs = set(external_inputs or set())
        
    def build_graph(self):
        """构建依赖有向无环图（DAG）"""
        # Step 1: 添加节点并建立变量生产者映射
        for block in self.blocks:
            self.graph.add_node(
                block.id, 
                type=block.type.value, 
                data=block,
                is_async=block.is_async
            )
            
            # 记录变量的生产者
            for out_var in block.outputs:
                if out_var in self.var_producer:
                    warning(f"变量 {out_var} 被重复定义")
                self.var_producer[out_var] = block.id
        
        # Step 2: 建立依赖边
        for block in self.blocks:
            for in_var in block.inputs:
                producer_id = self.var_producer.get(in_var)
                
                if producer_id and producer_id != block.id:
                    self.graph.add_edge(producer_id, block.id, var=in_var)
                    block.dependencies.add(producer_id)
                elif not producer_id and in_var not in self.external_inputs and block.type != BlockType.IF:
                    # IF 语句可以使用外部输入变量
                    warning(f"未定义变量: {in_var} 在块 {block.id} 中使用")
    
    def detect_cycles(self) -> bool:
        """检测循环依赖"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                error(f"检测到循环依赖: {cycles}")
                return True
            return False
        except:
            return False
    
    def find_dead_code(self) -> List[str]:
        """查找死代码（孤岛节点）"""
        dead_blocks = []
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0 and self.graph.out_degree(node) == 0:
                block_data = self.graph.nodes[node]['data']
                if block_data.type not in [BlockType.IF, BlockType.ELIF, BlockType.ELSE, BlockType.ENDIF]:
                    dead_blocks.append(node)
        return dead_blocks
    
    def topological_sort(self) -> List[str]:
        """拓扑排序 - 确定执行顺序"""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError as e:
            raise ValueError(f"拓扑排序失败（可能存在循环依赖）: {e}")
    
    def analyze_clusters(self, strategy="io_isolation") -> List[List[CodeBlock]]:
        """
        模块聚类 - 将代码块分组为逻辑模块

        策略:
        - io_isolation: IO 隔离策略（每个 LLM 调用独立成模块）
        - control_flow: 控制流内聚策略
        - hybrid: 混合策略（推荐）

        修复：确保完整的 IF-ELIF-ELSE-ENDIF 保持在同一模块中
        """
        # 按源代码行号排序（而非拓扑排序），保持控制流完整
        sorted_blocks = sorted(self.blocks, key=lambda b: b.line_number)

        modules = []
        current_module = []

        # 控制流块类型
        control_flow_start_types = [BlockType.IF, BlockType.FOR, BlockType.WHILE]
        control_flow_middle_types = [BlockType.ELIF, BlockType.ELSE]
        control_flow_end_types = [BlockType.ENDIF, BlockType.ENDFOR, BlockType.ENDWHILE]

        in_control_flow = False  # 标记是否在控制流内部
        control_flow_depth = 0  # 控制流嵌套深度

        for block in sorted_blocks:
            block_type = block.type

            # 处理控制流开始（IF/FOR/WHILE）
            if block_type in control_flow_start_types:
                # 只在最外层控制流开始时才保存之前的模块
                # 这样可以确保嵌套的控制流保持在同一模块中
                if control_flow_depth == 0:
                    if current_module:
                        modules.append(current_module)
                        current_module = []
                # 开始新的控制流
                in_control_flow = True
                control_flow_depth += 1
                current_module.append(block)
                continue

            # 处理控制流中间（ELIF/ELSE）
            if block_type in control_flow_middle_types:
                # ELIF/ELSE 必须在控制流内部
                if not in_control_flow:
                    error(f"发现独立的 {block_type.value} 语句，没有对应的 IF")
                    # 尝试合并到前一个模块
                    if not current_module and modules:
                        # 检查前一个模块是否以 IF 或 ELIF 结尾
                        last_module = modules[-1]
                        if last_module and last_module[-1].type in [BlockType.IF, BlockType.ELIF]:
                            # 将 ELIF/ELSE 合并到前一个模块
                            last_module.append(block)
                            continue
                    # 仍然添加到当前模块，避免丢失代码
                # 添加到当前模块
                current_module.append(block)
                continue

            # 处理控制流结束（ENDIF/ENDFOR/ENDWHILE）
            if block_type in control_flow_end_types:
                current_module.append(block)
                # 更新控制流状态
                control_flow_depth = max(0, control_flow_depth - 1)
                if control_flow_depth == 0:
                    in_control_flow = False
                    # 最外层控制流结束后保存模块
                    if current_module:
                        modules.append(current_module)
                        current_module = []
                continue

            # 处理普通代码块（CALL, ASSIGN）
            # 添加到当前模块
            current_module.append(block)

            # 决定是否切分模块
            should_split = False

            # 只在控制流外部时考虑切分
            if not in_control_flow:
                if strategy == "io_isolation":
                    # 每个 CALL 都独立成模块
                    should_split = block_type == BlockType.CALL
                elif strategy == "control_flow":
                    # 控制流边界切分（已在上方处理）
                    should_split = False
                elif strategy == "hybrid":
                    # CALL 时切分
                    should_split = block_type == BlockType.CALL

            # 执行切分
            if should_split and current_module:
                modules.append(current_module)
                current_module = []

        # 保存最后一个模块
        if current_module:
            modules.append(current_module)

        return modules
    
    def visualize(self, output_file="dependency_graph.png"):
        """可视化依赖图"""
        try:
            import matplotlib.pyplot as plt
            
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
            
            # 按类型着色
            color_map = {
                'CALL': '#ff6b6b',
                'ASSIGN': '#4ecdc4',
                'IF': '#ffe66d',
                'ELSE': '#ffe66d',
                'ENDIF': '#ffe66d'
            }
            
            node_colors = [color_map.get(self.graph.nodes[n]['type'], '#95a5a6') 
                          for n in self.graph.nodes()]
            
            plt.figure(figsize=(12, 8))
            nx.draw(self.graph, pos, 
                   node_color=node_colors,
                   node_size=2000,
                   with_labels=True,
                   font_size=8,
                   font_weight='bold',
                   arrows=True,
                   edge_color='gray')
            
            plt.title("代码依赖关系图 (Dependency DAG)", fontsize=14)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            info(f"依赖图已保存至: {output_file}")
        except ImportError:
            warning("matplotlib 未安装，跳过可视化")


# ============================================================================
# 3. 模块代码生成器
# ============================================================================

class ModuleSynthesizer:
    """将代码块簇转换为 Python 函数"""

    def __init__(self):
        self.llm_client_import = "from llm_client import invoke_function"
        self.sb_func_info = {}  # semantic block 函数信息映射 {func_name: {"input_vars": [...], ...}}
        
    def _translate_call(self, pseudo_line: str) -> str:
        """
        将伪代码的 CALL 转换为 Python LLM 调用（增强容错性）
        
        示例:
        {{result}} = CALL generate_outline({{topic}}, {{level}})
        =>
        result = await invoke_function('generate_outline', topic=topic, level=level)
        
        支持的容错模式:
        - 缺失右括号: {{result}} = CALL func(
        - 参数为空: {{result}} = CALL func()
        - 参数包含中文字符
        """
        # 提取变量名（去除 {{}}）
        cleaned = pseudo_line.replace("{{", "").replace("}}", "")

        # 新增模式0: 带实现的 CALL 语法
        # {{result}} = CALL func(args) = await actual_implementation
        # 这种格式直接提取 "= await ..." 部分作为实现
        match = re.match(r'(\w+)\s*=\s*CALL\s+(\w+)\s*\((.*?)\)\s*=\s*(.+)$', cleaned)
        if match:
            result_var = match.group(1)
            func_name = match.group(2)
            args_str = match.group(3)
            implementation = match.group(4).strip()
            return f"{result_var} = {implementation}"

        # 容错模式1: 标准格式 result = CALL func(arg1, arg2)
        match = re.match(r'(\w+)\s*=\s*CALL\s+(\w+)\s*\((.*?)\)\s*$', cleaned)
        if match:
            result_var = match.group(1)
            func_name = match.group(2)
            args_str = match.group(3)
            if _check_and_get_impl(func_name):
                impl = _check_and_get_impl(func_name)
                return f"{result_var} = {impl}"
            # 如果是 semantic block 函数，直接调用本地函数
            if func_name in self.sb_func_info:
                input_vars = self.sb_func_info[func_name].get('input_vars', [])
                return self._format_sb_call(result_var, func_name, args_str, input_vars)
            return self._format_call(result_var, func_name, args_str)

        # 容错模式2: 缺失右括号 result = CALL func(
        match = re.match(r'(\w+)\s*=\s*CALL\s+(\w+)\s*\(\s*$', cleaned)
        if match:
            result_var = match.group(1)
            func_name = match.group(2)
            if _check_and_get_impl(func_name):
                impl = _check_and_get_impl(func_name)
                return f"{result_var} = {impl}"
            if func_name in self.sb_func_info:
                input_vars = self.sb_func_info[func_name].get('input_vars', [])
                return self._format_sb_call(result_var, func_name, "", input_vars)
            return f"{result_var} = await invoke_function('{func_name}')"

        # 容错模式3: 参数后跟换行 result = CALL func(arg1, arg2
        match = re.match(r'(\w+)\s*=\s*CALL\s+(\w+)\s*\((.*?)\s*$', cleaned)
        if match:
            result_var = match.group(1)
            func_name = match.group(2)
            args_str = match.group(3)
            if _check_and_get_impl(func_name):
                impl = _check_and_get_impl(func_name)
                return f"{result_var} = {impl}"
            if func_name in self.sb_func_info:
                input_vars = self.sb_func_info[func_name].get('input_vars', [])
                return self._format_sb_call(result_var, func_name, args_str, input_vars)
            return self._format_call(result_var, func_name, args_str)

        # 容错模式5: 单纯 CALL 语句（没有等号）CALL func(arg1, arg2)
        match = re.match(r'CALL\s+(\w+)\s*\((.*?)\)\s*$', cleaned)
        if match:
            func_name = match.group(1)
            args_str = match.group(2)

            if _check_and_get_impl(func_name):
                impl = _check_and_get_impl(func_name)
                return impl

            if not args_str.strip():
                if func_name in self.sb_func_info:
                    input_vars = self.sb_func_info[func_name].get('input_vars', [])
                    return self._format_sb_call("_", func_name, "", input_vars)
                return f"await invoke_function('{func_name}')"

            # 处理参数
            args = []
            kwargs = []
            for arg in args_str.split(','):
                arg = arg.strip()
                if not arg:
                    continue
                if '=' in arg:
                    parts = arg.split('=', 1)
                    if len(parts) == 2:
                        param_name = self._sanitize_identifier(parts[0].strip())
                        param_value = parts[1].strip()
                        if param_value and not (param_value.startswith('"') or param_value.startswith("'")):
                            param_value = self._sanitize_identifier(param_value)
                        kwargs.append(f"{param_name}={param_value}")
                else:
                    # 纯变量名或数字字面量
                    if arg.isdigit():
                        args.append(arg)  # 使用原始数字
                    else:
                        cleaned_arg = self._sanitize_identifier(arg)
                        kwargs.append(f"{cleaned_arg}={cleaned_arg}")

            call_parts = []
            if args:
                call_parts.append(', '.join(args))
            if kwargs:
                call_parts.append(', '.join(kwargs))
            call_str = ', '.join(call_parts)
            return f"await invoke_function('{func_name}', {call_str})"

        # 容错模式6: 单纯 CALL 语句（没有等号，没有参数）CALL func
        match = re.match(r'CALL\s+(\w+)\s*$', cleaned)
        if match:
            func_name = match.group(1)
            return f"await invoke_function('{func_name}')"

        # 容错模式7: 多返回值元组解包 var1, var2 = CALL func(args)
        match = re.match(r'([\w,\s]+?)\s*=\s*CALL\s+(\w+)\s*\((.*?)\)\s*$', cleaned)
        if match:
            result_vars_str = match.group(1).strip()
            func_name = match.group(2)
            args_str = match.group(3)
            # 清理多个返回变量名
            result_vars = [self._sanitize_identifier(v.strip()) for v in result_vars_str.split(',') if v.strip()]
            result_vars_formatted = ', '.join(result_vars)
            if _check_and_get_impl(func_name):
                impl = _check_and_get_impl(func_name)
                return f"{result_vars_formatted} = {impl}"
            if func_name in self.sb_func_info:
                input_vars = self.sb_func_info[func_name].get('input_vars', [])
                return self._format_sb_call(result_vars_formatted, func_name, args_str, input_vars)
            return self._format_call(result_vars_formatted, func_name, args_str)

        # 容错模式8: 单返回值不完整 var = CALL func（无括号）
        match = re.match(r'(\w+)\s*=\s*CALL\s+(\w+)', cleaned)
        if match:
            result_var = match.group(1)
            func_name = match.group(2)
            warning(f"CALL 语句格式不完整，使用默认调用: {pseudo_line}")
            return f"{result_var} = await invoke_function('{func_name}')"

        # 无法解析，返回原始行并警告
        warning(f"无法解析 CALL 语句: {pseudo_line}")
        return cleaned
    
    def _format_call(self, result_var: str, func_name: str, args_str: str) -> str:
        """格式化参数并生成 invoke_function 调用"""
        # 清理结果变量名
        result_var = self._sanitize_identifier(result_var)

        if not args_str.strip():
            return f"{result_var} = await invoke_function('{func_name}')"

        # 解析参数（支持中文变量名、逗号分隔）
        args = []
        kwargs = []
        for arg in args_str.split(','):
            arg = arg.strip()
            if not arg:
                continue

            # 检查是否是参数赋值格式 param=value
            if '=' in arg:
                # 清理参数名和值
                parts = arg.split('=', 1)
                if len(parts) == 2:
                    param_name = self._sanitize_identifier(parts[0].strip())
                    param_value = parts[1].strip()
                    # 清理值部分（如果是变量名）
                    if param_value and not (param_value.startswith('"') or param_value.startswith("'")):
                        param_value = self._sanitize_identifier(param_value)
                    kwargs.append(f"{param_name}={param_value}")
            else:
                # 纯变量名或数字字面量
                if arg.isdigit():
                    args.append(arg)  # 使用原始数字
                else:
                    cleaned_arg = self._sanitize_identifier(arg)
                    kwargs.append(f"{cleaned_arg}={cleaned_arg}")

        # 构建调用字符串
        call_parts = []
        if args:
            call_parts.append(', '.join(args))
        if kwargs:
            call_parts.append(', '.join(kwargs))

        call_str = ', '.join(call_parts)
        return f"{result_var} = await invoke_function('{func_name}', {call_str})"

    def _format_sb_call(self, result_var: str, func_name: str, args_str: str,
                       input_vars: List[Dict[str, str]] = None) -> str:
        """生成 semantic block 的直接调用（而非通过 invoke_function）

        semantic block 函数签名: async def sb_xxx(ctx: dict) -> dict
        应该转换为: result = await sb_xxx({"param1": value1, "param2": value2, ...})
        """
        ctx_str = self._build_sb_ctx(args_str, input_vars or [])
        return f"{result_var} = await {func_name}({ctx_str})"

    def _build_sb_ctx(self, call_str: str, input_vars: List[Dict[str, str]]) -> str:
        """构建 semantic block 的 ctx 参数

        Args:
            call_str: 参数字符串，如 "key1=val1, key2=val2"
            input_vars: semantic block 定义的输入变量列表
        Returns:
            Python dict 字符串，如 '{"key1": var1, "key2": var2}'
        """
        ctx_parts = []
        args_list = [a.strip() for a in call_str.split(',') if a.strip()]

        if args_list:
            if '=' in args_list[0]:
                # 命名参数格式
                for arg in args_list:
                    if '=' in arg:
                        parts = arg.split('=', 1)
                        if len(parts) == 2:
                            key = self._sanitize_identifier(parts[0].strip())
                            val = self._sanitize_identifier(parts[1].strip())
                            ctx_parts.append(f'"{key}": {val}')
            else:
                # 位置参数格式
                for i, arg in enumerate(args_list):
                    if i < len(input_vars):
                        key = input_vars[i].get('name', f'arg{i}')
                        val = self._sanitize_identifier(arg)
                        ctx_parts.append(f'"{key}": {val}')

        return "{" + ", ".join(ctx_parts) + "}"

    def _clean_line(self, line: str) -> str:
        """清理伪代码为 Python 代码（增强 FOR 循环转换）"""
        # 去除 {{}}
        cleaned = line.replace("{{", "").replace("}}", "")

        # DSL 使用 JSON 风格布尔字面量；Python 代码必须使用 True/False。
        # 引号边界用于避免改写常见的字符串值（例如 "true"）。
        cleaned = re.sub(r'(?<!["\'])\btrue\b(?!["\'])', 'True', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(?<!["\'])\bfalse\b(?!["\'])', 'False', cleaned, flags=re.IGNORECASE)

        # 跳过标签行（如 block_main:、step_1: 等）
        stripped = cleaned.strip()
        if stripped.endswith(':') and not any(kw in stripped.upper() for kw in ['IF ', 'ELIF ', 'FOR ', 'WHILE ', 'DEFINE ']):
            # 纯标签行，没有关键字，是标签
            return ''
        
        # 转换控制流关键字
        cleaned = re.sub(r'^ELSE:?', 'else:', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^ENDIF:?', '', cleaned, flags=re.IGNORECASE)

        # 修复：移除 IF 语句中的非法 FOR ... MINUTES 语法
        # 例如：IF {{query_rate}} < 10 FOR {{duration}} MINUTES -> IF {{query_rate}} < 10
        if re.match(r'^IF\s+.+\s+FOR\s+\w+\s+MINUTES\s*$', cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(r'\s+FOR\s+\w+\s+MINUTES\s*$', '', cleaned, flags=re.IGNORECASE)
            warning(f"移除了非法的 FOR ... MINUTES 语法: {line}")

        # 修复 P1-1: 处理 IF CALL 语句（必须在去除 {{}} 后立即处理）
        # 检查是否是 "IF CALL func(...)" 格式
        if re.match(r'^IF\s+CALL\s+\w+', cleaned, flags=re.IGNORECASE):
            match = re.match(r'IF\s+CALL\s+(\w+)\s*\((.*?)\)\s*$', cleaned, flags=re.IGNORECASE)
            if match:
                func_name = match.group(1)
                args_str = match.group(2)
                # 格式化参数
                if args_str.strip():
                    args = []
                    kwargs = []
                    for arg in args_str.split(','):
                        arg = arg.strip()
                        if not arg:
                            continue
                        if '=' in arg:
                            parts = arg.split('=', 1)
                            if len(parts) == 2:
                                param_name = self._sanitize_identifier(parts[0].strip())
                                param_value = parts[1].strip()
                                if param_value and not (param_value.startswith('"') or param_value.startswith("'")):
                                    param_value = self._sanitize_identifier(param_value)
                                kwargs.append(f"{param_name}={param_value}")
                        else:
                            # 纯变量名或数字字面量
                            if arg.isdigit():
                                args.append(arg)  # 使用原始数字
                            else:
                                cleaned_arg = self._sanitize_identifier(arg)
                                kwargs.append(f"{cleaned_arg}={cleaned_arg}")
                    call_parts = []
                    if args:
                        call_parts.append(', '.join(args))
                    if kwargs:
                        call_parts.append(', '.join(kwargs))
                    call_str = ', '.join(call_parts)
                    if func_name in self.sb_func_info:
                        input_vars = self.sb_func_info[func_name].get('input_vars', [])
                        return f"if await {func_name}({self._build_sb_ctx(call_str, input_vars)}):"
                    return f"if await invoke_function('{func_name}', {call_str}):"
                else:
                    if func_name in self.sb_func_info:
                        return f"if await {func_name}({{}}):"
                    return f"if await invoke_function('{func_name}'):"

        # 修复 P0-1: 处理 RETURN CALL 语句
        # 将 "RETURN CALL func(...)" 转换为 "return await invoke_function('func', ...)"
        if re.match(r'^RETURN\s+CALL\s+\w+', cleaned, flags=re.IGNORECASE):
            match = re.match(r'RETURN\s+CALL\s+(\w+)\s*\((.*?)\)\s*$', cleaned, flags=re.IGNORECASE)
            if match:
                func_name = match.group(1)
                args_str = match.group(2)
                # 格式化参数
                if args_str.strip():
                    args = []
                    kwargs = []
                    for arg in args_str.split(','):
                        arg = arg.strip()
                        if not arg:
                            continue
                        if '=' in arg:
                            # 清理参数名和值
                            parts = arg.split('=', 1)
                            if len(parts) == 2:
                                param_name = self._sanitize_identifier(parts[0].strip())
                                param_value = parts[1].strip()
                                # 清理值部分（如果是变量名）
                                if param_value and not (param_value.startswith('"') or param_value.startswith("'")):
                                    param_value = self._sanitize_identifier(param_value)
                                kwargs.append(f"{param_name}={param_value}")
                        else:
                            # 纯变量名或数字字面量
                            if arg.isdigit():
                                args.append(arg)  # 使用原始数字
                            else:
                                cleaned_arg = self._sanitize_identifier(arg)
                                kwargs.append(f"{cleaned_arg}={cleaned_arg}")
                    call_parts = []
                    if args:
                        call_parts.append(', '.join(args))
                    if kwargs:
                        call_parts.append(', '.join(kwargs))
                    call_str = ', '.join(call_parts)
                    if func_name in self.sb_func_info:
                        input_vars = self.sb_func_info[func_name].get('input_vars', [])
                        return f"return await {func_name}({self._build_sb_ctx(call_str, input_vars)})"
                    return f"return await invoke_function('{func_name}', {call_str})"
                else:
                    if func_name in self.sb_func_info:
                        return f"return await {func_name}({{}})"
                    return f"return await invoke_function('{func_name}')"

        # 转换控制流
        cleaned = re.sub(r'^IF\s+', 'if ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^ELIF\s+', 'elif ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^ELSE$', 'else:', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^ENDIF$', '', cleaned, flags=re.IGNORECASE)

        # 确保 elif 语句以冒号结尾（如果还没有）
        if cleaned.strip().startswith('elif ') and not cleaned.strip().endswith(':'):
            cleaned = cleaned.strip() + ':'

        # 增强 FOR 循环转换（支持多种格式）
        # 格式1: FOR var IN range(end) 或 FOR var IN range(end):
        match = re.match(r'^FOR\s+(\w+)\s+IN\s+range\((\w+)\)(?::?\s*)$', cleaned, flags=re.IGNORECASE)
        if match:
            loop_var, end_var = match.groups()
            cleaned = f"for {loop_var} in range({end_var}):"
        # 格式2: FOR var IN range(start, end)
        elif re.match(r'^FOR\s+(\w+)\s+IN\s+range\([^)]+\)(?::?\s*)$', cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(r'^FOR\s+(\w+)\s+IN\s+', r'for \1 in ', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r':+\s*$', '', cleaned)
            cleaned = cleaned.rstrip() + ':'
        # 格式3: 通用 FOR var IN iterable（支持 Python 表达式）
        elif re.match(r'^FOR\s+(\w+)\s+IN\s+', cleaned, flags=re.IGNORECASE):
            # 提取变量名和 IN 后面的表达式
            match = re.match(r'^FOR\s+(\w+)\s+IN\s+(.+)$', cleaned, flags=re.IGNORECASE)
            if match:
                loop_var, iterable = match.groups()
                # 如果 iterable 已经是 Python 格式（包含 [、]、(、)、.、函数调用等），直接使用
                # 如果是简单的变量名，也直接使用
                cleaned = f"for {loop_var} in {iterable}:"

        cleaned = re.sub(r'^ENDFOR$', '', cleaned, flags=re.IGNORECASE)

        # 转换逻辑运算符（DSL -> Python）
        cleaned = re.sub(r'\bAND\b', ' and ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bOR\b', ' or ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bNOT\b', ' not ', cleaned, flags=re.IGNORECASE)

        # 转换 IS NOT 运算符（DSL -> Python）
        cleaned = re.sub(r'\bIS\s+NOT\b', ' is not ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bIS\b', ' is ', cleaned, flags=re.IGNORECASE)

        # 转换 CONTAINS 运算符（DSL -> Python）
        # 实际DSL格式1: {{user_input}} CONTAINS "返点" OR {{user_input}} CONTAINS "价格" OR ...
        # 实际DSL格式2: user_input CONTAINS ["返点", "价格", "预算", ...]
        # 期望Python: "返点" in user_input or "价格" in user_input or ...
        # 支持带 if/elif 前缀的格式: if user_input CONTAINS [...]
        # 注意：IF/ELIF 已经在上面转换成小写，所以这里检查小写形式
        cleaned_stripped = cleaned.strip()
        prefix = ''
        condition_part = cleaned_stripped
        if cleaned_stripped.startswith('if '):
            prefix = 'if '
            condition_part = cleaned_stripped[3:]
        elif cleaned_stripped.startswith('elif '):
            prefix = 'elif '
            condition_part = cleaned_stripped[5:]

        def _convert_contains(condition: str) -> str:
            """转换 CONTAINS 条件表达式"""
            condition = condition.strip()
            # 处理 DSL 格式: not CONTAINS [...] (not 在开头)
            if re.match(r'^not\s+', condition, flags=re.IGNORECASE):
                not_contains_arr_match = re.match(r'^not\s+(\S+)\s+CONTAINS\s+\[([^\]]+)\]$', condition, flags=re.IGNORECASE)
                if not_contains_arr_match:
                    var = not_contains_arr_match.group(1).strip().strip('{}')
                    items = [item.strip().strip('"\'') for item in not_contains_arr_match.group(2).split(',')]
                    conditions = ['"' + item + '" in ' + var for item in items]
                    return "not (" + " or ".join(conditions) + ")"
                not_contains_match = re.match(r"^not\s+(\S+)\s+CONTAINS\s+\"[^\"]+\"(\s+OR\s+\1\s+CONTAINS\s+\"[^\"]+\")*$", condition, flags=re.IGNORECASE)
                if not_contains_match:
                    var = not_contains_match.group(1).strip().strip('{}')
                    parts = re.split(r'\s+OR\s+', condition[4:].strip(), flags=re.IGNORECASE)
                    conditions = []
                    for part in parts:
                        m = re.match(r'(\S+)\s+CONTAINS\s+"([^"]+)"', part.strip(), flags=re.IGNORECASE)
                        if m:
                            conditions.append(f'"{m.group(2)}" in {var}')
                    if conditions:
                        return 'not (' + ' or '.join(conditions) + ')'
                match = re.match(r'^not\s+(\S+)\s+CONTAINS\s+"([^"]+)"$', condition, flags=re.IGNORECASE)
                if match:
                    var = match.group(1).strip().strip('{}')
                    item = match.group(2)
                    return f'not ("{item}" in {var})'
            # 处理 DSL 格式: var not CONTAINS [...] (not 在中间，变量名在前)
            else:
                not_contains_arr_match = re.match(r'^(\S+?)\s+not\s+CONTAINS\s+\[([^\]]+)\]$', condition, flags=re.IGNORECASE)
                if not_contains_arr_match:
                    var = not_contains_arr_match.group(1).strip().strip('{}')
                    items = [item.strip().strip('"\'') for item in not_contains_arr_match.group(2).split(',')]
                    conditions = ['"' + item + '" in ' + var for item in items]
                    return "not (" + " or ".join(conditions) + ")"
                contains_arr_match = re.match(r'^(\S+)\s+CONTAINS\s+\[([^\]]+)\]$', condition, flags=re.IGNORECASE)
                if contains_arr_match:
                    var = contains_arr_match.group(1).strip().strip('{}')
                    items = [item.strip().strip('"\'') for item in contains_arr_match.group(2).split(',')]
                    return ' or '.join(f'"{item}" in {var}' for item in items)
                contains_match = re.match(r"^(\S+)\s+CONTAINS\s+\"[^\"]+\"(\s+OR\s+\1\s+CONTAINS\s+\"[^\"]+\")*$", condition, flags=re.IGNORECASE)
                if contains_match:
                    var = contains_match.group(1).strip().strip('{}')
                    parts = re.split(r'\s+OR\s+', condition, flags=re.IGNORECASE)
                    conditions = []
                    for part in parts:
                        m = re.match(r'(\S+)\s+CONTAINS\s+"([^"]+)"', part.strip(), flags=re.IGNORECASE)
                        if m:
                            conditions.append(f'"{m.group(2)}" in {var}')
                    if conditions:
                        return ' or '.join(conditions)
                match = re.match(r'^(\S+)\s+CONTAINS\s+"([^"]+)"$', condition, flags=re.IGNORECASE)
                if match:
                    var = match.group(1).strip().strip('{}')
                    item = match.group(2)
                    return f'"{item}" in {var}'
            return condition

        converted_condition = _convert_contains(condition_part)
        if converted_condition != condition_part:
            cleaned = prefix + converted_condition
            if not cleaned.endswith(':'):
                cleaned = cleaned + ':'

        # 转换 IN 运算符（DSL -> Python）
        # 注意：必须在 CONTAINS 之后处理，避免误替换
        cleaned = re.sub(r'\bIN\b', ' in ', cleaned)

        # 确保if语句以冒号结尾（如果还没有）
        if cleaned.strip().startswith('if ') and not cleaned.strip().endswith(':'):
            cleaned = cleaned.strip() + ':'

        # 处理 RETURN 语句（将 RETURN 转换为小写的 return）
        if cleaned.strip().startswith("RETURN "):
            # 提取 RETURN 后的内容
            return_expr = cleaned.strip()[7:].strip()
            cleaned = f"return {return_expr}"

        # 处理 CALL（非 IF CALL 的情况）
        if "CALL" in line:
            cleaned = self._translate_call(line)

        # 修复 P0-2: 清理非 CALL 行中的变量引用
        # 查找所有可能的标识符（不以数字开头的字母数字下划线组合）
        # 但要跳过关键字、字符串、数字等
        cleaned = self._sanitize_line_variables(cleaned)

        # 修复 DSL 条件表达式错误序列化为 dict 字面量的问题
        # 例如: discount_rate = {'if': '...', 'then': '...', 'else': {...}
        # 这类 dict 字面量在 DSL 中没有闭合，会导致生成的 Python 语法错误
        if "= {" in cleaned and ("'if':" in cleaned or '"if":' in cleaned or "'then':" in cleaned or '"then":' in cleaned):
            # 检测到疑似条件表达式序列化为 dict 的情况
            # 尝试用 Python AST 检测括号是否匹配
            try:
                # 将该行替换为安全的 pass 语句，变量设为 None
                # 提取变量名
                assign_match = re.match(r'^\s*(\w+)\s*=', cleaned)
                if assign_match:
                    var_name = assign_match.group(1).strip()
                    cleaned = f"{var_name} = None  # 条件表达式转换失败，使用默认值"
                    warning(f"检测到未闭合的 dict 条件表达式，已替换为安全代码: {var_name} = None")
                else:
                    cleaned = "pass  # 无效代码行"
            except Exception:
                pass

        return cleaned
    
    def _sanitize_line_variables(self, line: str) -> str:
        """
        清理代码行中的变量引用，使其符合 Python 标识符规范
        只处理非 CALL 语句中的变量引用
        """
        # 定义 Python 关键字，不进行替换
        python_keywords = {
            'if', 'else', 'elif', 'for', 'while', 'in', 'and', 'or', 'not',
            'is', 'none', 'true', 'false', 'return', 'def', 'async', 'await',
            'import', 'from', 'class', 'pass', 'break', 'continue', 'elif'
        }
        
        # 查找所有标识符（包括以数字开头的）
        # 匹配：字母、数字、下划线的组合（包括以数字开头的）
        tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', line)
        
        # 对每个 token 检查是否需要清理
        for token in set(tokens):
            token_lower = token.lower()
            # 跳过关键字
            if token_lower in python_keywords:
                continue
            # 跳过纯数字
            if token.isdigit():
                continue
            # 如果以数字开头，需要清理
            if token[0].isdigit():
                sanitized = self._sanitize_identifier(token)
                # 替换所有出现的位置（作为独立单词）
                line = re.sub(r'\b' + re.escape(token) + r'\b', sanitized, line)
        
        return line
    
    def _sanitize_identifier(self, identifier: str) -> str:
        """
        清理变量名，确保符合 Python 标识符规范
        - 不能以数字开头
        - 只能包含字母、数字、下划线
        """
        # 如果是纯数字字符串，不需要清理
        if identifier.isdigit():
            return identifier

        # 如果以数字开头，添加下划线前缀
        if identifier and identifier[0].isdigit():
            identifier = '_' + identifier

        # 替换非字母数字下划线字符为下划线
        identifier = re.sub(r'[^a-zA-Z0-9_]', '_', identifier)

        return identifier
    
    def generate_module(self, cluster: List[CodeBlock], module_id: int,
                        semantic_blocks: List[Dict[str, Any]] = None) -> ModuleDefinition:
        """生成单个模块的 Python 代码

        Args:
            cluster: 代码块集群
            module_id: 模块 ID
            semantic_blocks: SEMANTIC_BLOCK 列表，用于判断是否应直接调用本地函数
        """
        if semantic_blocks is None:
            semantic_blocks = []

        # 构建 semantic block 函数信息映射
        sb_func_info = {}
        for sb in semantic_blocks:
            block_id = sb.get('block_id', '')
            func_name = f'sb_{block_id}' if not block_id.startswith('sb_') else block_id
            sb_func_info[func_name] = {
                'input_vars': sb.get('input_vars', []),
                'output_vars': sb.get('output_vars', [])
            }

        # 收集输入输出
        all_inputs = set()
        all_outputs = set()
        internal_vars = set()
        body_lines = []
        is_async = False

        for block in cluster:
            all_inputs.update(block.inputs)
            all_outputs.update(block.outputs)
            internal_vars.update(block.outputs)

            if block.is_async:
                is_async = True

        # 修复 P0-2: 清理变量名，确保符合 Python 标识符规范
        # 不再使用 sorted()，保持变量出现的原始顺序
        external_inputs = [self._sanitize_identifier(v) for v in all_inputs - internal_vars]
        final_outputs = [self._sanitize_identifier(v) for v in all_outputs]

        # 生成函数名
        func_name = self._generate_function_name(cluster, module_id)

        # 组装函数体
        async_kw = "async " if is_async else ""
        code_lines = [f"{async_kw}def {func_name}({', '.join(external_inputs)}):"]
        code_lines.append('    """Auto-generated module"""')

        # 控制流状态跟踪（使用栈来跟踪嵌套层级）
        control_flow_stack = []  # 跟踪每个控制流的缩进层级
        control_flow_types = [BlockType.IF, BlockType.FOR, BlockType.WHILE]
        control_flow_middle_types = [BlockType.ELIF, BlockType.ELSE]
        control_flow_end_types = [BlockType.ENDIF, BlockType.ENDFOR, BlockType.ENDWHILE]

        # 标记是否已经包含 RETURN 语句
        has_return_statement = False

        # 转换代码行并处理多行CALL语句
        for block in cluster:
            block_type = block.type

            # 处理 DEFINE 语句（如果包含 = value，则生成赋值代码）
            if block.code_lines and block.code_lines[0].strip().startswith("DEFINE"):
                for line in block.code_lines:
                    # 匹配 DEFINE {{var_name}}: Type = value 格式
                    match = re.match(r'DEFINE\s+\{\{(\w+)\}\}\s*:\s*\w+\s*=\s*(.+)$', line.strip())
                    if match:
                        var_name = self._sanitize_identifier(match.group(1))
                        value = match.group(2).strip()
                        code_lines.append(f"    {var_name} = {value}")
                        debug(f"生成常量赋值: {var_name} = {value}")
                    else:
                        # 没有赋值的 DEFINE 语句，跳过
                        debug(f"跳过无赋值的 DEFINE 语句: {line[:50]}...")
                continue

            # 处理控制流开始（IF/FOR/WHILE）
            if block_type in control_flow_types:
                for line in block.code_lines:
                    if line.strip():
                        cleaned = self._clean_line(line)
                        if cleaned and cleaned.strip():
                            # 控制流语句的缩进：基础缩进4个空格 + 当前栈深度
                            indent = "    " * (1 + len(control_flow_stack))
                            code_lines.append(f"{indent}{cleaned}")
                # 控制流入栈
                control_flow_stack.append(block_type)

            # 处理控制流结束（ENDIF/ENDFOR/ENDWHILE）
            elif block_type in control_flow_end_types:
                # 控制流出栈
                if control_flow_stack:
                    control_flow_stack.pop()
                # ENDIF/ENDFOR 等不生成代码（Python 使用缩进）
                continue

            # 处理 ELSE 分支（特殊处理：与 IF 同级缩进）
            elif block_type == BlockType.ELSE:
                # 判断是否在 IF 内部（栈顶是 IF 或之前已在 ELSE/ELIF 中）
                in_if_branch = bool(control_flow_stack) and control_flow_stack[-1] == BlockType.IF
                
                for line in block.code_lines:
                    if line.strip():
                        cleaned = self._clean_line(line)
                        if cleaned and cleaned.strip():
                            # ELSE 冒号与 IF 同级缩进
                            if cleaned.strip() == "else:" or cleaned.strip() == "else":
                                # 弹出 IF 标记，因为我们现在在 ELSE 分支中
                                if control_flow_stack and control_flow_stack[-1] == BlockType.IF:
                                    control_flow_stack.pop()
                                # 标记我们仍在 IF 内部但现在是 else 分支
                                control_flow_stack.append(BlockType.ELSE)
                                indent = "    " * len(control_flow_stack)
                                code_lines.append(f"{indent}else:")
                            # ELSE 内部代码（缩进比 ELSE 多一层）
                            else:
                                indent = "    " * (1 + len(control_flow_stack))
                                code_lines.append(f"{indent}{cleaned}")
                # ELSE 后面的代码缩进应该与 IF 体内相同
                continue

            # 处理 ELIF 分支（特殊处理：与 IF 同级缩进）
            elif block_type == BlockType.ELIF:
                for line in block.code_lines:
                    if line.strip():
                        cleaned = self._clean_line(line)
                        if cleaned and cleaned.strip():
                            # ELIF 冒号与 IF 同级缩进（如果栈不为空，使用栈深度；否则使用 0）
                            if cleaned.strip().startswith("elif"):
                                if control_flow_stack:
                                    indent = "    " * len(control_flow_stack)
                                else:
                                    indent = "    "  # 外层 ELIF，应该与 IF 同级
                                code_lines.append(f"{indent}{cleaned.strip()}")
                            # ELIF 内部代码（缩进比 ELIF 多一层）
                            else:
                                if control_flow_stack:
                                    indent = "    " * (1 + len(control_flow_stack))
                                else:
                                    indent = "        "  # 外层 ELIF 内部，应该缩进 8 个空格
                                code_lines.append(f"{indent}{cleaned}")
                # ELIF 后面的代码缩进应该与 IF 体内相同
                continue

            # 处理普通代码块（CALL, ASSIGN, RETURN）
            else:
                # 计算当前缩进级别：基础缩进4个空格 + 控制流栈深度
                indent = "    " * (1 + len(control_flow_stack))

                # 检查是否是多行CALL block
                is_multiline_call = len(block.code_lines) > 1 and any("CALL" in line for line in block.code_lines)

                if is_multiline_call:
                    # 多行CALL：合并后一次性转换
                    translated = self._parse_multiline_call(block.code_lines)
                    if translated and translated.strip():
                        code_lines.append(f"{indent}{translated}")
                else:
                    # 单行处理
                    for line in block.code_lines:
                        # 检查原始行是否包含CALL
                        if "CALL" in line:
                            # 单行CALL，使用 _translate_call 转换为合法 Python
                            # _translate_call 会处理：var = CALL func(args) → var = await func(args)
                            # 以及 CALL sb_xxx() → await sb_xxx(ctx) 等格式
                            translated = self._translate_call(line)
                            if translated and translated.strip():
                                code_lines.append(f"{indent}{translated}")
                            else:
                                # _translate_call 无法处理时，回退到 _clean_line
                                cleaned = self._clean_line(line)
                                if cleaned and cleaned.strip():
                                    warning(f"_translate_call 返回空，回退 _clean_line: {line[:60]}")
                                    code_lines.append(f"{indent}{cleaned}")
                        elif line.strip():
                            # 其他代码行（ASSIGN, RETURN等）
                            cleaned = self._clean_line(line)
                            if cleaned and cleaned.strip():
                                code_lines.append(f"{indent}{cleaned}")
                                # 检查是否是 RETURN 语句
                                if cleaned.strip().startswith('return '):
                                    has_return_statement = True

        # 返回值
        if has_return_statement:
            # 如果已经有 RETURN 语句，不再添加默认 return
            pass
        elif final_outputs:
            code_lines.append(f"    return {', '.join(final_outputs)}")
        else:
            code_lines.append("    return None")

        body_code = '\n'.join(code_lines)

        return ModuleDefinition(
            name=func_name,
            inputs=external_inputs,
            outputs=final_outputs,
            body_code=body_code,
            is_async=is_async,
            original_blocks=cluster
        )
    
    def _flush_call_buffer(self, code_lines: List[str], call_buffer: List[str]) -> None:
        """将多行CALL语句缓冲区转换为单行并添加到code_lines"""
        if not call_buffer:
            return
        
        # 检查缓冲区中是否有CALL语句
        has_call = any("CALL" in line for line in call_buffer)
        
        if has_call:
            # 直接解析多行CALL
            translated = self._parse_multiline_call(call_buffer)
            if translated and translated.strip():
                code_lines.append(f"    {translated}")
        else:
            # 普通参数行，直接转换后添加
            for line in call_buffer:
                cleaned = self._clean_line(line)
                if cleaned and cleaned.strip():
                    code_lines.append(f"    {cleaned}")
    
    def _parse_multiline_call(self, lines: List[str]) -> str:
        """解析多行CALL语句"""
        if not lines:
            return ""
        
        # 第一行: {{result}} = CALL func_name(
        first_line = lines[0].replace("{{", "").replace("}}", "").strip()
        match = re.match(r'(\w+)\s*=\s*CALL\s+(\w+)\s*\(\s*$', first_line)
        if not match:
            warning(f"无法解析多行CALL起始行: {first_line}")
            return ""
        
        result_var = match.group(1)
        func_name = match.group(2)
        
        # 收集参数
        args = []
        for line in lines[1:]:
            cleaned = line.replace("{{", "").replace("}}", "").strip()
            # 移除尾随逗号
            if cleaned.endswith(','):
                cleaned = cleaned[:-1].strip()
            if cleaned and '=' in cleaned:
                args.append(cleaned)
        
        # 生成调用
        kwargs_str = ', '.join(args)
        if kwargs_str:
            if func_name in self.sb_func_info:
                input_vars = self.sb_func_info[func_name].get('input_vars', [])
                return f"{result_var} = await {func_name}({self._build_sb_ctx(kwargs_str, input_vars)})"
            return f"{result_var} = await invoke_function('{func_name}', {kwargs_str})"
        else:
            if func_name in self.sb_func_info:
                return f"{result_var} = await {func_name}({{}})"
            return f"{result_var} = await invoke_function('{func_name}')"
        
        body_code = '\n'.join(code_lines)
        
        return ModuleDefinition(
            name=func_name,
            inputs=external_inputs,
            outputs=final_outputs,
            body_code=body_code,
            is_async=is_async,
            original_blocks=cluster
        )
    
    def _generate_function_name(self, cluster: List[CodeBlock], module_id: int) -> str:
        """智能生成函数名"""
        # 优先使用 CALL 的函数名
        for block in cluster:
            if block.type == BlockType.CALL:
                match = re.search(r'CALL\s+(\w+)', block.code_lines[0])
                if match:
                    return f"step_{module_id}_{match.group(1)}"

        # 否则使用输出变量名（需要清理 {{}} 和确保符合 Python 标识符规范）
        for block in cluster:
            if block.outputs:
                var_name = list(block.outputs)[0]
                # 先去除 {{}}，再清理标识符
                var_name_cleaned = var_name.replace("{{", "").replace("}}", "")
                var_name_sanitized = self._sanitize_identifier(var_name_cleaned)
                return f"step_{module_id}_compute_{var_name_sanitized}"

        return f"step_{module_id}_process"
    
    def generate_orchestrator(self, modules: List[ModuleDefinition],
                             main_inputs: List[str]) -> str:
        """生成主控函数（工作流编排器） - 支持短路逻辑"""

        has_async = any(m.is_async for m in modules)
        async_kw = "async " if has_async else ""

        lines = [
            f"{async_kw}def main_workflow(input_params: dict):",
            '    """',
            '    主工作流 - 自动生成（短路逻辑）',
            '    ',
            f'    Args:',
            f'        input_params: 包含 {main_inputs} 的字典',
            '    ',
            '    Returns:',
            '        执行结果上下文',
            '    """',
            '    # 初始化上下文',
            '    ctx = input_params.copy()',
            '',
            '    # 依次尝试每个模块，找到匹配就返回',
        ]

        for i, module in enumerate(modules, 1):
            lines.append(f'    # Module {i}: {module.name}')

            # 准备参数
            args = ', '.join([f'ctx.get("{arg}")' for arg in module.inputs])

            # 根据模块的 is_async 状态决定是否使用 await
            await_kw = "await " if module.is_async else ""

            if module.outputs:
                # 检查函数实际返回多少个值
                return_statements = re.findall(r'return\s+(.+)', module.body_code)
                use_single_output = False

                if return_statements:
                    first_return = return_statements[0].strip()
                    first_return = re.sub(r'[\s;#].*$', '', first_return)
                    if ',' in first_return and not first_return.startswith('('):
                        actual_return_count = first_return.count(',') + 1
                    else:
                        actual_return_count = 1

                    if actual_return_count == 1 and len(module.outputs) > 1:
                        use_single_output = True

                # 生成带检查的调用 - 如果已匹配则短路停止
                if use_single_output:
                    out_var = module.outputs[0]
                    lines.append(f'    if not ctx.get("{out_var}"):')
                    lines.append(f'        result = {await_kw}{module.name}({args})')
                    lines.append(f'        if result is not None:')
                    lines.append(f'            ctx["{out_var}"] = result')
                else:
                    out_vars_check = ' or '.join([f'ctx.get("{v}")' for v in module.outputs])
                    lines.append(f'    if not ({out_vars_check}):')
                    lines.append(f'        result = {await_kw}{module.name}({args})')
                    lines.append(f'        if result is not None:')
                    lines.append(f'            ctx["scene"], ctx["agent"] = result')
                    lines.append(f'            return ctx')
            else:
                lines.append(f'    # 无输出的模块: {await_kw}{module.name}({args})')
                lines.append(f'    {await_kw}{module.name}({args})')

            lines.append('')

        lines.append('    return ctx')

        return '\n'.join(lines)


# ============================================================================
# 4. 主编译器入口
# ============================================================================

class WaActCompiler:
    """WaAct 编译器 - 完整流程"""

    def __init__(self):
        self.parser = PseudoCodeParser()
        self.synthesizer = ModuleSynthesizer()
        self.semantic_converter = SemanticBlockConverter()

    def compile(self, dsl_code: str,
                clustering_strategy: str = "hybrid",
                visualize: bool = False) -> Tuple[List[ModuleDefinition], str, Dict[str, Any]]:
        """
        编译 DSL 为可执行 Python 代码

        Args:
            dsl_code: Prompt 3.0 伪代码
            clustering_strategy: 聚类策略 (io_isolation/control_flow/hybrid)
            visualize: 是否生成依赖图可视化

        Returns:
            (模块列表, 主函数代码, 编译步骤详情)
        """
        info("=" * 60)
        info("WaAct Compiler v2.0 - 开始编译")
        info("=" * 60)

        # 收集步骤详情
        compile_details = {}

        # Stage 0: DSL v1.2 预处理（检测并转换为中间格式）
        info("\n[Stage 0] DSL v1.2 预处理...")
        converter = DSL12ToIntermediateConverter()

        # 【重要】在转换之前先提取原始 DSL 的 INPUTS 定义
        # 因为转换器会删除 INPUTS 等元信息
        original_spec_inputs = self._extract_spec_inputs_from_dsl(dsl_code)

        semantic_blocks = []
        is_dsl_v12 = converter.is_dsl_v12(dsl_code)
        if is_dsl_v12:
            from files.validator import DSLValidator
            validation = DSLValidator().validate(dsl_code)
            validation.raise_if_invalid()
            compile_details["dsl_structure_validation"] = {"passed": True}
            info("  📋 检测到 DSL v1.2 格式，启动语法转换器...")
            dsl_code, semantic_blocks = converter.convert(dsl_code)
            info(f"  ✅ DSL v1.2 → 中间格式转换完成, 检测到 {len(semantic_blocks)} 个 SEMANTIC_BLOCK")
        else:
            info("  📋 使用标准伪代码格式")

        # Stage 1: 词法解析
        info("\n[Stage 1] 词法解析 (Lexical Parsing)...")
        blocks = self.parser.parse(dsl_code)
        declared_input_names: Set[str] = set()
        if is_dsl_v12:
            # Check parsed executable statements, not prompt text or metadata.
            # This is name resolution, not branch-sensitive definite assignment.
            declared = set()
            for declaration in original_spec_inputs:
                match = re.match(r"(?:REQUIRED\s+|OPTIONAL\s+)?\{\{(\w+)\}\}\s*:", declaration)
                if match:
                    declared.add(match.group(1))
                    declared_input_names.add(match.group(1))
            for block in blocks:
                declared.update(block.outputs)
            unresolved = []
            for block in blocks:
                for name in sorted(block.inputs - declared):
                    unresolved.append(f"line {block.line_number}: undefined variable {name}")
            if unresolved:
                raise ValueError("DSL name resolution failed:\n" + "\n".join(unresolved))
            compile_details["dsl_name_resolution"] = {
                "passed": True,
                "scope": "declared_inputs_and_parsed_assignment_targets",
                "definite_assignment_checked": False,
            }
        info(f"✅ 解析出 {len(blocks)} 个代码块")

        # 收集 Stage 1 详情
        block_types_count = {}
        for block in blocks:
            block_type = block.type.value
            block_types_count[block_type] = block_types_count.get(block_type, 0) + 1

        compile_details['step1_parsing'] = {
            'total_blocks': len(blocks),
            'block_types': block_types_count,
            'blocks': [
                {
                    'id': block.id,
                    'type': block.type.value,
                    'line_number': block.line_number,
                    'is_async': block.is_async,
                    'inputs': list(block.inputs),
                    'outputs': list(block.outputs),
                    'code_lines': block.code_lines
                }
                for block in blocks
            ]
        }

        # Stage 2: 依赖分析
        info("\n[Stage 2] 依赖分析 (Dependency Analysis)...")
        analyzer = DependencyAnalyzer(blocks, external_inputs=declared_input_names)
        analyzer.build_graph()

        # 循环检测
        has_cycles = analyzer.detect_cycles()
        if has_cycles:
            raise ValueError("❌ 编译失败：检测到循环依赖")

        # 死代码检测
        dead_code = analyzer.find_dead_code()
        if dead_code:
            warning(f"发现 {len(dead_code)} 个死代码块: {dead_code}")

        # 拓扑排序
        topological_order = analyzer.topological_sort()

        info("✅ 依赖图构建完成")

        # 收集 Stage 2 详情
        compile_details['step2_dependency'] = {
            'has_cycles': has_cycles,
            'dead_code_count': len(dead_code),
            'dead_code_blocks': dead_code,
            'topological_order': topological_order,
            'edge_count': analyzer.graph.number_of_edges(),
            'node_count': analyzer.graph.number_of_nodes()
        }

        if visualize:
            analyzer.visualize()

        # Stage 3.5: DSL 规范化（预处理代码块）
        info("\n[Stage 3.5] DSL 规范化 (Normalization)...")
        normalizer = DSLNormalizer()
        for block in blocks:
            block.code_lines = [
                normalizer._fix_placeholders(line) 
                for line in block.code_lines
            ]
        info("  ✅ DSL 语法规范化完成")

        # Stage 3: 模块聚类
        info(f"\n[Stage 3] 模块聚类 (Strategy: {clustering_strategy})...")
        clusters = analyzer.analyze_clusters(clustering_strategy)
        info(f"✅ 拆分为 {len(clusters)} 个模块")

        # 收集 Stage 3 详情
        compile_details['step3_clustering'] = {
            'strategy': clustering_strategy,
            'total_clusters': len(clusters),
            'clusters': [
                {
                    'cluster_id': i,
                    'block_count': len(cluster),
                    'blocks': [block.id for block in cluster]
                }
                for i, cluster in enumerate(clusters, 1)
            ]
        }

        # Stage 4: 代码生成
        info("\n[Stage 4] 代码生成 (Code Synthesis)...")

        # 设置 semantic block 函数信息映射，供代码生成时判断是否直接调用
        # 格式: {func_name: {"input_vars": [...], ...}}
        sb_func_info = {}
        for sb in semantic_blocks:
            block_id = sb.get('block_id', '')
            func_name = f'sb_{block_id}' if not block_id.startswith('sb_') else block_id
            sb_func_info[func_name] = {
                'input_vars': sb.get('input_vars', []),
                'output_vars': sb.get('output_vars', [])
            }
        self.synthesizer.sb_func_info = sb_func_info

        modules = []

        for i, cluster in enumerate(clusters, 1):
            module = self.synthesizer.generate_module(cluster, i, semantic_blocks)
            modules.append(module)
            info(f"  ├─ Module {i}: {module.name} "
                  f"({'async' if module.is_async else 'sync'})")

        # 收集 Stage 4 详情
        compile_details['step4_generation'] = {
            'total_modules': len(modules),
            'async_modules': sum(1 for m in modules if m.is_async),
            'sync_modules': sum(1 for m in modules if not m.is_async),
            'modules': [
                {
                    'name': module.name,
                    'inputs': module.inputs,
                    'outputs': module.outputs,
                    'is_async': module.is_async,
                    'body_code': module.body_code,
                    'original_block_count': len(module.original_blocks)
                }
                for module in modules
            ]
        }

        # Stage 4.7: 三阶段代码修复 (需要在语法验证之前!)
        info("\n[Stage 4.7] 代码自动修复 (Auto Fix)...")
        fixer = WorkflowFixer()
        dsl_code, modules = fixer.fix(dsl_code, blocks, modules)
        info("✅ 自动修复完成")

        # Stage 4.6: SEMANTIC_BLOCK 代码生成
        if semantic_blocks:
            info(f"\n[Stage 4.6] SEMANTIC_BLOCK 代码生成 ({len(semantic_blocks)} 个块)...")
            semantic_code = self.semantic_converter.generate_semantic_blocks_header()
            for sb in semantic_blocks:
                semantic_code += self.semantic_converter.convert_semantic_block_to_code(sb)
            info("  ✅ SEMANTIC_BLOCK 代码生成完成")
        else:
            semantic_code = ""

        # Stage 4.5: 语法验证
        info("\n[Stage 4.5] 语法验证 (Syntax Validation)...")
        self._validate_generated_code(modules)

        # Stage 5: 主控生成
        info("\n[Stage 5] 主控编排 (Orchestration)...")
        # 使用 Stage 0 中提取的 original_spec_inputs，确保所有定义的变量都被包含在 main_workflow 参数中
        main_inputs = self._extract_main_inputs(blocks, original_spec_inputs)
        main_code = self.synthesizer.generate_orchestrator(modules, main_inputs)
        if semantic_code:
            main_code = semantic_code + "\n\n" + main_code
        info("✅ 主工作流生成完成")

        # 收集 Stage 5 详情
        compile_details['step5_orchestration'] = {
            'main_inputs': main_inputs,
            'input_count': len(main_inputs),
            'main_code': main_code
        }

        compile_details['semantic_blocks'] = semantic_blocks

        info("\n" + "=" * 60)
        info("编译成功！")
        info("=" * 60)

        return modules, main_code, compile_details
    
    def _extract_main_inputs(self, blocks: List[CodeBlock], spec_inputs: List[str] = None) -> List[str]:
        """提取工作流的外部输入参数
        
        Args:
            blocks: 代码块列表
            spec_inputs: DSL中定义的inputs（可选），用于补充未在BLOCK中直接使用但需要的变量
        """
        all_inputs = set()
        all_outputs = set()
        
        for block in blocks:
            all_inputs.update(block.inputs)
            all_outputs.update(block.outputs)
        
        # 外部输入 = 使用但未生产的变量
        external = all_inputs - all_outputs
        
        # 【新增】从 spec_inputs 补充定义的变量（即使未在 BLOCK 中使用）
        # 这确保了 DSL INPUTS 中定义的所有变量（如 process）都被包含在 main_workflow 参数中
        if spec_inputs:
            for inp in spec_inputs:
                # 从 "REQUIRED {{process}}: String" 或 "OPTIONAL {{manuscript_link}}: String = """ 中提取变量名
                matches = re.findall(r'\{\{(\w+)\}\}', inp)
                for var in matches:
                    external.add(var)
        
        return sorted(list(external))

    def _extract_spec_inputs_from_dsl(self, dsl_code: str) -> List[str]:
        """从 DSL 代码中提取 INPUTS 定义
        
        用于确保 DSL INPUTS 中定义的所有变量都被包含在 main_workflow 参数中，
        即使这些变量没有在 BLOCK 内部直接使用（如 process）。
        """
        inputs = []
        in_inputs_block = False
        for line in dsl_code.splitlines():
            line = line.strip()
            if line == 'INPUTS:':
                in_inputs_block = True
                continue
            if line == 'ENDINPUTS':
                in_inputs_block = False
                continue
            if in_inputs_block and ('REQUIRED' in line or 'OPTIONAL' in line or '{{' in line):
                inputs.append(line)
        return inputs

    def _extract_config_policy_from_dsl(self, dsl_code: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        """从 DSL 代码中提取 CONFIG 和 POLICY 段。

        Returns:
            (configs, policies)
            configs: {key: value} 字典
            policies: [{key, description, trigger_condition, action}] 列表
        """
        configs = {}
        policies = []

        # 解析 CONFIG 段
        in_config = False
        for line in dsl_code.splitlines():
            line = line.strip()
            if line == 'CONFIG:':
                in_config = True
                continue
            if line == 'ENDCONFIG':
                in_config = False
                continue
            if in_config and ':' in line:
                key, _, value = line.partition(':')
                configs[key.strip()] = value.strip()

        # 解析 POLICY 段
        in_policy = False
        for line in dsl_code.splitlines():
            line = line.strip()
            if line == 'POLICY:':
                in_policy = True
                continue
            if line == 'ENDPOLICY':
                in_policy = False
                continue
            if in_policy and ':' in line:
                key, _, desc = line.partition(':')
                desc = desc.strip()
                policy = {
                    'key': key.strip(),
                    'description': desc,
                    'trigger_condition': '',
                    'action': ''
                }
                # 解析触发条件和动作（格式：描述，触发条件: xxx→动作）
                if '，触发条件:' in desc:
                    desc_part, _, trigger_part = desc.partition('，触发条件:')
                    policy['description'] = desc_part
                    if '→' in trigger_part:
                        trigger, _, action = trigger_part.partition('→')
                        policy['trigger_condition'] = trigger.strip()
                        policy['action'] = action.strip()
                    else:
                        policy['trigger_condition'] = trigger_part.strip()
                elif '→' in desc:
                    desc_part, _, action = desc.partition('→')
                    policy['description'] = desc_part.strip()
                    policy['action'] = action.strip()
                policies.append(policy)

        return configs, policies

    def generate_full_code(self, modules: List[ModuleDefinition], main_code: str,
                           dsl_code: str = "") -> str:
        """生成完整的可执行 Python 代码，包含中间件装饰器（路径C）。

        Args:
            modules: 模块列表
            main_code: 主工作流代码
            dsl_code: 原始 DSL 代码（用于提取 CONFIG/POLICY）
        """
        # 从 DSL 中提取 CONFIG 和 POLICY
        configs, policies = self._extract_config_policy_from_dsl(dsl_code) if dsl_code else ({}, [])

        lines = []

        # 导入
        lines.append("# Auto-generated by WaAct Compiler v2.0")
        lines.append("# Path C: 包含中间件/装饰器")
        lines.append("")
        imports = set()
        has_async = any(m.is_async for m in modules)

        # 基础导入
        imports.add("import json")
        imports.add("from typing import Dict, Any, Optional, List")
        generated_body = "\n".join([*(m.body_code for m in modules), main_code])
        if "invoke_function(" in generated_body:
            imports.add("from llm_client import invoke_function")
        if has_async:
            imports.add("import asyncio")
            imports.add("from functools import wraps")

        # 中间件导入
        if configs or policies:
            imports.add("import time")
            imports.add("from functools import wraps")
            if has_async:
                pass  # asyncio already imported

        lines.extend(sorted(imports))
        lines.append("")

        # ================================================================
        # 路径C: 生成中间件装饰器
        # ================================================================
        middleware_lines = self._generate_middleware_decorators(configs, policies, has_async)
        if middleware_lines:
            lines.extend(middleware_lines)
            lines.append("")

        # 模块函数
        for module in modules:
            lines.append(module.body_code)
            lines.append("")

        # 主工作流（带装饰器）
        decorator_lines = self._generate_decorator_applications(configs, policies)
        lines.extend(decorator_lines)
        lines.append(main_code)

        return "\n".join(lines)

    def _generate_middleware_decorators(self, configs: Dict[str, str],
                                        policies: List[Dict[str, str]],
                                        has_async: bool) -> List[str]:
        """生成中间件装饰器代码。"""
        lines = []
        async_prefix = "async " if has_async else ""
        await_prefix = "await " if has_async else ""

        # 1. 安全检查装饰器
        security_policies = [p for p in policies if p['key'].startswith('SECURITY')]
        if security_policies:
            lines.append("# === 安全策略中间件 ===")
            lines.append("")
            lines.append("SENSITIVE_WORDS = []  # TODO: 填充敏感词列表")
            lines.append("")
            lines.append("def contains_sensitive_content(text: str) -> bool:")
            lines.append('    """检查文本是否包含敏感内容"""')
            lines.append("    return any(word in text for word in SENSITIVE_WORDS)")
            lines.append("")
            lines.append("def is_finance_query(text: str) -> bool:")
            lines.append('    """检查是否为金融类查询"""')
            lines.append("    finance_keywords = ['金融', '银行', '投资', '理财', '贷款', '基金', '股票']")
            lines.append("    return any(kw in text for kw in finance_keywords)")
            lines.append("")
            lines.append(f"def security_check(func):")
            lines.append(f'    """安全策略装饰器：敏感词过滤 + 金融查询二次确认"""')
            lines.append(f"    @wraps(func)")
            lines.append(f"    {async_prefix}def wrapper(*args, **kwargs):")
            lines.append(f'        user_input = kwargs.get("user_input", "") or (args[0] if args else "")')
            # 安全策略内容
            for pol in security_policies:
                if '敏感' in pol['description'] or '过滤' in pol['description']:
                    lines.append(f'        if contains_sensitive_content(str(user_input)):')
                    lines.append(f'            return {{"scene": -1, "agent": "安全拦截", "reason": "检测到违规内容"}}')
                if '金融' in pol['description'] or '身份' in pol['description']:
                    lines.append(f'        if is_finance_query(str(user_input)) and not kwargs.get("identity_confirmed"):')
                    lines.append(f'            return {{"scene": -2, "agent": "身份确认", "reason": "金融类查询需要二次确认"}}')
            lines.append(f"        return {await_prefix}func(*args, **kwargs)")
            lines.append(f"    return wrapper")
            lines.append("")

        # 2. 并发控制装饰器
        max_concurrency = configs.get('MAX_CONCURRENCY')
        if max_concurrency:
            lines.append("# === 并发控制中间件 ===")
            lines.append("")
            lines.append(f"def rate_limit(max_concurrency={max_concurrency}):")
            lines.append(f'    """并发控制装饰器：限制最大并发数"""')
            if has_async:
                lines.append(f"    semaphore = asyncio.Semaphore(max_concurrency)")
            lines.append(f"    def decorator(func):")
            lines.append(f"        @wraps(func)")
            lines.append(f"        {async_prefix}def wrapper(*args, **kwargs):")
            if has_async:
                lines.append(f"            async with semaphore:")
                lines.append(f"                return {await_prefix}func(*args, **kwargs)")
            else:
                lines.append(f"            return func(*args, **kwargs)")
            lines.append(f"        return wrapper")
            lines.append(f"    return decorator")
            lines.append("")

        # 3. 性能监控装饰器
        monitoring_policies = [p for p in policies if p['key'] == 'MONITORING']
        has_sla = any('响应时间' in c.get('description', '') for c in (self._extract_constraints_from_policies(policies)))
        if monitoring_policies or has_sla:
            lines.append("# === 性能监控中间件 ===")
            lines.append("")
            sla_configs = {}
            # 从 CONSTRAINTS 提取 SLA 配置
            for pol in monitoring_policies:
                desc = pol.get('description', '')
                if '响应时间' in desc:
                    sla_configs['max_total'] = 3.0
                if '向量' in desc:
                    sla_configs['vector'] = 0.5
                if '图数据库' in desc:
                    sla_configs['graph'] = 1.0
                if '代码' in desc:
                    sla_configs['code'] = 2.0
            if not sla_configs:
                sla_configs = {"max_total": 3.0}

            lines.append(f"def performance_monitor(slas=None):")
            lines.append(f'    """性能监控装饰器：SLA 检查 + 告警"""')
            lines.append(f"    slas = slas or {sla_configs}")
            lines.append(f"    def decorator(func):")
            lines.append(f"        @wraps(func)")
            lines.append(f"        {async_prefix}def wrapper(*args, **kwargs):")
            lines.append(f"            start = time.time()")
            lines.append(f"            result = {await_prefix}func(*args, **kwargs)")
            lines.append(f"            elapsed = time.time() - start")
            lines.append(f"            max_total = slas.get('max_total', 3.0)")
            lines.append(f"            if elapsed > max_total:")
            lines.append(f'                print(f"[SLA告警] 响应时间 {{elapsed:.2f}}s 超过阈值 {{max_total}}s")')
            lines.append(f"            return result")
            lines.append(f"        return wrapper")
            lines.append(f"    return decorator")
            lines.append("")

        # 4. 日志策略装饰器
        logging_policies = [p for p in policies if p['key'] == 'LOGGING']
        if logging_policies:
            lines.append("# === 日志策略中间件 ===")
            lines.append("")
            lines.append("import logging as _logging")
            lines.append("_logger = _logging.getLogger('workflow')")
            lines.append("")
            lines.append(f"def logging_middleware(func):")
            lines.append(f'    """日志策略装饰器：记录请求和响应"""')
            lines.append(f"    @wraps(func)")
            lines.append(f"    {async_prefix}def wrapper(*args, **kwargs):")
            lines.append(f'        user_input = kwargs.get("user_input", "") or (args[0] if args else "")')
            lines.append(f'        _logger.info(f"[请求] user_input={{user_input[:50]}}")')
            lines.append(f"        result = {await_prefix}func(*args, **kwargs)")
            lines.append(f'        _logger.info(f"[响应] result={{result}}")')
            lines.append(f"        return result")
            lines.append(f"    return wrapper")
            lines.append("")

        return lines

    def _generate_decorator_applications(self, configs: Dict[str, str],
                                          policies: List[Dict[str, str]]) -> List[str]:
        """生成装饰器应用代码（放在 main_workflow 函数前面）。"""
        decorator_lines = []

        # 按逆序应用装饰器（最外层最先执行）
        # 顺序: performance_monitor → rate_limit → security_check → logging

        monitoring_policies = [p for p in policies if p['key'] == 'MONITORING']
        has_sla = bool(monitoring_policies)
        if has_sla:
            decorator_lines.append("@performance_monitor()")

        if configs.get('MAX_CONCURRENCY'):
            decorator_lines.append("@rate_limit()")

        security_policies = [p for p in policies if p['key'].startswith('SECURITY')]
        if security_policies:
            decorator_lines.append("@security_check")

        logging_policies = [p for p in policies if p['key'] == 'LOGGING']
        if logging_policies:
            decorator_lines.append("@logging_middleware")

        return decorator_lines

    @staticmethod
    def _extract_constraints_from_policies(policies):
        """辅助：从 policies 提取带 description 的策略（用于 SLA 检测）。"""
        return policies

    def parse_exclusion_rules(self, constraints: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """解析 CONSTRAINTS 中的 EXCLUSION 规则

        Args:
            constraints: DSL CONSTRAINTS 中的规则列表，如：
                ['OUTPUT: 仅输出一个数字（1~4）',
                 'EXCLUSION: 当 type 为图文时不能选择视频相关智能体']

        Returns:
            解析后的规则字典，如：
                {
                    'type=图文': {'forbidden': ['2', '4']},  # 2=视频执行, 4=视频修改
                    'type=视频': {'forbidden': ['1', '3']}   # 1=图文执行, 3=图文修改
                }
        """
        exclusion_map = {}

        for c in constraints:
            if 'EXCLUSION' not in c:
                continue

            # 解析 "当 type 为图文时不能选择视频相关智能体"
            if 'type 为图文' in c or 'type=图文' in c or 'type等于图文' in c:
                # 图文类型禁止选择视频智能体（scene 2 和 4）
                forbidden = []
                if '2' in c or '视频执行' in c or '视频相关' in c:
                    forbidden.append('2')
                if '4' in c or '视频修改' in c or '视频相关' in c:
                    forbidden.append('4')
                if forbidden:
                    exclusion_map['type=图文'] = {'forbidden': forbidden}

            # 解析 "当 type 为视频时不能选择图文相关智能体"
            elif 'type 为视频' in c or 'type=视频' in c or 'type等于视频' in c:
                # 视频类型禁止选择图文智能体（scene 1 和 3）
                forbidden = []
                if '1' in c or '图文执行' in c or '图文相关' in c:
                    forbidden.append('1')
                if '3' in c or '图文修改' in c or '图文相关' in c:
                    forbidden.append('3')
                if forbidden:
                    exclusion_map['type=视频'] = {'forbidden': forbidden}

        return exclusion_map

    def _validate_generated_code(self, modules: List[ModuleDefinition]) -> None:
        """验证生成的代码语法正确性（AST 解析）"""
        import ast
        import io
        import sys
        
        errors = []
        warnings_list = []
        
        for module in modules:
            try:
                # 尝试解析生成的代码为 AST
                ast.parse(module.body_code)
                debug(f"  ✅ {module.name}: 语法正确")
            except SyntaxError as e:
                error_msg = f"  ❌ {module.name}: 语法错误 - {e.msg} (行 {e.lineno})"
                errors.append(error_msg)
                error(error_msg)
                # 打印出错的代码用于调试
                error(f"\n--- {module.name} 代码内容 ---")
                error(module.body_code)
                error(f"--- {module.name} 代码结束 ---\n")
                continue
            
            # 检查是否有残留的 DSL 关键字
            dsl_keywords = ['CALL', 'FOR round IN', '{{', '}}', 'ENDIF', 'ENDFOR']
            for keyword in dsl_keywords:
                if keyword in module.body_code:
                    warning_msg = f"  ⚠️ {module.name}: 发现残留的 DSL 关键字 '{keyword}'"
                    warnings_list.append(warning_msg)
                    warning(warning_msg)
            
            # 检查是否所有 CALL 语句都已转换
            if 'CALL' in module.body_code and 'await invoke_function' not in module.body_code:
                error_msg = f"  ❌ {module.name}: CALL 语句未正确转换"
                errors.append(error_msg)
                error(error_msg)
            
            # 检查异步标记一致性
            if module.is_async and 'await' not in module.body_code and 'async def' not in module.body_code:
                warning_msg = f"  ⚠️ {module.name}: 标记为 async 但未使用 await"
                warnings_list.append(warning_msg)
                warning(warning_msg)
            elif not module.is_async and 'await' in module.body_code:
                warning_msg = f"  ⚠️ {module.name}: 标记为 sync 但包含 await"
                warnings_list.append(warning_msg)
                warning(warning_msg)
        
        # 汇总报告
        if errors:
            raise ValueError(
                f"语法验证失败，发现 {len(errors)} 个错误：\n" + "\n".join(errors)
            )
        
        if warnings_list:
            info(f"发现 {len(warnings_list)} 个警告，请检查代码质量")
        
        info("✅ 语法验证通过")
    
    def export_to_file(self, modules: List[ModuleDefinition], 
                       main_code: str, 
                       output_path: str = "agent_workflow.py"):
        """导出为完整的 Python 文件"""
        
        lines = [
            '"""',
            'Auto-generated by WaAct Compiler',
            'DO NOT EDIT THIS FILE MANUALLY',
            '"""',
            '',
            'from typing import Dict, Any',
            'from llm_client import invoke_function  # 需要实现 LLM 客户端',
            '',
            ''
        ]
        
        # 添加所有模块
        for module in modules:
            lines.append(module.to_python())
            lines.append('\n')
        
        # 添加主函数
        lines.append(main_code)
        lines.append('\n\n')
        
        # 添加测试入口
        lines.extend([
            'if __name__ == "__main__":',
            '    import asyncio',
            '    ',
            '    # 测试输入',
            '    test_input = {',
            '        # TODO: 填充实际参数',
            '    }',
            '    ',
            '    result = asyncio.run(main_workflow(test_input))',
            '    from logger import info',
            '    info("执行结果:", result)'
        ])
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        info(f"\n✅ 代码已导出至: {output_path}")


# ============================================================================
# 5. 测试用例
# ============================================================================

if __name__ == "__main__":
    # 测试 DSL
    test_dsl = """
# 用户等级判断与内容生成工作流

IF {{experience_years}} > 3
    {{level}} = "Expert"
    {{complexity}} = 0.9
ELSE
    {{level}} = "Novice"
    {{complexity}} = 0.3
ENDIF

{{outline}} = CALL generate_outline({{level}}, {{complexity}})
{{content}} = CALL expand_content({{outline}})
{{final_doc}} = CALL polish_text({{content}}, {{level}})
"""
    
    # 执行编译
    compiler = WaActCompiler()
    modules, main_code = compiler.compile(
        test_dsl, 
        clustering_strategy="hybrid",
        visualize=False  # 设为 True 需要 matplotlib
    )
    
    # 打印结果
    info("\n" + "=" * 60)
    info("生成的模块代码:")
    info("=" * 60)
    
    for i, module in enumerate(modules, 1):
        info(f"\n{'─' * 60}")
        info(f"Module {i}: {module.name}")
        info('─' * 60)
        info(module.to_python())
    
    info("\n" + "=" * 60)
    info("主工作流代码:")
    info("=" * 60)
    info(main_code)
    
    # 导出文件
    compiler.export_to_file(modules, main_code, "generated_workflow.py")



