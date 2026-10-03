"""SEMANTIC_BLOCK 语法解析器"""
import re
from typing import List, Tuple, Optional
from dsl_v2.parser.ast_nodes import (
    SemanticBlockNode,
    InputVarNode,
    OutputVarNode,
    CallSemanticNode,
)


class DSL12SemanticParser:
    """解析 SEMANTIC_BLOCK 语法"""

    TASK_TYPES = {"classification", "generation", "scoring", "similarity", "extraction"}

    def parse_semantic_block(self, lines: List[str], start_idx: int) -> Tuple[SemanticBlockNode, int]:
        """解析 SEMANTIC_BLOCK ... ENDSEMANTIC_BLOCK"""
        line = lines[start_idx].strip()

        block_match = re.match(r'SEMANTIC_BLOCK\s+(\w+)\s+"([^"]+)"', line)
        if not block_match:
            raise ValueError(f"Invalid SEMANTIC_BLOCK header: {line}")

        block_id = block_match.group(1)
        description = block_match.group(2)

        metadata = {
            "name": block_id,
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 2048,
            "task": "classification",
        }

        input_vars = []
        output_vars = []
        prompt_template = ""

        i = start_idx + 1
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("ENDSEMANTIC_BLOCK"):
                break

            if line.startswith("NAME "):
                metadata["name"] = line[5:].strip().strip('"')

            elif line.startswith("MODEL "):
                metadata["model"] = line[6:].strip()

            elif line.startswith("TASK "):
                task = line[5:].strip()
                if task not in self.TASK_TYPES:
                    raise ValueError(f"Invalid TASK: {task}. Must be one of {self.TASK_TYPES}")
                metadata["task"] = task

            elif line.startswith("TEMPERATURE "):
                try:
                    metadata["temperature"] = float(line[12:].strip())
                except ValueError:
                    metadata["temperature"] = 0.1

            elif line.startswith("MAX_TOKENS "):
                try:
                    metadata["max_tokens"] = int(line[11:].strip())
                except ValueError:
                    metadata["max_tokens"] = 2048

            elif line.startswith("INPUT:"):
                input_vars, i = self._parse_io_block(lines, i, "INPUT:")

            elif line.startswith("OUTPUT:"):
                output_vars, i = self._parse_io_block(lines, i, "OUTPUT:")

            elif line.startswith("PROMPT"):
                prompt_template, i = self._parse_prompt(lines, i)

            i += 1

        return SemanticBlockNode(
            block_id=block_id,
            description=description,
            name=metadata["name"],
            model=metadata["model"],
            task=metadata["task"],
            temperature=metadata["temperature"],
            max_tokens=metadata["max_tokens"],
            input_vars=input_vars,
            output_vars=output_vars,
            prompt_template=prompt_template,
        ), i

    def _parse_io_block(self, lines: List[str], start_idx: int, marker: str) -> Tuple[List, int]:
        """解析 INPUT: 或 OUTPUT: 块"""
        vars = []
        i = start_idx + 1

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            if line == "INPUT:" or line == "OUTPUT:":
                break

            if marker == "OUTPUT:" and line.startswith("dimension_scores:"):
                vars.append(OutputVarNode(name="dimension_scores", var_type="Map"))
                i += 1
                continue

            var_match = re.match(r'(\w+):\s*(\w+)', line)
            if var_match:
                var_name = var_match.group(1)
                var_type = var_match.group(2)
                vars.append(OutputVarNode(name=var_name, var_type=var_type) if marker == "OUTPUT:" else InputVarNode(name=var_name, var_type=var_type))
            elif line.startswith("PROMPT") or line.startswith('"""'):
                i -= 1
                break

            i += 1

        return vars, i - 1

    def _parse_prompt(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """解析 PROMPT 块"""
        line = lines[start_idx].strip()

        if line.startswith('PROMPT "'):
            if '"""' in line:
                prompt_match = re.match(r'PROMPT\s*"""(.*)"""', line, re.DOTALL)
                if prompt_match and prompt_match.group(1).strip():
                    return prompt_match.group(1).strip(), start_idx

            prompt_match = re.match(r'PROMPT\s*"([^"]*)"', line)
            if prompt_match:
                return prompt_match.group(1).strip(), start_idx

            prompt_lines = []
            i = start_idx + 1
            while i < len(lines):
                line = lines[i].strip()
                if line == '"""':
                    break
                prompt_lines.append(line)
                i += 1
            result = "\n".join(prompt_lines)
            return result, i

        return "", start_idx

    def parse_call_statement(self, line: str) -> CallSemanticNode:
        """解析 CALL sb_xxx INPUT {...} OUTPUT {...}"""
        line = line.strip()

        call_match = re.match(r'CALL\s+(\w+)\s+INPUT\s*\{(.+)\}\s+OUTPUT\s*\{(.+)\}', line)
        if not call_match:
            raise ValueError(f"Invalid CALL statement: {line}")

        target_block = call_match.group(1)
        input_str = call_match.group(2)
        output_str = call_match.group(3)

        input_mapping = []
        for item in input_str.split(','):
            item = item.strip()
            if ':' in item:
                key, value = item.split(':', 1)
                input_mapping.append((key.strip(), value.strip()))

        output_mapping = []
        for item in output_str.split(','):
            item = item.strip()
            if ':' in item:
                key, value = item.split(':', 1)
                output_mapping.append((key.strip(), value.strip()))

        return CallSemanticNode(
            target_block=target_block,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
        )

    def parse_source(self, source: str) -> List[SemanticBlockNode]:
        """解析完整的 DSL 源码，提取所有 SEMANTIC_BLOCK"""
        lines = source.split('\n')
        semantic_blocks = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("SEMANTIC_BLOCK "):
                try:
                    node, end_idx = self.parse_semantic_block(lines, i)
                    semantic_blocks.append(node)
                    i = end_idx + 1
                except Exception as e:
                    raise ValueError(f"Failed to parse SEMANTIC_BLOCK at line {i}: {e}")
            else:
                i += 1

        return semantic_blocks
