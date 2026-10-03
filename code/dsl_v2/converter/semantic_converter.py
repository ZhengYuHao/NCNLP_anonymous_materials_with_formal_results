"""SemanticBlockConverter: SEMANTIC_BLOCK → async Python 函数"""
import json
from typing import Dict, Any, List
from dsl_v2.parser.ast_nodes import (
    SemanticBlockNode,
    InputVarNode,
    OutputVarNode,
    CallSemanticNode,
)


class SemanticBlockConverter:
    """将 SEMANTIC_BLOCK 转换为 async Python 函数"""

    TASK_TEMPLATES = {
        "classification": "你是{role}。基于输入判断分类。输出JSON格式。",
        "generation": "你是{role}。基于输入生成内容。输出JSON格式。",
        "scoring": "你是{role}。基于输入进行多维度评分。输出JSON格式。",
        "similarity": "你是{role}。判断输入文本的语义相似度。输出JSON格式。",
        "extraction": "你是{role}。从输入中抽取信息。输出JSON格式。",
    }

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def convert_semantic_block(self, node: SemanticBlockNode) -> str:
        """将 SemanticBlockNode 转换为 Python 代码"""
        if node.task == "classification":
            return self._generate_classification_function(node)
        elif node.task == "generation":
            return self._generate_generation_function(node)
        elif node.task == "scoring":
            return self._generate_scoring_function(node)
        elif node.task == "similarity":
            return self._generate_similarity_function(node)
        elif node.task == "extraction":
            return self._generate_extraction_function(node)
        else:
            return self._generate_classification_function(node)

    def _generate_classification_function(self, node: SemanticBlockNode) -> str:
        """生成分类任务函数"""
        func_name = node.block_id if node.block_id.startswith("sb_") else f"sb_{node.block_id}"

        input_format = self._format_input_vars(node.input_vars)
        output_format = self._generate_output_format(node.output_vars)

        prompt = node.prompt_template or self.TASK_TEMPLATES.get("classification", "").format(role=node.name)

        return f"""
async def {func_name}(ctx: dict) -> dict:
    '''
    SEMANTIC_BLOCK: {node.block_id}
    TASK: {node.task}
    MODEL: {node.model}
    '''
    input_data = {input_format}

    system_prompt = {repr(prompt)}
    user_prompt = json.dumps(input_data, ensure_ascii=False, indent=2)

    messages = [
        {{"role": "system", "content": system_prompt}},
        {{"role": "user", "content": user_prompt}}
    ]

    if self.llm_client is None:
        raise RuntimeError("LLM client not configured")

    response = await self.llm_client.chat.completions.create(
        model="{node.model}",
        temperature={node.temperature},
        max_tokens={node.max_tokens},
        messages=messages
    )

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {{"error": "JSON parse failed", "raw": response.content}}

    {self._format_output_assignment(node.output_vars)}

    return ctx
"""

    def _generate_generation_function(self, node: SemanticBlockNode) -> str:
        """生成任务函数（generation 使用较高的 temperature）"""
        func_name = node.block_id if node.block_id.startswith("sb_") else f"sb_{node.block_id}"

        input_format = self._format_input_vars(node.input_vars)
        output_format = self._generate_output_format(node.output_vars)

        prompt = node.prompt_template or self.TASK_TEMPLATES.get("generation", "").format(role=node.name)

        temperature = max(node.temperature, 0.5)

        return f"""
async def {func_name}(ctx: dict) -> dict:
    '''
    SEMANTIC_BLOCK: {node.block_id}
    TASK: {node.task}
    MODEL: {node.model}
    '''
    input_data = {input_format}

    system_prompt = {repr(prompt)}
    user_prompt = json.dumps(input_data, ensure_ascii=False, indent=2)

    messages = [
        {{"role": "system", "content": system_prompt}},
        {{"role": "user", "content": user_prompt}}
    ]

    if self.llm_client is None:
        raise RuntimeError("LLM client not configured")

    response = await self.llm_client.chat.completions.create(
        model="{node.model}",
        temperature={temperature},
        max_tokens={node.max_tokens},
        messages=messages
    )

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {{"error": "JSON parse failed", "raw": response.content}}

    {self._format_output_assignment(node.output_vars)}

    return ctx
"""

    def _generate_scoring_function(self, node: SemanticBlockNode) -> str:
        """生成评分任务函数"""
        func_name = node.block_id if node.block_id.startswith("sb_") else f"sb_{node.block_id}"

        input_format = self._format_input_vars(node.input_vars)

        prompt = node.prompt_template or self.TASK_TEMPLATES.get("scoring", "").format(role=node.name)

        return f"""
async def {func_name}(ctx: dict) -> dict:
    '''
    SEMANTIC_BLOCK: {node.block_id}
    TASK: {node.task}
    MODEL: {node.model}
    '''
    input_data = {input_format}

    system_prompt = {repr(prompt)}
    user_prompt = json.dumps(input_data, ensure_ascii=False, indent=2)

    messages = [
        {{"role": "system", "content": system_prompt}},
        {{"role": "user", "content": user_prompt}}
    ]

    if self.llm_client is None:
        raise RuntimeError("LLM client not configured")

    response = await self.llm_client.chat.completions.create(
        model="{node.model}",
        temperature={node.temperature},
        max_tokens={node.max_tokens},
        messages=messages
    )

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {{"error": "JSON parse failed", "raw": response.content}}

    {self._format_output_assignment(node.output_vars)}

    return ctx
"""

    def _generate_similarity_function(self, node: SemanticBlockNode) -> str:
        """生成相似度任务函数"""
        func_name = node.block_id if node.block_id.startswith("sb_") else f"sb_{node.block_id}"

        input_format = self._format_input_vars(node.input_vars)

        prompt = node.prompt_template or self.TASK_TEMPLATES.get("similarity", "").format(role=node.name)

        return f"""
async def {func_name}(ctx: dict) -> dict:
    '''
    SEMANTIC_BLOCK: {node.block_id}
    TASK: {node.task}
    MODEL: {node.model}
    '''
    input_data = {input_format}

    system_prompt = {repr(prompt)}
    user_prompt = json.dumps(input_data, ensure_ascii=False, indent=2)

    messages = [
        {{"role": "system", "content": system_prompt}},
        {{"role": "user", "content": user_prompt}}
    ]

    if self.llm_client is None:
        raise RuntimeError("LLM client not configured")

    response = await self.llm_client.chat.completions.create(
        model="{node.model}",
        temperature={node.temperature},
        max_tokens={node.max_tokens},
        messages=messages
    )

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {{"error": "JSON parse failed", "raw": response.content}}

    {self._format_output_assignment(node.output_vars)}

    return ctx
"""

    def _generate_extraction_function(self, node: SemanticBlockNode) -> str:
        """生成信息抽取任务函数"""
        return self._generate_classification_function(node)

    def _format_input_vars(self, input_vars: List[InputVarNode]) -> str:
        """格式化输入变量为 dict 字面量"""
        if not input_vars:
            return "{}"

        parts = []
        for var in input_vars:
            parts.append(f'        "{var.name}": ctx.get("{var.name}")')
        return "{\n" + ",\n".join(parts) + "\n    }"

    def _generate_output_format(self, output_vars: List[OutputVarNode]) -> str:
        """生成输出格式说明"""
        fields = []
        for var in output_vars:
            if var.var_type == "Float":
                fields.append(f'"{var.name}": 0.0')
            elif var.var_type == "String":
                fields.append(f'"{var.name}": "..."')
            elif var.var_type == "Boolean":
                fields.append(f'"{var.name}": true')
            elif var.var_type == "Map":
                fields.append(f'"{var.name}": {{}}')
            elif var.var_type == "Integer":
                fields.append(f'"{var.name}": 0')
            else:
                fields.append(f'"{var.name}": null')
        return "{" + ", ".join(fields) + "}"

    def _format_output_assignment(self, output_vars: List[OutputVarNode]) -> str:
        """生成输出变量赋值到 ctx 的代码"""
        lines = []
        for var in output_vars:
            if var.var_type == "Map":
                lines.append(f'    ctx["{var.name}"] = result.get("{var.name}", {{}})')
            else:
                lines.append(f'    ctx["{var.name}"] = result.get("{var.name}")')
        return "\n".join(lines) if lines else "    pass"

    def convert_call_node(self, node: CallSemanticNode) -> str:
        """生成 CALL sb_xxx 的调用代码"""
        func_name = node.target_block if node.target_block.startswith("sb_") else f"sb_{node.target_block}"

        input_args = []
        for formal, actual in node.input_mapping:
            clean_actual = actual.replace("{{", "").replace("}}", "")
            input_args.append(f'"{formal}": {clean_actual}')

        input_dict = "{\n        " + ",\n        ".join(input_args) + "\n    }"

        return f"""
    # CALL {node.target_block}
    call_input = {input_dict}
    call_ctx = await {func_name}(call_input)
    ctx.update(call_ctx)
"""

    def generate_header(self) -> str:
        """生成文件头部（import 语句等）"""
        return '''"""
Auto-generated by WaAct Compiler v2 (SEMANTIC_BLOCK support)
"""

import json
from typing import Dict, Any
'''

    def generate_workflow_function(self, semantic_blocks: List[str], workflow_steps: List[str]) -> str:
        """生成主工作流函数"""
        steps_code = []
        for i, step in enumerate(workflow_steps):
            steps_code.append(f"    # Step {i+1}: {step}")

        steps_code.append("\n    return ctx")

        return f"""
async def main_workflow(input_params: dict):
    \"\"\"
    主工作流 - 自动生成
    \"\"\"
    ctx = input_params.copy()

{chr(10).join(steps_code)}
"""

    def generate_semantic_blocks_header(self) -> str:
        """生成 SEMANTIC_BLOCK 模块的头部"""
        return '''# ============================================================================
# SEMANTIC_BLOCK: LLM 语义理解模块
# ============================================================================

import json
from typing import Dict, Any
from types import SimpleNamespace

LLM_CLIENT = None  # 全局 LLM 客户端，需在使用前设置

'''

    def convert_semantic_block_to_code(self, sb_data: Dict[str, Any]) -> str:
        """将 SEMANTIC_BLOCK 数据字典转换为 Python 代码

        Args:
            sb_data: 从 DSL12Parser 解析出的 semantic_block 数据字典
        """
        block_id = sb_data.get('block_id', '')
        description = sb_data.get('description', '')
        name = sb_data.get('name', block_id)
        model = sb_data.get('model', 'gpt-5.4')
        task = sb_data.get('task', 'classification')
        temperature = sb_data.get('temperature', 0.1)
        max_tokens = sb_data.get('max_tokens', 2048)
        prompt = sb_data.get('prompt', '')
        input_vars = sb_data.get('input_vars', [])
        output_vars = sb_data.get('output_vars', [])

        func_name = f"sb_{block_id}" if not block_id.startswith('sb_') else block_id

        input_format = self._format_input_vars_from_dict(input_vars)
        output_assignment = self._format_output_assignment_from_dict(output_vars)
        return_value = self._format_return_value(output_vars)

        template = '''
async def {func_name}(ctx: dict) -> dict:
    """
    SEMANTIC_BLOCK: {block_id}
    描述: {description}
    TASK: {task}
    MODEL: {model}
    """
    input_data = {input_format}

    system_prompt = {prompt_repr}
    user_prompt = json.dumps(input_data, ensure_ascii=False, indent=2)

    messages = [
        {{"role": "system", "content": system_prompt}},
        {{"role": "user", "content": user_prompt}}
    ]

    if LLM_CLIENT is None:
        raise RuntimeError("LLM client not configured. Set global LLM_CLIENT.")

    response = await LLM_CLIENT.chat.completions.create(
        model="{model}",
        temperature={temperature},
        max_tokens={max_tokens},
        messages=messages
    )

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {{"error": "JSON parse failed", "raw": response.content}}

{output_assignment}

    {return_value}

'''
        return template.format(
            func_name=func_name,
            block_id=block_id,
            description=description,
            task=task,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            input_format=input_format,
            output_assignment=output_assignment,
            return_value=return_value,
            prompt_repr=repr(prompt)
        )

    def _format_input_vars_from_dict(self, input_vars: List[Dict[str, str]]) -> str:
        """格式化输入变量为 dict 字面量"""
        if not input_vars:
            return "{}"

        parts = []
        for var in input_vars:
            var_name = var.get('name', '')
            parts.append(f'        "{var_name}": ctx.get("{var_name}")')
        return "{\n" + ",\n".join(parts) + "\n    }"

    def _format_output_assignment_from_dict(self, output_vars: List[Dict[str, str]]) -> str:
        """生成输出变量赋值到 ctx 的代码"""
        lines = []
        for var in output_vars:
            var_name = var.get('name', '')
            lines.append(f'    ctx["{var_name}"] = result.get("{var_name}")')
        return "\n".join(lines) if lines else "    pass"

    def _format_return_value(self, output_vars: List[Dict[str, str]]) -> str:
        """生成返回值代码"""
        if not output_vars:
            return "return None"
        if len(output_vars) == 1:
            var_name = output_vars[0].get('name', '')
            return f"return result.get('{var_name}')"
        # DSL 后续步骤通过 result.field 引用多输出结果。
        var_names = [v.get('name', '') for v in output_vars]
        mapping = ", ".join([f"'{n}': result.get('{n}')" for n in var_names])
        return f"return SimpleNamespace(**{{{mapping}}})"
