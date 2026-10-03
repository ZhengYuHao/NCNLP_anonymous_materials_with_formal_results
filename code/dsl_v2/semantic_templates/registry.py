"""Prompt Template Registry for Semantic Blocks"""
from typing import Dict, Optional


CLASSIFICATION_TEMPLATE = """你是{role}。
任务：{task_description}
输入：{input_description}
输出：必须为有效 JSON 格式

{task_guidelines}

{few_shot_examples}
"""

GENERATION_TEMPLATE = """你是{role}。
任务：生成{output_name}
要求：
{requirements}

输入信息：
{input_content}

输出格式（必须是有效 JSON）：
{output_format}
"""

SCORING_TEMPLATE = """你是{role}。
任务：对以下输入进行多维度评分
评分维度：
{scoring_dimensions}

输入信息：
{input_content}

输出格式（必须是有效 JSON）：
{output_format}
"""

SIMILARITY_TEMPLATE = """你是{role}。
任务：判断两个文本的语义相似度
文本A：{text_a}
文本B：{text_b}

输出格式（必须是有效 JSON）：
{{"similarity": 0.0到1.0之间的浮点数}}
"""

EXTRACTION_TEMPLATE = """你是{role}。
任务：从输入中抽取以下信息：
抽取字段：{extract_fields}

输入信息：
{input_content}

输出格式（必须是有效 JSON）：
{output_format}
"""


class PromptTemplateRegistry:
    """语义 Block Prompt 模板注册表"""

    def __init__(self):
        self.templates = {
            "classification": CLASSIFICATION_TEMPLATE,
            "generation": GENERATION_TEMPLATE,
            "scoring": SCORING_TEMPLATE,
            "similarity": SIMILARITY_TEMPLATE,
            "extraction": EXTRACTION_TEMPLATE,
        }

    def get_template(self, task: str) -> str:
        """获取指定任务的模板"""
        return self.templates.get(task, "")

    def build_prompt(
        self,
        task: str,
        template: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """构建 Prompt

        Args:
            task: 任务类型
            template: 自定义模板（可选）
            variables: 模板变量
            **kwargs: 其他模板参数

        Returns:
            格式化后的 prompt 字符串
        """
        if template:
            prompt = template
        else:
            prompt = self.get_template(task)

        if variables:
            for key, value in variables.items():
                placeholder = f"{{{key}}}"
                prompt = prompt.replace(placeholder, str(value))

        return prompt


def test_prompt_registry():
    """测试 PromptRegistry"""
    registry = PromptTemplateRegistry()

    template = registry.get_template("classification")
    assert "classification" in template.lower() or "你是" in template
    print("✅ test_get_template passed")

    prompt = registry.build_prompt(
        task="classification",
        variables={"role": "测试角色", "task_description": "分类任务"}
    )
    assert "测试角色" in prompt
    assert "分类任务" in prompt
    print("✅ test_build_prompt passed")

    print("\n🎉 All PromptRegistry tests passed!")


if __name__ == "__main__":
    test_prompt_registry()
