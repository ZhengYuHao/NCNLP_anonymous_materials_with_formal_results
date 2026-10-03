"""Tests for AST nodes"""
import sys
sys.path.insert(0, '/mnt/e/pyProject/prompt3.0')

from dsl_v2.parser.ast_nodes import (
    InputVarNode,
    OutputVarNode,
    SemanticBlockNode,
    CallSemanticNode,
)

def test_semantic_block_node_creation():
    """测试 SemanticBlockNode 创建"""
    node = SemanticBlockNode(
        block_id="sb_analyze",
        description="商单匹配分析",
        name="sb_analyze",
        model="gpt-4o",
        task="classification",
        input_vars=[InputVarNode(name="blogger_info", var_type="BloggerInfo")],
        output_vars=[OutputVarNode(name="is_match", var_type="Boolean")],
        prompt_template="判断是否为优质商单推荐"
    )
    assert node.block_id == "sb_analyze", f"Expected sb_analyze, got {node.block_id}"
    assert node.task == "classification", f"Expected classification, got {node.task}"
    assert node.model == "gpt-4o", f"Expected gpt-4o, got {node.model}"
    print("✅ test_semantic_block_node_creation passed")

def test_input_var_node():
    """测试 InputVarNode"""
    var = InputVarNode(name="user_input", var_type="String")
    assert var.name == "user_input"
    assert var.var_type == "String"
    print("✅ test_input_var_node passed")

def test_output_var_node():
    """测试 OutputVarNode"""
    var = OutputVarNode(name="intent", var_type="String")
    assert var.name == "intent"
    assert var.var_type == "String"
    print("✅ test_output_var_node passed")

def test_call_semantic_node():
    """测试 CallSemanticNode"""
    node = CallSemanticNode(
        target_block="sb_analyze",
        input_mapping=[("blogger_info", "{{blogger_info}}")],
        output_mapping=[("is_match", "{{is_match}}")]
    )
    assert node.target_block == "sb_analyze", f"Expected sb_analyze, got {node.target_block}"
    print("✅ test_call_semantic_node passed")

def test_nested_output_vars():
    """测试嵌套输出变量（Map 类型）"""
    node = SemanticBlockNode(
        block_id="sb_score",
        description="多维度评分",
        name="sb_score",
        model="gpt-4o",
        task="scoring",
        input_vars=[],
        output_vars=[
            OutputVarNode(name="dimension_scores", var_type="Map"),
            OutputVarNode(name="confidence", var_type="Float")
        ],
        prompt_template=""
    )
    assert len(node.output_vars) == 2
    assert node.output_vars[0].var_type == "Map"
    assert node.output_vars[0].name == "dimension_scores"
    print("✅ test_nested_output_vars passed")

def test_default_values():
    """测试默认参数"""
    node = SemanticBlockNode(
        block_id="sb_test",
        description="测试",
        name="sb_test",
        model="gpt-4o",
        task="classification",
        input_vars=[],
        output_vars=[],
        prompt_template=""
    )
    assert node.temperature == 0.1
    assert node.max_tokens == 2048
    print("✅ test_default_values passed")

if __name__ == "__main__":
    test_semantic_block_node_creation()
    test_input_var_node()
    test_output_var_node()
    test_call_semantic_node()
    test_nested_output_vars()
    test_default_values()
    print("\n🎉 All AST node tests passed!")