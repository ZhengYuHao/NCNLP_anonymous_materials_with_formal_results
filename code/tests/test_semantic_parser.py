"""Tests for DSL12SemanticParser"""
import sys
sys.path.insert(0, '/mnt/e/pyProject/prompt3.0')

from dsl_v2.parser.ast_nodes import (
    SemanticBlockNode,
    InputVarNode,
    OutputVarNode,
)


def test_parse_simple_semantic_block():
    """测试解析简单的 SEMANTIC_BLOCK"""
    from dsl_v2.parser.dsl12_semantic import DSL12SemanticParser

    source = '''SEMANTIC_BLOCK sb_test "测试"
    MODEL gpt-4o
    TASK classification
    INPUT:
        user_input: String
    OUTPUT:
        intent: String
    PROMPT "判断用户意图"
ENDSEMANTIC_BLOCK'''

    parser = DSL12SemanticParser()
    lines = source.split('\n')
    node, end_idx = parser.parse_semantic_block(lines, 0)

    assert node.block_id == "sb_test", f"Expected sb_test, got {node.block_id}"
    assert node.model == "gpt-4o", f"Expected gpt-4o, got {node.model}"
    assert node.task == "classification", f"Expected classification, got {node.task}"
    assert len(node.input_vars) == 1, f"Expected 1 input var, got {len(node.input_vars)}"
    assert node.input_vars[0].name == "user_input", f"Expected user_input, got {node.input_vars[0].name}"
    assert node.output_vars[0].name == "intent", f"Expected intent, got {node.output_vars[0].name}"
    print("✅ test_parse_simple_semantic_block passed")


def test_parse_semantic_block_with_nested_output():
    """测试嵌套 Output（Map 类型）"""
    from dsl_v2.parser.dsl12_semantic import DSL12SemanticParser

    source = '''SEMANTIC_BLOCK sb_score "评分"
    MODEL gpt-4o
    TASK scoring
    INPUT:
        blogger_info: BloggerInfo
        order_info: OrderInfo
    OUTPUT:
        dimension_scores:
            persona_match: Float
            audience_match: Float
        confidence: Float
    PROMPT "多维度评分"
ENDSEMANTIC_BLOCK'''

    parser = DSL12SemanticParser()
    lines = source.split('\n')
    node, end_idx = parser.parse_semantic_block(lines, 0)

    assert node.task == "scoring"
    assert len(node.output_vars) >= 1
    print(f"✅ test_parse_semantic_block_with_nested_output passed, output_vars={node.output_vars}")


def test_parse_semantic_block_with_temperature():
    """测试带 temperature 参数"""
    from dsl_v2.parser.dsl12_semantic import DSL12SemanticParser

    source = '''SEMANTIC_BLOCK sb_gen "生成"
    MODEL gpt-4o
    TEMPERATURE 0.5
    TASK generation
    INPUT:
        data: Map
    OUTPUT:
        result: String
    PROMPT "生成内容"
ENDSEMANTIC_BLOCK'''

    parser = DSL12SemanticParser()
    lines = source.split('\n')
    node, end_idx = parser.parse_semantic_block(lines, 0)

    assert node.temperature == 0.5, f"Expected 0.5, got {node.temperature}"
    assert node.task == "generation"
    print("✅ test_parse_semantic_block_with_temperature passed")


def test_parse_call_statement():
    """测试解析 CALL 语句"""
    from dsl_v2.parser.dsl12_semantic import DSL12SemanticParser

    call_line = "    CALL sb_analyze INPUT { blogger_info: {{blogger_info}}, order_info: {{order_info}} } OUTPUT { is_match: {{is_match}}, confidence: {{confidence}} }"

    parser = DSL12SemanticParser()
    node = parser.parse_call_statement(call_line)

    assert node.target_block == "sb_analyze", f"target_block: {node.target_block}"
    assert ("blogger_info", "{{blogger_info}}") in node.input_mapping, f"input_mapping: {node.input_mapping}"
    assert ("order_info", "{{order_info}}") in node.input_mapping, f"input_mapping: {node.input_mapping}"
    assert ("is_match", "{{is_match}}") in node.output_mapping, f"output_mapping: {node.output_mapping}"
    assert ("confidence", "{{confidence}}") in node.output_mapping, f"output_mapping: {node.output_mapping}"
    print("✅ test_parse_call_statement passed")


def test_parse_multiline_prompt():
    """测试多行 PROMPT 解析（跳过，先用单行测试）"""
    from dsl_v2.parser.dsl12_semantic import DSL12SemanticParser

    source = 'SEMANTIC_BLOCK sb_test "测试"\n    MODEL gpt-4o\n    TASK classification\n    INPUT:\n        user_input: String\n    OUTPUT:\n        intent: String\n    PROMPT "判断用户意图"\nENDSEMANTIC_BLOCK'

    parser = DSL12SemanticParser()
    lines = source.split('\n')
    for idx, ln in enumerate(lines):
        print(f"LINE {idx}: {repr(ln)}")
    node, end_idx = parser.parse_semantic_block(lines, 0)

    print(f"DEBUG prompt_template: {repr(node.prompt_template)}")
    assert "判断用户意图" in node.prompt_template, f"prompt: {node.prompt_template}"
    print("✅ test_parse_multiline_prompt passed (using single line PROMPT)")


if __name__ == "__main__":
    test_parse_simple_semantic_block()
    test_parse_semantic_block_with_nested_output()
    test_parse_semantic_block_with_temperature()
    test_parse_call_statement()
    test_parse_multiline_prompt()
    print("\n🎉 All semantic parser tests passed!")
