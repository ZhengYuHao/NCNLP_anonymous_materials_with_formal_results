"""AST 节点定义"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class InputVarNode:
    """语义 Block 输入变量"""
    name: str
    var_type: str


@dataclass
class OutputVarNode:
    """语义 Block 输出变量"""
    name: str
    var_type: str


@dataclass
class SemanticBlockNode:
    """SEMANTIC_BLOCK AST 节点"""
    block_id: str
    description: str
    name: str
    model: str
    task: str
    input_vars: List[InputVarNode] = field(default_factory=list)
    output_vars: List[OutputVarNode] = field(default_factory=list)
    prompt_template: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class CallSemanticNode:
    """CALL sb_xxx 节点"""
    target_block: str
    input_mapping: List[Tuple[str, str]] = field(default_factory=list)
    output_mapping: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class RuleBlockNode:
    """普通 BLOCK AST 节点"""
    block_id: str
    description: str
    statements: List = field(default_factory=list)


@dataclass
class IfConditionNode:
    """IF 条件节点"""
    condition: str
    statements: List = field(default_factory=list)


@dataclass
class ReturnNode:
    """RETURN 节点"""
    values: List[str] = field(default_factory=list)
