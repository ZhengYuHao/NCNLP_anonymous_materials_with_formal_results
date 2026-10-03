"""
统一数据模型定义
提供所有模块共用的数据结构
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


# ============================================================================
# 枚举定义
# ============================================================================

class ProcessingMode(Enum):
    """处理模式枚举"""
    DICTIONARY = "dictionary"  # 基于词表
    SMART = "smart"           # 纯LLM智能
    HYBRID = "hybrid"         # 混合模式


class ProcessingStatus(Enum):
    """处理状态枚举"""
    SUCCESS = "success"       # 成功
    AMBIGUITY = "ambiguity"   # 检测到歧义
    ERROR = "error"           # 处理错误
    PARTIAL = "partial"       # 部分成功


class DataType(Enum):
    """变量数据类型白名单"""
    STRING = "String"
    INTEGER = "Integer"
    FLOAT = "Float"
    BOOLEAN = "Boolean"
    LIST = "List"
    ENUM = "Enum"


# ============================================================================
# 中间步骤快照
# ============================================================================

@dataclass
class StepSnapshot:
    """单个处理步骤的快照"""
    step_name: str           # 步骤名称
    step_index: int          # 步骤序号
    input_text: str          # 输入文本
    output_text: str         # 输出文本
    changes: Dict[str, str]  # 变更记录
    duration_ms: int         # 耗时（毫秒）
    notes: List[str] = field(default_factory=list)  # 备注信息
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def get_diff_summary(self) -> str:
        """获取变更摘要"""
        if self.input_text == self.output_text:
            return "无变更"
        if not self.changes:
            return "文本已修改"
        return f"变更 {len(self.changes)} 处: {list(self.changes.keys())}"


# ============================================================================
# 阶段1: Prompt 1.0 预处理结果
# ============================================================================

@dataclass
class Prompt10Result:
    """
    Prompt 1.0 预处理结果
    prompt_preprocessor.py 的输出，同时也是 prompt_structurizer.py 的输入
    """
    # 基本信息
    id: str                          # 唯一标识符
    timestamp: str                   # 处理时间戳
    mode: str                        # 处理模式
    
    # 文本信息
    original_text: str               # 原始输入文本
    processed_text: str              # 处理后的文本
    
    # 处理详情
    steps: List[StepSnapshot]        # 中间步骤快照
    terminology_changes: Dict[str, str]  # 术语替换记录
    
    # 状态信息
    status: str                      # 处理状态
    ambiguity_detected: bool         # 是否检测到歧义
    ambiguity_details: Optional[str] = None  # 歧义详情
    
    # 日志
    steps_log: List[str] = field(default_factory=list)  # 步骤日志
    warnings: List[str] = field(default_factory=list)   # 警告信息
    
    # 性能指标
    processing_time_ms: int = 0      # 总处理时间（毫秒）
    llm_calls_count: int = 0         # LLM 调用次数
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        # 将 StepSnapshot 列表转换为字典列表
        data['steps'] = [step.to_dict() if hasattr(step, 'to_dict') else step for step in self.steps]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Prompt10Result':
        """从字典创建实例"""
        # 将步骤字典转换为 StepSnapshot 对象
        if 'steps' in data and data['steps']:
            data['steps'] = [
                StepSnapshot(**step) if isinstance(step, dict) else step 
                for step in data['steps']
            ]
        return cls(**data)
    
    def is_success(self) -> bool:
        """判断是否成功"""
        return self.status == ProcessingStatus.SUCCESS.value
    
    def get_step_by_name(self, name: str) -> Optional[StepSnapshot]:
        """根据名称获取步骤"""
        for step in self.steps:
            if step.step_name == name:
                return step
        return None


# ============================================================================
# 阶段2: Prompt 2.0 结构化结果
# ============================================================================

@dataclass
class VariableMeta:
    """变量元数据"""
    name: str                  # 变量名 (英文, snake_case)
    original_text: str         # 原文中的精确片段
    value: Any                 # 提取出的具体值
    data_type: str             # 数据类型
    start_index: int           # 在原文中的起始位置
    end_index: int             # 在原文中的结束位置
    source_context: str = ""   # 来源上下文
    constraints: Optional[Dict] = None  # 约束条件
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Prompt20Result:
    """
    Prompt 2.0 结构化结果
    prompt_structurizer.py 的输出
    """
    # 基本信息
    id: str                          # 唯一标识符
    timestamp: str                   # 处理时间戳
    source_prompt10_id: str          # 来源 Prompt 1.0 的 ID
    
    # 模板信息
    template_text: str               # 带 {{variable}} 占位符的模板
    original_text: str               # 原始文本（来自 Prompt 1.0）
    
    # 变量信息
    variables: List[VariableMeta]    # 变量列表
    variable_registry: List[Dict]    # 变量注册表（兼容旧格式）
    
    # 日志
    extraction_log: List[str] = field(default_factory=list)  # 提取日志
    
    # 性能指标
    processing_time_ms: int = 0      # 处理时间（毫秒）
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['variables'] = [v.to_dict() if hasattr(v, 'to_dict') else v for v in self.variables]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Prompt20Result':
        """从字典创建实例"""
        if 'variables' in data and data['variables']:
            data['variables'] = [
                VariableMeta(**v) if isinstance(v, dict) else v 
                for v in data['variables']
            ]
        return cls(**data)
    
    def fill_template(self, values: Optional[Dict[str, Any]] = None) -> str:
        """
        使用值填充模板
        
        Args:
            values: 变量值字典，如果为None则使用默认值
            
        Returns:
            填充后的文本
        """
        result = self.template_text
        for var in self.variables:
            placeholder = f"{{{{{var.name}}}}}"
            value = values.get(var.name, var.value) if values else var.value
            result = result.replace(placeholder, str(value))
        return result


# ============================================================================
# 阶段 2.5: Prompt 2.5 结构化语义结果
# ============================================================================

@dataclass
class Operation:
    """
    操作单元
    表示一个具体的操作，如筛选、排序、限制等
    """
    type: str                          # 操作类型: filter, sort, limit, call, return, assign
    target: Optional[str] = None      # 操作目标（如表名、变量名）
    field: Optional[str] = None       # 字段名（如排序字段、筛选字段）
    condition: Optional[Dict] = None  # 条件 {field, op, value}
    value: Any = None                 # 值（如 limit 数量）
    order: Optional[str] = None       # 排序方向: asc, desc
    function: Optional[str] = None    # 函数名（用于 CALL 操作）
    params: Optional[Dict] = None     # 函数参数
    source_context: str = ""          # 原文中的原始表述
    
    implementation: Optional[str] = None
    impl_language: str = "python"
    impl_required_imports: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.impl_required_imports is None:
            self.impl_required_imports = []

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OutputField:
    """
    输出字段定义
    """
    name: str                          # 字段名
    data_type: str = "String"         # 数据类型
    source_field: Optional[str] = None # 源字段（如果有映射）
    aggregation: Optional[str] = None  # 聚合方式: sum, count, avg 等
    source_context: str = ""          # 原文表述

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ConditionBranch:
    """
    条件分支
    """
    condition: str                     # 条件表达式（DSL格式）
    operations: List[Operation]       # 该分支下的操作
    is_default: bool = False           # 是否为默认分支

    def to_dict(self) -> Dict:
        return {
            'condition': self.condition,
            'operations': [op.to_dict() for op in self.operations],
            'is_default': self.is_default
        }


@dataclass
class Prompt25Result:
    """
    Prompt 2.5 结构化语义结果
    在 Prompt 2.0 基础上，增加了完整的结构化语义表示

    改进点：
    - 不再依赖自然语言描述，而是用结构化操作序列
    - DSL 转译器可以直接基于此结构生成 DSL，无需再次理解自然语言
    """
    # 基本信息
    id: str                            # 唯一标识符
    timestamp: str                     # 处理时间戳
    source_prompt10_id: str           # 来源 Prompt 1.0 的 ID
    source_prompt20_id: Optional[str] = None  # 来源 Prompt 2.0 的 ID

    # 意图分类
    intent: str = ""                   # 意图: query, command, automation, analysis 等

    # 操作序列（核心改进）
    operations: List[Operation] = field(default_factory=list)

    # 条件分支（如果有）
    branches: List[ConditionBranch] = field(default_factory=list)

    # 输出格式定义
    output_schema: List[OutputField] = field(default_factory=list)

    # 变量信息（从 Prompt20 保留）
    variables: List[VariableMeta] = field(default_factory=list)
    variable_registry: List[Dict] = field(default_factory=list)

    # 原始文本（用于调试和回溯）
    original_text: str = ""

    # 歧义标记（如果有未解决的歧义）
    ambiguities: List[str] = field(default_factory=list)

    # 置信度
    confidence: float = 1.0           # 0.0 ~ 1.0

    # 日志
    extraction_log: List[str] = field(default_factory=list)

    # 性能指标
    processing_time_ms: int = 0

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'source_prompt10_id': self.source_prompt10_id,
            'source_prompt20_id': self.source_prompt20_id,
            'intent': self.intent,
            'operations': [op.to_dict() for op in self.operations],
            'branches': [br.to_dict() for br in self.branches],
            'output_schema': [f.to_dict() for f in self.output_schema],
            'variables': [v.to_dict() if hasattr(v, 'to_dict') else v for v in self.variables],
            'variable_registry': self.variable_registry,
            'original_text': self.original_text,
            'ambiguities': self.ambiguities,
            'confidence': self.confidence,
            'extraction_log': self.extraction_log,
            'processing_time_ms': self.processing_time_ms
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Prompt25Result':
        """从字典创建实例"""
        if 'variables' in data and data['variables']:
            data['variables'] = [
                VariableMeta(**v) if isinstance(v, dict) else v
                for v in data['variables']
            ]
        if 'operations' in data and data['operations']:
            data['operations'] = [
                Operation(**op) if isinstance(op, dict) else op
                for op in data['operations']
            ]
        if 'branches' in data and data['branches']:
            branches = []
            for br in data['branches']:
                if isinstance(br, dict):
                    br['operations'] = [
                        Operation(**op) if isinstance(op, dict) else op
                        for op in br.get('operations', [])
                    ]
                    branches.append(ConditionBranch(**br))
                else:
                    branches.append(br)
            data['branches'] = branches
        if 'output_schema' in data and data['output_schema']:
            data['output_schema'] = [
                OutputField(**f) if isinstance(f, dict) else f
                for f in data['output_schema']
            ]
        return cls(**data)

    def has_ambiguities(self) -> bool:
        """是否有未解决的歧义"""
        return len(self.ambiguities) > 0

    def get_operations_by_type(self, op_type: str) -> List[Operation]:
        """获取指定类型的操作"""
        return [op for op in self.operations if op.type == op_type]


def convert_prompt25_to_dsl_input(prompt25_result: Prompt25Result) -> Dict[str, Any]:
    """
    将 Prompt25Result 转换为 DSL 编译器所需的输入格式

    与 convert_prompt20_to_dsl_input 的区别：
    - 不再传递自然语言 logic 描述
    - 传递完整的结构化操作序列

    Args:
        prompt25_result: Prompt 2.5 结构化语义结果

    Returns:
        DSL 编译器输入字典
    """
    # 转换变量
    variables = []
    for var in prompt25_result.variables:
        var_dict = {
            'name': var.name,
            'type': var.data_type,
        }
        if var.value is not None:
            var_dict['default'] = var.value
        variables.append(var_dict)

    # 转换操作为结构化描述（供 DSL 生成器使用）
    operations_desc = []
    for op in prompt25_result.operations:
        op_desc = {
            'type': op.type,
            'target': op.target,
        }
        if op.field:
            op_desc['field'] = op.field
        if op.condition:
            op_desc['condition'] = op.condition
        if op.value is not None:
            op_desc['value'] = op.value
        if op.order:
            op_desc['order'] = op.order
        if op.function:
            op_desc['function'] = op.function
        if op.params:
            op_desc['params'] = op.params
        operations_desc.append(op_desc)

    return {
        'intent': prompt25_result.intent,
        'variables': variables,
        'operations': operations_desc,  # 结构化操作，不再是自然语言
        'branches': [br.to_dict() for br in prompt25_result.branches],
        'output_schema': [f.to_dict() for f in prompt25_result.output_schema],
        'confidence': prompt25_result.confidence,
    }


# ============================================================================
# 完整处理链结果
# ============================================================================

@dataclass
class FullPipelineResult:
    """
    完整处理链结果
    包含 Prompt 1.0 和 Prompt 2.0 的所有信息
    """
    # 基本信息
    pipeline_id: str                 # 流水线唯一标识
    timestamp: str                   # 开始处理时间
    
    # 原始输入
    raw_input: str                   # 用户原始输入
    
    # 阶段结果
    prompt10_result: Optional[Prompt10Result] = None  # 阶段1结果
    prompt20_result: Optional[Prompt20Result] = None  # 阶段2结果
    
    # 最终输出
    final_template: str = ""         # 最终模板
    final_variables: List[Dict] = field(default_factory=list)  # 最终变量表
    
    # 状态
    overall_status: str = "pending"  # 整体状态
    error_message: Optional[str] = None  # 错误信息
    
    # 性能
    total_time_ms: int = 0           # 总处理时间
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'pipeline_id': self.pipeline_id,
            'timestamp': self.timestamp,
            'raw_input': self.raw_input,
            'prompt10_result': self.prompt10_result.to_dict() if self.prompt10_result else None,
            'prompt20_result': self.prompt20_result.to_dict() if self.prompt20_result else None,
            'final_template': self.final_template,
            'final_variables': self.final_variables,
            'overall_status': self.overall_status,
            'error_message': self.error_message,
            'total_time_ms': self.total_time_ms
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    def is_success(self) -> bool:
        """判断整体是否成功"""
        return self.overall_status == ProcessingStatus.SUCCESS.value


# ============================================================================
# 工具函数
# ============================================================================

def generate_id() -> str:
    """生成唯一 ID"""
    import uuid
    return str(uuid.uuid4())[:8]


def get_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_prompt10_result(
    original_text: str,
    processed_text: str,
    mode: str,
    steps: List[StepSnapshot] = None,
    terminology_changes: Dict[str, str] = None,
    status: str = ProcessingStatus.SUCCESS.value,
    ambiguity_detected: bool = False,
    **kwargs
) -> Prompt10Result:
    """创建 Prompt10Result 的便捷函数"""
    return Prompt10Result(
        id=generate_id(),
        timestamp=get_timestamp(),
        mode=mode,
        original_text=original_text,
        processed_text=processed_text,
        steps=steps or [],
        terminology_changes=terminology_changes or {},
        status=status,
        ambiguity_detected=ambiguity_detected,
        **kwargs
    )


def create_prompt20_result(
    source_prompt10_id: str,
    original_text: str,
    template_text: str,
    variables: List[VariableMeta] = None,
    **kwargs
) -> Prompt20Result:
    """创建 Prompt20Result 的便捷函数"""
    vars_list = variables or []
    variable_registry = [
        {
            "variable": var.name,
            "value": var.value,
            "type": var.data_type,
            "original_text": var.original_text,
            "source_context": var.source_context
        }
        for var in vars_list
    ]
    
    return Prompt20Result(
        id=generate_id(),
        timestamp=get_timestamp(),
        source_prompt10_id=source_prompt10_id,
        original_text=original_text,
        template_text=template_text,
        variables=vars_list,
        variable_registry=variable_registry,
        **kwargs
    )


def convert_prompt20_to_dsl_input(prompt20_result: Prompt20Result) -> Dict[str, Any]:
    """
    将 Prompt20Result 转换为 DSL 编译器所需的输入格式
    
    Args:
        prompt20_result: Prompt 2.0 结构化结果
        
    Returns:
        DSL 编译器输入字典，包含 variables 和 logic 字段
    """
    variables = []
    for var in prompt20_result.variables:
        var_dict = {
            'name': var.name,
            'type': var.data_type,
        }
        if var.value is not None:
            var_dict['default'] = var.value
        variables.append(var_dict)
    
    return {
        'variables': variables,
        'logic': prompt20_result.original_text,  # 使用原始文本作为逻辑描述
        'context': 'Converted from Prompt 2.0'
    }
