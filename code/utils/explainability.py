"""
可解释性引擎
Explainability Engine

核心功能：
1. 决策追溯 - 完整记录每个处理步骤的证据链
2. 语义可视化 - 生成可交互的决策流程图
3. 置信度评估 - 为每个决策提供可靠性评分
4. 解释生成 - 将技术决策转化为可读解释
"""

import uuid
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from logger import info, debug, warning


class TraceLevel(Enum):
    """追溯级别"""
    MINIMAL = "minimal"          # 仅最终结果
    STANDARD = "standard"        # 关键步骤
    DETAILED = "detailed"        # 完整决策链
    DEBUG = "debug"              # 调试模式（包含所有细节）


class EvidenceType(Enum):
    """证据类型"""
    RULE_MATCH = "rule_match"           # 规则匹配
    LLM_INFERENCE = "llm_inference"     # LLM推理
    USER_INPUT = "user_input"         # 用户输入
    PATTERN_DETECTED = "pattern_detected"  # 模式检测
    AMBIGUITY_DETECTED = "ambiguity_detected"  # 歧义检测
    VALIDATION_PASSED = "validation_passed"  # 验证通过
    VALIDATION_FAILED = "validation_failed"  # 验证失败
    CACHE_HIT = "cache_hit"           # 缓存命中
    CACHE_MISS = "cache_miss"         # 缓存未命中
    TRANSFORMATION = "transformation" # 转换操作
    OPTIMIZATION = "optimization"     # 优化决策
    SEMANTIC_STRUCTURE = "semantic_structure" # 语义结构


@dataclass
class DecisionEvidence:
    """决策证据"""
    evidence_id: str
    evidence_type: EvidenceType
    content: str                    # 证据内容
    confidence: float               # 置信度 [0, 1]
    source: str                     # 来源模块
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 证据详情
    rule_name: Optional[str] = None  # 匹配的规则名
    pattern_matched: Optional[str] = None  # 匹配的模式
    llm_prompt: Optional[str] = None  # LLM提示（仅debug模式）
    llm_response: Optional[str] = None  # LLM响应（仅debug模式）
    
    def to_dict(self) -> Dict:
        return {
            'evidence_id': self.evidence_id,
            'evidence_type': self.evidence_type.value,
            'content': self.content,
            'confidence': self.confidence,
            'source': self.source,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'rule_name': self.rule_name,
            'pattern_matched': self.pattern_matched
        }


@dataclass
class ProcessingStepTrace:
    """处理步骤追溯"""
    step_id: str
    step_name: str
    step_index: int
    output_summary: str            # 输出摘要（必须在前）
    input_summary: str = ""       # 输入摘要
    
    # 输入输出快照
    input_snapshot: Optional[Dict] = None  # 输入快照（可选）
    output_snapshot: Optional[Dict] = None  # 输出快照（可选）
    
    # 决策证据链
    evidences: List[DecisionEvidence] = field(default_factory=list)
    
    # 推理过程
    reasoning: str = ""            # 推理过程描述
    decision_path: List[str] = field(default_factory=list)  # 决策路径
    
    # 性能指标
    duration_ms: int = 0
    status: str = "success"        # success/error/partial
    
    # 变更记录
    changes: Dict[str, Any] = field(default_factory=dict)
    
    def add_evidence(
        self,
        evidence_type: EvidenceType,
        content: str,
        confidence: float,
        source: str,
        **kwargs
    ):
        """添加证据"""
        evidence = DecisionEvidence(
            evidence_id=str(uuid.uuid4())[:8],
            evidence_type=evidence_type,
            content=content,
            confidence=confidence,
            source=source,
            timestamp=datetime.now().isoformat(),
            **kwargs
        )
        self.evidences.append(evidence)
    
    def to_dict(self) -> Dict:
        return {
            'step_id': self.step_id,
            'step_name': self.step_name,
            'step_index': self.step_index,
            'input_summary': self.input_summary,
            'output_summary': self.output_summary,
            'evidences': [e.to_dict() for e in self.evidences],
            'reasoning': self.reasoning,
            'decision_path': self.decision_path,
            'duration_ms': self.duration_ms,
            'status': self.status,
            'changes': self.changes
        }
    
    def get_confidence(self) -> float:
        """计算步骤置信度"""
        if not self.evidences:
            return 0.5
        return sum(e.confidence for e in self.evidences) / len(self.evidences)


@dataclass
class DecisionTrace:
    """完整决策追溯"""
    trace_id: str
    timestamp: str
    
    # 输入输出
    input_text: str
    output_result: Any
    
    # 处理步骤
    steps: List[ProcessingStepTrace] = field(default_factory=list)
    
    # 总体评估
    overall_confidence: float = 0.0
    trace_level: TraceLevel = TraceLevel.STANDARD
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: ProcessingStepTrace):
        """添加步骤"""
        self.steps.append(step)

    STEP_DESCRIPTIONS = {
        "术语对齐": "扫描词表将口语词替换为标准词 | 按长度降序排列避免短词误伤 | 零LLM消耗",
        "歧义阻断": "逐词匹配歧义词黑名单，命中即告警 | 快速阻断，不依赖LLM",
        "语义重构 (LLM 可选)": "规则引擎先行标准化(口语→书面化、标点规范化) | 规则失败才调用LLM | 置信度<0.7触发兜底",
        "LLM语义重构": "口语→书面化 | 标点规范化 | 规则失败才调用LLM | 置信度<0.7触发兜底",
        "规则引擎语义重构": "规则引擎先行 | 口语转书面化 | 规则无变化才回退LLM",
        "实体扫描 (LLM 可选)": "正则快速提取(数字/技术栈/时长) | 正则不足时LLM补充 | 隐含实体识别",
        "正则实体提取": "数字+单位 | 技术栈模式 | 时长模式 | 快速低成本覆盖显性实体",
        "LLM实体提取": "深度语义理解 | 隐含实体识别 | 正则失败时兜底调用",
        "幻觉防火墙": "精确匹配验证(原文比对) | 索引位置校验 | 模糊匹配容错 | 防LLM虚构",
        "精确匹配验证": "逐字比对实体是否存在于原文 | 空格/标点差异则进入模糊匹配",
        "索引位置验证": "校验start/end索引准确性 | 自动修正偏移错误",
        "词法解析": "逐行识别IF/FOR/CALL等块类型 | 正则提取{{变量}} | 生成原子CodeBlock列表",
        "依赖分析": "构建变量生产者映射表 | NetworkX DAG | 循环检测+拓扑排序 | 确定执行顺序",
        "结构歧义检测": "句法歧义审计(规则引擎) + 深度检测(LLM回退) | 捕获\"咬死了猎人的狗\"类歧义",
        "智能模式处理": "混合规则学习 | 边用边学机制 | 动态更新规则库",
        "Prompt 2.5 结构化语义": "提取操作/分支/条件 | 意图分类 | 歧义标记",
        "DSL 编译": "DSLv2编译 → 自动修复 → 验证通过 | 三次重试机制"
    }

    def get_step_by_name(self, name: str) -> Optional[ProcessingStepTrace]:
        """根据名称获取步骤"""
        for step in self.steps:
            if step.step_name == name:
                return step
        return None
    
    def get_decision_path(self) -> List[str]:
        """获取决策路径"""
        path = []
        for step in self.steps:
            path.append(f"{step.step_index}. {step.step_name}")
            path.extend(step.decision_path)
        return path
    
    def calculate_confidence(self):
        """计算整体置信度"""
        if not self.steps:
            self.overall_confidence = 0.5
            return
        
        # 加权平均（后面的步骤权重更高）
        total = 0
        weights = 0
        for i, step in enumerate(self.steps):
            weight = i + 1  # 递增权重
            total += step.get_confidence() * weight
            weights += weight
        
        self.overall_confidence = total / weights if weights > 0 else 0.5
    
    def to_dict(self) -> Dict:
        return {
            'trace_id': self.trace_id,
            'timestamp': self.timestamp,
            'input_text': self.input_text[:200] + "..." if len(self.input_text) > 200 else self.input_text,
            'overall_confidence': self.overall_confidence,
            'trace_level': self.trace_level.value,
            'steps': [s.to_dict() for s in self.steps],
            'metadata': self.metadata
        }
    
    def to_mermaid(self) -> str:
        """生成 Mermaid 流程图"""
        lines = ["graph TD"]

        # 添加节点
        for step in self.steps:
            # 节点ID
            node_id = f"S{step.step_index}"

            # 节点标签（包含关键信息和解释）
            confidence_emoji = "🟢" if step.get_confidence() > 0.8 else "🟡" if step.get_confidence() > 0.5 else "🔴"
            step_name = step.step_name
            description = self.STEP_DESCRIPTIONS.get(step_name, "")
            if description:
                label = f"{confidence_emoji} {step_name}\n<small>{description}</small>"
            else:
                label = f"{confidence_emoji} {step_name}"

            # 添加节点
            lines.append(f'    {node_id}["{label}"]')

        # 添加边
        for i in range(len(self.steps) - 1):
            lines.append(f'    S{i+1} --> S{i+2}')

        # 添加关键决策点
        for step in self.steps:
            if step.decision_path:
                lines.append(f'    S{step.step_index} -.-> |"关键决策"| S{step.step_index}_d')

        # 添加样式
        lines.append("")
        lines.append("    classDef high fill:#90EE90,stroke:#228B22")
        lines.append("    classDef medium fill:#FFFACD,stroke:#DAA520")
        lines.append("    classDef low fill:#FFB6C1,stroke:#DC143C")

        return "\n".join(lines)


class ExplainabilityEngine:
    """
    可解释性引擎
    
    为整个处理流水线提供可解释性支持
    """
    
    def __init__(
        self,
        trace_level: TraceLevel = TraceLevel.STANDARD,
        store_traces: bool = True
    ):
        self.trace_level = trace_level
        self.store_traces = store_traces
        
        # 当前追溯
        self.current_trace: Optional[DecisionTrace] = None
        self.current_step: Optional[ProcessingStepTrace] = None
        
        # 历史追溯存储
        self.trace_storage: List[DecisionTrace] = []
    
    def start_trace(self, input_text: str, **metadata) -> str:
        """
        开始追溯
        
        Args:
            input_text: 输入文本
            **metadata: 额外元数据
            
        Returns:
            trace_id
        """
        self.current_trace = DecisionTrace(
            trace_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            input_text=input_text,
            output_result=None,
            trace_level=self.trace_level,
            metadata=metadata
        )
        
        debug(f"开始追溯: {self.current_trace.trace_id}")
        return self.current_trace.trace_id
    
    def start_step(
        self,
        step_name: str,
        step_index: int,
        input_data: Any = None
    ) -> str:
        """
        开始处理步骤
        
        Args:
            step_name: 步骤名称
            step_index: 步骤索引
            input_data: 输入数据
            
        Returns:
            step_id
        """
        if not self.current_trace:
            raise RuntimeError("必须先调用 start_trace()")
        
        step = ProcessingStepTrace(
            step_id=str(uuid.uuid4())[:8],
            step_name=step_name,
            step_index=step_index,
            output_summary="",  # 初始为空，end_step 时填充
            input_summary=str(input_data)[:100] if input_data else "",
            input_snapshot=self._create_snapshot(input_data) if self.trace_level == TraceLevel.DEBUG else None
        )
        
        self.current_trace.add_step(step)
        self.current_step = step
        
        debug(f"开始步骤: {step_name}")
        return step.step_id
    
    def _create_snapshot(self, data: Any) -> Dict:
        """创建数据快照"""
        if isinstance(data, dict):
            return {k: str(v)[:100] for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return {"length": len(data), "preview": str(data)[:100]}
        else:
            return {"value": str(data)[:100]}
    
    def add_evidence(
        self,
        evidence_type: EvidenceType,
        content: str,
        confidence: float,
        source: str,
        **kwargs
    ):
        """
        添加证据
        
        Args:
            evidence_type: 证据类型
            content: 证据内容
            confidence: 置信度
            source: 来源模块
            **kwargs: 额外参数
        """
        if not self.current_step:
            warning("没有活动步骤，无法添加证据")
            return
        
        self.current_step.add_evidence(
            evidence_type=evidence_type,
            content=content,
            confidence=confidence,
            source=source,
            **kwargs
        )
    
    def add_reasoning(self, reasoning: str):
        """添加推理过程描述"""
        if not self.current_step:
            return
        self.current_step.reasoning = reasoning
    
    def add_decision(self, decision: str):
        """添加决策点"""
        if not self.current_step:
            return
        self.current_step.decision_path.append(decision)
    
    def record_change(self, key: str, value: Any):
        """记录变更"""
        if not self.current_step:
            return
        self.current_step.changes[key] = str(value)[:100]
    
    def end_step(
        self,
        output_data: Any = None,
        status: str = "success",
        duration_ms: int = 0
    ):
        """
        结束处理步骤
        
        Args:
            output_data: 输出数据
            status: 状态
            duration_ms: 耗时
        """
        if not self.current_step:
            return
        
        self.current_step.output_summary = str(output_data)[:100] if output_data else ""
        self.current_step.output_snapshot = self._create_snapshot(output_data) if self.trace_level == TraceLevel.DEBUG else None
        self.current_step.status = status
        self.current_step.duration_ms = duration_ms
        
        self.current_step = None
    
    def finalize_trace(self, output_result: Any) -> DecisionTrace:
        """
        完成追溯
        
        Args:
            output_result: 最终输出结果
            
        Returns:
            决策追溯对象
        """
        if not self.current_trace:
            raise RuntimeError("必须先调用 start_trace()")
        
        self.current_trace.output_result = str(output_result)[:200] if output_result else ""
        self.current_trace.calculate_confidence()
        
        if self.store_traces:
            self.trace_storage.append(self.current_trace)
        
        trace = self.current_trace
        self.current_trace = None
        
        debug(f"完成追溯: {trace.trace_id}, 置信度: {trace.overall_confidence:.2f}")
        return trace
    
    def explain_decision(self, trace: Optional[DecisionTrace] = None) -> str:
        """
        生成决策解释
        
        Args:
            trace: 追溯对象（默认当前）
            
        Returns:
            可读的决策解释
        """
        trace = trace or self.current_trace
        if not trace:
            return "无追溯数据"
        
        lines = []
        lines.append("=" * 60)
        lines.append("📋 决策解释报告")
        lines.append("=" * 60)
        lines.append(f"追溯ID: {trace.trace_id}")
        lines.append(f"整体置信度: {trace.overall_confidence:.1%}")
        lines.append("")
        
        for step in trace.steps:
            lines.append(f"【步骤 {step.step_index}】{step.step_name}")
            lines.append(f"  状态: {step.status} | 耗时: {step.duration_ms}ms")
            lines.append(f"  置信度: {step.get_confidence():.1%}")
            
            if step.reasoning:
                lines.append(f"  推理: {step.reasoning}")
            
            if step.evidences:
                lines.append("  证据链:")
                for e in step.evidences[:3]:  # 最多显示3条
                    emoji = "✓" if e.confidence > 0.7 else "?"
                    lines.append(f"    {emoji} [{e.evidence_type.value}] {e.content[:50]}")
            
            if step.changes:
                lines.append(f"  变更: {', '.join(step.changes.keys())}")
            
            lines.append("")
        
        lines.append("=" * 60)
        lines.append("决策路径: " + " → ".join(trace.get_decision_path()[:5]))
        
        return "\n".join(lines)
    
    def visualize_trace(
        self,
        trace: Optional[DecisionTrace] = None,
        format: str = "mermaid"
    ) -> str:
        """
        可视化追溯结果
        
        Args:
            trace: 追溯对象
            format: 输出格式 (mermaid/json)
            
        Returns:
            可视化代码
        """
        trace = trace or self.current_trace
        if not trace:
            return "无追溯数据"
        
        if format == "mermaid":
            return trace.to_mermaid()
        elif format == "json":
            return json.dumps(trace.to_dict(), ensure_ascii=False, indent=2)
        else:
            return "不支持的格式"
    
    def get_confidence_report(self, trace: Optional[DecisionTrace] = None) -> Dict:
        """获取置信度报告"""
        trace = trace or self.current_trace
        if not trace:
            return {}
        
        return {
            "overall": trace.overall_confidence,
            "by_step": [
                {
                    "step": s.step_name,
                    "confidence": s.get_confidence(),
                    "evidence_count": len(s.evidences)
                }
                for s in trace.steps
            ],
            "low_confidence_steps": [
                s.step_name for s in trace.steps 
                if s.get_confidence() < 0.5
            ]
        }
    
    def get_trace(self, trace_id: str) -> Optional[DecisionTrace]:
        """根据ID获取追溯"""
        for trace in self.trace_storage:
            if trace.trace_id == trace_id:
                return trace
        return None
    
    def get_all_traces(self) -> List[DecisionTrace]:
        """获取所有追溯"""
        return self.trace_storage


class SemanticVisualizer:
    """
    语义可视化引擎
    
    生成各类可视化图表：
    1. 实体关系图
    2. 数据流图
    3. 置信度热力图
    4. 对比视图
    """
    
    @staticmethod
    def generate_entity_graph(
        variables: List[Dict],
        relations: Optional[List[Dict]] = None
    ) -> str:
        """
        生成实体关系图 (Mermaid)
        
        Args:
            variables: 变量列表
            relations: 关系列表 (可选)
            
        Returns:
            Mermaid 图代码
        """
        lines = ["graph LR"]
        
        # 节点样式映射
        type_styles = {
            "String": "fill:#E1F5FE,stroke:#0277BD",
            "Integer": "fill:#E8F5E9,stroke:#2E7D32",
            "Float": "fill:#E0F2F1,stroke:#00695C",
            "Boolean": "fill:#FFF3E0,stroke:#EF6C00",
            "List": "fill:#F3E5F5,stroke:#7B1FA2",
        }
        
        # 添加节点
        for var in variables:
            var_name = var.get('name', 'unknown')
            var_type = var.get('type', 'String')
            style = type_styles.get(var_type, "fill:#ECEFF1,stroke:#546E7A")
            
            lines.append(f'    {var_name}["{var_name}: {var_type}"]')
            lines.append(f'    class {var_name} {style}')
        
        # 添加关系
        if relations:
            for rel in relations:
                source = rel.get('from', '')
                target = rel.get('to', '')
                label = rel.get('label', '')
                if source and target:
                    if label:
                        lines.append(f'    {source} -->|{label}| {target}')
                    else:
                        lines.append(f'    {source} --> {target}')
        
        lines.append("")
        lines.append("    classDef default fill:#ECEFF1,stroke:#546E7A")
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_data_flow(
        steps: List[ProcessingStepTrace]
    ) -> str:
        """
        生成数据流图
        
        Args:
            steps: 处理步骤列表
            
        Returns:
            Mermaid 图代码
        """
        lines = ["flowchart TB"]
        
        # 子图样式
        status_styles = {
            "success": "fill:#E8F5E9,stroke:#2E7D32",
            "error": "fill:#FFEBEE,stroke:#C62828",
            "partial": "fill:#FFF3E0,stroke:#EF6C00"
        }
        
        for i, step in enumerate(steps):
            step_id = f"step{i}"
            style = status_styles.get(step.status, "fill:#ECEFF1,stroke:#546E7A")
            
            # 节点
            lines.append(f'    {step_id}["{step.step_name}"]')
            lines.append(f'    {step_id}:::status')
            
            # 边
            if i > 0:
                lines.append(f'    step{i-1} --> {step_id}')
        
        # 添加样式定义
        lines.append("")
        lines.append("    classDef success fill:#E8F5E9,stroke:#2E7D32")
        lines.append("    classDef error fill:#FFEBEE,stroke:#C62828")
        lines.append("    classDef partial fill:#FFF3E0,stroke:#EF6C00")
        lines.append("    classDef default fill:#ECEFF1,stroke:#546E7A")
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_confidence_heatmap(
        trace: DecisionTrace
    ) -> str:
        """
        生成置信度热力图
        
        Args:
            trace: 追溯对象
            
        Returns:
            HTML 热力图
        """
        steps_data = []
        for step in trace.steps:
            conf = step.get_confidence()
            color = "#4CAF50" if conf > 0.8 else "#FFC107" if conf > 0.5 else "#F44336"
            steps_data.append({
                "name": step.step_name,
                "confidence": conf,
                "color": color
            })
        
        # 生成简单的HTML表格
        html = ['<div style="font-family: monospace;">']
        html.append('<h3>置信度热力图</h3>')
        
        for data in steps_data:
            width = data['confidence'] * 100
            html.append(f'<div style="margin: 4px 0;">')
            html.append(f'<span>{data["name"]:<20}</span>')
            html.append(f'<div style="display: inline-block; width: 200px; height: 20px; background: #eee; margin: 0 10px;">')
            html.append(f'<div style="width: {width}%; height: 100%; background: {data["color"]};"></div>')
            html.append(f'</div>')
            html.append(f'<span>{data["confidence"]:.1%}</span>')
            html.append(f'</div>')
        
        html.append('</div>')
        
        return "\n".join(html)
    
    @staticmethod
    def generate_comparison_view(
        before: str,
        after: str,
        changes: Optional[Dict] = None
    ) -> Dict:
        """
        生成对比视图
        
        Args:
            before: 变换前
            after: 变换后
            changes: 变更记录
            
        Returns:
            包含对比信息的字典
        """
        # 简单的diff计算
        before_lines = before.split('\n')
        after_lines = after.split('\n')
        
        return {
            "before": before,
            "after": after,
            "changes": changes or {},
            "summary": {
                "lines_before": len(before_lines),
                "lines_after": len(after_lines),
                "line_delta": len(after_lines) - len(before_lines)
            }
        }
    
    @staticmethod
    def generate_decision_tree(
        trace: DecisionTrace,
        max_depth: int = 3
    ) -> str:
        """
        生成决策树图
        
        Args:
            trace: 追溯对象
            max_depth: 最大深度
            
        Returns:
            Mermaid 决策树
        """
        lines = ["graph TB"]
        
        def add_branch(step: ProcessingStepTrace, depth: int = 0):
            if depth > max_depth:
                return
            
            node_id = f"d{depth}_{step.step_index}"
            decision = step.decision_path[0] if step.decision_path else ""
            
            if decision:
                lines.append(f'    {node_id}["{decision}"]')
                if depth > 0:
                    lines.append(f'    d{depth-1}_{step.step_index-1} --> {node_id}')
        
        for step in trace.steps:
            add_branch(step)
        
        return "\n".join(lines)


# 便捷函数
def create_explainer(**kwargs) -> ExplainabilityEngine:
    """创建可解释性引擎"""
    return ExplainabilityEngine(**kwargs)


def explain_trace(trace: DecisionTrace) -> str:
    """便捷函数：解释追溯"""
    return ExplainabilityEngine().explain_decision(trace)


def visualize_trace(trace: DecisionTrace, format: str = "mermaid") -> str:
    """便捷函数：可视化追溯"""
    return SemanticVisualizer.visualize_trace(trace, format)
