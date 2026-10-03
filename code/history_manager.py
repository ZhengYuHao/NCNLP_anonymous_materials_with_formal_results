"""
处理历史存储与对比展示模块
用于持久化存储每次处理的结果，并提供清晰的对比展示
支持完整流水线追踪：Prompt 1.0 + Prompt 2.0
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from logger import info, warning, error, debug


@dataclass
class ProcessingHistory:
    """单次处理历史记录（兼容旧格式 - Prompt 1.0）"""
    timestamp: str  # 处理时间戳
    original_text: str  # 原始输入
    processed_text: str  # 处理后文本
    mode: str  # 处理模式
    steps_log: List[str]  # 处理步骤日志
    warnings: List[str]  # 警告信息
    terminology_changes: Dict[str, str]  # 术语替换记录
    ambiguity_detected: bool  # 是否检测到歧义
    success: bool  # 是否成功处理（无歧义）
    processing_time_ms: Optional[int] = None  # 处理耗时（毫秒）

    # 新增：规则引擎统计（极窄化LLM优化）
    rule_engine_stats: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "llm_calls": 0,  # LLM调用次数
    #   "normalization_changes": 5,  # 规范化变更次数
    #   "ambiguity_detected": false,  # 是否检测到歧义
    #   "processing_mode": "dictionary",  # 处理模式
    #   "has_llm_fallback": false  # 是否回退到LLM
    # }


@dataclass
class Prompt20History:
    """Prompt 2.0 结构化历史记录"""
    id: str                              # 唯一标识
    timestamp: str                       # 处理时间戳
    source_prompt10_id: str              # 关联的 Prompt 1.0 ID
    
    # 输入输出
    input_text: str                      # 输入文本（来自 Prompt 1.0）
    template_text: str                   # 生成的模板
    
    # 变量信息
    variables: List[Dict] = field(default_factory=list)  # 变量列表
    variable_count: int = 0              # 变量数量
    
    # 类型统计
    type_stats: Dict[str, int] = field(default_factory=dict)  # 按类型统计
    
    # 日志
    extraction_log: List[str] = field(default_factory=list)
    
    # 性能
    processing_time_ms: int = 0


@dataclass
class PipelineHistory:
    """完整流水线历史记录"""
    pipeline_id: str  # 流水线ID
    timestamp: str  # 开始时间戳
    raw_input: str  # 用户原始输入
    
    # 阶段1结果
    prompt10_id: str = ""
    prompt10_original: str = ""
    prompt10_processed: str = ""
    prompt10_mode: str = ""
    prompt10_steps: List[Dict] = field(default_factory=list)
    prompt10_terminology_changes: Dict[str, str] = field(default_factory=dict)
    prompt10_ambiguity_detected: bool = False
    prompt10_status: str = ""
    prompt10_time_ms: int = 0
    
    # 阶段2结果
    prompt20_id: str = ""
    prompt20_template: str = ""
    prompt20_variables: List[Dict] = field(default_factory=list)
    prompt20_variable_count: int = 0
    prompt20_type_stats: Dict[str, int] = field(default_factory=dict)
    prompt20_extraction_log: List[str] = field(default_factory=list)
    prompt20_time_ms: int = 0
    
    # 阶段3结果 (DSL编译)
    prompt30_id: str = ""
    prompt30_dsl_code: str = ""
    prompt30_validation_result: Dict[str, Any] = field(default_factory=dict)
    prompt30_time_ms: int = 0
    prompt30_compile_history: Dict[str, Any] = field(default_factory=dict)  # 策略 D：编译历史
    prompt30_success: bool = True  # 策略 D：编译成功标志
    
    # 阶段4结果 (代码生成)
    prompt40_id: str = ""
    prompt40_modules: List[Dict] = field(default_factory=list)
    prompt40_module_count: int = 0
    prompt40_main_code: str = ""
    prompt40_time_ms: int = 0
    prompt40_module_bodies: Dict[str, str] = field(default_factory=dict)  # 添加模块函数体代码

    # 阶段4子步骤详情
    prompt40_step1_parsing: Dict[str, Any] = field(default_factory=dict)  # 词法解析
    prompt40_step2_dependency: Dict[str, Any] = field(default_factory=dict)  # 依赖分析
    prompt40_step3_clustering: Dict[str, Any] = field(default_factory=dict)  # 模块聚类
    prompt40_step4_generation: Dict[str, Any] = field(default_factory=dict)  # 代码生成
    prompt40_step5_orchestration: Dict[str, Any] = field(default_factory=dict)  # 主控编排

    # 整体状态
    overall_status: str = ""
    total_time_ms: int = 0
    error_message: Optional[str] = None

    # 优化统计字段（极窄化LLM优化）
    prompt10_rule_stats: Dict[str, Any] = field(default_factory=dict)
    prompt20_optimization_stats: Dict[str, Any] = field(default_factory=dict)
    prompt30_optimization_stats: Dict[str, Any] = field(default_factory=dict)
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    cache_hit_rate: float = 0.0
    validation_stats: Dict[str, Any] = field(default_factory=dict)
    auto_fix_stats: Dict[str, Any] = field(default_factory=dict)

    # ===== 智能优化与可解释性字段 =====
    # 术语学习结果
    term_learning: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "detected_terms": [...],      # 检测到的候选术语
    #   "confirmed_terms": {...},     # 已确认的术语映射
    #   "suggestions": [...],         # 建议的新术语
    #   "learning_count": 0           # 学习次数
    # }

    # 歧义检测与消解结果
    ambiguity_resolution: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "detected": [...],            # 检测到的歧义列表
    #   "resolved": [...],            # 已消解的歧义
    #   "pending": [...],             # 待确认的歧义
    #   "auto_resolved_count": 0,    # 自动消解数量
    #   "user_confirmed_count": 0     # 用户确认数量
    # }

    # 决策追溯数据
    decision_trace: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "trace_id": "...",
    #   "steps": [...],               # 处理步骤追溯
    #   "evidences": [...],           # 证据链
    #   "overall_confidence": 0.95,  # 整体置信度
    #   "decision_path": [...],      # 决策路径
    #   "explanations": {...}        # 步骤解释
    # }

    # 语义可视化数据
    visualization_data: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "entity_graph": "...",       # 实体关系图 (Mermaid)
    #   "data_flow": "...",          # 数据流图
    #   "confidence_heatmap": "...",  # 置信度热力图
    #   "decision_tree": "..."       # 决策树
    # }


class HistoryManager:
    """处理历史管理器"""

    def _generate_step_details_html(self, history: PipelineHistory) -> str:
        """生成第四步编译步骤详情的 HTML（完整展示，带滚动容器）"""
        # 添加滚动容器样式
        html = '''
<style>
    .step-scroll-container {
        max-height: 300px;
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: #667eea #f1f1f1;
    }
    .step-scroll-container::-webkit-scrollbar {
        width: 6px;
    }
    .step-scroll-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }
    .step-scroll-container::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 3px;
    }
    .step-code-block {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 12px;
        border-radius: 6px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 12px;
        white-space: pre-wrap;
        word-break: break-all;
        max-height: 250px;
        overflow-y: auto;
    }
    .step-item-card {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 8px;
        border-left: 3px solid #667eea;
    }
</style>
<div class="step-cards">
'''

        # Step 1: 词法解析
        if history.prompt40_step1_parsing:
            step1 = history.prompt40_step1_parsing
            total_blocks = step1.get('total_blocks', 0)
            block_types = step1.get('block_types', {})
            blocks = step1.get('blocks', [])

            block_types_str = ", ".join([f"{k}: {v}" for k, v in block_types.items()])

            blocks_preview = ""
            if blocks:
                for block in blocks:  # 显示所有块
                    block_id = block.get('id', 'N/A')
                    block_type = block.get('type', 'N/A')
                    block_inputs = ", ".join(block.get('inputs', []))
                    block_outputs = ", ".join(block.get('outputs', []))
                    is_async = '异步' if block.get('is_async') else '同步'
                    blocks_preview += f"""
                        <div class="step-item-card">
                            <strong>Block {block_id}</strong> ({block_type}, {is_async})<br>
                            &nbsp;&nbsp;输入: {block_inputs or '无'}<br>
                            &nbsp;&nbsp;输出: {block_outputs or '无'}
                        </div>
                    """

            html += f"""
                <div class="step-card">
                    <div class="step-header">
                        <span class="step-number">1</span>
                        <span class="step-title">词法解析</span>
                        <span class="step-duration"></span>
                    </div>
                    <div class="step-body">
                        <div class="step-section">
                            <h5>统计信息</h5>
                            <div class="change-item">代码块总数: {total_blocks}</div>
                            <div class="change-item">类型分布: {block_types_str}</div>
                        </div>
                        <div class="step-section">
                            <h5>代码块详情 (共 {len(blocks)} 个)</h5>
                            <div class="step-scroll-container">
                                {blocks_preview}
                            </div>
                        </div>
                    </div>
                </div>
            """

        # Step 2: 依赖分析
        if history.prompt40_step2_dependency:
            step2 = history.prompt40_step2_dependency
            has_cycles = step2.get('has_cycles', False)
            dead_code_count = step2.get('dead_code_count', 0)
            dead_code_blocks = step2.get('dead_code_blocks', [])
            node_count = step2.get('node_count', 0)
            edge_count = step2.get('edge_count', 0)
            topological_order = step2.get('topological_order', [])

            dead_code_str = ""
            if dead_code_blocks:
                dead_code_str = ", ".join(dead_code_blocks)  # 显示所有死代码块

            topological_str = ""
            if topological_order:
                topological_str = " → ".join(topological_order)  # 显示完整拓扑排序

            html += f"""
                <div class="step-card">
                    <div class="step-header">
                        <span class="step-number">2</span>
                        <span class="step-title">依赖分析</span>
                        <span class="step-duration"></span>
                    </div>
                    <div class="step-body">
                        <div class="step-section">
                            <h5>图结构</h5>
                            <div class="change-item">节点数量: {node_count}</div>
                            <div class="change-item">边数量: {edge_count}</div>
                            <div class="change-item">循环依赖: {'是 ❌' if has_cycles else '否 ✅'}</div>
                        </div>
                        <div class="step-section">
                            <h5>死代码检测</h5>
                            <div class="change-item">发现 {dead_code_count} 个死代码块</div>
                            <div class="step-scroll-container" style="max-height:100px;">
                                {f'<div class="change-item">死代码: {dead_code_str}</div>' if dead_code_blocks else ''}
                            </div>
                        </div>
                        <div class="step-section">
                            <h5>拓扑排序 (共 {len(topological_order)} 个)</h5>
                            <div class="step-scroll-container" style="max-height:100px;">
                                <div class="change-item" style="word-break:break-all;">{topological_str or '无'}</div>
                            </div>
                        </div>
                    </div>
                </div>
            """

        # Step 3: 模块聚类
        if history.prompt40_step3_clustering:
            step3 = history.prompt40_step3_clustering
            strategy = step3.get('strategy', 'hybrid')
            total_clusters = step3.get('total_clusters', 0)
            clusters = step3.get('clusters', [])

            clusters_str = ""
            if clusters:
                for cluster in clusters:  # 显示所有簇
                    cluster_id = cluster.get('cluster_id', 0)
                    block_count = cluster.get('block_count', 0)
                    block_ids = ", ".join(cluster.get('blocks', []))  # 显示所有块ID
                    clusters_str += f"""
                        <div class="step-item-card">
                            <strong>模块 {cluster_id}</strong> ({block_count} 个代码块)<br>
                            &nbsp;&nbsp;代码块: {block_ids}
                        </div>
                    """

            html += f"""
                <div class="step-card">
                    <div class="step-header">
                        <span class="step-number">3</span>
                        <span class="step-title">模块聚类</span>
                        <span class="step-duration"></span>
                    </div>
                    <div class="step-body">
                        <div class="step-section">
                            <h5>聚类策略</h5>
                            <div class="change-item">策略类型: <strong>{strategy}</strong></div>
                            <div class="change-item">模块总数: {total_clusters}</div>
                        </div>
                        <div class="step-section">
                            <h5>聚类结果 (共 {len(clusters)} 个模块)</h5>
                            <div class="step-scroll-container">
                                {clusters_str}
                            </div>
                        </div>
                    </div>
                </div>
            """

        # Step 4: 代码生成
        if history.prompt40_step4_generation:
            step4 = history.prompt40_step4_generation
            total_modules = step4.get('total_modules', 0)
            async_modules = step4.get('async_modules', 0)
            sync_modules = step4.get('sync_modules', 0)
            modules = step4.get('modules', [])

            modules_str = ""
            if modules:
                for i, module in enumerate(modules):  # 显示所有模块
                    name = module.get('name', 'N/A')
                    inputs = module.get('inputs', [])
                    outputs = module.get('outputs', [])
                    is_async = module.get('is_async', False)
                    body_code = module.get('body_code', '')
                    
                    inputs_str = ", ".join(inputs) if inputs else "无"
                    outputs_str = ", ".join(outputs) if outputs else "无"
                    mode_str = '<span class="badge-async">异步</span>' if is_async else '<span class="badge-sync">同步</span>'
                    
                    # 转义代码中的特殊字符，完整显示
                    if body_code:
                        escaped_code = body_code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        code_preview_html = f'<pre class="step-code-block">{escaped_code}</pre>'
                    else:
                        code_preview_html = '<p style="color:#999; font-style:italic;">暂无函数体代码</p>'
                    
                    modules_str += f"""
                        <div class="step-item-card">
                            <strong>{i}. {name}</strong><br>
                            &nbsp;&nbsp;输入: {inputs_str or '无'}<br>
                            &nbsp;&nbsp;输出: {outputs_str or '无'}<br>
                            &nbsp;&nbsp;{mode_str}
                        </div>
                        <div style="margin-top:10px; padding:10px; background:#f9f9fa; border-left:4px solid #f093fb; border-radius:4px;">
                            <strong style="color:#f093fb; display:block; margin-bottom:5px;">函数实现:</strong>
                            {code_preview_html}
                        </div>
                    """

            html += f"""
                <div class="step-card">
                    <div class="step-header">
                        <span class="step-number">4</span>
                        <span class="step-title">代码生成</span>
                        <span class="step-duration"></span>
                    </div>
                    <div class="step-body">
                        <div class="step-section">
                            <h5>生成统计</h5>
                            <div class="change-item">总模块数: {total_modules}</div>
                            <div class="change-item">异步模块: {async_modules}</div>
                            <div class="change-item">同步模块: {sync_modules}</div>
                        </div>
                        <div class="step-section">
                            <h5>模块详情 (共 {len(modules)} 个)</h5>
                            <div class="step-scroll-container">
                                {modules_str}
                            </div>
                        </div>
                    </div>
                </div>
            """

        # Step 5: 主控编排
        if history.prompt40_step5_orchestration:
            step5 = history.prompt40_step5_orchestration
            main_inputs = step5.get('main_inputs', [])
            input_count = step5.get('input_count', 0)
            main_code = step5.get('main_code', '')

            inputs_str = ", ".join(main_inputs) if main_inputs else "无"

            main_code_preview = ""
            if main_code:
                # 完整显示主代码
                escaped_code = main_code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                main_code_preview = f'<pre class="step-code-block" style="max-height:400px;">{escaped_code}</pre>'

            html += f"""
                <div class="step-card">
                    <div class="step-header">
                        <span class="step-number">5</span>
                        <span class="step-title">主控编排</span>
                        <span class="step-duration"></span>
                    </div>
                    <div class="step-body">
                        <div class="step-section">
                            <h5>外部输入参数</h5>
                            <div class="change-item">参数数量: {input_count}</div>
                            <div class="change-item" style="word-break:break-all;">{inputs_str}</div>
                        </div>
                        <div class="step-section">
                            <h5>主工作流代码 (完整)</h5>
                            {main_code_preview}
                        </div>
                    </div>
                </div>
            """

        html += '</div>'
        return html
    
    def __init__(self, storage_dir: str = "processing_history"):
        """
        初始化历史管理器

        Args:
            storage_dir: 存储目录路径
        """
        self.storage_dir = storage_dir
        self.history_file = os.path.join(storage_dir, "history.json")
        self._ensure_storage_dir()

        # ===== 初始化智能优化与可解释性模块 =====
        self._init_intelligence_modules()

    def _init_intelligence_modules(self):
        """初始化智能优化与可解释性模块"""
        try:
            # 延迟导入，避免循环依赖
            from utils.adaptive_terminology import AdaptiveTermLearner
            from utils.contextual_ambiguity_resolver import ContextualAmbiguityResolver
            from utils.explainability import ExplainabilityEngine

            # 术语学习器
            self.term_learner = AdaptiveTermLearner()

            # 歧义消解器
            self.ambiguity_resolver = ContextualAmbiguityResolver()

            # 可解释性引擎
            self.explainer = ExplainabilityEngine()

            info("智能优化与可解释性模块初始化完成")
        except ImportError as e:
            warning(f"智能优化模块导入失败: {e}")
            self.term_learner = None
            self.ambiguity_resolver = None
            self.explainer = None
    
    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            info(f"创建历史记录存储目录: {self.storage_dir}")
    
    def save_history(self, history: ProcessingHistory) -> str:
        """
        保存处理历史
        
        Args:
            history: 处理历史记录
            
        Returns:
            记录ID（时间戳）
        """
        # 加载现有历史
        all_history = self.load_all_history()
        
        # 添加新记录
        all_history[history.timestamp] = asdict(history)
        
        # 保存到文件
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(all_history, f, ensure_ascii=False, indent=2)
            info(f"历史记录已保存: {history.timestamp}")
            return history.timestamp
        except Exception as e:
            error(f"保存历史记录失败: {e}")
            raise
    
    def load_all_history(self) -> Dict[str, Dict]:
        """加载所有历史记录"""
        if not os.path.exists(self.history_file):
            return {}
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            warning(f"加载历史记录失败: {e}")
            return {}
    
    def get_history(self, timestamp: str) -> Optional[ProcessingHistory]:
        """获取指定时间戳的历史记录"""
        all_history = self.load_all_history()
        record = all_history.get(timestamp)
        if record:
            return ProcessingHistory(**record)
        return None
    
    def get_recent_history(self, limit: int = 10) -> List[ProcessingHistory]:
        """获取最近的处理历史"""
        all_history = self.load_all_history()
        # 按时间戳排序（最新的在前）
        sorted_timestamps = sorted(all_history.keys(), reverse=True)
        recent_timestamps = sorted_timestamps[:limit]
        
        return [
            ProcessingHistory(**all_history[ts])
            for ts in recent_timestamps
        ]
    
    def format_comparison(self, history: ProcessingHistory) -> str:
        """
        格式化对比展示
        
        Args:
            history: 处理历史记录
            
        Returns:
            格式化的对比文本
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"处理时间: {history.timestamp}")
        lines.append(f"处理模式: {history.mode}")
        lines.append(f"处理状态: {'✅ 成功' if history.success else '⚠️ 检测到歧义'}")
        lines.append("=" * 80)
        
        # 原始文本 vs 处理后文本对比
        lines.append("\n【文本对比】")
        lines.append("-" * 80)
        lines.append("原始文本:")
        lines.append(f"  {history.original_text}")
        lines.append("\n处理后文本:")
        lines.append(f"  {history.processed_text}")
        lines.append("-" * 80)
        
        # 术语替换
        if history.terminology_changes:
            lines.append("\n【术语替换】")
            lines.append("-" * 80)
            for old, new in history.terminology_changes.items():
                lines.append(f"  {old} → {new}")
            lines.append("-" * 80)
        
        # 处理步骤
        if history.steps_log:
            lines.append("\n【处理步骤】")
            lines.append("-" * 80)
            for i, step in enumerate(history.steps_log, 1):
                lines.append(f"  {i}. {step}")
            lines.append("-" * 80)
        
        # 警告信息
        if history.warnings:
            lines.append("\n【警告信息】")
            lines.append("-" * 80)
            for warning_msg in history.warnings:
                lines.append(f"  ⚠️  {warning_msg}")
            lines.append("-" * 80)
        
        # 歧义检测
        if history.ambiguity_detected:
            lines.append("\n【歧义检测】")
            lines.append("-" * 80)
            lines.append("  ⚠️  检测到歧义，已拦截")
            lines.append("-" * 80)
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def print_comparison(self, history: ProcessingHistory):
        """打印对比展示"""
        comparison_text = self.format_comparison(history)
        info("\n" + comparison_text)
    
    def export_comparison_html(self, history: ProcessingHistory, output_file: Optional[str] = None) -> str:
        """
        导出为HTML格式的对比展示
        
        Args:
            history: 处理历史记录
            output_file: 输出文件路径（可选）
            
        Returns:
            HTML内容
        """
        if output_file is None:
            output_file = os.path.join(
                self.storage_dir,
                f"comparison_{history.timestamp.replace(':', '-').replace(' ', '_')}.html"
            )
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>处理对比 - {history.timestamp}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .meta {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .meta-item {{
            margin: 5px 0;
        }}
        .status-success {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .status-warning {{
            color: #FF9800;
            font-weight: bold;
        }}
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .text-box {{
            border: 2px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background: #fafafa;
        }}
        .text-box.original {{
            border-color: #ff6b6b;
        }}
        .text-box.processed {{
            border-color: #4CAF50;
        }}
        .text-box h3 {{
            margin-top: 0;
            color: #333;
        }}
        .text-content {{
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
        }}
        .changes {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .changes h3 {{
            margin-top: 0;
            color: #856404;
        }}
        .change-item {{
            margin: 8px 0;
            padding: 8px;
            background: white;
            border-radius: 3px;
        }}
        .old {{
            color: #d32f2f;
            text-decoration: line-through;
        }}
        .new {{
            color: #388e3c;
            font-weight: bold;
        }}
        .steps {{
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .steps h3 {{
            margin-top: 0;
            color: #1565c0;
        }}
        .step-item {{
            margin: 8px 0;
            padding: 8px;
            background: white;
            border-radius: 3px;
        }}
        .warnings {{
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .warnings h3 {{
            margin-top: 0;
            color: #e65100;
        }}
        .warning-item {{
            margin: 8px 0;
            padding: 8px;
            background: white;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📝 处理对比报告</h1>
        
        <div class="meta">
            <div class="meta-item"><strong>处理时间:</strong> {history.timestamp}</div>
            <div class="meta-item"><strong>处理模式:</strong> {history.mode}</div>
            <div class="meta-item"><strong>处理状态:</strong> 
                <span class="{'status-success' if history.success else 'status-warning'}">
                    {'✅ 成功' if history.success else '⚠️ 检测到歧义'}
                </span>
            </div>
        </div>
        
        <div class="comparison">
            <div class="text-box original">
                <h3>📄 原始文本</h3>
                <div class="text-content">{history.original_text}</div>
            </div>
            <div class="text-box processed">
                <h3>✨ 处理后文本</h3>
                <div class="text-content">{history.processed_text}</div>
            </div>
        </div>
"""
        
        # 术语替换
        if history.terminology_changes:
            html += """
        <div class="changes">
            <h3>🔄 术语替换</h3>
"""
            for old, new in history.terminology_changes.items():
                html += f"""
            <div class="change-item">
                <span class="old">{old}</span> → <span class="new">{new}</span>
            </div>
"""
            html += """
        </div>
"""
        
        # 处理步骤
        if history.steps_log:
            html += """
        <div class="steps">
            <h3>⚙️ 处理步骤</h3>
"""
            for i, step in enumerate(history.steps_log, 1):
                html += f"""
            <div class="step-item">{i}. {step}</div>
"""
            html += """
        </div>
"""
        
        # 警告信息
        if history.warnings:
            html += """
        <div class="warnings">
            <h3>⚠️ 警告信息</h3>
"""
            for warning_msg in history.warnings:
                html += f"""
            <div class="warning-item">{warning_msg}</div>
"""
            html += """
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        # 保存HTML文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            info(f"HTML对比报告已保存: {output_file}")
        except Exception as e:
            error(f"保存HTML报告失败: {e}")
        
        return html
    
    # ========================================================================
    # Prompt 2.0 历史记录管理
    # ========================================================================
    
    def save_prompt20_history(self, history: Prompt20History) -> str:
        """
        保存 Prompt 2.0 处理历史
        
        Args:
            history: Prompt 2.0 历史记录
            
        Returns:
            记录ID
        """
        prompt20_file = os.path.join(self.storage_dir, "prompt20_history.json")
        
        # 加载现有历史
        all_history = {}
        if os.path.exists(prompt20_file):
            try:
                with open(prompt20_file, 'r', encoding='utf-8') as f:
                    all_history = json.load(f)
            except Exception:
                pass
        
        # 添加新记录
        all_history[history.id] = asdict(history)
        
        # 保存到文件
        try:
            with open(prompt20_file, 'w', encoding='utf-8') as f:
                json.dump(all_history, f, ensure_ascii=False, indent=2)
            info(f"Prompt 2.0 历史记录已保存: {history.id}")
            return history.id
        except Exception as e:
            error(f"保存 Prompt 2.0 历史记录失败: {e}")
            raise
    
    def load_prompt20_history(self, record_id: str) -> Optional[Prompt20History]:
        """加载指定的 Prompt 2.0 历史记录"""
        prompt20_file = os.path.join(self.storage_dir, "prompt20_history.json")
        
        if not os.path.exists(prompt20_file):
            return None
        
        try:
            with open(prompt20_file, 'r', encoding='utf-8') as f:
                all_history = json.load(f)
            record = all_history.get(record_id)
            if record:
                return Prompt20History(**record)
        except Exception as e:
            warning(f"加载 Prompt 2.0 历史记录失败: {e}")
        
        return None
    
    def get_recent_prompt20_history(self, limit: int = 10) -> List[Prompt20History]:
        """获取最近的 Prompt 2.0 处理历史"""
        prompt20_file = os.path.join(self.storage_dir, "prompt20_history.json")
        
        if not os.path.exists(prompt20_file):
            return []
        
        try:
            with open(prompt20_file, 'r', encoding='utf-8') as f:
                all_history = json.load(f)
            
            # 按时间戳排序
            sorted_ids = sorted(
                all_history.keys(),
                key=lambda x: all_history[x].get('timestamp', ''),
                reverse=True
            )
            
            return [
                Prompt20History(**all_history[id])
                for id in sorted_ids[:limit]
            ]
        except Exception as e:
            warning(f"加载 Prompt 2.0 历史记录失败: {e}")
            return []
    
    # ========================================================================
    # 完整流水线历史记录管理
    # ========================================================================
    
    def save_pipeline_history(self, history: PipelineHistory) -> str:
        """
        保存完整流水线历史记录
        
        Args:
            history: 流水线历史记录
            
        Returns:
            流水线ID
        """
        pipeline_file = os.path.join(self.storage_dir, "pipeline_history.json")
        
        # 加载现有历史
        all_history = {}
        if os.path.exists(pipeline_file):
            try:
                with open(pipeline_file, 'r', encoding='utf-8') as f:
                    all_history = json.load(f)
            except Exception:
                pass
        
        # 添加新记录
        all_history[history.pipeline_id] = asdict(history)
        
        # 保存到文件
        try:
            with open(pipeline_file, 'w', encoding='utf-8') as f:
                json.dump(all_history, f, ensure_ascii=False, indent=2)
            info(f"流水线历史记录已保存: {history.pipeline_id}")
            return history.pipeline_id
        except Exception as e:
            error(f"保存流水线历史记录失败: {e}")
            raise
    
    def load_pipeline_history(self, pipeline_id: str) -> Optional[PipelineHistory]:
        """加载指定的流水线历史记录"""
        pipeline_file = os.path.join(self.storage_dir, "pipeline_history.json")
        
        if not os.path.exists(pipeline_file):
            return None
        
        try:
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                all_history = json.load(f)
            record = all_history.get(pipeline_id)
            if record:
                return PipelineHistory(**record)
        except Exception as e:
            warning(f"加载流水线历史记录失败: {e}")
        
        return None
    
    def get_recent_pipeline_history(self, limit: int = 10) -> List[PipelineHistory]:
        """获取最近的流水线历史"""
        pipeline_file = os.path.join(self.storage_dir, "pipeline_history.json")
        
        if not os.path.exists(pipeline_file):
            return []
        
        try:
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                all_history = json.load(f)
            
            sorted_ids = sorted(
                all_history.keys(),
                key=lambda x: all_history[x].get('timestamp', ''),
                reverse=True
            )
            
            return [
                PipelineHistory(**all_history[id])
                for id in sorted_ids[:limit]
            ]
        except Exception as e:
            warning(f"加载流水线历史记录失败: {e}")
            return []

    # ========================================================================
    # 智能优化与可解释性集成方法
    # ========================================================================

    def process_with_intelligence(self, text: str) -> Dict[str, Any]:
        """
        使用智能优化处理文本

        Args:
            text: 输入文本

        Returns:
            包含处理结果和追溯数据的字典
        """
        result = {
            "processed_text": text,
            "term_learning": {},
            "ambiguity_resolution": {},
            "decision_trace": {},
            "visualization_data": {}
        }

        # 如果模块未初始化，返回原始结果
        if not self.term_learner or not self.ambiguity_resolver or not self.explainer:
            warning("智能优化模块未初始化")
            return result

        # 1. 术语学习与歧义检测
        try:
            # 术语学习
            self.term_learner.learn_from_text(text)
            term_suggestions = self.term_learner.analyze_candidates()

            # 歧义检测
            ambiguity_result = self.ambiguity_resolver.resolve(text)

            result["term_learning"] = {
                "detected_terms": term_suggestions,
                "confirmed_terms": self.term_learner.get_term_mapping(),
                "learning_count": self.term_learner.get_statistics()
            }

            result["ambiguity_resolution"] = {
                "original_text": ambiguity_result.original_text,
                "resolved_text": ambiguity_result.resolved_text,
                "detected_count": len(ambiguity_result.ambiguities_detected),
                "resolved_count": len(ambiguity_result.resolutions),
                "detected_ambiguities": []
            }
            # 安全地处理歧义类型
            for a in ambiguity_result.ambiguities_detected:
                amb_type = a.ambiguity_type
                if hasattr(amb_type, 'value'):
                    amb_type_str = amb_type.value
                elif hasattr(amb_type, 'name'):
                    amb_type_str = amb_type.name
                else:
                    amb_type_str = str(amb_type)
                result["ambiguity_resolution"]["detected_ambiguities"].append({
                    "type": amb_type_str,
                    "span": a.ambiguous_span,
                    "context": a.surrounding_text
                })
        except Exception as e:
            warning(f"智能处理失败: {e}")

        # 2. 决策追溯
        try:
            from utils.explainability import EvidenceType

            trace_id = self.explainer.start_trace(text)

            # 记录术语处理步骤
            self.explainer.start_step("术语学习", 1, text)
            self.explainer.add_evidence(
                EvidenceType.RULE_MATCH,
                f"检测到 {len(term_suggestions)} 个候选术语",
                confidence=0.9,
                source="AdaptiveTermLearner"
            )
            self.explainer.end_step(
                output_data={"term_count": len(term_suggestions)},
                status="success",
                duration_ms=10
            )

            # 记录歧义处理步骤
            self.explainer.start_step("歧义消解", 2, text)
            if ambiguity_result.ambiguities_detected:
                self.explainer.add_evidence(
                    EvidenceType.PATTERN_DETECTED,
                    f"检测到 {len(ambiguity_result.ambiguities_detected)} 个歧义",
                    confidence=0.85,
                    source="ContextualAmbiguityResolver"
                )
            self.explainer.end_step(
                output_data={"resolved": ambiguity_result.resolved_text},
                status="success",
                duration_ms=15
            )

            trace = self.explainer.finalize_trace(result["processed_text"])
            result["decision_trace"] = {
                "trace_id": trace.trace_id,
                "steps": [
                    {
                        "name": s.step_name,
                        "confidence": sum(e.confidence for e in s.evidences) / len(s.evidences) if s.evidences else 0.0,
                        "reasoning": s.reasoning,
                        "evidences": [
                            {"content": e.content, "confidence": e.confidence}
                            for e in s.evidences
                        ]
                    }
                    for s in trace.steps
                ],
                "overall_confidence": trace.overall_confidence,
                "explanation": self.explainer.explain_decision(trace)
            }

            # 3. 生成可视化数据 - 使用已生成的 mermaid_flowchart
            result["visualization_data"]["mermaid_flowchart"] = self.explainer.visualize_trace(trace)

        except Exception as e:
            warning(f"决策追溯失败: {e}")

        return result

    def generate_intelligence_report(self, history: PipelineHistory) -> str:
        """
        生成智能优化与可解释性 HTML 报告

        Args:
            history: 流水线历史记录

        Returns:
            HTML报告内容
        """
        # 收集数据
        term_data = history.term_learning or {}
        ambiguity_data = history.ambiguity_resolution or {}
        trace_data = history.decision_trace or {}
        viz_data = history.visualization_data or {}

        # 术语学习统计
        detected_terms = term_data.get("detected_terms", [])
        confirmed_terms = term_data.get("confirmed_terms", {})

        # 歧义数据
        detected_ambiguities = ambiguity_data.get("detected_ambiguities", [])
        resolved_count = ambiguity_data.get("resolved_count", 0)

        # 追溯数据
        steps = trace_data.get("steps", [])
        overall_confidence = trace_data.get("overall_confidence", 0)
        explanation = trace_data.get("explanation", "")

        # 生成HTML
        html = f"""
        <!-- 智能优化与可解释性报告 -->
        <div class="intelligence-section">
            <h2>🧠 智能优化与可解释性报告</h2>

            <!-- 置信度概览 -->
            <div class="confidence-overview">
                <div class="confidence-card">
                    <div class="confidence-label">整体置信度</div>
                    <div class="confidence-value">{overall_confidence * 100:.1f}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {overall_confidence * 100}%"></div>
                    </div>
                </div>
                <div class="confidence-card">
                    <div class="confidence-label">术语学习</div>
                    <div class="confidence-value">{len(confirmed_terms)} 个</div>
                </div>
                <div class="confidence-card">
                    <div class="confidence-label">歧义消解</div>
                    <div class="confidence-value">{resolved_count} 个</div>
                </div>
            </div>

            <!-- 决策追溯 -->
            <div class="trace-section">
                <h3>🔍 决策追溯</h3>
                <div class="steps-timeline">
"""

        for i, step in enumerate(steps, 1):
            step_confidence = step.get("confidence", 0) * 100
            step_evidences = step.get("evidences", [])

            html += f"""
                    <div class="timeline-step">
                        <div class="step-badge">{i}</div>
                        <div class="step-content">
                            <div class="step-title">{step.get("name", f"步骤 {i}")}</div>
                            <div class="step-confidence">置信度: {step_confidence:.1f}%</div>
                            <div class="step-reasoning">{step.get("reasoning", "")}</div>
                            <div class="step-evidences">
"""
            for ev in step_evidences:
                html += f'                                <span class="evidence-tag">{ev.get("content", "")}</span>\n'
            html += """                            </div>
                        </div>
                    </div>
"""

        html += """
                </div>
            </div>
"""

        # Mermaid 流程图
        mermaid_flow = viz_data.get("mermaid_flowchart", "")
        if mermaid_flow:
            html += f"""
            <!-- Mermaid 流程图 -->
            <div class="mermaid-section">
                <h3>📊 决策流程图</h3>
                <div class="mermaid-diagram">
                    <pre class="mermaid">
{mermaid_flow}
                    </pre>
                </div>
            </div>
"""

        html += """
        </div>

        <style>
        .intelligence-section {
            margin-top: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            color: white;
        }
        .intelligence-section h2 {
            margin-top: 0;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }
        .confidence-overview {
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }
        .confidence-card {
            flex: 1;
            background: rgba(255,255,255,0.15);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .confidence-label {
            font-size: 14px;
            opacity: 0.8;
        }
        .confidence-value {
            font-size: 28px;
            font-weight: bold;
            margin: 10px 0;
        }
        .confidence-bar {
            height: 6px;
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: #4CAF50;
            border-radius: 3px;
            transition: width 0.3s;
        }
        .trace-section {
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        .steps-timeline {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .timeline-step {
            display: flex;
            align-items: flex-start;
            gap: 15px;
        }
        .step-badge {
            width: 32px;
            height: 32px;
            background: #4CAF50;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }
        .step-content {
            flex: 1;
            background: rgba(255,255,255,0.9);
            color: #333;
            border-radius: 8px;
            padding: 15px;
        }
        .step-title {
            font-weight: bold;
            font-size: 16px;
        }
        .step-confidence {
            color: #667eea;
            font-size: 14px;
            margin: 5px 0;
        }
        .step-reasoning {
            font-size: 14px;
            margin: 10px 0;
            color: #666;
        }
        .step-evidences {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .evidence-tag {
            background: #e8f5e9;
            color: #2e7d32;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
        }
        .mermaid-section {
            margin-top: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 20px;
        }
        .mermaid-diagram {
            background: white;
            border-radius: 8px;
            padding: 15px;
            overflow-x: auto;
        }
        </style>
"""

        return html

    def export_pipeline_with_intelligence_html(
        self,
        history: PipelineHistory,
        output_file: Optional[str] = None
    ) -> str:
        """
        导出包含智能优化与可解释性的流水线HTML报告

        Args:
            history: 流水线历史记录
            output_file: 输出文件路径

        Returns:
            HTML内容
        """
        # 先调用原有的 HTML 生成方法
        base_html = self.export_pipeline_html(history, output_file)

        # 如果模块未初始化，返回基础HTML
        if not self.term_learner or not self.explainer:
            return base_html

        # 生成智能报告
        intelligence_html = self.generate_intelligence_report(history)

        # 插入到基础HTML中（在 </body> 之前）
        if "</body>" in base_html:
            html = base_html.replace("</body>", f"{intelligence_html}\n</body>")
        else:
            html = base_html + intelligence_html

        # 保存更新后的HTML
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                info(f"智能优化报告已保存: {output_file}")
            except Exception as e:
                error(f"保存智能优化报告失败: {e}")

        return html

    # ========================================================================
    # 完整流水线对比展示
    # ========================================================================
    
    def format_pipeline_comparison(self, history: PipelineHistory) -> str:
        """
        格式化完整流水线对比展示
        
        Args:
            history: 流水线历史记录
            
        Returns:
            格式化的对比文本
        """
        lines = []
        
        # 标题
        lines.append("█" * 80)
        lines.append("█" + " " * 28 + "完整流水线处理报告" + " " * 29 + "█")
        lines.append("█" * 80)
        lines.append("")
        lines.append(f"流水线 ID: {history.pipeline_id}")
        lines.append(f"处理时间: {history.timestamp}")
        lines.append(f"整体状态: {history.overall_status}")
        lines.append(f"总耗时: {history.total_time_ms}ms")
        lines.append("")
        
        # ===== 阶段 1: Prompt 1.0 =====
        lines.append("=" * 80)
        lines.append("【阶段 1: Prompt 1.0 预处理】")
        lines.append("=" * 80)
        lines.append("")
        lines.append("┌─ 原始输入 ─────────────────────────────────────────────────────────────────┐")
        for line in history.raw_input.split('\n'):
            lines.append(f"│ {line}")
        lines.append("└────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")
        lines.append("┌─ 标准化输出 (Prompt 1.0) ───────────────────────────────────────────────────┐")
        for line in history.prompt10_processed.split('\n'):
            lines.append(f"│ {line}")
        lines.append("└────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # 处理步骤详情
        if history.prompt10_steps:
            lines.append("【处理步骤详情】")
            for i, step in enumerate(history.prompt10_steps, 1):
                lines.append(f"\n  步骤 {i}: {step.get('step_name', 'N/A')}")
                lines.append(f"    耗时: {step.get('duration_ms', 0)}ms")
                changes = step.get('changes', {})
                if changes:
                    lines.append(f"    变更: {len(changes)} 处")
                    for old, new in list(changes.items())[:3]:  # 只显示前3个
                        new_str = f"'{new}'" if new else "(删除)"
                        lines.append(f"      • '{old}' → {new_str}")
                    if len(changes) > 3:
                        lines.append(f"      ... 还有 {len(changes) - 3} 处变更")
            lines.append("")
        
        # 术语替换
        if history.prompt10_terminology_changes:
            lines.append("【术语替换】")
            for old, new in history.prompt10_terminology_changes.items():
                if new:
                    lines.append(f"  • '{old}' → '{new}'")
                else:
                    lines.append(f"  • '{old}' → (删除)")
            lines.append("")
        
        lines.append(f"处理耗时: {history.prompt10_time_ms}ms | 状态: {history.prompt10_status}")
        lines.append("")
        
        # ===== 阶段 2: Prompt 2.0 =====
        lines.append("=" * 80)
        lines.append("【阶段 2: Prompt 2.0 结构化】")
        lines.append("=" * 80)
        lines.append("")
        lines.append("┌─ 参数化模板 (Prompt 2.0) ───────────────────────────────────────────────────┐")
        for line in history.prompt20_template.split('\n'):
            lines.append(f"│ {line}")
        lines.append("└────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # 变量注册表
        lines.append("【变量注册表】")
        lines.append(f"共 {history.prompt20_variable_count} 个变量")
        if history.prompt20_type_stats:
            stats_str = ", ".join([f"{k}: {v}" for k, v in history.prompt20_type_stats.items()])
            lines.append(f"类型分布: {stats_str}")
        lines.append("")
        
        for var in history.prompt20_variables[:10]:  # 只显示前10个
            lines.append(f"  • {var.get('variable', 'N/A')}: {var.get('value', 'N/A')} ({var.get('type', 'N/A')})")
            lines.append(f"    原文: \"{var.get('original_text', 'N/A')}\"")
        
        if len(history.prompt20_variables) > 10:
            lines.append(f"  ... 还有 {len(history.prompt20_variables) - 10} 个变量")
        
        lines.append("")
        lines.append(f"处理耗时: {history.prompt20_time_ms}ms")
        lines.append("")

        # ===== 阶段 3: Prompt 3.0 DSL 编译 =====
        lines.append("=" * 80)
        lines.append("【阶段 3: Prompt 3.0 DSL 编译】")
        lines.append("=" * 80)
        lines.append("")
        
        if history.prompt30_dsl_code:
            lines.append("┌─ 生成的 DSL 代码 (Prompt 3.0) ───────────────────────────────────────────────┐")
            dsl_lines = history.prompt30_dsl_code.split('\n')
            for line in dsl_lines[:20]:  # 最多显示20行
                lines.append(f"│ {line}")
            if len(dsl_lines) > 20:
                lines.append(f"│ ... 还有 {len(dsl_lines) - 20} 行")
            lines.append("└────────────────────────────────────────────────────────────────────────────┘")
            lines.append("")
            
            # 验证结果
            if history.prompt30_validation_result:
                valid = history.prompt30_validation_result.get('is_valid', False)
                errors = history.prompt30_validation_result.get('errors', [])
                warnings = history.prompt30_validation_result.get('warnings', [])
                defined_vars = history.prompt30_validation_result.get('defined_variables', {})
                function_calls = history.prompt30_validation_result.get('function_calls', [])
                
                lines.append(f"验证状态: {'✅ 通过' if valid else '❌ 失败'}")
                lines.append(f"定义变量数: {len(defined_vars)} 个")
                lines.append(f"函数调用数: {len(function_calls)} 个")
                if errors:
                    lines.append(f"错误数量: {len(errors)} 个")
                if warnings:
                    lines.append(f"警告数量: {len(warnings)} 个")
        else:
            lines.append("  未进行 DSL 编译")
        
        lines.append("")
        lines.append(f"处理耗时: {history.prompt30_time_ms}ms")
        lines.append("")

        # ===== 阶段 4: Prompt 4.0 代码生成 =====
        lines.append("=" * 80)
        lines.append("【阶段 4: Prompt 4.0 代码生成】")
        lines.append("=" * 80)
        lines.append("")
        
        if history.prompt40_modules:
            lines.append(f"【工作流模块】共 {history.prompt40_module_count} 个模块")
            lines.append("")
            for i, module in enumerate(history.prompt40_modules, 1):
                module_name = module.get('name', 'N/A')
                inputs = module.get('inputs', [])
                outputs = module.get('outputs', [])
                is_async = module.get('is_async', False)
                lines.append(f"  模块 {i}: {module_name}")
                lines.append(f"    输入变量: {', '.join(inputs) if inputs else '无'}")
                lines.append(f"    输出变量: {', '.join(outputs) if outputs else '无'}")
                lines.append(f"    执行模式: {'异步' if is_async else '同步'}")
                lines.append("")
            
            lines.append("┌─ 主工作流代码 (Prompt 4.0) ───────────────────────────────────────────────┐")
            code_lines = history.prompt40_main_code.split('\n')
            for line in code_lines[:30]:  # 最多显示30行
                lines.append(f"│ {line}")
            if len(code_lines) > 30:
                lines.append(f"│ ... 还有 {len(code_lines) - 30} 行")
            lines.append("└────────────────────────────────────────────────────────────────────────────┘")
        else:
            lines.append("  未进行代码生成")
        
        lines.append("")
        lines.append(f"处理耗时: {history.prompt40_time_ms}ms")
        lines.append("")

        # ===== 总结 =====
        lines.append("=" * 80)
        lines.append("【处理总结】")
        lines.append("=" * 80)
        lines.append(f"  原始输入长度: {len(history.raw_input)} 字符")
        lines.append(f"  标准化后长度: {len(history.prompt10_processed)} 字符")
        lines.append(f"  识别变量数量: {history.prompt20_variable_count} 个")
        lines.append(f"  DSL 编译状态: {'✅ 成功' if history.prompt30_dsl_code else '❌ 未执行'}")
        lines.append(f"  代码生成状态: {'✅ 成功' if history.prompt40_modules else '❌ 未执行'}")
        lines.append(f"  总处理耗时: {history.total_time_ms}ms")
        lines.append("")
        lines.append(f"  阶段 1 (预处理): {history.prompt10_time_ms}ms ({history.prompt10_time_ms / history.total_time_ms * 100:.1f}%)")
        lines.append(f"  阶段 2 (结构化): {history.prompt20_time_ms}ms ({history.prompt20_time_ms / history.total_time_ms * 100:.1f}%)")
        lines.append(f"  阶段 3 (DSL编译): {history.prompt30_time_ms}ms ({history.prompt30_time_ms / history.total_time_ms * 100:.1f}%)")
        lines.append(f"  阶段 4 (代码生成): {history.prompt40_time_ms}ms ({history.prompt40_time_ms / history.total_time_ms * 100:.1f}%)")
        lines.append("█" * 80)
        
        return "\n".join(lines)
    
    def print_pipeline_comparison(self, history: PipelineHistory):
        """打印流水线对比展示"""
        comparison_text = self.format_pipeline_comparison(history)
        info("\n" + comparison_text)
    
    def _generate_intelligence_section(self, history: PipelineHistory) -> str:
        """
        生成智能优化与可解释性报告HTML段落
        
        完整展示（带滑窗滚动效果）：
        1. 术语学习详情（候选术语、置信度、推荐理由）
        2. 歧义检测详情（歧义类型、位置、上下文）
        3. 决策追溯细节（步骤耗时、状态、证据详情）

        Args:
            history: 流水线历史记录

        Returns:
            HTML段落
        """
        term_data = history.term_learning or {}
        ambiguity_data = history.ambiguity_resolution or {}
        trace_data = history.decision_trace or {}
        viz_data = history.visualization_data or {}

        # 术语学习数据
        confirmed_terms = term_data.get("confirmed_terms", {})
        suggestions = term_data.get("suggestions", []) or term_data.get("detected_terms", [])
        known_mappings_count = term_data.get("known_mappings_count", 0)
        term_stats_count = term_data.get("term_stats_count", 0)

        # 歧义数据
        detected_ambiguities = ambiguity_data.get("ambiguities", []) or ambiguity_data.get("detected_ambiguities", [])
        detected_count = ambiguity_data.get("detected_count", 0)
        resolved_count = ambiguity_data.get("resolved_count", 0)

        # 追溯数据
        steps = trace_data.get("steps", [])
        overall_confidence = trace_data.get("overall_confidence", 0)
        trace_id = trace_data.get("trace_id", "N/A")

        # Mermaid流程图
        mermaid_flow = viz_data.get("mermaid_flowchart", "")

        # ========================================
        # 开始构建HTML - 添加CSS和JavaScript
        # ========================================
        html = f"""
        <style>
            /* 智能报告专用样式 */
            .intel-scroll-container {{
                max-height: 400px;
                overflow-y: auto;
                scrollbar-width: thin;
                scrollbar-color: #667eea #f1f1f1;
            }}
            .intel-scroll-container::-webkit-scrollbar {{
                width: 8px;
            }}
            .intel-scroll-container::-webkit-scrollbar-track {{
                background: #f1f1f1;
                border-radius: 4px;
            }}
            .intel-scroll-container::-webkit-scrollbar-thumb {{
                background: #667eea;
                border-radius: 4px;
            }}
            .intel-scroll-container::-webkit-scrollbar-thumb:hover {{
                background: #5a6fd6;
            }}
            .intel-code-block {{
                background: #1e1e1e;
                color: #d4d4d4;
                padding: 12px;
                border-radius: 6px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                white-space: pre-wrap;
                word-break: break-all;
                max-height: 150px;
                overflow-y: auto;
            }}
            .intel-expand-btn {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 11px;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .intel-expand-btn:hover {{
                transform: scale(1.05);
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
            }}
            .intel-detail-panel {{
                display: none;
                margin-top: 10px;
                animation: fadeIn 0.3s ease-in;
            }}
            .intel-detail-panel.show {{
                display: block;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(-10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .intel-tooltip {{
                position: relative;
                cursor: help;
            }}
            .intel-tooltip:hover::after {{
                content: attr(data-full);
                position: absolute;
                left: 0;
                top: 100%;
                background: #333;
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                white-space: normal;
                max-width: 300px;
                z-index: 1000;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
        </style>
        <script>
            function toggleDetail(id) {{
                const panel = document.getElementById(id);
                if (panel) {{
                    panel.classList.toggle('show');
                }}
            }}
            function toggleAll(containerClass) {{
                const containers = document.querySelectorAll('.' + containerClass);
                containers.forEach(c => {{
                    const isHidden = c.style.display === 'none' || c.classList.contains('collapsed');
                    if (isHidden) {{
                        c.style.display = 'block';
                        c.classList.remove('collapsed');
                    }} else {{
                        c.style.display = 'none';
                        c.classList.add('collapsed');
                    }}
                }});
            }}
        </script>

        <div class="intelligence-section" style="margin: 30px 0; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white;">
            <h2 style="margin-top: 0; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px;">
                🧠 智能优化与可解释性报告
            </h2>
            <div style="font-size: 12px; opacity: 0.7; margin-bottom: 15px;">
                追溯ID: <code style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 4px;">{trace_id}</code>
            </div>

            <!-- 统计概览卡片 -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0;">
                <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="font-size: 12px; opacity: 0.8;">整体置信度</div>
                    <div style="font-size: 28px; font-weight: bold; margin: 8px 0;">{overall_confidence * 100:.1f}%</div>
                    <div style="font-size: 11px; opacity: 0.6;">决策可靠性</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="font-size: 12px; opacity: 0.8;">处理步骤</div>
                    <div style="font-size: 28px; font-weight: bold; margin: 8px 0;">{len(steps)}</div>
                    <div style="font-size: 11px; opacity: 0.6;">决策追溯点</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="font-size: 12px; opacity: 0.8;">已确认术语</div>
                    <div style="font-size: 28px; font-weight: bold; margin: 8px 0;">{known_mappings_count}</div>
                    <div style="font-size: 11px; opacity: 0.6;">术语映射</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="font-size: 12px; opacity: 0.8;">候选术语</div>
                    <div style="font-size: 28px; font-weight: bold; margin: 8px 0;">{len(suggestions)}</div>
                    <div style="font-size: 11px; opacity: 0.6;">待确认建议</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="font-size: 12px; opacity: 0.8;">检测歧义</div>
                    <div style="font-size: 28px; font-weight: bold; margin: 8px 0;">{detected_count}</div>
                    <div style="font-size: 11px; opacity: 0.6;">潜在问题</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 15px; text-align: center;">
                    <div style="font-size: 12px; opacity: 0.8;">已消解歧义</div>
                    <div style="font-size: 28px; font-weight: bold; margin: 8px 0;">{resolved_count}</div>
                    <div style="font-size: 11px; opacity: 0.6;">自动处理</div>
                </div>
            </div>
"""

        # ========================================
        # 1. 术语学习详情 - 带滚动
        # ========================================
        html += """
            <!-- 术语学习详情 -->
            <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 20px; margin-top: 20px;">
                <h3 style="margin-top: 0; display: flex; align-items: center; gap: 8px;">
                    📚 术语学习详情
                </h3>
"""
        
        if suggestions:
            html += f"""
                <div style="background: rgba(255,255,255,0.95); border-radius: 8px; overflow: hidden; margin-top: 15px;">
                    <div style="background: #f8f9fa; padding: 10px 15px; border-bottom: 1px solid #dee2e6; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #333; font-size: 13px;">共 {len(suggestions)} 条术语建议</span>
                        <span style="color: #666; font-size: 11px;">↓ 向下滚动查看更多</span>
                    </div>
                    <div class="intel-scroll-container" style="max-height: 350px;">
                        <table style="width: 100%; border-collapse: collapse; color: #333;">
                            <thead style="position: sticky; top: 0; background: #f8f9fa; z-index: 10;">
                                <tr>
                                    <th style="padding: 12px 15px; text-align: left; border-bottom: 2px solid #dee2e6; font-size: 13px;">原始术语</th>
                                    <th style="padding: 12px 15px; text-align: left; border-bottom: 2px solid #dee2e6; font-size: 13px;">建议替换</th>
                                    <th style="padding: 12px 15px; text-align: center; border-bottom: 2px solid #dee2e6; font-size: 13px; width: 100px;">置信度</th>
                                    <th style="padding: 12px 15px; text-align: left; border-bottom: 2px solid #dee2e6; font-size: 13px;">推荐理由</th>
                                </tr>
                            </thead>
                            <tbody>
"""
            for idx, s in enumerate(suggestions):  # 展示所有术语
                orig = s.get("original", s.get("original_term", "N/A"))
                suggested = s.get("suggested", s.get("suggested_replacement", "N/A"))
                conf = s.get("confidence", 0)
                reason = s.get("reasoning", s.get("reason", "无"))
                
                # 置信度颜色
                if conf >= 0.8:
                    conf_color = "#28a745"
                    conf_bg = "#d4edda"
                elif conf >= 0.6:
                    conf_color = "#ffc107"
                    conf_bg = "#fff3cd"
                else:
                    conf_color = "#dc3545"
                    conf_bg = "#f8d7da"
                
                html += f"""
                                <tr style="border-bottom: 1px solid #dee2e6;">
                                    <td style="padding: 10px 15px;"><code style="background: #f8f9fa; padding: 2px 6px; border-radius: 4px;">{orig}</code></td>
                                    <td style="padding: 10px 15px;">{suggested}</td>
                                    <td style="padding: 10px 15px; text-align: center;">
                                        <span style="background: {conf_bg}; color: {conf_color}; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 12px;">{conf * 100:.0f}%</span>
                                    </td>
                                    <td style="padding: 10px 15px; color: #666; font-size: 13px;">{reason}</td>
                                </tr>
"""
            html += """
                            </tbody>
                        </table>
                    </div>
                </div>
"""
        else:
            html += """
                <div style="background: rgba(255,255,255,0.9); border-radius: 8px; padding: 20px; text-align: center; color: #666;">
                    <div style="font-size: 40px; margin-bottom: 10px;">📭</div>
                    <div>本次处理未发现新的术语建议</div>
                </div>
"""

        html += """
            </div>
"""

        # ========================================
        # 2. 歧义检测详情 - 带滚动和完整上下文
        # ========================================
        html += """
            <!-- 歧义检测详情 -->
            <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 20px; margin-top: 20px;">
                <h3 style="margin-top: 0; display: flex; align-items: center; gap: 8px;">
                    🔍 歧义检测详情
                </h3>
"""

        if detected_ambiguities:
            html += f"""
                <div style="background: rgba(255,255,255,0.15); padding: 10px 15px; border-radius: 8px 8px 0 0; display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: white; font-size: 13px;">共 {len(detected_ambiguities)} 处歧义</span>
                    <span style="color: rgba(255,255,255,0.7); font-size: 11px;">↓ 向下滚动查看更多</span>
                </div>
                <div class="intel-scroll-container" style="max-height: 400px; background: rgba(255,255,255,0.05); border-radius: 0 0 8px 8px; padding: 15px;">
                    <div style="display: grid; gap: 12px;">
"""
            for i, amb in enumerate(detected_ambiguities, 1):  # 展示所有歧义
                span = amb.get("span", amb.get("ambiguous_span", "N/A"))
                amb_type = amb.get("type", amb.get("ambiguity_type", "unknown"))
                context = amb.get("context", amb.get("surrounding_text", ""))
                resolution = amb.get("resolution", amb.get("suggested_resolution", ""))
                
                # 歧义类型标签颜色
                type_colors = {
                    "reference": ("#e3f2fd", "#1565c0", "指代歧义"),
                    "temporal": ("#fff8e1", "#f57c00", "时间序歧义"),
                    "quantifier": ("#f3e5f5", "#7b1fa2", "量词歧义"),
                    "semantic": ("#e8f5e9", "#2e7d32", "语义歧义"),
                    "structural": ("#fce4ec", "#c62828", "结构歧义"),
                }
                bg_color, text_color, type_label = type_colors.get(amb_type, ("#f5f5f5", "#666", amb_type))
                
                html += f"""
                        <div style="background: rgba(255,255,255,0.95); border-radius: 8px; padding: 15px; color: #333;">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap;">
                                <span style="background: #667eea; color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">{i}</span>
                                <code style="background: #fff3cd; padding: 4px 10px; border-radius: 4px; font-weight: bold; color: #856404;">"{span}"</code>
                                <span style="background: {bg_color}; color: {text_color}; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 500;">{type_label}</span>
                            </div>
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 6px; font-size: 13px; border-left: 3px solid #667eea; margin-bottom: 8px;">
                                <strong style="color: #667eea;">上下文:</strong>
                                <div style="color: #666; margin-top: 4px; white-space: pre-wrap; word-break: break-all;">{context}</div>
                            </div>
                            {f'<div style="background: #e8f5e9; padding: 10px; border-radius: 6px; font-size: 13px; border-left: 3px solid #28a745;"><strong style="color: #28a745;">建议消解:</strong> <span style="color: #666;">{resolution}</span></div>' if resolution else ''}
                        </div>
"""
            html += """
                    </div>
                </div>
"""
        else:
            html += """
                <div style="background: rgba(255,255,255,0.9); border-radius: 8px; padding: 20px; text-align: center; color: #666;">
                    <div style="font-size: 40px; margin-bottom: 10px;">✅</div>
                    <div>本次处理未检测到歧义</div>
                </div>
"""

        html += """
            </div>
"""

        # ========================================
        # 3. 决策追溯详情 - 带滚动和可展开详情
        # ========================================
        html += """
            <!-- 决策追溯详情 -->
            <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 20px; margin-top: 20px;">
                <h3 style="margin-top: 0; display: flex; align-items: center; gap: 8px;">
                    📋 决策追溯详情
                </h3>
"""

        if steps:
            html += f"""
                <div style="background: rgba(255,255,255,0.15); padding: 10px 15px; border-radius: 8px 8px 0 0; display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: white; font-size: 13px;">共 {len(steps)} 个处理步骤</span>
                    <span style="color: rgba(255,255,255,0.7); font-size: 11px;">↓ 向下滚动查看更多</span>
                </div>
                <div class="intel-scroll-container" style="max-height: 500px; padding: 20px 20px 20px 50px;">
                    <div style="position: relative;">
                        <!-- 时间线 -->
                        <div style="position: absolute; left: -30px; top: 0; bottom: 0; width: 2px; background: rgba(255,255,255,0.3);"></div>
"""
            
            total_duration = 0
            for i, step in enumerate(steps, 1):
                step_name = step.get("step_name", step.get("name", f"步骤 {i}"))
                step_confidence = step.get("confidence", 0)
                step_duration = step.get("duration_ms", 0)
                step_status = step.get("status", "success")
                step_reasoning = step.get("reasoning", "")
                step_evidences = step.get("evidences", [])
                step_changes = step.get("changes", {})
                input_full = step.get("input_summary", step.get("input", ""))
                output_full = step.get("output_summary", step.get("output", ""))
                
                total_duration += step_duration
                
                # 状态颜色
                status_colors = {
                    "success": ("#28a745", "✓ 成功"),
                    "partial": ("#ffc107", "⚠ 部分"),
                    "error": ("#dc3545", "✗ 失败"),
                }
                status_color, status_label = status_colors.get(step_status, ("#6c757d", step_status))
                
                # 置信度颜色
                if step_confidence >= 0.8:
                    conf_bg = "#28a745"
                elif step_confidence >= 0.6:
                    conf_bg = "#ffc107"
                else:
                    conf_bg = "#dc3545"
                
                html += f"""
                        <!-- 步骤 {i} -->
                        <div style="position: relative; margin-bottom: 25px;">
                            <!-- 节点 -->
                            <div style="position: absolute; left: -38px; top: 5px; width: 14px; height: 14px; background: {status_color}; border-radius: 50%; border: 2px solid white;"></div>
                            
                            <!-- 内容卡片 -->
                            <div style="background: rgba(255,255,255,0.95); border-radius: 8px; padding: 18px; color: #333;">
                                <!-- 标题行 -->
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                                    <div style="display: flex; align-items: center; gap: 10px;">
                                        <span style="background: #667eea; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 13px;">步骤 {i}</span>
                                        <span style="font-weight: bold; font-size: 15px;">{step_name}</span>
                                    </div>
                                    <div style="display: flex; align-items: center; gap: 10px;">
                                        <span style="background: {status_color}20; color: {status_color}; padding: 3px 10px; border-radius: 12px; font-size: 12px;">{status_label}</span>
                                        <span style="color: #666; font-size: 13px;">⏱ {step_duration}ms</span>
                                    </div>
                                </div>
                                
                                <!-- 置信度进度条 -->
                                <div style="margin-bottom: 12px;">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                        <span style="font-size: 12px; color: #666;">置信度</span>
                                        <span style="font-size: 12px; font-weight: bold; color: {conf_bg};">{step_confidence * 100:.1f}%</span>
                                    </div>
                                    <div style="height: 6px; background: #e9ecef; border-radius: 3px; overflow: hidden;">
                                        <div style="height: 100%; width: {step_confidence * 100}%; background: {conf_bg}; border-radius: 3px;"></div>
                                    </div>
                                </div>
"""
                
                # 推理过程
                if step_reasoning:
                    html += f"""
                                <div style="background: #f8f9fa; padding: 10px; border-radius: 6px; margin-bottom: 12px; font-size: 13px; border-left: 3px solid #667eea;">
                                    <strong style="color: #667eea;">推理:</strong> {step_reasoning}
                                </div>
"""
                
                # 输入输出 - 可展开
                html += f"""
                                <!-- 输入输出详情 -->
                                <div style="margin-bottom: 12px;">
                                    <button class="intel-expand-btn" onclick="toggleDetail('step-io-{i}')">
                                        📂 查看输入输出详情
                                    </button>
                                    <div id="step-io-{i}" class="intel-detail-panel">
                                        <div style="display: grid; gap: 10px; margin-top: 10px;">
                                            <div>
                                                <strong style="color: #1565c0; font-size: 12px;">📥 输入:</strong>
                                                <div class="intel-code-block" style="margin-top: 5px;">{input_full if input_full else '(无输入数据)'}</div>
                                            </div>
                                            <div>
                                                <strong style="color: #2e7d32; font-size: 12px;">📤 输出:</strong>
                                                <div class="intel-code-block" style="margin-top: 5px;">{output_full if output_full else '(无输出数据)'}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
"""
                
                # 证据链详情 - 完整展示
                if step_evidences:
                    html += f"""
                                <!-- 证据链详情 -->
                                <div style="margin-bottom: 12px;">
                                    <button class="intel-expand-btn" onclick="toggleDetail('step-ev-{i}')">
                                        🔗 查看证据链 ({len(step_evidences)} 条)
                                    </button>
                                    <div id="step-ev-{i}" class="intel-detail-panel">
                                        <div style="max-height: 200px; overflow-y: auto; margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 6px;">
"""
                    for ev_idx, ev in enumerate(step_evidences):
                        ev_type = ev.get("evidence_type", ev.get("type", ""))
                        ev_content = ev.get("content", "")
                        ev_conf = ev.get("confidence", 0)
                        ev_source = ev.get("source", "未知")
                        
                        # 证据类型图标
                        type_icons = {
                            "rule_match": "📏",
                            "llm_inference": "🤖",
                            "pattern_detected": "🔍",
                            "transformation": "🔄",
                            "validation_passed": "✅",
                            "validation_failed": "❌",
                        }
                        ev_icon = type_icons.get(ev_type, "📌")
                        
                        html += f"""
                                            <div style="padding: 8px; margin-bottom: 8px; background: white; border-radius: 6px; border-left: 3px solid #667eea;">
                                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                                    <span style="font-weight: bold; font-size: 12px;">{ev_icon} 证据 {ev_idx + 1}</span>
                                                    <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px;">{ev_conf * 100:.0f}%</span>
                                                </div>
                                                <div style="font-size: 12px; color: #666; margin-bottom: 4px;">
                                                    <strong>类型:</strong> {ev_type or '未知'}
                                                </div>
                                                <div style="font-size: 12px; color: #666; margin-bottom: 4px;">
                                                    <strong>内容:</strong> {ev_content}
                                                </div>
                                                <div style="font-size: 11px; color: #999;">
                                                    <strong>来源:</strong> {ev_source}
                                                </div>
                                            </div>
"""
                    html += """
                                        </div>
                                    </div>
                                </div>
"""
                
                # 变更记录 - 完整展示
                if step_changes:
                    html += f"""
                                <!-- 变更记录 -->
                                <div>
                                    <button class="intel-expand-btn" onclick="toggleDetail('step-ch-{i}')">
                                        📝 查看变更记录 ({len(step_changes)} 项)
                                    </button>
                                    <div id="step-ch-{i}" class="intel-detail-panel">
                                        <div style="max-height: 150px; overflow-y: auto; margin-top: 10px; padding: 10px; background: #fff3cd; border-radius: 6px;">
"""
                    for key, value in step_changes.items():
                        html += f"""
                                            <div style="padding: 6px 0; border-bottom: 1px dashed #ffc107;">
                                                <code style="background: white; padding: 2px 6px; border-radius: 4px;">{key}</code>
                                                <span style="color: #666;"> → </span>
                                                <strong>{str(value)}</strong>
                                            </div>
"""
                    html += """
                                        </div>
                                    </div>
                                </div>
"""
                
                html += """
                            </div>
                        </div>
"""
            
            html += f"""
                    </div>
                </div>
                
                <!-- 总耗时 -->
                <div style="background: rgba(255,255,255,0.2); border-radius: 8px; padding: 12px; margin-top: 15px; text-align: center;">
                    <span style="font-size: 14px;">⏱ 总处理耗时: <strong>{total_duration}ms</strong></span>
                </div>
"""
        else:
            html += """
                <div style="background: rgba(255,255,255,0.9); border-radius: 8px; padding: 20px; text-align: center; color: #666;">
                    <div style="font-size: 40px; margin-bottom: 10px;">📭</div>
                    <div>无决策追溯数据</div>
                </div>
"""

        html += """
            </div>
"""

        # ========================================
        # 4. Mermaid流程图
        # ========================================
        if mermaid_flow:
            html += f"""
            <!-- Mermaid 流程图 -->
            <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 20px; margin-top: 20px;">
                <h3 style="margin-top: 0; display: flex; align-items: center; gap: 8px;">
                    📊 决策流程图
                </h3>
                <div style="background: white; border-radius: 8px; padding: 15px; overflow-x: auto;">
                    <pre class="mermaid" style="margin: 0;">
{mermaid_flow}
                    </pre>
                </div>
            </div>
"""

        html += """
        </div>
"""

        return html

    def export_pipeline_html(self, history: PipelineHistory, output_file: Optional[str] = None) -> str:
        """
        导出完整流水线为HTML格式

        Args:
            history: 流水线历史记录
            output_file: 输出文件路径

        Returns:
            HTML内容
        """
        if output_file is None:
            output_file = os.path.join(
                self.storage_dir,
                f"pipeline_{history.pipeline_id}.html"
            )
        
        # 变量表格HTML
        variables_html = ""
        for var in history.prompt20_variables:
            variables_html += f"""
                <tr>
                    <td><code>{var.get('variable', 'N/A')}</code></td>
                    <td>"{var.get('original_text', 'N/A')}"</td>
                    <td><strong>{var.get('value', 'N/A')}</strong></td>
                    <td><span class="type-badge">{var.get('type', 'N/A')}</span></td>
                </tr>
"""
        
        # 类型统计HTML
        type_stats_html = ""
        if history.prompt20_type_stats:
            type_stats_html = '<div class="type-stats-box">'
            for dtype, count in history.prompt20_type_stats.items():
                type_stats_html += f'<div class="type-stat-item"><span class="type-badge">{dtype}</span>: <strong>{count}</strong></div>'
            type_stats_html += '</div>'
        else:
            type_stats_html = '<p style="color:#999; font-style:italic;">无类型统计数据</p>'
        
        # 提取日志HTML（验证过程）
        extraction_log_html = ""
        if history.prompt20_extraction_log:
            extraction_log_html = '<div class="extraction-log-box">'
            for i, log_msg in enumerate(history.prompt20_extraction_log, 1):
                # 添加不同图标来区分不同类型的日志
                icon = "•"
                if "LLM 识别到" in log_msg:
                    icon = "🤖"
                elif "❌" in log_msg or "幻觉检测" in log_msg:
                    icon = "❌"
                elif "✓ 验证通过" in log_msg or "精确匹配通过" in log_msg or "模糊匹配通过" in log_msg:
                    icon = "✓"
                elif "🔍" in log_msg or "过滤:" in log_msg:
                    icon = "🔍"
                elif "类型转换:" in log_msg:
                    icon = "🔄"
                elif "冲突解析完成" in log_msg or "后处理校验完成" in log_msg:
                    icon = "✅"
                
                extraction_log_html += f'<div class="log-item"><span class="log-icon">{icon}</span><span class="log-text">{log_msg}</span></div>'
            extraction_log_html += '</div>'
        else:
            extraction_log_html = '<p style="color:#999; font-style:italic;">无提取日志</p>'
        
        # 处理步骤HTML
        steps_html = ""
        for i, step in enumerate(history.prompt10_steps, 1):
            step_name = step.get('step_name', f'Step {i}')
            duration = step.get('duration_ms', 0)
            changes = step.get('changes', {})
            notes = step.get('notes', [])
            
            changes_html = ""
            if changes:
                for old, new in list(changes.items())[:3]:
                    new_str = f"'{new}'" if new else "(删除)"
                    changes_html += f'<div class="change-item"><span class="old">{old}</span> → <span class="new">{new_str}</span></div>'
                if len(changes) > 3:
                    changes_html += f'<div class="change-item">... 还有 {len(changes) - 3} 处变更</div>'
            else:
                changes_html = '<div class="change-item">无变更</div>'
            
            notes_html = "".join([f'<div class="note-item">• {note}</div>' for note in notes])
            
            steps_html += f"""
                <div class="step-card">
                    <div class="step-header">
                        <span class="step-number">{i}</span>
                        <span class="step-title">{step_name}</span>
                        <span class="step-duration">{duration}ms</span>
                    </div>
                    <div class="step-body">
                        <div class="step-section">
                            <h5>变更记录</h5>
                            {changes_html}
                        </div>
                        {f'<div class="step-section"><h5>备注</h5>{notes_html}</div>' if notes else ''}
                    </div>
                </div>
"""
        
        # 术语替换HTML
        terminology_html = ""
        for old, new in history.prompt10_terminology_changes.items():
            if new:
                terminology_html += f'<div class="term-item"><span class="old">{old}</span> → <span class="new">{new}</span></div>'
            else:
                terminology_html += f'<div class="term-item"><span class="old">{old}</span> → <span class="deleted">(删除)</span></div>'
        
        # DSL 代码 HTML
        dsl_code_html = ""
        if history.prompt30_dsl_code:
            escaped_dsl = history.prompt30_dsl_code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            dsl_code_html = f"""
                <div class="text-box dsl-code">
                    <pre>{escaped_dsl}</pre>
                </div>
            """
        else:
            dsl_code_html = '<p style="color:#999; font-style:italic;">未生成 DSL 代码</p>'
        
        # 验证结果 HTML
        validation_html = ""
        if history.prompt30_validation_result:
            valid = history.prompt30_validation_result.get('is_valid', False)
            errors = history.prompt30_validation_result.get('errors', [])
            warnings = history.prompt30_validation_result.get('warnings', [])
            defined_vars = history.prompt30_validation_result.get('defined_variables', {})
            function_calls = history.prompt30_validation_result.get('function_calls', [])
            
            validation_html = f"""
                <div class="validation-result">
                    <p><strong>验证状态:</strong> {'✅ 通过' if valid else '❌ 失败'}</p>
                    <p><strong>定义变量:</strong> {len(defined_vars)} 个</p>
                    <p><strong>函数调用:</strong> {len(function_calls)} 个</p>
                    <p><strong>错误数量:</strong> {len(errors)} 个</p>
                    <p><strong>警告数量:</strong> {len(warnings)} 个</p>
                </div>
            """
        else:
            validation_html = '<p style="color:#999; font-style:italic;">无验证结果</p>'
        
        # 模块列表HTML
        modules_html = ""
        if history.prompt40_modules:
            for i, module in enumerate(history.prompt40_modules, 1):
                module_name = module.get('name', 'N/A')
                inputs = module.get('inputs', [])
                outputs = module.get('outputs', [])
                is_async = module.get('is_async', False)
                
                inputs_str = ", ".join(inputs) if inputs else "无"
                outputs_str = ", ".join(outputs) if outputs else "无"
                mode_str = '<span class="badge-async">异步</span>' if is_async else '<span class="badge-sync">同步</span>'
                
                modules_html += f"""
                    <div class="module-card">
                        <div class="module-header">
                            <span class="module-number">{i}</span>
                            <span class="module-name">{module_name}</span>
                            {mode_str}
                        </div>
                        <div class="module-body">
                            <div><strong>输入:</strong> {inputs_str}</div>
                            <div><strong>输出:</strong> {outputs_str}</div>
                        </div>
                    </div>
"""
        else:
            modules_html = '<p style="color:#999; font-style:italic;">未生成工作流模块</p>'
        
        # 主代码 HTML
        main_code_html = ""
        if history.prompt40_main_code:
            escaped_code = history.prompt40_main_code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            main_code_html = f"""
                <div class="text-box main-code">
                    <pre>{escaped_code}</pre>
                </div>
            """
        else:
            main_code_html = '<p style="color:#999; font-style:italic;">未生成主代码</p>'

        # 第四步编译步骤详情 HTML
        step_details_html = self._generate_step_details_html(history)
        
        # 生成业务流程图（Approach 图）的 Mermaid 代码
        info("正在生成业务流程图...")
        approach_diagram_mermaid = self._generate_approach_diagram_mermaid(history)
        info(f"业务流程图生成完成 ({len(approach_diagram_mermaid)} 字符)")

        # 时间百分比计算
        time1_pct = history.prompt10_time_ms / history.total_time_ms * 100 if history.total_time_ms > 0 else 0
        time2_pct = history.prompt20_time_ms / history.total_time_ms * 100 if history.total_time_ms > 0 else 0
        time3_pct = history.prompt30_time_ms / history.total_time_ms * 100 if history.total_time_ms > 0 else 0
        time4_pct = history.prompt40_time_ms / history.total_time_ms * 100 if history.total_time_ms > 0 else 0
        
        # 定义变量和函数调用数量
        defined_vars_count = len(history.prompt30_validation_result.get('defined_variables', {})) if history.prompt30_validation_result else 0
        function_calls_count = len(history.prompt30_validation_result.get('function_calls', [])) if history.prompt30_validation_result else 0
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>流水线报告 - {history.pipeline_id}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .meta-bar {{
            display: flex;
            justify-content: space-around;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .meta-item {{
            text-align: center;
        }}
        .meta-item .label {{
            color: #666;
            font-size: 12px;
        }}
        .meta-item .value {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }}
        .stage {{
            margin: 30px 0;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
        }}
        .stage-header {{
            padding: 15px 20px;
            font-weight: bold;
            color: white;
        }}
        .stage-1 .stage-header {{ background: linear-gradient(90deg, #667eea, #764ba2); }}
        .stage-2 .stage-header {{ background: linear-gradient(90deg, #11998e, #38ef7d); }}
        .stage-3 .stage-header {{ background: linear-gradient(90deg, #ff7e5f, #feb47b); }}
        .stage-4 .stage-header {{ background: linear-gradient(90deg, #f093fb, #f5576c); }}
        .stage-content {{
            padding: 20px;
        }}
        .text-box {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
            white-space: pre-wrap;
            font-family: 'Consolas', monospace;
            line-height: 1.8;
            max-height: 400px;
            overflow-y: auto;
        }}
        .text-box.template {{ border-left-color: #11998e; }}
        .text-box.dsl-code {{ border-left-color: #ff7e5f; }}
        .text-box.main-code {{ border-left-color: #f093fb; max-height: 500px; }}
        .term-changes {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }}
        .term-item {{
            background: #fff3cd;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 14px;
        }}
        .old {{ color: #d32f2f; text-decoration: line-through; }}
        .new {{ color: #388e3c; font-weight: bold; }}
        .deleted {{ color: #999; font-style: italic; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f8f9fa;
            font-weight: bold;
        }}
        .type-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .type-badge {{ background: #e3f2fd; color: #1976d2; }}
        code {{
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card .number {{
            font-size: 32px;
            font-weight: bold;
        }}
        .stat-card .label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .step-cards {{
            display: grid;
            gap: 15px;
        }}
        .step-card {{
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .step-header {{
            background: #f8f9fa;
            padding: 12px 15px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .step-number {{
            background: #667eea;
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
        }}
        .step-title {{
            flex: 1;
            font-weight: bold;
        }}
        .step-duration {{
            color: #666;
            font-size: 14px;
        }}
        .step-body {{
            padding: 15px;
        }}
        .step-section {{
            margin-bottom: 15px;
        }}
        .step-section:last-child {{ margin-bottom: 0; }}
        .step-section h5 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
            font-weight: bold;
        }}
        .change-item {{
            padding: 5px 0;
            font-size: 14px;
        }}
        .note-item {{
            padding: 3px 0;
            color: #666;
            font-size: 14px;
        }}
        .module-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }}
        .module-card {{
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .module-header {{
            background: linear-gradient(90deg, #f093fb, #f5576c);
            color: white;
            padding: 12px 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .module-number {{
            background: rgba(255,255,255,0.3);
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 12px;
        }}
        .module-name {{
            flex: 1;
            font-weight: bold;
        }}
        .badge-async, .badge-sync {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
        }}
        .badge-async {{ background: #4CAF50; }}
        .badge-sync {{ background: #2196F3; }}
        .module-body {{
            padding: 15px;
            font-size: 14px;
        }}
        .module-body div {{
            margin: 5px 0;
        }}
        .time-breakdown {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }}
        .time-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .time-item:last-child {{ border-bottom: none; }}
        .time-bar {{
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            margin-top: 5px;
            overflow: hidden;
        }}
        .time-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }}
        .time-fill.stage-2 {{ background: linear-gradient(90deg, #11998e, #38ef7d); }}
        .time-fill.stage-3 {{ background: linear-gradient(90deg, #ff7e5f, #feb47b); }}
        .time-fill.stage-4 {{ background: linear-gradient(90deg, #f093fb, #f5576c); }}

        /* 调用关系图样式 - 优化显示空间 */
        .call-graph-container {{
            background: #f8f9fa;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            margin: 30px 0;
            overflow: hidden;
            position: relative;
            min-height: 600px;
        }}
        .call-graph-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .call-graph-controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .control-btn {{
            padding: 8px 16px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }}
        .control-btn:hover {{
            background: #0056b3;
        }}
        .control-btn.reset {{
            background: #6c757d;
        }}
        .control-btn.reset:hover {{
            background: #545b62;
        }}
        .zoom-level {{
            background: #e9ecef;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            min-width: 80px;
            text-align: center;
        }}
        .mermaid-wrapper {{
            width: 100%;
            height: 100%;
            overflow: auto;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: grab;
        }}
        .mermaid-wrapper:active {{
            cursor: grabbing;
        }}
        .mermaid {{
            display: block;
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.1s ease;
            transform-origin: center center;
            min-width: 1000px;
        }}
        .zoom-hint {{
            text-align: center;
            color: #6c757d;
            font-size: 12px;
            margin-top: 10px;
            font-style: italic;
        }}
        
        /* 类型统计样式 */
        .type-stats-box {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            padding: 15px;
            background: #f0f4ff;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #e0e7ff;
        }}
        .type-stat-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: white;
            border-radius: 6px;
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        /* 提取日志样式 */
        .log-section {{
            margin: 20px 0;
        }}
        .log-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 15px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 6px;
            font-size: 13px;
            border: 1px solid #e9ecef;
        }}
        .log-legend span {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .extraction-log-box {{
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            max-height: 500px;
            overflow-y: auto;
        }}
        .log-item {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 10px;
            margin: 8px 0;
            border-left: 3px solid #e9ecef;
            background: #fafbfc;
            border-radius: 4px;
            font-size: 14px;
            line-height: 1.6;
        }}
        .log-icon {{
            font-size: 18px;
            min-width: 24px;
            text-align: center;
            margin-top: -2px;
        }}
        .log-text {{
            flex: 1;
            word-break: break-word;
        }}
        /* 不同类型日志的特殊样式 */
        .log-item:nth-child(n) {{
            border-left-color: #dee2e6;
        }}
        /* 根据图标设置不同边框色（通过CSS选择器） */
        .log-item:has(.log-icon:contains("✓")) {{
            border-left-color: #28a745;
            background: #f0fff4;
        }}
        .log-item:has(.log-icon:contains("❌")) {{
            border-left-color: #dc3545;
            background: #fff5f5;
        }}
        .log-item:has(.log-icon:contains("🔍")) {{
            border-left-color: #fd7e14;
            background: #fff8f0;
        }}
        .log-item:has(.log-icon:contains("🔄")) {{
            border-left-color: #17a2b8;
            background: #e3f2fd;
        }}
    </style>
    <!-- 引入 Mermaid.js 用于渲染架构图 -->
    <!-- 使用 jsDelivr CDN (lib.baomitu.com/10.6.1 不存在，jsdelivr 可用) -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>📊 完整流水线处理报告</h1>
        <p class="subtitle">Prompt 1.0 预处理 → Prompt 2.0 结构化 → Prompt 3.0 DSL 编译 → Prompt 4.0 代码生成</p>
        
        <div class="meta-bar">
            <div class="meta-item">
                <div class="label">流水线 ID</div>
                <div class="value">{history.pipeline_id}</div>
            </div>
            <div class="meta-item">
                <div class="label">处理时间</div>
                <div class="value">{history.timestamp}</div>
            </div>
            <div class="meta-item">
                <div class="label">状态</div>
                <div class="value">{'✅ ' + history.overall_status if history.overall_status == 'success' else '⚠️ ' + history.overall_status}</div>
            </div>
            <div class="meta-item">
                <div class="label">总耗时</div>
                <div class="value">{history.total_time_ms}ms</div>
            </div>
        </div>
        
        <!-- 阶段 1 -->
        <div class="stage stage-1">
            <div class="stage-header">📝 阶段 1: Prompt 1.0 预处理 (耗时 {history.prompt10_time_ms}ms)</div>
            <div class="stage-content">
                <h4>原始输入</h4>
                <div class="text-box">{history.raw_input}</div>
                
                <h4>标准化输出</h4>
                <div class="text-box">{history.prompt10_processed}</div>
                
                <h4>处理步骤详情 ({len(history.prompt10_steps)} 个步骤)</h4>
                <div class="step-cards">
                    {steps_html if steps_html else '<p style="color:#999">无处理步骤记录</p>'}
                </div>
                
                <h4>术语替换 ({len(history.prompt10_terminology_changes)} 处)</h4>
                <div class="term-changes">{terminology_html if terminology_html else '<span style="color:#999">无术语替换</span>'}</div>
            </div>
        </div>
        
        <!-- 阶段 2 -->
        <div class="stage stage-2">
            <div class="stage-header">🔧 阶段 2: Prompt 2.0 结构化 (耗时 {history.prompt20_time_ms}ms)</div>
            <div class="stage-content">
                <h4>参数化模板</h4>
                <div class="text-box template">{history.prompt20_template}</div>
                
                <h4>变量注册表 ({history.prompt20_variable_count} 个变量)</h4>
                <table>
                    <thead>
                        <tr>
                            <th>变量名</th>
                            <th>原文片段</th>
                            <th>提取值</th>
                            <th>类型</th>
                        </tr>
                    </thead>
                    <tbody>
                        {variables_html}
                    </tbody>
                </table>
                
                <h4>类型统计</h4>
                {type_stats_html}
                
                <h4>验证过程提取日志 ({len(history.prompt20_extraction_log)} 条记录)</h4>
                <div class="log-section">
                    <div class="log-legend">
                        <span><span class="log-icon">🤖</span> LLM识别</span>
                        <span><span class="log-icon">✓</span> 验证通过</span>
                        <span><span class="log-icon">❌</span> 验证失败/幻觉</span>
                        <span><span class="log-icon">🔍</span> 规则过滤</span>
                        <span><span class="log-icon">🔄</span> 类型转换</span>
                        <span><span class="log-icon">✅</span> 步骤完成</span>
                    </div>
                    {extraction_log_html}
                </div>
            </div>
        </div>
        
        <!-- 阶段 3 -->
        <div class="stage stage-3">
            <div class="stage-header">⚙️ 阶段 3: Prompt 3.0 DSL 编译 (耗时 {history.prompt30_time_ms}ms)</div>
            <div class="stage-content">
                <h4>生成的 DSL 代码</h4>
                {dsl_code_html}
                
                <h4>验证结果</h4>
                {validation_html}
            </div>
        </div>
        
        <!-- 阶段 4 -->
        <div class="stage stage-4">
            <div class="stage-header">💻 阶段 4: Prompt 4.0 代码生成 (耗时 {history.prompt40_time_ms}ms)</div>
            <div class="stage-content">
                <h4>业务流程图（Approach 图）</h4>
                <div class="call-graph-container">
                    <div class="call-graph-header">
                        <span><strong>交互控制</strong></span>
                        <div class="call-graph-controls">
                            <button class="control-btn" onclick="zoomIn()">🔍 放大 (+)</button>
                            <button class="control-btn" onclick="zoomOut()">🔍 缩小 (-)</button>
                            <button class="control-btn reset" onclick="resetZoom()">↺ 重置</button>
                            <span class="zoom-level" id="zoomLevel">100%</span>
                        </div>
                    </div>
                    <div class="mermaid-wrapper" id="mermaidWrapper">
                        <div class="mermaid" id="mermaidDiagram">
{approach_diagram_mermaid}
                        </div>
                    </div>
                    <div class="zoom-hint">
                        💡 提示：此图展示业务逻辑和处理流程，使用鼠标滚轮缩放，或拖拽移动图片
                    </div>
                </div>

                <h4>编译步骤详情</h4>
                {step_details_html}

                <h4>工作流模块 ({history.prompt40_module_count} 个)</h4>
                <div class="module-cards">
                    {modules_html}
                </div>

                <h4>主工作流代码</h4>
                {main_code_html}
            </div>
        </div>
        
        <!-- 统计 -->
        <div class="stats">
            <div class="stat-card">
                <div class="number">{len(history.raw_input)}</div>
                <div class="label">原始字符数</div>
            </div>
            <div class="stat-card">
                <div class="number">{len(history.prompt10_processed)}</div>
                <div class="label">标准化字符数</div>
            </div>
            <div class="stat-card">
                <div class="number">{len(history.prompt10_terminology_changes)}</div>
                <div class="label">术语替换</div>
            </div>
            <div class="stat-card">
                <div class="number">{history.prompt20_variable_count}</div>
                <div class="label">识别变量</div>
            </div>
            <div class="stat-card">
                <div class="number">{history.prompt40_module_count}</div>
                <div class="label">工作流模块</div>
            </div>
            <div class="stat-card">
                <div class="number">{defined_vars_count}</div>
                <div class="label">定义变量</div>
            </div>
            <div class="stat-card">
                <div class="number">{function_calls_count}</div>
                <div class="label">函数调用</div>
            </div>
            <div class="stat-card">
                <div class="number">{history.total_time_ms}</div>
                <div class="label">总耗时(ms)</div>
            </div>
        </div>
        
        <!-- 时间分解 -->
        <div class="time-breakdown">
            <h3 style="margin-top:0;">⏱️ 耗时分解</h3>
            <div class="time-item">
                <span>阶段 1: 预处理</span>
                <span>{history.prompt10_time_ms}ms ({time1_pct:.1f}%)</span>
            </div>
            <div class="time-bar">
                <div class="time-fill" style="width: {time1_pct}%;"></div>
            </div>
            <div class="time-item">
                <span>阶段 2: 结构化</span>
                <span>{history.prompt20_time_ms}ms ({time2_pct:.1f}%)</span>
            </div>
            <div class="time-bar">
                <div class="time-fill stage-2" style="width: {time2_pct}%;"></div>
            </div>
            <div class="time-item">
                <span>阶段 3: DSL 编译</span>
                <span>{history.prompt30_time_ms}ms ({time3_pct:.1f}%)</span>
            </div>
            <div class="time-bar">
                <div class="time-fill stage-3" style="width: {time3_pct}%;"></div>
            </div>
            <div class="time-item">
                <span>阶段 4: 代码生成</span>
                <span>{history.prompt40_time_ms}ms ({time4_pct:.1f}%)</span>
            </div>
            <div class="time-bar">
                <div class="time-fill stage-4" style="width: {time4_pct}%;"></div>
            </div>
        </div>
    </div>

    <!-- 初始化 Mermaid.js 和缩放功能 -->
    <script>
        // 检查 mermaid 是否加载成功
        if (typeof mermaid === 'undefined') {{
            console.error('Mermaid.js 加载失败，请检查网络连接');
            document.getElementById('mermaidWrapper').innerHTML =
                '<div style="color: red; padding: 20px; text-align: center;">' +
                '❌ Mermaid.js 加载失败<br>' +
                '请检查网络连接或刷新页面' +
                '</div>';
        }} else {{
            // 初始化 Mermaid
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'default',
                securityLevel: 'loose',
                flowchart: {{
                    useMaxWidth: false,  // 禁用最大宽度限制
                    htmlLabels: true
                }},
                // 添加更多配置选项
                logLevel: 'error',  // 只显示错误日志
                suppressErrorRendering: false,  // 不隐藏错误
            }});
        }}

        // 监听 Mermaid 渲染错误
        window.addEventListener('error', function(e) {{
            if (e.message && e.message.includes('mermaid')) {{
                console.error('Mermaid 渲染错误:', e);
            }}
        }});

        // 缩放和平移功能
        let currentZoom = 1;
        let isDragging = false;
        let startX, startY, scrollLeft, scrollTop;
        const mermaidDiagram = document.getElementById('mermaidDiagram');
        const mermaidWrapper = document.getElementById('mermaidWrapper');
        const zoomLevelDisplay = document.getElementById('zoomLevel');

        function updateZoom() {{
            // 限制缩放范围: 0.3x - 3.0x
            currentZoom = Math.max(0.3, Math.min(3.0, currentZoom));
            mermaidDiagram.style.transform = `scale(${{currentZoom}})`;
            zoomLevelDisplay.textContent = `${{Math.round(currentZoom * 100)}}%`;
        }}

        function zoomIn() {{
            currentZoom += 0.1;
            updateZoom();
        }}

        function zoomOut() {{
            currentZoom -= 0.1;
            updateZoom();
        }}

        function resetZoom() {{
            currentZoom = 1;
            updateZoom();
            mermaidWrapper.scrollTo({{
                left: 0,
                top: 0,
                behavior: 'smooth'
            }});
        }}

        // 鼠标滚轮缩放
        mermaidWrapper.addEventListener('wheel', function(e) {{
            e.preventDefault();
            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            currentZoom += delta;
            updateZoom();
        }});

        // 鼠标拖拽平移
        mermaidWrapper.addEventListener('mousedown', function(e) {{
            isDragging = true;
            startX = e.pageX - mermaidWrapper.offsetLeft;
            startY = e.pageY - mermaidWrapper.offsetTop;
            scrollLeft = mermaidWrapper.scrollLeft;
            scrollTop = mermaidWrapper.scrollTop;
        }});

        mermaidWrapper.addEventListener('mouseleave', function() {{
            isDragging = false;
        }});

        mermaidWrapper.addEventListener('mouseup', function() {{
            isDragging = false;
        }});

        mermaidWrapper.addEventListener('mousemove', function(e) {{
            if (!isDragging) return;
            e.preventDefault();
            const x = e.pageX - mermaidWrapper.offsetLeft;
            const y = e.pageY - mermaidWrapper.offsetTop;
            const walkX = (x - startX) * 1.5;  // 平滑系数
            const walkY = (y - startY) * 1.5;
            mermaidWrapper.scrollLeft = scrollLeft - walkX;
            mermaidWrapper.scrollTop = scrollTop - walkY;
        }});

        // 双击重置缩放
        mermaidWrapper.addEventListener('dblclick', function() {{
            resetZoom();
        }});
    </script>
</body>
</html>
"""

        # ===== 生成智能优化与可解释性报告 =====
        intelligence_section = ""
        if self.explainer and (history.term_learning or history.ambiguity_resolution or history.decision_trace):
            try:
                intelligence_section = self._generate_intelligence_section(history)
            except Exception as e:
                warning(f"生成智能优化报告失败: {e}")

        # 插入智能优化报告到 </body> 之前
        if intelligence_section:
            html = html.replace("</body>", f"{intelligence_section}\n</body>")

        # 保存HTML文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            info(f"流水线HTML报告已保存: {output_file}")
        except Exception as e:
            error(f"保存流水线HTML报告失败: {e}")
        
        return html
    
    def _generate_call_graph_mermaid(self, history: PipelineHistory) -> str:
        """
        生成工作流调用关系图的 Mermaid 语法
        
        Args:
            history: 流水线历史记录
            
        Returns:
            Mermaid graph 语法
        """
        mermaid_lines = ["graph TD", "    %% 工作流调用关系图", ""]
        
        # 添加输入节点
        mermaid_lines.append("    Input([用户输入<br/>input_params]):::input")
        mermaid_lines.append("")
        
        # 添加主工作流节点
        mermaid_lines.append("    Main[main_workflow<br/>主控流程]:::main")
        mermaid_lines.append("")
        
        # 从主工作流到输入节点的连接
        mermaid_lines.append("    Input --> Main")
        mermaid_lines.append("")
        
        # 添加模块节点
        if history.prompt40_modules:
            modules = history.prompt40_modules
            step4_gen = history.prompt40_step4_generation
            
            # 获取模块聚类信息
            clusters = step4_gen.get('clusters', []) if step4_gen else []
            
            # 按照聚类分组显示模块
            if clusters:
                cluster_dict = {}
                # 为每个聚类分配模块
                for cluster_info in clusters:
                    cluster_id = cluster_info.get('cluster_id', 'unknown')
                    block_ids = cluster_info.get('blocks', [])
                    cluster_dict[cluster_id] = block_ids
                
                # 创建节点
                for i, module in enumerate(modules):
                    module_name = module.get('name', f'module_{i}')
                    is_async = module.get('is_async', False)
                    inputs = module.get('inputs', [])
                    outputs = module.get('outputs', [])
                    
                    # 节点样式
                    node_style = ":::async" if is_async else ":::sync"
                    node_id = f"Step{i}"
                    
                    # 节点标签
                    input_str = ", ".join(inputs[:3]) + ("..." if len(inputs) > 3 else "")
                    output_str = ", ".join(outputs[:3]) + ("..." if len(outputs) > 3 else "")
                    
                    label = f"{module_name}<br/><sub>({input_str}) → ({output_str})</sub>"
                    
                    mermaid_lines.append(f"    {node_id}[{label}]{node_style}")
                
                mermaid_lines.append("")
                
                # 添加边：主工作流调用所有模块
                for i, module in enumerate(modules):
                    node_id = f"Step{i}"
                    mermaid_lines.append(f"    Main --> {node_id}")
                
                mermaid_lines.append("")
            else:
                # 如果没有聚类信息，简单列出所有模块
                for i, module in enumerate(modules):
                    module_name = module.get('name', f'module_{i}')
                    is_async = module.get('is_async', False)
                    
                    node_style = ":::async" if is_async else ":::sync"
                    node_id = f"Step{i}"
                    
                    mermaid_lines.append(f"    {node_id}[{module_name}]{node_style}")
                    mermaid_lines.append(f"    Main --> {node_id}")
                
                mermaid_lines.append("")
        
        # 添加输出节点
        mermaid_lines.append("    Output([返回结果<br/>final_output]):::output")
        mermaid_lines.append("")
        
        # 从最后一个模块到输出节点的连接
        if history.prompt40_modules:
            last_step_id = f"Step{len(history.prompt40_modules) - 1}"
            mermaid_lines.append(f"    {last_step_id} --> Output")
        else:
            mermaid_lines.append("    Main --> Output")
        
        mermaid_lines.append("")
        
        # 添加图例
        mermaid_lines.append("    %% 样式定义")
        mermaid_lines.append("    classDef input fill:#d4edda,stroke:#28a745,stroke-width:2px;")
        mermaid_lines.append("    classDef main fill:#e1f5ff,stroke:#007bff,stroke-width:3px;")
        mermaid_lines.append("    classDef sync fill:#fff3cd,stroke:#ffc107,stroke-width:2px;")
        mermaid_lines.append("    classDef async fill:#f8d7da,stroke:#dc3545,stroke-width:2px;")
        mermaid_lines.append("    classDef output fill:#d4edda,stroke:#28a745,stroke-width:2px;")
        
        return "\n".join(mermaid_lines)
    
    def _generate_approach_diagram_mermaid(self, history: PipelineHistory) -> str:
        """
        生成业务流程图（Approach 图）的 Mermaid 语法
        
        使用 LLM 分析代码和需求，生成展示业务逻辑的流程图
        
        Args:
            history: 流水线历史记录
            
        Returns:
            Mermaid graph TD 语法
        """
        # 导入 LLM 客户端
        from llm_client import create_llm_client
        import json
        
        llm_client = create_llm_client()
        
        # 准备上下文信息
        context = {
            'business_requirement': history.prompt10_processed or "无需求描述",
            'module_count': len(history.prompt40_modules) if history.prompt40_modules else 0,
            'modules': [],
        }
        
        # 提取模块信息
        if history.prompt40_modules:
            for i, module in enumerate(history.prompt40_modules):
                context['modules'].append({
                    'index': i,
                    'name': module.get('name', f'module_{i}'),
                    'description': module.get('description', '无描述'),
                    'inputs': module.get('inputs', []),
                    'outputs': module.get('outputs', []),
                    'is_async': module.get('is_async', False),
                })
        
        # 获取 DSL 代码
        dsl_code = history.prompt30_dsl_code or ""
        
        # 构建 LLM Prompt
        prompt = f"""你是一个业务架构分析师。请基于以下信息，生成一个展示业务流程的 Mermaid 流程图。

## 业务需求
{context['business_requirement']}

## 模块信息（{context['module_count']} 个）
{json.dumps(context['modules'], indent=2, ensure_ascii=False)}

## DSL 代码
{dsl_code[:2000] if len(dsl_code) > 2000 else dsl_code}

## 要求

1. **重点展示业务逻辑，而非函数调用**
   - 体现数据处理的主要步骤
   - 展示决策点和分支逻辑
   - 使用业务术语，避免技术细节

2. **节点命名规范**
   - 使用业务术语（如：判断场景类型、生成响应内容）
   - 避免使用函数名（如：step_1_compute_xxx）
   - 节点名称简洁明了（不超过 15 个字）

3. **类名规范（重要）**
   - 结束节点类名使用 `endNode`，不要使用 `end`（`end` 是 Mermaid 保留关键字）
   - 其他类名使用有意义的名称（start、decision、process 等）
   - 示例：classDef endNode fill:#d4edda,stroke:#28a745,stroke-width:2px;

4. **流程图结构**
   - graph TD: 从上到下的流程图
   - 使用菱形（{{条件}}）表示决策点
   - 使用矩形（[步骤]）表示处理步骤
   - 使用圆角（（开始/结束））表示起点和终点

5. **样式要求**
   - 决策节点使用黄色系
   - 处理节点使用蓝色系
   - 起始/结束节点使用绿色系
   - 保持图表清晰易读

6. **输出格式**
   - 只输出 Mermaid graph TD 代码
   - 不包含任何解释文字
   - 不使用 ```mermaid ``` 标记

## 输出示例

graph TD
    Start([用户输入]) --> Check{{是否需要特殊处理?}}
    Check -->|是| Identify[识别场景类型]
    Check -->|否| Generate[直接生成]
    Identify --> Create[创建响应内容]
    Generate --> Format[格式化输出]
    Create --> Format
    Format --> End([返回结果])

    classDef start fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef endNode fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px;
    classDef process fill:#e1f5ff,stroke:#007bff,stroke-width:2px;

    class Start start;
    class End endNode;
    class Check decision;
    class Identify process;
    class Create process;
    class Generate process;
    class Format process;

请根据上述信息生成业务流程图："""
        
        try:
            # 调用 LLM 生成 Mermaid 代码
            system_prompt = """你是一个业务架构分析师，擅长将代码和技术实现转换为清晰的业务流程图。"""
            response = llm_client.call(
                system_prompt=system_prompt,
                user_content=prompt,
                temperature=0.3  # 使用较低的温度以获得更稳定的输出
            )
            mermaid_code = response.content.strip()

            # 调试：打印原始响应
            info(f"[Approach图LLM原始响应] 长度: {len(mermaid_code)} 字符")
            info(f"[Approach图LLM原始响应] 内容:\n{mermaid_code}")

            # 清理输出
            mermaid_code = mermaid_code.strip()

            # 更安全的 markdown 标记移除
            # 使用正则表达式移除 markdown 代码块标记
            import re
            # 移除开头的 ```mermaid 或 ```（更彻底）
            mermaid_code = re.sub(r'^[\s\S]*?```(?:mermaid)?\s*\n?', '', mermaid_code)
            # 移除结尾的 ```
            mermaid_code = re.sub(r'\n?```[\s\S]*?$', '', mermaid_code)

            # 确保第一行就是 graph TD，移除所有前面的无效内容
            lines = mermaid_code.split('\n')
            # 找到包含 "graph TD" 的行
            graph_idx = -1
            for i, line in enumerate(lines):
                if 'graph TD' in line:
                    graph_idx = i
                    break

            if graph_idx >= 0:
                # 从包含 "graph TD" 的行开始，不要 strip，保持原有的换行
                mermaid_code = '\n'.join(lines[graph_idx:])

            # 最后再 strip 一次，确保没有前后空白
            mermaid_code = mermaid_code.strip()

            # 清理 Mermaid 不支持的特殊字符
            # 将特殊符号替换为 ASCII 等价符号
            mermaid_code = mermaid_code.replace('≤', '<=')  # 小于等于
            mermaid_code = mermaid_code.replace('≥', '>=')  # 大于等于
            mermaid_code = mermaid_code.replace('≠', '!=')  # 不等于
            mermaid_code = mermaid_code.replace('→', '->')   # 箭头符号

            # 修复 Mermaid 保留关键字冲突
            # 'end' 是 Mermaid 的保留关键字，需要替换为 endNode
            # 替换 classDef end 为 classDef endNode
            mermaid_code = re.sub(r'classDef\s+end\b', 'classDef endNode', mermaid_code)
            # 替换 class ... end; 为 class ... endNode;
            mermaid_code = re.sub(r'class\s+(\w+)\s+end\b', r'class \1 endNode', mermaid_code)

            mermaid_code = mermaid_code.strip()

            # 清理不兼容的 Mermaid 10.x 语法
            # 移除 :::className 语法，替换为正确的 class 定义
            import re

            # 基本语法验证：检查括号是否匹配
            def validate_parentheses(code: str) -> tuple:
                """验证括号是否匹配"""
                stack = []
                brackets = {'(': ')', '[': ']', '{': '}'}
                position = 0

                for char in code:
                    position += 1
                    if char in brackets:
                        stack.append(char)
                    elif char in brackets.values():
                        if not stack:
                            return False, f"Unexpected closing bracket '{char}' at position {position}"
                        if brackets[stack[-1]] != char:
                            return False, f"Mismatched bracket at position {position}"
                        stack.pop()

                if stack:
                    return False, f"Unclosed brackets: {stack}"
                return True, ""

            # 验证基本语法（放宽要求）
            is_valid, error_msg = validate_parentheses(mermaid_code)
            if not is_valid:
                error(f"LLM 返回的 Mermaid 代码语法错误: {error_msg}")
                error(f"原始代码:\n{mermaid_code[:300]}...")
                # 不再直接返回默认图，尝试继续使用（即使有语法错误）
                warning(f"尝试使用可能包含错误的 Mermaid 代码")

            # 直接验证是否包含 graph TD，不需要复杂的清理逻辑
            # LLM 返回的 Mermaid 代码已经足够好了
            if 'graph TD' not in mermaid_code:
                # 如果 LLM 返回的不是 Mermaid 代码，使用默认模板
                error(f"[Approach图] 未找到 'graph TD'，使用默认图")
                return self._get_default_approach_diagram(context)

            info(f"[Approach图] 验证通过，使用 LLM 生成的代码，长度: {len(mermaid_code)} 字符")
            return mermaid_code
            
        except Exception as e:
            error(f"生成 Approach 图失败: {e}")
            # 返回默认的流程图
            return self._get_default_approach_diagram(context)
    
    def _get_default_approach_diagram(self, context: dict) -> str:
        """
        生成默认的业务流程图（当 LLM 调用失败时使用）

        Args:
            context: 上下文信息

        Returns:
            Mermaid graph TD 语法
        """
        mermaid_lines = ["graph TD", "    %% 业务流程图（默认生成）", ""]

        # 添加起始节点
        mermaid_lines.append("    Start([用户输入])")
        mermaid_lines.append("")

        # 根据模块数量生成处理步骤
        if context['module_count'] > 0:
            # 添加判断节点
            mermaid_lines.append("    Process1[处理输入数据]")
            mermaid_lines.append("    Start --> Process1")
            mermaid_lines.append("")

            # 添加中间处理步骤
            for i in range(1, context['module_count']):
                mermaid_lines.append(f"    Process{i+1}[处理步骤 {i+1}]")
                mermaid_lines.append(f"    Process{i} --> Process{i+1}")

            mermaid_lines.append("")

        # 添加结束节点
        mermaid_lines.append("    End([返回结果])")

        if context['module_count'] > 0:
            mermaid_lines.append(f"    Process{context['module_count']} --> End")
        else:
            mermaid_lines.append("    Start --> End")

        mermaid_lines.append("")

        # 添加样式定义（使用 style 语法，更稳定）
        mermaid_lines.append("    %% 样式定义")
        mermaid_lines.append("    style Start fill:#d4edda,stroke:#28a745,stroke-width:2px")
        mermaid_lines.append("    style End fill:#d4edda,stroke:#28a745,stroke-width:2px")
        for i in range(1, context['module_count'] + 1):
            mermaid_lines.append(f"    style Process{i} fill:#e1f5ff,stroke:#007bff,stroke-width:2px")

        return "\n".join(mermaid_lines)
