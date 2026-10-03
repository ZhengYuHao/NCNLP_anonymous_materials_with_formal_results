"""
S.E.D.E Framework - Step 2: 实体抽取与变量定义 (Entity Extraction & Variable Definition)
充当编译器前端的词法分析角色,将自然语言中的常量与变量分离
版本: 2.0
"""

import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
from logger import info, warning, error, debug, setup_logger
from llm_client import UnifiedLLMClient, create_llm_client
from data_models import (
    DataType, VariableMeta, Prompt10Result, Prompt20Result, Prompt25Result,
    Operation, OutputField, ConditionBranch,
    create_prompt20_result, generate_id, get_timestamp
)
from history_manager import HistoryManager, Prompt20History
from utils.hybrid_rule_learner import HybridRuleLearner, RuleType, create_hybrid_learner

# ========== 配置日志系统 ==========
logger = setup_logger("prompt3.0.step2")


# ========== 兼容性数据结构（保留旧接口） ==========
@dataclass
class VariableConstraints:
    """变量约束条件"""
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    allowed_values: Optional[List[str]] = None
    pattern: Optional[str] = None


@dataclass
class PromptStructure:
    """Prompt 2.0 结构化输出（兼容旧接口）"""
    template_text: str  # 带 {{variable}} 占位符的模板
    variable_registry: List[Dict]  # 变量注册表
    original_text: str  # 原始文本
    extraction_log: List[str]  # 提取日志


# ========== LLM 实体提取器 ==========
class LLMEntityExtractor:
    """LLM 实体提取接口 (语义扫描层) - 使用统一LLM客户端"""
    
    def __init__(
        self,
        llm_client: Optional[UnifiedLLMClient] = None,
        use_mock: bool = False,
        use_hybrid: bool = True
    ):
        """
        初始化实体提取器
        
        Args:
            llm_client: 统一LLM客户端实例
            use_mock: 是否使用模拟模式
            use_hybrid: 是否使用混合规则学习器
        """
        self.llm = llm_client or create_llm_client(use_mock=use_mock)
        self.use_mock = use_mock
        self.use_hybrid = use_hybrid
        
        if self.use_hybrid:
            self.hybrid_learner = create_hybrid_learner(use_mock=use_mock)
        else:
            self.hybrid_learner = None

    def extract(self, text: str, enable_optimization: bool = True) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        调用 LLM 进行实体提取（支持优化 + 混合规则学习）

        Args:
            text: 待抽取文本
            enable_optimization: 是否启用正则预处理优化

        Returns:
            (实体列表, 优化统计信息)
        """
        stats = {
            "regex_count": 0,
            "llm_count": 0,
            "merged_count": 0,
            "llm_called": False,
            "optimization_enabled": enable_optimization,
            "hybrid_rule_used": False,
            "rules_learned": 0
        }

        mustache_entities = self._extract_template_variables(text)
        dsl_entities = self._extract_dsl_variables(text)
        template_entities = mustache_entities + dsl_entities
        if template_entities:
            info(f"[模板变量] 提取到 {len(template_entities)} 个: {[e['name'] for e in template_entities]}")

        if self.use_hybrid and self.hybrid_learner:
            entities, rule_used = self.hybrid_learner.process_entity_extraction(text)
            if rule_used:
                stats["hybrid_rule_used"] = True
                stats["llm_called"] = True
                combined_entities = template_entities + entities if template_entities else entities
                stats["merged_count"] = len(combined_entities)
                return combined_entities, stats

        if self.use_mock:
            entities = self._mock_extract(text)
            stats["llm_count"] = len(entities)
            stats["merged_count"] = len(entities)
            combined_entities = template_entities + entities if template_entities else entities
            stats["merged_count"] = len(combined_entities)
            return combined_entities, stats

        entities, llm_stats = self.llm.extract_entities(text, enable_optimization=enable_optimization)

        stats["llm_count"] = llm_stats.get("llm_count", len(entities))
        stats["regex_count"] = llm_stats.get("regex_count", 0)
        stats["merged_count"] = llm_stats.get("merged_count", len(entities))
        stats["llm_called"] = True

        if self.use_hybrid and self.hybrid_learner and entities:
            new_rules = self.hybrid_learner.llm_handler.extract_entity_rule(text, entities)
            if new_rules:
                for rule in new_rules:
                    self.hybrid_learner.rule_library.add_rule(rule)
                stats["rules_learned"] = len(new_rules)
                info(f"[HYBRID] 从LLM提取结果中学习了 {len(new_rules)} 条实体规则")

        combined_entities = template_entities + entities if template_entities else entities
        stats["merged_count"] = len(combined_entities)

        return combined_entities, stats

    def _extract_template_variables(self, text: str) -> List[Dict[str, Any]]:
        """
        提取Mustache模板变量 {{variable}} 或 {{{variable}}}

        Returns:
            List[Dict]: 模板变量列表
        """
        entities = []
        mustache_pattern = r'(\{\{[^}]+\}\}|\{\{\{[^}]+\}\}\})'

        for match in re.finditer(mustache_pattern, text):
            original_text = match.group(1)
            name = original_text.strip('{}')

            entities.append({
                "name": name,
                "original_text": original_text,
                "start_index": match.start(),
                "end_index": match.end(),
                "type": "Mustache",
                "value": name,
                "category": "template"
            })

        return entities

    def _extract_dsl_variables(self, text: str) -> List[Dict[str, Any]]:
        """
        提取DSL简单变量 {identifier} 格式（不包括函数调用如 {output(chart)}）

        Returns:
            List[Dict]: DSL变量列表
        """
        entities = []
        dsl_simple_pattern = r'(\{[A-Za-z_][A-Za-z0-9_]*\})'
        dsl_func_pattern = r'\{[A-Za-z_][A-Za-z0-9_]*\([^)]*\)\}'

        for match in re.finditer(dsl_simple_pattern, text):
            original_text = match.group(1)
            if re.match(dsl_func_pattern, original_text):
                continue
            name = original_text.strip('{}')

            entities.append({
                "name": name,
                "original_text": original_text,
                "start_index": match.start(),
                "end_index": match.end(),
                "type": "DSL",
                "value": name,
                "category": "template"
            })

        return entities
    
    def _mock_extract(self, text: str) -> List[Dict]:
        """模拟实体抽取（用于测试）"""
        mock_entities = []
        
        # 识别数字+单位模式 (如 "3年", "2周")
        for match in re.finditer(r'(\d+)(年|周|月|天|小时|分钟)', text):
            mock_entities.append({
                "name": f"duration_{match.group(2)}",
                "original_text": match.group(0),
                "start_index": match.start(),
                "end_index": match.end(),
                "type": "Integer",
                "value": int(match.group(1))
            })
        
        # 识别专业术语
        tech_terms = ["Java程序员", "Python", "数据分析", "机器学习", "前端开发"]
        for term in tech_terms:
            if term in text:
                idx = text.find(term)
                mock_entities.append({
                    "name": f"tech_term_{len(mock_entities)}",
                    "original_text": term,
                    "start_index": idx,
                    "end_index": idx + len(term),
                    "type": "String",
                    "value": term
                })
        
        return mock_entities


# ========== 幻觉防火墙 ==========
class HallucinationFirewall:
    """防止 LLM 虚构原文中不存在的内容"""
    
    @staticmethod
    def validate_existence(entity: Dict, original_text: str) -> Tuple[bool, str]:
        """
        存在性校验: 确保 LLM 提取的片段真实存在于原文
        """
        original_snippet = entity.get('original_text', '')
        
        # 精确匹配
        if original_snippet in original_text:
            return True, "精确匹配通过"
        
        # 模糊匹配 (处理空格、标点差异)
        normalized_snippet = re.sub(r'\s+', '', original_snippet)
        normalized_text = re.sub(r'\s+', '', original_text)
        
        if normalized_snippet in normalized_text:
            return True, "模糊匹配通过"
        
        return False, f"幻觉检测: '{original_snippet}' 不存在于原文"
    
    @staticmethod
    def validate_index(entity: Dict, original_text: str) -> bool:
        """验证索引位置的准确性"""
        start = entity.get('start_index', -1)
        end = entity.get('end_index', -1)
        original_snippet = entity.get('original_text', '')
        
        if start == -1 or end == -1:
            return False
        
        if start < 0 or end > len(original_text) or start >= end:
            return False
        
        extracted = original_text[start:end]
        return extracted == original_snippet


# ========== 实体后处理校验器 ==========
class EntityPostValidator:
    """后处理实体校验器 - 过滤掉固定描述而非变量的实体"""
    
    # 固定描述关键词（包含这些词且没有数字的可能是固定需求）
    FIXED_DESCRIPTION_KEYWORDS = ["需要", "要", "支持", "确保", "实现", "提供", "用", "采用"]
    
    # 常见固定需求模式（整体不提取为变量）
    FIXED_DEMAND_PATTERNS = [
        r"需要支持.*对话",  # "需要支持多轮对话"
        r"支持.*双语",  # "支持中英文双语"
        r"用.*做底座",  # "用大模型做底座"
        r"响应时间控制在.*以内",  # 整体为配置描述
        r"上下文窗口存.*轮",  # 有数字但要单独提取
        r"用LangChain.*Milvus.*FastAPI",  # 技术栈整体提取
    ]
    
    # 变量关键词（表明是可配置参数）
    VARIABLE_KEYWORDS = ["个", "人", "万", "年", "周", "月", "天", "轮", "秒", "小时", "分钟"]
    
    @classmethod
    def filter_entities(cls, entities: List[Dict], original_text: str) -> Tuple[List[Dict], List[str]]:
        """
        过滤掉固定描述而非变量的实体
        
        Returns:
            (过滤后的实体列表, 过滤日志列表)
        """
        filtered = []
        filter_logs = []
        
        for entity in entities:
            orig_text = entity.get('original_text', '')
            entity_name = entity.get('name', '')
            entity_type = entity.get('type', None)
            filter_reason = cls._should_filter(orig_text, entity_name, entity_type)
            
            if filter_reason:
                filter_logs.append(f"过滤: '{orig_text}' - {filter_reason}")
                continue
            
            filtered.append(entity)
        
        return filtered, filter_logs
    
    @classmethod
    def _should_filter(cls, text: str, entity_name: str, entity_type: str = None) -> Optional[str]:
        """判断是否应该过滤该实体，返回过滤原因或 None"""
        # 规则0: Mustache模板变量和DSL变量不过滤
        if text.startswith('{{') or text.startswith('{{{') or (text.startswith('{') and text.endswith('}') and entity_type == 'DSL'):
            return None

        # 规则0.5: 业务概念实体（regex_fallback 提取的 concept_* 类型）不过滤
        # "订单管理"、"库存管理" 等是可配置的业务模块名，应当保留
        if entity_name and entity_name.startswith('concept_'):
            return None

        # 规则1: 包含固定描述关键词但没有数字 -> 可能是固定需求
        if any(kw in text for kw in cls.FIXED_DESCRIPTION_KEYWORDS):
            # 检查是否有数字
            if not re.search(r'\d+', text):
                # 特殊情况：技术栈列表（如 "LangChain、Milvus、FastAPI"）
                if not cls._is_tech_stack(text):
                    return "包含需求描述关键词但无数字，可能是固定需求"

        # 规则2: 明确的固定需求描述（即使有数字也要过滤）
        # "需要支持多轮对话" - 整体描述
        if text in ["需要支持多轮对话", "需要中英文双语", "用大模型做底座"]:
            return "明确的固定需求描述"

        # "响应时间控制在2秒以内" - 配置描述，应只提取 "2秒"
        if "响应时间控制" in text:
            return "配置描述，应只提取数值部分"

        # 规则3: 细分变量（如 "java_developers"）-> 如果原文有总人数，只保留总人数
        if re.search(r'(java|python|frontend|backend).*developers?', entity_name, re.IGNORECASE):
            return "细分变量，建议合并为团队总人数"

        # 规则3.5: regex_fallback 提取的数量实体不过滤
        if entity_name and entity_name.startswith('quantity_'):
            return None

        # 规则4: 检查是否包含变量关键词
        has_variable_keyword = any(kw in text for kw in cls.VARIABLE_KEYWORDS)
        if not has_variable_keyword:
            # 没有变量关键词的可能是固定描述
            # 但技术栈、专有名词例外
            if not cls._is_tech_stack(text) and not cls._is_proper_noun(text):
                return "不包含变量关键词，可能是固定描述"

        return None
    
    @classmethod
    def _is_tech_stack(cls, text: str) -> bool:
        """判断是否为技术栈列表"""
        tech_indicators = ["LangChain", "Milvus", "FastAPI", "K8s", "Kubernetes", "ELK", "Prometheus", "Grafana"]
        return any(tech in text for tech in tech_indicators) and "、" in text
    
    @classmethod
    def _is_proper_noun(cls, text: str) -> bool:
        """判断是否为专有名词（技术术语）"""
        proper_nouns = [
            # 技术框架/平台
            "RAG", "LLM", "API", "K8s", "Kubernetes",
            "LangChain", "Milvus", "FastAPI",
            # 编程语言
            "Python", "Java", "JavaScript", "TypeScript",
            "C++", "C#", "Go", "Rust", "PHP", "Ruby",
            # 数据科学/机器学习
            "机器学习", "深度学习", "数据分析", "人工智能",
            "AI", "ML", "DL", "TensorFlow", "PyTorch",
            # 数据库/存储
            "MySQL", "PostgreSQL", "MongoDB", "Redis",
            "Elasticsearch", "Milvus", "向量数据库",
            # 开发领域
            "前端开发", "后端开发", "全栈开发",
            "Web开发", "移动开发", "数据科学",
            # 其他技术术语
            "微服务", "容器化", "DevOps", "CI/CD"
        ]
        return any(noun in text for noun in proper_nouns)
    
    @staticmethod
    def merge_duplicate_entities(entities: List[Dict]) -> List[Dict]:
        """
        合并重复或重叠的实体
        
        例如：
        - "5个人" 和 "5" → 保留 "5个人"
        - "8周" 和 "周" → 保留 "8周"
        """
        if not entities:
            return []
        
        merged = []
        processed_indices = set()
        
        for i, entity1 in enumerate(entities):
            if i in processed_indices:
                continue
            
            orig_text1 = entity1.get('original_text', '')
            
            # 查找是否有其他实体包含当前实体
            for j, entity2 in enumerate(entities):
                if i >= j or j in processed_indices:
                    continue
                
                orig_text2 = entity2.get('original_text', '')
                
                # 如果 entity1 包含 entity2，保留 entity1（更长的）
                if orig_text2 in orig_text1:
                    processed_indices.add(j)
                    logger.info(f"合并实体: '{orig_text2}' → '{orig_text1}'")
            
            merged.append(entity1)
            processed_indices.add(i)
        
        return merged


# ========== 强类型清洗器 ==========
class TypeCleaner:
    """强类型清洗与转换 (Code-Layer) + 混合规则学习"""
    
    _hybrid_learner: Optional[HybridRuleLearner] = None
    _hybrid_enabled: bool = False
    
    BOOLEAN_MAP = {
        '是': True, '要': True, '需要': True, '对': True, 'yes': True, 'true': True, '1': True,
        '否': False, '不': False, '不要': False, '错': False, 'no': False, 'false': False, '0': False
    }
    
    @classmethod
    def init_hybrid(cls, use_mock: bool = True):
        """初始化混合学习器"""
        if not cls._hybrid_learner:
            cls._hybrid_learner = create_hybrid_learner(use_mock=use_mock)
            cls._hybrid_enabled = True
            info("[HYBRID] TypeCleaner 混合学习器已初始化")
    
    @classmethod
    def enable_hybrid(cls, enabled: bool = True):
        """启用/禁用混合学习"""
        cls._hybrid_enabled = enabled
    
    @classmethod
    def clean(cls, value: Any, target_type: str) -> Tuple[Any, str]:
        """
        类型清洗与转换（支持混合规则学习）
        返回: (转换后的值, 实际类型)
        """
        if cls._hybrid_enabled and cls._hybrid_learner:
            inferred_type, rule_used = cls._hybrid_learner.process_type_inference(value)
            if rule_used:
                info(f"[HYBRID] 使用规则推断类型: {value} -> {inferred_type}")
        
        try:
            if target_type == DataType.INTEGER.value:
                return cls._clean_integer(value)
            
            elif target_type == DataType.BOOLEAN.value:
                return cls._clean_boolean(value)
            
            elif target_type == DataType.LIST.value:
                return cls._clean_list(value)
            
            elif target_type == DataType.ENUM.value:
                return cls._clean_enum(value)
            
            else:
                return str(value), DataType.STRING.value
                
        except Exception as e:
            logger.warning(f"类型转换失败: {value} -> {target_type}, 错误: {e}, 降级为 String")
            return str(value), DataType.STRING.value
    
    @staticmethod
    def _clean_integer(value: Any) -> Tuple[int, str]:
        """Integer 清洗"""
        if isinstance(value, int):
            return value, DataType.INTEGER.value
        
        # 从字符串中提取数字
        if isinstance(value, str):
            numbers = re.findall(r'-?\d+', value)
            if numbers:
                return int(numbers[0]), DataType.INTEGER.value
        
        raise ValueError(f"无法解析为整数: {value}")
    
    @classmethod
    def _clean_boolean(cls, value: Any) -> Tuple[bool, str]:
        """Boolean 清洗"""
        if isinstance(value, bool):
            return value, DataType.BOOLEAN.value
        
        str_value = str(value).strip().lower()
        if str_value in cls.BOOLEAN_MAP:
            return cls.BOOLEAN_MAP[str_value], DataType.BOOLEAN.value
        
        raise ValueError(f"无法解析为布尔值: {value}")
    
    @staticmethod
    def _clean_list(value: Any) -> Tuple[List, str]:
        """List 清洗 - 检测逗号、顿号分隔符"""
        if isinstance(value, list):
            return value, DataType.LIST.value
        
        if isinstance(value, str):
            # 尝试多种分隔符
            for separator in [',', '、', '，', ';', '；']:
                if separator in value:
                    items = [item.strip() for item in value.split(separator)]
                    return items, DataType.LIST.value
            
            # 单元素列表
            return [value], DataType.LIST.value
        
        return [str(value)], DataType.LIST.value
    
    @staticmethod
    def _clean_enum(value: Any) -> Tuple[str, str]:
        """Enum 清洗"""
        return str(value), DataType.ENUM.value


# ========== 实体冲突解析器 ==========
class EntityConflictResolver:
    """处理重叠实体 (Overlapping Entities)"""

    @staticmethod
    def resolve_overlaps(entities: List[Dict]) -> List[Dict]:
        """
        最长覆盖原则: 优先保留更长的实体
        例: "3年经验" vs "3年", 保留 "3年经验"

        新增: 变量名去重逻辑
        """
        if not entities:
            return []

        # 按起始位置排序
        sorted_entities = sorted(entities, key=lambda x: (x['start_index'], -(x['end_index'] - x['start_index'])))

        non_overlapping = []
        last_end = -1

        # 第一步: 解决位置重叠
        for entity in sorted_entities:
            start = entity['start_index']
            end = entity['end_index']

            # 如果当前实体与上一个不重叠
            if start >= last_end:
                non_overlapping.append(entity)
                last_end = end
            else:
                # 发生重叠,比较长度
                if len(non_overlapping) > 0:
                    last_entity = non_overlapping[-1]
                    current_length = end - start
                    last_length = last_entity['end_index'] - last_entity['start_index']

                    if current_length > last_length:
                        # 替换为更长的实体
                        non_overlapping[-1] = entity
                        last_end = end

        # 第二步: 变量名去重（新增）
        name_counter = {}
        deduplicated_entities = []

        for entity in non_overlapping:
            name = entity['name']

            if name not in name_counter:
                # 第一次出现的变量名，直接保留
                name_counter[name] = 1
                deduplicated_entities.append(entity)
            else:
                # 重复的变量名，生成新名称
                count = name_counter[name] + 1
                name_counter[name] = count

                # 创建新变量名
                new_name = f"{name}_{count}"
                entity['name'] = new_name
                deduplicated_entities.append(entity)

        return deduplicated_entities


# ========== 核心处理引擎 ==========
class PromptStructurizer:
    """Prompt 结构化处理引擎 (主控制器)"""
    
    def __init__(
        self,
        llm_client: Optional[UnifiedLLMClient] = None,
        use_mock: bool = False
    ):
        """
        初始化结构化处理引擎
        
        Args:
            llm_client: 统一LLM客户端实例
            use_mock: 是否使用模拟模式
        """
        self.llm_extractor = LLMEntityExtractor(
            llm_client=llm_client,
            use_mock=use_mock
        )
        self.semantic_extractor = SemanticExtractor(
            llm_client=llm_client,
            use_mock=use_mock
        )
        self.firewall = HallucinationFirewall()
        self.type_cleaner = TypeCleaner()
        self.conflict_resolver = EntityConflictResolver()
        self.entity_validator = EntityPostValidator()  # 新增：后处理校验器
        self.extraction_log = []
        self.use_mock = use_mock
        self.history_manager = HistoryManager()
    
    def process_from_prompt10(
        self, 
        prompt10_result: Prompt10Result,
        save_history: bool = True
    ) -> Prompt20Result:
        """
        从 Prompt 1.0 结果进行处理
        
        Args:
            prompt10_result: Prompt 1.0 的处理结果
            save_history: 是否保存历史记录
            
        Returns:
            Prompt20Result: Prompt 2.0 结构化结果
        """
        start_time = time.time()

        # 使用 Prompt 1.0 的处理后文本
        clean_text = prompt10_result.processed_text

        # 调用核心处理流程
        structure, _ = self.process(clean_text)  # 注意：process 返回元组

        # 构建 VariableMeta 列表
        variables = [
            VariableMeta(
                name=reg["variable"],
                original_text=reg["original_text"],
                value=reg["value"],
                data_type=reg["type"],
                start_index=0,  # 简化处理
                end_index=len(reg["original_text"]),
                source_context=reg.get("source_context", "Prompt 1.0")
            )
            for reg in structure.variable_registry
        ]

        processing_time_ms = int((time.time() - start_time) * 1000)

        result = Prompt20Result(
            id=generate_id(),
            timestamp=get_timestamp(),
            source_prompt10_id=prompt10_result.id,
            original_text=prompt10_result.processed_text,
            template_text=structure.template_text,
            variables=variables,
            variable_registry=structure.variable_registry,
            extraction_log=structure.extraction_log,
            processing_time_ms=processing_time_ms
        )
        
        # 保存历史记录
        if save_history:
            self._save_history(result)
        
        return result
    
    def _save_history(self, result: Prompt20Result):
        """保存 Prompt 2.0 处理历史"""
        # 统计变量类型
        type_stats = {}
        for var in result.variables:
            dtype = var.data_type
            type_stats[dtype] = type_stats.get(dtype, 0) + 1
        
        history = Prompt20History(
            id=result.id,
            timestamp=result.timestamp,
            source_prompt10_id=result.source_prompt10_id,
            input_text=result.original_text,
            template_text=result.template_text,
            variables=result.variable_registry,
            variable_count=len(result.variables),
            type_stats=type_stats,
            extraction_log=result.extraction_log,
            processing_time_ms=result.processing_time_ms
        )
        
        try:
            self.history_manager.save_prompt20_history(history)
        except Exception as e:
            warning(f"保存 Prompt 2.0 历史记录失败: {e}")

    def process_to_prompt25(
        self,
        prompt10_result: Prompt10Result,
        save_history: bool = True
    ) -> Prompt25Result:
        """
        从 Prompt 1.0 结果生成 Prompt 2.5 结构化语义

        Args:
            prompt10_result: Prompt 1.0 的处理结果
            save_history: 是否保存历史记录

        Returns:
            Prompt25Result: Prompt 2.5 结构化语义结果
        """
        start_time = time.time()

        # 使用 Prompt 1.0 的处理后文本
        clean_text = prompt10_result.processed_text

        # 先执行标准的 Prompt 2.0 处理（获取变量）
        prompt20_result = self.process_from_prompt10(prompt10_result, save_history=False)

        # 调用语义提取器
        logger.info(f"开始语义提取: {clean_text[:50]}...")
        semantic_result = self.semantic_extractor.extract(clean_text)

        # 合并 Prompt 20 和 Prompt 25 的信息
        semantic_result.source_prompt10_id = prompt10_result.id
        semantic_result.source_prompt20_id = prompt20_result.id
        semantic_result.variables = prompt20_result.variables
        semantic_result.variable_registry = prompt20_result.variable_registry
        semantic_result.extraction_log = prompt20_result.extraction_log + semantic_result.extraction_log

        processing_time_ms = int((time.time() - start_time) * 1000)
        semantic_result.processing_time_ms = processing_time_ms

        logger.info(f"✅ Prompt 2.5 语义提取完成，置信度: {semantic_result.confidence}")
        logger.info(f"   意图: {semantic_result.intent}")
        logger.info(f"   操作数: {len(semantic_result.operations)}")
        logger.info(f"   分支数: {len(semantic_result.branches)}")
        logger.info(f"   输出字段: {len(semantic_result.output_schema)}")

        if semantic_result.ambiguities:
            logger.warning(f"   ⚠️ 存在 {len(semantic_result.ambiguities)} 个未解决的歧义")
            for amb in semantic_result.ambiguities:
                logger.warning(f"      - {amb}")

        return semantic_result

    def process(self, clean_text: str) -> Tuple[PromptStructure, Dict[str, Any]]:
        """
        主处理流程
        输入: Prompt 1.0 (已清洗文本)
        输出: (PromptStructure, 优化统计信息)
        """
        logger.info(f"开始结构化处理: {clean_text[:50]}...")
        self.extraction_log = []

        # ===== 阶段 2.1: 语义扫描与实体定位 (LLM-Layer) =====
        raw_entities, optimization_stats = self.llm_extractor.extract(clean_text)
        # 兼容处理：确保是列表
        if isinstance(raw_entities, str):
            raw_entities = json.loads(raw_entities)

        # 🔧 Fallback: LLM 提取结果为空时，使用规则引擎做兜底提取
        if not raw_entities:
            logger.info("[实体提取] LLM 返回空结果，启用规则引擎 fallback")
            raw_entities = self._regex_fallback_extract(clean_text)
            if raw_entities:
                self._log(f"规则引擎 fallback 提取到 {len(raw_entities)} 个候选实体")
                optimization_stats['regex_fallback'] = True
                optimization_stats['merged_count'] = len(raw_entities)

        # 记录优化统计
        if optimization_stats.get('optimization_enabled'):
            if optimization_stats.get('llm_called'):
                self._log(f"实体提取: 正则{optimization_stats.get('regex_count')} + LLM{optimization_stats.get('llm_count')} → {optimization_stats.get('merged_count')}")
            else:
                self._log(f"实体提取: 正则提取成功，跳过 LLM (提取{optimization_stats.get('regex_count')}个)")
        else:
            self._log(f"LLM 识别到 {len(raw_entities)} 个候选实体")
        
        # ===== 阶段 2.2: 幻觉防火墙与存在性校验 (Code-Layer) =====
        validated_entities = []
        for entity in raw_entities:
            is_valid, msg = self.firewall.validate_existence(entity, clean_text)
            if not is_valid:
                self._log(f"❌ {msg}")
                continue
            
            # 验证索引准确性
            if not self.firewall.validate_index(entity, clean_text):
                self._log(f"⚠️  索引不匹配: {entity['original_text']}, 尝试修正")
                # 自动修正索引
                entity = self._fix_entity_index(entity, clean_text)
            
            validated_entities.append(entity)
            self._log(f"✓ 验证通过: {entity['original_text']}")
        
        # ===== 解决实体冲突 =====
        resolved_entities = self.conflict_resolver.resolve_overlaps(validated_entities)
        self._log(f"冲突解析完成,保留 {len(resolved_entities)} 个实体")
        
        # ===== 阶段 2.2.5: 后处理校验（过滤固定描述）=====
        filtered_entities, filter_logs = self.entity_validator.filter_entities(resolved_entities, clean_text)
        for log in filter_logs:
            self._log(f"🔍 {log}")
        self._log(f"后处理校验完成,保留 {len(filtered_entities)} 个变量")
        
        # ===== 阶段 2.3: 强类型清洗与转换 (Code-Layer) =====
        variable_metas = []
        for entity in filtered_entities:  # 修复：使用过滤后的实体
            cleaned_value, actual_type = self.type_cleaner.clean(
                entity['value'],
                entity['type']
            )

            var_meta = VariableMeta(
                name=entity['name'],
                original_text=entity['original_text'],
                value=cleaned_value,
                data_type=actual_type,
                start_index=entity['start_index'],
                end_index=entity['end_index']
            )
            variable_metas.append(var_meta)
            self._log(f"类型转换: {entity['original_text']} -> {actual_type} = {cleaned_value}")
        
        # ===== 阶段 2.4: 模板生成与变量注入 (Code-Layer) =====
        template_text = self._generate_template(clean_text, variable_metas)
        
        # ===== 生成变量注册表 =====
        variable_registry = [
            {
                "variable": var.name,
                "value": var.value,
                "type": var.data_type,
                "original_text": var.original_text,
                "source_context": var.source_context
            }
            for var in variable_metas
        ]

        logger.info("✅ Prompt 2.0 生成完毕")

        result = PromptStructure(
            template_text=template_text,
            variable_registry=variable_registry,
            original_text=clean_text,
            extraction_log=self.extraction_log
        )

        return result, optimization_stats
    
    def _generate_template(self, original_text: str, variables: List[VariableMeta]) -> str:
        """
        生成模板文本 (防误伤策略: 索引定位替换)
        """
        # 按位置倒序排序,从后往前替换 (防止索引偏移)
        sorted_vars = sorted(variables, key=lambda v: v.start_index, reverse=True)
        
        result = original_text
        for var in sorted_vars:
            placeholder = f"{{{{{var.name}}}}}"
            result = (
                result[:var.start_index] + 
                placeholder + 
                result[var.end_index:]
            )
        
        return result
    
    def _fix_entity_index(self, entity: Dict, text: str) -> Dict:
        """自动修正实体索引"""
        snippet = entity['original_text']
        idx = text.find(snippet)
        if idx != -1:
            entity['start_index'] = idx
            entity['end_index'] = idx + len(snippet)
        return entity

    def _regex_fallback_extract(self, text: str) -> List[Dict]:
        """规则引擎 fallback：当 LLM 提取结果为空时，用正则提取基础实体"""
        entities = []

        # 1. 数字+单位模式 (如 "3年", "2周", "100个")
        for match in re.finditer(r'(\d+\.?\d*)(年|周|月|天|小时|分钟|秒|个|条|次|件|份|笔|万|亿|%|%)', text):
            entities.append({
                "name": f"quantity_{len(entities)}",
                "original_text": match.group(0),
                "start_index": match.start(),
                "end_index": match.end(),
                "type": "Number",
                "value": match.group(0),
                "category": "regex_fallback"
            })

        # 2. 专业名词短语（XX管理、XX系统 等业务概念 — 只保留核心名词）
        # 策略：匹配所有 X管理/X系统 形式，然后清洗前缀和后缀
        _BUSINESS_SUFFIX = r'(?:管理|系统|服务|模块|功能|平台|引擎|中心|流程|组件|配置|策略|规则|监控|检测|分析|推荐|搜索|通知|认证|授权|日志|缓存)'
        _NOISY_PREFIXES = ["包含", "需要", "支持", "开发", "实现", "提供", "采用", "使用", "一个", "一套", "这种", "那种", "和", "与", "及", "的"]
        _NOISY_SUFFIXES = ["模块", "功能", "组件"]
        seen_concepts = set()
        for match in re.finditer(rf'([\u4e00-\u9fa5]{{1,4}}{_BUSINESS_SUFFIX})', text):
            core_concept = match.group(1)
            # 清理前缀
            cleaned = core_concept
            for prefix in _NOISY_PREFIXES:
                if cleaned.startswith(prefix) and len(cleaned) > len(prefix) + 1:
                    cleaned = cleaned[len(prefix):]
            # 清理后缀
            for suffix in _NOISY_SUFFIXES:
                if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 1:
                    cleaned = cleaned[:-len(suffix)]
            if len(cleaned) < 2 or cleaned in seen_concepts:
                continue
            seen_concepts.add(cleaned)
            # 用 cleaned 在 text 中重新定位（更精确）
            concept_start = text.find(cleaned)
            if concept_start == -1:
                concept_start = match.start(1)
            concept_end = concept_start + len(cleaned)
            # 验证位置
            if text[concept_start:concept_end] != cleaned:
                concept_start = match.start(1)
                concept_end = match.end(1)
                cleaned = core_concept
            entities.append({
                "name": f"concept_{len(entities)}",
                "original_text": cleaned,
                "start_index": concept_start,
                "end_index": concept_end,
                "type": "String",
                "value": cleaned,
                "category": "regex_fallback"
            })

        # 3. 布尔条件关键词 (是/否/需要/必须/可选)
        for match in re.finditer(r'(必须|需要|可选|启用|禁用|支持|不支持)', text):
            entities.append({
                "name": f"condition_{len(entities)}",
                "original_text": match.group(0),
                "start_index": match.start(),
                "end_index": match.end(),
                "type": "Boolean",
                "value": match.group(0) in ("必须", "需要", "启用", "支持"),
                "category": "regex_fallback"
            })

        # 去重（同一位置不重复）
        seen_positions = set()
        deduped = []
        for e in entities:
            pos = (e['start_index'], e['end_index'])
            if pos not in seen_positions:
                seen_positions.add(pos)
                deduped.append(e)

        return deduped
    
    def _log(self, message: str):
        """记录处理日志"""
        self.extraction_log.append(message)
        logger.info(message)


# ========== 语义提取器（Prompt 2.5） ==========

class SemanticExtractor:
    """
    语义提取器 - 从自然语言中提取完整的结构化语义

    改进点：将自然语言转换为结构化操作序列，
    而不是让 DSL 转译器再次理解自然语言
    """

    # 意图分类
    INTENT_PATTERNS = {
        "query": ["查询", "获取", "搜索", "找", "获取", "retrieve", "get", "fetch", "search"],
        "command": ["执行", "运行", "启动", "调用", "发送", "execute", "run", "call", "trigger"],
        "automation": ["自动", "定时", "周期", "监控", "automate", "schedule", "periodic"],
        "analysis": ["分析", "统计", "计算", "汇总", "analyze", "count", "calculate", "aggregate"],
        "generation": ["生成", "创建", "构建", "编写", "生成", "create", "generate", "build"],
    }

    # 操作模式识别
    OPERATION_PATTERNS = {
        "filter": [
            (r"筛选[出]?(.*?)(?:的|那些)?", "field_from_context"),
            (r"只[取用]?(?:具有|有|包含)(.*)", "condition_extraction"),
            (r"(.*?)的(.*?)[等于是]", "target_field_value"),
        ],
        "sort": [
            (r"按(.*?)(?:排序|升序|降序)", "sort_field"),
            (r"(升序|降序|从高到低|从低到高)", "sort_order"),
        ],
        "limit": [
            (r"前?(\d+)个?", "limit_value"),
            (r"返回[出]?(\d+)", "limit_value"),
        ],
        "call": [
            (r"调用(.*?)\(", "function_name"),
            (r"执行(.*?)\(", "function_name"),
        ],
    }

    def __init__(self, llm_client=None, use_mock: bool = False):
        self.llm = llm_client or create_llm_client(use_mock=use_mock)
        self.use_mock = use_mock

    def extract(self, text: str) -> Prompt25Result:
        """
        从自然语言中提取完整的结构化语义

        Args:
            text: 自然语言文本

        Returns:
            Prompt25Result: 包含完整结构化语义的结果
        """
        start_time = time.time()
        result_id = generate_id()
        timestamp = get_timestamp()

        if self.use_mock:
            return self._mock_extract(text, result_id, timestamp, start_time)

        # 使用 LLM 进行语义提取
        return self._llm_extract(text, result_id, timestamp, start_time)

    def _llm_extract(self, text: str, result_id: str, timestamp: str, start_time) -> Prompt25Result:
        """使用 LLM 进行语义提取"""

        system_prompt = """你是一个专业的语义提取专家。你的任务是从自然语言需求中提取完整的结构化语义表示。

## 任务说明
1. 仔细分析用户输入的自然语言
2. 提取所有明确提到的操作和条件
3. 对于隐含的语义，做出合理推断但标记为歧义
4. 输出结构化的 JSON 表示

## 输出格式（必须严格遵循）
{
    "intent": "意图类型",
    "operations": [...],
    "branches": [...],
    "output_schema": [...],
    "ambiguities": [...],
    "confidence": 0.0-1.0
}

## intent 意图类型
- "query": 查询/检索数据（如：查询用户列表、搜索产品）
- "command": 执行操作（如：发送邮件、执行计算）
- "automation": 自动化任务（如：定时监控、周期清理）
- "analysis": 数据分析（如：统计销售额、分析用户行为）
- "generation": 内容生成（如：生成报告、创建文档）

## operations 操作序列
每个操作是一个对象，包含：

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| type | string | 操作类型 | "filter", "sort", "limit", "call", "return", "assign" |
| target | string | 操作目标 | "products", "users", "results" |
| field | string | 字段名 | "price", "created_at", "category" |
| condition | object | 筛选条件 | {"field": "category", "op": "==", "value": "electronics"} |
| value | any | 值 | 10, "vip", true |
| order | string | 排序方向 | "asc", "desc" |
| function | string | 函数名 | "send_email", "calculate_discount" |
| params | object | 函数参数 | {"template": "vip_email"} |
| source_context | string | 原文表述 | "按价格降序排列" |

### 操作类型说明

**filter - 筛选**
- condition: 必须包含 field, op, value
- op 支持: ==, !=, >, <, >=, <=, contains, in

**sort - 排序**
- field: 排序字段
- order: "asc" 或 "desc"

**limit - 限制数量**
- value: 数字

**call - 函数调用**
- function: 函数名
- params: 参数对象

**return - 返回结果**
- 通常是最后一个操作

**assign - 赋值**
- target: 变量名
- value: 赋的值

## output_schema 输出格式
每个输出字段是一个对象：

| 字段 | 类型 | 说明 |
|------|------|------|
| name | string | 字段名 |
| data_type | string | 数据类型: String, Integer, Float, Boolean, List |
| source_field | string | 源字段（如果有映射） |
| source_context | string | 原文表述 |

## branches 条件分支
{
    "condition": "DSL 条件表达式",
    "operations": [操作列表],
    "is_default": false
}

## ambiguities 未解决的歧义
列出所有无法确定的语义：
- 字段名不确定
- 数量默认值不确定
- 操作顺序不明确

## confidence 置信度
- 1.0: 完全确定，所有信息都明确
- 0.8-0.9: 基本确定，少量推断
- 0.6-0.7: 部分确定，有歧义
- <0.6: 不确定，需要用户确认

## 示例

输入: "查询产品列表，返回按价格降序排列的前10个结果，只需要名称和价格"
输出:
{
    "intent": "query",
    "operations": [
        {"type": "query", "target": "products", "source_context": "查询产品列表"},
        {"type": "sort", "target": "results", "field": "price", "order": "desc", "source_context": "按价格降序排列"},
        {"type": "limit", "target": "results", "value": 10, "source_context": "前10个结果"}
    ],
    "output_schema": [
        {"name": "name", "data_type": "String", "source_context": "名称"},
        {"name": "price", "data_type": "Float", "source_context": "价格"}
    ],
    "ambiguities": [],
    "confidence": 0.95
}

输入: "如果用户是VIP，发送折扣邮件，否则发送普通邮件"
输出:
{
    "intent": "command",
    "operations": [],
    "branches": [
        {
            "condition": "{{user_type}} == 'vip'",
            "operations": [{"type": "call", "function": "send_discount_email", "source_context": "发送折扣邮件"}],
            "is_default": false
        },
        {
            "condition": "true",
            "operations": [{"type": "call", "function": "send_normal_email", "source_context": "发送普通邮件"}],
            "is_default": true
        }
    ],
    "output_schema": [],
    "ambiguities": [],
    "confidence": 0.95
}

输入: "帮我查下最近一周的订单"
输出:
{
    "intent": "query",
    "operations": [
        {"type": "query", "target": "orders", "source_context": "订单"},
        {"type": "filter", "condition": {"field": "created_at", "op": ">=", "value": "7天前"}, "source_context": "最近一周"}
    ],
    "output_schema": [],
    "ambiguities": ["订单状态未指定", "排序方式未指定"],
    "confidence": 0.6
}

现在提取以下文本的语义：
"""

        try:
            response = self.llm.call(system_prompt, text)
            import json as json_module
            parsed = json_module.loads(response.content)

            # 转换操作列表
            operations = []
            for op_dict in parsed.get("operations", []):
                operations.append(Operation(**op_dict))

            # 转换条件分支
            branches = []
            for br_dict in parsed.get("branches", []):
                br_ops = [Operation(**op) for op in br_dict.get("operations", [])]
                branches.append(ConditionBranch(
                    condition=br_dict.get("condition", ""),
                    operations=br_ops,
                    is_default=br_dict.get("is_default", False)
                ))

            # 转换输出字段
            output_fields = []
            for field_dict in parsed.get("output_schema", []):
                output_fields.append(OutputField(**field_dict))

            processing_time_ms = int((time.time() - start_time) * 1000)

            return Prompt25Result(
                id=result_id,
                timestamp=timestamp,
                source_prompt10_id="",
                intent=parsed.get("intent", ""),
                operations=operations,
                branches=branches,
                output_schema=output_fields,
                original_text=text,
                ambiguities=parsed.get("ambiguities", []),
                confidence=parsed.get("confidence", 0.8),
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            logger.error(f"LLM 语义提取失败: {e}")
            return self._mock_extract(text, result_id, timestamp, start_time)

    def _mock_extract(self, text: str, result_id: str, timestamp: str, start_time) -> Prompt25Result:
        """模拟语义提取（用于测试）"""
        operations = []
        output_fields = []
        branches = []
        ambiguities = []

        # ========== 意图检测 ==========
        intent = "query"
        intent_keywords = {
            "query": ["查询", "获取", "搜索", "找", "retrieve", "get", "fetch"],
            "command": ["执行", "运行", "启动", "调用", "发送", "execute", "run", "call"],
            "automation": ["自动", "定时", "周期", "监控", "automate", "schedule"],
            "analysis": ["分析", "统计", "计算", "汇总", "analyze", "count"],
            "generation": ["生成", "创建", "构建", "编写", "create", "generate"],
        }
        for intent_type, keywords in intent_keywords.items():
            if any(kw in text for kw in keywords):
                intent = intent_type
                break

        # ========== 查询目标提取 ==========
        target = None
        target_patterns = [
            r"(?:查询|获取|搜索|找)(?:到?)?(.+?)(?:列表|结果|信息|数据)",
            r"(.+?)(?:列表|结果|信息|数据)",
            r"(?:产品|用户|订单|客户|员工)",
        ]
        for pattern in target_patterns:
            match = re.search(pattern, text)
            if match:
                target = match.group(1) if match.lastindex else match.group(0)
                if target and len(target) > 1:
                    operations.append(Operation(
                        type="query",
                        target=target,
                        source_context=match.group(0)
                    ))
                    break

        # ========== 排序提取 ==========
        if "排序" in text or "按" in text:
            sort_match = re.search(r"按(\w+?)(?:排序|升序|降序)?", text)
            order = "desc" if "降序" in text or "从高到低" in text else "asc"
            if sort_match:
                operations.append(Operation(
                    type="sort",
                    field=sort_match.group(1),
                    order=order,
                    source_context=sort_match.group(0)
                ))
            else:
                # 尝试直接从文本中提取字段
                field_match = re.search(r"(价格|时间|创建|更新|名称|销量)", text)
                if field_match:
                    operations.append(Operation(
                        type="sort",
                        field=field_match.group(1),
                        order=order,
                        source_context=f"按{field_match.group(1)}{'降序' if order == 'desc' else '升序'}"
                    ))

        # ========== 数量限制提取 ==========
        limit_match = re.search(r"前?(\d+)个?", text)
        if limit_match:
            operations.append(Operation(
                type="limit",
                value=int(limit_match.group(1)),
                source_context=limit_match.group(0)
            ))

        # ========== 筛选条件提取 ==========
        if "筛选" in text or "过滤" in text or "只" in text:
            filter_match = re.search(r"(?:筛选|过滤|只要|只看)(.+?)(?:的|那些)?", text)
            if filter_match:
                operations.append(Operation(
                    type="filter",
                    target=filter_match.group(1),
                    source_context=filter_match.group(0)
                ))
            else:
                # 尝试提取类别条件
                category_match = re.search(r"(热门|新品|特价|推荐)", text)
                if category_match:
                    operations.append(Operation(
                        type="filter",
                        condition={"field": "category", "op": "==", "value": category_match.group(1)},
                        source_context=category_match.group(0)
                    ))

        # ========== 输出字段提取 ==========
        field_patterns = [
            (r"(?:需要|只要|包含)(.+?)(?:和|以及)", "fields"),
            (r"(?:名称|名字|name)", "name"),
            (r"(?:价格|price)", "price"),
            (r"(?:时间|time|date)", "time"),
            (r"(?:描述|说明|desc)", "description"),
            (r"(?:销量|sales)", "sales"),
        ]
        found_fields = set()
        for pattern, field_name in field_patterns:
            if re.search(pattern, text):
                if field_name != "fields":
                    found_fields.add(field_name)
                else:
                    # 提取多个字段
                    field_match = re.search(pattern, text)
                    if field_match:
                        fields_text = field_match.group(1)
                        # 分割字段
                        for f in re.split(r"[、和以及,]", fields_text):
                            f = f.strip()
                            if f:
                                found_fields.add(f)

        for field_name in found_fields:
            output_fields.append(OutputField(
                name=field_name,
                data_type="String" if field_name in ["name", "名称", "description", "描述"] else "Unknown",
                source_context=f"提取自: {field_name}"
            ))

        # ========== 条件分支提取 ==========
        if "如果" in text or "假如" in text or "若" in text:
            branch_match = re.search(r"如果(.+?)[，,]?(?:那么|就|则)(.+?)(?:否则|不然|要是)", text)
            if branch_match:
                condition = branch_match.group(1).strip()
                action = branch_match.group(2).strip()
                branches.append(ConditionBranch(
                    condition=f"{{{{condition}}}} == '{condition}'",
                    operations=[Operation(type="call", target=action, source_context=action)],
                    is_default=False
                ))
                # 添加默认分支
                default_match = re.search(r"否?则(.*?)$", text)
                if default_match:
                    branches.append(ConditionBranch(
                        condition="true",
                        operations=[Operation(type="call", target=default_match.group(1).strip(), source_context=default_match.group(1).strip())],
                        is_default=True
                    ))

        # ========== 函数调用提取 ==========
        call_match = re.search(r"(?:调用|执行|运行)(.+?)\(", text)
        if call_match:
            operations.append(Operation(
                type="call",
                function=call_match.group(1),
                source_context=call_match.group(0)
            ))

        # ========== 返回操作 ==========
        if "返回" in text or "给出" in text:
            operations.append(Operation(
                type="return",
                source_context="返回结果"
            ))

        processing_time_ms = int((time.time() - start_time) * 1000)

        return Prompt25Result(
            id=result_id,
            timestamp=timestamp,
            source_prompt10_id="",
            intent=intent,
            operations=operations,
            branches=branches,
            output_schema=output_fields,
            original_text=text,
            ambiguities=ambiguities,
            confidence=0.6,
            processing_time_ms=processing_time_ms
        )


# ========== 使用示例 ==========
def main():
    """演示完整流程"""
    
    # 输入: Prompt 1.0 (已清洗文本)
    input_text = "请为一位有3年经验的Java程序员生成一个为期2周的Python学习计划,重点关注数据分析。"
    
    info("=" * 60)
    info("S.E.D.E Framework - Step 2: 实体抽取与变量定义")
    info("=" * 60)
    info(f"\n【输入 - Prompt 1.0】:\n{input_text}\n")
    
    # 创建处理器（使用模拟模式）
    structurizer = PromptStructurizer(use_mock=False)  # 使用真实LLM
    
    # 方式1: 直接处理文本
    result = structurizer.process(input_text)
    
    # 输出结果
    info("\n" + "=" * 60)
    info("【输出 - Prompt 2.0 模板】:")
    info("=" * 60)
    info(result.template_text)
    
    info("\n" + "=" * 60)
    info("【变量注册表 (Variable Registry)】:")
    info("=" * 60)
    info(json.dumps(result.variable_registry, indent=2, ensure_ascii=False))
    
    info("\n" + "=" * 60)
    info("【处理日志】:")
    info("=" * 60)
    for log in result.extraction_log:
        info(f"  {log}")
    
    # 方式2: 从 Prompt10Result 处理
    info("\n\n" + "=" * 60)
    info("【演示: 从 Prompt10Result 处理】")
    info("=" * 60)
    
    # 模拟一个 Prompt10Result
    mock_prompt10 = Prompt10Result(
        id="test123",
        timestamp=get_timestamp(),
        mode="dictionary",
        original_text="帮我搞个RAG应用",
        processed_text=input_text,  # 使用已处理文本
        steps=[],
        terminology_changes={},
        status="success",
        ambiguity_detected=False
    )
    
    prompt20_result = structurizer.process_from_prompt10(mock_prompt10)
    
    info(f"\n来源 Prompt 1.0 ID: {prompt20_result.source_prompt10_id}")
    info(f"生成 Prompt 2.0 ID: {prompt20_result.id}")
    info(f"模板: {prompt20_result.template_text}")
    info(f"变量数量: {len(prompt20_result.variables)}")
    
    info("\n" + "=" * 60)
    info("【验证: 模板回填】:")
    info("=" * 60)
    # 演示如何使用模板
    filled_template = result.template_text
    for var in result.variable_registry:
        placeholder = f"{{{{{var['variable']}}}}}"
        filled_template = filled_template.replace(placeholder, str(var['value']))
    info(filled_template)
    info(f"\n原文一致性: {filled_template == input_text}")


if __name__ == "__main__":
    main()



    #如何使用
    # 1. 创建处理器
    structurizer = PromptStructurizer()

    # 2. 处理文本
    result = structurizer.process("您的 Prompt 1.0 文本")

    # 3. 获取结果
    template = result.template_text  # 带占位符的模板
    variables = result.variable_registry  # 变量定义表