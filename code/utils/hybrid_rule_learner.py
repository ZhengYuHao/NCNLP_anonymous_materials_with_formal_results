"""
混合规则学习器 (Hybrid Rule Learner)
=====================================

实现"边用边学"的机制：
1. 优先使用规则库中的规则
2. 规则库无法处理时，使用 LLM 兜底
3. 将 LLM 处理结果转化为规则，存入规则库

支持规则类型：
- TERMINOLOGY: 术语映射规则
- ENTITY_EXTRACTION: 实体抽取规则
- TYPE_INFERENCE: 类型推断规则
- PATTERN_MATCH: 模式匹配规则

Author: WaAct Compiler Team
"""

import re
import json
import os
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from collections import defaultdict
from logger import info, warning, error, debug


class RuleType(Enum):
    """规则类型"""
    TERMINOLOGY = "terminology"           
    ENTITY_EXTRACTION = "entity_extraction"  
    TYPE_INFERENCE = "type_inference"    
    PATTERN_MATCH = "pattern_match"      


class RuleSource(Enum):
    """规则来源"""
    MANUAL = "manual"         
    LLM_LEARNED = "llm_learned"   
    AUTO_DISCOVERED = "auto_discovered"  


@dataclass
class Rule:
    """规则定义"""
    id: str
    rule_type: RuleType
    trigger_pattern: str      
    handler: str              
    confidence: float         
    source: RuleSource        
    usage_count: int = 0     
    success_count: int = 0   
    created_at: str = ""     
    last_used: str = ""       
    examples: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'rule_type': self.rule_type.value,
            'source': self.source.value
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Rule':
        data = dict(data)
        data['rule_type'] = RuleType(data['rule_type'])
        data['source'] = RuleSource(data['source'])
        return cls(**data)


class RuleLibrary:
    """规则库管理器"""
    
    def __init__(self, storage_path: str = ".rule_library"):
        self.storage_path = storage_path
        self.rules: Dict[str, Rule] = {}
        self.rule_index: Dict[RuleType, List[str]] = defaultdict(list)
        self._load_library()
    
    def _load_library(self):
        """加载规则库"""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)
            info(f"创建规则库目录: {self.storage_path}")
            return
        
        rules_file = os.path.join(self.storage_path, "rules.json")
        if not os.path.exists(rules_file):
            return
        
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for rule_dict in data.get('rules', []):
                    rule = Rule.from_dict(rule_dict)
                    self.rules[rule.id] = rule
                    self.rule_index[rule.rule_type].append(rule.id)
            info(f"已加载 {len(self.rules)} 条规则")
        except Exception as e:
            warning(f"加载规则库失败: {e}")
    
    def _save_library(self):
        """保存规则库"""
        rules_file = os.path.join(self.storage_path, "rules.json")
        
        try:
            data = {
                'version': '1.0',
                'updated_at': datetime.now().isoformat(),
                'rules': [rule.to_dict() for rule in self.rules.values()]
            }
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            debug(f"已保存 {len(self.rules)} 条规则")
        except Exception as e:
            error(f"保存规则库失败: {e}")
    
    def add_rule(self, rule: Rule) -> str:
        """添加规则"""
        if not rule.created_at:
            rule.created_at = datetime.now().isoformat()
        
        self.rules[rule.id] = rule
        self.rule_index[rule.rule_type].append(rule.id)
        self._save_library()
        return rule.id
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[Rule]:
        """获取指定类型的规则"""
        rule_ids = self.rule_index.get(rule_type, [])
        return [self.rules[rid] for rid in rule_ids if rid in self.rules]
    
    def find_matching_rules(self, text: str, rule_type: RuleType) -> List[Rule]:
        """查找匹配文本的规则"""
        matching = []
        for rule in self.get_rules_by_type(rule_type):
            try:
                if re.search(rule.trigger_pattern, text):
                    matching.append(rule)
            except re.error as e:
                warning(f"规则 {rule.id} 的正则表达式无效: {e}")
        return matching
    
    def update_rule_usage(self, rule_id: str, success: bool):
        """更新规则使用统计"""
        if rule_id not in self.rules:
            return
        
        rule = self.rules[rule_id]
        rule.usage_count += 1
        if success:
            rule.success_count += 1
        rule.last_used = datetime.now().isoformat()
        self._save_library()
    
    def get_statistics(self) -> Dict:
        """获取规则库统计"""
        stats = {
            'total_rules': len(self.rules),
            'by_type': {},
            'by_source': {},
            'avg_confidence': 0.0,
            'total_usage': 0,
            'overall_success_rate': 0.0
        }
        
        total_confidence = 0.0
        total_usage = 0
        total_success = 0
        
        for rule in self.rules.values():
            t = rule.rule_type.value
            s = rule.source.value
            stats['by_type'][t] = stats['by_type'].get(t, 0) + 1
            stats['by_source'][s] = stats['by_source'].get(s, 0) + 1
            
            total_confidence += rule.confidence
            total_usage += rule.usage_count
            total_success += rule.success_count
        
        if self.rules:
            stats['avg_confidence'] = total_confidence / len(self.rules)
        if total_usage > 0:
            stats['overall_success_rate'] = total_success / total_usage
        
        stats['total_usage'] = total_usage
        return stats


class LLMFallbackHandler:
    """LLM 兜底处理器"""
    
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self._llm_client = None
    
    def _get_llm_client(self):
        """获取 LLM 客户端"""
        if self._llm_client is None:
            try:
                from llm_client import invoke_function
                self._llm_client = invoke_function
            except ImportError:
                warning("LLM 客户端未找到，将使用模拟模式")
                self.use_mock = True
        return self._llm_client
    
    def handle_terminology(self, text: str, unknown_terms: List[str], context: str = "") -> Dict[str, str]:
        """处理未知术语"""
        if self.use_mock:
            return self._mock_terminology_handling(unknown_terms)
        
        llm_client = self._get_llm_client()
        if not llm_client:
            return {}
        
        try:
            result = llm_client('standardize_terms', input_text=text, unknown_terms=unknown_terms, context=context)
            if isinstance(result, dict):
                return result.get('mappings', {})
        except Exception as e:
            warning(f"LLM 术语处理失败: {e}")
            return {}
        return {}
    
    def _mock_terminology_handling(self, unknown_terms: List[str]) -> Dict[str, str]:
        """模拟术语处理"""
        mock_mappings = {}
        for term in unknown_terms:
            if term in ['那个', '这个', '它']:
                mock_mappings[term] = ''
            elif term in ['搞', '弄', '整']:
                mock_mappings[term] = '处理'
            elif term in ['套壳']:
                mock_mappings[term] = '基于API封装'
            elif term in ['大模型']:
                mock_mappings[term] = '大型语言模型'
        return mock_mappings
    
    def handle_entity_extraction(self, text: str, context: str = "") -> List[Dict]:
        """处理实体抽取（LLM兜底）"""
        if self.use_mock:
            return self._mock_entity_extraction(text)
        
        llm_client = self._get_llm_client()
        if not llm_client:
            return []
        
        try:
            result = llm_client('extract_entities', text=text, context=context)
            if isinstance(result, list):
                return result
        except Exception as e:
            warning(f"LLM 实体抽取失败: {e}")
        return []
    
    def _mock_entity_extraction(self, text: str) -> List[Dict]:
        """模拟实体抽取"""
        entities = []
        
        int_patterns = [
            (r'(\d+)\s*种', 'mode_count'),
            (r'(\d+)\s*个人', 'team_size'),
            (r'(\d+)\s*个', 'count'),
            (r'(\d+)\s*周', 'duration_weeks'),
            (r'(\d+)\s*秒', 'response_time_sec'),
        ]
        
        for pattern, name in int_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    "name": f"{name}_{len(entities)}",
                    "original_text": match.group(0),
                    "start_index": match.start(),
                    "end_index": match.end(),
                    "type": "Integer",
                    "value": match.group(1)
                })
        
        return entities
    
    def handle_type_inference(self, value: Any, context: str = "") -> str:
        """处理类型推断（LLM兜底）"""
        if self.use_mock:
            return self._mock_type_inference(value)
        
        llm_client = self._get_llm_client()
        if not llm_client:
            return "String"
        
        try:
            result = llm_client('infer_type', value=str(value), context=context)
            if isinstance(result, dict):
                return result.get('inferred_type', 'String')
        except Exception as e:
            warning(f"LLM 类型推断失败: {e}")
        return "String"
    
    def _mock_type_inference(self, value: Any) -> str:
        """模拟类型推断"""
        str_value = str(value).strip().lower()
        
        if str_value in ['true', 'false', '是', '否', '要', '不要', '1', '0']:
            return "Boolean"
        
        if re.match(r'^-?\d+$', str_value):
            return "Integer"
        
        if re.match(r'^-?\d+\.?\d*$', str_value):
            return "Float"
        
        if ',' in str_value or '、' in str_value or '，' in str_value:
            return "List"
        
        return "String"
    
    def extract_entity_rule(self, text: str, entities: List[Dict]) -> List[Rule]:
        """从实体抽取结果中提取规则"""
        rules = []
        
        for entity in entities:
            original = entity.get('original_text', '')
            entity_type = entity.get('type', 'String')
            
            if not original:
                continue
            
            pattern = re.escape(original)
            
            rule_id = f"entity_{original[:10]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            rule = Rule(
                id=rule_id,
                rule_type=RuleType.ENTITY_EXTRACTION,
                trigger_pattern=pattern,
                handler=entity_type,
                confidence=0.7,
                source=RuleSource.LLM_LEARNED,
                examples=[{
                    'text': text[:200],
                    'extracted': original,
                    'type': entity_type
                }],
                metadata={
                    'entity_name': entity.get('name', ''),
                    'value': entity.get('value', '')
                }
            )
            rules.append(rule)
        
        return rules
    
    def extract_type_rule(self, original_value: Any, inferred_type: str) -> Optional[Rule]:
        """从类型推断结果中提取规则"""
        str_value = str(original_value)
        
        if inferred_type == "Integer":
            pattern = r'-?\d+'
        elif inferred_type == "Float":
            pattern = r'-?\d+\.?\d*'
        elif inferred_type == "Boolean":
            pattern = '|'.join(['true', 'false', '是', '否', '要', '不要', '1', '0', '对', '错'])
        else:
            return None
        
        rule_id = f"type_{inferred_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return Rule(
            id=rule_id,
            rule_type=RuleType.TYPE_INFERENCE,
            trigger_pattern=pattern,
            handler=inferred_type,
            confidence=0.75,
            source=RuleSource.LLM_LEARNED,
            examples=[{'value': str_value, 'type': inferred_type}],
            metadata={'original_value': str_value}
        )


class HybridRuleLearner:
    """混合规则学习器"""
    
    def __init__(
        self,
        storage_path: str = ".rule_library",
        use_mock_llm: bool = False,
        auto_learn: bool = True,
        min_confidence_threshold: float = 0.6
    ):
        self.rule_library = RuleLibrary(storage_path)
        self.llm_handler = LLMFallbackHandler(use_mock=use_mock_llm)
        self.auto_learn = auto_learn
        self.min_confidence_threshold = min_confidence_threshold
        
        self._llm_call_count = 0
        self._rules_learned_count = 0
    
    def process_with_fallback(
        self, 
        text: str, 
        rule_type: RuleType, 
        context: str = ""
    ) -> Tuple[str, Optional[Rule]]:
        """
        统一的规则处理入口（支持多种规则类型）
        
        根据规则类型调用对应的处理方法，实现"边用边学"机制：
        1. 优先使用规则库中的已有规则
        2. 规则无法处理时，使用 LLM 兜底
        3. 将 LLM 处理结果转化为规则，存入规则库
        
        Args:
            text: 输入文本
            rule_type: 规则类型 (TERMINOLOGY, ENTITY_EXTRACTION, TYPE_INFERENCE, PATTERN_MATCH)
            context: 上下文信息
            
        Returns:
            (处理后的文本, 使用的规则对象或None)
        """
        if rule_type == RuleType.TERMINOLOGY:
            return self.process_terminology(text, context)
        elif rule_type == RuleType.ENTITY_EXTRACTION:
            # 实体抽取返回的是实体列表，这里统一返回文本和规则
            entities, rule_used = self.process_entity_extraction(text, context)
            # 对于实体抽取，我们返回原始文本（因为实体抽取不修改文本）
            # 规则使用情况通过 rule_used 标记
            return text, None if not rule_used else Rule(
                id="entity_extraction_used",
                rule_type=RuleType.ENTITY_EXTRACTION,
                trigger_pattern="",
                handler=str(len(entities)),
                confidence=1.0,
                source=RuleSource.LLM_LEARNED
            )
        elif rule_type == RuleType.TYPE_INFERENCE:
            # 类型推断需要传入值而非文本
            inferred_type, rule_used = self.process_type_inference(text, context)
            return text, None if not rule_used else Rule(
                id="type_inference_used",
                rule_type=RuleType.TYPE_INFERENCE,
                trigger_pattern="",
                handler=inferred_type,
                confidence=1.0,
                source=RuleSource.LLM_LEARNED
            )
        else:
            # 未知类型，直接返回原文本
            warning(f"未知的规则类型: {rule_type}")
            return text, None
    
    def process_terminology(self, text: str, context: str = "") -> Tuple[str, Optional[Rule]]:
        """术语处理"""
        matching_rules = self.rule_library.find_matching_rules(text, RuleType.TERMINOLOGY)
        
        if matching_rules:
            best_rule = max(matching_rules, key=lambda r: r.confidence)
            self.rule_library.update_rule_usage(best_rule.id, success=True)
            processed_text = re.sub(best_rule.trigger_pattern, best_rule.handler, text)
            return processed_text, best_rule
        
        if self.auto_learn:
            return self._llm_learn_terminology(text, context)
        
        return text, None
    
    def _llm_learn_terminology(self, text: str, context: str) -> Tuple[str, Optional[Rule]]:
        """LLM兜底学习术语"""
        self._llm_call_count += 1
        
        unknown_terms = self._detect_unknown_terms(text)
        if not unknown_terms:
            return text, None
        
        mappings = self.llm_handler.handle_terminology(text, unknown_terms, context)
        
        if mappings:
            applied_text = text
            for original, standardized in mappings.items():
                applied_text = applied_text.replace(original, standardized)
            
            if self.auto_learn and standardized:
                rule_id = f"term_{original}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                rule = Rule(
                    id=rule_id,
                    rule_type=RuleType.TERMINOLOGY,
                    trigger_pattern=re.escape(original),
                    handler=standardized,
                    confidence=0.7,
                    source=RuleSource.LLM_LEARNED,
                    examples=[{'input': original, 'output': standardized}]
                )
                self.rule_library.add_rule(rule)
                self._rules_learned_count += 1
                info(f"学习到术语规则: {original} -> {standardized}")
                return applied_text, rule
        
        return text, None
    
    def process_entity_extraction(self, text: str, context: str = "") -> Tuple[List[Dict], bool]:
        """
        实体抽取处理
        
        Returns:
            (实体列表, 是否使用了规则)
        """
        matching_rules = self.rule_library.find_matching_rules(text, RuleType.ENTITY_EXTRACTION)
        
        if matching_rules:
            entities = []
            for rule in matching_rules:
                self.rule_library.update_rule_usage(rule.id, success=True)
                matches = re.finditer(rule.trigger_pattern, text)
                for match in matches:
                    entities.append({
                        "name": rule.metadata.get('entity_name', 'unknown'),
                        "original_text": match.group(0),
                        "start_index": match.start(),
                        "end_index": match.end(),
                        "type": rule.handler,
                        "value": rule.metadata.get('value', match.group(0))
                    })
            return entities, True
        
        if self.auto_learn:
            return self._llm_learn_entity_extraction(text, context)
        
        return [], False
    
    def _llm_learn_entity_extraction(self, text: str, context: str) -> Tuple[List[Dict], bool]:
        """LLM兜底学习实体抽取"""
        self._llm_call_count += 1
        
        entities = self.llm_handler.handle_entity_extraction(text, context)
        
        if entities and self.auto_learn:
            rules = self.llm_handler.extract_entity_rule(text, entities)
            for rule in rules:
                self.rule_library.add_rule(rule)
            self._rules_learned_count += len(rules)
            info(f"学习到 {len(rules)} 条实体抽取规则")
        
        return entities, bool(entities)
    
    def process_type_inference(self, value: Any, context: str = "") -> Tuple[str, bool]:
        """
        类型推断处理
        
        Returns:
            (推断的类型, 是否使用了规则)
        """
        str_value = str(value)
        
        matching_rules = self.rule_library.find_matching_rules(str_value, RuleType.TYPE_INFERENCE)
        
        if matching_rules:
            best_rule = max(matching_rules, key=lambda r: r.confidence)
            self.rule_library.update_rule_usage(best_rule.id, success=True)
            return best_rule.handler, True
        
        if self.auto_learn:
            return self._llm_learn_type_inference(value, context)
        
        return "String", False
    
    def _llm_learn_type_inference(self, value: Any, context: str) -> Tuple[str, bool]:
        """LLM兜底学习类型推断"""
        self._llm_call_count += 1
        
        inferred_type = self.llm_handler.handle_type_inference(value, context)
        
        if inferred_type and self.auto_learn:
            rule = self.llm_handler.extract_type_rule(value, inferred_type)
            if rule:
                self.rule_library.add_rule(rule)
                self._rules_learned_count += 1
                info(f"学习到类型推断规则: {value} -> {inferred_type}")
        
        return inferred_type, False
    
    def _detect_unknown_terms(self, text: str) -> List[str]:
        """检测未知术语"""
        existing_rules = self.rule_library.get_rules_by_type(RuleType.TERMINOLOGY)
        existing_patterns = [r.trigger_pattern for r in existing_rules]
        
        unknown_terms = []
        colloquial_terms = ['那个', '这个', '它', '他们', '这些', '那些']
        
        for term in colloquial_terms:
            if term in text:
                is_known = False
                for pattern in existing_patterns:
                    if re.search(pattern, term):
                        is_known = True
                        break
                if not is_known and term not in unknown_terms:
                    unknown_terms.append(term)
        
        return unknown_terms
    
    def get_statistics(self) -> Dict:
        """获取学习统计"""
        lib_stats = self.rule_library.get_statistics()
        return {
            **lib_stats,
            'llm_calls': self._llm_call_count,
            'rules_learned': self._rules_learned_count,
            'auto_learn_enabled': self.auto_learn
        }
    
    def confirm_rule(self, rule_id: str, new_handler: Optional[str] = None):
        """确认/更新规则"""
        if rule_id not in self.rule_library.rules:
            return
        
        rule = self.rule_library.rules[rule_id]
        rule.confidence = 1.0
        rule.source = RuleSource.MANUAL
        
        if new_handler:
            rule.handler = new_handler
        
        self.rule_library._save_library()
        info(f"已确认规则: {rule_id}")


def create_hybrid_learner(
    storage_path: str = ".rule_library",
    use_mock: bool = False
) -> HybridRuleLearner:
    """创建混合学习器"""
    return HybridRuleLearner(
        storage_path=storage_path,
        use_mock_llm=use_mock,
        auto_learn=True
    )
