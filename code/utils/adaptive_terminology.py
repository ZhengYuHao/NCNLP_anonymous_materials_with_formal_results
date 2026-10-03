"""
自适应术语学习系统
从历史处理数据中自动学习新术语，更新术语映射表

核心功能：
1. 从历史数据中提取候选术语
2. 频率统计与领域相关性分析
3. 置信度评估与术语推荐
4. 自动更新术语映射表
"""

import re
import json
import os
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from datetime import datetime
from logger import info, warning, debug


@dataclass
class TermStats:
    """术语统计信息"""
    term: str
    frequency: int = 0           # 出现频率
    contexts: List[str] = field(default_factory=list)  # 上下文样本
    first_seen: str = ""         # 首次出现时间
    last_seen: str = ""          # 最后出现时间
    confirmed: bool = False      # 是否已确认
    suggested_replacement: str = ""  # 建议的标准化替换
    confidence: float = 0.0      # 置信度 [0, 1]
    source: str = "auto"         # 来源: auto/user/history


@dataclass
class TermSuggestion:
    """术语建议"""
    original_term: str           # 原始术语
    suggested_replacement: str   # 建议替换
    confidence: float            # 置信度
    evidence_count: int          # 证据数量
    contexts: List[str]          # 上下文样本
    reasoning: str               # 推荐理由


class AdaptiveTermLearner:
    """
    自适应术语学习器
    
    从历史处理数据中学习，不断优化术语映射表
    """
    
    # 术语候选模式 (基于中文特点)
    CANDIDATE_PATTERNS = [
        # 技术术语模式
        r'[\u4e00-\u9fa5]+(?:技术|系统|平台|框架|引擎|模块|组件)',
        r'(?:基于|采用|使用)[\u4e00-\u9fa5]+(?:技术|系统|平台)',
        # 缩写/简称模式
        r'[A-Z]{2,}(?:\+[A-Z]+)*(?:\s*[\u4e00-\u9fa5]+)?',
        # 网络/软件术语
        r'[\u4e00-\u9fa5]+(?:服务|接口|协议|标准|规范)',
        # 领域专业术语
        r'[\u4e00-\u9fa5]{2,}(?:化|性|度|率|值|量|法|式|类|型)',
    ]
    
    # 常见需要标准化的词汇模式
    COLLOQUIAL_PATTERNS = [
        (r'那个', ''),
        (r'这个', ''),
        (r'弄|整|搞', '处理'),
        (r'套壳', '基于API封装'),
        (r'大模型', '大型语言模型'),
        (r'RAG', '检索增强生成'),
        (r'LLM', '大型语言模型'),
    ]
    
    def __init__(
        self,
        min_frequency: int = 2,
        min_confidence: float = 0.6,
        storage_path: str = ".term_learning"
    ):
        """
        初始化术语学习器
        
        Args:
            min_frequency: 最小出现频率（低于此值不考虑）
            min_confidence: 最小置信度阈值
            storage_path: 术语库存储路径
        """
        self.min_frequency = min_frequency
        self.min_confidence = min_confidence
        self.storage_path = storage_path
        
        # 术语统计库
        self.term_stats: Dict[str, TermStats] = {}
        
        # 已确认的术语映射
        self.confirmed_mappings: Dict[str, str] = {}
        
        # 加载已有术语库
        self._load_term_library()
    
    def _load_term_library(self):
        """加载已有术语库"""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)
            return
        
        stats_file = os.path.join(self.storage_path, "term_stats.json")
        mappings_file = os.path.join(self.storage_path, "confirmed_mappings.json")
        
        # 加载术语统计
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for term, stats_dict in data.items():
                        self.term_stats[term] = TermStats(**stats_dict)
                info(f"已加载 {len(self.term_stats)} 个术语统计")
            except Exception as e:
                warning(f"加载术语统计失败: {e}")
        
        # 加载已确认映射
        if os.path.exists(mappings_file):
            try:
                with open(mappings_file, 'r', encoding='utf-8') as f:
                    self.confirmed_mappings = json.load(f)
                info(f"已加载 {len(self.confirmed_mappings)} 个已确认映射")
            except Exception as e:
                warning(f"加载已确认映射失败: {e}")
    
    def _save_term_library(self):
        """保存术语库到文件"""
        stats_file = os.path.join(self.storage_path, "term_stats.json")
        mappings_file = os.path.join(self.storage_path, "confirmed_mappings.json")
        
        # 保存术语统计
        try:
            stats_data = {
                term: {
                    'term': stats.term,
                    'frequency': stats.frequency,
                    'contexts': stats.contexts,
                    'first_seen': stats.first_seen,
                    'last_seen': stats.last_seen,
                    'confirmed': stats.confirmed,
                    'suggested_replacement': stats.suggested_replacement,
                    'confidence': stats.confidence,
                    'source': stats.source
                }
                for term, stats in self.term_stats.items()
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            warning(f"保存术语统计失败: {e}")
        
        # 保存已确认映射
        try:
            with open(mappings_file, 'w', encoding='utf-8') as f:
                json.dump(self.confirmed_mappings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            warning(f"保存已确认映射失败: {e}")
    
    def _extract_candidate_terms(self, text: str) -> List[Tuple[str, str]]:
        """
        从文本中提取候选术语
        
        Returns:
            List[(候选术语, 上下文)]
        """
        candidates = []
        
        # 使用正则模式提取
        for pattern in self.CANDIDATE_PATTERNS:
            for match in re.finditer(pattern, text):
                term = match.group(0)
                # 获取上下文 (前后各20个字符)
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]
                
                # 过滤太短或太长的术语
                if 2 <= len(term) <= 20:
                    candidates.append((term, context))
        
        # 额外检测：常见的口语词/缩写
        for pattern, replacement in self.COLLOQUIAL_PATTERNS:
            if re.search(pattern, text):
                # 找到匹配，但不确定具体内容
                # 这种情况通常需要结合上下文判断
                pass
        
        return candidates
    
    def learn_from_text(self, text: str, original_terms: Optional[Set[str]] = None):
        """
        从单个文本中学习术语
        
        Args:
            text: 处理后的文本
            original_terms: 已知的原始术语集合（用于对比学习）
        """
        candidates = self._extract_candidate_terms(text)
        now = datetime.now().isoformat()
        
        for term, context in candidates:
            if term in self.term_stats:
                # 更新已有术语
                stats = self.term_stats[term]
                stats.frequency += 1
                stats.last_seen = now
                if context not in stats.contexts:
                    stats.contexts.append(context)
                    # 只保留最近的5个上下文
                    if len(stats.contexts) > 5:
                        stats.contexts = stats.contexts[-5:]
            else:
                # 新术语
                self.term_stats[term] = TermStats(
                    term=term,
                    frequency=1,
                    contexts=[context],
                    first_seen=now,
                    last_seen=now
                )
        
        # 如果提供了原始术语，学习标准化映射
        if original_terms:
            self._learn_mappings(text, original_terms)
    
    def _learn_mappings(self, processed_text: str, original_terms: Set[str]):
        """
        学习术语映射关系
        
        通过对比原始文本和处理后文本，推断映射关系
        """
        # 这是一个简化实现，实际需要更复杂的对齐算法
        # 目前主要依赖 LLM 提供的一致性映射
        
        # 检测是否有已知的映射模式
        for term in original_terms:
            if term in self.confirmed_mappings:
                continue  # 已有映射，跳过
            
            # 检查术语是否在处理后的文本中消失（被替换）
            if term not in processed_text:
                # 术语被替换了，但不知道替换成什么
                # 需要结合其他信息（如用户反馈）来学习
                pass
    
    def analyze_candidates(self) -> List[TermSuggestion]:
        """
        分析候选术语，生成建议
        
        Returns:
            术语建议列表（按置信度排序）
        """
        suggestions = []
        
        for term, stats in self.term_stats.items():
            # 跳过已确认的术语
            if stats.confirmed:
                continue
            
            # 跳过频率过低的
            if stats.frequency < self.min_frequency:
                continue
            
            # 计算置信度
            confidence = self._calculate_confidence(stats)
            stats.confidence = confidence
            
            # 低于阈值的跳过
            if confidence < self.min_confidence:
                continue
            
            # 生成建议
            suggestion = self._generate_suggestion(term, stats)
            if suggestion:
                suggestions.append(suggestion)
        
        # 按置信度排序
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions
    
    def _calculate_confidence(self, stats: TermStats) -> float:
        """
        计算术语的置信度
        
        置信度因素：
        1. 出现频率 (40%)
        2. 上下文多样性 (30%)
        3. 模式匹配质量 (30%)
        """
        # 频率得分 (归一化到 0-1)
        freq_score = min(stats.frequency / 10.0, 1.0) * 0.4
        
        # 上下文多样性得分
        unique_contexts = len(set(stats.contexts))
        context_score = min(unique_contexts / 5.0, 1.0) * 0.3
        
        # 模式匹配得分 (基于术语长度和结构)
        pattern_score = 0.0
        if re.match(r'[\u4e00-\u9fa5]+(?:技术|系统|平台|框架)', stats.term):
            pattern_score = 0.3  # 明显的专业术语
        elif re.match(r'[\u4e00-\u9fa5]{2,4}', stats.term):
            pattern_score = 0.2  # 常规中文词
        elif re.match(r'[A-Z]{2,}', stats.term):
            pattern_score = 0.25  # 英文缩写
        
        return freq_score + context_score + pattern_score
    
    def _generate_suggestion(self, term: str, stats: TermStats) -> Optional[TermSuggestion]:
        """生成术语建议"""
        
        # 根据术语特征推断建议
        suggested = ""
        reasoning = ""
        
        # 常见模式匹配
        if '系统' in term or '平台' in term:
            suggested = term  # 保持原样
            reasoning = "检测到专业系统/平台术语"
        elif re.match(r'[A-Z]{2,}', term):
            suggested = term
            reasoning = "检测到英文缩写术语"
        elif '化' in term[-1]:
            suggested = term
            reasoning = "检测到动名词化术语"
        else:
            # 需要更多上下文
            return None
        
        return TermSuggestion(
            original_term=term,
            suggested_replacement=suggested,
            confidence=stats.confidence,
            evidence_count=stats.frequency,
            contexts=stats.contexts[:3],
            reasoning=reasoning
        )
    
    def confirm_mapping(self, original: str, replacement: str):
        """
        确认术语映射
        
        Args:
            original: 原始术语
            replacement: 标准化替换
        """
        self.confirmed_mappings[original] = replacement
        
        if original in self.term_stats:
            self.term_stats[original].confirmed = True
            self.term_stats[original].suggested_replacement = replacement
            self.term_stats[original].source = "user"
        
        self._save_term_library()
        info(f"确认术语映射: {original} -> {replacement}")
    
    def reject_term(self, term: str):
        """
        拒绝术语（从候选中移除）
        
        Args:
            term: 要拒绝的术语
        """
        if term in self.term_stats:
            self.term_stats[term].confirmed = True  # 标记为已处理（但不使用）
            self.term_stats[term].confidence = 0.0
        
        self._save_term_library()
        info(f"拒绝术语: {term}")
    
    def get_term_mapping(self) -> Dict[str, str]:
        """
        获取当前的术语映射表
        
        Returns:
            术语映射字典
        """
        mapping = self.confirmed_mappings.copy()
        
        # 添加高置信度的自动建议
        for term, stats in self.term_stats.items():
            if not stats.confirmed and stats.confidence >= self.min_confidence:
                if stats.suggested_replacement:
                    mapping[term] = stats.suggested_replacement
        
        return mapping
    
    def get_statistics(self) -> Dict:
        """获取学习统计信息"""
        total = len(self.term_stats)
        confirmed = sum(1 for s in self.term_stats.values() if s.confirmed)
        high_confidence = sum(1 for s in self.term_stats.values() if s.confidence >= 0.8)
        
        return {
            "total_candidates": total,
            "confirmed_terms": confirmed,
            "high_confidence_terms": high_confidence,
            "total_mappings": len(self.confirmed_mappings),
            "min_frequency": self.min_frequency,
            "min_confidence": self.min_confidence
        }


class TermLearningPipeline:
    """
    术语学习流水线
    
    整合历史数据学习、新术语发现、映射更新
    """
    
    def __init__(self, learner: Optional[AdaptiveTermLearner] = None):
        self.learner = learner or AdaptiveTermLearner()
    
    def learn_from_history(self, history_entries: List[Dict]) -> Dict:
        """
        从历史记录中学习
        
        Args:
            history_entries: 历史记录列表
            
        Returns:
            学习结果摘要
        """
        total_texts = 0
        new_terms = 0
        
        for entry in history_entries:
            # 提取文本
            raw_text = entry.get('raw_input', '')
            processed_text = entry.get('prompt10_processed', '')
            
            if raw_text and processed_text:
                # 提取原始术语（从术语变化中）
                original_terms = set()
                term_changes = entry.get('prompt10_terminology_changes', {})
                if term_changes:
                    original_terms = set(term_changes.keys())
                
                # 学习
                self.learner.learn_from_text(processed_text, original_terms)
                total_texts += 1
        
        # 分析候选
        suggestions = self.learner.analyze_candidates()
        
        return {
            "texts_processed": total_texts,
            "new_suggestions": len(suggestions),
            "suggestions": [
                {
                    "term": s.original_term,
                    "replacement": s.suggested_replacement,
                    "confidence": s.confidence,
                    "reasoning": s.reasoning
                }
                for s in suggestions[:10]  # 返回前10个
            ],
            "statistics": self.learner.get_statistics()
        }
    
    def auto_update_mappings(self, threshold: float = 0.85) -> int:
        """
        自动更新高置信度映射
        
        Args:
            threshold: 自动确认的置信度阈值
            
        Returns:
            更新的映射数量
        """
        suggestions = self.learner.analyze_candidates()
        updated = 0
        
        for suggestion in suggestions:
            if suggestion.confidence >= threshold:
                self.learner.confirm_mapping(
                    suggestion.original_term,
                    suggestion.suggested_replacement
                )
                updated += 1
        
        return updated


# 便捷函数
def create_term_learner(**kwargs) -> AdaptiveTermLearner:
    """创建术语学习器"""
    return AdaptiveTermLearner(**kwargs)
