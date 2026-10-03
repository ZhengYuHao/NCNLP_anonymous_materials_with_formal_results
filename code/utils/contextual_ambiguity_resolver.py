"""
上下文感知歧义消解器
Contextual Ambiguity Resolver

核心功能：
1. 基于规则的歧义模式检测
2. 上下文编码与特征提取
3. 消解候选生成
4. 用户交互确认
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from logger import info, warning, debug


class AmbiguityType(Enum):
    """歧义类型枚举"""
    REFERENCE = "reference"           # 指代歧义 (这个/那个/它)
    QUANTIFIER_SCOPE = "quantifier"   # 量词作用域歧义
    TEMPORAL_SEQUENCE = "temporal"    # 时间序歧义 (先/然后/最后)
    SEMANTIC = "semantic"             # 语义歧义 (多义词)
    STRUCTURAL = "structural"         # 结构歧义


@dataclass
class AmbiguityContext:
    """歧义上下文"""
    ambiguous_span: str           # 歧义片段
    ambiguity_type: AmbiguityType  # 歧义类型
    start_pos: int               # 起始位置
    end_pos: int                 # 结束位置
    surrounding_text: str         # 周围文本
    context_left: str = ""       # 左侧上下文
    context_right: str = ""      # 右侧上下文
    matched_pattern: str = ""    # 匹配的模式


@dataclass
class DisambiguationCandidate:
    """消解候选"""
    resolution: str              # 消解方案
    confidence: float            # 置信度
    reasoning: str               # 推荐理由
    evidence: str = ""           # 支持证据
    is_user_confirmed: bool = False  # 用户是否确认


@dataclass
class AmbiguityResolution:
    """歧义消解结果"""
    original_text: str
    resolved_text: str
    ambiguities_detected: List[AmbiguityContext]
    resolutions: Dict[int, DisambiguationCandidate]  # 按位置索引的消解结果
    needs_user_confirmation: bool  # 是否需要用户确认


class ContextualAmbiguityResolver:
    """
    上下文感知歧义消解器
    
    策略：
    1. 规则模式匹配 -> 快速检测常见歧义
    2. 上下文特征提取 -> 为消解提供依据
    3. 候选生成 -> 提供多种消解选择
    4. 置信度评估 -> 自动消解或请求确认
    """
    
    # 歧义检测模式
    AMBIGUITY_PATTERNS = {
        AmbiguityType.REFERENCE: [
            # 指代词模式
            (r'(?<![a-zA-Z])(这个|那个|它|他们|这些|那些|此|该)(?=[^\s]{0,10})', 
             "指代词歧义"),
            (r'前面的?|后面的?|上一次的?|刚才的?', 
             "时间指代歧义"),
        ],
        AmbiguityType.QUANTIFIER_SCOPE: [
            # 量词作用域
            (r'(\w+)的(\w+)个?(\w+)?', 
             "量词作用域歧义"),
            (r'(所有|每个|各|全部)(\w+)', 
             "量词范围歧义"),
        ],
        AmbiguityType.TEMPORAL_SEQUENCE: [
            # 时间序
            (r'(先|首先|第一|一开始)(?:再|然后|接着|之后|第二|最后)?', 
             "时间序歧义"),
            (r'(?:之前|之后|之前|之后)(?:再|然后)?', 
             "时序关系歧义"),
        ],
        AmbiguityType.SEMANTIC: [
            # 多义词 (简化版)
            (r'(?:配置|设置)(?:\w+)?', 
             "配置相关歧义"),
            (r'(?:处理|加工|操作)', 
             "处理方式歧义"),
        ],
        AmbiguityType.STRUCTURAL: [
            # 句法结构歧义
            (r'(?:是否|能不能|可不可以)(?:需要|应该|必须)?', 
             "条件确认歧义"),
        ]
    }
    
    # 消解规则
    RESOLUTION_RULES = {
        AmbiguityType.REFERENCE: {
            "这个": {
                "nearest_noun": "指代距离最近的名词",
                "context_topic": "指代上下文主题",
            },
            "那个": {
                "previous_mention": "指向前一次提及",
                "context_topic": "指代上下文主题",
            },
            "它": {
                "singular_entity": "指向单数实体",
                "previous_subject": "指向前一主语",
            }
        },
        AmbiguityType.TEMPORAL_SEQUENCE: {
            "default": [
                "按时间顺序排列",
                "依赖关系排序"
            ]
        }
    }
    
    # 常见需要消解的词汇及其消解方案
    COMMON_RESOLUTIONS = {
        "这个": [
            ("当前系统", 0.9, "基于上下文判断"),
            ("指定模块", 0.7, "基于位置判断"),
            ("整个项目", 0.5, "基于范围判断"),
        ],
        "那个": [
            ("之前提到的", 0.85, "基于历史引用"),
            ("当前讨论的", 0.7, "基于上下文"),
        ],
        "它们": [
            ("所有相关实体", 0.8, "基于复数指代"),
            ("前文实体列表", 0.75, "基于列表上下文"),
        ],
        "先": [
            ("第一步执行", 0.9, "时间序标志"),
            ("优先处理", 0.7, "优先级标志"),
        ],
    }
    
    def __init__(
        self,
        context_window: int = 50,
        min_confidence: float = 0.7,
        auto_resolve: bool = True
    ):
        """
        初始化歧义消解器
        
        Args:
            context_window: 上下文窗口大小（字符数）
            min_confidence: 最小置信度阈值
            auto_resolve: 是否自动消解高置信度歧义
        """
        self.context_window = context_window
        self.min_confidence = min_confidence
        self.auto_resolve = auto_resolve
        
        # 已确认的消解方案
        self.confirmed_resolutions: Dict[str, str] = {}
        
        # 消解历史
        self.resolution_history: List[Dict] = []
    
    def detect(self, text: str) -> List[AmbiguityContext]:
        """
        检测文本中的歧义
        
        Args:
            text: 输入文本
            
        Returns:
            歧义上下文列表
        """
        ambiguities = []
        
        for amb_type, patterns in self.AMBIGUITY_PATTERNS.items():
            for pattern, description in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # 提取上下文
                    start = max(0, match.start() - self.context_window)
                    end = min(len(text), match.end() + self.context_window)
                    
                    context = AmbiguityContext(
                        ambiguous_span=match.group(0),
                        ambiguity_type=amb_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        surrounding_text=text[start:end],
                        context_left=text[start:match.start()],
                        context_right=text[match.end():end],
                        matched_pattern=description
                    )
                    ambiguities.append(context)
        
        # 按位置排序
        ambiguities.sort(key=lambda x: x.start_pos)
        
        # 过滤重叠的歧义（保留更具体的）
        ambiguities = self._filter_overlapping(ambiguities)
        
        return ambiguities
    
    def _filter_overlapping(self, ambiguities: List[AmbiguityContext]) -> List[AmbiguityContext]:
        """过滤重叠的歧义检测结果"""
        if not ambiguities:
            return []
        
        filtered = [ambiguities[0]]
        
        for current in ambiguities[1:]:
            last = filtered[-1]
            # 如果不重叠，添加
            if current.start_pos >= last.end_pos:
                filtered.append(current)
            else:
                # 如果重叠，保留置信度更高的（这里简化为保留更长的）
                if len(current.ambiguous_span) > len(last.ambiguous_span):
                    filtered[-1] = current
        
        return filtered
    
    def generate_candidates(
        self, 
        ambiguity: AmbiguityContext,
        context: Optional[str] = None
    ) -> List[DisambiguationCandidate]:
        """
        为歧义生成消解候选
        
        Args:
            ambiguity: 歧义上下文
            context: 完整文本上下文
            
        Returns:
            消解候选列表
        """
        term = ambiguity.ambiguous_span
        candidates = []
        
        # 1. 查表：常见词汇的消解方案
        if term in self.COMMON_RESOLUTIONS:
            for resolution, confidence, reasoning in self.COMMON_RESOLUTIONS[term]:
                candidates.append(DisambiguationCandidate(
                    resolution=resolution,
                    confidence=confidence,
                    reasoning=reasoning
                ))
        
        # 2. 规则生成：根据歧义类型
        if ambiguity.ambiguity_type == AmbiguityType.REFERENCE:
            candidates.extend(self._resolve_reference(ambiguity, context or ""))
        elif ambiguity.ambiguity_type == AmbiguityType.TEMPORAL_SEQUENCE:
            candidates.extend(self._resolve_temporal(ambiguity))
        
        # 3. 去重
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.resolution not in seen:
                seen.add(c.resolution)
                unique_candidates.append(c)
        
        # 4. 排序
        unique_candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_candidates
    
    def _resolve_reference(
        self, 
        ambiguity: AmbiguityContext,
        full_context: str
    ) -> List[DisambiguationCandidate]:
        """生成指代歧义的消解候选"""
        candidates = []
        term = ambiguity.ambiguous_span
        
        # 分析上下文中的名词实体
        nouns = self._extract_nouns(full_context)
        
        if "这个" in term or "那个" in term:
            # 指代词：优先指向最近的名词
            for i, noun in enumerate(nouns[:3]):
                confidence = 0.9 - (i * 0.2)  # 距离越近置信度越高
                candidates.append(DisambiguationCandidate(
                    resolution=noun,
                    confidence=confidence,
                    reasoning=f"距离最近的第{i+1}个名词"
                ))
        
        elif "它" in term or "它们" in term:
            # 代词：指向前文的主语
            for noun in nouns[:2]:
                candidates.append(DisambiguationCandidate(
                    resolution=noun,
                    confidence=0.75,
                    reasoning="基于代词指代规则"
                ))
        
        return candidates
    
    def _resolve_temporal(
        self, 
        ambiguity: AmbiguityContext
    ) -> List[DisambiguationCandidate]:
        """生成时间序歧义的消解候选"""
        candidates = []
        
        # 时间序关键词
        temporal_keywords = {
            "先": "第一步",
            "然后": "第二步", 
            "接着": "下一步",
            "最后": "最终步骤",
            "首先": "初始步骤"
        }
        
        term = ambiguity.ambiguous_span
        for key, resolution in temporal_keywords.items():
            if key in term:
                candidates.append(DisambiguationCandidate(
                    resolution=resolution,
                    confidence=0.85,
                    reasoning="基于时间序关键词"
                ))
        
        return candidates
    
    def _extract_nouns(self, text: str) -> List[str]:
        """提取文本中的名词（简化版）"""
        # 简化实现：基于规则的简单名词提取
        # 实际项目中可结合分词工具 (jieba, ltp)
        
        noun_patterns = [
            r'[\u4e00-\u9fa5]{2,4}(?:系统|平台|模块|服务|接口|功能|数据|信息|用户|管理员)',
            r'[\u4e00-\u9fa5]{2,4}(?:器|端|点|源|库|表|单|据|例)',
        ]
        
        nouns = []
        for pattern in noun_patterns:
            for match in re.finditer(pattern, text):
                nouns.append(match.group(0))
        
        return nouns
    
    def resolve(
        self, 
        text: str, 
        require_confirmation: bool = True
    ) -> AmbiguityResolution:
        """
        执行歧义消解
        
        Args:
            text: 输入文本
            require_confirmation: 是否需要用户确认
            
        Returns:
            消解结果
        """
        # 1. 检测歧义
        ambiguities = self.detect(text)
        
        if not ambiguities:
            return AmbiguityResolution(
                original_text=text,
                resolved_text=text,
                ambiguities_detected=[],
                resolutions={},
                needs_user_confirmation=False
            )
        
        # 2. 生成消解方案
        resolutions = {}
        needs_confirmation = False
        
        for amb in ambiguities:
            candidates = self.generate_candidates(amb, text)
            
            if not candidates:
                continue
            
            # 检查是否有已确认的消解
            if amb.ambiguous_span in self.confirmed_resolutions:
                confirmed = self.confirmed_resolutions[amb.ambiguous_span]
                resolutions[amb.start_pos] = DisambiguationCandidate(
                    resolution=confirmed,
                    confidence=1.0,
                    reasoning="使用历史确认的消解方案",
                    is_user_confirmed=True
                )
                continue
            
            # 自动消解或需要确认
            if self.auto_resolve and candidates[0].confidence >= self.min_confidence:
                resolutions[amb.start_pos] = candidates[0]
            else:
                needs_confirmation = True
                resolutions[amb.start_pos] = candidates[0]
        
        # 3. 应用消解
        resolved_text = self._apply_resolutions(text, resolutions)
        
        return AmbiguityResolution(
            original_text=text,
            resolved_text=resolved_text,
            ambiguities_detected=ambiguities,
            resolutions=resolutions,
            needs_user_confirmation=needs_confirmation and require_confirmation
        )
    
    def _apply_resolutions(
        self, 
        text: str, 
        resolutions: Dict[int, DisambiguationCandidate]
    ) -> str:
        """应用消解方案到文本"""
        result = text
        
        # 按位置倒序处理（避免位置偏移）
        sorted_positions = sorted(resolutions.keys(), reverse=True)
        
        for pos in sorted_positions:
            candidate = resolutions[pos]
            # 找到对应的歧义
            for amb in self.detect(text):
                if amb.start_pos == pos:
                    # 替换歧义词为消解方案
                    result = result[:pos] + candidate.resolution + result[amb.end_pos:]
                    break
        
        return result
    
    def confirm_resolution(
        self, 
        ambiguous_term: str, 
        resolution: str
    ):
        """
        确认消解方案
        
        Args:
            ambiguous_term: 歧义术语
            resolution: 消解方案
        """
        self.confirmed_resolutions[ambiguous_term] = resolution
        
        # 记录历史
        self.resolution_history.append({
            "term": ambiguous_term,
            "resolution": resolution,
            "timestamp": str(datetime.now())
        })
        
        info(f"确认消解: {ambiguous_term} -> {resolution}")
    
    def get_unresolved(self, text: str) -> List[Tuple[AmbiguityContext, List[DisambiguationCandidate]]]:
        """获取需要用户确认的歧义"""
        result = []
        
        for amb in self.detect(text):
            if amb.ambiguous_span not in self.confirmed_resolutions:
                candidates = self.generate_candidates(amb, text)
                result.append((amb, candidates))
        
        return result
    
    def get_statistics(self) -> Dict:
        """获取消解统计"""
        return {
            "confirmed_resolutions": len(self.confirmed_resolutions),
            "resolution_history": len(self.resolution_history),
            "auto_resolve": self.auto_resolve,
            "min_confidence": self.min_confidence
        }


# 便捷函数
def create_resolver(**kwargs) -> ContextualAmbiguityResolver:
    """创建歧义消解器"""
    return ContextualAmbiguityResolver(**kwargs)


# 辅助函数
from datetime import datetime

def detect_ambiguity(text: str) -> List[Dict]:
    """
    便捷函数：检测文本中的歧义
    
    Args:
        text: 输入文本
        
    Returns:
        歧义列表
    """
    resolver = ContextualAmbiguityResolver()
    ambiguities = resolver.detect(text)
    
    return [
        {
            "term": amb.ambiguous_span,
            "type": amb.ambiguity_type.value,
            "position": (amb.start_pos, amb.end_pos),
            "description": amb.matched_pattern,
            "context": amb.surrounding_text
        }
        for amb in ambiguities
    ]
