"""
提示词预处理与标准化系统 (Prompt Preprocessing & Standardization System)
版本: 3.0
支持模式: 
  - 词表模式 (Dictionary Mode): 基于预定义术语映射和歧义词黑名单
  - 智能模式 (Smart Mode): 纯LLM驱动的自适应处理
  - 混合模式 (Hybrid Mode): 词表优先 + LLM兜底
"""

import os
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
from logger import info, warning, error, debug
from history_manager import HistoryManager, ProcessingHistory
from llm_client import UnifiedLLMClient, create_llm_client
from data_models import (
    ProcessingMode, ProcessingStatus, StepSnapshot,
    Prompt10Result, create_prompt10_result, generate_id, get_timestamp
)

# 优化模块：规则引擎
from utils.rule_based_normalizer import RuleBasedTextNormalizer, SyntacticAmbiguityDetector
from utils.hybrid_rule_learner import HybridRuleLearner, RuleType, create_hybrid_learner
from enum import Enum


class InputType(Enum):
    """输入文本类型枚举"""
    NATURAL_LANGUAGE = "natural_language"      # 口语化自然语言需求
    STRUCTURED_PROMPT = "structured_prompt"    # 结构化提示词规范文档
    MIXED = "mixed"                            # 混合类型


# ============================================================================
# 核心预处理器
# ============================================================================

class PromptPreprocessor:
    """提示词预处理核心引擎"""
    
    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.DICTIONARY,
        term_mapping: Optional[Dict[str, str]] = None,
        ambiguity_blacklist: Optional[List[str]] = None,
        llm_client: Optional[UnifiedLLMClient] = None,
        enable_deep_check: bool = True,
        use_mock_llm: bool = False
    ):
        """
        初始化预处理器
        
        Args:
            mode: 处理模式 (dictionary/smart/hybrid)
            term_mapping: 术语映射表 {'旧词': '标准词'}
            ambiguity_blacklist: 歧义词黑名单
            llm_client: 统一LLM客户端实例
            enable_deep_check: 是否启用深度结构歧义检测
            use_mock_llm: 是否使用模拟LLM（用于测试）
        """
        self.mode = mode
        self.term_mapping = term_mapping or {}
        self.ambiguity_blacklist = ambiguity_blacklist or []
        self.llm = llm_client or create_llm_client(use_mock=use_mock_llm)
        self.enable_deep_check = enable_deep_check
        
        # 混合规则学习器（边用边学机制）
        self.hybrid_learner = create_hybrid_learner(use_mock=use_mock_llm)
        
        # 日志记录
        self.processing_log: List[str] = []
        self.warnings: List[str] = []
        
        # 中间步骤快照
        self.steps: List[StepSnapshot] = []
        self.llm_calls_count: int = 0
        
        # 历史记录管理器
        self.history_manager = HistoryManager()

        # 输入类型检测结果
        self.detected_input_type: Optional[InputType] = None

    def detect_input_type(self, text: str) -> InputType:
        """
        检测输入文本的类型

        策略：
        1. 检查Mustache模板变量 {{...}} - 结构化提示词特征
        2. 检查DSL占位符 {...} - 结构化提示词特征
        3. 检查结构化标记（##、###、**加粗**等）- 规范文档特征
        4. 检查口语化指标（语气词、问号比例等）- 口语化特征

        Returns:
            InputType: 检测到的输入类型
        """
        mustache_pattern = r'\{\{[^}]+\}\}|\{\{\{[^}]+\}\}\}'
        dsl_pattern = r'\{[A-Za-z_][A-Za-z0-9_]*\([^)]*\)\}'
        structured_markers = [r'^##\s+', r'^###\s+', r'\*\*[^*]+\*\*', r'^\d+\.\s+', r'^-\s+']
        colloquial_words = ['那个', '吧', '嘛', '呢', '哈', '呀', '噢', '哇', '呃', '嗯', '喂']
        question_marks = text.count('？') + text.count('?')
        total_chars = len(text)

        mustache_matches = re.findall(mustache_pattern, text)
        dsl_matches = re.findall(dsl_pattern, text)
        structured_count = sum(1 for marker in structured_markers if re.search(marker, text, re.MULTILINE))
        colloquial_count = sum(text.count(word) for word in colloquial_words)

        has_mustache = len(mustache_matches) > 0
        has_dsl = len(dsl_matches) > 0
        is_highly_structured = structured_count >= 3
        is_highly_colloquial = colloquial_count >= 3 or (total_chars > 0 and question_marks / total_chars > 0.05)

        if has_mustache or has_dsl:
            self.detected_input_type = InputType.STRUCTURED_PROMPT
        elif is_highly_structured and not is_highly_colloquial:
            self.detected_input_type = InputType.STRUCTURED_PROMPT
        elif is_highly_colloquial and not is_highly_structured:
            self.detected_input_type = InputType.NATURAL_LANGUAGE
        else:
            self.detected_input_type = InputType.MIXED

        debug(f"[输入类型检测] Mustache={len(mustache_matches)}, DSL={len(dsl_matches)}, "
              f"结构化标记={structured_count}, 口语化词={colloquial_count}")
        debug(f"[输入类型检测] 判定结果: {self.detected_input_type.value}")

        return self.detected_input_type

    def _protect_structured_placeholders(self, text: str) -> Dict[str, str]:
        """
        保护Mustache变量和DSL占位符，将其替换为临时占位符

        Returns:
            Dict[str, str]: 临时占位符 -> 原始值的映射
        """
        protected = {}
        counter = 0

        mustache_pattern = r'(\{\{[^}]+\}\}|\{\{\{[^}]+\}\}\})'
        for match in re.finditer(mustache_pattern, text):
            placeholder = f"__PROTECTED_MUSTACHE_{counter}__"
            protected[placeholder] = match.group(1)
            text = text.replace(match.group(1), placeholder)
            counter += 1

        dsl_pattern = r'(\{[A-Za-z_][A-Za-z0-9_]*\([^)]*\)\})'
        for match in re.finditer(dsl_pattern, text):
            placeholder = f"__PROTECTED_DSL_{counter}__"
            protected[placeholder] = match.group(1)
            text = text.replace(match.group(1), placeholder)
            counter += 1

        return protected

    def _restore_structured_placeholders(self, text: str, protected: Dict[str, str]) -> str:
        """
        恢复被保护的Mustache变量和DSL占位符
        """
        for placeholder, original in protected.items():
            text = text.replace(placeholder, original)
        return text

    # ========================================================================
    # 阶段 1.1: 术语标准化 (基于规则)
    # ========================================================================
    
    def _normalize_terminology(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        【纯代码操作】术语强一致性对齐
        
        策略: 最长匹配原则 (Longest Match First)
        返回: (处理后文本, 替换记录字典)
        """
        if not self.term_mapping:
            return text, {}
        
        start_time = time.time()
        
        changes: Dict[str, str] = {}
        occurrences: List[str] = []
        
        # 按关键词长度降序排列 (防止短词误伤长词)
        sorted_terms = sorted(self.term_mapping.keys(), key=len, reverse=True)
        
        # 编译正则表达式 (提升性能)
        pattern = re.compile('|'.join(map(re.escape, sorted_terms)))
        
        def replace_func(match):
            original = match.group(0)
            replacement = self.term_mapping[original]
            changes[original] = replacement
            occurrences.append(original)
            return replacement
        
        normalized_text = pattern.sub(replace_func, text)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        if changes:
            n_types, n_total = len(changes), len(occurrences)
            msg = f"[OK] 术语对齐: {n_types} 类术语共 {n_total} 处"
            self.processing_log.append(msg)
            
            # 记录步骤快照
            self._add_step_snapshot(
                step_name="术语对齐",
                input_text=text,
                output_text=normalized_text,
                changes=changes,
                duration_ms=duration_ms,
                notes=[f"替换 {n_total} 处"]
            )
        
        return normalized_text, changes
    
    # ========================================================================
    # 阶段 1.1.5: 混合规则学习（边用边学机制）
    # ========================================================================
    
    def _learn_unknown_terminology(
        self, 
        text: str, 
        existing_changes: Dict[str, str]
    ) -> Tuple[str, Dict[str, str]]:
        """
        【混合模式】未知术语的边用边学处理
        
        当词表无法处理时，使用 HybridRuleLearner 进行兜底处理，
        并将学习到的规则添加到规则库中。
        
        Returns:
            (处理后文本, 新增的替换记录字典)
        """
        new_changes: Dict[str, str] = {}
        
        if self.mode == ProcessingMode.DICTIONARY:
            return text, new_changes
        
        unknown_terms = []
        for term in self.ambiguity_blacklist:
            if term in text and term not in existing_changes:
                unknown_terms.append(term)
        
        if not unknown_terms:
            return text, new_changes
        
        processed_text, used_rule = self.hybrid_learner.process_with_fallback(
            text,
            RuleType.TERMINOLOGY,
            context=f"mode:{self.mode.value}"
        )
        
        if used_rule:
            new_changes[used_rule.trigger_pattern] = used_rule.handler
            info(f"[HYBRID] 使用学习到的规则处理未知术语: {used_rule.trigger_pattern} -> {used_rule.handler}")
        
        return processed_text, new_changes
    
    # ========================================================================
    # 阶段 1.2: 歧义阻断 (启发式规则)
    # ========================================================================
    
    def _check_heuristic_ambiguity(self, text: str) -> Tuple[bool, List[str]]:
        """
        【纯代码操作】基于黑名单的硬性歧义检测
        
        返回: (是否存在歧义, 歧义词列表)
        """
        found_ambiguities = [
            word for word in self.ambiguity_blacklist 
            if word in text
        ]
        
        if found_ambiguities:
            n = len(found_ambiguities)
            self.warnings.append(
                f"[WARN] 检测到歧义词共 {n} 个: {found_ambiguities}"
            )
        
        return bool(found_ambiguities), found_ambiguities
    
    # ========================================================================
    # 阶段 1.3: 语义重构 (规则引擎优先)
    # ========================================================================
    
    def _smooth_with_llm(self, text: str, step_name: str = "规则引擎语义重构") -> Tuple[str, Dict[str, str]]:
        """
        【规则操作】受限语义重构（替代LLM）

        策略：
          - 优先使用规则引擎
          - 规则引擎失败时才回退到LLM
          - 记录LLM回退情况
        
        Returns:
            (处理后文本, 变更映射字典)
        """
        import difflib
        start_time = time.time()

        # 步骤1：尝试规则引擎
        normalized_text, changes = RuleBasedTextNormalizer.normalize(text)

        # 步骤2：如果规则引擎没有变化，且启用LLM，则回退到LLM
        if not changes and hasattr(self, 'llm'):
            debug(f"[规则引擎] 未能识别标准化模式，回退到LLM")
            result = self.llm.standardize_text(text)
            self.llm_calls_count += 1
            changes_desc = "LLM标准化"
        else:
            result = normalized_text
            changes_desc = f"规则引擎: {len(changes)}处变更"

        duration_ms = int((time.time() - start_time) * 1000)

        # 🔧 从原始文本和结果文本的差异中提取变更映射
        change_map = {}
        if result != text:
            matcher = difflib.SequenceMatcher(None, text, result)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    old_part = text[i1:i2]
                    new_part = result[j1:j2]
                    if len(old_part) <= 8 and len(new_part) <= 12:
                        change_map[old_part] = new_part
                elif tag == 'delete':
                    old_part = text[i1:i2]
                    if len(old_part) <= 8:
                        change_map[old_part] = ""
                elif tag == 'insert':
                    new_part = result[j1:j2]
                    if len(new_part) <= 12:
                        change_map[""] = new_part  # 插入的变更

        # 记录步骤快照
        self._add_step_snapshot(
            step_name=step_name,
            input_text=text,
            output_text=result,
            changes={"标准化": changes_desc},
            duration_ms=duration_ms,
            notes=[f"变更: {change}" for change in changes] if changes else []
        )

        self.processing_log.append(f"[OK] {step_name}完成 (耗时 {duration_ms}ms, 变更{len(changes)}处)")

        return result, change_map
    
    # ========================================================================
    # 阶段 1.4: 深层歧义检测 (规则引擎优先)
    # ========================================================================

    def _detect_structural_ambiguity(self, text: str) -> Optional[str]:
        """
        【规则操作】结构性歧义审计（替代LLM）

        检测代码无法识别的句法歧义
        经典案例: "咬死了猎人的狗" - 是狗死了还是猎人死了?

        返回: 如果存在歧义则返回描述,否则返回None
        """
        start_time = time.time()

        # 步骤1：尝试规则引擎
        ambiguity = SyntacticAmbiguityDetector.detect(text)

        # 步骤2：如果规则引擎未检测到歧义，且启用深度检查和LLM，则回退到LLM
        if not ambiguity and self.enable_deep_check and hasattr(self, 'llm'):
            debug(f"[规则引擎] 未能识别歧义模式，回退到LLM深度检查")
            ambiguity = self.llm.detect_ambiguity(text)
            self.llm_calls_count += 1
            detection_method = "LLM深度检查"
        else:
            detection_method = "规则引擎"

        duration_ms = int((time.time() - start_time) * 1000)

        if ambiguity:
            self.warnings.append(f"[WARN] 检测到句法歧义 ({detection_method}): {ambiguity}")
            self._add_step_snapshot(
                step_name="结构歧义检测",
                input_text=text,
                output_text=text,
                changes={"检测结果": f"歧义: {ambiguity}"},
                duration_ms=duration_ms,
                notes=[f"检测方法: {detection_method}", f"歧义详情: {ambiguity}"]
            )
        else:
            self.processing_log.append(f"[OK] 结构歧义检查: 通过 (耗时 {duration_ms}ms, 方法:{detection_method})")
            self._add_step_snapshot(
                step_name="结构歧义检测",
                input_text=text,
                output_text=text,
                changes={"检测结果": "通过"},
                duration_ms=duration_ms,
                notes=[f"检测方法: {detection_method}"]
            )

        return ambiguity if ambiguity else None
    
    # ========================================================================
    # 智能模式: 纯LLM端到端处理
    # ========================================================================
    
    def _smart_process(self, text: str) -> str:
        """
        【纯LLM模式】端到端智能处理
        
        适用场景:
          - 没有预定义词表
          - 需要处理全新领域
          - 对灵活性要求高于确定性
        """
        start_time = time.time()
        
        system_prompt = """你是一个高级提示词预处理系统。你的任务是将用户的原始输入标准化为清晰、无歧义的规范文本。

处理要求:
1. 术语标准化: 将口语化、非正式术语转换为技术标准术语
   - "套壳" → "基于API的封装应用"
   - "大模型" → "大型语言模型(LLM)"
   
2. 语法规范化: 修正口语化表达
   - 去除语气词("吧"、"嗯"、"那个")
   - 转换为书面语
   
3. 歧义消除: 如果发现严重歧义,在文本后添加 [AMBIGUITY: 描述]
   
4. 信息保真: 绝对不添加原文不存在的信息

输出格式: 纯文本(如有歧义则在末尾添加标记)

示例:
输入: "帮我搞一个RAG的大模型应用,chain要复杂一点"
输出: "开发一个基于检索增强生成(RAG)的大型语言模型应用,处理链(Chain)需要具备较高的复杂度"

输入: "这个需求没啥意思"
输出: "这个需求价值较低 [AMBIGUITY: '没意思'可能指:1.无趣 2.无意义 3.无商业价值]"
"""
        
        response = self.llm.call(system_prompt, text)
        result = response.content
        self.llm_calls_count += 1
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # 检查是否包含歧义标记
        ambiguity_note = None
        if "[AMBIGUITY:" in result:
            ambiguity_part = result.split("[AMBIGUITY:")[1].split("]")[0]
            self.warnings.append(f"[WARN] LLM检测到歧义: {ambiguity_part}")
            ambiguity_note = f"检测到歧义: {ambiguity_part}"
            result = result.split("[AMBIGUITY:")[0].strip()
        
        # 记录步骤快照
        notes = [ambiguity_note] if ambiguity_note else []
        self._add_step_snapshot(
            step_name="智能模式处理",
            input_text=text,
            output_text=result,
            changes={"模式": "端到端LLM处理"},
            duration_ms=duration_ms,
            notes=notes
        )
        
        self.processing_log.append(f"[OK] 智能模式处理完成 (耗时 {duration_ms}ms)")
        return result
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _add_step_snapshot(
        self,
        step_name: str,
        input_text: str,
        output_text: str,
        changes: Dict[str, str],
        duration_ms: int,
        notes: List[str] = None
    ):
        """添加步骤快照"""
        step = StepSnapshot(
            step_name=step_name,
            step_index=len(self.steps) + 1,
            input_text=input_text,
            output_text=output_text,
            changes=changes,
            duration_ms=duration_ms,
            notes=notes or []
        )
        self.steps.append(step)
    
    def _reset_state(self):
        """重置处理状态"""
        self.processing_log = []
        self.warnings = []
        self.steps = []
        self.llm_calls_count = 0
    
    # ========================================================================
    # 主处理流程
    # ========================================================================
    
    def process(self, raw_text: str, save_history: bool = True, show_comparison: bool = True) -> Prompt10Result:
        """
        主处理入口
        
        执行流程:
          Dictionary模式: 规则 → LLM润色 → 歧义检测
          Smart模式: 纯LLM端到端
          Hybrid模式: 词表优先 + LLM兜底
        
        Args:
            raw_text: 原始输入文本
            save_history: 是否保存处理历史
            show_comparison: 是否显示对比结果
            
        Returns:
            Prompt10Result: 处理结果对象
        """
        # 记录处理开始时间
        start_time = time.time()
        result_id = generate_id()
        timestamp = get_timestamp()
        
        # 重置状态
        self._reset_state()

        # ===== 输入类型检测 =====
        input_type = self.detect_input_type(raw_text)
        info(f"[输入类型检测] 判定为: {input_type.value}")

        info(f"\n{'='*60}")
        info(f"[开始处理] ID: {result_id} | 模式: {self.mode.value}")
        info(f"[原始输入] {raw_text}")
        info(f"{'='*60}\n")

        terminology_changes = {}
        ambiguity_detected = False
        ambiguity_details = None
        processed_text = raw_text
        status = ProcessingStatus.SUCCESS.value

        try:
            # ===== 针对结构化提示词文档的特殊处理 =====
            if input_type == InputType.STRUCTURED_PROMPT:
                info("[结构化提示词] 检测到结构化提示词文档，采用轻量化处理策略")

                # 步骤1: 仅对非结构化部分进行术语对齐
                processed_text, terminology_changes = self._normalize_terminology(processed_text)

                # 步骤2: 对Mustache变量和DSL占位符进行保护性处理
                protected_placeholders = self._protect_structured_placeholders(processed_text)

                # 步骤3: LLM仅对口语化内容进行标准化，结构化部分保持不变
                if hasattr(self, 'llm') and self.llm:
                    result = self.llm.standardize_text(processed_text)
                    processed_text = result
                    self.llm_calls_count += 1

                # 步骤4: 恢复被保护的占位符
                processed_text = self._restore_structured_placeholders(processed_text, protected_placeholders)

                self.processing_log.append(f"[结构化提示词] 完成轻量化处理")

            elif self.mode == ProcessingMode.SMART:
                # 纯智能模式
                processed_text = self._smart_process(raw_text)
                
            elif self.mode == ProcessingMode.HYBRID:
                # 混合模式：词表优先 + LLM兜底
                processed_text, terminology_changes = self._normalize_terminology(processed_text)
                
                # 记录术语对齐步骤
                if terminology_changes:
                    self._add_step_snapshot(
                        step_name="术语对齐",
                        input_text=raw_text,
                        output_text=processed_text,
                        changes=terminology_changes,
                        duration_ms=0
                    )
                
                # 边用边学：处理未知术语（混合规则学习器兜底）
                processed_text, learned_changes = self._learn_unknown_terminology(
                    processed_text, terminology_changes
                )
                if learned_changes:
                    terminology_changes.update(learned_changes)
                    info(f"[HYBRID] 学习到 {len(learned_changes)} 条新规则")
                
                # LLM 进一步优化
                processed_text, llm_changes = self._smooth_with_llm(processed_text, "混合模式LLM优化")
                if llm_changes:
                    terminology_changes.update(llm_changes)
                
            else:
                # 词表模式 (DICTIONARY)
                
                # 步骤1: 术语对齐 (代码层)
                step_start = time.time()
                processed_text, terminology_changes = self._normalize_terminology(processed_text)
                step_duration = int((time.time() - step_start) * 1000)
                
                if terminology_changes:
                    self._add_step_snapshot(
                        step_name="术语对齐",
                        input_text=raw_text,
                        output_text=processed_text,
                        changes=terminology_changes,
                        duration_ms=step_duration
                    )
                
                # 步骤1.5: 边用边学（可选，通过配置启用）
                # 默认在 DICTIONARY 模式也启用混合学习
                processed_text, learned_changes = self._learn_unknown_terminology(
                    processed_text, terminology_changes
                )
                if learned_changes:
                    terminology_changes.update(learned_changes)
                    info(f"[HYBRID] 学习到 {len(learned_changes)} 条新规则")
                
                # 步骤2: 硬性歧义阻断 (代码层)
                has_ambiguity, ambiguous_words = self._check_heuristic_ambiguity(processed_text)
                
                if has_ambiguity:
                    ambiguity_detected = True
                    ambiguity_details = f"检测到歧义词: {ambiguous_words}"
                    
                    # 即使检测到歧义词，也先进行LLM修复
                    info("[INFO] 检测到歧义词，但将继续进行LLM修复以展示修复结果...")
                    processed_text, _ = self._smooth_with_llm(processed_text, "歧义词场景LLM修复")
                    self.processing_log.append(f"[WARN] 检测到歧义词 {ambiguous_words}，但已生成修复版本")
                    status = ProcessingStatus.AMBIGUITY.value
                else:
                    # 步骤3: LLM语义重构 (LLM层)
                    processed_text, _ = self._smooth_with_llm(processed_text)
                    
                    # 步骤4: 深层歧义检测 (LLM层)
                    if self.enable_deep_check:
                        structural_ambiguity = self._detect_structural_ambiguity(processed_text)
                        if structural_ambiguity:
                            ambiguity_detected = True
                            ambiguity_details = structural_ambiguity
                            status = ProcessingStatus.AMBIGUITY.value
                            self.processing_log.append(f"[WARN] 检测到句法歧义: {structural_ambiguity}")
            
            if status == ProcessingStatus.SUCCESS.value:
                info("\n[处理完成] [OK]")
            else:
                warning(f"\n[处理完成] [有歧义] {ambiguity_details}")
            
        except Exception as e:
            error(f"\n[处理失败] [ERROR] {e}")
            status = ProcessingStatus.ERROR.value
            ambiguity_details = str(e)
        
        # 计算总处理时间
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # 构建结果对象
        result = Prompt10Result(
            id=result_id,
            timestamp=timestamp,
            mode=self.mode.value,
            original_text=raw_text,
            processed_text=processed_text,
            steps=self.steps.copy(),
            terminology_changes=terminology_changes,
            status=status,
            ambiguity_detected=ambiguity_detected,
            ambiguity_details=ambiguity_details,
            steps_log=self.processing_log.copy(),
            warnings=self.warnings.copy(),
            processing_time_ms=processing_time_ms,
            llm_calls_count=self.llm_calls_count
        )
        
        # 打印处理日志
        self._print_result(result)
        
        # 保存历史记录
        if save_history:
            self._save_processing_history_v2(result)
            
            # 显示对比
            if show_comparison:
                self._print_comparison(result)
        
        return result
    
    def _save_processing_history_v2(self, result: Prompt10Result):
        """保存处理历史（使用新数据结构，包含优化统计）"""
        # 收集优化统计信息
        rule_stats = {
            "llm_calls": result.llm_calls_count,
            "normalization_changes": len(result.terminology_changes) if result.terminology_changes else 0,
            "ambiguity_detected": result.ambiguity_detected,
            "processing_mode": result.mode,
            "has_llm_fallback": result.llm_calls_count > 0
        }

        history = ProcessingHistory(
            timestamp=result.timestamp,
            original_text=result.original_text,
            processed_text=result.processed_text,
            mode=result.mode,
            steps_log=result.steps_log,
            warnings=result.warnings,
            terminology_changes=result.terminology_changes,
            ambiguity_detected=result.ambiguity_detected,
            success=result.is_success(),
            processing_time_ms=result.processing_time_ms,
            rule_engine_stats=rule_stats  # 新增：优化统计
        )

        try:
            self.history_manager.save_history(history)
        except Exception as e:
            warning(f"保存处理历史失败: {e}")
    
    def _print_result(self, result: Prompt10Result):
        """打印处理结果"""
        info("\n" + "─"*60)
        info(f"处理ID: {result.id} | 状态: {result.status}")
        info("─"*60)
        
        # 打印中间步骤
        if result.steps:
            info("\n【处理步骤】:")
            for step in result.steps:
                info(f"  {step.step_index}. {step.step_name} (耗时 {step.duration_ms}ms)")
                if step.changes:
                    for key, val in step.changes.items():
                        info(f"     └─ {key}: {val}")
        
        # 打印日志
        if result.steps_log:
            info("\n【处理日志】:")
            for log in result.steps_log:
                info(f"  {log}")
        
        # 打印警告
        if result.warnings:
            info("\n【警告信息】:")
            for warn_msg in result.warnings:
                warning(f"  {warn_msg}")
        
        # 打印术语替换
        if result.terminology_changes:
            info("\n【术语替换】:")
            for old, new in result.terminology_changes.items():
                info(f"  {old} → {new}")
        
        info("\n" + "─"*60)
        info(f"【最终输出】\n{result.processed_text}")
        info("="*60 + "\n")
    
    def _print_comparison(self, result: Prompt10Result):
        """打印详细对比"""
        info("\n" + "="*60)
        info("【处理对比】")
        info("="*60)
        
        info(f"\n📄 原始文本:")
        info(f"   {result.original_text}")
        
        info(f"\n✨ 处理后文本:")
        info(f"   {result.processed_text}")
        
        # 显示每个步骤的变化
        if result.steps:
            info(f"\n📊 处理步骤详情:")
            for step in result.steps:
                info(f"\n   步骤 {step.step_index}: {step.step_name}")
                info(f"   ├─ 输入: {step.input_text[:50]}{'...' if len(step.input_text) > 50 else ''}")
                info(f"   ├─ 输出: {step.output_text[:50]}{'...' if len(step.output_text) > 50 else ''}")
                info(f"   └─ 耗时: {step.duration_ms}ms")
        
        info(f"\n⏱️ 总耗时: {result.processing_time_ms}ms | LLM调用: {result.llm_calls_count}次")
        info("="*60 + "\n")


# ============================================================================
# 配置加载器
# ============================================================================

class ConfigLoader:
    """配置文件加载器"""
    
    @staticmethod
    def load_term_mapping(file_path: str) -> Dict[str, str]:
        """加载术语映射表 (JSON格式)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            warning(f"配置文件不存在: {file_path},使用默认配置")
            return {}
    
    @staticmethod
    def load_blacklist(file_path: str) -> List[str]:
        """加载歧义词黑名单 (每行一个词)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            warning(f"配置文件不存在: {file_path},使用默认配置")
            return []
    
    @staticmethod
    def get_default_term_mapping() -> Dict[str, str]:
        """默认术语映射表"""
        return {
            "套壳": "基于API的封装应用",
            "大模型": "大型语言模型(LLM)",
            "chain": "处理链(Chain)",
            "RAG": "检索增强生成(RAG)",
            "幻觉": "模型幻觉(Hallucination)",
            "微调": "模型微调(Fine-tuning)",
            "Agent": "智能代理(Agent)",
            "Prompt": "提示词(Prompt)",
            "CoT": "思维链(Chain-of-Thought)"
        }
    
    @staticmethod
    def get_default_blacklist() -> List[str]:
        """默认歧义词黑名单"""
        return [
            "意思", "那个", "随便", "搞一下", "弄一下", "弄好",
            "稍微", "简单点", "复杂点", "看着办",
            "差不多", "大概", "可能"
        ]


# ============================================================================
# 复杂示例（约200字，用于验证术语密集 + 歧义阻断）
# ============================================================================

COMPLEX_EXAMPLE_TERMINOLOGY = (
    "我们想做一个基于大模型的智能应用，用RAG做检索增强，配合Agent做决策。"
    "链式处理chain要设计得完整一点，避免出现幻觉。可以考虑用CoT做推理，Prompt要写清楚。"
    "如果效果不好再考虑微调，不要搞成那种套壳的玩意儿。"
    "另外检索模块要能处理多轮对话，保持上下文一致。"
    "整体希望检索准、生成稳，尽量少出现幻觉，必要时再上微调优化效果。"
)

COMPLEX_EXAMPLE_AMBIGUITY = (
    "这个需求你看着办吧，我们大概想要一个RAG加大模型的应用，chain要复杂点，别搞一下那种简单的。"
    "意思就是能检索、能生成就行，稍微像样一点，差不多满足业务就成。"
    "具体你随便弄一下，弄好就成。不需要搞太复杂点，可能加个Agent、用点RAG就差不多了。"
    "你那边先弄好一版，我们再看效果，那个具体的实现细节你们随便定就行。"
)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # ------------------------------------------------------------------------
    # 场景1: 词表模式 (Dictionary Mode) - 最高确定性
    # ------------------------------------------------------------------------
    info("\n" + "█"*60)
    info("█  场景1: 词表模式 (基于预定义规则)")
    info("█"*60)
    
    processor_dict = PromptPreprocessor(
        mode=ProcessingMode.DICTIONARY,
        term_mapping=ConfigLoader.get_default_term_mapping(),
        ambiguity_blacklist=ConfigLoader.get_default_blacklist(),
        enable_deep_check=True
    )
    
    test_case_1 = "帮我搞一个RAG的大模型应用,chain要复杂一点。"
    
    try:
        result1 = processor_dict.process(test_case_1)
    except ValueError as e:
        error(f"处理失败: {e}")
    
    
    # ------------------------------------------------------------------------
    # 场景2: 歧义词触发中断
    # ------------------------------------------------------------------------
    info("\n" + "█"*60)
    info("█  场景2: 歧义词检测 (触发中断)")
    info("█"*60)
    
    test_case_2 = "这个需求没啥意思,你看着办。"
    
    try:
        result2 = processor_dict.process(test_case_2)
    except ValueError as e:
        info("[OK] 成功拦截歧义输入")
    
    
    # ------------------------------------------------------------------------
    # 场景3: 智能模式 (Smart Mode) - 无需配置词表
    # ------------------------------------------------------------------------
    info("\n" + "█"*60)
    info("█  场景3: 智能模式 (纯LLM驱动)")
    info("█"*60)
    
    processor_smart = PromptPreprocessor(
        mode=ProcessingMode.SMART,
        enable_deep_check=False  # 智能模式已内置歧义检测
    )
    
    test_case_3 = "我想做个类似ChatGPT的东西,但要能联网搜索,还要能记住上下文。"
    
    try:
        result3 = processor_smart.process(test_case_3)
    except ValueError as e:
        error(f"处理失败: {e}")
    
    
    # ------------------------------------------------------------------------
    # 场景4: 从配置文件加载 (生产环境推荐)
    # ------------------------------------------------------------------------
    info("\n" + "█"*60)
    info("█  场景4: 配置文件加载示例")
    info("█"*60)
    
    # 示例: 如何保存配置到文件
    # import json
    # with open('term_mapping.json', 'w', encoding='utf-8') as f:
    #     json.dump(ConfigLoader.get_default_term_mapping(), f, 
    #               ensure_ascii=False, indent=2)
    
    # with open('ambiguity_blacklist.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(ConfigLoader.get_default_blacklist()))
    
    # 加载配置
    # term_map = ConfigLoader.load_term_mapping('term_mapping.json')
    # blacklist = ConfigLoader.load_blacklist('ambiguity_blacklist.txt')
    
    info("""
配置文件格式:

1. term_mapping.json (术语映射表):
{
  "套壳": "基于API的封装应用",
  "大模型": "大型语言模型(LLM)"
}

2. ambiguity_blacklist.txt (歧义词黑名单,每行一个):
意思
那个
随便
搞一下
    """)
    
    
    # ------------------------------------------------------------------------
    # 场景5: 复杂示例 - 术语密集（约200字，无歧义词，应完整通过）
    # ------------------------------------------------------------------------
    info("\n" + "█"*60)
    info("█  场景5: 复杂示例 - 术语密集（约200字）")
    info("█"*60)
    
    try:
        result5 = processor_dict.process(COMPLEX_EXAMPLE_TERMINOLOGY)
        info(f"  [统计] 术语替换 {len(result5.terminology_changes)} 类，"
              f"步骤数 {len(result5.steps_log)}，警告数 {len(result5.warnings)}")
    except ValueError as e:
        error(f"  处理失败: {e}")
    
    
    # ------------------------------------------------------------------------
    # 场景6: 复杂示例 - 歧义触发（约200字，含多处黑名单词，应拦截）
    # 说明：此场景预期「处理失败」— 流程在歧义检测阶段中断，即成功拦截。
    # ------------------------------------------------------------------------
    info("\n" + "█"*60)
    info("█  场景6: 复杂示例 - 歧义触发（约200字）")
    info("█"*60)
    
    try:
        result6 = processor_dict.process(COMPLEX_EXAMPLE_AMBIGUITY)
        warning("  [未预期] 应被歧义拦截却通过了")
    except ValueError as e:
        info("  [OK] 成功拦截歧义输入（复杂长文本）")
        try:
            detail = str(e).replace("\n", " ").strip()
            info(f"  [详情] {detail[:200]}{'...' if len(detail) > 200 else ''}")
        except Exception:
            info("  [详情] 已拦截，含黑名单歧义词。")
    
    
    info("\n" + "█"*60)
    info("█  系统演示完成")
    info("█"*60)
    info("""
使用建议:

1. 词表模式: 适用于术语固定的专业领域(法律、医疗、金融)
   - 优势: 100%准确,零Token消耗(术语对齐阶段)
   - 劣势: 需要维护词表

2. 智能模式: 适用于通用场景或新兴领域
   - 优势: 无需配置,适应性强
   - 劣势: 完全依赖LLM,成本较高

3. 生产环境推荐: 混合模式
   - 核心术语用词表强制对齐
   - 未覆盖的部分交给LLM智能处理

4. 历史记录功能:
   - 每次处理都会自动保存到 processing_history/history.json
   - 使用 history_manager 可以查看和对比历史记录
   - 支持导出HTML格式的对比报告
""")
    
    
    # ------------------------------------------------------------------------
    # 场景7: 查看处理历史记录
    # ------------------------------------------------------------------------
    info("\n" + "█"*60)
    info("█  场景7: 查看处理历史记录")
    info("█"*60)
    
    history_manager = HistoryManager()
    recent_history = history_manager.get_recent_history(limit=5)
    
    if recent_history:
        info(f"\n找到 {len(recent_history)} 条最近的处理记录:")
        for i, hist in enumerate(recent_history, 1):
            info(f"\n记录 {i}:")
            info(f"  时间: {hist.timestamp}")
            info(f"  模式: {hist.mode}")
            info(f"  状态: {'✅ 成功' if hist.success else '⚠️ 检测到歧义'}")
            info(f"  原始文本: {hist.original_text[:50]}...")
            info(f"  处理后文本: {hist.processed_text[:50]}...")
        
        # 展示最新一条记录的详细对比
        if recent_history:
            info("\n" + "="*60)
            info("最新记录的详细对比:")
            info("="*60)
            history_manager.print_comparison(recent_history[0])
    else:
        info("暂无处理历史记录")



# -----------------------------------------------------------------------------
# 生产环境：建议通过环境变量配置 RCOUYI_BASE_URL / RCOUYI_API_KEY，勿提交 key。
# 依赖: pip install openai
# -----------------------------------------------------------------------------



