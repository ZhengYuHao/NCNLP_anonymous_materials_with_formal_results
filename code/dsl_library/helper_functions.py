"""
辅助函数实现
提供DSL函数实现库中需要的辅助函数

这些函数会被模板引用，需要在生成的代码中可用
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def _get_top_n(results: List[Any], n: int) -> List[Any]:
    """获取前N个结果"""
    if not results:
        return []
    return results[:n]


def _limit_results(results: List[Any], n: int) -> List[Any]:
    """限制结果数量"""
    return results[:n] if results else []


def _deduplicate_results(results: List[Dict], key: str = 'id') -> List[Dict]:
    """结果去重"""
    seen = set()
    deduplicated = []
    for item in results:
        item_key = item.get(key)
        if item_key and item_key not in seen:
            seen.add(item_key)
            deduplicated.append(item)
    return deduplicated


def _sort_results(results: List[Dict], by: str, reverse: bool = True) -> List[Dict]:
    """结果排序"""
    if not results:
        return []
    return sorted(results, key=lambda x: x.get(by, 0), reverse=reverse)


def _compare_version(current: str, target: str) -> bool:
    """比较版本号"""
    try:
        current_parts = [int(x) for x in current.split('.')]
        target_parts = [int(x) for x in target.split('.')]
        
        for c, t in zip(current_parts, target_parts):
            if c > t:
                return True
            elif c < t:
                return False
        return len(current_parts) >= len(target_parts)
    except:
        return False


def _detect_query_type(query: str) -> str:
    """检测查询类型"""
    code_keywords = ['代码', '函数', 'class', 'def ', 'import ', 'python', 'java']
    for keyword in code_keywords:
        if keyword.lower() in query.lower():
            return 'code'
    
    complex_keywords = ['为什么', '分析', '比较', '推理', '计算', '原因']
    for keyword in complex_keywords:
        if keyword in query:
            return 'complex'
    
    return 'simple'


def _filter_sensitive_content(content: str) -> bool:
    """敏感词过滤，返回是否通过"""
    sensitive_words = ['政治', '暴力', '色情']
    for word in sensitive_words:
        if word in content:
            logger.warning(f"检测到敏感词: {word}")
            return False
    return True


def _validate_input(input_data: Any, rules: Dict[str, Any]) -> bool:
    """输入验证"""
    if not input_data:
        return False
    
    for rule_name, rule_value in rules.items():
        if rule_name == 'max_length' and len(str(input_data)) > rule_value:
            return False
        if rule_name == 'min_length' and len(str(input_data)) < rule_value:
            return False
    
    return True


def _check_user_permission(user_id: str, resource: str) -> bool:
    """权限检查（模拟）"""
    return True


def _check_rate_limit(user_id: str, limit: int = 50) -> bool:
    """限流检查（模拟）"""
    return True


def _switch_retrieval_mode(from_mode: str, to_mode: str):
    """切换检索模式"""
    logger.info(f"切换检索模式: {from_mode} -> {to_mode}")


def _escalate_to_human_queue(query: str, reason: str):
    """升级到人工队列"""
    logger.warning(f"升级到人工: query={query}, reason={reason}")


def _trigger_stronger_model_regeneration(query: str):
    """触发更强模型重新生成"""
    logger.info(f"触发更强模型重新生成: {query}")


def _check_user_dissatisfaction_count(user_id: str) -> int:
    """检查用户不满次数（模拟）"""
    return 0