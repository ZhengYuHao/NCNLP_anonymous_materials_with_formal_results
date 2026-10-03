"""
增强版WaAct编译器
在原有编译器基础上，支持从Operation获取函数实现代码

核心改进：
1. 接受已填充implementation的Operation列表
2. 在翻译CALL时优先使用实现代码
3. 收集必要的导入语句
"""

from typing import List, Dict, Tuple, Any, Optional
import logging

from prompt_codegenetate import WaActCompiler, ModuleDefinition, PseudoCodeParser, ModuleSynthesizer
from dsl_library import FUNCTION_REGISTRY, render_implementation, has_implementation

logger = logging.getLogger(__name__)


class EnhancedWaActCompiler(WaActCompiler):
    """
    增强版WaAct编译器
    支持函数实现填充
    """
    
    def __init__(self):
        super().__init__()
        self._operation_impl_map: Dict[str, str] = {}
        self._required_imports: List[str] = []
        
        self.parser = EnhancedPseudoCodeParser(self)
    
    def set_operations_impl(self, operations: List) -> 'EnhancedWaActCompiler':
        """
        设置操作的实现代码映射
        
        Args:
            operations: 操作列表，每个操作包含function和implementation字段
            
        Returns:
            self（支持链式调用）
        """
        self._operation_impl_map.clear()
        self._required_imports.clear()
        
        for op in operations:
            if hasattr(op, 'function') and hasattr(op, 'implementation') and op.implementation:
                func_name = op.function
                impl = op.implementation
                
                self._operation_impl_map[func_name] = impl
                
                if hasattr(op, 'impl_required_imports') and op.impl_required_imports:
                    for imp in op.impl_required_imports:
                        if imp not in self._required_imports:
                            self._required_imports.append(imp)
                
                logger.info(f"[EnhancedCompiler] 注册函数实现: {func_name}")
        
        logger.info(f"[EnhancedCompiler] 已注册 {len(self._operation_impl_map)} 个函数实现")
        return self
    
    def get_implementation(self, func_name: str) -> Optional[str]:
        """获取函数的实现代码"""
        return self._operation_impl_map.get(func_name)
    
    def get_all_imports(self) -> List[str]:
        """获取所有需要的导入语句"""
        return self._required_imports.copy()
    
    def has_implementation(self, func_name: str) -> bool:
        """检查函数是否有实现"""
        return func_name in self._operation_impl_map

    def compile(self, dsl_code: str,
                clustering_strategy: str = "hybrid",
                visualize: bool = False):
        """重写compile方法，使用增强的parser"""
        from prompt_codegenetate import ModuleSynthesizer

        self.parser = EnhancedPseudoCodeParser(self)
        self.synthesizer = ModuleSynthesizer()

        return super().compile(dsl_code, clustering_strategy, visualize)

    compile.__doc__ = WaActCompiler.compile.__doc__


class EnhancedPseudoCodeParser(PseudoCodeParser):
    """
    增强版伪代码解析器
    支持使用函数实现代码而非invoke_function
    """
    
    def __init__(self, compiler: EnhancedWaActCompiler):
        super().__init__()
        self.compiler = compiler
    
    def _translate_call(self, pseudo_line: str) -> str:
        """
        将伪代码的CALL转换为Python代码（优先使用实现代码）

        改进逻辑：
        1. 提取函数名
        2. 检查是否有注册的实现（优先查注册表，其次查本地map）
        3. 有实现 → 使用实现代码
        4. 无实现 → 回退到invoke_function
        """
        import re

        cleaned = pseudo_line.replace("{{", "").replace("}}", "")

        func_name = self._extract_function_name(cleaned)
        if not func_name:
            return super()._translate_call(pseudo_line)

        impl = None
        if has_implementation(func_name):
            impl = render_implementation(func_name, {})
        elif self.compiler.get_implementation(func_name):
            impl = self.compiler.get_implementation(func_name)

        if impl:
            result_var = self._extract_result_var(cleaned)
            if result_var:
                return f"{result_var} = {impl}"
            else:
                return impl

        return super()._translate_call(pseudo_line)
    
    def _extract_function_name(self, line: str) -> Optional[str]:
        """提取函数名"""
        import re
        
        patterns = [
            r'(\w+)\s*=\s*CALL\s+(\w+)\s*\(',
            r'(\w+)\s*=\s*CALL\s+(\w+)\s*$',
            r'CALL\s+(\w+)\s*\(',
            r'CALL\s+(\w+)\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                groups = match.groups()
                if len(groups) == 2 and match.group(1) not in ['CALL', 'IF']:
                    return groups[1]
                elif len(groups) == 1:
                    return groups[0]
        
        return None
    
    def _extract_result_var(self, line: str) -> Optional[str]:
        """提取结果变量名"""
        import re
        
        match = re.match(r'(\w+)\s*=\s*CALL', line)
        if match:
            return match.group(1)
        
        return None


def compile_with_implementation(
    dsl_code: str,
    operations: List,
    clustering_strategy: str = "hybrid",
    visualize: bool = False
) -> Tuple[List[ModuleDefinition], str, Dict[str, Any]]:
    """
    编译DSL代码（带函数实现）
    
    Args:
        dsl_code: DSL伪代码
        operations: 操作列表（包含implementation字段）
        clustering_strategy: 聚类策略
        visualize: 是否可视化
        
    Returns:
        (模块列表, 主函数代码, 编译详情)
    """
    compiler = EnhancedWaActCompiler()
    compiler.set_operations_impl(operations)
    
    modules, main_code, details = compiler.compile(
        dsl_code,
        clustering_strategy=clustering_strategy,
        visualize=visualize
    )
    
    details['implementation_info'] = {
        'registered_count': len(compiler._operation_impl_map),
        'required_imports': compiler.get_all_imports(),
        'using_impl_fallback': False
    }
    
    return modules, main_code, details


def compile_with_fallback(
    dsl_code: str,
    operations: List,
    clustering_strategy: str = "hybrid"
) -> Tuple[List[ModuleDefinition], str, Dict[str, Any]]:
    """
    编译DSL代码（带回退机制）
    
    有实现 → 使用实现代码
    无实现 → 使用invoke_function
    
    Args:
        dsl_code: DSL伪代码
        operations: 操作列表
        clustering_strategy: 聚类策略
        
    Returns:
        (模块列表, 主函数代码, 编译详情)
    """
    compiler = EnhancedWaActCompiler()
    compiler.set_operations_impl(operations)
    
    modules, main_code, details = compiler.compile(
        dsl_code,
        clustering_strategy=clustering_strategy,
        visualize=False
    )
    
    impl_count = len(compiler._operation_impl_map)
    fallback_count = len(operations) - impl_count
    
    details['implementation_info'] = {
        'registered_count': impl_count,
        'fallback_count': fallback_count,
        'fallback_rate': fallback_count / len(operations) if operations else 0,
        'required_imports': compiler.get_all_imports(),
        'using_impl_fallback': fallback_count > 0
    }
    
    logger.info(f"[EnhancedCompile] 实现填充: {impl_count}/{len(operations)}, "
                f"兜底: {fallback_count}/{len(operations)}")
    
    return modules, main_code, details