"""
统一的日志系统模块
提供带时间戳和行号的日志功能
"""

import logging
import sys
from typing import Optional


class LineNumberFormatter(logging.Formatter):
    """自定义格式化器，显示文件名和行号"""
    
    def format(self, record):
        # 获取调用日志的代码位置信息
        if record.pathname:
            # 只显示文件名，不显示完整路径
            filename = record.pathname.split('/')[-1]
            record.filename = filename
        else:
            record.filename = record.module
        
        # 格式化日志消息
        return super().format(record)


def setup_logger(
    name: str = "prompt3.0",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置并返回配置好的日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别 (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径（可选，如果提供则同时写入文件）
        format_string: 自定义格式字符串（可选）
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 默认格式：时间戳 | 级别 | 文件名:行号 | 函数名 | 消息
    if format_string is None:
        format_string = (
            '%(asctime)s | %(levelname)-8s | '
            '%(filename)s:%(lineno)d | %(funcName)s() | %(message)s'
        )
    
    formatter = LineNumberFormatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 创建默认的日志记录器实例
default_logger = setup_logger()

# 导出便捷函数
def debug(msg: str, *args, **kwargs):
    """记录DEBUG级别日志"""
    # 使用stacklevel=2来跳过当前函数，显示调用者的位置
    default_logger.debug(msg, *args, stacklevel=2, **kwargs)


def info(msg: str, *args, **kwargs):
    """记录INFO级别日志"""
    # 使用stacklevel=2来跳过当前函数，显示调用者的位置
    default_logger.info(msg, *args, stacklevel=2, **kwargs)


def warning(msg: str, *args, **kwargs):
    """记录WARNING级别日志"""
    # 使用stacklevel=2来跳过当前函数，显示调用者的位置
    default_logger.warning(msg, *args, stacklevel=2, **kwargs)


def error(msg: str, *args, **kwargs):
    """记录ERROR级别日志"""
    # 使用stacklevel=2来跳过当前函数，显示调用者的位置
    default_logger.error(msg, *args, stacklevel=2, **kwargs)


def critical(msg: str, *args, **kwargs):
    """记录CRITICAL级别日志"""
    # 使用stacklevel=2来跳过当前函数，显示调用者的位置
    default_logger.critical(msg, *args, stacklevel=2, **kwargs)


def exception(msg: str, *args, exc_info=True, **kwargs):
    """记录异常信息（ERROR级别）"""
    # 使用stacklevel=2来跳过当前函数，显示调用者的位置
    default_logger.exception(msg, *args, exc_info=exc_info, stacklevel=2, **kwargs)
