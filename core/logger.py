# core/logger.py
"""
日志系统模块
提供彩色日志输出和文件记录功能
"""

import logging
from logging.handlers import RotatingFileHandler
import os

class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

def get_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    获取配置好的logger实例
    
    Args:
        name: logger名称
        log_file: 日志文件路径
        level: 日志级别
    
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 控制台处理器 - 带颜色
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        ColoredFormatter('%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
    )
    logger.addHandler(console_handler)
    
    # 文件处理器 - 如果提供了文件路径
    if log_file:
        # 确保目录存在
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
        )
        logger.addHandler(file_handler)
    
    return logger

# 创建默认logger
default_logger = get_logger(__name__)