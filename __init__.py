# 根目录 __init__.py (项目根目录)
"""
小虞沁XYQ终极交易系统 V1 增强版

Enhanced Trading System with Advanced AI Features
"""

# 版本信息
__version__ = "1.0.0"
__title__ = "小虞沁XYQ终极交易系统 V1 增强版"
__author__ = "小虞沁XYQ"
__description__ = "Advanced AI-powered cryptocurrency trading system"

# 主要组件导入
from config import config
from core import get_logger, DatabaseManager, ExchangeManager
from ai import UltraEnhancedAIPredictor, PatternRecognition
from strategies import SignalGenerator, EnhancedWhaleTracker
from utils import NotificationManager
from data import Signal, MarketRegime

# 主要类导出
__all__ = [
    # 配置
    'config',
    
    # 核心组件
    'get_logger',
    'DatabaseManager',
    'ExchangeManager',
    
    # AI组件
    'UltraEnhancedAIPredictor',
    'PatternRecognition',
    
    # 策略组件
    'SignalGenerator',
    'EnhancedWhaleTracker',
    
    # 工具组件
    'NotificationManager',
    
    # 数据模型
    'Signal',
    'MarketRegime'
]

# 系统功能检查
def check_system_capabilities():
    """检查整个系统的功能可用性"""
    from ai import check_ai_capabilities
    from utils import check_notification_capabilities
    from analysis import check_analysis_capabilities
    
    return {
        "ai": check_ai_capabilities(),
        "notifications": check_notification_capabilities(), 
        "analysis": check_analysis_capabilities(),
        "version": __version__,
        "core_features": True
    }

# 快速启动函数
def quick_start(config_overrides=None):
    """
    快速启动交易系统
    
    Args:
        config_overrides: 配置覆盖参数
    
    Returns:
        配置好的交易机器人实例
    """
    from main import UltraXYQTradingBot
    
    # 应用配置覆盖
    if config_overrides:
        config.update(config_overrides)
    
    # 创建机器人实例
    bot = UltraXYQTradingBot()
    
    return bot

# 系统信息
def get_system_info():
    """获取系统信息"""
    import platform
    import sys
    
    return {
        "system_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "capabilities": check_system_capabilities(),
        "supported_exchanges": ["okx", "binance", "bybit", "gateio"],
        "ai_features": [
            "ensemble_learning",
            "deep_learning", 
            "pattern_recognition",
            "sentiment_analysis",
            "uncertainty_estimation",
            "adaptive_learning"
        ]
    }

# 欢迎信息
def print_welcome():
    """打印欢迎信息"""
    print("="*60)
    print(f"  {__title__}")
    print(f"  版本: {__version__}")
    print(f"  作者: {__author__}")
    print("="*60)
    print("  💙 献给每一位勇敢的朋友")
    print("  🌈 您的梦想永远在这里闪光") 
    print("  🚀 Version1 - 更智能，更精准，更强大")
    print("="*60)
    
    # 显示功能状态
    capabilities = check_system_capabilities()
    print("\n  功能状态检查:")
    
    if capabilities["ai"]["basic_ml"]:
        print("  ✅ 基础机器学习 - 可用")
    else:
        print("  ❌ 基础机器学习 - 不可用 (需要sklearn)")
    
    if capabilities["ai"]["deep_learning"]:
        print("  ✅ 深度学习 - 可用")
    else:
        print("  ⚠️  深度学习 - 不可用 (需要tensorflow)")
    
    if capabilities["notifications"]["telegram"]:
        print("  ✅ Telegram通知 - 已配置")
    else:
        print("  ⚠️  Telegram通知 - 未配置")
    
    print("="*60)

# 模块级初始化
if __name__ != "__main__":
    # 当模块被导入时，检查基本依赖
    try:
        import pandas
        import numpy
        import ccxt
        import requests
    except ImportError as e:
        print(f"❌ 缺少必要依赖: {e}")
        print("请运行: pip install -r requirements.txt")