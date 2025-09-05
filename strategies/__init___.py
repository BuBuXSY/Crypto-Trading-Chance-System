# strategies/__init__.py
"""
策略模块
包含所有交易策略和分析方法
"""

from .signal_generator import SignalGenerator
from .whale_tracker import EnhancedWhaleTracker
from .arbitrage_scanner import EnhancedArbitrageScanner
from .anomaly_detector import EnhancedAnomalyDetector
from .market_regime import MarketRegimeDetector
from .signal_tracker import SignalTracker
from .self_learning import SelfLearningSystem

__all__ = [
    'SignalGenerator',
    'EnhancedWhaleTracker',
    'EnhancedArbitrageScanner', 
    'EnhancedAnomalyDetector',
    'MarketRegimeDetector',
    'SignalTracker',
    'SelfLearningSystem'
]

# 策略版本信息
STRATEGY_VERSION = "1.0.0"
SUPPORTED_EXCHANGES = ["okx", "binance", "bybit", "gateio"]