# indicators/__init__.py
"""
技术指标模块
包含所有技术分析指标
"""

from .technical import TechnicalIndicators
from .advanced import AdvancedIndicators

__all__ = [
    'TechnicalIndicators',
    'AdvancedIndicators'
]

# ============================================================================

# ai/__init__.py
"""
AI模块
包含机器学习、深度学习和模式识别功能
"""

from .predictor import UltraEnhancedAIPredictor
from .pattern_recognition import PatternRecognition
from .sentiment import MarketSentimentAnalyzer
from .uncertainty import UncertaintyEstimator

# 检查可选依赖
try:
    from .adaptive_learning import AdaptiveLearningSystem
    ADAPTIVE_LEARNING_AVAILABLE = True
except ImportError:
    ADAPTIVE_LEARNING_AVAILABLE = False

__all__ = [
    'UltraEnhancedAIPredictor',
    'PatternRecognition', 
    'MarketSentimentAnalyzer',
    'UncertaintyEstimator'
]

if ADAPTIVE_LEARNING_AVAILABLE:
    __all__.append('AdaptiveLearningSystem')

# AI模块状态检查
def check_ai_capabilities():
    """检查AI功能可用性"""
    capabilities = {
        "basic_ml": False,
        "deep_learning": False,
        "advanced_ml": False,
        "pattern_recognition": True,
        "sentiment_analysis": True
    }
    
    try:
        import sklearn
        capabilities["basic_ml"] = True
    except ImportError:
        pass
    
    try:
        import tensorflow
        capabilities["deep_learning"] = True
    except ImportError:
        pass
    
    try:
        import xgboost
        capabilities["advanced_ml"] = True
    except ImportError:
        pass
    
    return capabilities
