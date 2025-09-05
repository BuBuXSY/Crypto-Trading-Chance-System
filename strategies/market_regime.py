# strategies/market_regime.py
"""
市场状态检测模块
分析市场趋势、波动率和所处阶段
"""

import pandas as pd
import numpy as np
from typing import Optional
from core.logger import get_logger
from core.utils import safe_div, now_local
from data.models import MarketRegime

logger = get_logger(__name__)

class MarketRegimeDetector:
    """市场状态检测器"""
    
    def __init__(self, config: dict):
        logger.info("初始化市场状态检测器")
        self.config = config
        self.current_regime = None
        self.regime_history = []
    
    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        检测市场状态
        
        Args:
            df: K线数据DataFrame
        
        Returns:
            市场状态对象
        """
        logger.debug("分析市场状态")
        
        if len(df) < self.config["market_regime"]["lookback_periods"]:
            return MarketRegime(
                timestamp=now_local(),
                regime_type="UNKNOWN",
                trend_direction="UNKNOWN",
                volatility_level="UNKNOWN",
                market_phase="UNKNOWN",
                confidence=0.0
            )
        
        close = df['close'].values
        returns = df['close'].pct_change().dropna()
        
        # 趋势分析
        trend_direction, trend_strength = self.analyze_trend(df)
        
        # 确定市场类型
        if trend_strength > 0.05:
            regime_type = "TRENDING"
        else:
            regime_type = "RANGING"
        
        # 波动率分析
        volatility_level = self.analyze_volatility(returns)
        
        # 市场阶段分析
        market_phase = self.determine_market_phase(df, trend_direction)
        
        # 计算置信度
        confidence = self.calculate_confidence(trend_strength, volatility_level)
        
        regime = MarketRegime(
            timestamp=now_local(),
            regime_type=regime_type,
            trend_direction=trend_direction,
            volatility_level=volatility_level,
            market_phase=market_phase,
            confidence=confidence
        )
        
        self.current_regime = regime
        self.regime_history.append(regime)
        
        # 限制历史记录数量
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        logger.info(f"市场状态: {regime_type}/{trend_direction} 波动: {volatility_level}")
        return regime
    
    def analyze_trend(self, df: pd.DataFrame) -> tuple:
        """
        分析趋势方向和强度
        
        Args:
            df: K线数据
        
        Returns:
            (趋势方向, 趋势强度)
        """
        close = df['close']
        
        # 短期和长期均线
        sma_short = close.rolling(20).mean().iloc[-1]
        sma_long = close.rolling(50).mean().iloc[-1]
        
        # 趋势强度
        trend_strength = safe_div(abs(sma_short - sma_long), sma_long)
        
        # 趋势方向
        if sma_short > sma_long:
            trend_direction = "UP"
        elif sma_short < sma_long:
            trend_direction = "DOWN"
        else:
            trend_direction = "SIDEWAYS"
        
        # 使用价格斜率确认趋势
        price_slope = self.calculate_price_slope(close)
        if abs(price_slope) < 0.001:
            trend_direction = "SIDEWAYS"
        
        return trend_direction, trend_strength
    
    def analyze_volatility(self, returns: pd.Series) -> str:
        """
        分析波动率水平
        
        Args:
            returns: 收益率序列
        
        Returns:
            波动率水平字符串
        """
        current_vol = returns.std()
        historical_vol = returns.rolling(20).std().mean()
        
        if current_vol > historical_vol * 1.5:
            return "HIGH"
        elif current_vol > historical_vol * 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    def calculate_price_slope(self, prices: pd.Series, window: int = 20) -> float:
        """
        计算价格斜率
        
        Args:
            prices: 价格序列
            window: 计算窗口
        
        Returns:
            价格斜率
        """
        try:
            recent_prices = prices.tail(window).values
            x = np.arange(len(recent_prices))
            slope = np.polyfit(x, recent_prices, 1)[0]
            return slope / recent_prices[-1]  # 标准化
        except:
            return 0.0
    
    def determine_market_phase(self, df: pd.DataFrame, trend_direction: str) -> str:
        """
        确定市场阶段
        
        Args:
            df: K线数据
            trend_direction: 趋势方向
        
        Returns:
            市场阶段
        """
        # 获取RSI (如果存在)
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
        else:
            # 简单计算RSI
            close = df['close']
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # 根据趋势和RSI判断阶段
        if trend_direction == "UP" and rsi < 70:
            return "BULL"
        elif trend_direction == "UP" and rsi >= 70:
            return "OVERBOUGHT"
        elif trend_direction == "DOWN" and rsi > 30:
            return "BEAR"
        elif trend_direction == "DOWN" and rsi <= 30:
            return "OVERSOLD"
        else:
            return "NEUTRAL"
    
    def calculate_confidence(self, trend_strength: float, volatility_level: str) -> float:
        """
        计算检测置信度
        
        Args:
            trend_strength: 趋势强度
            volatility_level: 波动率水平
        
        Returns:
            置信度 (0-1)
        """
        base_confidence = 0.5
        
        # 趋势强度越高，置信度越高
        base_confidence += min(trend_strength * 5, 0.3)
        
        # 波动率调整
        if volatility_level == "HIGH":
            base_confidence -= 0.1
        elif volatility_level == "LOW":
            base_confidence += 0.1
        
        return np.clip(base_confidence, 0.1, 0.95)
    
    def get_regime_summary(self) -> dict:
        """
        获取市场状态摘要
        
        Returns:
            状态摘要字典
        """
        if not self.current_regime:
            return {"status": "未检测"}
        
        return {
            "regime_type": self.current_regime.regime_type,
            "trend_direction": self.current_regime.trend_direction,
            "volatility_level": self.current_regime.volatility_level,
            "market_phase": self.current_regime.market_phase,
            "confidence": f"{self.current_regime.confidence:.2%}",
            "timestamp": self.current_regime.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def is_trending_market(self) -> bool:
        """检查是否为趋势市场"""
        return self.current_regime and self.current_regime.regime_type == "TRENDING"
    
    def is_bullish_phase(self) -> bool:
        """检查是否为看涨阶段"""
        return (self.current_regime and 
                self.current_regime.market_phase in ["BULL", "OVERSOLD"])
    
    def is_bearish_phase(self) -> bool:
        """检查是否为看跌阶段"""
        return (self.current_regime and 
                self.current_regime.market_phase in ["BEAR", "OVERBOUGHT"])