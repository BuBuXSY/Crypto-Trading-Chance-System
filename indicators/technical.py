# indicators/technical.py
"""
技术指标计算模块
包含所有技术分析指标
"""

import pandas as pd
import numpy as np
from core.utils import safe_div
from core.logger import get_logger

logger = get_logger(__name__)

class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标
        
        Args:
            df: K线数据DataFrame
        
        Returns:
            添加了指标的DataFrame
        """
        if df.empty:
            return df
        
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        # 移动平均线
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma{period}'] = close.rolling(period).mean()
            df[f'ema{period}'] = close.ewm(span=period).mean()
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["signal"]
        
        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = safe_div(gain.rolling(14).mean(), loss.rolling(14).mean() + 1e-9)
        df["rsi"] = 100 - 100/(1 + rs)
        
        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_percent"] = safe_div(df["atr"], close) * 100
        
        # 布林带
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_middle"] = sma20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["bb_percent"] = safe_div(close - df["bb_lower"], df["bb_width"])
        
        # 成交量指标
        df["volume_sma"] = volume.rolling(20).mean()
        df["volume_ratio"] = safe_div(volume, df["volume_sma"])
        
        # Stochastic
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        df["stoch_k"] = 100 * safe_div(close - lowest_low, highest_high - lowest_low + 1e-9)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        
        # OBV (On Balance Volume)
        df['obv'] = (np.sign(close.diff()) * volume).cumsum()
        
        # Williams %R
        df["williams_r"] = -100 * safe_div(highest_high - close, highest_high - lowest_low + 1e-9)
        
        # CCI (Commodity Channel Index)
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = (typical_price - sma_tp).abs().rolling(20).mean()
        df["cci"] = safe_div(typical_price - sma_tp, 0.015 * mad)
        
        # MFI (Money Flow Index)
        mf = typical_price * volume
        mf_pos = pd.Series(np.where(typical_price > typical_price.shift(), mf, 0), index=df.index)
        mf_neg = pd.Series(np.where(typical_price < typical_price.shift(), mf, 0), index=df.index)
        mf_ratio = safe_div(mf_pos.rolling(14).sum(), mf_neg.rolling(14).sum() + 1e-9)
        df["mfi"] = 100 - 100/(1 + mf_ratio)
        
        return df
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        计算RSI
        
        Args:
            prices: 价格序列
            period: 周期
        
        Returns:
            RSI序列
        """
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        计算MACD
        
        Args:
            prices: 价格序列
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
        
        Returns:
            MACD, 信号线, 柱状图
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2):
        """
        计算布林带
        
        Args:
            prices: 价格序列
            period: 周期
            std_dev: 标准差倍数
        
        Returns:
            上轨, 中轨, 下轨
        """
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower