# indicators/advanced.py
"""
高级技术指标模块
包含Ichimoku、Elliott Wave等高级指标
"""

import pandas as pd
import numpy as np
from core.utils import safe_div
from core.logger import get_logger

logger = get_logger(__name__)

# 检查scipy是否可用
try:
    from scipy.signal import argrelextrema
    SCIPY_AVAILABLE = True
except:
    SCIPY_AVAILABLE = False
    logger.warning("scipy未安装，部分高级指标将不可用")

class AdvancedIndicators:
    """高级技术指标计算器"""
    
    @staticmethod
    def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算一目均衡表
        
        Args:
            df: K线数据DataFrame
        
        Returns:
            添加了Ichimoku指标的DataFrame
        """
        logger.debug("计算一目均衡表")
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 转换线 (Tenkan-sen) - 9日
        period9_high = high.rolling(9).max()
        period9_low = low.rolling(9).min()
        df['tenkan_sen'] = (period9_high + period9_low) / 2
        
        # 基准线 (Kijun-sen) - 26日
        period26_high = high.rolling(26).max()
        period26_low = low.rolling(26).min()
        df['kijun_sen'] = (period26_high + period26_low) / 2
        
        # 先行跨度A (Senkou Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # 先行跨度B (Senkou Span B)
        period52_high = high.rolling(52).max()
        period52_low = low.rolling(52).min()
        df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        
        # 迟行跨度 (Chikou Span)
        df['chikou_span'] = close.shift(-26)
        
        # 云层颜色
        df['cloud_color'] = np.where(df['senkou_span_a'] > df['senkou_span_b'], 'green', 'red')
        
        return df
    
    @staticmethod
    def elliott_wave_detection(df: pd.DataFrame) -> dict:
        """
        检测艾略特波浪
        
        Args:
            df: K线数据DataFrame
        
        Returns:
            波浪检测结果
        """
        logger.debug("检测艾略特波浪")
        prices = df['close'].values
        
        if len(prices) < 50:
            return {"wave": None, "position": None, "direction": None}
        
        if not SCIPY_AVAILABLE:
            logger.warning("scipy未安装，跳过波浪检测")
            return {"wave": None, "position": None}
            
        try:
            # 寻找局部极值点
            highs = argrelextrema(prices, np.greater, order=5)[0]
            lows = argrelextrema(prices, np.less, order=5)[0]
            
            if len(highs) >= 3 and len(lows) >= 2:
                # 检测推进浪
                if prices[highs[-1]] > prices[highs[-2]] > prices[highs[-3]]:
                    return {"wave": "IMPULSE", "position": "Wave 5", "direction": "UP"}
                elif prices[lows[-1]] < prices[lows[-2]] < prices[lows[-3]]:
                    return {"wave": "IMPULSE", "position": "Wave 5", "direction": "DOWN"}
        except Exception as e:
            logger.debug(f"波浪检测异常: {e}")
        
        return {"wave": "CORRECTIVE", "position": "ABC", "direction": "SIDEWAYS"}
    
    @staticmethod
    def wyckoff_phase(df: pd.DataFrame) -> str:
        """
        分析威科夫阶段
        
        Args:
            df: K线数据DataFrame
        
        Returns:
            当前威科夫阶段
        """
        logger.debug("分析威科夫阶段")
        
        if len(df) < 50:
            return "UNKNOWN"
        
        close = df['close']
        volume = df['volume']
        
        sma_50 = close.rolling(50).mean()
        volume_sma = volume.rolling(20).mean()
        
        # 计算价格和成交量趋势
        price_trend = safe_div((close.iloc[-1] - close.iloc[-20]), close.iloc[-20])
        volume_trend = safe_div((volume.iloc[-1] - volume_sma.iloc[-1]), volume_sma.iloc[-1])
        
        # 判断威科夫阶段
        if price_trend > 0.1 and volume_trend > 0.5:
            return "MARKUP"  # 上涨阶段
        elif price_trend < -0.1 and volume_trend > 0.5:
            return "MARKDOWN"  # 下跌阶段
        elif abs(price_trend) < 0.05 and volume_trend < -0.3:
            return "ACCUMULATION"  # 吸筹阶段
        elif abs(price_trend) < 0.05 and volume_trend > 0.3:
            return "DISTRIBUTION"  # 派发阶段
        else:
            return "BACKING_UP"  # 回撤阶段
    
    @staticmethod
    def market_profile(df: pd.DataFrame, periods: int = 20) -> dict:
        """
        计算市场轮廓
        
        Args:
            df: K线数据DataFrame
            periods: 计算周期
        
        Returns:
            市场轮廓数据
        """
        logger.debug("计算市场轮廓")
        
        if len(df) < periods:
            return {
                "poc": 0,
                "value_area_high": 0,
                "value_area_low": 0,
                "profile_shape": "UNKNOWN"
            }
        
        recent = df.tail(periods)
        prices = recent['close'].values
        volumes = recent['volume'].values
        
        # 计算价格分布
        price_bins = np.histogram(prices, bins=10, weights=volumes)
        
        # POC (Point of Control) - 成交量最大的价格
        poc_index = np.argmax(price_bins[0])
        poc_price = (price_bins[1][poc_index] + price_bins[1][poc_index + 1]) / 2
        
        # 计算价值区域 (70%成交量)
        cumsum = np.cumsum(price_bins[0])
        total_volume = cumsum[-1] if len(cumsum) > 0 else 1
        value_area_low_idx = np.where(cumsum >= total_volume * 0.15)[0][0] if total_volume > 0 else 0
        value_area_high_idx = np.where(cumsum >= total_volume * 0.85)[0][0] if total_volume > 0 else 0
        
        # 判断轮廓形状
        if abs(poc_index - 5) < 2:
            profile_shape = "NORMAL"  # 正态分布
        elif poc_index < 3:
            profile_shape = "P_SHAPED"  # P型
        elif poc_index > 7:
            profile_shape = "B_SHAPED"  # B型
        else:
            profile_shape = "SKEWED"  # 偏态
        
        return {
            "poc": poc_price,
            "value_area_high": price_bins[1][value_area_high_idx],
            "value_area_low": price_bins[1][value_area_low_idx],
            "profile_shape": profile_shape
        }
    
    @staticmethod
    def pivot_points(df: pd.DataFrame) -> dict:
        """
        计算枢轴点
        
        Args:
            df: K线数据DataFrame
        
        Returns:
            枢轴点数据
        """
        if df.empty:
            return {}
        
        # 使用前一日的数据
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # 计算枢轴点
        pivot = (high + low + close) / 3
        
        # 阻力位
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        # 支撑位
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            "pivot": pivot,
            "r1": r1, "r2": r2, "r3": r3,
            "s1": s1, "s2": s2, "s3": s3
        }