# ai/pattern_recognition.py
"""
模式识别模块
检测各种技术形态
"""

import numpy as np
import pandas as pd
from typing import Dict
from core.logger import get_logger

logger = get_logger(__name__)

class PatternRecognition:
    """模式识别器"""
    
    def __init__(self):
        logger.info("初始化模式识别系统")
        self.patterns = {}
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """
        检测技术形态
        
        Args:
            df: K线数据
        
        Returns:
            检测到的模式字典
        """
        patterns = {}
        
        if len(df) < 50:
            return patterns
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # 检测各种形态
        patterns['head_shoulders'] = self.detect_head_shoulders(close)
        patterns['double_top'] = self.detect_double_top(high, low, close)
        patterns['triangle'] = self.detect_triangle(high, low)
        patterns['flag'] = self.detect_flag(close)
        patterns['wedge'] = self.detect_wedge(high, low)
        
        return patterns
    
    def detect_head_shoulders(self, prices):
        """检测头肩形态"""
        if len(prices) < 50:
            return 0.0
            
        try:
            # 简化的头肩形态检测
            window = 10
            peaks = []
            valleys = []
            
            for i in range(window, len(prices) - window):
                if prices[i] == max(prices[i-window:i+window]):
                    peaks.append(i)
                if prices[i] == min(prices[i-window:i+window]):
                    valleys.append(i)
            
            # 检查是否形成头肩形态
            if len(peaks) >= 3 and len(valleys) >= 2:
                # 检查肩部高度相似
                left_shoulder = prices[peaks[-3]]
                head = prices[peaks[-2]]
                right_shoulder = prices[peaks[-1]]
                
                if head > left_shoulder and head > right_shoulder:
                    shoulder_similarity = 1 - abs(left_shoulder - right_shoulder) / head
                    if shoulder_similarity > 0.9:
                        return shoulder_similarity
            
            return 0.0
        except:
            return 0.0
    
    def detect_double_top(self, high, low, close):
        """检测双顶形态"""
        try:
            if len(close) < 30:
                return 0.0
                
            # 寻找两个相似高点
            recent_high = high[-30:]
            max_idx1 = np.argmax(recent_high[:15])
            max_idx2 = np.argmax(recent_high[15:]) + 15
            
            peak1 = recent_high[max_idx1]
            peak2 = recent_high[max_idx2]
            
            # 检查两个高点是否相似
            similarity = 1 - abs(peak1 - peak2) / max(peak1, peak2)
            
            if similarity > 0.95:
                return similarity
            
            return 0.0
        except:
            return 0.0
    
    def detect_triangle(self, high, low):
        """检测三角形态"""
        try:
            if len(high) < 20:
                return 0.0
                
            # 检查高点和低点是否收敛
            high_slope = np.polyfit(range(len(high[-20:])), high[-20:], 1)[0]
            low_slope = np.polyfit(range(len(low[-20:])), low[-20:], 1)[0]
            
            # 收敛三角形
            if high_slope < 0 and low_slope > 0:
                convergence = abs(high_slope) + abs(low_slope)
                return min(1.0, convergence)
            
            return 0.0
        except:
            return 0.0
    
    def detect_flag(self, prices):
        """检测旗形形态"""
        try:
            if len(prices) < 20:
                return 0.0
                
            # 检查是否有强烈趋势后的整理
            trend = np.polyfit(range(10), prices[-20:-10], 1)[0]
            consolidation = np.std(prices[-10:])
            
            if abs(trend) > 0.01 and consolidation < np.std(prices[-20:-10]) * 0.5:
                return 0.8
            
            return 0.0
        except:
            return 0.0
    
    def detect_wedge(self, high, low):
        """检测楔形形态"""
        try:
            if len(high) < 20:
                return 0.0
                
            # 检查高低点是否形成楔形
            high_slope = np.polyfit(range(len(high[-20:])), high[-20:], 1)[0]
            low_slope = np.polyfit(range(len(low[-20:])), low[-20:], 1)[0]
            
            # 上升楔形或下降楔形
            if (high_slope > 0 and low_slope > 0 and high_slope < low_slope) or \
               (high_slope < 0 and low_slope < 0 and high_slope > low_slope):
                return 0.7
            
            return 0.0
        except:
            return 0.0
    
    def get_pattern_signal(self, patterns: Dict) -> float:
        """
        获取综合形态信号
        
        Args:
            patterns: 形态检测结果
        
        Returns:
            综合信号强度 (-1到1)
        """
        if not patterns:
            return 0.0
            
        # 加权平均
        weights = {
            'head_shoulders': 0.3,
            'double_top': 0.25,
            'triangle': 0.2,
            'flag': 0.15,
            'wedge': 0.1
        }
        
        signal = sum(patterns.get(p, 0) * w for p, w in weights.items())
        return np.clip(signal, -1, 1)