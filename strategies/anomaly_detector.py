# strategies/anomaly_detector.py
"""
异常检测模块
监控价格和成交量异常
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, List
from core.logger import get_logger

logger = get_logger(__name__)

# 检查ML库
try:
    from sklearn.ensemble import IsolationForest
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False
    logger.warning("scikit-learn未安装，ML异常检测不可用")

class EnhancedAnomalyDetector:
    """增强异常检测器"""
    
    def __init__(self, config: Dict):
        logger.info("初始化异常检测系统")
        self.config = config
        self.historical_data = defaultdict(deque)
        self.alerts = []
        self.anomaly_model = None
        
        if ML_AVAILABLE and config["anomaly_detection"]["use_ml"]:
            self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
    
    def update_and_detect(self, symbol: str, price: float, volume: float) -> List[Dict]:
        """
        更新数据并检测异常
        
        Args:
            symbol: 交易对符号
            price: 当前价格
            volume: 当前成交量
        
        Returns:
            异常列表
        """
        # 初始化历史数据
        if symbol not in self.historical_data:
            self.historical_data[symbol] = deque(maxlen=200)
        
        # 保存当前数据
        self.historical_data[symbol].append({
            "price": price,
            "volume": volume,
            "timestamp": pd.Timestamp.now()
        })
        
        anomalies = []
        
        # 传统异常检测
        anomalies.extend(self.detect_traditional_anomalies(symbol, price, volume))
        
        # ML异常检测
        if self.config["anomaly_detection"]["use_ml"] and ML_AVAILABLE:
            anomalies.extend(self.detect_ml_anomalies(symbol))
        
        # 保存警报
        for anomaly in anomalies:
            self.alerts.append(anomaly)
        
        # 限制警报数量
        if len(self.alerts) > 200:
            self.alerts = self.alerts[-200:]
        
        return anomalies
    
    def detect_traditional_anomalies(self, symbol, current_price, current_volume):
        """
        传统统计方法检测异常
        
        Args:
            symbol: 交易对
            current_price: 当前价格
            current_volume: 当前成交量
        
        Returns:
            异常列表
        """
        anomalies = []
        
        if len(self.historical_data[symbol]) < 20:
            return anomalies
        
        history = list(self.historical_data[symbol])
        prices = [h["price"] for h in history]
        volumes = [h["volume"] for h in history]
        
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        volume_mean = np.mean(volumes)
        
        # 价格Z-score异常
        if price_std > 0:
            price_zscore = abs((current_price - price_mean) / price_std)
            if price_zscore > 3:
                anomalies.append({
                    "type": "PRICE_ZSCORE",
                    "severity": "HIGH",
                    "message": f"价格Z-score异常: {price_zscore:.2f}",
                    "value": price_zscore,
                    "symbol": symbol
                })
        
        # 成交量异常
        volume_spike_threshold = self.config["anomaly_detection"]["volume_spike_threshold"]
        if volume_mean > 0 and current_volume > volume_mean * volume_spike_threshold:
            anomalies.append({
                "type": "VOLUME_SPIKE",
                "severity": "MEDIUM",
                "message": f"成交量异常放大 {current_volume/volume_mean:.1f}x",
                "value": current_volume,
                "symbol": symbol
            })
        
        # 价格突变
        if len(prices) >= 2:
            price_change = abs(current_price - prices[-2]) / prices[-2]
            price_spike_threshold = self.config["anomaly_detection"]["price_spike_threshold"]
            if price_change > price_spike_threshold:
                anomalies.append({
                    "type": "PRICE_SPIKE",
                    "severity": "HIGH" if price_change > 0.1 else "MEDIUM",
                    "message": f"价格突变 {price_change*100:.1f}%",
                    "value": price_change,
                    "symbol": symbol
                })
        
        return anomalies
    
    def detect_ml_anomalies(self, symbol):
        """
        机器学习方法检测异常
        
        Args:
            symbol: 交易对
        
        Returns:
            异常列表
        """
        anomalies = []
        
        if len(self.historical_data[symbol]) < 50 or not self.anomaly_model:
            return anomalies
        
        try:
            history = list(self.historical_data[symbol])
            features = []
            
            # 构建特征
            for i in range(1, len(history)):
                price_change = (history[i]["price"] - history[i-1]["price"]) / history[i-1]["price"]
                volume_change = (history[i]["volume"] - history[i-1]["volume"]) / (history[i-1]["volume"] + 1e-9)
                features.append([price_change, volume_change])
            
            if len(features) >= 20:
                X = np.array(features)
                self.anomaly_model.fit(X)
                
                # 预测最新数据
                latest_feature = features[-1:]
                prediction = self.anomaly_model.predict(latest_feature)
                
                if prediction[0] == -1:
                    anomalies.append({
                        "type": "ML_ANOMALY",
                        "severity": "MEDIUM",
                        "message": "机器学习检测到异常",
                        "value": self.anomaly_model.score_samples(latest_feature)[0],
                        "symbol": symbol
                    })
        except:
            pass
        
        return anomalies