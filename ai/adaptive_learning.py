# ai/adaptive_learning.py
"""
自适应学习系统模块
"""

import numpy as np
from typing import Dict
from collections import defaultdict
from core.logger import get_logger

logger = get_logger(__name__)

class AdaptiveLearningSystem:
    """自适应学习系统"""
    
    def __init__(self, db_manager):
        logger.info("初始化自适应学习系统")
        self.db = db_manager
        self.feature_importance_history = defaultdict(list)
        self.model_weights = defaultdict(dict)
        self.performance_tracker = defaultdict(list)
    
    def update_model_weights(self, symbol: str, model_performances: Dict):
        """
        基于性能更新模型权重
        
        Args:
            symbol: 交易对符号
            model_performances: 各模型性能
        """
        for model_name, performance in model_performances.items():
            self.performance_tracker[f"{symbol}_{model_name}"].append(performance)
            
            # 计算指数移动平均
            if len(self.performance_tracker[f"{symbol}_{model_name}"]) > 5:
                recent_performance = self.performance_tracker[f"{symbol}_{model_name}"][-5:]
                avg_performance = np.mean([p.get('r2_score', 0) for p in recent_performance])
                
                # 更新权重
                self.model_weights[symbol][model_name] = avg_performance
        
        # 归一化权重
        total = sum(self.model_weights[symbol].values())
        if total > 0:
            for model in self.model_weights[symbol]:
                self.model_weights[symbol][model] /= total
    
    def adapt_features(self, symbol: str, feature_importance: np.ndarray):
        """
        自适应特征选择
        
        Args:
            symbol: 交易对符号
            feature_importance: 特征重要性数组
        
        Returns:
            选择的特征索引
        """
        self.feature_importance_history[symbol].append(feature_importance)
        
        if len(self.feature_importance_history[symbol]) > 10:
            # 计算特征重要性的稳定性
            recent_importance = np.array(self.feature_importance_history[symbol][-10:])
            avg_importance = np.mean(recent_importance, axis=0)
            std_importance = np.std(recent_importance, axis=0)
            
            # 选择稳定且重要的特征
            stability_score = 1 - (std_importance / (avg_importance + 1e-9))
            feature_score = avg_importance * stability_score
            
            # 返回top特征索引
            top_features = np.argsort(feature_score)[-50:]  # 选择top 50特征
            return top_features
        
        return None
    
    def get_optimal_weights(self, symbol: str) -> Dict:
        """
        获取最优模型权重
        
        Args:
            symbol: 交易对符号
        
        Returns:
            模型权重字典
        """
        if symbol in self.model_weights:
            return self.model_weights[symbol]
        else:
            # 返回默认权重
            return {
                'rf': 0.25,
                'gb': 0.25,
                'xgb': 0.25,
                'lstm': 0.25
            }