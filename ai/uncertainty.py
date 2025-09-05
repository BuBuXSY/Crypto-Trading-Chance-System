# ai/uncertainty.py
"""
不确定性估计模块
"""

import numpy as np
from typing import Dict
from core.logger import get_logger

logger = get_logger(__name__)

class UncertaintyEstimator:
    """不确定性估计器"""
    
    def __init__(self):
        logger.info("初始化不确定性估计器")
    
    def estimate(self, predictions: Dict, performance: Dict) -> float:
        """
        估计预测不确定性
        
        Args:
            predictions: 各模型预测值
            performance: 模型性能指标
        
        Returns:
            不确定性评分 (0-1之间，越高越不确定)
        """
        uncertainties = []
        
        # 基于预测分歧
        if len(predictions) > 1:
            pred_values = list(predictions.values())
            std = np.std(pred_values)
            mean = np.mean(np.abs(pred_values))
            if mean > 0:
                cv = std / mean  # 变异系数
                uncertainties.append(cv)
        
        # 基于模型性能
        if performance:
            r2_scores = [p.get('r2_score', 0) for p in performance.values()]
            if r2_scores:
                avg_r2 = np.mean(r2_scores)
                uncertainties.append(1 - avg_r2)
        
        if uncertainties:
            return np.mean(uncertainties)
        return 0.5