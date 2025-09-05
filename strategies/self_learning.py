# strategies/self_learning.py
"""
自学习系统模块
根据历史表现动态调整指标权重
"""

import os
import json
import time
import numpy as np
from typing import Dict, List
from core.logger import get_logger
from ai.adaptive_learning import AdaptiveLearningSystem

logger = get_logger(__name__)

class SelfLearningSystem:
    """自学习系统"""
    
    def __init__(self, db_manager, config: Dict):
        logger.info("初始化自学习系统")
        self.db = db_manager
        self.config = config
        self.indicator_weights = self.load_weights()
        self.last_update = time.time()
        self.adaptive_learning = AdaptiveLearningSystem(db_manager)
        self.performance_history = {}
        
    def load_weights(self) -> Dict:
        """
        加载指标权重
        
        Returns:
            权重字典
        """
        weight_file = "indicator_weights.json"
        default_weights = {
            "ai_prediction": 1.0,
            "whale_activity": 1.0,
            "technical_indicators": 1.0,
            "market_regime": 1.0,
            "onchain_data": 1.0,
            "volume_analysis": 1.0,
            "advanced_indicators": 1.0,
            "pattern_recognition": 1.0,
            "market_sentiment": 1.0,
        }
        
        try:
            if os.path.exists(weight_file):
                with open(weight_file, 'r') as f:
                    loaded = json.load(f)
                    # 确保所有必要的键都存在
                    for key in default_weights:
                        if key not in loaded:
                            loaded[key] = default_weights[key]
                    return loaded
        except Exception as e:
            logger.error(f"加载权重文件失败: {e}")
        
        return default_weights
    
    def save_weights(self):
        """保存权重到文件"""
        try:
            with open("indicator_weights.json", 'w') as f:
                json.dump(self.indicator_weights, f, indent=2)
            logger.info("权重已保存")
        except Exception as e:
            logger.error(f"保存权重失败: {e}")
    
    def update_weights(self):
        """更新指标权重"""
        logger.info("更新指标权重")
        
        # 获取信号统计数据
        stats = self.db.get_signal_statistics(30)
        
        if not stats or len(stats) < self.config["self_learning"]["min_samples"]:
            logger.info("样本不足，跳过更新")
            return
        
        # 基于性能调整权重
        if self.config["self_learning"]["performance_based_adjustment"]:
            self.adjust_weights_by_performance(stats)
        
        # 标准化权重
        self.normalize_weights()
        
        # 保存权重
        self.save_weights()
        self.last_update = time.time()
        
        logger.info(f"权重已更新: {self.indicator_weights}")
    
    def adjust_weights_by_performance(self, stats: List):
        """
        基于表现调整权重
        
        Args:
            stats: 统计数据列表
        """
        # 计算总体成功率
        total_signals = sum(stat[0] for stat in stats)
        total_success = sum(stat[1] or 0 for stat in stats)
        overall_success_rate = total_success / total_signals if total_signals > 0 else 0
        
        logger.info(f"总体成功率: {overall_success_rate:.2%}")
        
        # 分析各个指标的贡献
        indicator_performance = self.analyze_indicator_performance(stats)
        
        # 调整权重
        adjustment_rate = self.config["self_learning"]["weight_adjustment_rate"]
        for indicator, performance_score in indicator_performance.items():
            if indicator in self.indicator_weights:
                if performance_score > 0.6:
                    # 表现好的指标增加权重
                    adjustment = adjustment_rate * 0.1
                elif performance_score < 0.4:
                    # 表现差的指标减少权重
                    adjustment = -adjustment_rate * 0.1
                else:
                    adjustment = 0
                
                self.indicator_weights[indicator] *= (1 + adjustment)
                
                # 限制权重范围
                self.indicator_weights[indicator] = max(0.1, min(2.0, self.indicator_weights[indicator]))
    
    def analyze_indicator_performance(self, stats: List) -> Dict:
        """
        分析各指标表现
        
        Args:
            stats: 统计数据
        
        Returns:
            各指标表现评分
        """
        # 这里简化处理，实际应该更复杂的分析
        performance = {}
        
        # 基于信号质量分析各指标贡献
        quality_performance = {}
        for stat in stats:
            total, success, avg_conf, quality, side = stat
            if quality not in quality_performance:
                quality_performance[quality] = {"total": 0, "success": 0}
            quality_performance[quality]["total"] += total
            quality_performance[quality]["success"] += success or 0
        
        # 计算质量成功率
        for quality, data in quality_performance.items():
            if data["total"] > 0:
                success_rate = data["success"] / data["total"]
                
                # 根据质量等级推断指标贡献
                if quality == "VIP":
                    # VIP信号主要依赖AI和模式识别
                    performance["ai_prediction"] = success_rate
                    performance["pattern_recognition"] = success_rate
                elif quality == "STRONG":
                    # STRONG信号综合多个指标
                    performance["technical_indicators"] = success_rate
                    performance["whale_activity"] = success_rate
                elif quality == "MEDIUM":
                    # MEDIUM信号主要技术指标
                    performance["technical_indicators"] = success_rate * 0.8
                    performance["volume_analysis"] = success_rate * 0.8
        
        # 填充缺失的指标（使用默认值）
        all_indicators = list(self.indicator_weights.keys())
        for indicator in all_indicators:
            if indicator not in performance:
                performance[indicator] = 0.5  # 中性表现
        
        return performance
    
    def normalize_weights(self):
        """标准化权重，确保总和合理"""
        total_weight = sum(self.indicator_weights.values())
        target_sum = len(self.indicator_weights)  # 期望平均权重为1.0
        
        if total_weight > 0:
            normalization_factor = target_sum / total_weight
            for key in self.indicator_weights:
                self.indicator_weights[key] *= normalization_factor
    
    def should_update(self) -> bool:
        """
        检查是否应该更新权重
        
        Returns:
            是否需要更新
        """
        update_interval = self.config["self_learning"]["update_interval"]
        return time.time() - self.last_update > update_interval
    
    def get_weight_for_indicator(self, indicator: str) -> float:
        """
        获取指定指标的权重
        
        Args:
            indicator: 指标名称
        
        Returns:
            权重值
        """
        return self.indicator_weights.get(indicator, 1.0)
    
    def record_signal_performance(self, symbol: str, indicator: str, 
                                success: bool, confidence: float):
        """
        记录信号表现
        
        Args:
            symbol: 交易对
            indicator: 指标名称
            success: 是否成功
            confidence: 置信度
        """
        key = f"{symbol}_{indicator}"
        if key not in self.performance_history:
            self.performance_history[key] = []
        
        self.performance_history[key].append({
            "success": success,
            "confidence": confidence,
            "timestamp": time.time()
        })
        
        # 限制历史记录长度
        if len(self.performance_history[key]) > 100:
            self.performance_history[key] = self.performance_history[key][-100:]
    
    def get_indicator_reliability(self, indicator: str, symbol: str = None) -> float:
        """
        获取指标可靠性评分
        
        Args:
            indicator: 指标名称
            symbol: 交易对（可选）
        
        Returns:
            可靠性评分 (0-1)
        """
        if symbol:
            key = f"{symbol}_{indicator}"
            history = self.performance_history.get(key, [])
        else:
            # 所有交易对的平均可靠性
            history = []
            for key, records in self.performance_history.items():
                if key.endswith(f"_{indicator}"):
                    history.extend(records)
        
        if not history:
            return 0.5  # 默认中等可靠性
        
        # 计算成功率
        recent_history = history[-20:]  # 最近20个记录
        success_count = sum(1 for record in recent_history if record["success"])
        
        return success_count / len(recent_history)
    
    def generate_learning_report(self) -> str:
        """
        生成学习报告
        
        Returns:
            报告文本
        """
        report = []
        report.append("=== 自学习系统报告 ===")
        report.append(f"更新时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_update))}")
        report.append("")
        
        # 当前权重
        report.append("【当前指标权重】")
        sorted_weights = sorted(self.indicator_weights.items(), 
                              key=lambda x: x[1], reverse=True)
        for indicator, weight in sorted_weights:
            report.append(f"{indicator}: {weight:.3f}")
        report.append("")
        
        # 可靠性评分
        report.append("【指标可靠性评分】")
        for indicator in self.indicator_weights:
            reliability = self.get_indicator_reliability(indicator)
            report.append(f"{indicator}: {reliability:.2%}")
        report.append("")
        
        # 学习建议
        report.append("【优化建议】")
        best_indicator = max(self.indicator_weights.items(), key=lambda x: x[1])
        worst_indicator = min(self.indicator_weights.items(), key=lambda x: x[1])
        
        report.append(f"表现最佳: {best_indicator[0]} (权重: {best_indicator[1]:.3f})")
        report.append(f"需要改进: {worst_indicator[0]} (权重: {worst_indicator[1]:.3f})")
        
        if worst_indicator[1] < 0.5:
            report.append(f"建议检查 {worst_indicator[0]} 指标的计算逻辑")
        
        return "\n".join(report)
    
    def reset_weights(self):
        """重置权重到默认值"""
        logger.info("重置权重到默认值")
        self.indicator_weights = {
            "ai_prediction": 1.0,
            "whale_activity": 1.0,
            "technical_indicators": 1.0,
            "market_regime": 1.0,
            "onchain_data": 1.0,
            "volume_analysis": 1.0,
            "advanced_indicators": 1.0,
            "pattern_recognition": 1.0,
            "market_sentiment": 1.0,
        }
        self.save_weights()
    
    def backup_weights(self) -> str:
        """
        备份当前权重
        
        Returns:
            备份文件路径
        """
        backup_file = f"weights_backup_{int(time.time())}.json"
        try:
            with open(backup_file, 'w') as f:
                json.dump(self.indicator_weights, f, indent=2)
            logger.info(f"权重已备份到: {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(f"备份权重失败: {e}")
            return ""
    
    def restore_weights(self, backup_file: str) -> bool:
        """
        从备份恢复权重
        
        Args:
            backup_file: 备份文件路径
        
        Returns:
            是否成功恢复
        """
        try:
            with open(backup_file, 'r') as f:
                self.indicator_weights = json.load(f)
            self.save_weights()
            logger.info(f"已从 {backup_file} 恢复权重")
            return True
        except Exception as e:
            logger.error(f"恢复权重失败: {e}")
            return False