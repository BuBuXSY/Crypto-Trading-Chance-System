# utils/dynamic_parameters.py
"""
动态参数调整器
根据系统运行状态动态调整交易参数
"""

import time
from collections import deque
from typing import Dict, List
import numpy as np

from core.logger import get_logger

logger = get_logger(__name__)


class DynamicParameterAdjuster:
    """动态参数调整器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.signal_history = deque(maxlen=50)  # 保存最近50个周期的信号数量
        self.adjustment_history = deque(maxlen=20)  # 调整历史
        self.performance_history = deque(maxlen=30)  # 性能历史
        
        # 调整参数
        self.adjustment_params = {
            "signal_count_target_min": 3,    # 目标最少信号数
            "signal_count_target_max": 15,   # 目标最多信号数
            "adjustment_factor": 0.1,        # 调整幅度
            "min_adjustment_interval": 300,  # 最小调整间隔（秒）
            "stability_threshold": 5,        # 稳定性阈值
        }
        
        self.last_adjustment_time = 0
        
        logger.info("[动态调整] 初始化完成")
    
    def adjust_thresholds(self, current_signal_count: int, success_rate: float = None):
        """根据信号数量和成功率动态调整阈值"""
        try:
            current_time = time.time()
            
            # 记录当前状态
            self.signal_history.append(current_signal_count)
            if success_rate is not None:
                self.performance_history.append(success_rate)
            
            # 检查是否需要调整
            if not self._should_adjust(current_time):
                return
            
            # 计算调整策略
            adjustments = self._calculate_adjustments()
            
            # 应用调整
            if adjustments:
                self._apply_adjustments(adjustments)
                self.last_adjustment_time = current_time
                
        except Exception as e:
            logger.error(f"[动态调整] 调整阈值失败: {e}")
    
    def _should_adjust(self, current_time: float) -> bool:
        """判断是否应该进行调整"""
        # 检查调整间隔
        if current_time - self.last_adjustment_time < self.adjustment_params["min_adjustment_interval"]:
            return False
        
        # 需要足够的历史数据
        if len(self.signal_history) < 10:
            return False
        
        # 检查是否有明显的趋势需要调整
        recent_signals = list(self.signal_history)[-10:]
        avg_signals = np.mean(recent_signals)
        
        return (avg_signals < self.adjustment_params["signal_count_target_min"] or 
                avg_signals > self.adjustment_params["signal_count_target_max"])
    
    def _calculate_adjustments(self) -> Dict:
        """计算需要进行的调整"""
        adjustments = {}
        
        try:
            # 分析信号数量趋势
            recent_signals = list(self.signal_history)[-10:]
            avg_signals = np.mean(recent_signals)
            signal_variance = np.var(recent_signals)
            
            # 分析成功率趋势
            avg_success_rate = None
            if len(self.performance_history) >= 5:
                recent_performance = list(self.performance_history)[-5:]
                avg_success_rate = np.mean(recent_performance)
            
            logger.info(f"[动态调整] 平均信号数: {avg_signals:.1f}, 方差: {signal_variance:.1f}")
            if avg_success_rate:
                logger.info(f"[动态调整] 平均成功率: {avg_success_rate:.1%}")
            
            # 信号数量调整
            if avg_signals < self.adjustment_params["signal_count_target_min"]:
                # 信号太少，降低要求
                adjustments.update(self._get_signal_increase_adjustments(avg_signals, avg_success_rate))
                
            elif avg_signals > self.adjustment_params["signal_count_target_max"]:
                # 信号太多，提高要求
                adjustments.update(self._get_signal_decrease_adjustments(avg_signals, avg_success_rate))
            
            # 基于成功率的调整
            if avg_success_rate is not None:
                success_adjustments = self._get_performance_based_adjustments(avg_success_rate)
                adjustments.update(success_adjustments)
            
            return adjustments
            
        except Exception as e:
            logger.error(f"[动态调整] 计算调整参数失败: {e}")
            return {}
    
    def _get_signal_increase_adjustments(self, avg_signals: float, success_rate: float = None) -> Dict:
        """获取增加信号数量的调整参数"""
        adjustments = {}
        
        try:
            # 计算调整幅度
            signal_deficit = self.adjustment_params["signal_count_target_min"] - avg_signals
            adjustment_factor = min(0.2, signal_deficit * 0.1)  # 最大调整20%
            
            # 调整信号质量阈值
            if "signal_quality" in self.config:
                quality_config = self.config["signal_quality"]
                
                # 降低最低分数要求
                for score_key in ["min_score_weak", "min_score_medium", "min_score_strong"]:
                    if score_key in quality_config:
                        current_val = quality_config[score_key]
                        new_val = max(1, int(current_val * (1 - adjustment_factor)))
                        adjustments[f"signal_quality.{score_key}"] = new_val
                
                # 降低优势要求
                if "require_advantage" in quality_config:
                    current_val = quality_config["require_advantage"]
                    new_val = max(1.0, current_val * (1 - adjustment_factor))
                    adjustments["signal_quality.require_advantage"] = round(new_val, 2)
            
            # 调整扫描参数
            adjustments["scan_symbols_count"] = min(60, self.config.get("scan_symbols_count", 30) + 5)
            adjustments["max_workers"] = min(30, self.config.get("max_workers", 15) + 2)
            
            logger.info(f"[动态调整] 增加信号策略，调整幅度: {adjustment_factor:.1%}")
            
        except Exception as e:
            logger.error(f"[动态调整] 计算信号增加调整失败: {e}")
        
        return adjustments
    
    def _get_signal_decrease_adjustments(self, avg_signals: float, success_rate: float = None) -> Dict:
        """获取减少信号数量的调整参数"""
        adjustments = {}
        
        try:
            # 计算调整幅度
            signal_excess = avg_signals - self.adjustment_params["signal_count_target_max"]
            adjustment_factor = min(0.15, signal_excess * 0.05)  # 最大调整15%
            
            # 调整信号质量阈值
            if "signal_quality" in self.config:
                quality_config = self.config["signal_quality"]
                
                # 提高最低分数要求
                for score_key in ["min_score_weak", "min_score_medium", "min_score_strong"]:
                    if score_key in quality_config:
                        current_val = quality_config[score_key]
                        new_val = min(20, int(current_val * (1 + adjustment_factor)))
                        adjustments[f"signal_quality.{score_key}"] = new_val
                
                # 提高优势要求
                if "require_advantage" in quality_config:
                    current_val = quality_config["require_advantage"]
                    new_val = min(2.0, current_val * (1 + adjustment_factor))
                    adjustments["signal_quality.require_advantage"] = round(new_val, 2)
            
            logger.info(f"[动态调整] 减少信号策略，调整幅度: {adjustment_factor:.1%}")
            
        except Exception as e:
            logger.error(f"[动态调整] 计算信号减少调整失败: {e}")
        
        return adjustments
    
    def _get_performance_based_adjustments(self, success_rate: float) -> Dict:
        """基于成功率的调整"""
        adjustments = {}
        
        try:
            # 成功率过低，放宽要求
            if success_rate < 0.4:
                adjustments["performance_adjustment"] = "relaxed"
                logger.info(f"[动态调整] 成功率偏低 ({success_rate:.1%})，放宽要求")
                
            # 成功率很高，可以提高要求
            elif success_rate > 0.7:
                adjustments["performance_adjustment"] = "strict"
                logger.info(f"[动态调整] 成功率较高 ({success_rate:.1%})，提高要求")
                
        except Exception as e:
            logger.error(f"[动态调整] 基于成功率调整失败: {e}")
        
        return adjustments
    
    def _apply_adjustments(self, adjustments: Dict):
        """应用调整参数"""
        try:
            applied_count = 0
            
            for key, value in adjustments.items():
                try:
                    if "." in key:
                        # 嵌套配置
                        parts = key.split(".")
                        config_section = self.config
                        
                        # 导航到嵌套位置
                        for part in parts[:-1]:
                            if part not in config_section:
                                config_section[part] = {}
                            config_section = config_section[part]
                        
                        # 设置值
                        old_value = config_section.get(parts[-1], "无")
                        config_section[parts[-1]] = value
                        
                        logger.info(f"[动态调整] {key}: {old_value} → {value}")
                        applied_count += 1
                        
                    else:
                        # 顶级配置
                        old_value = self.config.get(key, "无")
                        self.config[key] = value
                        
                        logger.info(f"[动态调整] {key}: {old_value} → {value}")
                        applied_count += 1
                        
                except Exception as e:
                    logger.warning(f"[动态调整] 应用调整 {key} 失败: {e}")
            
            # 记录调整历史
            adjustment_record = {
                "timestamp": time.time(),
                "adjustments": adjustments,
                "applied_count": applied_count
            }
            self.adjustment_history.append(adjustment_record)
            
            logger.info(f"[动态调整] 成功应用 {applied_count} 项调整")
            
        except Exception as e:
            logger.error(f"[动态调整] 应用调整失败: {e}")
    
    def get_adjustment_summary(self) -> Dict:
        """获取调整摘要"""
        try:
            if not self.adjustment_history:
                return {"total_adjustments": 0, "last_adjustment": None}
            
            return {
                "total_adjustments": len(self.adjustment_history),
                "last_adjustment": self.adjustment_history[-1]["timestamp"],
                "recent_signal_avg": float(np.mean(list(self.signal_history)[-10:])) if len(self.signal_history) >= 10 else 0,
                "recent_success_rate": float(np.mean(list(self.performance_history)[-5:])) if len(self.performance_history) >= 5 else None,
            }
            
        except Exception as e:
            logger.error(f"[动态调整] 获取摘要失败: {e}")
            return {"error": str(e)}
    
    def reset_history(self):
        """重置历史记录"""
        try:
            self.signal_history.clear()
            self.adjustment_history.clear()
            self.performance_history.clear()
            self.last_adjustment_time = 0
            
            logger.info("[动态调整] 历史记录已重置")
            
        except Exception as e:
            logger.error(f"[动态调整] 重置历史失败: {e}")


class AdvancedParameterOptimizer:
    """高级参数优化器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.optimization_history = deque(maxlen=100)
        
    def optimize_for_market_regime(self, market_regime: str, volatility: float):
        """根据市场制度优化参数"""
        try:
            optimizations = {}
            
            if market_regime == "BULL":
                # 牛市：更积极的参数
                optimizations["signal_quality.min_score_weak"] = max(3, 
                    int(self.config.get("signal_quality", {}).get("min_score_weak", 8) * 0.8))
                optimizations["risk_tolerance"] = 1.2
                
            elif market_regime == "BEAR":
                # 熊市：更保守的参数
                optimizations["signal_quality.min_score_weak"] = min(15, 
                    int(self.config.get("signal_quality", {}).get("min_score_weak", 8) * 1.3))
                optimizations["risk_tolerance"] = 0.7
                
            elif market_regime == "SIDEWAYS":
                # 横盘：中性参数
                optimizations["signal_quality.min_score_weak"] = self.config.get("signal_quality", {}).get("min_score_weak", 8)
                optimizations["risk_tolerance"] = 1.0
            
            # 基于波动率调整
            if volatility > 0.05:  # 高波动
                optimizations["volatility_adjustment"] = "high"
                optimizations["stop_loss_multiplier"] = 1.5
            elif volatility < 0.02:  # 低波动
                optimizations["volatility_adjustment"] = "low"
                optimizations["stop_loss_multiplier"] = 0.8
            
            return optimizations
            
        except Exception as e:
            logger.error(f"[参数优化] 市场制度优化失败: {e}")
            return {}
    
    def optimize_for_time_of_day(self, hour: int):
        """根据时间优化参数"""
        try:
            optimizations = {}
            
            if 0 <= hour <= 6:  # 亚洲时段
                optimizations["asia_session"] = True
                optimizations["activity_multiplier"] = 0.8
                
            elif 7 <= hour <= 15:  # 欧洲时段
                optimizations["europe_session"] = True
                optimizations["activity_multiplier"] = 1.2
                
            elif 16 <= hour <= 23:  # 美洲时段
                optimizations["america_session"] = True
                optimizations["activity_multiplier"] = 1.0
            
            return optimizations
            
        except Exception as e:
            logger.error(f"[参数优化] 时间优化失败: {e}")
            return {}