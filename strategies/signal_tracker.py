# strategies/signal_tracker.py
"""
信号追踪模块
追踪和验证交易信号的表现
"""

from typing import Dict, List
from core.logger import get_logger
from core.utils import safe_div, now_local
from data.models import Signal

logger = get_logger(__name__)

class SignalTracker:
    """信号追踪器"""
    
    def __init__(self, db_manager, exchange, config: Dict):
        logger.info("初始化信号追踪系统")
        self.db = db_manager
        self.exchange = exchange
        self.config = config
        self.tracked_signals = {}
    
    def track_signal(self, signal_data: Dict):
        """
        追踪新信号
        
        Args:
            signal_data: 信号数据字典
        """
        signal = Signal(
            timestamp=now_local(),
            symbol=signal_data['symbol'],
            side=signal_data['side'],
            entry=signal_data['entry'],
            sl=signal_data['sl'],
            tps=signal_data['tps'],
            score=signal_data['score'],
            confidence=signal_data['confidence'],
            quality=signal_data['quality'],
            reason=signal_data['reason'],
            predicted_outcome=signal_data.get('predicted_change', 0),
            ai_confidence=signal_data.get('ai_confidence', 0),
            pattern_detected=signal_data.get('pattern_detected', '')
        )
        
        # 保存到数据库
        self.db.save_signal(signal)
        
        # 添加到内存追踪
        signal_id = f"{signal.symbol}_{signal.timestamp.isoformat()}"
        self.tracked_signals[signal_id] = signal_data
        
        logger.info(f"信号已记录: {signal.symbol} {signal.side} ({signal.quality})")
    
    def check_signal_outcomes(self):
        """检查历史信号结果"""
        logger.info("检查历史信号结果")
        
        check_hours = self.config["signal_tracking"]["check_after_hours"]
        unchecked = self.db.get_unchecked_signals(check_hours)
        
        checked_count = 0
        for signal_record in unchecked:
            try:
                signal_id = signal_record[0]
                symbol = signal_record[2]
                side = signal_record[3]
                entry = signal_record[4]
                sl = signal_record[5]
                tps_str = signal_record[6]
                
                # 解析止盈目标
                import json
                tps = json.loads(tps_str) if tps_str else []
                
                # 获取当前价格
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # 计算实际结果
                result = self.calculate_signal_outcome(
                    side, entry, current_price, sl, tps
                )
                
                # 更新数据库
                self.db.update_signal_outcome(
                    signal_id, 
                    result['actual_outcome'], 
                    result['success']
                )
                
                checked_count += 1
                logger.info(f"信号 {signal_id} 结果: {'成功' if result['success'] else '失败'} "
                           f"({result['actual_outcome']:.2%})")
                
            except Exception as e:
                logger.error(f"检查信号失败: {e}")
                continue
        
        logger.info(f"检查完成: {checked_count} 个信号")
    
    def calculate_signal_outcome(self, side: str, entry: float, current: float, 
                                sl: float, tps: List[float]) -> Dict:
        """
        计算信号结果
        
        Args:
            side: 交易方向
            entry: 入场价格
            current: 当前价格
            sl: 止损价格
            tps: 止盈目标列表
        
        Returns:
            结果字典
        """
        if side == "LONG":
            actual_outcome = safe_div(current - entry, entry)
            
            # 检查是否触及止损
            if current <= sl:
                success = False
                outcome_type = "STOP_LOSS"
            # 检查是否达到止盈目标
            elif tps and current >= tps[0]:
                success = True
                outcome_type = "TAKE_PROFIT"
            # 按当前价格计算
            else:
                success = actual_outcome > self.config["signal_tracking"]["success_threshold"]
                outcome_type = "CURRENT_PRICE"
        else:  # SHORT
            actual_outcome = safe_div(entry - current, entry)
            
            # 检查是否触及止损
            if current >= sl:
                success = False
                outcome_type = "STOP_LOSS"
            # 检查是否达到止盈目标
            elif tps and current <= tps[0]:
                success = True
                outcome_type = "TAKE_PROFIT"
            # 按当前价格计算
            else:
                success = actual_outcome > self.config["signal_tracking"]["success_threshold"]
                outcome_type = "CURRENT_PRICE"
        
        return {
            "actual_outcome": actual_outcome,
            "success": success,
            "outcome_type": outcome_type
        }
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """
        获取信号表现统计
        
        Args:
            days: 统计天数
        
        Returns:
            统计数据字典
        """
        stats = self.db.get_signal_statistics(days)
        
        performance = {
            "total_signals": 0,
            "success_rate": 0.0,
            "by_quality": {},
            "by_side": {},
            "average_return": 0.0,
            "best_signal": None,
            "worst_signal": None
        }
        
        total_return = 0
        total_successful = 0
        returns = []
        
        for stat in stats:
            total, success, avg_conf, quality, side = stat
            
            performance["total_signals"] += total
            total_successful += success or 0
            
            # 按质量分组
            if quality not in performance["by_quality"]:
                performance["by_quality"][quality] = {"total": 0, "success": 0, "rate": 0}
            performance["by_quality"][quality]["total"] += total
            performance["by_quality"][quality]["success"] += success or 0
            
            # 按方向分组
            if side not in performance["by_side"]:
                performance["by_side"][side] = {"total": 0, "success": 0, "rate": 0}
            performance["by_side"][side]["total"] += total
            performance["by_side"][side]["success"] += success or 0
        
        # 计算成功率
        if performance["total_signals"] > 0:
            performance["success_rate"] = safe_div(total_successful, performance["total_signals"])
        
        # 计算各组成功率
        for quality_data in performance["by_quality"].values():
            if quality_data["total"] > 0:
                quality_data["rate"] = safe_div(quality_data["success"], quality_data["total"])
        
        for side_data in performance["by_side"].values():
            if side_data["total"] > 0:
                side_data["rate"] = safe_div(side_data["success"], side_data["total"])
        
        return performance
    
    def get_quality_analysis(self) -> Dict:
        """
        获取信号质量分析
        
        Returns:
            质量分析数据
        """
        performance = self.get_performance_stats()
        
        quality_ranking = []
        for quality, data in performance["by_quality"].items():
            if data["total"] >= 5:  # 至少5个信号才有统计意义
                quality_ranking.append({
                    "quality": quality,
                    "success_rate": data["rate"],
                    "total_signals": data["total"],
                    "confidence_level": self.get_quality_confidence(quality)
                })
        
        # 按成功率排序
        quality_ranking.sort(key=lambda x: x["success_rate"], reverse=True)
        
        return {
            "quality_ranking": quality_ranking,
            "best_quality": quality_ranking[0]["quality"] if quality_ranking else None,
            "reliability_threshold": 0.6,  # 60%以上认为可靠
            "recommendation": self.generate_quality_recommendation(quality_ranking)
        }
    
    def get_quality_confidence(self, quality: str) -> str:
        """获取质量等级的置信度描述"""
        confidence_map = {
            "VIP": "极高",
            "STRONG": "高",
            "MEDIUM": "中等", 
            "WEAK": "低"
        }
        return confidence_map.get(quality, "未知")
    
    def generate_quality_recommendation(self, quality_ranking: List[Dict]) -> str:
        """生成质量建议"""
        if not quality_ranking:
            return "数据不足，需要更多信号历史"
        
        best = quality_ranking[0]
        if best["success_rate"] > 0.7:
            return f"建议重点关注{best['quality']}级别信号，成功率达{best['success_rate']:.1%}"
        elif best["success_rate"] > 0.5:
            return f"{best['quality']}级别信号表现最佳，但成功率仅{best['success_rate']:.1%}，需要优化"
        else:
            return "所有信号质量都需要改进，建议检查策略参数"
    
    def get_recent_signals(self, hours: int = 24) -> List[Dict]:
        """
        获取最近的信号记录
        
        Args:
            hours: 小时数
        
        Returns:
            信号列表
        """
        # 这里简化实现，实际可以从数据库查询
        recent = []
        cutoff_time = now_local().timestamp() - (hours * 3600)
        
        for signal_id, signal_data in self.tracked_signals.items():
            timestamp = signal_data.get('timestamp')
            if timestamp and timestamp > cutoff_time:
                recent.append(signal_data)
        
        return sorted(recent, key=lambda x: x.get('timestamp', 0), reverse=True)
    
    def export_performance_report(self) -> str:
        """
        导出表现报告
        
        Returns:
            报告文本
        """
        stats = self.get_performance_stats()
        quality_analysis = self.get_quality_analysis()
        
        report = []
        report.append("=== 信号追踪表现报告 ===")
        report.append(f"统计时间: {now_local().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 总体统计
        report.append("【总体表现】")
        report.append(f"总信号数: {stats['total_signals']}")
        report.append(f"成功率: {stats['success_rate']:.1%}")
        report.append("")
        
        # 按质量统计
        report.append("【按质量统计】")
        for quality, data in stats["by_quality"].items():
            report.append(f"{quality}: {data['success']}/{data['total']} ({data['rate']:.1%})")
        report.append("")
        
        # 按方向统计
        report.append("【按方向统计】")
        for side, data in stats["by_side"].items():
            report.append(f"{side}: {data['success']}/{data['total']} ({data['rate']:.1%})")
        report.append("")
        
        # 质量分析
        report.append("【质量分析】")
        report.append(quality_analysis["recommendation"])
        
        return "\n".join(report)