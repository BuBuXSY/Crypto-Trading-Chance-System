# strategies/whale_tracker.py
"""
鲸鱼追踪模块
监控大额交易活动
"""

from typing import Dict, List
from collections import defaultdict
from datetime import datetime
from core.logger import get_logger
from core.utils import now_local

logger = get_logger(__name__)

class EnhancedWhaleTracker:
    """增强鲸鱼追踪器"""
    
    def __init__(self, exchange, db_manager, config: Dict):
        logger.info("初始化鲸鱼追踪系统")
        self.exchange = exchange
        self.db = db_manager
        self.config = config
        self.whale_activities = defaultdict(list)
        self.alert_counter = defaultdict(int)
    
    def fetch_recent_trades(self, symbol: str, limit: int = 200):
        """
        获取最近成交记录
        
        Args:
            symbol: 交易对符号
            limit: 记录数量
        
        Returns:
            成交记录列表
        """
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"获取成交失败 {symbol}: {e}")
            return []
    
    def analyze_whale_activity(self, symbol: str) -> Dict:
        """
        分析鲸鱼活动
        
        Args:
            symbol: 交易对符号
        
        Returns:
            鲸鱼活动分析结果
        """
        logger.debug(f"分析 {symbol} 活动")
        trades = self.fetch_recent_trades(symbol)
        if not trades:
            return None
            
        whale_trades = []
        total_whale_volume = 0
        whale_buy_volume = 0
        whale_sell_volume = 0
        
        min_trade_usdt = self.config['whale_tracking']['min_trade_usdt']
        
        for trade in trades:
            try:
                amount_usdt = trade.get('cost', trade['amount'] * trade['price'])
                
                if amount_usdt >= min_trade_usdt:
                    whale_trades.append({
                        "time": trade['datetime'],
                        "side": trade['side'],
                        "amount": trade['amount'],
                        "price": trade['price'],
                        "cost": amount_usdt
                    })
                    
                    total_whale_volume += amount_usdt
                    if trade['side'] == 'buy':
                        whale_buy_volume += amount_usdt
                    else:
                        whale_sell_volume += amount_usdt
            except:
                continue
        
        if not whale_trades:
            return None
            
        # 判断行为模式
        behavior_pattern = "NEUTRAL"
        if whale_buy_volume > whale_sell_volume * 1.5:
            behavior_pattern = "ACCUMULATION"
        elif whale_sell_volume > whale_buy_volume * 1.5:
            behavior_pattern = "DISTRIBUTION"
        
        # 更新警报计数
        alert_threshold = self.config['whale_tracking']['alert_threshold']
        if len(whale_trades) >= alert_threshold:
            self.alert_counter[symbol] += 1
        
        result = {
            "symbol": symbol,
            "whale_trades": whale_trades[:20],
            "total_volume": total_whale_volume,
            "buy_volume": whale_buy_volume,
            "sell_volume": whale_sell_volume,
            "behavior_pattern": behavior_pattern,
            "alert_level": min(self.alert_counter[symbol], 5),
            "timestamp": now_local().isoformat()
        }
        
        # 保存到内存
        self.whale_activities[symbol].append(result)
        if len(self.whale_activities[symbol]) > 100:
            self.whale_activities[symbol] = self.whale_activities[symbol][-100:]
        
        return result
    
    def get_whale_report(self) -> Dict:
        """
        获取鲸鱼活动报告
        
        Returns:
            综合报告
        """
        report = {
            "active_symbols": [],
            "total_whale_volume": 0,
            "alerts": []
        }
        
        for symbol, activities in self.whale_activities.items():
            if activities:
                recent = activities[-1]
                report["active_symbols"].append({
                    "symbol": symbol,
                    "pattern": recent["behavior_pattern"],
                    "volume": recent["total_volume"]
                })
                report["total_whale_volume"] += recent["total_volume"]
                
                if recent["alert_level"] >= 3:
                    report["alerts"].append({
                        "symbol": symbol,
                        "pattern": recent["behavior_pattern"],
                        "level": recent["alert_level"]
                    })
        
        return report