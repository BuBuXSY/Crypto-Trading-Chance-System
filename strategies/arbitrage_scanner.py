# strategies/arbitrage_scanner.py
"""
套利扫描模块
寻找交易所间价差机会
"""

import ccxt
import time
from typing import Dict, List
from core.logger import get_logger
from core.utils import safe_div, now_local

logger = get_logger(__name__)

class EnhancedArbitrageScanner:
    """增强套利扫描器"""
    
    def __init__(self, config: Dict):
        logger.info("初始化套利扫描器")
        self.config = config
        self.exchanges = {}
        self.opportunities = []
        self.setup_exchanges()
    
    def setup_exchanges(self):
        """设置多个交易所连接"""
        for ex_name in self.config["backup_exchanges"]:
            try:
                exchange_class = getattr(ccxt, ex_name)
                self.exchanges[ex_name] = exchange_class({
                    "options": {"defaultType": "swap"},
                    "enableRateLimit": True
                })
                self.exchanges[ex_name].load_markets()
                logger.info(f"已连接 {ex_name}")
            except Exception as e:
                logger.error(f"连接 {ex_name} 失败: {e}")
    
    def scan_opportunities(self, symbols: List[str]) -> List[Dict]:
        """
        扫描套利机会
        
        Args:
            symbols: 交易对列表
        
        Returns:
            套利机会列表
        """
        logger.debug("扫描套利机会")
        opportunities = []
        
        for symbol in symbols:
            try:
                prices_and_depths = {}
                
                # 获取各交易所价格
                for ex_name, exchange in self.exchanges.items():
                    try:
                        ticker = exchange.fetch_ticker(symbol)
                        
                        depth_quality = 1.0
                        if self.config["arbitrage"]["check_depth"]:
                            try:
                                order_book = exchange.fetch_order_book(symbol, limit=10)
                                depth_quality = self.assess_depth_quality(order_book)
                            except:
                                pass
                        
                        prices_and_depths[ex_name] = {
                            "bid": ticker.get('bid', 0),
                            "ask": ticker.get('ask', 0),
                            "last": ticker.get('last', 0),
                            "volume": ticker.get('quoteVolume', 0),
                            "depth_quality": depth_quality,
                            "fee_rate": self.config["arbitrage"]["fee_rates"].get(ex_name, 0.001)
                        }
                    except:
                        continue
                
                if len(prices_and_depths) < 2:
                    continue
                
                # 计算套利机会
                for ex1 in prices_and_depths:
                    for ex2 in prices_and_depths:
                        if ex1 == ex2:
                            continue
                        
                        opportunity = self.calculate_arbitrage_profit(
                            symbol, ex1, ex2, prices_and_depths
                        )
                        
                        if opportunity and opportunity['net_profit_percent'] > self.config["arbitrage"]["min_profit_percent"]:
                            opportunities.append(opportunity)
                            
            except Exception as e:
                logger.debug(f"分析 {symbol} 失败: {e}")
        
        # 排序并保存
        opportunities.sort(key=lambda x: x['net_profit_percent'], reverse=True)
        self.opportunities = opportunities[:20]
        return self.opportunities
    
    def calculate_arbitrage_profit(self, symbol, ex1, ex2, prices_and_depths):
        """
        计算套利利润
        
        Args:
            symbol: 交易对
            ex1: 买入交易所
            ex2: 卖出交易所
            prices_and_depths: 价格深度数据
        
        Returns:
            套利机会字典
        """
        data1 = prices_and_depths[ex1]
        data2 = prices_and_depths[ex2]
        
        buy_price = data1['ask']
        sell_price = data2['bid']
        
        if not buy_price or not sell_price or buy_price >= sell_price:
            return None
        
        # 计算费用
        buy_fee = buy_price * data1['fee_rate']
        sell_fee = sell_price * data2['fee_rate']
        
        # 计算净利润
        gross_profit = sell_price - buy_price
        net_profit = gross_profit - buy_fee - sell_fee
        net_profit_percent = safe_div(net_profit, buy_price) * 100
        
        # 评估可行性
        feasibility_score = self.assess_feasibility(
            data1['volume'], data2['volume'],
            data1['depth_quality'], data2['depth_quality']
        )
        
        return {
            "symbol": symbol,
            "buy_exchange": ex1,
            "sell_exchange": ex2,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "net_profit_percent": net_profit_percent,
            "feasibility_score": feasibility_score,
            "volume_available": min(data1['volume'], data2['volume']),
            "timestamp": now_local().isoformat()
        }
    
    def assess_depth_quality(self, order_book):
        """
        评估订单深度质量
        
        Args:
            order_book: 订单簿
        
        Returns:
            深度质量分数
        """
        if not order_book or not order_book.get('bids') or not order_book.get('asks'):
            return 0.0
        
        try:
            spread = order_book['asks'][0][0] - order_book['bids'][0][0]
            mid_price = (order_book['asks'][0][0] + order_book['bids'][0][0]) / 2
            spread_percent = safe_div(spread, mid_price) * 100
            
            # 计算深度
            bid_depth = sum(bid[1] for bid in order_book['bids'][:5])
            ask_depth = sum(ask[1] for ask in order_book['asks'][:5])
            total_depth = bid_depth + ask_depth
            
            # 评分
            spread_score = max(0, 1 - spread_percent / 1)
            depth_score = min(1, total_depth / 1000)
            
            return (spread_score + depth_score) / 2
        except:
            return 0.5
    
    def assess_feasibility(self, vol1, vol2, depth1, depth2):
        """
        评估套利可行性
        
        Args:
            vol1: 交易所1成交量
            vol2: 交易所2成交量
            depth1: 交易所1深度
            depth2: 交易所2深度
        
        Returns:
            可行性分数
        """
        min_volume = min(vol1, vol2)
        volume_score = min(1, safe_div(min_volume, self.config["arbitrage"]["min_volume_usdt"]))
        depth_score = (depth1 + depth2) / 2
        return volume_score * 0.6 + depth_score * 0.4