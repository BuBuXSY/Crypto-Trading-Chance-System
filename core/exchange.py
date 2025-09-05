# core/exchange.py
"""
交易所管理模块
处理所有交易所连接和数据获取
"""

import ccxt
import pandas as pd
import time
from typing import Dict, List, Optional
from core.logger import get_logger

logger = get_logger(__name__)

class ExchangeManager:
    """交易所管理器"""
    
    def __init__(self, config: Dict):
        """
        初始化交易所管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.exchange = None
        self.setup_exchange()
    
    def setup_exchange(self):
        """设置交易所连接"""
        primary_exchange = self.config["exchange"]
        backup_exchanges = self.config["backup_exchanges"]
        
        # 尝试连接主交易所和备用交易所
        for exchange_name in [primary_exchange] + backup_exchanges:
            try:
                logger.info(f"尝试连接 {exchange_name}")
                
                opts = {
                    "options": {"defaultType": "swap"},
                    "enableRateLimit": True,
                    "timeout": 30000,
                    "rateLimit": 100,
                }
                
                exchange_class = getattr(ccxt, exchange_name)
                self.exchange = exchange_class(opts)
                
                # 测试连接
                self.exchange.load_markets()
                logger.info(f"成功连接 {exchange_name}")
                self.config["exchange"] = exchange_name  # 更新配置
                return
                
            except Exception as e:
                logger.error(f"连接 {exchange_name} 失败: {e}")
                continue
        
        raise Exception("所有交易所连接失败")
    
    def fetch_tickers(self) -> Dict:
        """
        获取所有交易对的行情
        
        Returns:
            行情字典
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                tickers = self.exchange.fetch_tickers()
                if tickers:
                    return tickers
            except ccxt.NetworkError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"获取行情失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(5 * (attempt + 1))
                    continue
                logger.error(f"网络连接最终失败: {e}")
            except Exception as e:
                logger.error(f"获取行情失败: {e}")
                break
        
        return {}
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """
        获取单个交易对的行情
        
        Args:
            symbol: 交易对符号
        
        Returns:
            行情数据
        """
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"获取 {symbol} 行情失败: {e}")
            return {}
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            limit: 数据条数
        
        Returns:
            K线数据DataFrame
        """
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # 获取OHLCV数据
                data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not data or len(data) < 10:
                    logger.warning(f"{symbol} 返回数据不足: {len(data) if data else 0} 条")
                    return pd.DataFrame()
                
                # 转换为DataFrame
                df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df.set_index("time", inplace=True)
                
                # 数据清洗
                df = df.dropna()
                df = df[df['volume'] > 0]  # 移除无交易量的数据
                
                if len(df) < 50:
                    logger.warning(f"{symbol} 清洗后数据不足: {len(df)} 条")
                    return pd.DataFrame()
                    
                logger.debug(f"成功获取 {symbol} 数据: {len(df)} 条")
                return df
                
            except ccxt.NetworkError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"网络错误 {symbol} (尝试 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                logger.error(f"网络连接最终失败 {symbol}: {e}")
            except ccxt.ExchangeError as e:
                logger.error(f"交易所错误 {symbol}: {e}")
                break
            except Exception as e:
                logger.error(f"获取K线失败 {symbol}: {e}")
                break
        
        return pd.DataFrame()
    
    def fetch_trades(self, symbol: str, limit: int = 200) -> List[Dict]:
        """
        获取最近交易记录
        
        Args:
            symbol: 交易对符号
            limit: 记录数量
        
        Returns:
            交易记录列表
        """
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            return trades if trades else []
        except Exception as e:
            logger.error(f"获取 {symbol} 交易记录失败: {e}")
            return []
    
    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        获取订单簿
        
        Args:
            symbol: 交易对符号
            limit: 深度限制
        
        Returns:
            订单簿数据
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit=limit)
            return order_book if order_book else {}
        except Exception as e:
            logger.error(f"获取 {symbol} 订单簿失败: {e}")
            return {}
    
    def get_exchange_info(self) -> Dict:
        """
        获取交易所信息
        
        Returns:
            交易所信息字典
        """
        try:
            return {
                "id": self.exchange.id,
                "name": self.exchange.name,
                "countries": getattr(self.exchange, 'countries', []),
                "urls": getattr(self.exchange, 'urls', {}),
                "version": getattr(self.exchange, 'version', 'unknown'),
                "has": self.exchange.has,
                "timeframes": getattr(self.exchange, 'timeframes', {}),
                "limits": getattr(self.exchange, 'limits', {}),
                "fees": getattr(self.exchange, 'fees', {}),
                "status": "connected"
            }
        except Exception as e:
            logger.error(f"获取交易所信息失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def check_symbol_validity(self, symbol: str) -> bool:
        """
        检查交易对是否有效
        
        Args:
            symbol: 交易对符号
        
        Returns:
            是否有效
        """
        try:
            markets = self.exchange.markets
            return symbol in markets
        except Exception as e:
            logger.error(f"检查交易对有效性失败: {e}")
            return False
    
    def get_available_symbols(self, quote_currency: str = "USDT") -> List[str]:
        """
        获取可用交易对列表
        
        Args:
            quote_currency: 计价货币
        
        Returns:
            交易对列表
        """
        try:
            markets = self.exchange.markets
            symbols = []
            
            for symbol, market in markets.items():
                if market.get('quote') == quote_currency and market.get('active', True):
                    symbols.append(symbol)
            
            return sorted(symbols)
        except Exception as e:
            logger.error(f"获取交易对列表失败: {e}")
            return []
    
    def test_connection(self) -> Dict:
        """
        测试交易所连接
        
        Returns:
            测试结果
        """
        try:
            # 测试获取服务器时间
            start_time = time.time()
            
            if hasattr(self.exchange, 'fetch_time'):
                server_time = self.exchange.fetch_time()
            else:
                # 如果没有fetch_time方法，尝试获取BTC行情作为连接测试
                self.exchange.fetch_ticker('BTC/USDT')
                server_time = int(time.time() * 1000)
            
            latency = (time.time() - start_time) * 1000
            
            return {
                "status": "success",
                "exchange": self.exchange.id,
                "server_time": server_time,
                "latency_ms": round(latency, 2),
                "markets_count": len(self.exchange.markets),
                "rate_limit": self.exchange.rateLimit
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "exchange": getattr(self.exchange, 'id', 'unknown'),
                "error": str(e)
            }
    
    def get_rate_limit_info(self) -> Dict:
        """
        获取API限制信息
        
        Returns:
            限制信息字典
        """
        try:
            return {
                "rate_limit": self.exchange.rateLimit,
                "enable_rate_limit": self.exchange.enableRateLimit,
                "last_request_time": getattr(self.exchange, 'lastRestRequestTimestamp', None),
                "remaining_calls": getattr(self.exchange, 'remaining', None)
            }
        except Exception as e:
            logger.error(f"获取限制信息失败: {e}")
            return {}
    
    def handle_rate_limit(self, delay_multiplier: float = 1.5):
        """
        处理API限制
        
        Args:
            delay_multiplier: 延迟倍数
        """
        try:
            rate_limit = self.exchange.rateLimit
            delay = (rate_limit / 1000) * delay_multiplier
            logger.info(f"API限制延迟: {delay:.2f}秒")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"处理API限制失败: {e}")
            time.sleep(1)  # 默认延迟1秒
    
    def reconnect(self) -> bool:
        """
        重新连接交易所
        
        Returns:
            是否重连成功
        """
        try:
            logger.info("尝试重新连接交易所...")
            self.setup_exchange()
            test_result = self.test_connection()
            
            if test_result["status"] == "success":
                logger.info("重新连接成功")
                return True
            else:
                logger.error(f"重新连接失败: {test_result.get('error', '未知错误')}")
                return False
                
        except Exception as e:
            logger.error(f"重新连接过程失败: {e}")
            return False
    
    def close(self):
        """关闭交易所连接"""
        try:
            if hasattr(self.exchange, 'close') and callable(self.exchange.close):
                self.exchange.close()
                logger.info("交易所连接已关闭")
        except Exception as e:
            logger.error(f"关闭连接失败: {e}")