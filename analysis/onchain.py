# analysis/onchain.py
"""
链上数据分析模块
获取和分析区块链相关数据
"""

import time
import requests
from typing import Dict, Optional
from core.logger import get_logger
from core.utils import now_local

logger = get_logger(__name__)

class OnChainAnalyzer:
    """链上数据分析器"""
    
    def __init__(self, config: Dict):
        logger.info("初始化链上数据分析器")
        self.config = config
        self.fear_greed_cache = None
        self.last_check = 0
        self.btc_metrics_cache = {}
        self.funding_rates_cache = {}
        
    def fetch_fear_greed_index(self) -> Optional[Dict]:
        """
        获取恐惧贪婪指数
        
        Returns:
            恐惧贪婪指数数据或None
        """
        try:
            # 检查缓存
            if time.time() - self.last_check < 3600:  # 1小时缓存
                return self.fear_greed_cache
                
            response = requests.get(
                self.config["onchain"]["fear_greed_api"], 
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and data['data']:
                    index_value = int(data['data'][0]['value'])
                    classification = data['data'][0]['value_classification']
                    
                    self.fear_greed_cache = {
                        "value": index_value,
                        "classification": classification,
                        "signal": self._interpret_fear_greed(index_value),
                        "timestamp": now_local().isoformat()
                    }
                    self.last_check = time.time()
                    
                    logger.info(f"恐惧贪婪指数: {index_value} - {classification}")
                    return self.fear_greed_cache
                    
        except requests.exceptions.Timeout:
            logger.warning("恐惧贪婪指数API超时")
        except Exception as e:
            logger.error(f"获取恐惧贪婪指数失败: {e}")
        
        return None
    
    def _interpret_fear_greed(self, value: int) -> str:
        """
        解释恐惧贪婪指数
        
        Args:
            value: 指数值 (0-100)
        
        Returns:
            交易信号
        """
        if value <= 20:
            return "EXTREME_FEAR"  # 极度恐惧，可能是买入机会
        elif value <= 40:
            return "FEAR"  # 恐惧
        elif value <= 60:
            return "NEUTRAL"  # 中性
        elif value <= 80:
            return "GREED"  # 贪婪
        else:
            return "EXTREME_GREED"  # 极度贪婪，可能是卖出信号
    
    def fetch_btc_network_metrics(self) -> Optional[Dict]:
        """
        获取比特币网络指标
        
        Returns:
            网络指标数据或None
        """
        try:
            # 这里应该接入真实的链上数据API，如Glassnode、CoinMetrics等
            # 示例使用模拟数据
            
            current_time = time.time()
            if current_time - self.btc_metrics_cache.get('timestamp', 0) < 3600:
                return self.btc_metrics_cache
            
            # 模拟数据 - 实际应该从API获取
            metrics = {
                "active_addresses": 850000,  # 活跃地址数
                "hash_rate": 450_000_000,    # 算力 (TH/s)
                "difficulty": 55_000_000_000_000,  # 挖矿难度
                "mempool_size": 15.5,        # 内存池大小 (MB)
                "avg_fee": 25.0,             # 平均手续费 (sat/vB)
                "hodl_waves": {
                    "1d_to_1w": 3.2,         # 1天到1周的币龄分布
                    "1w_to_1m": 8.5,
                    "1m_to_3m": 12.3,
                    "3m_to_6m": 15.8,
                    "6m_to_1y": 22.4,
                    "1y_to_2y": 18.9,
                    "2y_plus": 18.9
                },
                "timestamp": current_time
            }
            
            self.btc_metrics_cache = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"获取BTC网络指标失败: {e}")
            return None
    
    def fetch_funding_rates(self, symbols: list = None) -> Dict:
        """
        获取合约资金费率
        
        Args:
            symbols: 交易对列表
        
        Returns:
            资金费率数据
        """
        if symbols is None:
            symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        
        funding_data = {}
        
        for symbol in symbols:
            try:
                # 这里应该从交易所API获取真实资金费率
                # 示例使用模拟数据
                funding_rate = self._simulate_funding_rate(symbol)
                
                funding_data[symbol] = {
                    "current_rate": funding_rate,
                    "next_funding_time": "2024-01-01T08:00:00Z",
                    "signal": self._interpret_funding_rate(funding_rate),
                    "historical_avg": funding_rate * 0.8,  # 模拟历史平均
                }
                
            except Exception as e:
                logger.debug(f"获取 {symbol} 资金费率失败: {e}")
                continue
        
        self.funding_rates_cache = funding_data
        return funding_data
    
    def _simulate_funding_rate(self, symbol: str) -> float:
        """模拟资金费率数据"""
        import random
        # 实际应该从交易所API获取
        base_rate = 0.0001  # 0.01%
        variation = random.uniform(-0.0005, 0.0005)
        return base_rate + variation
    
    def _interpret_funding_rate(self, rate: float) -> str:
        """
        解释资金费率
        
        Args:
            rate: 资金费率
        
        Returns:
            信号解释
        """
        if rate > 0.001:  # 0.1%
            return "VERY_BULLISH"  # 多头情绪极强
        elif rate > 0.0005:  # 0.05%
            return "BULLISH"  # 多头情绪强
        elif rate > -0.0005:  # -0.05% to 0.05%
            return "NEUTRAL"  # 中性
        elif rate > -0.001:  # -0.1%
            return "BEARISH"  # 空头情绪强
        else:
            return "VERY_BEARISH"  # 空头情绪极强
    
    def analyze_whale_movements(self, symbol: str = "BTC") -> Dict:
        """
        分析鲸鱼资金流动
        
        Args:
            symbol: 加密货币符号
        
        Returns:
            鲸鱼活动分析
        """
        try:
            # 这里应该接入链上数据提供商的API
            # 示例返回模拟数据
            
            analysis = {
                "large_transactions_24h": 45,  # 24小时大额交易数量
                "whale_netflow": -120.5,       # 鲸鱼净流入/流出 (负值表示流出)
                "exchange_inflow": 850.2,      # 交易所流入
                "exchange_outflow": 970.7,     # 交易所流出
                "net_exchange_flow": -120.5,   # 净流入（负值表示净流出）
                "signal": "ACCUMULATION",      # ACCUMULATION/DISTRIBUTION/NEUTRAL
                "confidence": 0.75,
                "timestamp": now_local().isoformat()
            }
            
            # 根据净流入确定信号
            if analysis["net_exchange_flow"] < -100:
                analysis["signal"] = "ACCUMULATION"  # 净流出，可能是囤积
            elif analysis["net_exchange_flow"] > 100:
                analysis["signal"] = "DISTRIBUTION"  # 净流入，可能是抛售
            else:
                analysis["signal"] = "NEUTRAL"
            
            return analysis
            
        except Exception as e:
            logger.error(f"分析鲸鱼活动失败: {e}")
            return {}
    
    def get_long_short_ratio(self, symbol: str = "BTC") -> Dict:
        """
        获取多空比数据
        
        Args:
            symbol: 交易对符号
        
        Returns:
            多空比数据
        """
        try:
            # 示例返回模拟数据
            # 实际应该从交易所API获取
            
            long_ratio = 0.65  # 65%多头
            short_ratio = 0.35  # 35%空头
            
            ratio_data = {
                "long_percentage": long_ratio * 100,
                "short_percentage": short_ratio * 100,
                "long_short_ratio": long_ratio / short_ratio,
                "signal": self._interpret_long_short_ratio(long_ratio / short_ratio),
                "timestamp": now_local().isoformat()
            }
            
            return ratio_data
            
        except Exception as e:
            logger.error(f"获取多空比失败: {e}")
            return {}
    
    def _interpret_long_short_ratio(self, ratio: float) -> str:
        """
        解释多空比
        
        Args:
            ratio: 多空比值
        
        Returns:
            信号解释
        """
        if ratio > 3.0:
            return "EXTREME_LONG"  # 极度偏多，可能反转
        elif ratio > 2.0:
            return "VERY_LONG"     # 非常偏多
        elif ratio > 1.5:
            return "BULLISH"       # 偏多
        elif ratio > 0.67:
            return "NEUTRAL"       # 相对平衡
        elif ratio > 0.5:
            return "BEARISH"       # 偏空
        else:
            return "EXTREME_SHORT"  # 极度偏空，可能反转
    
    def analyze_stablecoin_flows(self) -> Dict:
        """
        分析稳定币流动
        
        Returns:
            稳定币流动分析
        """
        try:
            # 示例返回模拟数据
            analysis = {
                "usdt_supply_change_24h": 50_000_000,    # USDT供应量变化
                "usdc_supply_change_24h": 25_000_000,    # USDC供应量变化
                "total_stablecoin_inflow": 75_000_000,   # 总稳定币流入
                "signal": "BULLISH",                     # 稳定币流入通常看涨
                "interpretation": "大量稳定币流入，可能准备购买",
                "timestamp": now_local().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"分析稳定币流动失败: {e}")
            return {}
    
    def get_comprehensive_onchain_signal(self) -> Dict:
        """
        获取综合链上信号
        
        Returns:
            综合链上分析结果
        """
        try:
            # 收集各项指标
            fear_greed = self.fetch_fear_greed_index()
            btc_metrics = self.fetch_btc_network_metrics()
            funding_rates = self.fetch_funding_rates()
            whale_analysis = self.analyze_whale_movements()
            long_short = self.get_long_short_ratio()
            stablecoin_flows = self.analyze_stablecoin_flows()
            
            # 计算综合评分
            signals = []
            weights = []
            
            if fear_greed:
                fear_signal = self._fear_greed_to_score(fear_greed["signal"])
                signals.append(fear_signal)
                weights.append(0.25)
            
            if whale_analysis:
                whale_signal = self._whale_signal_to_score(whale_analysis["signal"])
                signals.append(whale_signal)
                weights.append(0.20)
            
            if funding_rates:
                avg_funding_signal = sum(
                    self._funding_signal_to_score(data["signal"]) 
                    for data in funding_rates.values()
                ) / len(funding_rates)
                signals.append(avg_funding_signal)
                weights.append(0.20)
            
            if long_short:
                ls_signal = self._long_short_signal_to_score(long_short["signal"])
                signals.append(ls_signal)
                weights.append(0.15)
            
            if stablecoin_flows:
                stable_signal = 0.6 if stablecoin_flows["signal"] == "BULLISH" else 0.4
                signals.append(stable_signal)
                weights.append(0.20)
            
            # 计算加权平均
            if signals:
                weighted_score = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
            else:
                weighted_score = 0.5
            
            # 确定最终信号
            if weighted_score > 0.7:
                final_signal = "STRONG_BULLISH"
            elif weighted_score > 0.6:
                final_signal = "BULLISH"
            elif weighted_score > 0.4:
                final_signal = "NEUTRAL"
            elif weighted_score > 0.3:
                final_signal = "BEARISH"
            else:
                final_signal = "STRONG_BEARISH"
            
            comprehensive_analysis = {
                "overall_signal": final_signal,
                "confidence_score": weighted_score,
                "components": {
                    "fear_greed": fear_greed,
                    "whale_analysis": whale_analysis,
                    "funding_rates": funding_rates,
                    "long_short_ratio": long_short,
                    "stablecoin_flows": stablecoin_flows
                },
                "recommendation": self._generate_recommendation(final_signal, weighted_score),
                "timestamp": now_local().isoformat()
            }
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"生成综合链上信号失败: {e}")
            return {"overall_signal": "NEUTRAL", "confidence_score": 0.5}
    
    def _fear_greed_to_score(self, signal: str) -> float:
        """恐惧贪婪信号转评分"""
        signal_map = {
            "EXTREME_FEAR": 0.8,    # 极度恐惧 -> 看涨
            "FEAR": 0.6,
            "NEUTRAL": 0.5,
            "GREED": 0.4,
            "EXTREME_GREED": 0.2    # 极度贪婪 -> 看跌
        }
        return signal_map.get(signal, 0.5)
    
    def _whale_signal_to_score(self, signal: str) -> float:
        """鲸鱼信号转评分"""
        signal_map = {
            "ACCUMULATION": 0.7,
            "NEUTRAL": 0.5,
            "DISTRIBUTION": 0.3
        }
        return signal_map.get(signal, 0.5)
    
    def _funding_signal_to_score(self, signal: str) -> float:
        """资金费率信号转评分"""
        signal_map = {
            "VERY_BULLISH": 0.2,    # 极度多头 -> 反向看跌
            "BULLISH": 0.4,
            "NEUTRAL": 0.5,
            "BEARISH": 0.6,
            "VERY_BEARISH": 0.8     # 极度空头 -> 反向看涨
        }
        return signal_map.get(signal, 0.5)
    
    def _long_short_signal_to_score(self, signal: str) -> float:
        """多空比信号转评分"""
        signal_map = {
            "EXTREME_LONG": 0.3,    # 极度偏多 -> 可能反转
            "VERY_LONG": 0.4,
            "BULLISH": 0.6,
            "NEUTRAL": 0.5,
            "BEARISH": 0.4,
            "EXTREME_SHORT": 0.7    # 极度偏空 -> 可能反转
        }
        return signal_map.get(signal, 0.5)
    
    def _generate_recommendation(self, signal: str, score: float) -> str:
        """生成操作建议"""
        recommendations = {
            "STRONG_BULLISH": "强烈看涨，可考虑加仓或入场",
            "BULLISH": "看涨，适合逢低买入",
            "NEUTRAL": "中性，观望为主",
            "BEARISH": "看跌，建议减仓或止盈",
            "STRONG_BEARISH": "强烈看跌，建议避险或做空"
        }
        
        base_rec = recommendations.get(signal, "信号不明确，谨慎操作")
        confidence_note = f"（置信度: {score:.1%}）"
        
        return base_rec + confidence_note