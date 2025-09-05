# strategies/signal_generator.py
"""
核心信号生成模块
整合所有指标和AI预测生成交易信号
"""

import time
import pandas as pd
from typing import Optional, Dict, List
from core.logger import get_logger
from core.utils import safe_div, now_local
from indicators.technical import TechnicalIndicators
from indicators.advanced import AdvancedIndicators

logger = get_logger(__name__)

class SignalGenerator:
    """核心信号生成器"""
    
    def __init__(self, config: Dict, ai_predictor=None, whale_tracker=None, 
                 market_regime_detector=None, pattern_recognizer=None,
                 sentiment_analyzer=None, self_learning=None):
        """
        初始化信号生成器
        
        Args:
            config: 配置字典
            ai_predictor: AI预测器
            whale_tracker: 鲸鱼追踪器
            market_regime_detector: 市场状态检测器
            pattern_recognizer: 模式识别器
            sentiment_analyzer: 情绪分析器
            self_learning: 自学习系统
        """
        logger.info("初始化核心信号生成器")
        self.config = config
        self.ai_predictor = ai_predictor
        self.whale_tracker = whale_tracker
        self.market_regime_detector = market_regime_detector
        self.pattern_recognizer = pattern_recognizer
        self.sentiment_analyzer = sentiment_analyzer
        self.self_learning = self_learning
        
        # 信号冷却期管理
        self.signal_cooldown = {}
        
        # 统计数据
        self.generated_signals = 0
        self.signal_quality_stats = {"VIP": 0, "STRONG": 0, "MEDIUM": 0, "WEAK": 0}
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        生成交易信号 - 核心方法
        
        Args:
            df: K线数据DataFrame
            symbol: 交易对符号
        
        Returns:
            信号字典或None
        """
        try:
            # 数据验证
            if df.empty or len(df) < 200:
                logger.debug(f"{symbol} 数据不足: {len(df) if not df.empty else 0}")
                return None
            
            # 检查冷却期
            if not self._check_cooldown(symbol):
                return None
            
            # 计算技术指标
            df = self._compute_indicators(df)
            
            # 获取当前数据
            last = df.iloc[-1]
            
            # 获取权重
            weights = self._get_indicator_weights()
            
            # 计算各项评分
            bull_score, bear_score, reasons, confidence_factors = self._calculate_comprehensive_scores(
                df, symbol, last, weights
            )
            
            # 生成最终信号
            signal = self._create_final_signal(
                symbol, bull_score, bear_score, reasons, confidence_factors, last
            )
            
            if signal:
                # 更新冷却期
                self.signal_cooldown[symbol] = time.time()
                self.generated_signals += 1
                self.signal_quality_stats[signal['quality']] += 1
                
                logger.info(f"生成{signal['quality']}信号: {symbol} {signal['side']} "
                           f"(评分: {signal['score']:.1f}, AI置信: {signal.get('ai_confidence', 0):.0%})")
            
            return signal
            
        except Exception as e:
            logger.error(f"生成信号失败 {symbol}: {e}")
            return None
    
    def _check_cooldown(self, symbol: str) -> bool:
        """
        检查信号冷却期
        
        Args:
            symbol: 交易对符号
        
        Returns:
            是否可以生成信号
        """
        if symbol in self.signal_cooldown:
            time_since_last = time.time() - self.signal_cooldown[symbol]
            if time_since_last < 3600:  # 1小时冷却
                logger.debug(f"{symbol} 冷却中，距离上次: {time_since_last/60:.1f}分钟")
                return False
        
        return True
    
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: K线数据
        
        Returns:
            添加了指标的DataFrame
        """
        # 计算基础技术指标
        df = TechnicalIndicators.compute_all(df)
        
        # 计算高级指标
        if self.config["advanced_indicators"]["ichimoku"]:
            df = AdvancedIndicators.ichimoku(df)
        
        return df
    
    def _get_indicator_weights(self) -> Dict:
        """获取指标权重"""
        if self.self_learning:
            return self.self_learning.indicator_weights
        else:
            # 默认权重
            return {
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
    
    def _calculate_comprehensive_scores(self, df: pd.DataFrame, symbol: str, 
                                      last: pd.Series, weights: Dict) -> tuple:
        """
        计算综合评分
        
        Args:
            df: K线数据
            symbol: 交易对
            last: 最新数据
            weights: 指标权重
        
        Returns:
            (多头评分, 空头评分, 原因列表, 置信度因子列表)
        """
        bull_score = 0
        bear_score = 0
        reasons = []
        confidence_factors = []
        
        # === AI预测评分 ===
        ai_score, ai_reasons, ai_confidence = self._evaluate_ai_prediction(symbol, df, weights)
        bull_score += ai_score['bull']
        bear_score += ai_score['bear']
        reasons.extend(ai_reasons)
        if ai_confidence > 0:
            confidence_factors.append(ai_confidence)
        
        # === 鲸鱼活动评分 ===
        whale_score, whale_reasons, whale_confidence = self._evaluate_whale_activity(symbol, weights)
        bull_score += whale_score['bull']
        bear_score += whale_score['bear']
        reasons.extend(whale_reasons)
        if whale_confidence > 0:
            confidence_factors.append(whale_confidence)
        
        # === 市场状态评分 ===
        regime_score, regime_reasons, regime_confidence = self._evaluate_market_regime(weights)
        bull_score += regime_score['bull']
        bear_score += regime_score['bear']
        reasons.extend(regime_reasons)
        if regime_confidence > 0:
            confidence_factors.append(regime_confidence)
        
        # === 技术指标评分 ===
        tech_score, tech_reasons = self._evaluate_technical_indicators(last, df, weights)
        bull_score += tech_score['bull']
        bear_score += tech_score['bear']
        reasons.extend(tech_reasons)
        
        # === 成交量分析评分 ===
        volume_score, volume_reasons = self._evaluate_volume_analysis(last, weights)
        bull_score += volume_score['bull']
        bear_score += volume_score['bear']
        reasons.extend(volume_reasons)
        
        # === 高级指标评分 ===
        advanced_score, advanced_reasons = self._evaluate_advanced_indicators(df, weights)
        bull_score += advanced_score['bull']
        bear_score += advanced_score['bear']
        reasons.extend(advanced_reasons)
        
        return bull_score, bear_score, reasons, confidence_factors
    
    def _evaluate_ai_prediction(self, symbol: str, df: pd.DataFrame, weights: Dict) -> tuple:
        """评估AI预测"""
        if not self.ai_predictor:
            return {"bull": 0, "bear": 0}, [], 0
        
        try:
            ai_prediction = self.ai_predictor.predict_with_uncertainty(symbol, df)
            if not ai_prediction:
                return {"bull": 0, "bear": 0}, [], 0
            
            weight = weights.get("ai_prediction", 1.0)
            ai_conf = ai_prediction['confidence']
            
            if ai_conf < self.config["ai_prediction"]["min_confidence"]:
                return {"bull": 0, "bear": 0}, [], 0
            
            # 计算强度倍数
            strength_map = {
                "VERY_STRONG": 4,
                "STRONG": 3,
                "MODERATE": 2,
                "WEAK": 1,
                "VERY_WEAK": 0.5
            }
            strength_multiplier = strength_map.get(ai_prediction['strength'], 2)
            
            if ai_prediction['direction'] == "BULLISH":
                bull_score = strength_multiplier * weight * ai_conf
                bear_score = 0
                reason = f"AI看涨({ai_prediction['strength']},置信{ai_conf:.0%})"
            else:
                bull_score = 0
                bear_score = strength_multiplier * weight * ai_conf
                reason = f"AI看跌({ai_prediction['strength']},置信{ai_conf:.0%})"
            
            reasons = [reason]
            
            # 模式识别加成
            if ai_prediction.get('pattern_signal', 0) > 0.5:
                pattern_weight = weights.get("pattern_recognition", 1.0)
                if ai_prediction['direction'] == "BULLISH":
                    bull_score += 2 * pattern_weight
                    reasons.append("技术形态看涨")
                else:
                    bear_score += 2 * pattern_weight
                    reasons.append("技术形态看跌")
            
            # 市场情绪加成
            sentiment = ai_prediction.get('market_sentiment', 0)
            if abs(sentiment) > 0.3:
                sentiment_weight = weights.get("market_sentiment", 1.0)
                if sentiment > 0:
                    bull_score += sentiment * sentiment_weight
                    reasons.append("市场情绪积极")
                else:
                    bear_score += abs(sentiment) * sentiment_weight
                    reasons.append("市场情绪消极")
            
            return {"bull": bull_score, "bear": bear_score}, reasons, ai_conf
            
        except Exception as e:
            logger.debug(f"AI预测评估失败: {e}")
            return {"bull": 0, "bear": 0}, [], 0
    
    def _evaluate_whale_activity(self, symbol: str, weights: Dict) -> tuple:
        """评估鲸鱼活动"""
        if not self.whale_tracker:
            return {"bull": 0, "bear": 0}, [], 0
        
        try:
            whale_data = self.whale_tracker.whale_activities.get(symbol, [])
            if not whale_data:
                return {"bull": 0, "bear": 0}, [], 0
            
            recent_whale = whale_data[-1]
            weight = weights.get("whale_activity", 1.0)
            
            if recent_whale['behavior_pattern'] == "ACCUMULATION":
                return {"bull": 3 * weight, "bear": 0}, ["鲸鱼吸筹"], 0.75
            elif recent_whale['behavior_pattern'] == "DISTRIBUTION":
                return {"bull": 0, "bear": 3 * weight}, ["鲸鱼派发"], 0.75
            else:
                return {"bull": 0, "bear": 0}, [], 0
                
        except Exception as e:
            logger.debug(f"鲸鱼活动评估失败: {e}")
            return {"bull": 0, "bear": 0}, [], 0
    
    def _evaluate_market_regime(self, weights: Dict) -> tuple:
        """评估市场状态"""
        if not self.market_regime_detector or not self.market_regime_detector.current_regime:
            return {"bull": 0, "bear": 0}, [], 0
        
        try:
            regime = self.market_regime_detector.current_regime
            weight = weights.get("market_regime", 1.0)
            
            if regime.regime_type == "TRENDING":
                if regime.trend_direction == "UP":
                    bull_score = 2 * weight * regime.confidence
                    bear_score = 0
                    reason = "上升趋势"
                else:
                    bull_score = 0
                    bear_score = 2 * weight * regime.confidence
                    reason = "下降趋势"
                
                return {"bull": bull_score, "bear": bear_score}, [reason], regime.confidence
            else:
                return {"bull": 0, "bear": 0}, [], 0
                
        except Exception as e:
            logger.debug(f"市场状态评估失败: {e}")
            return {"bull": 0, "bear": 0}, [], 0
    
    def _evaluate_technical_indicators(self, last: pd.Series, df: pd.DataFrame, weights: Dict) -> tuple:
        """评估技术指标"""
        bull_score = 0
        bear_score = 0
        reasons = []
        weight = weights.get("technical_indicators", 1.0)
        
        try:
            # RSI
            rsi = last.get("rsi", 50)
            if rsi < 30:
                bull_score += 3 * weight
                reasons.append("RSI超卖")
            elif rsi > 70:
                bear_score += 3 * weight
                reasons.append("RSI超买")
            
            # MACD
            if "macd" in last.index and "signal" in last.index:
                macd = last["macd"]
                signal = last["signal"]
                prev_macd = df.iloc[-2]["macd"] if len(df) > 1 else macd
                prev_signal = df.iloc[-2]["signal"] if len(df) > 1 else signal
                
                if macd > signal and prev_macd <= prev_signal:
                    bull_score += 2 * weight
                    reasons.append("MACD金叉")
                elif macd < signal and prev_macd >= prev_signal:
                    bear_score += 2 * weight
                    reasons.append("MACD死叉")
            
            # 布林带
            if all(col in last.index for col in ["close", "bb_upper", "bb_lower"]):
                close = last["close"]
                bb_upper = last["bb_upper"]
                bb_lower = last["bb_lower"]
                
                if close < bb_lower:
                    bull_score += 2 * weight
                    reasons.append("跌破下轨")
                elif close > bb_upper:
                    bear_score += 2 * weight
                    reasons.append("突破上轨")
            
            # 移动平均线
            if all(col in last.index for col in ["close", "sma20", "sma50"]):
                close = last["close"]
                sma20 = last["sma20"]
                sma50 = last["sma50"]
                
                if close > sma20 > sma50:
                    bull_score += 1.5 * weight
                    reasons.append("均线多头排列")
                elif close < sma20 < sma50:
                    bear_score += 1.5 * weight
                    reasons.append("均线空头排列")
            
        except Exception as e:
            logger.debug(f"技术指标评估失败: {e}")
        
        return {"bull": bull_score, "bear": bear_score}, reasons
    
    def _evaluate_volume_analysis(self, last: pd.Series, weights: Dict) -> tuple:
        """评估成交量分析"""
        bull_score = 0
        bear_score = 0
        reasons = []
        weight = weights.get("volume_analysis", 1.0)
        
        try:
            if "volume_ratio" in last.index and "close" in last.index:
                volume_ratio = last["volume_ratio"]
                
                if volume_ratio > 2.0:
                    # 放量上涨或下跌需要结合价格变化
                    if hasattr(last, 'close') and len(last) > 1:
                        # 这里需要前一个价格来判断，简化处理
                        bull_score += 1.5 * weight
                        reasons.append("放量上涨")
        
        except Exception as e:
            logger.debug(f"成交量分析失败: {e}")
        
        return {"bull": bull_score, "bear": bear_score}, reasons
    
    def _evaluate_advanced_indicators(self, df: pd.DataFrame, weights: Dict) -> tuple:
        """评估高级指标"""
        bull_score = 0
        bear_score = 0
        reasons = []
        weight = weights.get("advanced_indicators", 1.0)
        
        try:
            # Wyckoff分析
            if self.config["advanced_indicators"]["wyckoff"]:
                wyckoff = AdvancedIndicators.wyckoff_phase(df)
                if wyckoff == "MARKUP":
                    bull_score += 2 * weight
                    reasons.append("Wyckoff上涨期")
                elif wyckoff == "MARKDOWN":
                    bear_score += 2 * weight
                    reasons.append("Wyckoff下跌期")
            
            # 一目均衡表
            if self.config["advanced_indicators"]["ichimoku"] and len(df) > 26:
                last = df.iloc[-1]
                if all(col in last.index for col in ["close", "senkou_span_a", "senkou_span_b"]):
                    close = last["close"]
                    span_a = last["senkou_span_a"]
                    span_b = last["senkou_span_b"]
                    
                    if close > max(span_a, span_b):
                        bull_score += 1.5 * weight
                        reasons.append("一目云层之上")
                    elif close < min(span_a, span_b):
                        bear_score += 1.5 * weight
                        reasons.append("一目云层之下")
        
        except Exception as e:
            logger.debug(f"高级指标评估失败: {e}")
        
        return {"bull": bull_score, "bear": bear_score}, reasons
    
    def _create_final_signal(self, symbol: str, bull_score: float, bear_score: float,
                           reasons: List[str], confidence_factors: List[float], 
                           last: pd.Series) -> Optional[Dict]:
        """
        创建最终信号
        
        Args:
            symbol: 交易对
            bull_score: 多头评分
            bear_score: 空头评分
            reasons: 原因列表
            confidence_factors: 置信度因子
            last: 最新数据
        
        Returns:
            信号字典或None
        """
        # 获取配置
        min_score = self.config["signal_quality"]["min_score_weak"]
        advantage_ratio = self.config["signal_quality"]["require_advantage"]
        
        # 计算综合置信度
        overall_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        
        # 判断信号质量
        max_score = max(bull_score, bear_score)
        if max_score >= self.config["signal_quality"]["min_score_vip"]:
            quality = "VIP"
        elif max_score >= self.config["signal_quality"]["min_score_strong"]:
            quality = "STRONG"
        elif max_score >= self.config["signal_quality"]["min_score_medium"]:
            quality = "MEDIUM"
        elif max_score >= min_score:
            quality = "WEAK"
        else:
            return None
        
        # 确定信号方向
        if bull_score >= min_score and bull_score > bear_score * advantage_ratio:
            side = "LONG"
            entry = float(last["close"])
            sl = float(last["close"] * 0.97)
            tps = [
                float(last["close"] * 1.01),
                float(last["close"] * 1.02),
                float(last["close"] * 1.05),
            ]
            score = bull_score
            
        elif bear_score >= min_score and bear_score > bull_score * advantage_ratio:
            side = "SHORT"
            entry = float(last["close"])
            sl = float(last["close"] * 1.03)
            tps = [
                float(last["close"] * 0.99),
                float(last["close"] * 0.98),
                float(last["close"] * 0.95),
            ]
            score = bear_score
        else:
            return None
        
        # 获取AI相关信息
        ai_confidence = 0
        pattern_detected = ""
        predicted_change = 0
        uncertainty = 0.5
        
        if self.ai_predictor and symbol in self.ai_predictor.predictions_cache:
            ai_pred = self.ai_predictor.predictions_cache[symbol]
            ai_confidence = ai_pred.get('confidence', 0)
            if 'patterns_detected' in ai_pred:
                patterns = [k for k, v in ai_pred['patterns_detected'].items() if v > 0.5]
                pattern_detected = ",".join(patterns)
            predicted_change = ai_pred.get('ensemble_prediction', 0)
            uncertainty = ai_pred.get('uncertainty', 0.5)
        
        signal = {
            "symbol": symbol,
            "side": side,
            "entry": entry,
            "sl": sl,
            "tps": tps,
            "score": score,
            "confidence": overall_confidence,
            "ai_confidence": ai_confidence,
            "quality": quality,
            "reason": " + ".join(reasons[:5]),  # 限制原因数量
            "predicted_change": predicted_change,
            "pattern_detected": pattern_detected,
            "uncertainty": uncertainty,
            "timestamp": now_local().isoformat()
        }
        
        return signal
    
    def get_generation_stats(self) -> Dict:
        """获取信号生成统计"""
        return {
            "total_generated": self.generated_signals,
            "quality_distribution": self.signal_quality_stats.copy(),
            "cooldown_count": len(self.signal_cooldown),
            "avg_quality_score": self._calculate_avg_quality_score()
        }
    
    def _calculate_avg_quality_score(self) -> float:
        """计算平均质量分数"""
        quality_scores = {"VIP": 4, "STRONG": 3, "MEDIUM": 2, "WEAK": 1}
        total_score = 0
        total_count = 0
        
        for quality, count in self.signal_quality_stats.items():
            total_score += quality_scores[quality] * count
            total_count += count
        
        return total_score / total_count if total_count > 0 else 0
    
    def reset_cooldowns(self):
        """重置所有冷却期"""
        self.signal_cooldown.clear()
        logger.info("已重置所有信号冷却期")
    
    def force_signal_generation(self, symbol: str) -> bool:
        """
        强制允许生成信号（忽略冷却期）
        
        Args:
            symbol: 交易对符号
        
        Returns:
            是否成功重置
        """
        if symbol in self.signal_cooldown:
            del self.signal_cooldown[symbol]
            logger.info(f"已重置 {symbol} 的信号冷却期")
            return True
        return False