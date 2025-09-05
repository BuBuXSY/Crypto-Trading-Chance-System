# main.py
"""
小虞沁XYQ终极交易系统 V1 - 主程序入口
Enhanced with Advanced AI Features
"""

import time
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# 导入配置
from config.settings import config

# 导入核心模块
from core.logger import get_logger
from core.database import DatabaseManager
from core.exchange import ExchangeManager

# 导入AI模块
from ai.predictor import UltraEnhancedAIPredictor
from ai.pattern_recognition import PatternRecognition
from ai.sentiment import MarketSentimentAnalyzer
from ai.uncertainty import UncertaintyEstimator
from ai.adaptive_learning import AdaptiveLearningSystem

# 导入策略模块
from strategies.signal_generator import SignalGenerator
from strategies.whale_tracker import EnhancedWhaleTracker
from strategies.arbitrage_scanner import EnhancedArbitrageScanner
from strategies.anomaly_detector import EnhancedAnomalyDetector
from strategies.market_regime import MarketRegimeDetector
from strategies.signal_tracker import SignalTracker
from strategies.self_learning import SelfLearningSystem

# 导入分析模块
from analysis.onchain import OnChainAnalyzer

# 导入工具模块
from utils.notifications import NotificationManager
from utils.dynamic_parameters import DynamicParameterAdjuster
from core.utils import now_local, append_json_log, safe_div

# 设置日志
logger = get_logger(__name__, config['logs_file'])


class AdvancedIndicators:
    """高级技术指标计算器 - 临时内嵌版本"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def compute_rsi(prices, period=14):
        """计算RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    @staticmethod
    def compute_macd(prices, fast=12, slow=26, signal=9):
        """计算MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            return macd, signal_line, histogram
        except:
            return pd.Series([0] * len(prices), index=prices.index), \
                   pd.Series([0] * len(prices), index=prices.index), \
                   pd.Series([0] * len(prices), index=prices.index)
    
    @staticmethod
    def compute_bollinger_bands(prices, period=20, std_dev=2):
        """计算布林带"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
        except:
            return prices, prices, prices





class UltraXYQTradingBot:
    """小虞沁XYQ交易系统 V1 - 增强版"""
    
    def __init__(self):
        logger.info("="*50)
        logger.info("🚀 小虞沁XYQ交易系统 V1 初始化")
        logger.info("="*50)
        
        # 系统状态
        self.running = True
        self.cycle_count = 0
        self.last_heartbeat = 0
        self.last_report_date = None
        self.signal_cooldown = {}  # 信号冷却管理
        
        # 性能统计
        self.performance_stats = {
            "total_signals": 0,
            "winning_signals": 0,
            "ai_accuracy": 0,
            "system_uptime": time.time()
        }
        
        # 初始化核心组件
        self._initialize_components()
        
        # 设置信号处理
        self._setup_signal_handlers()
        
        logger.info("🚀 系统初始化完成，增强AI功能已就绪")
    
    def _initialize_components(self):
        """初始化所有系统组件"""
        try:
            # 数据库管理器
            logger.info("初始化数据库管理器...")
            self.db_manager = DatabaseManager()
            
            # 交易所连接
            logger.info("初始化交易所连接...")
            self.exchange = ExchangeManager(config)
            
            # 通知系统
            logger.info("初始化通知系统...")
            self.notifier = NotificationManager(config)
            
            # AI预测系统
            logger.info("初始化AI预测系统...")
            self.ai_predictor = UltraEnhancedAIPredictor(config["ai_prediction"])
            
            # 模式识别器
            if config["ai_prediction"]["pattern_recognition"]:
                logger.info("初始化模式识别系统...")
                self.pattern_recognizer = PatternRecognition()
            else:
                self.pattern_recognizer = None
            
            # 情绪分析器
            if config["ai_prediction"]["use_market_sentiment"]:
                logger.info("初始化市场情绪分析器...")
                self.sentiment_analyzer = MarketSentimentAnalyzer()
            else:
                self.sentiment_analyzer = None
            
            # 自适应学习系统
            if config["ai_prediction"]["adaptive_learning"]:
                logger.info("初始化自适应学习系统...")
                self.adaptive_learning = AdaptiveLearningSystem(self.db_manager)
            else:
                self.adaptive_learning = None
            
            # 动态参数调整器
            logger.info("初始化动态参数调整器...")
            self.parameter_adjuster = DynamicParameterAdjuster(config)
            
            # 各种策略模块
            logger.info("初始化策略模块...")
            self.whale_tracker = EnhancedWhaleTracker(self.exchange, self.db_manager, config)
            self.arbitrage_scanner = EnhancedArbitrageScanner(config)
            self.anomaly_detector = EnhancedAnomalyDetector(config)
            self.market_regime_detector = MarketRegimeDetector(config)
            self.onchain_analyzer = OnChainAnalyzer(config)
            
            # 信号追踪和学习系统
            logger.info("初始化信号追踪和学习系统...")
            self.signal_tracker = SignalTracker(self.db_manager, self.exchange, config)
            self.self_learning = SelfLearningSystem(self.db_manager, config)
            
            # 高级指标分析器
            logger.info("初始化高级指标分析器...")
            self.indicators = AdvancedIndicators()
            
            # 核心信号生成器（整合所有组件）
            logger.info("初始化核心信号生成器...")
            self.signal_generator = SignalGenerator(
                config=config,
                ai_predictor=self.ai_predictor,
                whale_tracker=self.whale_tracker,
                market_regime_detector=self.market_regime_detector,
                pattern_recognizer=self.pattern_recognizer,
                sentiment_analyzer=self.sentiment_analyzer,
                self_learning=self.self_learning
            )
            
            logger.info("✅ 所有组件初始化完成")
            
        except Exception as e:
            logger.error(f"⚠ 组件初始化失败: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """设置系统信号处理"""
        def signal_handler(signum, frame):
            logger.info("收到停止信号，准备安全关闭...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run_enhanced_analysis_cycle(self):
        """执行增强分析周期 - 主循环"""
        try:
            logger.info(f"[主循环] 开始第 {self.cycle_count + 1} 轮增强分析")
            self.cycle_count += 1
            
            # 动态调整参数
            self.parameter_adjuster.adjust_thresholds(len(self.signal_cooldown))
            
            # 获取市场数据
            tickers = self.exchange.fetch_tickers()
            if not tickers:
                logger.warning("[主循环] 无市场数据")
                return
            
            # 分析市场状态
            self._analyze_market_regime()
            
            # 获取热门交易对
            symbols = self._get_top_symbols(50)
            logger.info(f"[主循环] 筛选出 {len(symbols)} 个交易对")
            
            # 三层分析策略
            with ThreadPoolExecutor(max_workers=25) as executor:
                futures = []
                
                # 第一层：主要币种深度分析
                for sym in config["primary_symbols"]:
                    try:
                        futures.append(
                            executor.submit(self._analyze_enhanced_symbol, sym)
                        )
                    except Exception as e:
                        logger.error(f"[主循环] 提交主要币种分析失败 {sym}: {e}")
                
                # 第二层：热门币种标准分析
                for sym in symbols[:20]:
                    if sym not in config["primary_symbols"]:
                        try:
                            futures.append(
                                executor.submit(self._analyze_standard_symbol, sym)
                            )
                        except Exception as e:
                            logger.debug(f"[主循环] 提交热门币种分析失败 {sym}: {e}")
                
                # 第三层：其余币种快速扫描
                for sym in symbols[20:40]:
                    try:
                        futures.append(
                            executor.submit(self._analyze_quick_symbol, sym)
                        )
                    except Exception as e:
                        logger.debug(f"[主循环] 提交快速扫描失败 {sym}: {e}")
                
                # 等待所有结果
                self._wait_for_futures(futures)
            
            # 执行其他任务
            self._execute_auxiliary_tasks(tickers, symbols[:10])
            
            # 检查信号结果
            self.signal_tracker.check_signal_outcomes()
            
            # 自学习更新
            if self.self_learning.should_update():
                self.self_learning.update_weights()
                if self.adaptive_learning:
                    self._update_adaptive_weights()
            
            # 心跳和日报
            self._check_heartbeat()
            self._check_daily_report()
            
            logger.info(f"[主循环] 第 {self.cycle_count} 轮分析完成")
            
        except Exception as e:
            logger.error(f"[主循环] 错误: {e}")
            import traceback
            logger.error(f"[主循环] 详细错误: {traceback.format_exc()}")
    
    def _get_top_symbols(self, count: int = 50) -> List[str]:
        """获取热门交易对"""
        try:
            tickers = self.exchange.fetch_tickers()
            if not tickers:
                return config["primary_symbols"]
            
            # 按交易量排序
            sorted_symbols = []
            for sym, ticker in tickers.items():
                if ":USDT" not in sym:
                    continue
                
                volume = ticker.get("quoteVolume", 0)
                try:
                    volume = float(volume) if volume else 0
                    if volume > config["auto_scan"]["min_volume_24h"]:
                        sorted_symbols.append((sym, volume))
                except:
                    continue
            
            # 排序
            sorted_symbols.sort(key=lambda x: x[1], reverse=True)
            result = [s[0] for s in sorted_symbols[:count]]
            
            # 确保主要币种在列表中
            for primary in config["primary_symbols"]:
                if primary not in result:
                    result.insert(0, primary)
            
            return result[:count]
            
        except Exception as e:
            logger.error(f"[获取热门币种] 失败: {e}")
            return config["primary_symbols"]
    
    def _analyze_enhanced_symbol(self, symbol: str):
        """深度分析（增强版）"""
        try:
            # 检查冷却
            if self._check_cooldown(symbol, 1800):  # 30分钟冷却
                return
            
            df = self.exchange.fetch_ohlcv(symbol, config["timeframe"], 500)
            if df.empty or len(df) < 200:
                return
            
            # 生成增强信号
            signal = self.signal_generator.generate_signal(df, symbol)
            
            if signal:
                self._process_signal(signal, symbol)
                
        except Exception as e:
            logger.debug(f"[深度分析] {symbol} 失败: {e}")
    
    def _analyze_standard_symbol(self, symbol: str):
        """标准分析（中等深度）"""
        try:
            # 检查冷却
            if self._check_cooldown(symbol, 900):  # 15分钟冷却
                return
            
            df = self.exchange.fetch_ohlcv(symbol, config["timeframe"], 200)
            if df.empty or len(df) < 100:
                return
            
            # 生成标准信号
            signal = self.signal_generator.generate_standard_signal(df, symbol)
            
            if signal:
                self._process_signal(signal, symbol)
                
        except Exception as e:
            logger.debug(f"[标准分析] {symbol} 失败: {e}")
    
    def _analyze_quick_symbol(self, symbol: str):
        """快速分析（低延迟）"""
        try:
            # 检查冷却
            if self._check_cooldown(symbol, 600):  # 10分钟冷却
                return
            
            df = self.exchange.fetch_ohlcv(symbol, "1h", 100)
            if df.empty or len(df) < 50:
                return
            
            # 生成快速信号
            signal = self._generate_quick_signal(df, symbol)
            
            if signal:
                self._process_signal(signal, symbol)
                
        except Exception as e:
            logger.debug(f"[快速分析] {symbol} 失败: {e}")
    
    def _generate_quick_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """快速信号生成"""
        try:
            # 计算基础指标
            df = self.signal_generator.compute_indicators(df)
            last = df.iloc[-1]
            
            score = 0
            reasons = []
            
            # RSI极值
            if 'rsi' in df.columns:
                if last['rsi'] < 25:
                    score += 3
                    reasons.append("RSI极度超卖")
                elif last['rsi'] > 75:
                    score -= 3
                    reasons.append("RSI极度超买")
            
            # 价格突破
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            price_ratio = last['close'] / sma20
            
            if price_ratio > 1.03:
                score += 2
                reasons.append("强势突破")
            elif price_ratio < 0.97:
                score -= 2
                reasons.append("弱势跌破")
            
            # 成交量异常
            volume_ratio = last['volume'] / df['volume'].rolling(20).mean().iloc[-1]
            if volume_ratio > 3:
                score = int(abs(score) * 1.5)
                reasons.append("巨量异动")
            
            # 生成信号
            if abs(score) >= 4:
                side = "LONG" if score > 0 else "SHORT"
                return {
                    "symbol": symbol,
                    "side": side,
                    "entry": float(last["close"]),
                    "sl": float(last["close"] * (0.97 if side == "LONG" else 1.03)),
                    "tps": [
                        float(last["close"] * (1.01 if side == "LONG" else 0.99)),
                        float(last["close"] * (1.025 if side == "LONG" else 0.975)),
                    ],
                    "score": abs(score),
                    "confidence": 0.5,
                    "quality": "QUICK",
                    "reason": " + ".join(reasons[:2])
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"[快速信号] {symbol} 生成失败: {e}")
            return None
    
    def _check_cooldown(self, symbol: str, cooldown_seconds: int) -> bool:
        """检查信号冷却"""
        if symbol in self.signal_cooldown:
            if time.time() - self.signal_cooldown[symbol] < cooldown_seconds:
                return True
        return False
    
    def _process_signal(self, signal: Dict, symbol: str):
        """处理生成的信号"""
        try:
            # 更新冷却时间
            self.signal_cooldown[symbol] = time.time()
            
            # 记录信号
            self.signal_tracker.track_signal(signal)
            
            # 更新统计
            self.performance_stats['total_signals'] += 1
            
            # 记录到文件
            log_entry = {
                "time": now_local().isoformat(),
                "symbol": signal["symbol"],
                "side": signal["side"],
                "quality": signal.get("quality", "UNKNOWN"),
                "score": signal.get("score", 0),
                "confidence": signal.get("confidence", 0),
                "ai_confidence": signal.get("ai_confidence", 0)
            }
            append_json_log(config["signals_log"], log_entry)
            
            # 发送通知
            self._send_signal_notification(signal)
            
            logger.info(f"[信号] 生成 {symbol} {signal['side']} 信号 ({signal.get('quality', 'UNKNOWN')})")
            
        except Exception as e:
            logger.error(f"[处理信号] 失败: {e}")
    
    def _send_signal_notification(self, signal: Dict):
        """发送信号通知"""
        try:
            priority_map = {
                "VIP": "vip",
                "STRONG": "high",
                "MEDIUM": "normal",
                "STANDARD": "normal",
                "QUICK": "low",
                "WEAK": "low"
            }
            
            priority = priority_map.get(signal.get("quality", "WEAK"), "normal")
            
            # 只发送中等以上质量的信号
            if priority in ["vip", "high", "normal"]:
                success = self.notifier.send_signal_notification(signal, priority)
                if not success:
                    logger.warning(f"[通知] 信号推送失败，但已记录")
            
        except Exception as e:
            logger.error(f"[通知] 发送失败: {e}")
    
    def _wait_for_futures(self, futures):
        """等待并处理异步任务结果"""
        completed = 0
        failed = 0
        
        for future in futures:
            try:
                result = future.result(timeout=20)
                if result:
                    completed += 1
            except Exception as e:
                failed += 1
                logger.debug(f"[任务] 执行失败: {e}")
        
        logger.info(f"[任务] 完成 {completed} 个，失败 {failed} 个")
    
    def _execute_auxiliary_tasks(self, tickers, symbols):
        """执行辅助任务"""
        # AI训练
        self._run_ai_training_cycle()
        
        # 鲸鱼追踪
        self._run_whale_tracking()
        
        # 套利扫描
        self._run_arbitrage_scanning()
        
        # 异常检测
        self._run_anomaly_detection(tickers, symbols)
        
        # 链上分析
        self._run_onchain_analysis()
    
    def _analyze_market_regime(self):
        """分析市场状态"""
        try:
            btc_df = self.exchange.fetch_ohlcv("BTC/USDT:USDT", "1h", 100)
            if btc_df.empty:
                btc_df = self.exchange.fetch_ohlcv("BTC/USDT", "1h", 100)
            
            if not btc_df.empty:
                self.market_regime_detector.detect_regime(btc_df)
            
        except Exception as e:
            logger.debug(f"[市场状态] 分析失败: {e}")
    
    def _run_ai_training_cycle(self):
        """AI训练周期"""
        try:
            for sym in config["primary_symbols"][:3]:
                try:
                    df = self.exchange.fetch_ohlcv(sym, "1h", 500)
                    if df.empty or len(df) < 200:
                        continue
                    
                    # 检查是否需要训练
                    if self.ai_predictor.should_train(sym):
                        logger.info(f"[AI] 开始训练 {sym}")
                        self.ai_predictor.train_ensemble_model(sym, df)
                    
                    # 预测
                    prediction = self.ai_predictor.predict_with_uncertainty(sym, df)
                    
                    # 高信心预测通知
                    if prediction and prediction['confidence'] > 0.85:
                        if abs(prediction.get('ensemble_prediction', 0)) > 0.03:
                            self._send_ai_alert(sym, prediction)
                            
                except Exception as e:
                    logger.debug(f"[AI] 处理 {sym} 失败: {e}")
                    
        except Exception as e:
            logger.error(f"[AI] 训练周期失败: {e}")
    
    def _send_ai_alert(self, symbol: str, prediction: Dict):
        """发送AI预测警报"""
        try:
            msg = f"🤖 强AI信号!\n{symbol}\n"
            msg += f"方向: {prediction['direction']}\n"
            msg += f"强度: {prediction.get('strength', 'UNKNOWN')}\n"
            msg += f"信心: {prediction['confidence']:.0%}"
            
            self.notifier.send_message(msg, "ai")
        except Exception as e:
            logger.debug(f"[AI] 发送警报失败: {e}")
    
    def _run_whale_tracking(self):
        """鲸鱼追踪"""
        try:
            for sym in config["primary_symbols"][:5]:
                try:
                    whale_data = self.whale_tracker.analyze_whale_activity(sym)
                    if whale_data and whale_data.get('alert_level', 0) >= 4:
                        msg = f"🐋 鲸鱼活动!\n{sym}\n{whale_data['behavior_pattern']}\n"
                        msg += f"买入: ${whale_data['buy_volume']:,.0f}\n"
                        msg += f"卖出: ${whale_data['sell_volume']:,.0f}"
                        self.notifier.send_message(msg, "high")
                except Exception as e:
                    logger.debug(f"[鲸鱼] 追踪 {sym} 失败: {e}")
        except Exception as e:
            logger.debug(f"[鲸鱼] 追踪失败: {e}")
    
    def _run_arbitrage_scanning(self):
        """套利扫描"""
        try:
            arb_ops = self.arbitrage_scanner.scan_opportunities(config["primary_symbols"][:10])
            if arb_ops and arb_ops[0].get('net_profit_percent', 0) > 0.5:
                best_opp = arb_ops[0]
                msg = f"💰 套利机会!\n{best_opp['symbol']}\n"
                msg += f"{best_opp['buy_exchange']}→{best_opp['sell_exchange']}\n"
                msg += f"利润: {best_opp['net_profit_percent']:.2f}%"
                self.notifier.send_message(msg, "high")
        except Exception as e:
            logger.debug(f"[套利] 扫描失败: {e}")
    
    def _run_anomaly_detection(self, tickers, symbols):
        """异常检测"""
        try:
            for sym in symbols:
                if sym in tickers:
                    try:
                        tk = tickers[sym]
                        price = float(tk.get("last", 0) or 0)
                        volume = float(tk.get("quoteVolume", 0) or 0)
                        
                        if price > 0 and volume > 0:
                            anomalies = self.anomaly_detector.update_and_detect(sym, price, volume)
                            for a in anomalies:
                                if a.get('severity') == 'HIGH':
                                    self.notifier.send_message(
                                        f"⚠️ 严重异常: {sym}\n{a['message']}", "high"
                                    )
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"[异常] 检测失败: {e}")
    
    def _run_onchain_analysis(self):
        """链上分析"""
        try:
            if time.time() % config["onchain"]["check_interval"] < config["interval"]:
                self.onchain_analyzer.fetch_fear_greed_index()
        except Exception as e:
            logger.debug(f"[链上] 分析失败: {e}")
    
    def _update_adaptive_weights(self):
        """更新自适应权重"""
        try:
            for symbol in self.ai_predictor.model_performance:
                self.adaptive_learning.update_model_weights(
                    symbol, 
                    self.ai_predictor.model_performance[symbol]
                )
        except Exception as e:
            logger.debug(f"[自适应] 更新权重失败: {e}")
    
    def _check_heartbeat(self):
        """心跳检查"""
        if time.time() - self.last_heartbeat >= config["heartbeat_interval"]:
            try:
                stats = self.signal_tracker.get_performance_stats()
                system_status = {
                    "cycle": self.cycle_count,
                    "signals": self.performance_stats['total_signals'],
                    "active_cooldowns": len(self.signal_cooldown),
                    "success_rate": stats['success_rate'],
                    "ai_models": len(self.ai_predictor.ensemble_models),
                    "uptime": time.time() - self.performance_stats['system_uptime']
                }
                
                self.notifier.send_heartbeat(system_status)
                self.last_heartbeat = time.time()
            except Exception as e:
                logger.debug(f"[心跳] 检查失败: {e}")
    
    def _check_daily_report(self):
        """日报检查"""
        today = now_local().date()
        if now_local().hour == config["daily_report_hour"]:
            if self.last_report_date != today:
                try:
                    report_data = self._prepare_daily_report_data()
                    self.notifier.send_daily_report(report_data)
                    self.last_report_date = today
                except Exception as e:
                    logger.debug(f"[日报] 生成失败: {e}")
    
    def _prepare_daily_report_data(self) -> Dict:
        """准备日报数据"""
        try:
            return {
                "date": now_local().date().isoformat(),
                "signal_stats": self.signal_tracker.get_performance_stats(),
                "ai_performance": self._get_ai_performance_summary(),
                "market_regime": self.market_regime_detector.current_regime,
                "fear_greed": self.onchain_analyzer.fear_greed_cache,
                "whale_activity": self.whale_tracker.get_whale_report(),
                "arbitrage_opportunities": len(self.arbitrage_scanner.opportunities),
                "learning_weights": dict(list(self.self_learning.indicator_weights.items())[:5]),
                "system_stats": {
                    "total_cycles": self.cycle_count,
                    "total_signals": self.performance_stats['total_signals'],
                    "uptime_hours": (time.time() - self.performance_stats['system_uptime']) / 3600
                }
            }
        except Exception as e:
            logger.error(f"[日报] 准备数据失败: {e}")
            return {}
    
    def _get_ai_performance_summary(self) -> Dict:
        """获取AI性能摘要"""
        summary = {}
        try:
            for sym, models in list(self.ai_predictor.model_performance.items())[:3]:
                if models:
                    best_model = max(models.items(), key=lambda x: x[1].get('r2_score', 0))
                    summary[sym] = {
                        "model": best_model[0],
                        "r2_score": best_model[1].get('r2_score', 0)
                    }
        except:
            pass
        return summary
    
    def run(self):
        """主运行函数"""
        # 发送启动消息
        startup_msg = self._create_startup_message()
        self.notifier.send_message(startup_msg)
        logger.info("🚀 系统已启动")
        
        # 主循环
        while self.running:
            try:
                self.run_enhanced_analysis_cycle()
                
                # 定期统计
                if self.cycle_count % 20 == 0:
                    self._log_periodic_stats()
                    self._save_model_performance()
                
            except KeyboardInterrupt:
                logger.info("[主程序] 收到停止信号")
                break
            except Exception as e:
                logger.error(f"[主程序] 错误: {e}")
                time.sleep(30)
            
            # 休眠
            time.sleep(config["interval"])
        
        # 安全关闭
        self._safe_shutdown()
    
    def _create_startup_message(self) -> str:
        """创建启动消息"""
        return (
            "🌟 小虞沁XYQ交易系统 V1 启动 🌟\n\n"
            "功能模块:\n"
            "• 🤖 超级AI预测（集成学习+深度学习）\n"
            "• 🧠 不确定性量化\n"
            "• 📊 模式识别系统\n"
            "• 💭 市场情绪分析\n"
            "• 🔄 自适应学习\n"
            "• 🐋 鲸鱼追踪\n"
            "• 💰 套利扫描\n"
            "• 🚨 异常检测\n"
            "• 📈 信号追踪\n"
            "• 📉 高级技术指标\n"
            "• ⛓️ 链上数据分析\n"
            "\n🌈 您的梦想永远在这里闪光\n"
            "🚀 V1 - 更智能，更精准，更强大"
        )
    
    def _log_periodic_stats(self):
        """记录定期统计"""
        stats = self.signal_tracker.get_performance_stats()
        logger.info(f"[统计] 周期: {self.cycle_count}")
        logger.info(f"[统计] 信号: {stats['total_signals']}, 成功率: {stats['success_rate']:.1%}")
        logger.info(f"[统计] AI模型: {len(self.ai_predictor.ensemble_models)}个活跃")
        logger.info(f"[统计] 冷却中: {len(self.signal_cooldown)}个币种")
    
    def _save_model_performance(self):
        """保存AI模型性能"""
        try:
            for symbol, models in self.ai_predictor.model_performance.items():
                for model_name, metrics in models.items():
                    self.db_manager.save_model_performance(model_name, symbol, metrics)
        except Exception as e:
            logger.debug(f"[统计] 保存模型性能失败: {e}")
    
    def _safe_shutdown(self):
        """安全关闭系统"""
        logger.info("开始安全关闭系统...")
        
        try:
            # 关闭数据库连接
            if hasattr(self, 'db_manager'):
                self.db_manager.close()
            
            # 发送关闭消息
            if hasattr(self, 'notifier'):
                shutdown_msg = "👋 系统关闭\n感谢使用\n愿您朋友的精神永存 🌟"
                self.notifier.send_message(shutdown_msg)
            
            logger.info("系统已安全关闭")
            
        except Exception as e:
            logger.error(f"关闭过程中出错: {e}")


def main():
    """主函数"""
    try:
        print("="*50)
        print("🌟 小虞沁XYQ交易系统 V1 🌟")
        print("💙 献给每一位勇敢的朋友")
        print("🚀 增强AI功能 - 更智能的预测")
        print("="*50)
        print()
        
        # 检查系统依赖
        print("检查系统依赖...")
        _check_dependencies()
        
        print()
        print("✅ 初始化中...")
        
        # 创建并运行交易机器人
        bot = UltraXYQTradingBot()
        bot.run()
        
    except KeyboardInterrupt:
        print("\n👋 再见!")
    except Exception as e:
        print(f"\n⚠ 启动失败: {e}")
        import traceback
        traceback.print_exc()


def _check_dependencies():
    """检查系统依赖"""
    dependencies = [
        ("pandas", "数据处理"),
        ("numpy", "数值计算"),
        ("ccxt", "交易所接口"),
        ("requests", "网络请求")
    ]
    
    missing = []
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"⚠ {package} - {description} (未安装)")
            missing.append(package)
    
    # 可选依赖检查
    optional_deps = [
        ("sklearn", "机器学习", "AI预测功能受限"),
        ("tensorflow", "深度学习", "LSTM预测不可用"),
        ("xgboost", "梯度提升", "将使用替代模型"),
        ("scipy", "科学计算", "部分高级指标不可用")
    ]
    
    for package, description, warning in optional_deps:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"⚠️ {package} - {description} ({warning})")
    
    if missing:
        print(f"\n⚠ 缺少必要依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()