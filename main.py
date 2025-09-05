# main.py
"""
小虞沁XYQ终极交易系统 V1 - 主程序入口

Enhanced with Advanced AI Features
"""

import time
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

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
from core.utils import now_local, append_json_log

# 设置日志
logger = get_logger(__name__, config['logs_file'])

class UltraXYQTradingBot:
    """小虞沁XYQ交易系统 V1 """
    
    def __init__(self):
        logger.info("="*50)
        logger.info("🚀 小虞沁XYQ交易系统 V1 初始化")
        logger.info("="*50)
        
        # 系统状态
        self.running = True
        self.cycle_count = 0
        self.last_heartbeat = 0
        self.last_report_date = None
        
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
            logger.error(f"❌ 组件初始化失败: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """设置系统信号处理"""
        def signal_handler(signum, frame):
            logger.info("收到停止信号，准备安全关闭...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run_enhanced_analysis_cycle(self):
        """执行增强分析周期"""
        try:
            self.cycle_count += 1
            logger.info(f"[主循环] 开始第{self.cycle_count}轮增强分析")
            
            # 获取市场数据
            tickers = self._fetch_tickers_with_retry()
            if not tickers:
                logger.warning("[主循环] 无市场数据，跳过本轮")
                return
            
            # 市场状态检测
            self._analyze_market_regime()
            
            # 筛选交易对
            symbols = self._filter_trading_symbols(tickers)
            logger.info(f"[主循环] 分析 {len(symbols)} 个标的")
            
            # 并行分析 - 减少并发数避免API限制
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 异常检测
                self._run_anomaly_detection(tickers, symbols[:10])
                
                # 分析每个币种
                futures = []
                for sym in symbols[:5]:  # 限制同时分析的数量
                    time.sleep(0.5)  # 添加延迟避免API限制
                    try:
                        future = executor.submit(self._analyze_enhanced_symbol, sym)
                        futures.append(future)
                    except Exception as e:
                        logger.debug(f"[主循环] 提交 {sym} 分析任务失败: {e}")
                        continue
                
                # 等待任务完成
                completed = 0
                for future in futures:
                    try:
                        future.result(timeout=60)
                        completed += 1
                    except Exception as e:
                        logger.debug(f"[主循环] 任务执行失败: {e}")
                        pass
                
                logger.info(f"[主循环] 完成 {completed}/{len(futures)} 个分析任务")
            
            # AI训练和预测
            self._run_ai_training_cycle()
            
            # 其他模块
            self._run_whale_tracking()
            self._run_arbitrage_scanning()
            self._run_onchain_analysis()
            
            # 信号验证
            try:
                self.signal_tracker.check_signal_outcomes()
            except Exception as e:
                logger.debug(f"[主循环] 信号验证失败: {e}")
            
            # 自学习更新
            if self.self_learning.should_update():
                try:
                    self.self_learning.update_weights()
                except Exception as e:
                    logger.debug(f"[主循环] 自学习更新失败: {e}")
            
            # 心跳和日报
            self._check_heartbeat()
            self._check_daily_report()
            
            logger.info("[主循环] 本轮增强分析完成")
            
        except Exception as e:
            logger.error(f"[主循环] 错误: {e}")
            import traceback
            logger.error(f"[主循环] 详细错误: {traceback.format_exc()}")
    
    def _fetch_tickers_with_retry(self, max_retries=3):
        """获取行情 - 带重试"""
        for attempt in range(max_retries):
            try:
                tickers = self.exchange.fetch_tickers()
                if tickers:
                    return tickers
            except Exception as e:
                logger.warning(f"[主系统] 获取行情失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
        
        logger.error("[主系统] 获取行情最终失败")
        return {}
    
    def _analyze_market_regime(self):
        """分析市场状态"""
        try:
            btc_df = self.exchange.fetch_ohlcv("BTC/USDT:USDT", "1h", 100)
            if btc_df.empty:
                btc_df = self.exchange.fetch_ohlcv("BTC/USDT", "1h", 100)
            
            if not btc_df.empty:
                self.market_regime_detector.detect_regime(btc_df)
            else:
                logger.warning("[主循环] 无法获取BTC数据进行市场分析")
        except Exception as e:
            logger.debug(f"[主循环] 市场状态分析失败: {e}")
    
    def _filter_trading_symbols(self, tickers):
        """筛选交易对"""
        symbols = []
        
        for sym, tk in tickers.items():
            try:
                if ":USDT" not in sym:
                    continue
                
                volume = tk.get("quoteVolume", 0) or 0
                
                try:
                    volume_float = float(volume) if volume else 0
                except (ValueError, TypeError):
                    continue
                
                if volume_float > config["auto_scan"]["min_volume_24h"]:
                    symbols.append(sym)
                    
            except Exception:
                continue
        
        # 添加主要币种
        for primary in config["primary_symbols"]:
            if primary not in symbols:
                symbols.append(primary)
        
        # 去重并限制数量
        symbols = list(dict.fromkeys(symbols))[:20]
        return symbols
    
    def _analyze_enhanced_symbol(self, symbol: str):
        """分析单个币种（增强版）"""
        try:
            df = self.exchange.fetch_ohlcv(symbol, config["timeframe"], 500)
            if df.empty:
                return
            
            # 生成信号
            signal = self.signal_generator.generate_signal(df, symbol)
            if not signal:
                logger.debug(f"[分析] {symbol} 未生成信号")
                return
            
            logger.info(f"[分析] {symbol} 生成{signal['quality']}信号: {signal['side']}")
            
            # 更新统计
            self.performance_stats['total_signals'] += 1
            
            # 记录信号
            log_entry = {
                "time": now_local().isoformat(),
                "symbol": signal["symbol"],
                "side": signal["side"],
                "quality": signal["quality"],
                "score": signal["score"],
                "confidence": signal["confidence"],
                "ai_confidence": signal.get("ai_confidence", 0),
                "pattern": signal.get("pattern_detected", ""),
                "uncertainty": signal.get("uncertainty", 0)
            }
            append_json_log(config["signals_log"], log_entry)
            
            # 发送通知
            self._send_signal_notification(signal)
            
        except Exception as e:
            logger.error(f"[分析] {symbol} 失败: {e}")
    
    def _send_signal_notification(self, signal: Dict):
        """发送信号通知"""
        try:
            priority_map = {
                "VIP": "vip",
                "STRONG": "high", 
                "MEDIUM": "normal",
                "WEAK": "normal"
            }
            
            priority = priority_map.get(signal["quality"], "normal")
            success = self.notifier.send_signal_notification(signal, priority)
            
            if not success:
                logger.warning(f"[分析] {signal['symbol']} 信号推送失败，但已记录到日志")
            
        except Exception as e:
            logger.error(f"[分析] 发送通知失败: {e}")
    
    def _run_ai_training_cycle(self):
        """AI训练周期"""
        try:
            for sym in config["primary_symbols"][:3]:
                try:
                    df = self.exchange.fetch_ohlcv(sym, "1h", 300)
                    if df.empty or len(df) < 100:
                        continue
                    
                    # 检查是否需要训练
                    should_train = (sym not in self.ai_predictor.last_train or 
                                  time.time() - self.ai_predictor.last_train.get(sym, 0) > 
                                  config["ai_prediction"]["retrain_interval"])
                    
                    if should_train:
                        logger.info(f"[AI] 开始训练 {sym}")
                        result = self.ai_predictor.train_ensemble_model(sym, df)
                        if result:
                            logger.info(f"[AI] {sym} 训练完成")
                    
                    # 预测
                    prediction = self.ai_predictor.predict_with_uncertainty(sym, df, 12)
                    
                    # 高信心预测通知
                    if prediction and prediction['confidence'] > 0.8:
                        if abs(prediction.get('ensemble_prediction', 0)) > 0.025:
                            self._send_ai_alert(sym, prediction)
                            
                except Exception as e:
                    logger.debug(f"[AI] 处理 {sym} 失败: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"[AI] AI训练周期失败: {e}")
    
    def _send_ai_alert(self, symbol: str, prediction: Dict):
        """发送AI预测警报"""
        try:
            msg = f"🤖 强AI信号!\n{symbol}\n"
            msg += f"方向: {prediction['direction']}\n"
            msg += f"强度: {prediction['strength']}\n"
            msg += f"信心: {prediction['confidence']:.0%}"
            self.notifier.send_message(msg, "ai")
        except Exception as e:
            logger.debug(f"[AI] 发送AI警报失败: {e}")
    
    def _run_whale_tracking(self):
        """鲸鱼追踪"""
        try:
            for sym in config["primary_symbols"][:3]:
                try:
                    whale_data = self.whale_tracker.analyze_whale_activity(sym)
                    if whale_data and whale_data.get('alert_level', 0) >= 3:
                        msg = f"🐋 鲸鱼活动!\n{sym}\n{whale_data['behavior_pattern']}\n"
                        msg += f"买入: ${whale_data['buy_volume']:,.0f}\n"
                        msg += f"卖出: ${whale_data['sell_volume']:,.0f}"
                        self.notifier.send_message(msg, "high")
                except Exception as e:
                    logger.debug(f"[主循环] 鲸鱼追踪 {sym} 失败: {e}")
        except Exception as e:
            logger.debug(f"[主循环] 鲸鱼追踪模块失败: {e}")
    
    def _run_arbitrage_scanning(self):
        """套利扫描"""
        try:
            arb_ops = self.arbitrage_scanner.scan_opportunities(config["primary_symbols"][:5])
            if arb_ops and len(arb_ops) > 0 and arb_ops[0].get('net_profit_percent', 0) > 0.5:
                msg = f"💰 套利机会!\n{arb_ops[0]['symbol']}\n"
                msg += f"{arb_ops[0]['buy_exchange']}→{arb_ops[0]['sell_exchange']}\n"
                msg += f"净利润: {arb_ops[0]['net_profit_percent']:.2f}%"
                self.notifier.send_message(msg, "high")
        except Exception as e:
            logger.debug(f"[主循环] 套利扫描失败: {e}")
    
    def _run_onchain_analysis(self):
        """链上分析"""
        try:
            if time.time() % config["onchain"]["check_interval"] < config["interval"]:
                self.onchain_analyzer.fetch_fear_greed_index()
        except Exception as e:
            logger.debug(f"[主循环] 链上分析失败: {e}")
    
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
            logger.debug(f"[主循环] 异常检测失败: {e}")
    
    def _check_heartbeat(self):
        """心跳检查"""
        if time.time() - self.last_heartbeat >= config["heartbeat_interval"]:
            try:
                stats = self.signal_tracker.get_performance_stats()
                system_status = {
                    "signal_count": len(self.signal_generator.signal_cooldown),
                    "success_rate": stats['success_rate'],
                    "ai_models_active": len(self.ai_predictor.ensemble_models)
                }
                
                self.notifier.send_heartbeat(system_status)
                self.last_heartbeat = time.time()
            except Exception as e:
                logger.debug(f"[主循环] 心跳检查失败: {e}")
    
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
                    logger.debug(f"[主循环] 日报生成失败: {e}")
    
    def _prepare_daily_report_data(self) -> Dict:
        """准备日报数据"""
        try:
            return {
                "signal_stats": self.signal_tracker.get_performance_stats(),
                "ai_performance": {
                    sym: max(models.items(), key=lambda x: x[1].get('r2_score', 0))[1].get('r2_score', 0)
                    for sym, models in list(self.ai_predictor.model_performance.items())[:3]
                },
                "market_regime": self.market_regime_detector.get_regime_summary(),
                "fear_greed": self.onchain_analyzer.fear_greed_cache,
                "whale_activity": self.whale_tracker.get_whale_report(),
                "arbitrage_opportunities": self.arbitrage_scanner.opportunities,
                "learning_system": self.self_learning.indicator_weights
            }
        except Exception as e:
            logger.error(f"[主循环] 准备日报数据失败: {e}")
            return {}
    
    def run(self):
        """主运行函数"""
        # 发送启动消息
        startup_msg = self._create_startup_message()
        self.notifier.send_message(startup_msg)
        logger.info("🚀 系统已启动 - Version 1")
        
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
            
            time.sleep(config["interval"])
        
        # 安全关闭
        self._safe_shutdown()
    
    def _create_startup_message(self) -> str:
        """创建启动消息"""
        return (
            "🌟 小虞沁XYQ交易系统 Version1 启动 🌟\n\n"
            "\n"
            "功能模块:\n"
            "\n"
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
            "\n"
            "🌈 您的梦想永远在这里闪光\n"
            "🚀 V1 - 更智能，更精准，更强大"
        )
    
    def _log_periodic_stats(self):
        """记录定期统计"""
        stats = self.signal_tracker.get_performance_stats()
        logger.info(f"[统计] 信号: {stats['total_signals']}, 成功率: {stats['success_rate']:.1%}")
        
        # AI模型状态
        ai_models = len(self.ai_predictor.ensemble_models)
        logger.info(f"[统计] AI模型: {ai_models}个活跃")
    
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
            self.db_manager.close()
            
            # 发送关闭消息
            shutdown_msg = "👋 系统关闭\n感谢使用 \n愿您朋友的精神永存 🌟"
            self.notifier.send_message(shutdown_msg)
            
            logger.info("系统已安全关闭")
            
        except Exception as e:
            logger.error(f"关闭过程中出错: {e}")

def main():
    """主函数"""
    try:
        print("="*50)
        print("🌟 小虞沁XYQ交易系统 Version 1 🌟")
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
        print("\n👋 再见！")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
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
            print(f"❌ {package} - {description} (未安装)")
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
        print(f"\n❌ 缺少必要依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()