# config/settings.py
"""
交易系统配置文件
包含所有系统配置参数
"""

config = {
    # ============= 交易所配置 =============
    "exchange": "okx",
    "backup_exchanges": ["binance", "bybit", "gateio"],
    
    "primary_symbols": [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", 
        "BNB/USDT:USDT", "XRP/USDT:USDT", "ADA/USDT:USDT",
        "DOGE/USDT:USDT", "AVAX/USDT:USDT", "DOT/USDT:USDT",
        "LINK/USDT:USDT", "UNI/USDT:USDT"
    ],
    
    # ============= 自动扫描配置 =============
    "auto_scan": {
        "enabled": True,
        "min_volume_24h": 10_000_000,  
        "top_n": 20,
        "exclude_patterns": ["UP", "DOWN", "BULL", "BEAR"],
    },
    
    # ============= 时间配置 =============
    "timeframe": "1h",
    "interval": 300,
    "heartbeat_interval": 3600,
    "daily_report_hour": 0,
    "timezone_offset_hours": 8,
    
    # ============= 信号质量配置 =============
    "signal_quality": {
        "min_score_weak": 5,
        "min_score_medium": 7,
        "min_score_strong": 10,
        "min_score_vip": 13,
        "require_advantage": 1.2,
    },
    
    # 缩短冷却时间
    "signal_cooldown_seconds": 1800,  # 新增：30分钟冷却
        
    # ============= AI预测配置 =============
    "ai_prediction": {
        "enabled": True,
        "lookback_hours": 168,
        "predict_hours": 24,
        "min_confidence": 0.65,
        "retrain_interval": 21600,
        "ensemble_enabled": True,
        "deep_learning_enabled": True,
        "adaptive_learning": True,
        "use_market_sentiment": True,
        "pattern_recognition": True,
    },
    
    # ============= 高级AI配置 =============
    "advanced_ai": {
        "lstm_layers": 3,
        "lstm_units": 128,
        "dropout_rate": 0.2,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10,
        "min_delta": 0.001,
        "feature_engineering_v2": True,
        "quantile_regression": True,
        "uncertainty_threshold": 0.3,
        "ensemble_weights_optimization": True,
        "cross_validation_splits": 5,
    },
    
    # ============= 模式识别配置 =============
    "pattern_recognition": {
        "enabled": True,
        "patterns": ["head_shoulders", "double_top", "triangle", "flag", "wedge"],
        "min_pattern_score": 0.7,
        "lookback_candles": 100,
    },
    
    # ============= 鲸鱼追踪配置 =============
    "whale_tracking": {
        "enabled": True,
        "min_trade_usdt": 500_000,
        "alert_threshold": 3,
        "track_wallets": True,
        "whale_database": "whale_wallets.db",
    },
    
    # ============= 套利配置 =============
    "arbitrage": {
        "enabled": True,
        "min_profit_percent": 0.3,
        "consider_fees": True,
        "check_depth": True,
        "min_volume_usdt": 50000,
        "fee_rates": {
            "okx": 0.001,
            "binance": 0.001,
            "bybit": 0.001,
            "gateio": 0.001,
        }
    },
    
    # ============= 异常检测配置 =============
    "anomaly_detection": {
        "enabled": True,
        "use_ml": True,
        "volume_spike_threshold": 3.0,
        "price_spike_threshold": 0.03,
    },
    
    # ============= 市场状态配置 =============
    "market_regime": {
        "enabled": True,
        "lookback_periods": 50,
    },
    
    # ============= 信号追踪配置 =============
    "signal_tracking": {
        "enabled": True,
        "database": "signal_history.db",
        "check_after_hours": 24,
        "success_threshold": 0.01,
    },
    
    # ============= 自学习配置 =============
    "self_learning": {
        "enabled": True,
        "update_interval": 86400,
        "min_samples": 100,
        "weight_adjustment_rate": 0.1,
        "performance_based_adjustment": True,
    },
    
    # ============= 链上数据配置 =============
    "onchain": {
        "enabled": True,
        "check_interval": 1800,
        "fear_greed_api": "https://api.alternative.me/fng/",
    },
    
    # ============= 高级指标配置 =============
    "advanced_indicators": {
        "ichimoku": True,
        "elliott_wave": True,
        "wyckoff": True,
        "market_profile": True,
    },
    
    # ============= 日志文件配置 =============
    "logs_file": "trading_xyq_ultimate.log",
    "signals_log": "signals_xyq_ultimate.json",
}