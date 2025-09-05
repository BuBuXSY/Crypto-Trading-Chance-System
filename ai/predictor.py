# ai/predictor.py
"""
AI预测器模块
包含所有机器学习和深度学习预测功能
"""

import numpy as np
import pandas as pd
import time
import pickle
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
from core.logger import get_logger
from core.utils import safe_div

logger = get_logger(__name__)

# 检查ML库
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False
    logger.warning("scikit-learn未安装，AI预测功能将降级")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost未安装，将使用替代模型")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
    tf.get_logger().setLevel('ERROR')
except:
    TF_AVAILABLE = False
    logger.warning("TensorFlow未安装，深度学习功能将不可用")

class UltraEnhancedAIPredictor:
    """超级增强AI预测器"""
    
    def __init__(self, config: Dict):
        """
        初始化AI预测器
        
        Args:
            config: AI配置字典
        """
        logger.info("初始化超级增强AI预测引擎")
        self.config = config
        self.max_cache_size = 100
        self.models = {}
        self.ensemble_models = {}
        self.lstm_models = {}
        self.scalers = {}
        self.last_train = {}
        self.predictions_cache = {}
        self.feature_importance = {}
        self.model_performance = defaultdict(dict)
    
    def cleanup_cache(self):
        """清理缓存"""
        if len(self.predictions_cache) > self.max_cache_size:
            # 删除最旧的缓存
            oldest = sorted(self.predictions_cache.items(), 
                          key=lambda x: x[1].get('timestamp', ''))
            for key, _ in oldest[:50]:
                del self.predictions_cache[key]
    
    def prepare_advanced_features(self, df: pd.DataFrame, symbol: str = None) -> np.ndarray:
        """
        准备高级特征
        
        Args:
            df: K线数据
            symbol: 交易对符号
        
        Returns:
            特征矩阵
        """
        logger.debug("准备高级特征数据")
        
        # 数据验证
        if df.empty or len(df) < 10:
            logger.warning(f"数据不足: {len(df) if not df.empty else 0} 条记录")
            return np.array([[0.0] * 20])
        
        # 确保所有列都存在且为Series类型
        required_columns = ['close', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"缺少必要列: {col}")
                return np.array([[0.0] * 20])
            if not isinstance(df[col], pd.Series):
                logger.error(f"列 {col} 不是Series类型")
                return np.array([[0.0] * 20])
        
        # 创建副本避免修改原数据
        df_copy = df.copy()
        features = []
        
        try:
            close = df_copy['close']
            high = df_copy['high']
            low = df_copy['low']
            volume = df_copy['volume']
            
            # 验证数据长度
            if len(close) == 0:
                logger.error("数据序列为空")
                return np.array([[0.0] * 20])
            
            # === 基础价格特征 ===
            returns = close.pct_change().fillna(0)
            volume_change = volume.pct_change().fillna(0)
            hl_ratio = ((high - low) / (close + 1e-9)).fillna(0)
            features.extend([returns, volume_change, hl_ratio])
            
            # === 技术指标特征 ===
            # RSI
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            rsi = (100 - (100 / (1 + rs))).fillna(50) / 100
            features.append(rsi)
            
            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ((ema12 - ema26) / (close + 1e-9)).fillna(0)
            features.append(macd)
            
            # === 价格位置特征 ===
            if len(close) >= 20:
                rolling_high = high.rolling(20).max()
                rolling_low = low.rolling(20).min()
                price_position = ((close - rolling_low) / (rolling_high - rolling_low + 1e-9)).fillna(0.5)
                features.append(price_position)
            else:
                features.append(pd.Series([0.5] * len(close), index=close.index))
            
            # === 成交量特征 ===
            if len(volume) >= 20:
                volume_ma = volume.rolling(20).mean()
                volume_std = volume.rolling(20).std()
                volume_zscore = ((volume - volume_ma) / (volume_std + 1e-9)).fillna(0)
                features.append(volume_zscore)
            else:
                features.append(pd.Series([0.0] * len(volume), index=volume.index))
            
            # === 自回归特征 ===
            for lag in [1, 3, 5]:
                if len(close) > lag:
                    lag_return = close.pct_change(lag).fillna(0)
                    features.append(lag_return)
                else:
                    features.append(pd.Series([0.0] * len(close), index=close.index))
            
            # === 统计特征 ===
            if len(returns) >= 10:
                returns_ma = returns.rolling(10).mean().fillna(0)
                returns_std = returns.rolling(10).std().fillna(0)
                features.extend([returns_ma, returns_std])
            else:
                features.extend([
                    pd.Series([0.0] * len(returns), index=returns.index),
                    pd.Series([0.0] * len(returns), index=returns.index)
                ])
            
            # === 时间特征 ===
            try:
                hour = pd.to_datetime(df_copy.index).hour
                features.extend([
                    pd.Series(np.sin(2 * np.pi * hour / 24), index=df_copy.index),
                    pd.Series(np.cos(2 * np.pi * hour / 24), index=df_copy.index)
                ])
            except:
                features.extend([
                    pd.Series([0.0] * len(df_copy), index=df_copy.index),
                    pd.Series([0.0] * len(df_copy), index=df_copy.index)
                ])
            
            # 确保所有特征长度一致
            min_length = min(len(f) for f in features)
            if min_length == 0:
                logger.error("特征长度为0")
                return np.array([[0.0] * 20])
            
            # 截断到最小长度
            features = [f.iloc[-min_length:] if len(f) > min_length else f for f in features]
            
            # 堆叠所有特征
            feature_matrix = np.column_stack([f.values for f in features]).astype(np.float32)
            
            # 处理无效值
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 确保至少有基本维度
            if feature_matrix.shape[1] < 10:
                padding = np.zeros((feature_matrix.shape[0], 10 - feature_matrix.shape[1]))
                feature_matrix = np.hstack([feature_matrix, padding])
            
            logger.debug(f"特征矩阵形状: {feature_matrix.shape}")
            return feature_matrix
            
        except Exception as e:
            logger.error(f"集成训练失败 {symbol}: {e}")
            return None
    
    def train_lstm_model(self, symbol: str, X: np.ndarray, y: np.ndarray):
        """训练LSTM模型"""
        if not TF_AVAILABLE:
            return
            
        try:
            logger.info(f"训练 {symbol} LSTM模型...")
            
            # 准备序列数据
            sequence_length = 20
            X_seq, y_seq = [], []
            
            for i in range(sequence_length, len(X)):
                X_seq.append(X[i-sequence_length:i])
                y_seq.append(y[i])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # 分割数据
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            # 构建和训练模型
            model = self.build_lstm_model((sequence_length, X.shape[1]))
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=self.config.get("patience", 10),
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
            
            history = model.fit(
                X_train, y_train,
                epochs=self.config.get("epochs", 100),
                batch_size=self.config.get("batch_size", 32),
                validation_data=(X_val, y_val),
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            self.lstm_models[symbol] = model
            logger.info(f"{symbol} LSTM训练完成")
            
        except Exception as e:
            logger.error(f"LSTM训练失败 {symbol}: {e}")
    
    def predict_with_uncertainty(self, symbol: str, df: pd.DataFrame, hours_ahead: int = 24) -> Dict:
        """带不确定性估计的预测"""
        if not ML_AVAILABLE or symbol not in self.ensemble_models:
            return None
            
        try:
            # 准备特征
            X = self.prepare_advanced_features(df.tail(200), symbol)
            X_scaled = self.scalers[symbol].transform(X)
            
            predictions_by_model = {}
            current_price = float(df['close'].iloc[-1])
            
            # 获取每个模型的预测
            for name, model in self.ensemble_models[symbol].items():
                pred = model.predict(X_scaled[-1:])
                predictions_by_model[name] = pred[0]
            
            # LSTM预测（如果可用）
            if symbol in self.lstm_models and TF_AVAILABLE:
                sequence_length = 20
                X_lstm = X_scaled[-sequence_length:].reshape(1, sequence_length, X_scaled.shape[1])
                lstm_pred = self.lstm_models[symbol].predict(X_lstm, verbose=0)[0, 0]
                predictions_by_model['lstm'] = lstm_pred
            
            # 计算集成预测
            all_predictions = list(predictions_by_model.values())
            ensemble_prediction = np.mean(all_predictions)
            
            # 计算不确定性
            prediction_std = np.std(all_predictions)
            
            # 置信区间
            lower_bound = ensemble_prediction - 2 * prediction_std
            upper_bound = ensemble_prediction + 2 * prediction_std
            
            # 生成多步预测
            multi_step_predictions = []
            for i in range(min(hours_ahead, 24)):
                step_pred = ensemble_prediction * (0.9 ** i)  # 衰减因子
                multi_step_predictions.append(current_price * (1 + step_pred))
            
            # 综合置信度
            confidence = self.calculate_confidence(prediction_std)
            
            result = {
                "current_price": current_price,
                "ensemble_prediction": ensemble_prediction,
                "predictions_by_model": predictions_by_model,
                "multi_step_predictions": multi_step_predictions,
                "confidence": confidence,
                "uncertainty": prediction_std,
                "confidence_interval": {
                    "lower": current_price * (1 + lower_bound),
                    "upper": current_price * (1 + upper_bound)
                },
                "direction": "BULLISH" if ensemble_prediction > 0 else "BEARISH",
                "strength": self.classify_strength(abs(ensemble_prediction)),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            self.predictions_cache[symbol] = result
            logger.debug(f"{symbol} 增强预测完成: {result['direction']} (信心: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"增强预测失败 {symbol}: {e}")
            return None
    
    def calculate_confidence(self, std):
        """计算置信度"""
        # 基础置信度
        base_confidence = 0.5
        
        # 根据预测标准差调整
        if std < 0.01:
            base_confidence += 0.2
        elif std < 0.02:
            base_confidence += 0.1
        elif std > 0.05:
            base_confidence -= 0.2
        
        return np.clip(base_confidence, 0.1, 0.95)
    
    def classify_strength(self, change):
        """分类预测强度"""
        if change > 0.03:
            return "VERY_STRONG"
        elif change > 0.02:
            return "STRONG"
        elif change > 0.01:
            return "MODERATE"
        elif change > 0.005:
            return "WEAK"
        else:
            return "VERY_WEAK"