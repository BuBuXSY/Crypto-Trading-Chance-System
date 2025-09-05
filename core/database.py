# core/database.py
"""
数据库管理模块
处理所有数据库操作
"""

import sqlite3
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from core.logger import get_logger
from data.models import Signal

logger = get_logger(__name__)

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, signal_db_path: str = "signal_history.db", 
                 whale_db_path: str = "whale_wallets.db"):
        """
        初始化数据库管理器
        
        Args:
            signal_db_path: 信号数据库路径
            whale_db_path: 鲸鱼数据库路径
        """
        self.signal_db_path = signal_db_path
        self.whale_db_path = whale_db_path
        self.lock = threading.Lock()
        self.setup_databases()
    
    def setup_databases(self):
        """初始化数据库表结构"""
        # 信号历史数据库
        self.signal_db = sqlite3.connect(self.signal_db_path, check_same_thread=False)
        cursor = self.signal_db.cursor()
        
        # 创建信号表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                entry REAL,
                sl REAL,
                tps TEXT,
                score REAL,
                confidence REAL,
                quality TEXT,
                reason TEXT,
                predicted_outcome REAL,
                actual_outcome REAL,
                success INTEGER,
                checked INTEGER,
                ai_confidence REAL,
                pattern_detected TEXT
            )
        ''')
        
        # 创建模型性能表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                symbol TEXT,
                timestamp TEXT,
                accuracy REAL,
                mse REAL,
                mae REAL,
                r2_score REAL,
                parameters TEXT
            )
        ''')
        self.signal_db.commit()
        
        # 鲸鱼钱包数据库
        self.whale_db = sqlite3.connect(self.whale_db_path, check_same_thread=False)
        cursor = self.whale_db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS whale_wallets (
                address TEXT PRIMARY KEY,
                first_seen TEXT,
                last_seen TEXT,
                total_volume REAL,
                trade_count INTEGER,
                behavior_pattern TEXT
            )
        ''')
        self.whale_db.commit()
        
        logger.info("数据库初始化完成")
    
    def save_signal(self, signal: Signal):
        """
        保存交易信号
        
        Args:
            signal: 信号对象
        """
        with self.lock:
            try:
                cursor = self.signal_db.cursor()
                cursor.execute('''
                    INSERT INTO signals (timestamp, symbol, side, entry, sl, tps, score, 
                                        confidence, quality, reason, predicted_outcome, 
                                        actual_outcome, success, checked, ai_confidence, pattern_detected)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal.timestamp.isoformat(), signal.symbol, signal.side, signal.entry,
                    signal.sl, json.dumps(signal.tps), signal.score, signal.confidence,
                    signal.quality, signal.reason, signal.predicted_outcome,
                    signal.actual_outcome, signal.success, signal.checked,
                    signal.ai_confidence, signal.pattern_detected
                ))
                self.signal_db.commit()
                logger.debug(f"信号已保存: {signal.symbol} {signal.side}")
            except Exception as e:
                logger.error(f"保存信号失败: {e}")
    
    def save_model_performance(self, model_name: str, symbol: str, metrics: Dict):
        """
        保存模型性能指标
        
        Args:
            model_name: 模型名称
            symbol: 交易对
            metrics: 性能指标字典
        """
        with self.lock:
            try:
                cursor = self.signal_db.cursor()
                cursor.execute('''
                    INSERT INTO model_performance (model_name, symbol, timestamp, accuracy, mse, mae, r2_score, parameters)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_name, symbol, datetime.now().isoformat(),
                    metrics.get('accuracy', 0), metrics.get('mse', 0),
                    metrics.get('mae', 0), metrics.get('r2_score', 0),
                    json.dumps(metrics.get('parameters', {}))
                ))
                self.signal_db.commit()
            except Exception as e:
                logger.error(f"保存模型性能失败: {e}")
    
    def get_unchecked_signals(self, hours_ago: int = 24) -> List:
        """
        获取未检查的信号
        
        Args:
            hours_ago: 多少小时前的信号
        
        Returns:
            未检查的信号列表
        """
        with self.lock:
            try:
                cursor = self.signal_db.cursor()
                cutoff_time = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
                cursor.execute('''
                    SELECT * FROM signals 
                    WHERE checked = 0 AND timestamp < ?
                ''', (cutoff_time,))
                return cursor.fetchall()
            except Exception as e:
                logger.error(f"获取未检查信号失败: {e}")
                return []
    
    def update_signal_outcome(self, signal_id: int, actual_outcome: float, success: bool):
        """
        更新信号结果
        
        Args:
            signal_id: 信号ID
            actual_outcome: 实际结果
            success: 是否成功
        """
        with self.lock:
            try:
                cursor = self.signal_db.cursor()
                cursor.execute('''
                    UPDATE signals 
                    SET actual_outcome = ?, success = ?, checked = 1
                    WHERE id = ?
                ''', (actual_outcome, success, signal_id))
                self.signal_db.commit()
            except Exception as e:
                logger.error(f"更新信号结果失败: {e}")
    
    def get_signal_statistics(self, days: int = 30) -> List:
        """
        获取信号统计数据
        
        Args:
            days: 统计天数
        
        Returns:
            统计数据列表
        """
        with self.lock:
            try:
                cursor = self.signal_db.cursor()
                cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total,
                        SUM(success) as success_count,
                        AVG(confidence) as avg_confidence,
                        quality,
                        side
                    FROM signals 
                    WHERE checked = 1 AND timestamp > ?
                    GROUP BY quality, side
                ''', (cutoff_time,))
                return cursor.fetchall()
            except Exception as e:
                logger.error(f"获取信号统计失败: {e}")
                return []
    
    def save_whale_wallet(self, address: str, volume: float, pattern: str):
        """
        保存鲸鱼钱包信息
        
        Args:
            address: 钱包地址
            volume: 交易量
            pattern: 行为模式
        """
        with self.lock:
            try:
                cursor = self.whale_db.cursor()
                now = datetime.now().isoformat()
                
                # 尝试更新现有记录
                cursor.execute('''
                    UPDATE whale_wallets 
                    SET last_seen = ?, total_volume = total_volume + ?, trade_count = trade_count + 1
                    WHERE address = ?
                ''', (now, volume, address))
                
                if cursor.rowcount == 0:
                    # 如果不存在则插入新记录
                    cursor.execute('''
                        INSERT INTO whale_wallets (address, first_seen, last_seen, total_volume, trade_count, behavior_pattern)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (address, now, now, volume, 1, pattern))
                
                self.whale_db.commit()
            except Exception as e:
                logger.error(f"保存鲸鱼钱包失败: {e}")
    
    def close(self):
        """关闭数据库连接"""
        try:
            if hasattr(self, 'signal_db'):
                self.signal_db.close()
            if hasattr(self, 'whale_db'):
                self.whale_db.close()
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库失败: {e}")