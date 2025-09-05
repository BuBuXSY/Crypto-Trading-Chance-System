# utils/notifications.py
"""
消息通知模块
处理Telegram、邮件等各种通知方式
"""

import os
import requests
import time
from typing import Dict, List, Optional
from core.logger import get_logger
from core.utils import append_json_log, now_local

logger = get_logger(__name__)

class NotificationManager:
    """通知管理器"""
    
    def __init__(self, config: Dict):
        logger.info("初始化通知系统")
        self.config = config
        self.telegram_config = self._load_telegram_config()
        self.failed_messages = []
        self.send_statistics = {
            "total_sent": 0,
            "successful": 0,
            "failed": 0
        }
    
    def _load_telegram_config(self) -> Dict:
        """加载Telegram配置"""
        return {
            "token": os.getenv("TELEGRAM_TOKEN", ""),
            "chat_id": os.getenv("TELEGRAM_CHAT", ""),
            "enabled": bool(os.getenv("TELEGRAM_TOKEN") and os.getenv("TELEGRAM_CHAT"))
        }
    
    def send_signal_notification(self, signal: Dict, priority: str = "normal") -> bool:
        """
        发送信号通知
        
        Args:
            signal: 信号数据
            priority: 优先级 (vip, high, normal, low)
        
        Returns:
            是否发送成功
        """
        message = self._format_signal_message(signal)
        return self.send_message(message, priority)
    
    def send_message(self, message: str, priority: str = "normal") -> bool:
        """
        发送消息
        
        Args:
            message: 消息内容
            priority: 优先级
        
        Returns:
            是否发送成功
        """
        # 添加优先级标识
        formatted_message = self._add_priority_prefix(message, priority)
        
        # 记录发送统计
        self.send_statistics["total_sent"] += 1
        
        # 尝试发送
        success = self._send_telegram_message(formatted_message)
        
        if success:
            self.send_statistics["successful"] += 1
            logger.info(f"消息发送成功 (优先级: {priority})")
        else:
            self.send_statistics["failed"] += 1
            logger.error(f"消息发送失败 (优先级: {priority})")
            self._save_failed_message(formatted_message, priority)
        
        return success
    
    def _send_telegram_message(self, message: str) -> bool:
        """
        发送Telegram消息
        
        Args:
            message: 消息内容
        
        Returns:
            是否发送成功
        """
        if not self.telegram_config["enabled"]:
            logger.warning("Telegram配置缺失，请设置TELEGRAM_TOKEN和TELEGRAM_CHAT环境变量")
            return False
        
        url = f"https://api.telegram.org/bot{self.telegram_config['token']}/sendMessage"
        data = {
            "chat_id": self.telegram_config["chat_id"],
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, data=data, timeout=10)
                
                if response.status_code == 200:
                    return True
                elif response.status_code == 429:
                    # 速率限制，等待后重试
                    retry_after = response.json().get("retry_after", 60)
                    logger.warning(f"Telegram速率限制，等待{retry_after}秒")
                    time.sleep(retry_after)
                else:
                    logger.error(f"Telegram API错误: {response.status_code} - {response.text}")
                    break
                    
            except requests.exceptions.Timeout:
                logger.error(f"Telegram请求超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(5)
            except Exception as e:
                logger.error(f"Telegram发送失败: {e}")
                break
        
        return False
    
    def _add_priority_prefix(self, message: str, priority: str) -> str:
        """
        添加优先级前缀
        
        Args:
            message: 原始消息
            priority: 优先级
        
        Returns:
            添加了前缀的消息
        """
        prefixes = {
            "vip": "🌟⭐ VIP信号 ⭐🌟\n",
            "high": "🔴 重要提醒\n",
            "normal": "",
            "low": "ℹ️ 信息\n",
            "ai": "🤖 AI预测\n"
        }
        
        prefix = prefixes.get(priority, "")
        return prefix + message
    
    def _format_signal_message(self, signal: Dict) -> str:
        """
        格式化信号消息
        
        Args:
            signal: 信号数据
        
        Returns:
            格式化的消息
        """
        line = "=" * 30
        message = f"{line}\n"
        message += f"📢 小虞沁XYQ信号 V1 ({signal['quality']})\n"
        message += f"{line}\n\n"
        message += f"币种: {signal['symbol']}\n"
        message += f"方向: {'🚀多头' if signal['side'] == 'LONG' else '🐻空头'}\n"
        message += f"质量: {signal['quality']}\n"
        message += f"评分: {signal['score']:.1f}\n"
        message += f"置信度: {signal['confidence']:.1%}\n"
        
        # AI相关信息
        if signal.get('ai_confidence', 0) > 0:
            message += f"AI置信度: {signal['ai_confidence']:.1%}\n"
        
        if signal.get('uncertainty'):
            uncertainty_level = '低' if signal['uncertainty'] < 0.3 else '中' if signal['uncertainty'] < 0.6 else '高'
            message += f"不确定性: {uncertainty_level}\n"
        
        if signal.get('pattern_detected'):
            message += f"检测形态: {signal['pattern_detected']}\n"
        
        # 交易信息
        message += f"\n进场: ${signal['entry']:.6f}\n"
        message += f"止损: ${signal['sl']:.6f}\n"
        message += f"止盈:\n"
        for i, tp in enumerate(signal['tps'], 1):
            message += f"  TP{i}: ${tp:.6f}\n"
        
        message += f"\n理由: {signal['reason']}\n"
        message += f"\n⏰ {now_local().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"{line}\n"
        message += "⚠️ 交易有风险，入市需谨慎"
        
        return message
    
    def _save_failed_message(self, message: str, priority: str):
        """
        保存发送失败的消息
        
        Args:
            message: 消息内容
            priority: 优先级
        """
        try:
            failed_msg = {
                "timestamp": now_local().isoformat(),
                "priority": priority,
                "message": message,
                "length": len(message)
            }
            append_json_log("failed_messages.json", failed_msg)
            self.failed_messages.append(failed_msg)
            
            # 限制内存中的失败消息数量
            if len(self.failed_messages) > 100:
                self.failed_messages = self.failed_messages[-100:]
                
            logger.info("失败消息已保存到文件")
        except Exception as e:
            logger.error(f"保存失败消息出错: {e}")
    
    def send_daily_report(self, report_data: Dict) -> bool:
        """
        发送日报
        
        Args:
            report_data: 报告数据
        
        Returns:
            是否发送成功
        """
        message = self._format_daily_report(report_data)
        return self.send_message(message, "high")
    
    def _format_daily_report(self, data: Dict) -> str:
        """
        格式化日报消息
        
        Args:
            data: 报告数据
        
        Returns:
            格式化的日报消息
        """
        line = "=" * 35
        report = f"{line}\n"
        report += f"📊 小虞沁XYQ每日报告\n"
        report += f"📅 {now_local().strftime('%Y-%m-%d')}\n"
        report += f"{line}\n\n"
        
        # 信号统计
        if 'signal_stats' in data:
            stats = data['signal_stats']
            report += f"📈 信号统计\n"
            report += f"总信号: {stats.get('total_signals', 0)}\n"
            report += f"成功率: {stats.get('success_rate', 0):.1%}\n\n"
        
        # AI性能
        if 'ai_performance' in data:
            ai_perf = data['ai_performance']
            report += f"🤖 AI模型性能\n"
            for symbol, performance in list(ai_perf.items())[:3]:
                report += f"{symbol}: R² {performance:.3f}\n"
            report += "\n"
        
        # 市场状态
        if 'market_regime' in data:
            regime = data['market_regime']
            report += f"🌍 市场状态\n"
            report += f"类型: {regime.get('regime_type', '未知')}\n"
            report += f"趋势: {regime.get('trend_direction', '未知')}\n"
            report += f"波动: {regime.get('volatility_level', '未知')}\n\n"
        
        # 恐惧贪婪指数
        if 'fear_greed' in data:
            fg = data['fear_greed']
            report += f"😱 恐惧贪婪指数\n"
            report += f"{fg.get('value', 50)}/100 - {fg.get('classification', '未知')}\n\n"
        
        # 鲸鱼活动
        if 'whale_activity' in data:
            whale = data['whale_activity']
            report += f"🐋 鲸鱼活动\n"
            report += f"活跃币种: {whale.get('active_symbols', 0)}个\n"
            report += f"总交易量: ${whale.get('total_volume', 0):,.0f}\n\n"
        
        # 套利机会
        if 'arbitrage_opportunities' in data:
            arb = data['arbitrage_opportunities']
            if arb:
                report += f"💰 套利机会TOP3\n"
                for opp in arb[:3]:
                    report += f"• {opp['symbol']}: {opp['buy_exchange']}→{opp['sell_exchange']} (+{opp['net_profit_percent']:.2f}%)\n"
                report += "\n"
        
        # 学习系统状态
        if 'learning_system' in data:
            learning = data['learning_system']
            report += f"🧠 自学习系统\n"
            top_weights = sorted(learning.items(), key=lambda x: x[1], reverse=True)[:3]
            for indicator, weight in top_weights:
                report += f"• {indicator}: {weight:.2f}\n"
        
        report += f"\n{line}\n"
        report += f"💙 献给一位勇敢的朋友\n"
        report += f"🚀 Version1 更智能，更精准\n"
        report += f"{line}\n"
        
        return report
    
    def send_heartbeat(self, system_status: Dict) -> bool:
        """
        发送心跳消息
        
        Args:
            system_status: 系统状态数据
        
        Returns:
            是否发送成功
        """
        message = f"💗 小虞沁XYQ V1运行中\n"
        message += f"信号: {system_status.get('signal_count', 0)}\n"
        message += f"成功率: {system_status.get('success_rate', 0):.1%}\n"
        
        # 添加AI状态
        if 'ai_models_active' in system_status:
            message += f"AI模型: {system_status['ai_models_active']}个活跃"
        
        return self.send_message(message, "low")
    
    def send_alert(self, alert_type: str, message: str, severity: str = "medium") -> bool:
        """
        发送警报消息
        
        Args:
            alert_type: 警报类型
            message: 警报内容
            severity: 严重程度 (low, medium, high)
        
        Returns:
            是否发送成功
        """
        priority_map = {
            "low": "normal",
            "medium": "high", 
            "high": "vip"
        }
        
        alert_message = f"🚨 {alert_type}警报\n{message}"
        return self.send_message(alert_message, priority_map.get(severity, "normal"))
    
    def retry_failed_messages(self) -> int:
        """
        重试发送失败的消息
        
        Returns:
            成功重发的消息数量
        """
        if not self.failed_messages:
            return 0
        
        success_count = 0
        remaining_failures = []
        
        for failed_msg in self.failed_messages:
            try:
                if self._send_telegram_message(failed_msg["message"]):
                    success_count += 1
                    logger.info(f"重发成功: {failed_msg['timestamp']}")
                else:
                    remaining_failures.append(failed_msg)
            except Exception as e:
                logger.error(f"重发失败: {e}")
                remaining_failures.append(failed_msg)
        
        self.failed_messages = remaining_failures
        logger.info(f"重发完成: {success_count}成功, {len(remaining_failures)}仍失败")
        
        return success_count
    
    def get_statistics(self) -> Dict:
        """
        获取发送统计
        
        Returns:
            统计数据
        """
        return {
            **self.send_statistics,
            "success_rate": (
                self.send_statistics["successful"] / self.send_statistics["total_sent"]
                if self.send_statistics["total_sent"] > 0 else 0
            ),
            "failed_messages_count": len(self.failed_messages),
            "telegram_enabled": self.telegram_config["enabled"]
        }
    
    def test_connection(self) -> Dict:
        """
        测试连接
        
        Returns:
            测试结果
        """
        test_message = f"🧪 测试消息 {now_local().strftime('%H:%M:%S')}"
        success = self._send_telegram_message(test_message)
        
        return {
            "telegram": {
                "enabled": self.telegram_config["enabled"],
                "test_success": success,
                "config_valid": bool(self.telegram_config["token"] and self.telegram_config["chat_id"])
            }
        }
    
    def clear_failed_messages(self):
        """清除失败消息记录"""
        self.failed_messages.clear()
        logger.info("已清除失败消息记录")
    
    def update_telegram_config(self, token: str = None, chat_id: str = None):
        """
        更新Telegram配置
        
        Args:
            token: Bot token
            chat_id: Chat ID
        """
        if token:
            self.telegram_config["token"] = token
            os.environ["TELEGRAM_TOKEN"] = token
        
        if chat_id:
            self.telegram_config["chat_id"] = chat_id
            os.environ["TELEGRAM_CHAT"] = chat_id
        
        self.telegram_config["enabled"] = bool(
            self.telegram_config["token"] and self.telegram_config["chat_id"]
        )
        
        logger.info("Telegram配置已更新")