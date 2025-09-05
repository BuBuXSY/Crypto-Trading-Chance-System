# utils/__init__.py
"""
工具模块
包含通知和其他实用工具
"""

from .notifications import NotificationManager

__all__ = [
    'NotificationManager'
]

# 通知功能检查
def check_notification_capabilities():
    """检查通知功能可用性"""
    import os
    
    capabilities = {
        "telegram": bool(os.getenv("TELEGRAM_TOKEN") and os.getenv("TELEGRAM_CHAT")),
        "file_logging": True,
        "console_output": True
    }
    
    return capabilities