# core/utils.py
"""
工具函数模块
包含各种辅助函数
"""

import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

def now_local(offset_hours: int = 8):
    """
    获取本地时间
    
    Args:
        offset_hours: 时区偏移小时数
    
    Returns:
        带时区的datetime对象
    """
    tz = timezone(timedelta(hours=offset_hours))
    return datetime.now(tz)

def safe_div(a, b):
    """
    安全除法，避免除零错误
    
    Args:
        a: 被除数
        b: 除数
    
    Returns:
        除法结果，如果除数为0则返回0.0
    """
    try:
        return a/b if b and b != 0 else 0.0
    except:
        return 0.0

def ensure_file(filepath: str, default_content: str = "[]"):
    """
    确保文件存在，如果不存在则创建
    
    Args:
        filepath: 文件路径
        default_content: 默认内容
    """
    try:
        if not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(default_content)
    except Exception as e:
        print(f"创建文件失败 {filepath}: {e}")

def append_json_log(filepath: str, entry: Dict[str, Any], max_entries: int = 2000):
    """
    追加JSON日志
    
    Args:
        filepath: JSON文件路径
        entry: 要追加的日志条目
        max_entries: 最大条目数
    """
    ensure_file(filepath)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        data = []
    
    data.append(entry)
    
    # 限制条目数量
    if len(data) > max_entries:
        data = data[-max_entries:]
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)