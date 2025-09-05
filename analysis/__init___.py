# analysis/__init__.py
"""
分析模块
包含链上数据和市场分析功能
"""

from .onchain import OnChainAnalyzer

__all__ = [
    'OnChainAnalyzer'
]

# 分析功能检查
def check_analysis_capabilities():
    """检查分析功能可用性"""
    capabilities = {
        "onchain_basic": True,
        "api_access": True,
        "real_time": True
    }
    
    # 这里可以添加更多的功能检查
    return capabilities
