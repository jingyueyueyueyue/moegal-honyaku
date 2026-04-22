"""
MIT 渲染模块工具函数适配层
"""

import os
from pathlib import Path

# 基础路径
BASE_PATH = Path(__file__).parent.parent.parent

# 字体目录
FONTS_DIR = BASE_PATH / "fonts"

# 确保字体目录存在
FONTS_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str):
    """获取日志记录器"""
    from loguru import logger
    return logger.bind(name=name)
