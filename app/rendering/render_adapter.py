"""
MIT 渲染模块适配接口

为 moegal 项目提供简化的渲染接口
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

from . import text_render
from .auto_linebreak import solve_no_br_layout


# 默认字体路径
DEFAULT_FONT_PATH = Path(__file__).parent.parent.parent / "fonts" / "LXGWWenKai-Regular.ttf"
FALLBACK_FONTS = [
    Path(__file__).parent.parent.parent / "fonts" / "msyh.ttc",
    Path(__file__).parent.parent.parent / "fonts" / "msgothic.ttc",
]


def ensure_font_available(font_path: str = None) -> str:
    """确保字体文件可用"""
    if font_path and os.path.exists(font_path):
        return font_path
    
    # 检查默认字体
    if DEFAULT_FONT_PATH.exists():
        return str(DEFAULT_FONT_PATH)
    
    # 检查备选字体
    for fallback in FALLBACK_FONTS:
        if fallback.exists():
            return str(fallback)
    
    # 返回空字符串，将使用 Qt 默认字体
    return ""


class TextRenderConfig:
    """渲染配置"""
    def __init__(
        self,
        font_path: str = None,
        stroke_width: float = 0.07,
        line_spacing: float = 1.0,
        letter_spacing: float = 1.0,
    ):
        self.font_path = ensure_font_available(font_path)
        self.stroke_width = stroke_width
        self.line_spacing = line_spacing
        self.letter_spacing = letter_spacing
        
        # 设置全局字体
        if self.font_path:
            text_render.set_font(self.font_path)


def calculate_font_size(
    text: str,
    box_width: int,
    box_height: int,
    is_horizontal: bool = True,
    config: TextRenderConfig = None,
    target_lang: str = "CHS",
    min_font_size: int = 8,
    max_font_size: int = 256,
) -> Tuple[int, str]:
    """
    计算适合气泡的字体大小和换行文本
    
    Args:
        text: 文本内容
        box_width: 气泡宽度
        box_height: 气泡高度
        is_horizontal: 是否横排
        config: 渲染配置
        target_lang: 目标语言
        min_font_size: 最小字体大小
        max_font_size: 最大字体大小
        
    Returns:
        (font_size, text_with_br): 字体大小和带换行的文本
    """
    if not text:
        return min_font_size, ""
    
    if config is None:
        config = TextRenderConfig()
    
    # 初始字体大小估算（基于文本长度和气泡尺寸）
    text_len = len(text)
    
    if is_horizontal:
        # 横排：字体大小主要基于高度
        # 估算行数：假设每行放 sqrt(text_len) 个字符
        import math
        chars_per_line_estimate = max(1, int(math.sqrt(text_len) * (box_width / box_height)))
        n_lines_estimate = max(1, (text_len + chars_per_line_estimate - 1) // chars_per_line_estimate)
        initial_font_size = int(box_height / n_lines_estimate * 0.85)
    else:
        # 竖排：字体大小主要基于宽度
        # 估算列数：假设每列放 box_height/font_size 个字符
        import math
        # 使用气泡的宽高比来估算合理的列数
        aspect_ratio = box_width / box_height
        cols_estimate = max(1, int(round(text_len * aspect_ratio / 3)))
        initial_font_size = int(box_width / cols_estimate * 0.85)
    
    # 确保初始字体大小在合理范围内，且不超过框的尺寸
    initial_font_size = max(min_font_size, min(initial_font_size, max_font_size, box_width, box_height))
    
    # 使用 MIT 的换行引擎计算最优布局
    result = solve_no_br_layout(
        text=text,
        horizontal=is_horizontal,
        seed_segments=0,  # 自动计算
        seed_font_size=initial_font_size,
        bubble_width=float(box_width),
        bubble_height=float(box_height),
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        line_spacing_multiplier=config.line_spacing,
        letter_spacing_multiplier=config.letter_spacing,
        target_lang=target_lang,
        config=None,
    )
    
    return result.font_size, result.text_with_br


def render_text(
    image: np.ndarray,
    text: str,
    bbox: Tuple[int, int, int, int],
    is_horizontal: bool = True,
    font_size: int = None,
    fg_color: Tuple[int, int, int] = (0, 0, 0),
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    config: TextRenderConfig = None,
    target_lang: str = "CHS",
) -> np.ndarray:
    """
    在图像上渲染文本
    
    Args:
        image: BGR 图像
        text: 文本内容
        bbox: 边界框 (x1, y1, x2, y2)
        is_horizontal: 是否横排
        font_size: 字体大小（None 则自动计算）
        fg_color: 前景色 (R, G, B)
        bg_color: 背景色/描边色 (R, G, B)
        config: 渲染配置
        target_lang: 目标语言
        
    Returns:
        渲染后的 BGR 图像
    """
    if not text:
        return image
    
    x1, y1, x2, y2 = map(int, bbox)
    box_width = x2 - x1
    box_height = y2 - y1
    
    if box_width <= 0 or box_height <= 0:
        return image
    
    if config is None:
        config = TextRenderConfig()
    
    # 计算字体大小和换行
    if font_size is None:
        font_size, text = calculate_font_size(
            text, box_width, box_height, is_horizontal, config, target_lang
        )
    
    # 渲染文本
    if is_horizontal:
        text_bitmap = text_render.put_text_horizontal(
            font_size=font_size,
            text=text,
            width=box_width,
            height=box_height,
            alignment='center',
            reversed_direction=False,
            fg=fg_color,
            bg=bg_color,
            lang=target_lang,
            hyphenate=True,
            line_spacing=int(config.line_spacing * 100),
            stroke_width=config.stroke_width,
            letter_spacing=config.letter_spacing,
        )
    else:
        text_bitmap = text_render.put_text_vertical(
            font_size=font_size,
            text=text,
            h=box_height,
            alignment='center',
            fg=fg_color,
            bg=bg_color,
            line_spacing=int(config.line_spacing * 100),
            stroke_width=config.stroke_width,
            letter_spacing=config.letter_spacing,
        )
    
    if text_bitmap is None:
        return image
    
    # 将文本叠加到图像上
    return _overlay_text_bitmap(image, text_bitmap, x1, y1, box_width, box_height)


def _overlay_text_bitmap(
    image: np.ndarray,
    text_bitmap: np.ndarray,
    x: int,
    y: int,
    box_width: int,
    box_height: int,
) -> np.ndarray:
    """
    将渲染好的文本位图叠加到图像上
    """
    result = image.copy()
    
    th, tw = text_bitmap.shape[:2]
    
    # 计算居中位置
    paste_x = x + (box_width - tw) // 2
    paste_y = y + (box_height - th) // 2
    
    # 确保不越界
    img_h, img_w = image.shape[:2]
    paste_x = max(0, paste_x)
    paste_y = max(0, paste_y)
    
    # 计算实际粘贴区域
    src_x = max(0, -paste_x)
    src_y = max(0, -paste_y)
    
    paste_x = min(paste_x, img_w - 1)
    paste_y = min(paste_y, img_h - 1)
    
    dst_w = min(tw - src_x, img_w - paste_x)
    dst_h = min(th - src_y, img_h - paste_y)
    
    if dst_w <= 0 or dst_h <= 0:
        return result
    
    # 提取 alpha 通道
    if text_bitmap.shape[2] == 4:
        alpha = text_bitmap[src_y:src_y+dst_h, src_x:src_x+dst_w, 3:4] / 255.0
        color = text_bitmap[src_y:src_y+dst_h, src_x:src_x+dst_w, :3]
    else:
        # 如果没有 alpha 通道，假设非黑色区域为文本
        alpha = np.any(text_bitmap[src_y:src_y+dst_h, src_x:src_x+dst_w] > 0, axis=2, keepdims=True).astype(np.float32)
        color = text_bitmap[src_y:src_y+dst_h, src_x:src_x+dst_w, :3] if len(text_bitmap.shape) == 3 else np.stack([text_bitmap[src_y:src_y+dst_h, src_x:src_x+dst_w]] * 3, axis=2)
    
    # 颜色转换 (RGB -> BGR)
    color_bgr = color[:, :, ::-1]
    
    # 混合
    roi = result[paste_y:paste_y+dst_h, paste_x:paste_x+dst_w]
    result[paste_y:paste_y+dst_h, paste_x:paste_x+dst_w] = (alpha * color_bgr + (1 - alpha) * roi).astype(np.uint8)
    
    return result
