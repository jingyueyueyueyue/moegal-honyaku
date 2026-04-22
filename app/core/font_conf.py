from functools import lru_cache

from PIL import ImageFont

from app.core.paths import FONTS_DIR

FONT_PATH = FONTS_DIR / "LXGWWenKai-Regular.ttf"


@lru_cache(maxsize=2048)
def _load_font(font_path: str, font_size: int):
    return ImageFont.truetype(font_path, font_size)


def _glyph_dimensions(font) -> tuple[int, int]:
    """获取字符'中'的宽度和高度"""
    bbox = font.getbbox("中")
    h = max(1, bbox[3] - bbox[1])
    w = max(1, bbox[2] - bbox[0])
    return w, h


def _glyph_area(font) -> int:
    w, h = _glyph_dimensions(font)
    return h * w


@lru_cache(maxsize=4096)
def _calc_font_size(font_path: str, max_height: int, max_width: int, text_len: int, text_direction: str = "horizontal") -> int:
    """
    计算合适的字体大小
    
    Args:
        font_path: 字体路径
        max_height: 气泡最大高度
        max_width: 气泡最大宽度
        text_len: 文本长度
        text_direction: 文字方向 "horizontal" 或 "vertical"
    
    Returns:
        合适的字体大小
    """
    if text_len <= 0 or max_height <= 0 or max_width <= 0:
        return 10
    
    # 字体大小搜索范围
    high = max(10, min(512, max(max_height, max_width) * 2))
    low = 8
    best = 8
    
    # 二分搜索合适的字体大小
    while low <= high:
        mid = (low + high) // 2
        font = _load_font(font_path, mid)
        char_w, char_h = _glyph_dimensions(font)
        
        if text_direction == "vertical":
            # 竖排模式
            col_spacing = 4
            
            # 对于短文本（<=3字符），尽量放一列，使用最大字体
            if text_len <= 3:
                # 检查是否能放入一列
                total_height = text_len * char_h
                fits = total_height <= max_height * 0.95 and char_w <= max_width * 0.95
            else:
                # 计算需要的列数
                max_chars_per_col = max(1, max_height // char_h) if char_h > 0 else 1
                cols_needed = max(1, (text_len + max_chars_per_col - 1) // max_chars_per_col)
                total_width = cols_needed * char_w + max(0, cols_needed - 1) * col_spacing
                fits = total_width <= max_width * 0.95 and char_h <= max_height * 0.9
        else:
            # 横排模式
            line_spacing = 4
            max_chars_per_line = max(1, max_width // char_w) if char_w > 0 else 1
            lines_needed = max(1, (text_len + max_chars_per_line - 1) // max_chars_per_line)
            total_height = lines_needed * char_h + max(0, lines_needed - 1) * line_spacing
            fits = total_height <= max_height * 0.9 and char_w <= max_width * 0.9
        
        if fits:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return max(8, best)


class FontConfig:
    def __init__(self, max_height, max_width, text, font_path=FONT_PATH, text_direction: str = "horizontal"):
        self.font_path = str(font_path)
        self.text_direction = text_direction
        self.font_size = _calc_font_size(
            self.font_path, 
            int(max_height), 
            int(max_width), 
            len(text),
            text_direction
        )

    @property
    def font(self):
        return _load_font(self.font_path, self.font_size)
