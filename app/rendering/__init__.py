"""
MIT 渲染模块适配层

提供高质量的文本渲染功能：
- Qt 渲染引擎（精确字形渲染）
- CJK 标点禁则
- 竖排/横排自动换行
- <H> 标签支持（竖排中嵌入横排）
"""

from .text_render import (
    set_font,
    get_string_width,
    get_string_height,
    get_char_offset_x,
    get_char_offset_y,
    get_vertical_char_bitmap_width,
    calc_horizontal_block_height,
    put_text_horizontal,
    put_text_vertical,
    calc_horizontal,
    calc_vertical,
    compact_special_symbols,
    auto_add_horizontal_tags,
    prepare_text_for_direction_rendering,
    CJK_Compatibility_Forms_translate,
    DEFAULT_FONT,
)

from .auto_linebreak import (
    solve_no_br_layout,
    should_force_no_wrap_single_region,
)

from .render_adapter import (
    TextRenderConfig,
    calculate_font_size,
    render_text,
    ensure_font_available,
)

__all__ = [
    # text_render
    'set_font',
    'get_string_width',
    'get_string_height',
    'get_char_offset_x',
    'get_char_offset_y',
    'get_vertical_char_bitmap_width',
    'calc_horizontal_block_height',
    'put_text_horizontal',
    'put_text_vertical',
    'calc_horizontal',
    'calc_vertical',
    'compact_special_symbols',
    'auto_add_horizontal_tags',
    'prepare_text_for_direction_rendering',
    'CJK_Compatibility_Forms_translate',
    'DEFAULT_FONT',
    # auto_linebreak
    'solve_no_br_layout',
    'should_force_no_wrap_single_region',
    # render_adapter
    'TextRenderConfig',
    'calculate_font_size',
    'render_text',
    'ensure_font_available',
]
