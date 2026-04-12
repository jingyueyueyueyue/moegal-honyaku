import asyncio
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.core.font_conf import FontConfig
from app.core.paths import SAVED_DIR
from app.services.ocr import ocr_recognize

# ============ 可配置参数（通过环境变量调整）============
OCR_MAX_CONCURRENCY = max(1, int(os.getenv("OCR_MAX_CONCURRENCY", "2")))
# Inpaint 半径（默认3，范围2-5）
INPAINT_RADIUS = int(os.getenv("INPAINT_RADIUS", "3"))
# 掩码膨胀迭代次数（默认1，范围1-3）
MASK_DILATE_ITERATIONS = int(os.getenv("MASK_DILATE_ITERATIONS", "1"))
# 掩码膨胀核大小（默认3，范围2-5）
MASK_DILATE_KERNEL_SIZE = int(os.getenv("MASK_DILATE_KERNEL_SIZE", "3"))
# 边界框扩展像素（默认2，范围0-5）
BBOX_PADDING = int(os.getenv("BBOX_PADDING", "2"))
# 是否使用增强版掩码算法
USE_ENHANCED_MASK = os.getenv("USE_ENHANCED_MASK", "false").lower() in ("true", "1", "yes")
# 是否进行二次inpaint
DOUBLE_INPAINT = os.getenv("DOUBLE_INPAINT", "false").lower() in ("true", "1", "yes")


def _sanitize_bbox(bbox, width: int, height: int):
    x1, y1, x2, y2 = map(int, bbox)
    # 扩展边界框，确保完全覆盖文字
    x1 = max(0, min(x1 - BBOX_PADDING, width - 1))
    y1 = max(0, min(y1 - BBOX_PADDING, height - 1))
    x2 = max(x1 + 1, min(x2 + BBOX_PADDING, width))
    y2 = max(y1 + 1, min(y2 + BBOX_PADDING, height))
    return x1, y1, x2, y2


def _build_text_mask_basic(cropped_cv: np.ndarray) -> np.ndarray:
    """基础版掩码生成（智能检测深色或浅色文字）"""
    if cropped_cv.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    gray = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    height, width = gray.shape[:2]
    area = height * width

    kernel_size = max(3, min(11, (min(height, width) // 10) * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # ===== 判断背景亮度 =====
    # 使用中位数判断背景是深色还是浅色
    median_luma = float(np.median(blur))

    if median_luma < 128:
        # ===== 深色背景（黑底）：检测浅色文字 =====
        # 白帽运算突出比背景更亮的细节
        tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
        _, th_mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 极亮像素检测
        p90 = float(np.percentile(blur, 90))
        bright_threshold = int(max(180, min(250, p90)))
        bright_mask = cv2.inRange(blur, bright_threshold, 255)

        merged = cv2.bitwise_or(th_mask, bright_mask)
    else:
        # ===== 浅色背景（白底）：检测深色文字（原算法）=====
        # 黑帽运算突出比背景更暗的细节
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
        _, bh_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 极深像素检测
        p15 = float(np.percentile(blur, 15))
        dark_threshold = int(max(30, min(120, p15)))
        dark_mask = cv2.inRange(blur, 0, dark_threshold)

        merged = cv2.bitwise_or(bh_mask, dark_mask)

    # 连通域过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    filtered = np.zeros_like(merged)
    min_area = max(6, int(area * 0.0002))
    max_area = max(20, int(area * 0.2))
    for idx in range(1, num_labels):
        component_area = int(stats[idx, cv2.CC_STAT_AREA])
        if min_area <= component_area <= max_area:
            filtered[labels == idx] = 255

    return filtered


def _build_text_mask_enhanced(cropped_cv: np.ndarray) -> np.ndarray:
    """增强版掩码生成 - 智能检测深色或浅色文字"""
    if cropped_cv.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    gray = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    area = height * width

    # 高斯模糊降噪
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    kernel_size = max(3, min(15, (min(height, width) // 8) * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # ===== 判断背景亮度 =====
    median_luma = float(np.median(blur))
    mean_luma = float(np.mean(blur))

    # 边缘检测（用于辅助）
    edges = cv2.Canny(blur, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    if median_luma < 128:
        # ===== 深色背景（黑底）：检测浅色文字 =====
        # 白帽运算突出比背景更亮的细节
        tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
        _, th_mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 自适应阈值检测亮色文字
        adaptive_mask = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=11, C=2
        )

        # 极亮像素检测
        p90 = float(np.percentile(blur, 90))
        bright_threshold = int(max(180, min(250, p90)))
        bright_mask = cv2.inRange(blur, bright_threshold, 255)

        # 融合策略
        merged = cv2.bitwise_or(th_mask, bright_mask)
        # 自适应阈值结果与边缘结合，减少噪声
        merged = cv2.bitwise_or(merged, cv2.bitwise_and(adaptive_mask, edges_dilated))
    else:
        # ===== 浅色背景（白底）：检测深色文字 =====
        # 黑帽运算突出比背景更暗的细节
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
        _, bh_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 自适应阈值检测暗色文字
        adaptive_mask = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=11, C=2
        )

        # 极深像素检测
        p15 = float(np.percentile(blur, 15))
        dark_threshold = int(max(30, min(120, p15)))
        dark_mask = cv2.inRange(blur, 0, dark_threshold)

        # 融合策略
        merged = cv2.bitwise_or(bh_mask, dark_mask)
        # 自适应阈值结果与边缘结合，减少噪声
        merged = cv2.bitwise_or(merged, cv2.bitwise_and(adaptive_mask, edges_dilated))

    # ===== 连通域过滤 =====
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    filtered = np.zeros_like(merged)

    min_area = max(4, int(area * 0.0001))
    max_area = max(30, int(area * 0.4))

    for idx in range(1, num_labels):
        component_area = int(stats[idx, cv2.CC_STAT_AREA])
        if min_area <= component_area <= max_area:
            w = stats[idx, cv2.CC_STAT_WIDTH]
            h = stats[idx, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(w, h) / max(1, min(w, h))
            if aspect_ratio < 20:
                filtered[labels == idx] = 255

    return filtered


def _build_text_mask(cropped_cv: np.ndarray) -> np.ndarray:
    """主掩码生成函数"""
    if USE_ENHANCED_MASK:
        mask = _build_text_mask_enhanced(cropped_cv)
    else:
        mask = _build_text_mask_basic(cropped_cv)

    if mask.size == 0:
        return mask

    # 膨胀掩码，确保完全覆盖文字边缘
    kernel = np.ones((MASK_DILATE_KERNEL_SIZE, MASK_DILATE_KERNEL_SIZE), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=MASK_DILATE_ITERATIONS)

    return mask


async def get_text_masked_pic(image_pil, image_cv, bboxes, inpaint=True, ocr_engine: str = None):
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
    if len(bboxes) == 0:
        return [], image_cv

    height, width = image_cv.shape[:2]
    semaphore = asyncio.Semaphore(min(OCR_MAX_CONCURRENCY, len(bboxes)))

    async def ocr_and_mask(bbox):
        x1, y1, x2, y2 = _sanitize_bbox(bbox, width, height)
        cropped_image = image_pil.crop((x1, y1, x2, y2))
        async with semaphore:
            text = await asyncio.to_thread(ocr_recognize, cropped_image, ocr_engine)
        local_mask = _build_text_mask(image_cv[y1:y2, x1:x2])
        return text, (x1, y1, x2, y2), local_mask

    tasks = [ocr_and_mask(bbox) for bbox in bboxes]
    results = await asyncio.gather(*tasks)
    all_text = []
    for text, (x1, y1, x2, y2), local_mask in results:
        all_text.append(text)
        if local_mask.size == 0:
            continue
        target = mask[y1:y2, x1:x2]
        np.maximum(target, local_mask, out=target)

    if inpaint and np.any(mask):
        # 可选算法：TELEA（快）或 NS（质量更好）
        inpaint_method = os.getenv("INPAINT_METHOD", "TELEA").upper()
        if inpaint_method == "NS":
            flags = cv2.INPAINT_NS
        else:
            flags = cv2.INPAINT_TELEA

        image_cv = cv2.inpaint(image_cv, mask, inpaintRadius=INPAINT_RADIUS, flags=flags)

    return all_text, image_cv


def _normalize_newlines(text: str) -> str:
    """
    将字面量换行符转换为真正的换行符
    处理 AI 返回的文本中 "\\n" 字面量
    """
    if not text:
        return text
    # 将字面量 \n (两个字符) 转换为真正的换行符
    return text.replace('\\n', '\n')


def wrap_text_by_width(draw, text, font, max_width):
    """
    将文字根据实际像素宽度换行，返回行列表（横排模式）
    """
    # 先处理字面量换行符
    text = _normalize_newlines(text)
    
    lines = []
    line = ''
    for char in text:
        # 遇到换行符直接换行
        if char == '\n':
            if line:
                lines.append(line)
            line = ''
            continue
        test_line = line + char
        w = draw.textlength(test_line, font=font)
        if w <= max_width:
            line = test_line
        else:
            if line:
                lines.append(line)
            line = char
    if line:
        lines.append(line)
    return lines


def wrap_text_vertical(text, font, max_height):
    """
    将文字根据实际像素高度分列，返回列列表（竖排模式）
    竖排：从右到左，每列从上到下
    返回的列表中，索引0是最先读到的文字（应画在最右边）
    """
    # 先处理字面量换行符
    text = _normalize_newlines(text)
    
    columns = []
    col = ''
    col_height = 0
    char_height = font.getbbox("中")[3] - font.getbbox("中")[1]
    
    for char in text:
        if char == '\n':
            if col:
                columns.append(col)
            col = ''
            col_height = 0
            continue
        if col_height + char_height <= max_height:
            col += char
            col_height += char_height
        else:
            if col:
                columns.append(col)
            col = char
            col_height = char_height
    if col:
        columns.append(col)
    return columns


def _pick_text_style(cropped_cv: np.ndarray):
    if cropped_cv.size == 0:
        return (20, 20, 20), (245, 245, 245)
    gray = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2GRAY)
    median_luma = float(np.median(gray))
    if median_luma >= 145:
        return (20, 20, 20), (245, 245, 245)
    return (245, 245, 245), (20, 20, 20)


def draw_text_on_boxes(image: np.ndarray, boxes: list, texts: list, text_direction: str = "horizontal") -> np.ndarray:
    """
    在指定区域绘制文字
    
    Args:
        image: BGR 图像
        boxes: 边界框列表
        texts: 文本列表
        text_direction: 文字方向 "horizontal"（横排）或 "vertical"（竖排）
    
    Returns:
        绘制后的 BGR 图像
    """
    height, width = image.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    line_spacing = 4  # 行间距/列间距
    char_height = None  # 用于竖排模式
    
    for box, text in zip(boxes, texts):
        # 绘制文字使用原始边界框（不扩展），确保文字定位准确
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        if not text:
            continue
        box_width = x2 - x1
        box_height = y2 - y1
        font_config = FontConfig(box_height, box_width, text)
        font = font_config.font
        fill_color, stroke_color = _pick_text_style(image[y1:y2, x1:x2])
        stroke_width = max(1, int(font.size * 0.12))
        
        if text_direction == "vertical":
            # ===== 竖排模式 =====
            char_height = font.getbbox("中")[3] - font.getbbox("中")[1]
            char_width = int(draw.textlength("中", font=font))
            columns = wrap_text_vertical(text, font, box_height)
            
            # 计算总宽度
            col_spacing = line_spacing
            total_width = len(columns) * char_width + max(0, len(columns) - 1) * col_spacing
            
            # 起始位置（水平居中）
            start_x = x1 + max(0, (box_width - total_width) // 2)
            
            for col_idx, col in enumerate(columns):
                # 从右到左绘制列
                col_x = start_x + (len(columns) - 1 - col_idx) * (char_width + col_spacing)
                # 垂直居中每列
                col_height = len(col) * char_height
                start_y = y1 + max(0, (box_height - col_height) // 2)
                
                for char_idx, char in enumerate(col):
                    char_y = start_y + char_idx * char_height
                    draw.text(
                        (col_x, char_y),
                        char,
                        font=font,
                        fill=fill_color,
                        stroke_width=stroke_width,
                        stroke_fill=stroke_color,
                    )
        else:
            # ===== 横排模式（原逻辑）=====
            lines = wrap_text_by_width(draw, text, font, box_width)
            line_height = font.getbbox("中")[3] - font.getbbox("中")[1]
            total_height = line_height * len(lines) + line_spacing * max(0, len(lines) - 1)
            start_y = y1 + max(0, (box_height - total_height) // 2)
            
            for i, line in enumerate(lines):
                y = start_y + i * (line_height + line_spacing)
                line_width = draw.textlength(line, font=font)
                x = x1 + max(0, int((box_width - line_width) / 2))
                draw.text(
                    (x, y),
                    line,
                    font=font,
                    fill=fill_color,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_color,
                )
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def save_img(file_bytes, pre: str, file_name: str):
    folder_path = SAVED_DIR / pre
    folder_path.mkdir(parents=True, exist_ok=True)
    with open(folder_path / file_name, "wb") as f:
        f.write(file_bytes)
