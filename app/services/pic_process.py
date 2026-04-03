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
# Inpaint 半径：越大修复范围越广（默认8，针对边缘残留问题）
INPAINT_RADIUS = int(os.getenv("INPAINT_RADIUS", "8"))
# 掩码膨胀迭代次数：越大覆盖越完整（默认3）
MASK_DILATE_ITERATIONS = int(os.getenv("MASK_DILATE_ITERATIONS", "3"))
# 掩码膨胀核大小（默认7）
MASK_DILATE_KERNEL_SIZE = int(os.getenv("MASK_DILATE_KERNEL_SIZE", "7"))
# 边界框扩展像素：确保完全覆盖文字边缘（默认5）
BBOX_PADDING = int(os.getenv("BBOX_PADDING", "5"))
# 全局掩码扩展（在整个图像上额外扩展掩码，单位像素）
GLOBAL_MASK_EXPAND = int(os.getenv("GLOBAL_MASK_EXPAND", "4"))
# 是否使用增强版掩码算法
USE_ENHANCED_MASK = os.getenv("USE_ENHANCED_MASK", "true").lower() in ("true", "1", "yes")
# 是否进行二次inpaint（针对顽固残留）
DOUBLE_INPAINT = os.getenv("DOUBLE_INPAINT", "true").lower() in ("true", "1", "yes")


def _sanitize_bbox(bbox, width: int, height: int):
    x1, y1, x2, y2 = map(int, bbox)
    # 扩展边界框，确保完全覆盖文字
    x1 = max(0, min(x1 - BBOX_PADDING, width - 1))
    y1 = max(0, min(y1 - BBOX_PADDING, height - 1))
    x2 = max(x1 + 1, min(x2 + BBOX_PADDING, width))
    y2 = max(y1 + 1, min(y2 + BBOX_PADDING, height))
    return x1, y1, x2, y2


def _build_text_mask_basic(cropped_cv: np.ndarray) -> np.ndarray:
    """基础版掩码生成（原算法）"""
    if cropped_cv.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    gray = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 黑帽突出深色细节
    kernel_size = max(3, min(11, (min(cropped_cv.shape[:2]) // 10) * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    _, bh_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 兜底保留极深像素
    p15 = float(np.percentile(blur, 15))
    dark_threshold = int(max(30, min(120, p15)))
    dark_mask = cv2.inRange(blur, 0, dark_threshold)

    merged = cv2.bitwise_or(bh_mask, dark_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    filtered = np.zeros_like(merged)
    area = cropped_cv.shape[0] * cropped_cv.shape[1]
    min_area = max(6, int(area * 0.0002))
    max_area = max(20, int(area * 0.2))
    for idx in range(1, num_labels):
        component_area = int(stats[idx, cv2.CC_STAT_AREA])
        if min_area <= component_area <= max_area:
            filtered[labels == idx] = 255

    return filtered


def _build_text_mask_enhanced(cropped_cv: np.ndarray) -> np.ndarray:
    """增强版掩码生成 - 多策略融合"""
    if cropped_cv.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    gray = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    area = height * width

    # 高斯模糊降噪
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # ===== 策略1: 黑帽运算检测深色文字 =====
    kernel_size = max(3, min(15, (min(height, width) // 8) * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    _, bh_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ===== 策略2: 自适应阈值检测 =====
    # 适用于光照不均匀的气泡
    adaptive_mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=11, C=2
    )

    # ===== 策略3: 深色像素直方图检测 =====
    # 动态确定深色阈值
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    # 找到直方图的低谷作为阈值
    p10 = float(np.percentile(blur, 10))
    p25 = float(np.percentile(blur, 25))
    dark_threshold = int(max(20, min(100, (p10 + p25) / 2)))
    dark_mask = cv2.inRange(blur, 0, dark_threshold)

    # ===== 策略4: 边缘检测辅助 =====
    edges = cv2.Canny(blur, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    # ===== 融合多策略 =====
    merged = cv2.bitwise_or(bh_mask, dark_mask)
    merged = cv2.bitwise_or(merged, cv2.bitwise_and(adaptive_mask, edges_dilated))

    # ===== 连通域过滤 =====
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    filtered = np.zeros_like(merged)

    min_area = max(4, int(area * 0.0001))  # 降低最小面积，保留更多细节
    max_area = max(30, int(area * 0.4))    # 提高最大面积，处理大字体

    for idx in range(1, num_labels):
        component_area = int(stats[idx, cv2.CC_STAT_AREA])
        if min_area <= component_area <= max_area:
            # 检查宽高比，过滤掉极端形状
            w = stats[idx, cv2.CC_STAT_WIDTH]
            h = stats[idx, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(w, h) / max(1, min(w, h))
            if aspect_ratio < 20:  # 过滤掉过长的线条
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


async def get_text_masked_pic(image_pil, image_cv, bboxes, inpaint=True):
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
    if len(bboxes) == 0:
        return [], image_cv

    height, width = image_cv.shape[:2]
    semaphore = asyncio.Semaphore(min(OCR_MAX_CONCURRENCY, len(bboxes)))

    async def ocr_and_mask(bbox):
        x1, y1, x2, y2 = _sanitize_bbox(bbox, width, height)
        cropped_image = image_pil.crop((x1, y1, x2, y2))
        async with semaphore:
            text = await asyncio.to_thread(ocr_recognize, cropped_image)
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
        # 全局掩码扩展：在整个图像层面再扩展一次
        if GLOBAL_MASK_EXPAND > 0:
            expand_kernel = np.ones((GLOBAL_MASK_EXPAND * 2 + 1, GLOBAL_MASK_EXPAND * 2 + 1), dtype=np.uint8)
            mask = cv2.dilate(mask, expand_kernel, iterations=1)

        # 可选算法：TELEA（快）或 NS（质量更好）
        inpaint_method = os.getenv("INPAINT_METHOD", "NS").upper()
        if inpaint_method == "NS":
            flags = cv2.INPAINT_NS
        else:
            flags = cv2.INPAINT_TELEA

        image_cv = cv2.inpaint(image_cv, mask, inpaintRadius=INPAINT_RADIUS, flags=flags)

        # 二次inpaint：针对顽固残留
        if DOUBLE_INPAINT:
            # 检测是否还有残留文字边缘
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            # 在原掩码区域边缘检测残留
            mask_edge = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
            mask_edge = mask_edge - mask
            
            if np.any(mask_edge):
                # 在边缘区域检测残留
                edge_region = cv2.bitwise_and(gray, gray, mask=mask_edge)
                # 检测是否有残留的深色像素
                residual_mask = cv2.inRange(edge_region, 0, 80)
                residual_mask = cv2.dilate(residual_mask, np.ones((2, 2), np.uint8), iterations=1)
                
                if np.any(residual_mask):
                    # 对残留区域进行二次修复
                    image_cv = cv2.inpaint(image_cv, residual_mask, inpaintRadius=3, flags=flags)

        # 后处理：轻微模糊修复区域边缘，使过渡更自然
        if os.getenv("INPAINT_BLUR_EDGE", "true").lower() in ("true", "1", "yes"):
            mask_dilated = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
            mask_edge = mask_dilated - mask
            if np.any(mask_edge):
                blurred = cv2.GaussianBlur(image_cv, (5, 5), 0)
                image_cv = np.where(mask_edge[:, :, np.newaxis] > 0, blurred, image_cv)

    return all_text, image_cv


def wrap_text_by_width(draw, text, font, max_width):
    """
    将文字根据实际像素宽度换行，返回行列表
    """
    lines = []
    line = ''
    for char in text:
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


def _pick_text_style(cropped_cv: np.ndarray):
    if cropped_cv.size == 0:
        return (20, 20, 20), (245, 245, 245)
    gray = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2GRAY)
    median_luma = float(np.median(gray))
    if median_luma >= 145:
        return (20, 20, 20), (245, 245, 245)
    return (245, 245, 245), (20, 20, 20)


def draw_text_on_boxes(image: np.ndarray, boxes: list, texts: list) -> np.ndarray:
    height, width = image.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    line_spacing = 4  # 行间距
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
        lines = wrap_text_by_width(draw, text, font, box_width)
        line_height = font.getbbox("中")[3] - font.getbbox("中")[1]
        total_height = line_height * len(lines) + line_spacing * max(0, len(lines) - 1)
        start_y = y1 + max(0, (box_height - total_height) // 2)
        fill_color, stroke_color = _pick_text_style(image[y1:y2, x1:x2])
        stroke_width = max(1, int(font.size * 0.12))
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
