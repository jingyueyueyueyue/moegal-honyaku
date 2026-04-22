import asyncio
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.core.font_conf import FontConfig
from app.core.logger import logger
from app.core.paths import PROJECT_ROOT, SAVED_DIR
from app.services.ocr import ocr_recognize

# MIT 高质量渲染模块
try:
    from app.rendering import (
        set_font,
        put_text_horizontal,
        put_text_vertical,
        compact_special_symbols,
        calculate_font_size,
        TextRenderConfig,
        get_string_width,
        get_string_height,
    )
    HAS_MIT_RENDER = True
except ImportError as e:
    logger.warning(f"MIT 渲染模块未加载，将使用 PIL 渲染: {e}")
    HAS_MIT_RENDER = False

# 是否使用 MIT 渲染模块（默认启用）
USE_MIT_RENDER = os.getenv("USE_MIT_RENDER", "true").lower() in ("true", "1", "yes")

# ============ 可配置参数（通过环境变量调整）============
OCR_MAX_CONCURRENCY = max(1, int(os.getenv("OCR_MAX_CONCURRENCY", "2")))
# Inpaint 半径（默认4，范围2-7）
INPAINT_RADIUS = int(os.getenv("INPAINT_RADIUS", "4"))
# 掩码膨胀迭代次数（默认2，范围1-4）
MASK_DILATE_ITERATIONS = int(os.getenv("MASK_DILATE_ITERATIONS", "2"))
# 掩码膨胀核大小（默认5，范围3-9）
MASK_DILATE_KERNEL_SIZE = int(os.getenv("MASK_DILATE_KERNEL_SIZE", "5"))
# 边界框扩展像素（默认2，范围0-5）

# 是否启用文本框合并（合并相邻的文本框）
MERGE_TEXTBOXES = os.getenv("MERGE_TEXTBOXES", "true").lower() in ("true", "1", "yes")
# 文本框合并的距离阈值（像素，默认25，放宽以适应同一气泡内的多列文字）
MERGE_DISTANCE_THRESHOLD = int(os.getenv("MERGE_DISTANCE_THRESHOLD", "25"))
# 文本框合并的高度差阈值（像素，默认30，用于判断是否在同一行）
MERGE_HEIGHT_THRESHOLD = int(os.getenv("MERGE_HEIGHT_THRESHOLD", "30"))
BBOX_PADDING = int(os.getenv("BBOX_PADDING", "2"))
# 是否使用增强版掩码算法
USE_ENHANCED_MASK = os.getenv("USE_ENHANCED_MASK", "false").lower() in ("true", "1", "yes")
# 是否进行二次inpaint
DOUBLE_INPAINT = os.getenv("DOUBLE_INPAINT", "false").lower() in ("true", "1", "yes")
# Lama inpainting 最大处理尺寸（默认 2048，与 MTU 一致）
LAMA_INPAINT_SIZE = int(os.getenv("LAMA_INPAINT_SIZE", "2048"))
MTU_MASK_DILATION_OFFSET = int(os.getenv("MTU_MASK_DILATION_OFFSET", "20"))
DEBUG_DIAG = os.getenv("MOEGAL_DEBUG_DIAG", "false").lower() in ("true", "1", "yes")


def _diag_print(message: str) -> None:
    if DEBUG_DIAG:
        print(message, flush=True)


def _summarize_log_text(text: str, max_chars: int = 120) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _log_ocr_summary(texts: list[str], merged: bool = False, max_items: int = 20) -> None:
    if texts is None:
        return
    suffix = " (merged)" if merged else ""
    logger.info(f"OCR extracted {len(texts)} text lines{suffix}")
    preview_count = min(len(texts), max_items)
    for idx, text in enumerate(texts[:preview_count], start=1):
        logger.info(f"  [{idx}] {_summarize_log_text(text)}")
    remaining = len(texts) - preview_count
    if remaining > 0:
        logger.info(f"  ... {remaining} more lines")


def _get_inpaint_config():
    """
    获取 inpainting 配置
    优先从 custom_conf 读取，回退到环境变量
    默认使用 aot（MIT 默认模型，效果最好）
    """
    try:
        from app.core.custom_conf import INPAINT_METHOD_QUALITY, custom_conf, normalize_inpaint_method
        method = normalize_inpaint_method(getattr(custom_conf, "inpaint_method", None)) or INPAINT_METHOD_QUALITY
        use_crf = getattr(custom_conf, "use_crf_refine", None)
        
        # 如果 custom_conf 有值，使用它
        if method:
            return method, use_crf if use_crf is not None else (method == INPAINT_METHOD_QUALITY)
    except Exception:
        pass
    
    # 回退到环境变量，默认使用 aot（效果最好）
    method = os.getenv("INPAINT_METHOD", "quality").lower()
    use_crf = os.getenv("USE_CRF_REFINE", "false").lower() in ("true", "1", "yes")
    return method, use_crf


def _normalize_inpaint_mode(method: str | None) -> str:
    try:
        from app.core.custom_conf import INPAINT_METHOD_QUALITY, normalize_inpaint_method

        normalized = normalize_inpaint_method(method)
        if normalized:
            return normalized
        return INPAINT_METHOD_QUALITY
    except Exception:
        normalized = (method or "").strip().lower()
        if normalized in {"fast", "telea", "ns", "aot"}:
            return "fast"
        if normalized in {"quality", "lama", "lama_mpe", "lama_large"}:
            return "quality"
        return "quality"


def _resolve_inpaint_runtime(method: str, use_crf: bool | None = None) -> tuple[str, str, bool, bool, bool]:
    mode = _normalize_inpaint_mode(method)
    if mode == "fast":
        return mode, "telea", False, False, False

    try:
        from app.services.mtu_inpaint_bridge import supports_mtu_inpaint
        from app.services.mtu_mask_refinement_bridge import supports_mtu_mask_refinement

        if supports_mtu_inpaint("lama_large"):
            return mode, "lama_large", True, supports_mtu_mask_refinement(), True
    except Exception:
        pass

    return mode, "aot_local", True, False, False


def _resolve_bg_fix_repair_method(method: str) -> str:
    forced_method = os.getenv("BG_FIX_REPAIR_METHOD", "").strip().lower()
    if forced_method in {"aot", "lama", "lama_mpe", "lama_large"}:
        return forced_method

    normalized = (method or "").strip().lower()
    if normalized in {"fast", "quality"}:
        normalized = "lama_large" if normalized == "quality" else "telea"
    if normalized == "aot":
        return "lama_large"
    if normalized == "aot_local":
        return "lama_large"
    if normalized in {"lama", "lama_mpe", "lama_large"}:
        return normalized
    return "lama_large"


def _cluster_vertical_columns(indices: list, centers: list) -> list:
    """将竖排框按列聚类，并按阅读顺序（右到左、上到下）输出。"""
    if not indices:
        return []

    sorted_by_x = sorted(indices, key=lambda idx: centers[idx][0], reverse=True)
    max_width = max(centers[idx][2] for idx in indices)
    x_tolerance = max(max_width * 0.9, 12)
    columns = []

    for idx in sorted_by_x:
        cx = centers[idx][0]
        placed = False
        for column in columns:
            if abs(column["anchor_x"] - cx) <= x_tolerance:
                column["indices"].append(idx)
                current_xs = [centers[i][0] for i in column["indices"]]
                column["anchor_x"] = sum(current_xs) / len(current_xs)
                placed = True
                break
        if not placed:
            columns.append({"anchor_x": cx, "indices": [idx]})

    ordered_columns = []
    for column in sorted(columns, key=lambda item: item["anchor_x"], reverse=True):
        ordered_columns.append(sorted(column["indices"], key=lambda idx: centers[idx][1]))
    return ordered_columns



def _merge_textboxes(
    texts: list, 
    bboxes: list, 
    distance_threshold: int = None,
    height_threshold: int = None
) -> tuple:
    """
    合并相邻的文本框
    
    根据距离和方向判断是否为同一气泡内的文本，合并后一起翻译。
    
    Args:
        texts: 文本列表
        bboxes: 边界框列表 [[x1,y1,x2,y2], ...]
        distance_threshold: 距离阈值（像素）
        height_threshold: 高度差阈值（像素）
        
    Returns:
        (合并后的文本列表, 合并后的边界框列表, 合并映射)
    """
    if not texts or not bboxes:
        return texts, bboxes, []
    
    if distance_threshold is None:
        distance_threshold = MERGE_DISTANCE_THRESHOLD
    if height_threshold is None:
        height_threshold = MERGE_HEIGHT_THRESHOLD
    
    n = len(texts)
    if n <= 1:
        return texts, bboxes, [[0]]
    
    # 计算每个框的中心点和尺寸
    centers = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        centers.append((cx, cy, w, h))
    
    # 使用 Union-Find 合并相邻的框
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # 判断两个框是否应该合并（基于空间邻近性）
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, w1, h1 = centers[i]
            cx2, cy2, w2, h2 = centers[j]
            x1_i, y1_i, x2_i, y2_i = bboxes[i]
            x1_j, y1_j, x2_j, y2_j = bboxes[j]
            
            # 计算扩展后的边界框（扩展 30% 用于判断邻近性）
            expand = 0.3
            ex1_i, ey1_i, ex2_i, ey2_i = x1_i - w1 * expand, y1_i - h1 * expand, x2_i + w1 * expand, y2_i + h1 * expand
            ex1_j, ey1_j, ex2_j, ey2_j = x1_j - w2 * expand, y1_j - h2 * expand, x2_j + w2 * expand, y2_j + h2 * expand
            
            # 检查扩展边界框是否重叠
            overlaps = not (ex2_i < ex1_j or ex2_j < ex1_i or ey2_i < ey1_j or ey2_j < ey1_i)
            
            # 或者计算实际距离
            h_dist = max(0, max(x1_i, x1_j) - min(x2_i, x2_j))
            v_dist = max(0, max(y1_i, y1_j) - min(y2_i, y2_j))
            
            # 判断是否为竖排文字
            is_vertical_i = h1 > w1 * 1.5
            is_vertical_j = h2 > w2 * 1.5
            
            should_merge = False
            
            if is_vertical_i and is_vertical_j:
                # 两个都是竖排：水平或垂直相邻都可合并（同一气泡内可能有多列竖排）
                # 放宽条件：只要扩展边界框重叠，或者距离很近
                if overlaps or (h_dist < distance_threshold * 2 and v_dist < max(h1, h2) * 0.5):
                    should_merge = True
            elif not is_vertical_i and not is_vertical_j:
                # 两个都是横排：需要垂直对齐且水平相邻
                v_align = abs(cy1 - cy2) < max(h1, h2) * 0.8
                h_adjacent = h_dist < distance_threshold * 2
                if v_align and h_adjacent:
                    should_merge = True
            else:
                # 一个竖排一个横排：扩展边界框重叠则合并
                if overlaps:
                    should_merge = True
            
            if should_merge:
                union(i, j)
    
    # 收集每个组的成员
    groups = {}
    for i in range(n):
        p = find(i)
        if p not in groups:
            groups[p] = []
        groups[p].append(i)
    
    # 合并文本和边界框
    merged_texts = []
    merged_bboxes = []
    merge_map = []
    
    for group_indices in groups.values():
        merge_map.append(group_indices)
        
        if len(group_indices) == 1:
            # 单个框，不合并
            idx = group_indices[0]
            merged_texts.append(texts[idx])
            merged_bboxes.append(bboxes[idx])
        else:
            # 多个框，合并
            # 判断是否都是竖排
            all_vertical = all(centers[idx][3] > centers[idx][2] * 1.5 for idx in group_indices)
            
            if all_vertical:
                column_groups = _cluster_vertical_columns(group_indices, centers)
                sorted_indices = [idx for column in column_groups for idx in column]
            else:
                # 横排：按 x 坐标排序（从左到右）
                sorted_indices = sorted(group_indices, key=lambda i: centers[i][0])
            
            # 合并文本
            merged_text = ""
            if all_vertical:
                for column in column_groups:
                    column_text = "".join(texts[idx].strip() for idx in column if texts[idx].strip())
                    if not column_text:
                        continue
                    if merged_text:
                        merged_text += " "
                    merged_text += column_text
            else:
                for idx in sorted_indices:
                    text = texts[idx].strip()
                    if text:
                        merged_text += text
            merged_texts.append(merged_text)
            
            # 合并边界框
            min_x = min(bboxes[i][0] for i in group_indices)
            min_y = min(bboxes[i][1] for i in group_indices)
            max_x = max(bboxes[i][2] for i in group_indices)
            max_y = max(bboxes[i][3] for i in group_indices)
            merged_bboxes.append([min_x, min_y, max_x, max_y])
    
    logger.debug(f"文本框合并: {n} -> {len(merged_texts)} 个")
    for i, group in enumerate(merge_map):
        if len(group) > 1:
            original_texts = [texts[j] for j in group]
            logger.debug(f"  合并组 {i+1}: {original_texts} -> {merged_texts[i]}")
    
    return merged_texts, merged_bboxes, merge_map


def _build_render_groups(
    boxes: list,
    text_direction: str = "horizontal",
    distance_threshold: int = None,
    height_threshold: int = None,
) -> list:
    """Group nearby boxes so we can keep the same visual style inside one bubble."""
    if not boxes:
        return []

    if distance_threshold is None:
        distance_threshold = MERGE_DISTANCE_THRESHOLD
    if height_threshold is None:
        height_threshold = MERGE_HEIGHT_THRESHOLD

    n = len(boxes)
    if n <= 1:
        return [[0]]

    metrics = []
    for bbox in boxes:
        x1, y1, x2, y2 = map(float, bbox)
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        metrics.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "w": w,
            "h": h,
        })

    is_vertical_layout = text_direction == "vertical"
    if is_vertical_layout:
        parent = list(range(n))

        def find_column(x):
            if parent[x] != x:
                parent[x] = find_column(parent[x])
            return parent[x]

        def union_column(x, y):
            px, py = find_column(x), find_column(y)
            if px != py:
                parent[px] = py

        # Pass 1: merge fragments that clearly belong to the same vertical column.
        for i in range(n):
            for j in range(i + 1, n):
                m1 = metrics[i]
                m2 = metrics[j]
                overlap_w = max(0.0, min(m1["x2"], m2["x2"]) - max(m1["x1"], m2["x1"]))
                overlap_h = max(0.0, min(m1["y2"], m2["y2"]) - max(m1["y1"], m2["y1"]))
                gap_x = max(0.0, max(m1["x1"], m2["x1"]) - min(m1["x2"], m2["x2"]))
                gap_y = max(0.0, max(m1["y1"], m2["y1"]) - min(m1["y2"], m2["y2"]))
                overlap_w_ratio = overlap_w / max(1.0, min(m1["w"], m2["w"]))
                overlap_h_ratio = overlap_h / max(1.0, min(m1["h"], m2["h"]))
                center_x_close = abs(m1["cx"] - m2["cx"]) <= max(6.0, min(m1["w"], m2["w"]) * 0.25)

                same_column_stack = (
                    center_x_close
                    and gap_x <= max(4.0, min(m1["w"], m2["w"]) * 0.2)
                    and (
                        overlap_h_ratio >= 0.2
                        or (
                            overlap_w_ratio >= 0.45
                            and gap_y <= max(height_threshold, min(m1["h"], m2["h"]) * 0.8)
                        )
                    )
                )
                if same_column_stack:
                    union_column(i, j)

        columns = {}
        for idx in range(n):
            root = find_column(idx)
            columns.setdefault(root, []).append(idx)

        ordered_columns = sorted(
            (sorted(indices) for indices in columns.values()),
            key=lambda indices: indices[0],
        )
        if len(ordered_columns) <= 1:
            return ordered_columns

        column_metrics = []
        for indices in ordered_columns:
            x1 = min(metrics[idx]["x1"] for idx in indices)
            y1 = min(metrics[idx]["y1"] for idx in indices)
            x2 = max(metrics[idx]["x2"] for idx in indices)
            y2 = max(metrics[idx]["y2"] for idx in indices)
            widths = [metrics[idx]["w"] for idx in indices]
            heights = [metrics[idx]["h"] for idx in indices]
            column_metrics.append({
                "indices": indices,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "w": max(x2 - x1, 1.0),
                "h": max(y2 - y1, 1.0),
                "avg_w": sum(widths) / max(len(widths), 1),
                "avg_h": sum(heights) / max(len(heights), 1),
            })

        bubble_parent = list(range(len(column_metrics)))

        def find_bubble(x):
            if bubble_parent[x] != x:
                bubble_parent[x] = find_bubble(bubble_parent[x])
            return bubble_parent[x]

        def union_bubble(x, y):
            px, py = find_bubble(x), find_bubble(y)
            if px != py:
                bubble_parent[px] = py

        # Pass 2: only merge side-by-side columns when they actually share the same bubble height band.
        for i in range(len(column_metrics)):
            for j in range(i + 1, len(column_metrics)):
                m1 = column_metrics[i]
                m2 = column_metrics[j]
                overlap_h = max(0.0, min(m1["y2"], m2["y2"]) - max(m1["y1"], m2["y1"]))
                gap_x = max(0.0, max(m1["x1"], m2["x1"]) - min(m1["x2"], m2["x2"]))
                overlap_h_ratio = overlap_h / max(1.0, min(m1["h"], m2["h"]))
                top_aligned = abs(m1["y1"] - m2["y1"]) <= max(height_threshold, min(m1["h"], m2["h"]) * 0.22)
                bottom_aligned = abs(m1["y2"] - m2["y2"]) <= max(height_threshold, min(m1["h"], m2["h"]) * 0.22)
                center_y_close = abs(m1["cy"] - m2["cy"]) <= max(height_threshold, min(m1["h"], m2["h"]) * 0.3)
                close_columns = gap_x <= max(distance_threshold * 1.5, min(m1["avg_w"], m2["avg_w"]) * 2.0)

                if close_columns and overlap_h_ratio >= 0.42 and (top_aligned or bottom_aligned or center_y_close):
                    union_bubble(i, j)

        bubble_groups = {}
        for idx in range(len(column_metrics)):
            root = find_bubble(idx)
            bubble_groups.setdefault(root, []).append(idx)

        grouped_indices = []
        for column_group in bubble_groups.values():
            merged = []
            for column_idx in column_group:
                merged.extend(column_metrics[column_idx]["indices"])
            grouped_indices.append(sorted(merged))

        return sorted(grouped_indices, key=lambda indices: indices[0])

    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    def expanded_overlap(m1, m2, expand_x: float, expand_y: float) -> bool:
        return not (
            (m1["x2"] + expand_x) < (m2["x1"] - expand_x)
            or (m2["x2"] + expand_x) < (m1["x1"] - expand_x)
            or (m1["y2"] + expand_y) < (m2["y1"] - expand_y)
            or (m2["y2"] + expand_y) < (m1["y1"] - expand_y)
        )

    for i in range(n):
        for j in range(i + 1, n):
            m1 = metrics[i]
            m2 = metrics[j]

            overlap_w = max(0.0, min(m1["x2"], m2["x2"]) - max(m1["x1"], m2["x1"]))
            overlap_h = max(0.0, min(m1["y2"], m2["y2"]) - max(m1["y1"], m2["y1"]))
            gap_x = max(0.0, max(m1["x1"], m2["x1"]) - min(m1["x2"], m2["x2"]))
            gap_y = max(0.0, max(m1["y1"], m2["y1"]) - min(m1["y2"], m2["y2"]))

            expand_x = max(distance_threshold, int(round(min(m1["w"], m2["w"]) * 0.35)))
            expand_y = max(height_threshold, int(round(min(m1["h"], m2["h"]) * 0.8)))
            overlap_w_ratio = overlap_w / max(1.0, min(m1["w"], m2["w"]))
            row_aligned = abs(m1["cy"] - m2["cy"]) <= max(height_threshold, int(round(min(m1["h"], m2["h"]) * 0.8)))
            should_group = (
                expanded_overlap(m1, m2, expand_x, expand_y)
                or (
                    row_aligned
                    and (
                        overlap_w_ratio >= 0.35
                        or gap_x <= max(distance_threshold * 2, int(round(min(m1["w"], m2["w"]) * 0.35)))
                    )
                )
            )

            if should_group:
                union(i, j)

    groups = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    return sorted(
        (sorted(indices) for indices in groups.values()),
        key=lambda indices: indices[0],
    )


def _merge_textboxes_with_groups(
    texts: list,
    bboxes: list,
    groups: list,
    text_direction: str = "horizontal",
) -> tuple:
    """Merge OCR results with an explicit group layout so render and translation stay aligned."""
    if not texts or not bboxes:
        return texts, bboxes, []

    centers = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))

    ordered_groups = sorted(
        (sorted(group) for group in groups if group),
        key=lambda group: group[0],
    )
    if not ordered_groups:
        return texts, bboxes, []

    merged_texts = []
    merged_bboxes = []

    for group_indices in ordered_groups:
        if len(group_indices) == 1:
            idx = group_indices[0]
            merged_texts.append(texts[idx])
            merged_bboxes.append(bboxes[idx])
            continue

        all_vertical = text_direction == "vertical" or all(
            centers[idx][3] > centers[idx][2] * 1.5 for idx in group_indices
        )

        if all_vertical:
            column_groups = _cluster_vertical_columns(group_indices, centers)
            merged_text = " ".join(
                "".join(texts[idx].strip() for idx in column if texts[idx].strip())
                for column in column_groups
                if any(texts[idx].strip() for idx in column)
            )
        else:
            ordered_indices = sorted(group_indices, key=lambda idx: centers[idx][0])
            merged_text = "".join(texts[idx].strip() for idx in ordered_indices if texts[idx].strip())

        merged_texts.append(merged_text)
        merged_bboxes.append([
            min(bboxes[idx][0] for idx in group_indices),
            min(bboxes[idx][1] for idx in group_indices),
            max(bboxes[idx][2] for idx in group_indices),
            max(bboxes[idx][3] for idx in group_indices),
        ])

    return merged_texts, merged_bboxes, ordered_groups


def _estimate_mask_font_size(bbox) -> int:
    """Estimate glyph size from the short edge so vertical text columns do not over-expand."""
    x1, y1, x2, y2 = map(int, bbox)
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    shorter_side = min(box_width, box_height)
    longer_side = max(box_width, box_height)

    if longer_side >= shorter_side * 2.0:
        estimated = shorter_side * 0.95
    else:
        estimated = shorter_side * 0.85

    return max(12, min(int(round(estimated)), 96))


def _render_text_bitmap_with_fit(
    *,
    text: str,
    font_size: int,
    box_width: int,
    box_height: int,
    is_horizontal: bool,
    fill_color,
    stroke_color,
    target_lang: str,
    align_x: str = "center",
    align_y: str = "center",
    bubble_mask: np.ndarray | None = None,
):
    """Render text and keep shrinking until the final bitmap is safely inside the target box."""
    if not text or box_width <= 0 or box_height <= 0:
        return None, max(int(font_size), 1)

    current_font_size = max(int(font_size), 6)
    min_font_size = 6
    fitted_bitmap = None
    fitted_size = current_font_size

    while current_font_size >= min_font_size:
        if is_horizontal:
            text_bitmap = put_text_horizontal(
                font_size=current_font_size,
                text=text,
                width=box_width,
                height=box_height,
                alignment='center',
                reversed_direction=False,
                fg=fill_color,
                bg=stroke_color,
                lang=target_lang,
                hyphenate=True,
                line_spacing=0,
                stroke_width=0.07,
                letter_spacing=1.0,
            )
        else:
            text_bitmap = put_text_vertical(
                font_size=current_font_size,
                text=text,
                h=box_height,
                alignment=align_y,
                fg=fill_color,
                bg=stroke_color,
                line_spacing=0,
                stroke_width=0.07,
                letter_spacing=1.0,
            )

        if text_bitmap is None:
            return None, current_font_size

        fitted_bitmap = text_bitmap
        fitted_size = current_font_size
        text_h, text_w = text_bitmap.shape[:2]
        width_limit = max(box_width - 2, 1)
        height_limit = max(box_height - 2, 1)
        fits_box = text_w <= width_limit and text_h <= height_limit
        fits_bubble = _bitmap_fits_bubble_mask(
            text_bitmap,
            bubble_mask,
            box_width,
            box_height,
            align_x=align_x,
            align_y=align_y,
        )
        if fits_box and fits_bubble:
            return text_bitmap, current_font_size

        if fits_box and not fits_bubble:
            next_font_size = max(int(round(current_font_size * 0.92)), current_font_size - 2)
        else:
            scale = min(width_limit / max(text_w, 1), height_limit / max(text_h, 1))
            if scale >= 0.98:
                next_font_size = current_font_size - 1
            else:
                next_font_size = int(current_font_size * max(scale * 0.95, 0.55))
        if next_font_size >= current_font_size:
            next_font_size = current_font_size - 1
        if next_font_size < min_font_size and current_font_size == min_font_size:
            break
        current_font_size = max(next_font_size, min_font_size)

    return fitted_bitmap, fitted_size


def _extract_bubble_mask_from_crop(cropped_cv: np.ndarray) -> np.ndarray | None:
    """Approximate the speech-bubble interior inside a render crop using MTU-style contour flood fill."""
    if cropped_cv is None or cropped_cv.size == 0:
        return None

    h, w = cropped_cv.shape[:2]
    if h < 24 or w < 24:
        return None

    work = cropped_cv.copy()
    scale = 1.0
    if h > 300 and w > 300:
        scale = 0.6
    elif h < 120 or w < 120:
        scale = 1.4

    if scale != 1.0:
        work = cv2.resize(
            work,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    wh, ww = work.shape[:2]
    img_area = max(wh * ww, 1)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    detected_edges = cv2.Canny(blurred, 70, 140, L2gradient=True, apertureSize=3)
    cv2.rectangle(detected_edges, (0, 0), (ww - 1, wh - 1), 255, 1, cv2.LINE_8)
    contours, _ = cv2.findContours(detected_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.rectangle(detected_edges, (0, 0), (ww - 1, wh - 1), 0, 1, cv2.LINE_8)

    bubble_mask = np.zeros((wh, ww), np.uint8)
    min_area = float("inf")
    difres = 10
    seed = (ww // 2, wh // 2)

    for contour_idx, contour in enumerate(contours):
        rect = cv2.boundingRect(contour)
        if rect[2] * rect[3] < img_area * 0.4:
            continue

        contour_mask = np.zeros((wh, ww), np.uint8)
        cv2.drawContours(contour_mask, contours, contour_idx, 255, 2)
        candidate = contour_mask.copy()
        flood_area, _, _, _ = cv2.floodFill(
            candidate,
            mask=None,
            seedPoint=seed,
            flags=4,
            newVal=127,
            loDiff=difres,
            upDiff=difres,
        )
        if img_area * 0.24 < flood_area < min_area:
            min_area = flood_area
            bubble_mask = candidate

    if not np.any(bubble_mask):
        return None

    bubble_mask = 127 - bubble_mask
    bubble_mask = cv2.dilate(
        bubble_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    bubble_area, _, _, _ = cv2.floodFill(
        bubble_mask,
        mask=None,
        seedPoint=seed,
        flags=4,
        newVal=30,
        loDiff=difres,
        upDiff=difres,
    )
    bubble_mask = 30 - bubble_mask
    _, bubble_mask = cv2.threshold(bubble_mask, 1, 255, cv2.THRESH_BINARY)
    bubble_mask = cv2.bitwise_not(bubble_mask)

    morph_side = int(np.sqrt(max(bubble_area, 1)) / 30)
    if morph_side > 1:
        morph_kernel = np.ones((morph_side, morph_side), np.uint8)
        bubble_mask = cv2.dilate(bubble_mask, morph_kernel, iterations=1)
        bubble_mask = cv2.erode(bubble_mask, morph_kernel, iterations=1)

    if scale != 1.0:
        bubble_mask = cv2.resize(bubble_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    positive_pixels = int(np.count_nonzero(bubble_mask))
    area_ratio = positive_pixels / max(h * w, 1)
    if area_ratio < 0.12 or area_ratio > 0.98:
        return None

    return bubble_mask


def _get_bubble_crop_margin_ratio(box_width: int, box_height: int) -> float:
    short_side = max(1, min(int(box_width), int(box_height)))
    long_side = max(int(box_width), int(box_height))
    aspect_ratio = float(long_side) / float(short_side)
    if aspect_ratio <= 1.4:
        return 0.45
    return min(1.8, 0.45 + (aspect_ratio - 1.4) * 0.32)


def _build_bubble_mask_from_bboxes(
    image_cv: np.ndarray,
    bboxes: list,
    erode_px: int = 3,
) -> np.ndarray:
    h, w = image_cv.shape[:2]
    bubble_mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            continue
        margin = int(round(min(bw, bh) * 0.35))
        cx1 = max(0, x1 - margin)
        cy1 = max(0, y1 - margin)
        cx2 = min(w, x2 + margin)
        cy2 = min(h, y2 + margin)
        crop = image_cv[cy1:cy2, cx1:cx2]
        local_bubble = _extract_bubble_mask_from_crop(crop)
        if local_bubble is not None:
            bubble_mask[cy1:cy2, cx1:cx2] = cv2.bitwise_or(
                bubble_mask[cy1:cy2, cx1:cx2], local_bubble
            )
        else:
            inner_x1 = max(0, x1 - 2)
            inner_y1 = max(0, y1 - 2)
            inner_x2 = min(w, x2 + 2)
            inner_y2 = min(h, y2 + 2)
            cv2.rectangle(bubble_mask, (inner_x1, inner_y1), (inner_x2, inner_y2), 255, -1)
    if erode_px > 0 and np.any(bubble_mask):
        kernel_size = 2 * erode_px + 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(bubble_mask, erode_kernel, iterations=1)
        if np.any(eroded):
            bubble_mask = eroded
    return bubble_mask


def _clip_mask_to_bubble_mask(refined_mask: np.ndarray, bubble_mask: np.ndarray) -> np.ndarray:
    if not np.any(bubble_mask):
        return refined_mask
    refined_bin = np.where(refined_mask > 0, 255, 0).astype(np.uint8)
    bubble_bin = np.where(bubble_mask > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_bin, connectivity=8)
    clipped_mask = np.zeros_like(refined_bin)
    for label_idx in range(1, num_labels):
        x, y, bw, bh, area = stats[label_idx]
        if area <= 0:
            continue
        label_view = labels[y:y + bh, x:x + bw]
        region = label_view == label_idx
        bubble_region = bubble_bin[y:y + bh, x:x + bw] > 0
        intersection = region & bubble_region
        dst = clipped_mask[y:y + bh, x:x + bw]
        if np.any(intersection):
            dst[intersection] = 255
        else:
            dst[region] = 255
        clipped_mask[y:y + bh, x:x + bw] = dst
    return clipped_mask


def _sample_bbox_background(
    original: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    raw_mask: np.ndarray = None,
) -> np.ndarray | None:
    h, w = original.shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w, x2), min(h, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    margin = 30
    rx1, ry1 = max(0, x1 - margin), max(0, y1 - margin)
    rx2, ry2 = min(w, x2 + margin), min(h, y2 + margin)
    ring_mask = np.zeros((h, w), dtype=bool)
    ring_mask[ry1:ry2, rx1:rx2] = True
    ring_mask[cy1:cy2, cx1:cx2] = False
    ring_pixels_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)[ring_mask]
    if ring_pixels_gray.size >= 10:
        ring_dark_frac = np.count_nonzero(ring_pixels_gray < 120) / ring_pixels_gray.size
        if ring_dark_frac >= 0.4:
            ring_crop = original[ry1:ry2, rx1:rx2]
            ring_gray = cv2.cvtColor(ring_crop, cv2.COLOR_BGR2GRAY)
            non_bright = ring_gray < 180
            if np.count_nonzero(non_bright) >= 10:
                return ring_crop[non_bright]

    crop = original[cy1:cy2, cx1:cx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bg_mask = np.ones(crop.shape[:2], dtype=bool)
    if raw_mask is not None:
        local_raw = raw_mask[cy1:cy2, cx1:cx2]
        bg_mask[local_raw > 127] = False
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_mask = binary == 0
        if np.count_nonzero(text_mask) > 0 and np.count_nonzero(text_mask) < crop.size * 0.7:
            bg_mask[text_mask] = False
    bg_pixels = crop[bg_mask]
    if bg_pixels.size < 10:
        bg_pixels = crop.reshape(-1, 3)
    return bg_pixels


def _iter_safe_source_segments(
    safe_line: np.ndarray,
    start: int,
    end: int,
    sample_width: int,
) -> list[tuple[int, int]]:
    segments = []
    in_segment = False
    segment_start = 0
    line_length = int(safe_line.shape[0])

    for idx in range(line_length):
        is_safe = bool(safe_line[idx])
        if is_safe and not in_segment:
            segment_start = idx
            in_segment = True
        elif not is_safe and in_segment:
            segments.append((segment_start, idx))
            in_segment = False
    if in_segment:
        segments.append((segment_start, line_length))

    clipped_segments = []
    for seg_start, seg_end in segments:
        if seg_end - seg_start < 6:
            continue
        if not (seg_end <= start or seg_start >= end):
            continue
        if seg_end <= start:
            seg_start = max(seg_start, seg_end - sample_width)
        else:
            seg_end = min(seg_end, seg_start + sample_width)
        if seg_end - seg_start >= 6:
            clipped_segments.append((seg_start, seg_end))
    return clipped_segments


def _estimate_target_sample_luma(nearest_means: list[float]) -> float | None:
    if not nearest_means:
        return None
    clipped = np.clip(np.rint(nearest_means).astype(np.int32), 0, 255)
    if clipped.size < 6:
        return float(np.median(clipped))
    histogram = np.bincount(clipped, minlength=256).astype(np.float32)
    smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32) / 9.0
    histogram = np.convolve(histogram, smooth_kernel, mode="same")
    return float(int(np.argmax(histogram)))


def _sample_directional_background(
    original_patch: np.ndarray,
    target_mask: np.ndarray,
    safe_mask: np.ndarray,
    axis: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    target_pixels = cv2.findNonZero(np.where(target_mask > 0, 255, 0).astype(np.uint8))
    if target_pixels is None:
        return None, np.zeros_like(target_mask, dtype=np.uint8)

    bx, by, bw, bh = cv2.boundingRect(target_pixels)
    if bw <= 0 or bh <= 0:
        return None, np.zeros_like(target_mask, dtype=np.uint8)

    gray = cv2.cvtColor(original_patch, cv2.COLOR_BGR2GRAY)
    sampled = original_patch.copy()
    resolved_mask = np.zeros_like(target_mask, dtype=np.uint8)

    if axis == "row":
        sample_width = max(8, min(28, int(round(bw * 0.45))))
        line_range = range(by, by + bh)
        start = bx
        end = bx + bw
    else:
        sample_width = max(8, min(28, int(round(bh * 0.45))))
        line_range = range(bx, bx + bw)
        start = by
        end = by + bh

    def _collect_line_candidates(active_safe_mask: np.ndarray) -> tuple[dict, list[float]]:
        collected = {}
        means = []
        for line_idx in line_range:
            if axis == "row":
                line_mask = target_mask[line_idx, :] > 0
                safe_line = active_safe_mask[line_idx, :] > 0
                gray_line = gray[line_idx, :]
            else:
                line_mask = target_mask[:, line_idx] > 0
                safe_line = active_safe_mask[:, line_idx] > 0
                gray_line = gray[:, line_idx]

            if not np.any(line_mask):
                continue

            line_candidates = []
            for seg_start, seg_end in _iter_safe_source_segments(safe_line, start, end, sample_width):
                segment_values = gray_line[seg_start:seg_end]
                if segment_values.size < 6:
                    continue
                segment_mean = float(np.mean(segment_values))
                distance_to_target = float(min(abs(start - seg_end), abs(seg_start - end)))
                line_candidates.append((seg_start, seg_end, segment_mean, distance_to_target, seg_end - seg_start))

            if not line_candidates:
                continue

            collected[line_idx] = line_candidates
            nearest = min(line_candidates, key=lambda item: (item[3], abs(item[2] - 128.0), -item[4]))
            means.append(nearest[2])
        return collected, means

    candidates_by_line, nearest_means = _collect_line_candidates(np.where(safe_mask > 0, 255, 0).astype(np.uint8))

    target_luma = _estimate_target_sample_luma(nearest_means)
    if target_luma is None:
        return None, resolved_mask

    blurred_gray = cv2.GaussianBlur(gray, (0, 0), 2.0)
    tone_tolerance = max(18, min(34, int(round(np.std(nearest_means) * 1.15 + 16.0))))
    tone_safe_mask = np.where(
        (safe_mask > 0) & (np.abs(blurred_gray.astype(np.float32) - float(target_luma)) <= float(tone_tolerance)),
        255,
        0,
    ).astype(np.uint8)
    tone_safe_mask = cv2.morphologyEx(
        tone_safe_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    toned_candidates, _ = _collect_line_candidates(tone_safe_mask)
    if toned_candidates:
        candidates_by_line = toned_candidates

    for line_idx, line_candidates in candidates_by_line.items():
        best_start, best_end, _, _, _ = min(
            line_candidates,
            key=lambda item: (abs(item[2] - target_luma), item[3], -item[4]),
        )

        if axis == "row":
            source_pixels = original_patch[line_idx, best_start:best_end].copy()
            destination_positions = np.where(target_mask[line_idx, :] > 0)[0]
            if source_pixels.shape[0] == 0 or destination_positions.size == 0:
                continue
            for pixel_idx, dst_pos in enumerate(destination_positions):
                sampled[line_idx, dst_pos] = source_pixels[pixel_idx % source_pixels.shape[0]]
                resolved_mask[line_idx, dst_pos] = 255
        else:
            source_pixels = original_patch[best_start:best_end, line_idx].copy()
            destination_positions = np.where(target_mask[:, line_idx] > 0)[0]
            if source_pixels.shape[0] == 0 or destination_positions.size == 0:
                continue
            for pixel_idx, dst_pos in enumerate(destination_positions):
                sampled[dst_pos, line_idx] = source_pixels[pixel_idx % source_pixels.shape[0]]
                resolved_mask[dst_pos, line_idx] = 255

    if not np.any(resolved_mask):
        return None, resolved_mask

    return sampled, resolved_mask


def _propagate_unresolved_background(
    original_patch: np.ndarray,
    sampled_patch: np.ndarray,
    target_mask: np.ndarray,
    resolved_mask: np.ndarray,
    safe_mask: np.ndarray,
) -> np.ndarray:
    unresolved_mask = np.where((target_mask > 0) & (resolved_mask == 0), 255, 0).astype(np.uint8)
    if not np.any(unresolved_mask):
        return sampled_patch

    background_pixels = original_patch[safe_mask > 0]
    if background_pixels.size < 24:
        return sampled_patch

    median_background = np.median(background_pixels, axis=0).astype(np.uint8)
    seed_image = sampled_patch.copy()
    seed_image[(safe_mask == 0) & (target_mask == 0)] = median_background
    seed_image[unresolved_mask > 0] = median_background

    try:
        propagated = cv2.inpaint(seed_image, unresolved_mask, 3, cv2.INPAINT_TELEA)
    except Exception:
        return sampled_patch

    repaired = sampled_patch.copy()
    repaired[unresolved_mask > 0] = propagated[unresolved_mask > 0]
    return repaired


def _soften_fill_edges(sampled_patch: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    mask_uint8 = np.where(target_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask_uint8):
        return sampled_patch

    outer = cv2.dilate(mask_uint8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    inner = cv2.erode(mask_uint8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    edge_mask = np.where((outer > 0) & (inner == 0), 255, 0).astype(np.uint8)
    if not np.any(edge_mask):
        return sampled_patch

    smoothed = cv2.GaussianBlur(sampled_patch, (3, 3), 0)
    result = sampled_patch.copy()
    result[edge_mask > 0] = smoothed[edge_mask > 0]
    return result


def _build_sampled_background_fill(
    original_patch: np.ndarray,
    target_mask: np.ndarray,
    safe_mask: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    target_pixels = int(np.count_nonzero(target_mask))
    if target_pixels == 0 or np.count_nonzero(safe_mask) < 32:
        return None, 0.0

    points = cv2.findNonZero(np.where(target_mask > 0, 255, 0).astype(np.uint8))
    if points is None:
        return None, 0.0
    _, _, bw, bh = cv2.boundingRect(points)
    if bh >= int(round(bw * 1.15)):
        primary_axis = "row"
    elif bw >= int(round(bh * 1.15)):
        primary_axis = "col"
    else:
        primary_axis = "row"

    sampled, resolved_mask = _sample_directional_background(
        original_patch,
        target_mask,
        safe_mask,
        axis=primary_axis,
    )
    if sampled is None:
        fallback_axis = "col" if primary_axis == "row" else "row"
        sampled, resolved_mask = _sample_directional_background(
            original_patch,
            target_mask,
            safe_mask,
            axis=fallback_axis,
        )
        if sampled is None:
            return None, 0.0

    coverage = float(np.count_nonzero(resolved_mask > 0)) / float(target_pixels)
    if coverage <= 0.0:
        return None, 0.0

    sampled = _propagate_unresolved_background(
        original_patch,
        sampled,
        target_mask,
        resolved_mask,
        safe_mask,
    )
    sampled = _soften_fill_edges(sampled, target_mask)
    return sampled, coverage


def _build_target_ring_mask(target_mask: np.ndarray, safe_mask: np.ndarray) -> np.ndarray:
    target_uint8 = np.where(target_mask > 0, 255, 0).astype(np.uint8)
    inner = cv2.dilate(target_uint8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    outer = cv2.dilate(target_uint8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), iterations=1)
    ring_mask = np.where((outer > 0) & (inner == 0) & (safe_mask > 0), 255, 0).astype(np.uint8)
    if np.count_nonzero(ring_mask) < 24:
        ring_mask = np.where(safe_mask > 0, 255, 0).astype(np.uint8)
    return ring_mask


def _measure_mask_gray_stats(image_patch: np.ndarray, region_mask: np.ndarray) -> tuple[float, float] | None:
    gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
    values = gray[region_mask > 0]
    if values.size < 16:
        return None
    return float(np.mean(values)), float(np.std(values))


def _score_fill_candidate(
    image_patch: np.ndarray,
    target_mask: np.ndarray,
    reference_stats: tuple[float, float] | None,
) -> float | None:
    if reference_stats is None:
        return None
    candidate_stats = _measure_mask_gray_stats(image_patch, target_mask)
    if candidate_stats is None:
        return None
    mean_diff = abs(candidate_stats[0] - reference_stats[0])
    std_diff = abs(candidate_stats[1] - reference_stats[1])
    return mean_diff + std_diff * 0.85


def _compute_local_gray_std_map(gray_image: np.ndarray, window_size: int = 41) -> np.ndarray:
    kernel_size = max(3, int(window_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    gray_float = gray_image.astype(np.float32)
    mean = cv2.boxFilter(gray_float, cv2.CV_32F, (kernel_size, kernel_size), normalize=True)
    sq_mean = cv2.boxFilter(gray_float * gray_float, cv2.CV_32F, (kernel_size, kernel_size), normalize=True)
    variance = np.maximum(sq_mean - mean * mean, 0.0)
    return np.sqrt(variance)


def _blend_repaired_patch(
    current_patch: np.ndarray,
    repaired_patch: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    mask_uint8 = np.where(target_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask_uint8):
        return current_patch

    feather_mask = cv2.dilate(
        mask_uint8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    feather_mask = cv2.GaussianBlur(feather_mask, (0, 0), 2.2)
    alpha = (feather_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    blended = current_patch.astype(np.float32)
    repaired_float = repaired_patch.astype(np.float32)
    blended = repaired_float * alpha + blended * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _write_debug_image(path, image: np.ndarray) -> None:
    try:
        ok, buffer = cv2.imencode(".png", image)
        if ok:
            buffer.tofile(str(path))
    except Exception:
        pass


def _repair_background_from_current(
    current_patch: np.ndarray,
    target_mask: np.ndarray,
    safe_mask: np.ndarray,
) -> np.ndarray | None:
    target_uint8 = np.where(target_mask > 0, 255, 0).astype(np.uint8)
    safe_uint8 = np.where(safe_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(target_uint8):
        return None

    if np.count_nonzero(safe_uint8) < 24:
        safe_uint8 = np.where(target_uint8 == 0, 255, 0).astype(np.uint8)

    ring_mask = _build_target_ring_mask(target_uint8, safe_uint8)
    reference_stats = _measure_mask_gray_stats(current_patch, ring_mask)
    if reference_stats is None:
        return None

    try:
        telea_fill = cv2.inpaint(current_patch, target_uint8, 3, cv2.INPAINT_TELEA)
        telea_score = _score_fill_candidate(telea_fill, target_uint8, reference_stats)
        if telea_score is None:
            telea_score = 0.0
    except Exception:
        telea_fill = None
        telea_score = None

    best_candidate = telea_fill
    best_score = float(telea_score) if telea_score is not None else None

    try:
        ns_fill = cv2.inpaint(current_patch, target_uint8, 3, cv2.INPAINT_NS)
        ns_score = _score_fill_candidate(ns_fill, target_uint8, reference_stats)
        if ns_score is None:
            ns_score = 0.0
        if best_candidate is None or float(ns_score) < float(best_score):
            best_candidate = ns_fill
            best_score = float(ns_score)
    except Exception:
        pass

    if best_candidate is None:
        return None
    return _blend_repaired_patch(current_patch, best_candidate, target_uint8)


def _repair_background_with_mtu_patches(
    image: np.ndarray,
    repair_mask: np.ndarray,
    preferred_method: str = "lama_large",
) -> np.ndarray:
    repair_mask_u8 = np.where(repair_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(repair_mask_u8):
        return image

    from app.services.inpainting_lama import advanced_inpaint

    result = image.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (repair_mask_u8 > 0).astype(np.uint8),
        connectivity=8,
    )
    repaired_components = 0
    selected_method = preferred_method if preferred_method in {"aot", "lama", "lama_mpe", "lama_large"} else "lama_large"

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 160:
            continue

        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        ys, xs = np.where(component_mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue

        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        margin = 48
        rx1, ry1 = max(0, x1 - margin), max(0, y1 - margin)
        rx2, ry2 = min(image.shape[1], x2 + margin), min(image.shape[0], y2 + margin)

        patch = result[ry1:ry2, rx1:rx2].copy()
        patch_mask = component_mask[ry1:ry2, rx1:rx2]
        if np.count_nonzero(patch_mask) < 64:
            continue

        repaired_patch = advanced_inpaint(
            patch,
            patch_mask,
            method=selected_method,
            use_crf=False,
            verbose=False,
        )
        feather = cv2.GaussianBlur((patch_mask > 0).astype(np.float32), (0, 0), 3.0)
        feather = np.clip(feather, 0.0, 1.0)[:, :, np.newaxis]
        blended = repaired_patch.astype(np.float32) * feather + patch.astype(np.float32) * (1.0 - feather)
        result[ry1:ry2, rx1:rx2] = np.clip(blended, 0, 255).astype(np.uint8)
        repaired_components += 1

    logger.debug(
        f"[BG_FIX] MTU patch repair completed: components={repaired_components}, "
        f"mask_pixels={int(np.count_nonzero(repair_mask_u8))}, method={selected_method}"
    )
    return result


def _fix_dark_background_after_inpaint(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    bboxes: list,
    raw_mask: np.ndarray,
    repair_method: str = "lama_large",
) -> np.ndarray:
    result = inpainted.copy()
    h, w = original.shape[:2]
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    diag_dir = (PROJECT_ROOT / "diag") if DEBUG_DIAG else None
    debug_fix_mask = np.zeros((h, w), dtype=np.uint8)
    fallback_repairs: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    if diag_dir is not None:
        try:
            diag_dir.mkdir(parents=True, exist_ok=True)
            _write_debug_image(diag_dir / "before_bg_fix.png", inpainted)
        except Exception:
            diag_dir = None

    dark_binary = (gray_orig < 120).astype(np.float32)
    local_dark_frac = cv2.boxFilter(dark_binary, -1, (51, 51), normalize=True)
    local_texture_std = _compute_local_gray_std_map(gray_orig, window_size=41)
    refined_mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)
    raw_mask_bin = (
        np.where(raw_mask > 0, 255, 0).astype(np.uint8)
        if raw_mask is not None
        else np.zeros_like(refined_mask_bin)
    )

    _diag_print(f"[BG_FIX] Processing {len(bboxes)} bboxes, image={w}x{h}")

    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(w, x2), min(h, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            _diag_print(f"[BG_FIX] bbox[{idx}] ({x1},{y1})-({x2},{y2}): invalid coords, skip")
            continue

        margin = 30
        rx1, ry1 = max(0, x1 - margin), max(0, y1 - margin)
        rx2, ry2 = min(w, x2 + margin), min(h, y2 + margin)

        ring_mask = np.zeros((h, w), dtype=bool)
        ring_mask[ry1:ry2, rx1:rx2] = True
        ring_mask[cy1:cy2, cx1:cx2] = False

        ring_pixels = gray_orig[ring_mask]
        if ring_pixels.size < 10:
            continue
        ring_dark_frac = np.count_nonzero(ring_pixels < 120) / ring_pixels.size
        ring_std = float(np.std(ring_pixels))

        ex1, ey1 = max(0, x1 - 80), max(0, y1 - 80)
        ex2, ey2 = min(w, x2 + 80), min(h, y2 + 80)

        local_frac_region = local_dark_frac[ey1:ey2, ex1:ex2]
        local_texture_region = local_texture_std[ey1:ey2, ex1:ex2]
        gray_region = gray_orig[ey1:ey2, ex1:ex2]
        bubble_region = (local_frac_region > 0.28) | (
            (local_texture_region > max(8.0, ring_std * 0.55))
            & (gray_region < 245)
        )

        inp_region = result[ey1:ey2, ex1:ex2]
        inp_gray = cv2.cvtColor(inp_region, cv2.COLOR_BGR2GRAY)

        focus_mask = cv2.bitwise_or(
            refined_mask_bin[ey1:ey2, ex1:ex2],
            raw_mask_bin[ey1:ey2, ex1:ex2],
        )
        if np.any(focus_mask):
            focus_mask = cv2.dilate(
                focus_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                iterations=1,
            )
        else:
            focus_mask = np.zeros_like(inp_gray, dtype=np.uint8)
            local_x1 = max(0, x1 - ex1)
            local_y1 = max(0, y1 - ey1)
            local_x2 = min(ex2 - ex1, x2 - ex1)
            local_y2 = min(ey2 - ey1, y2 - ey1)
            focus_mask[local_y1:local_y2, local_x1:local_x2] = 255

        focus_active = focus_mask > 0
        focus_pixels = max(int(np.count_nonzero(focus_active)), 1)
        bubble_focus_ratio = float(np.count_nonzero(bubble_region & focus_active)) / float(focus_pixels)
        provisional_bg_mask = bubble_region & (~focus_active)
        provisional_bg_pixels = gray_region[provisional_bg_mask]
        if provisional_bg_pixels.size < 24:
            provisional_bg_pixels = gray_region[~focus_active]
        provisional_bg_luma = (
            float(np.median(provisional_bg_pixels))
            if provisional_bg_pixels.size > 0
            else float(np.median(gray_region))
        )
        is_dark_context = bool(
            ((bubble_focus_ratio >= 0.24) and (provisional_bg_luma < 170.0))
            or (ring_dark_frac >= 0.22)
        )

        if is_dark_context:
            bg_sample_mask = bubble_region & (~focus_active)
        else:
            bg_sample_mask = ~focus_active
        bg_pixels_gray = inp_gray[bg_sample_mask]
        if bg_pixels_gray.size < 24:
            bg_pixels_gray = gray_region[bg_sample_mask]
        if bg_pixels_gray.size < 24:
            bg_pixels_gray = inp_gray[~focus_active]
        bg_luma = float(np.median(bg_pixels_gray)) if bg_pixels_gray.size > 0 else float(np.median(inp_gray))
        bright_threshold = max(bg_luma + 34, 170)
        bright_mask = inp_gray > bright_threshold
        dark_threshold = min(140, max(36, int(round(bg_luma - 26))))
        dark_mask = inp_gray < dark_threshold

        if is_dark_context:
            candidate_region = bubble_region & (focus_mask > 0)
            fix_mask = np.where(candidate_region & bright_mask, 255, 0).astype(np.uint8)
        else:
            candidate_region = focus_mask > 0
            fix_mask = np.where(candidate_region & dark_mask, 255, 0).astype(np.uint8)
        if np.any(fix_mask):
            fix_mask = cv2.morphologyEx(
                fix_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            expand_kernel = 11 if is_dark_context else 7
            fix_mask = cv2.dilate(
                fix_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_kernel, expand_kernel)),
                iterations=1,
            )
            if is_dark_context:
                allowed_region = cv2.dilate(
                    np.where(bubble_region, 255, 0).astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                    iterations=1,
                )
            else:
                allowed_region = cv2.dilate(
                    focus_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                    iterations=1,
                )
            fix_mask = cv2.bitwise_and(fix_mask, allowed_region)

        if not np.any(fix_mask):
            continue

        fix_count = int(np.count_nonzero(fix_mask))
        logger.debug(
            f"[BG_FIX] bbox[{idx}] ({x1},{y1})-({x2},{y2}): "
            f"mode={'dark' if is_dark_context else 'light'}, "
            f"bubble_focus_ratio={bubble_focus_ratio:.2f}, "
            f"ring_dark_frac={ring_dark_frac:.2f}, bg_luma={bg_luma:.1f}, "
            f"bright_thresh={bright_threshold:.0f}, dark_thresh={dark_threshold:.0f}, "
            f"fix_pixels={fix_count}"
        )
        _diag_print(
            f"[BG_FIX] FIXED bbox[{idx}] ({x1},{y1})-({x2},{y2}): "
            f"mode={'dark' if is_dark_context else 'light'}, "
            f"bubble_focus_ratio={bubble_focus_ratio:.2f}, "
            f"ring_dark={ring_dark_frac:.2f}, bg_luma={bg_luma:.1f}, "
            f"fix_pixels={fix_count}"
        )

        safe_mask = np.where(
            (bubble_region if is_dark_context else (focus_mask == 0)) & (focus_mask == 0),
            255,
            0,
        ).astype(np.uint8)
        fallback_repairs.append((ey1, ey2, ex1, ex2, fix_mask.copy(), safe_mask.copy()))
        debug_fix_mask[ey1:ey2, ex1:ex2] = np.maximum(debug_fix_mask[ey1:ey2, ex1:ex2], fix_mask)

    if np.any(debug_fix_mask):
        try:
            result = _repair_background_with_mtu_patches(
                result,
                debug_fix_mask,
                preferred_method=repair_method,
            )
        except Exception as exc:
            logger.warning(f"[BG_FIX] MTU patch repair failed, fallback to local repair: {exc}")
            for ey1, ey2, ex1, ex2, local_fix_mask, local_safe_mask in fallback_repairs:
                result_crop = result[ey1:ey2, ex1:ex2]
                repaired_crop = _repair_background_from_current(
                    result_crop,
                    local_fix_mask,
                    local_safe_mask,
                )
                if repaired_crop is not None:
                    result[ey1:ey2, ex1:ex2] = repaired_crop

    if diag_dir is not None:
        try:
            _write_debug_image(diag_dir / "bg_fix_mask.png", debug_fix_mask)
            _write_debug_image(diag_dir / "after_bg_fix.png", result)
            (diag_dir / "bg_fix_meta.txt").write_text(
                (
                    f"ts={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"pid={os.getpid()}\n"
                    f"repair_method={repair_method}\n"
                    f"mask_pixels={int(np.count_nonzero(debug_fix_mask))}\n"
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
    return result


def _get_text_bitmap_alpha_mask(text_bitmap: np.ndarray) -> np.ndarray:
    if text_bitmap is None or text_bitmap.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if len(text_bitmap.shape) == 3 and text_bitmap.shape[2] == 4:
        return np.where(text_bitmap[:, :, 3] > 0, 255, 0).astype(np.uint8)
    if len(text_bitmap.shape) == 3:
        return np.where(np.any(text_bitmap[:, :, :3] > 0, axis=2), 255, 0).astype(np.uint8)
    return np.where(text_bitmap > 0, 255, 0).astype(np.uint8)


def _resolve_bitmap_paste_origin(
    x: int,
    y: int,
    box_width: int,
    box_height: int,
    bitmap_width: int,
    bitmap_height: int,
    align_x: str = "center",
    align_y: str = "center",
) -> tuple[int, int]:
    if align_x == "left":
        paste_x = x
    elif align_x == "right":
        paste_x = x + box_width - bitmap_width
    else:
        paste_x = x + (box_width - bitmap_width) // 2

    if align_y == "top":
        paste_y = y
    elif align_y in ("bottom", "right"):
        paste_y = y + box_height - bitmap_height
    else:
        paste_y = y + (box_height - bitmap_height) // 2

    return paste_x, paste_y


def _bitmap_fits_bubble_mask(
    text_bitmap: np.ndarray,
    bubble_mask: np.ndarray | None,
    box_width: int,
    box_height: int,
    align_x: str = "center",
    align_y: str = "center",
) -> bool:
    if bubble_mask is None or bubble_mask.size == 0 or not np.any(bubble_mask):
        return True

    alpha_mask = _get_text_bitmap_alpha_mask(text_bitmap)
    if alpha_mask.size == 0 or not np.any(alpha_mask):
        return True

    mask_canvas = bubble_mask
    if mask_canvas.shape[:2] != (box_height, box_width):
        mask_canvas = cv2.resize(mask_canvas, (box_width, box_height), interpolation=cv2.INTER_NEAREST)

    mask_pixels = int(np.count_nonzero(mask_canvas))
    if mask_pixels == 0:
        return True

    safe_mask = mask_canvas
    if mask_pixels > 64:
        eroded = cv2.erode(
            mask_canvas,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        if np.count_nonzero(eroded) > max(int(mask_pixels * 0.55), 32):
            safe_mask = eroded

    th, tw = alpha_mask.shape[:2]
    paste_x, paste_y = _resolve_bitmap_paste_origin(
        0, 0, box_width, box_height, tw, th, align_x=align_x, align_y=align_y
    )

    src_x = max(0, -paste_x)
    src_y = max(0, -paste_y)
    dst_x = max(0, paste_x)
    dst_y = max(0, paste_y)
    dst_w = min(tw - src_x, box_width - dst_x)
    dst_h = min(th - src_y, box_height - dst_y)
    if dst_w <= 0 or dst_h <= 0:
        return False

    placed = np.zeros((box_height, box_width), dtype=np.uint8)
    placed[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w] = alpha_mask[src_y:src_y + dst_h, src_x:src_x + dst_w]
    text_pixels = int(np.count_nonzero(placed))
    if text_pixels == 0:
        return True

    inside_pixels = int(np.count_nonzero(cv2.bitwise_and(placed, safe_mask)))
    overflow_pixels = text_pixels - inside_pixels
    return overflow_pixels <= max(6, int(text_pixels * 0.01))


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

    # 使用椭圆形核膨胀（更自然的边缘过渡）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE_KERNEL_SIZE, MASK_DILATE_KERNEL_SIZE))
    mask = cv2.dilate(mask, kernel, iterations=MASK_DILATE_ITERATIONS)

    return mask

def _solidify_local_text_mask(local_mask: np.ndarray) -> np.ndarray:
    """Turn edge-heavy local text features back into solid glyph masks before inpainting."""
    if local_mask is None or local_mask.size == 0 or not np.any(local_mask):
        return local_mask

    height, width = local_mask.shape[:2]
    close_side = 3 if min(height, width) < 72 else 5
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_side, close_side))
    solid = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros_like(solid)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    else:
        filled = solid

    return cv2.dilate(
        filled,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )


def _tighten_refined_mask_with_local_text_features(
    image_cv: np.ndarray,
    refined_mask: np.ndarray,
    bboxes: list,
) -> np.ndarray:
    """Shrink oversized detector masks back toward local text strokes for simple bubbles."""
    if refined_mask is None or refined_mask.size == 0 or not np.any(refined_mask) or not bboxes:
        return refined_mask

    height, width = image_cv.shape[:2]
    tightened = np.zeros_like(refined_mask)

    for bbox in bboxes:
        x1, y1, x2, y2 = _sanitize_bbox(bbox, width, height)
        region_mask = refined_mask[y1:y2, x1:x2]
        if region_mask.size == 0 or not np.any(region_mask):
            continue

        cropped = image_cv[y1:y2, x1:x2]
        local_mask = _build_text_mask_enhanced(cropped)
        if local_mask.size == 0 or not np.any(local_mask):
            tightened[y1:y2, x1:x2] = cv2.bitwise_or(tightened[y1:y2, x1:x2], region_mask)
            continue

        local_mask = _solidify_local_text_mask(local_mask)
        overlap_mask = cv2.bitwise_and(region_mask, local_mask)
        overlap_pixels = int(np.count_nonzero(overlap_mask))
        region_pixels = int(np.count_nonzero(region_mask))
        local_pixels = int(np.count_nonzero(local_mask))
        if region_pixels == 0 or local_pixels == 0:
            tightened[y1:y2, x1:x2] = cv2.bitwise_or(tightened[y1:y2, x1:x2], region_mask)
            continue

        coverage = overlap_pixels / max(local_pixels, 1)
        shrink_ratio = local_pixels / max(region_pixels, 1)
        median_luma = float(np.median(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)))
        use_local_tightening = coverage >= 0.18 and (
            shrink_ratio <= 0.82
            or median_luma < 145
            or median_luma > 215
        )

        if use_local_tightening:
            chosen_mask = local_mask
        else:
            chosen_mask = region_mask

        tightened[y1:y2, x1:x2] = cv2.bitwise_or(tightened[y1:y2, x1:x2], chosen_mask)

    if np.count_nonzero(tightened) == 0:
        return refined_mask

    return tightened


def _apply_inpaint(image_cv: np.ndarray, mask: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    应用 inpainting 算法（增强版）
    
    移植自 manga-translator-ui (MTU) 项目的改进：
    - 分块处理：对极端长宽比图片自动分块
    - Alpha 通道处理：支持 RGBA 图像
    - 内存清理：推理后自动清理 GPU 内存
    
    支持的方法:
    - aot: AOT Inpainting（MIT 默认模型，效果最好，推荐）
    - telea: OpenCV TELEA 算法（快速）
    - ns: OpenCV Navier-Stokes 算法（质量更好）
    - lama: Lama-MPE 深度学习模型
    - lama_large: Lama-Large 深度学习模型（更大更准确）
    """
    if not np.any(mask):
        return image_cv
    
    # 从配置获取 inpainting 方法
    configured_method, configured_use_crf = _get_inpaint_config()
    mode, runtime_method, runtime_use_crf, _, _ = _resolve_inpaint_runtime(
        configured_method,
        configured_use_crf,
    )
    
    # 使用高级 inpainting（所有方法都走增强版接口）
    try:
        from app.services.inpainting_lama import advanced_inpaint
        return advanced_inpaint(
            image_cv, mask, 
            method=runtime_method,
            use_crf=runtime_use_crf,
            inpainting_size=LAMA_INPAINT_SIZE,
            verbose=verbose
        )
    except Exception as e:
        import traceback
        logger.warning(f"高级 inpainting 失败，回退到 OpenCV: {e}\n{traceback.format_exc()}")
        _diag_print(f"[INPAINT_DIAG] _apply_inpaint failed: {e}\n{traceback.format_exc()}")
        runtime_method = "telea" if mode == "fast" else "ns"
    
    # OpenCV 传统方法（回退）
    if runtime_method == "ns":
        return cv2.inpaint(image_cv, mask, inpaintRadius=INPAINT_RADIUS, flags=cv2.INPAINT_NS)
    else:
        return cv2.inpaint(image_cv, mask, inpaintRadius=INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)


async def get_text_masked_pic(image_pil, image_cv, bboxes, inpaint=True, ocr_engine: str = None):
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
    if len(bboxes) == 0:
        return [], image_cv, mask

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
        image_cv = _apply_inpaint(image_cv, mask)

    return all_text, image_cv, mask


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


def wrap_text_vertical(text, font, max_height, max_width=None):
    """
    将文字根据实际像素高度分列，返回列列表（竖排模式）
    竖排：从右到左，每列从上到下
    返回的列表中，索引0是最先读到的文字（应画在最右边）
    
    Args:
        text: 文本
        font: 字体
        max_height: 最大列高度
        max_width: 最大总宽度（用于限制列数）
    """
    # 先处理字面量换行符
    text = _normalize_newlines(text)
    
    if not text:
        return []
    
    # 获取字符高度（使用多个字符取平均）
    char_height = font.getbbox("中")[3] - font.getbbox("中")[1]
    char_width = font.getbbox("中")[2] - font.getbbox("中")[0]
    
    # 计算最多能容纳多少列
    col_spacing = 4
    if max_width:
        max_cols = max(1, int((max_width + col_spacing) / (char_width + col_spacing)))
    else:
        max_cols = 999
    
    # 计算每列最多能放多少字符
    max_chars_per_col = max(1, max_height // char_height)
    
    # 如果文本很短，直接放一列
    if len(text) <= max_chars_per_col:
        return [text]
    
    # 计算最优列数：尽量让每列字符数接近
    text_len = len(text)
    best_cols = 1
    best_score = float('inf')
    
    for num_cols in range(1, min(max_cols, text_len) + 1):
        # 计算每列字符数
        base_chars = text_len // num_cols
        remainder = text_len % num_cols
        
        # 检查是否超出高度限制
        if base_chars + (1 if remainder > 0 else 0) > max_chars_per_col:
            continue
        
        # 评分：字符数差异越小越好
        max_chars = base_chars + (1 if remainder > 0 else 0)
        min_chars = base_chars
        score = max_chars - min_chars  # 差异
        
        if score < best_score:
            best_score = score
            best_cols = num_cols
    
    # 按最优列数分配字符
    columns = []
    chars_per_col = text_len // best_cols
    remainder = text_len % best_cols
    
    idx = 0
    for i in range(best_cols):
        col_len = chars_per_col + (1 if i < remainder else 0)
        columns.append(text[idx:idx + col_len])
        idx += col_len
    
    return columns


def _sample_surrounding_region(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, margin: int = 20) -> np.ndarray:
    h, w = image.shape[:2]
    sx1 = max(0, x1 - margin)
    sy1 = max(0, y1 - margin)
    sx2 = min(w, x2 + margin)
    sy2 = min(h, y2 + margin)
    outer = image[sy1:sy2, sx1:sx2]
    inner_y1 = y1 - sy1
    inner_x1 = x1 - sx1
    inner_y2 = inner_y1 + (y2 - y1)
    inner_x2 = inner_x1 + (x2 - x1)
    mask = np.ones(outer.shape[:2], dtype=bool)
    iy1 = max(0, inner_y1)
    ix1 = max(0, inner_x1)
    iy2 = min(outer.shape[0], inner_y2)
    ix2 = min(outer.shape[1], inner_x2)
    mask[iy1:iy2, ix1:ix2] = False
    return outer[mask]


def _pick_text_style(cropped_cv: np.ndarray, surrounding_region: np.ndarray = None, bg_pixels: np.ndarray = None):
    if bg_pixels is not None and bg_pixels.size > 0:
        b, g, r = bg_pixels[:, 0].astype(np.float32), bg_pixels[:, 1].astype(np.float32), bg_pixels[:, 2].astype(np.float32)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        median_luma = float(np.median(luma))
    elif surrounding_region is not None and surrounding_region.size > 0:
        if surrounding_region.ndim == 2 and surrounding_region.shape[1] == 3:
            b, g, r = surrounding_region[:, 0].astype(np.float32), surrounding_region[:, 1].astype(np.float32), surrounding_region[:, 2].astype(np.float32)
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            median_luma = float(np.median(luma))
        else:
            gray = cv2.cvtColor(surrounding_region, cv2.COLOR_BGR2GRAY)
            median_luma = float(np.median(gray))
    elif cropped_cv.size > 0:
        gray = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2GRAY)
        median_luma = float(np.median(gray))
    else:
        return (20, 20, 20), (245, 245, 245)
    if median_luma >= 145:
        return (20, 20, 20), (245, 245, 245)
    return (245, 245, 245), (20, 20, 20)


def _adjust_font_to_fit(draw, text, initial_font, box_width, box_height, text_direction: str, font_path: str):
    """
    自适应调整字体大小，确保文字能够放入气泡
    
    Args:
        draw: PIL ImageDraw 对象
        text: 待渲染文本
        initial_font: 初始字体
        box_width: 气泡宽度
        box_height: 气泡高度
        text_direction: 文字方向
        font_path: 字体路径
    
    Returns:
        调整后的字体
    """
    line_spacing = 4
    font = initial_font
    font_size = font.size
    
    # 最大尝试次数，避免无限循环
    max_attempts = 20
    
    for _ in range(max_attempts):
        # 获取字符尺寸
        char_bbox = font.getbbox("中")
        char_height = char_bbox[3] - char_bbox[1]
        char_width = int(draw.textlength("中", font=font))
        
        if text_direction == "vertical":
            # 竖排模式检查
            columns = wrap_text_vertical(text, font, box_height, box_width)
            if not columns:
                break
            
            # 计算总宽度（所有列）
            col_spacing = line_spacing
            total_width = len(columns) * char_width + max(0, len(columns) - 1) * col_spacing
            
            # 检查是否超出
            if total_width <= box_width and char_height <= box_height:
                break
        else:
            # 横排模式检查
            lines = wrap_text_by_width(draw, text, font, box_width)
            if not lines:
                break
            
            # 计算总高度（所有行）
            total_height = char_height * len(lines) + line_spacing * max(0, len(lines) - 1)
            
            # 检查最宽的一行
            max_line_width = max(draw.textlength(line, font=font) for line in lines)
            
            # 检查是否超出
            if total_height <= box_height and max_line_width <= box_width:
                break
        
        # 字体太大，缩小
        font_size = max(8, font_size - 2)
        if font_size <= 8:
            break
        # 使用 FontConfig 的缓存字体加载
        from app.core.font_conf import _load_font
        font = _load_font(font_path, font_size)
    
    return font


def draw_text_on_boxes(image: np.ndarray, boxes: list, texts: list, text_direction: str = "horizontal", raw_mask: np.ndarray = None) -> np.ndarray:
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
        
        # 创建字体配置（传入文字方向）
        font_config = FontConfig(box_height, box_width, text, text_direction=text_direction)
        font = font_config.font
        font_path = font_config.font_path
        
        fill_color, stroke_color = _pick_text_style(
            image[y1:y2, x1:x2],
            _sample_surrounding_region(image, x1, y1, x2, y2),
            bg_pixels=_sample_bbox_background(image, x1, y1, x2, y2, raw_mask),
        )
        stroke_width = max(1, int(font.size * 0.12))
        
        if text_direction == "vertical":
            # ===== 竖排模式 =====
            char_bbox = font.getbbox("中")
            char_height = char_bbox[3] - char_bbox[1]
            char_width = int(draw.textlength("中", font=font))
            columns = wrap_text_vertical(text, font, box_height, box_width)
            
            # 计算总宽度
            col_spacing = line_spacing
            total_width = len(columns) * char_width + max(0, len(columns) - 1) * col_spacing
            
            # 自适应调整：如果总宽度超出气泡，缩小字体
            if total_width > box_width:
                font = _adjust_font_to_fit(draw, text, font, box_width, box_height, text_direction, font_path)
                char_bbox = font.getbbox("中")
                char_height = char_bbox[3] - char_bbox[1]
                char_width = int(draw.textlength("中", font=font))
                columns = wrap_text_vertical(text, font, box_height, box_width)
                total_width = len(columns) * char_width + max(0, len(columns) - 1) * col_spacing
                stroke_width = max(1, int(font.size * 0.12))
            
            # 起始位置（水平居中）
            start_x = x1 + max(0, (box_width - total_width) // 2)
            
            for col_idx, col in enumerate(columns):
                # 从右到左绘制列
                col_x = start_x + (len(columns) - 1 - col_idx) * (char_width + col_spacing)
                # 所有列从同一高度开始（居中），保持视觉统一
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
            # ===== 横排模式 =====
            lines = wrap_text_by_width(draw, text, font, box_width)
            char_bbox = font.getbbox("中")
            line_height = char_bbox[3] - char_bbox[1]
            total_height = line_height * len(lines) + line_spacing * max(0, len(lines) - 1)
            
            # 检查是否有行超出宽度
            max_line_width = max((draw.textlength(line, font=font) for line in lines), default=0)
            
            # 自适应调整：如果总高度超出气泡，缩小字体
            if total_height > box_height or max_line_width > box_width:
                font = _adjust_font_to_fit(draw, text, font, box_width, box_height, text_direction, font_path)
                lines = wrap_text_by_width(draw, text, font, box_width)
                char_bbox = font.getbbox("中")
                line_height = char_bbox[3] - char_bbox[1]
                total_height = line_height * len(lines) + line_spacing * max(0, len(lines) - 1)
                stroke_width = max(1, int(font.size * 0.12))
            
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


# ============ MIT 高级处理流程（推荐使用）============

# 是否使用 MIT 检测器（默认 True）
USE_MIT_DETECTOR = os.getenv("USE_MIT_DETECTOR", "true").lower() in ("true", "1", "yes")

async def get_text_masked_pic_mit(
    image_pil, 
    image_cv, 
    detect_size: int = 1024,
    text_threshold: float = 0.3,
    box_threshold: float = 0.5,
    inpaint: bool = True, 
    ocr_engine: str = None,
    text_direction: str = "horizontal",
    use_crf: bool = True,
    use_yolo_obb: bool = True,  # 是否启用 YOLO OBB 辅助检测
    yolo_obb_conf: float = 0.4,  # YOLO OBB 置信度阈值
):
    """
    使用 MIT 检测器的高级处理流程（推荐）
    
    整合了以下改进（移植自 manga-translator-ui）：
    1. MIT DBNet 检测器（输出 bbox + raw_mask）
    2. YOLO OBB 辅助检测器（可选）
    3. 增强版掩码细化（连通组件分析 + CRF）
    4. 增强版 Inpainting（分块处理 + 内存优化 + 精度控制）
    
    Args:
        image_pil: PIL 图像
        image_cv: OpenCV BGR 图像
        detect_size: 检测尺寸
        text_threshold: 文本阈值
        box_threshold: 框阈值
        inpaint: 是否进行 inpainting
        ocr_engine: OCR 引擎
        use_crf: 是否使用 CRF 细化
        use_yolo_obb: 是否启用 YOLO OBB 辅助检测（默认启用）
        yolo_obb_conf: YOLO OBB 置信度阈值（默认 0.4）
        
    Returns:
        all_text: OCR 识别的文本列表
        image_cv: 处理后的图像
        bboxes: 边界框列表
        raw_mask: 原始掩码（可用于调试）
        merge_map: 合并映射
    """
    from app.services.mit_detector import detect_text_regions_hybrid
    from app.services.mask_refinement import complete_mask
    from app.services.inpainting_lama import advanced_inpaint
    from app.services.mtu_mask_refinement_bridge import (
        mtu_refine_mask_async,
        supports_mtu_mask_refinement,
    )
    
    # 1. 使用混合检测器（MIT + YOLO OBB）检测文本区域
    bboxes, raw_mask = detect_text_regions_hybrid(
        image_cv,
        detect_size=detect_size,
        text_threshold=text_threshold,
        box_threshold=box_threshold,
        use_yolo_obb=use_yolo_obb,
        yolo_obb_conf=yolo_obb_conf,
    )
    
    if len(bboxes) == 0:
        return [], image_cv, [], raw_mask, None
    
    height, width = image_cv.shape[:2]
    
    # 2. OCR 识别
    semaphore = asyncio.Semaphore(min(OCR_MAX_CONCURRENCY, len(bboxes)))
    
    async def ocr_region(bbox):
        x1, y1, x2, y2 = _sanitize_bbox(bbox, width, height)
        cropped_image = image_pil.crop((x1, y1, x2, y2))
        async with semaphore:
            text = await asyncio.to_thread(ocr_recognize, cropped_image, ocr_engine)
        return text, (x1, y1, x2, y2)
    
    tasks = [ocr_region(bbox) for bbox in bboxes]
    results = await asyncio.gather(*tasks)
    all_text = [text for text, _ in results]
    all_bboxes = [list(bbox) for _, bbox in results]
    mask_bboxes = [bbox[:] for bbox in all_bboxes]
    
    # 输出 OCR 识别结果（带位置信息）
    logger.debug(f"OCR 识别完成，共 {len(all_text)} 条文本:")
    for i, (text, bbox) in enumerate(results):
        x1, y1, x2, y2 = bbox
        logger.debug(f"  [{i+1}] ({x1},{y1})-({x2},{y2}): {text}")
    
    # 文本框合并（如果启用）
    merge_map = None
    render_groups = _build_render_groups(all_bboxes, text_direction)
    try:
        from app.core.custom_conf import custom_conf
        merge_enabled = getattr(custom_conf, "merge_textboxes", MERGE_TEXTBOXES)
    except Exception:
        merge_enabled = MERGE_TEXTBOXES
    auto_merge_vertical = text_direction == "vertical" and any(len(group) > 1 for group in render_groups)
    if len(all_text) > 1 and (merge_enabled or auto_merge_vertical):
        if auto_merge_vertical and not merge_enabled:
            logger.debug("Auto-merge split vertical boxes inside the same bubble for translation/render.")
        if text_direction == "vertical":
            all_text, all_bboxes, merge_map = _merge_textboxes_with_groups(
                all_text,
                all_bboxes,
                render_groups,
                text_direction=text_direction,
            )
        else:
            all_text, all_bboxes, merge_map = _merge_textboxes(all_text, all_bboxes)
        logger.debug(f"OCR 识别到 {len(all_text)} 条文本（合并后）:")
        for i, text in enumerate(all_text):
            logger.debug(f"  [{i+1}] {text}")
    
    # 计算每个文本框的字体大小（基于框高度，用于自适应掩码膨胀）
    # 移植自 MTU：font_size = min(int(box_height * 0.8), 128)
    _log_ocr_summary(all_text, merged=merge_map is not None)

    mask_font_sizes = [_estimate_mask_font_size(bbox) for bbox in mask_bboxes]
    mask_kernel_size = max(3, MASK_DILATE_KERNEL_SIZE - 2)
    if mask_kernel_size % 2 == 0:
        mask_kernel_size += 1

    configured_method, configured_use_crf = _get_inpaint_config()
    inpaint_mode, runtime_method, runtime_use_crf, allow_mtu_mask_bridge, enable_bg_fix = _resolve_inpaint_runtime(
        configured_method,
        configured_use_crf,
    )
    logger.debug(
        f"[INPAINT_PROFILE] mode={inpaint_mode}, runtime_method={runtime_method}, "
        f"mtu_mask_bridge={allow_mtu_mask_bridge}, bg_fix={enable_bg_fix}"
    )
    
    # 3. 掩码细化（优先直接走 MTU 原生 refined mask，失败时回退到本地实现）
    used_mtu_mask_bridge = False
    refined_mask = None
    if runtime_use_crf and allow_mtu_mask_bridge and supports_mtu_mask_refinement():
        try:
            refined_mask = await mtu_refine_mask_async(
                image_cv,
                raw_mask,
                mask_bboxes,
                font_sizes=mask_font_sizes,
                text_direction=text_direction,
                dilation_offset=MTU_MASK_DILATION_OFFSET,
                kernel_size=mask_kernel_size,
                verbose=False,
            )
            used_mtu_mask_bridge = bool(np.any(refined_mask))
            if used_mtu_mask_bridge:
                logger.debug(
                    f"[MASK] Embedded MTU refined mask ready: pixels={int(np.count_nonzero(refined_mask))}"
                )
        except Exception as e:
            logger.warning(f"[MASK] Embedded MTU mask refinement failed, fallback to local path: {e}")

    if refined_mask is None:
        if runtime_use_crf:
            refined_mask = complete_mask(
                image_cv,
                raw_mask,
                mask_bboxes,
                use_crf=True,
                kernel_size=mask_kernel_size,
                font_sizes=mask_font_sizes,
            )
        else:
            # 简单膨胀
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (mask_kernel_size, mask_kernel_size)
            )
            refined_mask = cv2.dilate(raw_mask, kernel, iterations=max(1, MASK_DILATE_ITERATIONS - 1))

    if not used_mtu_mask_bridge:
        refined_mask = _tighten_refined_mask_with_local_text_features(
            image_cv,
            refined_mask,
            mask_bboxes,
        )
        if np.any(refined_mask):
            refined_mask = cv2.dilate(
                refined_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
    
    # 3.5 Bubble mask 约束（照搬 MTU 的 limit_mask_dilation_to_bubble_mask）
    original_for_inpaint = image_cv.copy()
    if not used_mtu_mask_bridge:
        try:
            bubble_mask = _build_bubble_mask_from_bboxes(image_cv, mask_bboxes, erode_px=3)
            if np.any(bubble_mask):
                mask_before_clip = refined_mask.copy()
                refined_mask = _clip_mask_to_bubble_mask(refined_mask, bubble_mask)
                removed_pixels = int(np.count_nonzero(mask_before_clip)) - int(np.count_nonzero(refined_mask))
                logger.debug(
                    f"[BUBBLE_CLIP] bubble_mask_pixels={np.count_nonzero(bubble_mask)}, removed_pixels={removed_pixels}"
                )
                _diag_print(f"[INPAINT_DIAG] BUBBLE_CLIP: removed_pixels={removed_pixels}")
        except Exception as e:
            logger.warning(f"[BUBBLE_CLIP] bubble mask 约束失败: {e}")
    
    # 4. Inpainting（使用增强版接口）
    if inpaint and np.any(refined_mask):
        logger.debug(
            f"[INPAINT] 开始 inpainting: mode={inpaint_mode}, runtime_method={runtime_method}, "
            f"mask_pixels={np.count_nonzero(refined_mask)}, mask_shape={refined_mask.shape}"
        )
        try:
            image_cv = advanced_inpaint(
                image_cv,
                refined_mask,
                method=runtime_method,
                use_crf=False,
                inpainting_size=LAMA_INPAINT_SIZE,
                verbose=DEBUG_DIAG
            )
            logger.debug("[INPAINT] inpainting 完成")
        except Exception as e:
            import traceback
            logger.error(f"[INPAINT] inpainting 失败: {e}\n{traceback.format_exc()}")
    
    # 4.5 深色背景修复（inpainting 后，对被错误涂白的深色区域用原始背景色替换）
    if enable_bg_fix and inpaint and np.any(refined_mask):
        try:
            bg_fix_repair_method = _resolve_bg_fix_repair_method(runtime_method)
            logger.debug(
                f"[BG_FIX] 开始深色背景修复: bboxes={len(mask_bboxes)}, "
                f"mask_shape={refined_mask.shape}, repair_method={bg_fix_repair_method}"
            )
            _diag_print(f"[BG_FIX] 开始深色背景修复: bboxes={len(mask_bboxes)}, mask_shape={refined_mask.shape}")
            image_cv = _fix_dark_background_after_inpaint(
                original_for_inpaint,
                image_cv,
                refined_mask,
                mask_bboxes,
                raw_mask,
                repair_method=bg_fix_repair_method,
            )
        except Exception as e:
            import traceback
            logger.warning(f"[BG_FIX] 深色背景修复失败: {e}\n{traceback.format_exc()}")
            _diag_print(f"[BG_FIX] 深色背景修复失败: {e}")
    
    return all_text, image_cv, all_bboxes, refined_mask, merge_map


async def get_text_masked_pic_auto(
    image_pil, 
    image_cv, 
    bboxes=None,
    inpaint: bool = True, 
    ocr_engine: str = None,
    text_direction: str = "horizontal",
):
    """
    自动选择最佳处理流程
    
    - 如果使用 MIT 检测器模式（USE_MIT_DETECTOR=true），使用新的高级流程
    - 否则使用原有的 YOLO 检测 + 颜色掩码流程
    
    Args:
        image_pil: PIL 图像
        image_cv: OpenCV BGR 图像
        bboxes: 预计算的边界框（可选，仅用于旧流程）
        inpaint: 是否进行 inpainting
        ocr_engine: OCR 引擎
        
    Returns:
        all_text: OCR 识别的文本列表
        image_cv: 处理后的图像
        bboxes: 边界框列表（用于绘制文字）
        raw_mask: 原始掩码（用于文字颜色选择）
    """
    if USE_MIT_DETECTOR:
        # 使用 MIT 高级流程（支持混合检测）
        try:
            from app.core.custom_conf import custom_conf
            use_yolo_obb = getattr(custom_conf, "use_yolo_obb", True)
            yolo_obb_conf = getattr(custom_conf, "yolo_obb_conf", 0.4)
        except Exception:
            use_yolo_obb = True
            yolo_obb_conf = 0.4
        
        all_text, image_cv, bboxes, raw_mask, _ = await get_text_masked_pic_mit(
            image_pil,
            image_cv,
            inpaint=inpaint,
            ocr_engine=ocr_engine,
            text_direction=text_direction,
            use_yolo_obb=use_yolo_obb,
            yolo_obb_conf=yolo_obb_conf,
        )
        return all_text, image_cv, bboxes, raw_mask
    else:
        # 使用原有流程
        if bboxes is None:
            from app.services.ocr import detect_text_regions
            bboxes = detect_text_regions(image_cv)
        all_text, image_cv, raw_mask = await get_text_masked_pic(
            image_pil, 
            image_cv, 
            bboxes, 
            inpaint=inpaint, 
            ocr_engine=ocr_engine
        )
        return all_text, image_cv, bboxes, raw_mask


# ============ MIT 高质量渲染函数 ============

def draw_text_on_boxes_mit(
    image: np.ndarray, 
    boxes: list, 
    texts: list, 
    text_direction: str = "horizontal",
    target_lang: str = "CHS",
    original_image: np.ndarray = None,
    raw_mask: np.ndarray = None,
) -> np.ndarray:
    """
    使用 MIT 渲染模块在指定区域绘制文字（高质量）
    
    特性：
    - Qt 渲染引擎（精确字形渲染）
    - CJK 标点禁则
    - 自动换行优化
    - <H> 标签支持（竖排中嵌入横排）
    - 页面级统一字体大小（确保同一气泡内字体一致）
    
    Args:
        image: BGR 图像
        boxes: 边界框列表
        texts: 文本列表
        text_direction: 文字方向 "horizontal"（横排）或 "vertical"（竖排）
        target_lang: 目标语言（用于换行优化）
    
    Returns:
        绘制后的 BGR 图像
    """
    if not HAS_MIT_RENDER or not USE_MIT_RENDER:
        logger.info("MIT 渲染模块未启用，使用 PIL 渲染")
        return draw_text_on_boxes(image, boxes, texts, text_direction, raw_mask=raw_mask)
    
    height, width = image.shape[:2]
    result = image.copy()
    is_horizontal = text_direction == "horizontal"
    
    # 创建渲染配置
    render_config = TextRenderConfig(
        stroke_width=0.07,
        line_spacing=1.0,
        letter_spacing=1.0,
    )
    render_groups = _build_render_groups(boxes, text_direction)
    group_index_map = {}
    for group_id, group in enumerate(render_groups):
        for idx in group:
            group_index_map[idx] = group_id

    vertical_group_layouts = {}
    if not is_horizontal:
        for group_id, group in enumerate(render_groups):
            if len(group) <= 1:
                continue
            group_top = min(int(boxes[idx][1]) for idx in group)
            group_bottom = max(int(boxes[idx][3]) for idx in group)
            vertical_group_layouts[group_id] = {
                "top": group_top,
                "height": max(1, group_bottom - group_top),
            }
    
    # ============ 第一步：计算所有文本框的字体大小和换行 ============
    font_sizes = {}
    text_with_br_list = {}
    render_layouts = {}
    valid_indices = []
    bubble_mask_cache = {}
    
    for i, (box, text) in enumerate(zip(boxes, texts)):
        if not text:
            continue
        
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        box_width = x2 - x1
        box_height = y2 - y1
        
        if box_width <= 0 or box_height <= 0:
            continue

        render_x1, render_y1, render_x2, render_y2 = x1, y1, x2, y2
        group_id = group_index_map.get(i, -1)
        if not is_horizontal and group_id in vertical_group_layouts:
            group_layout = vertical_group_layouts[group_id]
            render_y1 = max(0, min(group_layout["top"], height - 1))
            render_y2 = max(render_y1 + 1, min(group_layout["top"] + group_layout["height"], height))

        render_box_width = render_x2 - render_x1
        render_box_height = render_y2 - render_y1
        if render_box_width <= 0 or render_box_height <= 0:
            continue

        bubble_key = (render_x1, render_y1, render_x2, render_y2)
        bubble_layout = bubble_mask_cache.get(bubble_key)
        if bubble_layout is None:
            bubble_pad = int(round(
                min(render_box_width, render_box_height) *
                _get_bubble_crop_margin_ratio(render_box_width, render_box_height)
            ))
            bubble_crop_x1 = max(0, render_x1 - bubble_pad)
            bubble_crop_y1 = max(0, render_y1 - bubble_pad)
            bubble_crop_x2 = min(width, render_x2 + bubble_pad)
            bubble_crop_y2 = min(height, render_y2 + bubble_pad)
            bubble_mask = None
            local_bubble_mask = _extract_bubble_mask_from_crop(
                image[bubble_crop_y1:bubble_crop_y2, bubble_crop_x1:bubble_crop_x2]
            )
            if local_bubble_mask is not None:
                inner_x1 = render_x1 - bubble_crop_x1
                inner_y1 = render_y1 - bubble_crop_y1
                inner_x2 = inner_x1 + render_box_width
                inner_y2 = inner_y1 + render_box_height
                bubble_mask = local_bubble_mask[inner_y1:inner_y2, inner_x1:inner_x2]
            hint_bbox = None
            if bubble_mask is not None and np.any(bubble_mask):
                bubble_pixels = cv2.findNonZero(bubble_mask)
                if bubble_pixels is not None:
                    bx, by, bw, bh = cv2.boundingRect(bubble_pixels)
                    if bw >= max(10, int(render_box_width * 0.45)) and bh >= max(10, int(render_box_height * 0.45)):
                        hint_bbox = (bx, by, bw, bh)
            bubble_layout = {
                "mask": bubble_mask,
                "hint_bbox": hint_bbox,
            }
            bubble_mask_cache[bubble_key] = bubble_layout

        layout_box_width = render_box_width
        layout_box_height = render_box_height
        hint_bbox = bubble_layout["hint_bbox"]
        if hint_bbox is not None:
            _, _, hint_w, hint_h = hint_bbox
            layout_box_width = max(1, min(render_box_width, hint_w - 4))
            layout_box_height = max(1, min(render_box_height, hint_h - 4))
        
        # 处理特殊符号
        text = compact_special_symbols(text, convert_ascii_ellipsis=not is_horizontal)
        
        # 计算最优字体大小和换行
        font_size, text_with_br = calculate_font_size(
            text=text,
            box_width=layout_box_width,
            box_height=layout_box_height,
            is_horizontal=is_horizontal,
            config=render_config,
            target_lang=target_lang,
        )
        
        font_sizes[i] = font_size
        text_with_br_list[i] = text_with_br
        render_layouts[i] = {
            "render_x1": render_x1,
            "render_y1": render_y1,
            "render_x2": render_x2,
            "render_y2": render_y2,
            "bubble_mask": bubble_layout["mask"],
        }
        valid_indices.append(i)

    for group in render_groups:
        valid_group = [idx for idx in group if idx in font_sizes]
        if len(valid_group) <= 1:
            continue

        calculated_sizes = [font_sizes[idx] for idx in valid_group]
        original_sizes = [_estimate_mask_font_size(boxes[idx]) for idx in valid_group]
        reference_size = int(round(np.median(original_sizes)))
        min_calculated = min(calculated_sizes)

        if reference_size <= min_calculated:
            shared_font_size = reference_size
        elif len(calculated_sizes) >= 3:
            sorted_sizes = sorted(calculated_sizes)
            shared_font_size = sorted_sizes[len(sorted_sizes) // 2]
        else:
            shared_font_size = min_calculated

        size_span = max(calculated_sizes) - shared_font_size
        if size_span > 0 or shared_font_size != min_calculated:
            logger.debug(
                f"同气泡统一字号: group={valid_group}, "
                f"calculated={calculated_sizes}, original_ref={reference_size} -> {shared_font_size}"
            )
        for idx in valid_group:
            font_sizes[idx] = shared_font_size
    
    # ============ 第二步：渲染每个文本框 ============
    # 对同一气泡内的相邻框统一字号，避免双竖排被分开识别后大小失衡。
    for idx, i in enumerate(valid_indices):
        box = boxes[i]
        text = texts[i]
        text_with_br = text_with_br_list[i]
        font_size = font_sizes[i]
        
        if not text:
            continue
        
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        box_width = x2 - x1
        box_height = y2 - y1
        
        if box_width <= 0 or box_height <= 0:
            continue

        group_id = group_index_map.get(i, -1)
        render_layout = render_layouts.get(i)
        if render_layout is None:
            continue
        render_x1 = render_layout["render_x1"]
        render_y1 = render_layout["render_y1"]
        render_x2 = render_layout["render_x2"]
        render_y2 = render_layout["render_y2"]
        render_box_width = render_x2 - render_x1
        render_box_height = render_y2 - render_y1
        if render_box_width <= 0 or render_box_height <= 0:
            continue
        
        cropped = result[render_y1:render_y2, render_x1:render_x2]
        color_source = original_image if original_image is not None else image
        bg_pixels = _sample_bbox_background(color_source, render_x1, render_y1, render_x2, render_y2, raw_mask)
        fill_color, stroke_color = _pick_text_style(
            cropped,
            _sample_surrounding_region(color_source, render_x1, render_y1, render_x2, render_y2),
            bg_pixels=bg_pixels,
        )
        
        # 处理特殊符号
        text = compact_special_symbols(text, convert_ascii_ellipsis=not is_horizontal)
        group_member_count = len([
            member for member in render_groups[group_id]
            if member in font_sizes
        ]) if group_id >= 0 else 1
        vertical_alignment = "top" if (not is_horizontal and group_member_count > 1) else "center"
        
        logger.debug(f"渲染文本 '{text[:10]}...' 字体大小={font_size}px, 框={render_box_width}x{render_box_height}")
        
        try:
            render_text = text_with_br if '[BR]' in text_with_br else text
            text_bitmap, fitted_font_size = _render_text_bitmap_with_fit(
                text=render_text,
                font_size=font_size,
                box_width=render_box_width,
                box_height=render_box_height,
                is_horizontal=is_horizontal,
                fill_color=fill_color,
                stroke_color=stroke_color,
                target_lang=target_lang,
                align_y=vertical_alignment,
                bubble_mask=render_layout["bubble_mask"],
            )
            if fitted_font_size < font_size:
                logger.debug(
                    f"Render bitmap shrink-to-fit: idx={i}, font={font_size}->{fitted_font_size}, "
                    f"box={render_box_width}x{render_box_height}"
                )

            if text_bitmap is not None:
                result = _overlay_text_bitmap(
                    result,
                    text_bitmap,
                    render_x1,
                    render_y1,
                    render_box_width,
                    render_box_height,
                    align_y=vertical_alignment if not is_horizontal else 'center',
                )
        except Exception as e:
            logger.warning(f"MIT 渲染失败，回退到 PIL: {e}")
            # 回退到 PIL 渲染单个区域
            result_pil = draw_text_on_boxes(result, [box], [text], text_direction)
            result = result_pil
    
    return result


def _overlay_text_bitmap(
    image: np.ndarray,
    text_bitmap: np.ndarray,
    x: int,
    y: int,
    box_width: int,
    box_height: int,
    align_x: str = "center",
    align_y: str = "center",
) -> np.ndarray:
    result = image
    
    th, tw = text_bitmap.shape[:2]
    
    paste_x, paste_y = _resolve_bitmap_paste_origin(
        x,
        y,
        box_width,
        box_height,
        tw,
        th,
        align_x=align_x,
        align_y=align_y,
    )
    
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
    if len(text_bitmap.shape) == 3 and text_bitmap.shape[2] == 4:
        alpha = text_bitmap[src_y:src_y+dst_h, src_x:src_x+dst_w, 3:4] / 255.0
        color = text_bitmap[src_y:src_y+dst_h, src_x:src_x+dst_w, :3]
    else:
        # 如果没有 alpha 通道，假设非黑色区域为文本
        if len(text_bitmap.shape) == 3:
            alpha = np.any(text_bitmap[src_y:src_y+dst_h, src_x:src_x+dst_w] > 0, axis=2, keepdims=True).astype(np.float32)
            color = text_bitmap[src_y:src_y+dst_h, src_x:src_x+dst_w, :3]
        else:
            alpha = (text_bitmap[src_y:src_y+dst_h, src_x:src_x+dst_w] > 0).astype(np.float32)[:, :, np.newaxis]
            color = np.stack([text_bitmap[src_y:src_y+dst_h, src_x:src_x+dst_w]] * 3, axis=2)
    
    # 颜色转换 (RGB -> BGR)
    color_bgr = color[:, :, ::-1]
    
    # 混合
    roi = result[paste_y:paste_y+dst_h, paste_x:paste_x+dst_w]
    result[paste_y:paste_y+dst_h, paste_x:paste_x+dst_w] = (alpha * color_bgr + (1 - alpha) * roi).astype(np.uint8)
    
    return result
