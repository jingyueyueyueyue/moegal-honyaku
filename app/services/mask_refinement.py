"""
掩码细化模块

包含 CRF (条件随机场) 细化算法，用于优化掩码边界
基于 manga-image-translator 和 manga-translator-ui (MTU) 的实现

移植自 MTU 的增强功能：
1. 连通组件分析与文本框匹配
2. 对每个文本区域单独 CRF 细化
3. 基于字体大小的自适应膨胀
"""

import logging
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np

from app.core.logger import logger

# 尝试导入 pydensecrf
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    HAS_CRF = True
except ImportError:
    HAS_CRF = False
    logger.debug("pydensecrf 未安装，CRF 细化功能不可用。安装: pip install pydensecrf")

# 尝试导入 Shapely（用于精确的多边形相交计算）
try:
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    logger.debug("Shapely 未安装，将使用简化的边界框计算")

# 兼容不同版本的 pydensecrf
DIAG_KERNEL = getattr(dcrf, 'DIAG_KERNEL', 0) if HAS_CRF else 0
NO_NORMALIZATION = getattr(dcrf, 'NO_NORMALIZATION', 0) if HAS_CRF else 0


def _extend_rect(x: int, y: int, w: int, h: int, max_x: int, max_y: int, extend_size: int) -> Tuple[int, int, int, int]:
    """扩展矩形边界"""
    x1 = max(x - extend_size, 0)
    y1 = max(y - extend_size, 0)
    w1 = min(w + extend_size * 2, max_x - x1 - 1)
    h1 = min(h + extend_size * 2, max_y - y1 - 1)
    return x1, y1, w1, h1


def _adaptive_dilate_size(text_size: int, bbox_width: int, bbox_height: int, dilation_offset: int) -> int:
    """Keep dilation close to glyph width so tall columns do not flood the bubble."""
    shorter_side = max(1, min(int(bbox_width), int(bbox_height)))
    longer_side = max(1, max(int(bbox_width), int(bbox_height)))
    adjusted_text = max(1, int(round(text_size + dilation_offset)))

    scale = 0.18 if longer_side >= shorter_side * 2.0 else 0.24
    dilate_size = max(1, int(round(adjusted_text * scale)))
    dilate_cap = max(1, int(round(shorter_side * 0.45)))
    dilate_size = min(dilate_size, dilate_cap)

    if shorter_side >= 6:
        dilate_size = max(dilate_size, 3)
    if dilate_size % 2 == 0:
        dilate_size += 1
    return dilate_size


def refine_mask_with_crf(rgbimg: np.ndarray, rawmask: np.ndarray, iterations: int = 5) -> np.ndarray:
    """
    使用 CRF (条件随机场) 细化掩码边界
    
    CRF 可以利用图像的颜色信息来优化掩码边界，
    特别适合处理复杂的背景和前景分离。
    
    Args:
        rgbimg: RGB 图像 (H, W, 3)
        rawmask: 原始掩码 (H, W)，值为 0 或 255
        iterations: CRF 迭代次数，默认 5
        
    Returns:
        细化后的掩码 (H, W)，值为 0 或 255
    """
    if not HAS_CRF:
        logger.debug("pydensecrf 未安装，跳过 CRF 细化")
        return rawmask
    
    # 优化：空掩码或全黑掩码直接返回
    if rawmask is None or rawmask.size == 0:
        return rawmask
    if np.max(rawmask) == 0:
        return rawmask
    
    # 优化：小区域跳过 CRF（开销大于收益）
    if rawmask.size < 100:
        return rawmask
    
    # 确保输入数组是 C 连续的（修复 "ndarray is not C-contiguous" 错误）
    rgbimg = np.ascontiguousarray(rgbimg)
    rawmask = np.ascontiguousarray(rawmask)
    
    if len(rawmask.shape) == 2:
        rawmask = rawmask[:, :, None]
    
    # 复用数组，减少内存分配
    mask_softmax = np.empty((rawmask.shape[0], rawmask.shape[1], 2), dtype=np.float32)
    float_mask = rawmask[:, :, 0].astype(np.float32) / 255.0
    mask_softmax[:, :, 0] = 1.0 - float_mask
    mask_softmax[:, :, 1] = float_mask
    
    n_classes = 2
    feat_first = mask_softmax.transpose((2, 0, 1)).reshape((n_classes, -1))
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(rgbimg.shape[1], rgbimg.shape[0], n_classes)
    d.setUnaryEnergy(unary)
    
    # 高斯成对势（空间平滑）
    d.addPairwiseGaussian(sxy=1, compat=3, kernel=DIAG_KERNEL, normalization=NO_NORMALIZATION)
    
    # 双边成对势（颜色相似性）
    d.addPairwiseBilateral(
        sxy=23, srgb=7, rgbim=rgbimg,
        compat=20,
        kernel=DIAG_KERNEL,
        normalization=NO_NORMALIZATION
    )
    
    # 推理
    Q = d.inference(iterations)
    res = np.argmax(Q, axis=0).reshape((rgbimg.shape[0], rgbimg.shape[1]))
    
    return (res * 255).astype(np.uint8)


def complete_mask(
    img: np.ndarray,
    mask: np.ndarray,
    bboxes: List[np.ndarray],
    keep_threshold: float = 0.01,
    dilation_offset: int = 0,
    kernel_size: int = 3,
    use_crf: bool = True,
    font_sizes: Optional[List[int]] = None,
) -> np.ndarray:
    """
    完成掩码精修（增强版）
    
    移植自 manga-translator-ui (MTU) 项目，包含以下改进：
    1. 连通组件分析与文本框匹配
    2. 使用 Shapely 进行精确的多边形相交计算
    3. 基于字体大小的自适应膨胀
    4. 对每个文本区域单独 CRF 细化
    
    Args:
        img: BGR 图像
        mask: 检测器输出的原始掩码 (H, W)
        bboxes: 边界框列表 [[x1,y1,x2,y2], ...]
        keep_threshold: 保留阈值
        dilation_offset: 膨胀偏移
        kernel_size: 内核大小
        use_crf: 是否使用 CRF 细化
        font_sizes: 字体大小列表（用于自适应膨胀）
        
    Returns:
        精修后的掩码 (H, W)
    """
    if mask is None or mask.size == 0:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    
    # 确保掩码是二维的
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    height, width = img.shape[:2]
    M = len(bboxes)
    
    if M == 0:
        return mask
    
    # 构建 Shapely 多边形（如果可用）
    polys = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        if HAS_SHAPELY:
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            polys.append(Polygon(pts))
        else:
            polys.append(None)
    
    for bbox in bboxes:
        bx1, by1, bx2, by2 = map(int, bbox)
        bx1 = max(0, min(bx1, width - 1))
        by1 = max(0, min(by1, height - 1))
        bx2 = max(bx1 + 1, min(bx2, width))
        by2 = max(by1 + 1, min(by2, height))
        cv2.rectangle(mask, (bx1, by1), (bx2, by2), (0), 1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 文本行连通组件掩码
    textline_ccs = [np.zeros_like(mask) for _ in range(M)]
    iinfo = np.iinfo(labels.dtype)
    textline_rects = np.full(shape=(M, 4), fill_value=[iinfo.max, iinfo.max, iinfo.min, iinfo.min], dtype=labels.dtype)
    
    # 重叠矩阵和距离矩阵
    ratio_mat = np.zeros(shape=(num_labels, M), dtype=np.float32)
    dist_mat = np.zeros(shape=(num_labels, M), dtype=np.float32)
    
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] <= 9:
            continue
        
        x1 = stats[label, cv2.CC_STAT_LEFT]
        y1 = stats[label, cv2.CC_STAT_TOP]
        w1 = stats[label, cv2.CC_STAT_WIDTH]
        h1 = stats[label, cv2.CC_STAT_HEIGHT]
        area1 = stats[label, cv2.CC_STAT_AREA]
        
        # 构建连通组件的多边形
        if HAS_SHAPELY:
            cc_pts = np.array([[x1, y1], [x1 + w1, y1], [x1 + w1, y1 + h1], [x1, y1 + h1]])
            cc_poly = Polygon(cc_pts)
        else:
            cc_poly = None
        
        for tl_idx in range(M):
            if polys[tl_idx] is not None and cc_poly is not None:
                # 使用 Shapely 计算精确的重叠面积
                area2 = polys[tl_idx].area
                try:
                    overlapping_area = polys[tl_idx].intersection(cc_poly).area
                except Exception:
                    try:
                        fixed_poly = polys[tl_idx].buffer(0)
                        fixed_cc_poly = cc_poly.buffer(0)
                        overlapping_area = fixed_poly.intersection(fixed_cc_poly).area
                    except Exception:
                        overlapping_area = 0
                
                ratio_mat[label, tl_idx] = overlapping_area / min(area1, area2)
                
                try:
                    dist_mat[label, tl_idx] = polys[tl_idx].distance(cc_poly.centroid)
                except Exception:
                    dist_mat[label, tl_idx] = float('inf')
            else:
                # 使用简化的边界框计算
                bbox = bboxes[tl_idx]
                bx1, by1, bx2, by2 = map(int, bbox)
                area2 = (bx2 - bx1) * (by2 - by1)
                
                # 计算重叠区域
                overlap_x = max(0, min(x1 + w1, bx2) - max(x1, bx1))
                overlap_y = max(0, min(y1 + h1, by2) - max(y1, by1))
                overlapping_area = overlap_x * overlap_y
                
                ratio_mat[label, tl_idx] = overlapping_area / min(area1, area2)
                
                # 计算中心点距离
                cc_cx, cc_cy = x1 + w1 / 2, y1 + h1 / 2
                bbox_cx, bbox_cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                dist_mat[label, tl_idx] = math.sqrt((cc_cx - bbox_cx) ** 2 + (cc_cy - bbox_cy) ** 2)
        
        # 找到最佳匹配的文本框
        avg = np.argmax(ratio_mat[label])
        max_overlap = ratio_mat[label, avg]
        
        if max_overlap < 0.1:
            continue
        
        bbox = bboxes[avg]
        bx1, by1, bx2, by2 = map(int, bbox)
        area2 = (bx2 - bx1) * (by2 - by1)
        
        if area1 >= area2:
            continue
        
        if ratio_mat[label, avg] <= keep_threshold:
            avg = np.argmin(dist_mat[label])
            bbox = bboxes[avg]
            bx1, by1, bx2, by2 = map(int, bbox)
            area2 = (bx2 - bx1) * (by2 - by1)
            
            # 估计字体大小
            if font_sizes and avg < len(font_sizes):
                text_size = font_sizes[avg]
            else:
                text_size = min(w1, h1)
            
            unit = max(min([text_size, w1, h1]), 10)
            if dist_mat[label, avg] >= 0.5 * unit:
                continue
        
        # 更新文本行连通组件
        textline_ccs[avg][y1:y1+h1, x1:x1+w1][labels[y1:y1+h1, x1:x1+w1] == label] = 255
        textline_rects[avg, 0] = min(textline_rects[avg, 0], x1)
        textline_rects[avg, 1] = min(textline_rects[avg, 1], y1)
        textline_rects[avg, 2] = max(textline_rects[avg, 2], x1 + w1)
        textline_rects[avg, 3] = max(textline_rects[avg, 3], y1 + h1)
    
    # 计算矩形宽高
    textline_rects[:, 2] -= textline_rects[:, 0]
    textline_rects[:, 3] -= textline_rects[:, 1]
    
    # 双边滤波（减少噪声）
    img_filtered = cv2.bilateralFilter(img, 17, 80, 80)
    rgb_img = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)
    
    # 生成最终掩码
    final_mask = np.zeros_like(mask)
    
    for i, cc in enumerate(textline_ccs):
        if not np.any(cc > 0):
            continue
        
        x1, y1, w1, h1 = textline_rects[i]
        
        # 计算字体大小（用于自适应膨胀）
        if font_sizes and i < len(font_sizes):
            text_size = font_sizes[i]
        else:
            text_size = min(w1, h1)
        
        # 扩展矩形边界
        extend_size = min(
            int(text_size * 0.08),
            max(2, int(round(min(w1, h1) * 0.18))),
        )
        x1_ext, y1_ext, w1_ext, h1_ext = _extend_rect(
            x1, y1, w1, h1, width, height, extend_size
        )
        
        dilate_size = _adaptive_dilate_size(text_size, w1, h1, dilation_offset)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        
        # 提取区域
        cc_region = np.ascontiguousarray(cc[y1_ext: y1_ext + h1_ext, x1_ext: x1_ext + w1_ext])
        if cc_region.size == 0:
            continue
        
        # CRF 细化（如果启用）
        if use_crf and HAS_CRF:
            img_region = np.ascontiguousarray(rgb_img[y1_ext: y1_ext + h1_ext, x1_ext: x1_ext + w1_ext])
            try:
                cc_region = refine_mask_with_crf(img_region, cc_region)
            except Exception as e:
                logger.debug(f"CRF 细化失败: {e}")
        
        cc[y1_ext: y1_ext + h1_ext, x1_ext: x1_ext + w1_ext] = cc_region
        
        # 膨胀处理
        x2_ext, y2_ext, w2_ext, h2_ext = _extend_rect(
            x1_ext, y1_ext, w1_ext, h1_ext, width, height, -(-dilate_size // 2)
        )
        cc[y2_ext:y2_ext+h2_ext, x2_ext:x2_ext+w2_ext] = cv2.dilate(
            cc[y2_ext:y2_ext+h2_ext, x2_ext:x2_ext+w2_ext], kern
        )
        
        # 合并到最终掩码
        final_mask[y2_ext:y2_ext+h2_ext, x2_ext:x2_ext+w2_ext] = cv2.bitwise_or(
            final_mask[y2_ext:y2_ext+h2_ext, x2_ext:x2_ext+w2_ext],
            cc[y2_ext:y2_ext+h2_ext, x2_ext:x2_ext+w2_ext]
        )
    
    # 最终膨胀
    if kernel_size > 0:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        final_mask = cv2.dilate(final_mask, kern, iterations=1)
    
    return final_mask


def refine_textline_mask(
    img: np.ndarray,
    bbox: np.ndarray,
    initial_mask: np.ndarray = None,
    use_crf: bool = True,
) -> np.ndarray:
    """
    细化单个文本行的掩码
    
    Args:
        img: BGR 图像
        bbox: 边界框 [x1, y1, x2, y2]
        initial_mask: 初始掩码（可选）
        use_crf: 是否使用 CRF
        
    Returns:
        细化后的掩码
    """
    x1, y1, x2, y2 = map(int, bbox)
    height, width = img.shape[:2]
    
    # 确保坐标有效
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    
    # 提取区域
    region_img = img[y1:y2, x1:x2]
    
    if initial_mask is not None:
        region_mask = initial_mask[y1:y2, x1:x2]
    else:
        # 如果没有初始掩码，使用整个区域
        region_mask = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)
    
    if use_crf and HAS_CRF:
        rgb_region = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
        region_mask = refine_mask_with_crf(rgb_region, region_mask)
    
    # 创建完整掩码
    full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = region_mask
    
    return full_mask


def merge_masks(masks: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """
    合并多个掩码
    
    Args:
        masks: 掩码列表
        shape: 输出形状 (H, W)
        
    Returns:
        合并后的掩码
    """
    result = np.zeros(shape, dtype=np.uint8)
    
    for mask in masks:
        if mask is not None and mask.size > 0:
            # 确保尺寸匹配
            if mask.shape[:2] != shape:
                mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            result = cv2.bitwise_or(result, mask)
    
    return result
