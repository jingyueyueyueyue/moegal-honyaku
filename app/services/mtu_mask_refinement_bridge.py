import os
from functools import lru_cache

import cv2
import numpy as np

from app.core.logger import logger
from app.services.mtu_bridge_compat import ensure_py3langid_shim

@lru_cache(maxsize=1)
def _load_mtu_runtime():
    ensure_py3langid_shim()

    from manga_translator.mask_refinement import dispatch as dispatch_mask_refinement
    from manga_translator.utils.textblock import TextBlock

    return dispatch_mask_refinement, TextBlock


@lru_cache(maxsize=1)
def _has_embedded_mtu() -> bool:
    try:
        _load_mtu_runtime()
        return True
    except Exception:
        return False


def supports_mtu_mask_refinement() -> bool:
    return _has_embedded_mtu()


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("raw_mask is required")
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    return np.where(mask_np > 0, 255, 0).astype(np.uint8)


def _bgr_to_mtu_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _build_text_regions(bboxes: list, font_sizes: list | None, text_direction: str):
    _, text_block_cls = _load_mtu_runtime()
    direction = "v" if text_direction == "vertical" else "h"
    regions = []

    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(float, bbox)
        polygon = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        font_size = -1
        if font_sizes and idx < len(font_sizes):
            font_size = float(font_sizes[idx])
        regions.append(
            text_block_cls(
                lines=[polygon],
                texts=[""],
                font_size=font_size,
                direction=direction,
            )
        )
    return regions


async def mtu_refine_mask_async(
    image: np.ndarray,
    raw_mask: np.ndarray,
    bboxes: list,
    *,
    font_sizes: list | None = None,
    text_direction: str = "horizontal",
    dilation_offset: int | None = None,
    kernel_size: int = 3,
    verbose: bool = False,
) -> np.ndarray:
    if raw_mask is None:
        raise ValueError("raw_mask is required")

    mask_binary = _normalize_mask(raw_mask)
    if not bboxes or not np.any(mask_binary):
        return np.zeros(mask_binary.shape[:2], dtype=np.uint8)

    dispatch_mask_refinement, _ = _load_mtu_runtime()
    mtu_dilation_offset = int(
        os.getenv("MTU_MASK_DILATION_OFFSET", str(20 if dilation_offset is None else dilation_offset))
    )
    mtu_kernel_size = max(1, int(kernel_size))
    rgb_image = _bgr_to_mtu_rgb(image)
    text_regions = _build_text_regions(bboxes, font_sizes, text_direction)

    log_message = (
        "Using embedded MTU mask refinement: "
        f"regions={len(text_regions)}, dilation_offset={mtu_dilation_offset}, "
        f"kernel_size={mtu_kernel_size}, mask_pixels={int(np.count_nonzero(mask_binary))}"
    )
    if verbose:
        logger.info(log_message)
    else:
        logger.debug(log_message)

    refined_mask = await dispatch_mask_refinement(
        text_regions,
        rgb_image,
        mask_binary,
        method="fit_text",
        dilation_offset=mtu_dilation_offset,
        verbose=verbose,
        kernel_size=mtu_kernel_size,
        use_model_bubble_repair_intersection=False,
        limit_mask_dilation_to_bubble_mask=False,
    )

    return _normalize_mask(refined_mask)
