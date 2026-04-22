import asyncio
import os
import threading
from functools import lru_cache

import cv2
import numpy as np

from app.core.logger import logger
from app.services.mtu_bridge_compat import ensure_py3langid_shim

MTU_METHODS = {"aot", "lama", "lama_mpe", "lama_large"}


def _env_truthy(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _choose_mtu_device() -> str:
    forced_device = os.getenv("MTU_INPAINT_DEVICE", "").strip().lower()
    if forced_device in {"cpu", "cuda", "mps"}:
        return forced_device

    gpu_mode = _env_truthy("MOEGAL_USE_GPU")
    try:
        import torch

        if gpu_mode is not False and torch.cuda.is_available():
            return "cuda"
        if gpu_mode is not False and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


@lru_cache(maxsize=1)
def _load_mtu_runtime():
    ensure_py3langid_shim()

    from manga_translator.config import Inpainter, InpainterConfig
    from manga_translator.inpainting import dispatch as dispatch_inpainting

    return dispatch_inpainting, Inpainter, InpainterConfig


@lru_cache(maxsize=1)
def _has_embedded_mtu() -> bool:
    try:
        _load_mtu_runtime()
        return True
    except Exception:
        return False


def _run_coroutine_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_holder = {}
    error_holder = {}

    def _runner():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result_holder["value"] = loop.run_until_complete(coro)
        except Exception as exc:
            error_holder["error"] = exc
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            asyncio.set_event_loop(None)
            loop.close()

    thread = threading.Thread(target=_runner, name="mtu-inpaint-bridge", daemon=True)
    thread.start()
    thread.join()

    if "error" in error_holder:
        raise error_holder["error"]
    return result_holder.get("value")


def _map_method(method: str, inpainter_enum):
    normalized = (method or "lama_large").lower()
    if normalized == "aot":
        return inpainter_enum.default
    if normalized in {"lama", "lama_mpe"}:
        return inpainter_enum.lama_mpe
    if normalized == "lama_large":
        return inpainter_enum.lama_large
    raise ValueError(f"Unsupported MTU inpaint method: {method}")


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("mask is required")
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


def _mtu_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def supports_mtu_inpaint(method: str) -> bool:
    return (method or "").lower() in MTU_METHODS and _has_embedded_mtu()


def mtu_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "lama_large",
    inpainting_size: int = 2048,
    verbose: bool = False,
) -> np.ndarray:
    dispatch_inpainting, inpainter_enum, inpainter_config_cls = _load_mtu_runtime()
    mask_binary = _normalize_mask(mask)
    rgb_image = _bgr_to_mtu_rgb(image)
    inpainter_key = _map_method(method, inpainter_enum)
    config = inpainter_config_cls(inpainter=inpainter_key, inpainting_size=inpainting_size)
    device = _choose_mtu_device()

    log_message = (
        f"Using embedded MTU inpaint: method={method}, mapped={inpainter_key}, "
        f"device={device}, mask_pixels={int(np.count_nonzero(mask_binary))}"
    )
    if verbose:
        logger.info(log_message)
    else:
        logger.debug(log_message)

    result = _run_coroutine_sync(
        dispatch_inpainting(
            inpainter_key,
            rgb_image,
            mask_binary,
            config,
            inpainting_size,
            device,
            verbose,
        )
    )
    return _mtu_rgb_to_bgr(np.asarray(result, dtype=np.uint8))
