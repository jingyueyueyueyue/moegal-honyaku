"""
Advanced Inpainting Module
Based on manga-image-translator's Lama-MPE and AOT implementation

Supports:
1. CRF mask refinement - Uses pydensecrf to optimize mask boundaries
2. AOT Inpainting - MIT default model, best results (recommended)
3. Lama-MPE deep learning inpainting - Specialized manga image repair model
4. Lama-Large - Larger model

Model Download URLs:
- AOT (recommended): https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt
- Lama-MPE: https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt
- Lama-Large: https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt
"""

import gc
import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from threading import Lock

from app.core.logger import logger
from app.core.paths import MODELS_DIR
from app.services.mtu_inpaint_bridge import mtu_inpaint, supports_mtu_inpaint

# ============ Model Download URLs ============
LAMA_MODEL_URLS = {
    "aot": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt",
    "lama_mpe": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt",
    "lama_large": "https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt",
}

_MODEL_LOCK = Lock()
_LAMA_MODELS = {}
DEBUG_DIAG = os.getenv("MOEGAL_DEBUG_DIAG", "false").lower() in ("true", "1", "yes")

INPAINT_SPLIT_RATIO = 3.0
SIMPLE_REGION_MIN_AREA = max(16, int(os.getenv("INPAINT_SIMPLE_REGION_MIN_AREA", "80")))
SIMPLE_REGION_NS_RADIUS = max(1, int(os.getenv("INPAINT_SIMPLE_REGION_NS_RADIUS", "4")))
SIMPLE_REGION_STD_THRESHOLD = float(os.getenv("INPAINT_SIMPLE_REGION_STD_THRESHOLD", "18.0"))
SIMPLE_REGION_GRAD_THRESHOLD = float(os.getenv("INPAINT_SIMPLE_REGION_GRAD_THRESHOLD", "12.0"))
SIMPLE_REGION_EDGE_THRESHOLD = float(os.getenv("INPAINT_SIMPLE_REGION_EDGE_THRESHOLD", "0.08"))
SIMPLE_REGION_NOISE_EXCESS = float(os.getenv("INPAINT_SIMPLE_REGION_NOISE_EXCESS", "6.0"))
SIMPLE_REGION_FLAT_STD_THRESHOLD = float(os.getenv("INPAINT_SIMPLE_REGION_FLAT_STD_THRESHOLD", "9.0"))
SIMPLE_REGION_FLAT_GRAD_THRESHOLD = float(os.getenv("INPAINT_SIMPLE_REGION_FLAT_GRAD_THRESHOLD", "6.0"))
SIMPLE_REGION_FLAT_EDGE_THRESHOLD = float(os.getenv("INPAINT_SIMPLE_REGION_FLAT_EDGE_THRESHOLD", "0.03"))
SIMPLE_REGION_LUMA_MISMATCH = float(os.getenv("INPAINT_SIMPLE_REGION_LUMA_MISMATCH", "25.0"))


def _diag_print(message: str) -> None:
    if DEBUG_DIAG:
        print(message, flush=True)


def _download_file(url: str, path: str) -> None:
    """Download file"""
    import requests
    from tqdm import tqdm
    
    logger.info(f"Downloading {url} -> {path}")
    
    response = requests.get(url, stream=True, allow_redirects=True)
    total = int(response.headers.get("content-length", 0))
    
    with open(path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(path)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def _ensure_lama_model(model_type: str = "aot") -> str:
    """Ensure Lama model exists"""
    model_dir = MODELS_DIR / "inpainting"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_filename = f"{model_type}.ckpt"
    model_path = model_dir / model_filename
    
    if not model_path.exists():
        url = LAMA_MODEL_URLS[model_type]
        _download_file(url, str(model_path))
    
    return str(model_path)


# ============ CRF Mask Refinement ============

def refine_mask_with_crf(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Refine mask boundaries using CRF (Conditional Random Field)
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        logger.warning("pydensecrf not installed, skipping CRF refinement")
        return mask
    
    # Optimization: Early exit for empty or trivial masks
    if mask is None or mask.size == 0:
        return mask
    if np.max(mask) == 0:
        return mask
    
    # Optimization: Skip expensive CRF for very small regions
    if mask.size < 100:
        return mask
    
    # Ensure arrays are C-contiguous
    img = np.ascontiguousarray(img)
    mask = np.ascontiguousarray(mask)
    
    if len(mask.shape) == 2:
        mask = mask[:, :, None]
    
    # Reuse array to reduce memory allocation
    mask_softmax = np.empty((mask.shape[0], mask.shape[1], 2), dtype=np.float32)
    float_mask = mask[:, :, 0].astype(np.float32) / 255.0
    mask_softmax[:, :, 0] = 1.0 - float_mask
    mask_softmax[:, :, 1] = float_mask
    
    n_classes = 2
    feat_first = mask_softmax.transpose((2, 0, 1)).reshape((n_classes, -1))
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_classes)
    d.setUnaryEnergy(unary)
    
    # Gaussian pairwise potential (spatial smoothness)
    DIAG_KERNEL = getattr(dcrf, 'DIAG_KERNEL', 0)
    NO_NORMALIZATION = getattr(dcrf, 'NO_NORMALIZATION', 0)
    
    d.addPairwiseGaussian(sxy=1, compat=3, kernel=DIAG_KERNEL, normalization=NO_NORMALIZATION)
    
    # Bilateral pairwise potential (color similarity)
    d.addPairwiseBilateral(
        sxy=23, srgb=7, rgbim=img,
        compat=20,
        kernel=DIAG_KERNEL,
        normalization=NO_NORMALIZATION
    )
    
    # Inference
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
    
    return (res * 255).astype(np.uint8)


# ============ Memory Cleanup ============

def _cleanup_memory():
    """Clean up GPU memory after inpainting"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


# ============ AOT Model Architecture (from manga-image-translator) ============

def relu_nf(x):
    return F.relu(x) * 1.7139588594436646


class LambdaLayer(nn.Module):
    def __init__(self, f):
        super(LambdaLayer, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class ScaledWSConv2d(nn.Conv2d):
    """2D Conv layer with Scaled Weight Standardization."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, gain=True, eps=1e-4):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
        scale = torch.rsqrt(torch.max(var * fan_in, torch.tensor(self.eps).to(var.device)))
        if self.gain is not None:
            scale = scale * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class ScaledWSTransposeConv2d(nn.ConvTranspose2d):
    """2D Transpose Conv layer with Scaled Weight Standardization."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, gain=True, eps=1e-4):
        nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, 'zeros')
        if gain:
            self.gain = nn.Parameter(torch.ones(self.in_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
        scale = torch.rsqrt(torch.max(var * fan_in, torch.tensor(self.eps).to(var.device)))
        if self.gain is not None:
            scale = scale * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose2d(x, self.get_weight(), self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)


class GatedWSConvPadded(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride=1, dilation=1):
        super(GatedWSConvPadded, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.padding = nn.ReflectionPad2d(((ks - 1) * dilation) // 2)
        self.conv = ScaledWSConv2d(in_ch, out_ch, kernel_size=ks, stride=stride, dilation=dilation)
        self.conv_gate = ScaledWSConv2d(in_ch, out_ch, kernel_size=ks, stride=stride, dilation=dilation)

    def forward(self, x):
        x = self.padding(x)
        signal = self.conv(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return signal * gate * 1.8


class GatedWSTransposeConvPadded(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride=1):
        super(GatedWSTransposeConvPadded, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = ScaledWSTransposeConv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=(ks - 1) // 2)
        self.conv_gate = ScaledWSTransposeConv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=(ks - 1) // 2)

    def forward(self, x):
        signal = self.conv(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return signal * gate * 1.8


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


class AOTBlock(nn.Module):
    def __init__(self, dim, rates=[2, 4, 8, 16]):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


class AOTGenerator(nn.Module):
    """AOT Inpainting Generator"""
    def __init__(self, in_ch=4, out_ch=3, ch=32, alpha=0.0):
        super(AOTGenerator, self).__init__()
        self.head = nn.Sequential(
            GatedWSConvPadded(in_ch, ch, 3, stride=1),
            LambdaLayer(relu_nf),
            GatedWSConvPadded(ch, ch * 2, 4, stride=2),
            LambdaLayer(relu_nf),
            GatedWSConvPadded(ch * 2, ch * 4, 4, stride=2),
        )
        self.body_conv = nn.Sequential(*[AOTBlock(ch * 4) for _ in range(10)])
        self.tail = nn.Sequential(
            GatedWSConvPadded(ch * 4, ch * 4, 3, 1),
            LambdaLayer(relu_nf),
            GatedWSConvPadded(ch * 4, ch * 4, 3, 1),
            LambdaLayer(relu_nf),
            GatedWSTransposeConvPadded(ch * 4, ch * 2, 4, 2),
            LambdaLayer(relu_nf),
            GatedWSTransposeConvPadded(ch * 2, ch, 4, 2),
            LambdaLayer(relu_nf),
            GatedWSConvPadded(ch, out_ch, 3, stride=1),
        )

    def forward(self, img, mask):
        x = torch.cat([mask, img], dim=1)
        x = self.head(x)
        conv = self.body_conv(x)
        x = self.tail(conv)
        if self.training:
            return x
        else:
            return torch.clip(x, -1, 1)


def get_aot_model(device: str = "cpu"):
    """Get AOT model (lazy loading)"""
    global _LAMA_MODELS
    
    with _MODEL_LOCK:
        cache_key = f"aot:{device}"
        if cache_key in _LAMA_MODELS:
            return _LAMA_MODELS[cache_key]
        
        model_path = _ensure_lama_model("aot")
        model = AOTGenerator()
        sd = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd['model'] if 'model' in sd else sd)
        model.eval()
        
        if device.startswith('cuda') or device == 'mps':
            model = model.to(device)
        
        _LAMA_MODELS[cache_key] = model
        logger.info(f"AOT Inpainting model loaded, device: {device}")
        return model


def aot_inpaint(image: np.ndarray, mask: np.ndarray, inpainting_size: int = 2048, merge_result: bool = True) -> np.ndarray:
    """
    Inpainting using AOT model (MIT default model, best results)
    
    Note: AOT model does not support bf16 precision, uses float32 by default.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_aot_model(device)
    _diag_print(
        f"[INPAINT_DIAG] aot_inpaint: device={device}, image_shape={image.shape}, merge_result={merge_result}"
    )
    
    img_original = image.copy()
    mask_original = mask.copy()
    original_height, original_width = image.shape[:2]
    
    # Normalize mask
    mask_original[mask_original < 127] = 0
    mask_original[mask_original >= 127] = 1
    
    height, width = image.shape[:2]
    needs_resize_back = False
    
    # Keep aspect ratio scaling
    if max(image.shape[:2]) > inpainting_size:
        needs_resize_back = True
        scale = inpainting_size / max(image.shape[:2])
        new_w = int(width * scale)
        new_h = int(height * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        height, width = new_h, new_w
    
    # Pad to multiple of 64
    h, w = image.shape[:2]
    pad_size = 64
    padded_h = h if h % pad_size == 0 else (pad_size - (h % pad_size)) + h
    padded_w = w if w % pad_size == 0 else (pad_size - (w % pad_size)) + w
    
    if padded_h != h or padded_w != w:
        img_pad = np.pad(image, ((0, padded_h - h), (0, padded_w - w), (0, 0)), mode='symmetric')
        mask_pad = np.pad(mask, ((0, padded_h - h), (0, padded_w - w)), mode='constant', constant_values=0)
    else:
        img_pad = image
        mask_pad = mask
    
    logger.debug(f"AOT inpainting resolution: {padded_w}x{padded_h}")
    
    # Convert to tensor (range -1 to 1)
    img_torch = torch.from_numpy(img_pad).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0
    mask_torch = torch.from_numpy(mask_pad).unsqueeze_(0).unsqueeze_(0).float() / 255.0
    mask_torch[mask_torch < 0.5] = 0
    mask_torch[mask_torch >= 0.5] = 1
    
    if device.startswith('cuda'):
        img_torch = img_torch.to(device)
        mask_torch = mask_torch.to(device)
    
    # Inference
    with torch.no_grad():
        img_torch = img_torch * (1 - mask_torch)
        img_inpainted_torch = model(img_torch, mask_torch)
    
    # Convert back to numpy (range -1 to 1 -> 0 to 255)
    img_inpainted = ((img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
    img_inpainted = np.clip(img_inpainted, 0, 255)
    
    # Remove padding
    img_inpainted = img_inpainted[:h, :w, :]
    
    # Scale back to original size
    if needs_resize_back:
        img_inpainted = cv2.resize(img_inpainted, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    
    # Ensure mask size matches
    if mask_original.shape[:2] != img_inpainted.shape[:2]:
        mask_original = cv2.resize(mask_original.astype(np.uint8), (img_inpainted.shape[1], img_inpainted.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Merge result
    mask_original = mask_original[:, :, None] if len(mask_original.shape) == 2 else mask_original
    if merge_result:
        mask_float = mask_original.astype(np.float32)
        ans = img_inpainted.astype(np.float32) * mask_float + img_original.astype(np.float32) * (1.0 - mask_float)
        ans = np.clip(ans, 0, 255).astype(np.uint8)
        return ans
    return img_inpainted


# ============ Utility Functions ============

def _normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return None
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    return np.where(mask_np > 0, 255, 0).astype(np.uint8)


def _inpaint_handle_alpha_channel(original_alpha: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep RGBA alpha stable around inpainted areas based on surrounding alpha."""
    alpha_2d = original_alpha[:, :, 0] if original_alpha.ndim == 3 else original_alpha
    result_alpha = alpha_2d.copy()
    mask_bin = (_normalize_binary_mask(mask) > 0).astype(np.uint8)

    if not np.any(mask_bin > 0):
        return result_alpha

    mask_dilated = cv2.dilate(mask_bin, np.ones((15, 15), np.uint8), iterations=1)
    surrounding_mask = mask_dilated - mask_bin

    if np.any(surrounding_mask > 0):
        surrounding_alpha = result_alpha[surrounding_mask > 0]
        if surrounding_alpha.size > 0:
            median_surrounding_alpha = np.median(surrounding_alpha)
            if median_surrounding_alpha < 128:
                result_alpha[mask_bin > 0] = np.uint8(np.clip(np.rint(median_surrounding_alpha), 0, 255))

    return result_alpha


def _odd_kernel_size(value: int) -> int:
    value = max(int(value), 1)
    return value if value % 2 == 1 else value + 1


def _measure_ring_complexity(image: np.ndarray, ring_mask: np.ndarray):
    ring_pixels = ring_mask > 0
    if np.count_nonzero(ring_pixels) < 32:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ring_values = gray[ring_pixels]
    if ring_values.size < 32:
        return None

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    edges = cv2.Canny(gray, 50, 150)

    return (
        float(np.std(ring_values)),
        float(np.mean(grad_mag[ring_pixels])),
        float(np.mean(edges[ring_pixels] > 0)),
    )


def _build_flat_tone_fill(
    original_patch: np.ndarray,
    component_mask: np.ndarray,
    ring_mask: np.ndarray,
    blur_sigma: float,
) -> np.ndarray | None:
    ring_pixels = ring_mask > 0
    component_pixels = component_mask > 0
    if np.count_nonzero(ring_pixels) < 24 or np.count_nonzero(component_pixels) == 0:
        return None

    dist_from_mask = cv2.distanceTransform((~component_pixels).astype(np.uint8), cv2.DIST_L2, 5)
    border_dist = 6.0
    near_ring = ring_pixels & (dist_from_mask <= border_dist)
    if np.count_nonzero(near_ring) < 12:
        near_ring = ring_pixels

    fill_color = np.median(original_patch[near_ring], axis=0)
    filled = original_patch.copy().astype(np.float32)
    filled[component_pixels] = fill_color.astype(np.float32)

    smooth = cv2.GaussianBlur(
        filled,
        (0, 0),
        sigmaX=max(float(blur_sigma), 1.0),
        sigmaY=max(float(blur_sigma), 1.0),
    )
    smooth[~component_pixels] = original_patch[~component_pixels]
    return np.clip(smooth, 0, 255).astype(np.uint8)


def _build_edge_propagation_fill(
    original_patch: np.ndarray,
    component_mask: np.ndarray,
    max_iterations: int = 100,
) -> np.ndarray:
    component_pixels = component_mask > 0
    filled = original_patch.copy().astype(np.float32)

    border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    border_dilated = cv2.dilate(component_mask, border_kernel, iterations=1)
    border_mask = (border_dilated > 0) & (~component_pixels)
    if np.count_nonzero(border_mask) > 0:
        border_color = np.median(original_patch[border_mask], axis=0).astype(np.float32)
    else:
        border_color = np.array([127.0, 127.0, 127.0])
    filled[component_pixels] = border_color

    known = (~component_pixels).astype(np.float32)

    for _ in range(max_iterations):
        unknown = component_pixels & (known < 0.5)
        if not np.any(unknown):
            break
        filled_blur = cv2.GaussianBlur(filled, (5, 5), 1.0)
        known_blur = cv2.GaussianBlur(known, (5, 5), 1.0)
        known_blur = np.clip(known_blur, 1e-6, 1.0)
        normalized = filled_blur / known_blur[:, :, None]
        border = unknown & cv2.dilate(known.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
        if not np.any(border):
            break
        filled[border] = normalized[border]
        known[border] = 1.0

    filled[~component_pixels] = original_patch[~component_pixels].astype(np.float32)
    return np.clip(filled, 0, 255).astype(np.uint8)


def _stabilize_simple_regions_with_ns(
    original: np.ndarray,
    mask: np.ndarray,
    inpainted: np.ndarray,
) -> np.ndarray:
    """Replace bad deep-model output with classical inpaint.
    Handles two cases:
    1. Color mismatch: inpainted region color differs greatly from surroundings
    2. Noisy output: inpainted region has excessive texture on simple backgrounds
    """
    mask_binary = (_normalize_binary_mask(mask) > 0).astype(np.uint8)
    mask_pixel_count = np.count_nonzero(mask_binary > 0)
    logger.debug(f"stabilize: mask_pixels={mask_pixel_count}, mask_shape={mask_binary.shape}")
    _diag_print(f"[INPAINT_DIAG] stabilize: mask_pixels={mask_pixel_count}, mask_shape={mask_binary.shape}")
    if mask_pixel_count == 0:
        return inpainted

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    logger.debug(f"stabilize: num_components={num_labels - 1}")
    result = inpainted.copy()
    repaired_components = 0
    skipped_small = 0
    skipped_no_ring = 0
    skipped_no_issue = 0
    color_mismatch_count = 0
    noisy_count = 0

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < SIMPLE_REGION_MIN_AREA:
            skipped_small += 1
            continue

        pad = max(8, int(round(max(w, h) * 0.25)))
        x0 = max(int(x) - pad, 0)
        y0 = max(int(y) - pad, 0)
        x1 = min(int(x + w) + pad, original.shape[1])
        y1 = min(int(y + h) + pad, original.shape[0])

        component_mask = np.where(labels[y0:y1, x0:x1] == label_idx, 255, 0).astype(np.uint8)
        if np.count_nonzero(component_mask) == 0:
            continue

        ring_kernel_size = _odd_kernel_size(min(max(9, int(round(max(w, h) * 0.35))), 31))
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_kernel_size, ring_kernel_size))
        dilated_mask = cv2.dilate(component_mask, ring_kernel, iterations=1)
        ring_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(component_mask))
        ring_stats = _measure_ring_complexity(original[y0:y1, x0:x1], ring_mask)
        if ring_stats is None:
            skipped_no_ring += 1
            continue

        ring_std, ring_grad, ring_edge_density = ring_stats

        inpainted_gray = cv2.cvtColor(result[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        current_std = float(np.std(inpainted_gray[component_mask > 0]))
        current_luma = float(np.mean(inpainted_gray[component_mask > 0]))

        ring_gray = cv2.cvtColor(original[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

        border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        border_dilated = cv2.dilate(component_mask, border_kernel, iterations=1)
        border_ring = cv2.bitwise_and(border_dilated, cv2.bitwise_not(component_mask))
        border_ring_pixels = ring_gray[border_ring > 0]
        if border_ring_pixels.size > 0:
            border_luma_mean = float(np.mean(border_ring_pixels))
        else:
            border_luma_mean = float(np.mean(ring_gray[ring_mask > 0])) if np.any(ring_mask > 0) else 127.0

        wide_ring_pixels = ring_gray[ring_mask > 0]
        if wide_ring_pixels.size > 0:
            wide_luma_mean = float(np.mean(wide_ring_pixels))
        else:
            wide_luma_mean = 127.0

        ring_luma_mean = border_luma_mean
        luma_mismatch = abs(current_luma - ring_luma_mean)

        is_color_mismatch = luma_mismatch > SIMPLE_REGION_LUMA_MISMATCH

        ring_is_simple = (
            ring_std <= SIMPLE_REGION_STD_THRESHOLD
            and ring_grad <= SIMPLE_REGION_GRAD_THRESHOLD
            and ring_edge_density <= SIMPLE_REGION_EDGE_THRESHOLD
        )

        is_noisy = ring_is_simple and current_std > max(ring_std + SIMPLE_REGION_NOISE_EXCESS, ring_std * 1.5)

        if not is_color_mismatch and not is_noisy:
            skipped_no_issue += 1
            continue

        logger.debug(
            f"stabilize component {label_idx}: area={area}, "
            f"current_luma={current_luma:.1f}, border_luma={border_luma_mean:.1f}, wide_luma={wide_luma_mean:.1f}, "
            f"mismatch={luma_mismatch:.1f}, color_mismatch={is_color_mismatch}, "
            f"noisy={is_noisy}, ring_std={ring_std:.1f}, ring_edge={ring_edge_density:.3f}"
        )
        _diag_print(
            f"[INPAINT_DIAG] stabilize component {label_idx}: area={area}, "
            f"current_luma={current_luma:.1f}, border_luma={border_luma_mean:.1f}, wide_luma={wide_luma_mean:.1f}, "
            f"mismatch={luma_mismatch:.1f}, color_mismatch={is_color_mismatch}"
        )

        if is_color_mismatch:
            color_mismatch_count += 1
            use_flat_tone_fill = (
                ring_std <= SIMPLE_REGION_FLAT_STD_THRESHOLD
                and ring_grad <= SIMPLE_REGION_FLAT_GRAD_THRESHOLD
                and ring_edge_density <= SIMPLE_REGION_FLAT_EDGE_THRESHOLD
            )
        else:
            noisy_count += 1
            use_flat_tone_fill = (
                ring_std <= SIMPLE_REGION_FLAT_STD_THRESHOLD
                and ring_grad <= SIMPLE_REGION_FLAT_GRAD_THRESHOLD
                and ring_edge_density <= SIMPLE_REGION_FLAT_EDGE_THRESHOLD
                and (ring_luma_mean <= 120.0 or ring_luma_mean >= 150.0)
            )

        ns_patch = None
        if is_color_mismatch:
            ns_patch = _build_edge_propagation_fill(
                original_patch=original[y0:y1, x0:x1],
                component_mask=component_mask,
            )
            ns_patch_flat = _build_flat_tone_fill(
                original_patch=original[y0:y1, x0:x1],
                component_mask=component_mask,
                ring_mask=ring_mask,
                blur_sigma=max(1.0, ring_kernel_size / 5.0),
            )
            if ns_patch_flat is not None:
                ns_gray = cv2.cvtColor(ns_patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
                flat_gray = cv2.cvtColor(ns_patch_flat, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if abs(np.mean(flat_gray[component_mask > 0]) - ring_luma_mean) < abs(np.mean(ns_gray[component_mask > 0]) - ring_luma_mean):
                    ns_patch = ns_patch_flat
        elif use_flat_tone_fill:
            ns_patch = _build_flat_tone_fill(
                original_patch=original[y0:y1, x0:x1],
                component_mask=component_mask,
                ring_mask=ring_mask,
                blur_sigma=max(1.0, ring_kernel_size / 5.0),
            )
        
        if ns_patch is None:
            ns_patch = cv2.inpaint(
                original[y0:y1, x0:x1],
                component_mask,
                inpaintRadius=max(SIMPLE_REGION_NS_RADIUS, 7),
                flags=cv2.INPAINT_NS,
            )
        feather = cv2.GaussianBlur(
            component_mask.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=max(1.0, ring_kernel_size / 4.0),
            sigmaY=max(1.0, ring_kernel_size / 4.0),
        )[:, :, None]

        current_patch = result[y0:y1, x0:x1].astype(np.float32)
        ns_patch = ns_patch.astype(np.float32)
        blended = np.where(
            component_mask[:, :, None] > 0,
            ns_patch * feather + current_patch * (1.0 - feather),
            current_patch,
        )
        result[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        repaired_components += 1

    logger.debug(
        f"Background stabilization: {num_labels - 1} components, "
        f"skipped(small={skipped_small}, no_ring={skipped_no_ring}, no_issue={skipped_no_issue}), "
        f"detected(color_mismatch={color_mismatch_count}, noisy={noisy_count}), "
        f"repaired={repaired_components}"
    )
    _diag_print(
        f"[INPAINT_DIAG] Background stabilization: {num_labels - 1} components, "
        f"skipped(small={skipped_small}, no_ring={skipped_no_ring}, no_issue={skipped_no_issue}), "
        f"detected(color_mismatch={color_mismatch_count}, noisy={noisy_count}), "
        f"repaired={repaired_components}"
    )

    return result


def _build_inpaint_split_ranges(long_side: int, short_side: int) -> Tuple[list, int, int]:
    """Build overlapped 1D split ranges for extreme-aspect-ratio inpainting."""
    long_side = max(int(long_side), 0)
    short_side = max(int(short_side), 1)
    if long_side <= 0:
        return [], 0, 0

    num_splits = max(int(np.ceil(long_side / (short_side * INPAINT_SPLIT_RATIO))), 1)
    overlap = max(int(short_side * 0.1), 0)
    tile_size = max(1, (long_side + overlap * (num_splits - 1) + num_splits - 1) // num_splits)

    ranges = []
    step = max(tile_size - overlap, 1)
    for ii in range(num_splits):
        start = min(ii * step, max(long_side - tile_size, 0))
        end = min(long_side, start + tile_size)
        ranges.append((start, end))

    if ranges:
        ranges[-1] = (max(long_side - tile_size, 0), long_side)

    return ranges, overlap, tile_size


def _inpaint_single_tile(image: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    if method in ("aot", "aot_local", "lama", "lama_mpe", "lama_large"):
        return aot_inpaint(image, mask, merge_result=False)
    elif method == "ns":
        return cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    else:
        return cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def _dispatch_inpaint_with_split(image: np.ndarray, mask: np.ndarray, method: str, verbose: bool = False) -> np.ndarray:
    """Split inpainting for extreme aspect ratio images"""
    h, w = image.shape[:2]
    is_vertical = h > w
    long_side = h if is_vertical else w
    short_side = w if is_vertical else h

    split_ranges, overlap, tile_size = _build_inpaint_split_ranges(long_side, short_side)
    num_splits = len(split_ranges)

    if verbose:
        logger.info(f"[Inpainting Split] image={w}x{h}, splitting into {num_splits} tiles")

    tiles = []
    for ii, (start, end) in enumerate(split_ranges):
        if is_vertical:
            tile_img = image[start:end, :, :].copy()
            tile_mask = mask[start:end, :].copy()
        else:
            tile_img = image[:, start:end, :].copy()
            tile_mask = mask[:, start:end].copy()

        tile_inpainted = _inpaint_single_tile(tile_img, tile_mask, method)
        tile_mask_3ch = tile_mask[:, :, None] if len(tile_mask.shape) == 2 else tile_mask
        tile_mask_3ch = np.where(tile_mask_3ch > 0, 1, 0).astype(np.float32)
        tile_merged = (tile_inpainted.astype(np.float32) * tile_mask_3ch + tile_img.astype(np.float32) * (1.0 - tile_mask_3ch))
        tile_merged = np.clip(tile_merged, 0, 255).astype(np.uint8)
        tiles.append({'image': tile_merged, 'start': start, 'end': end})

    result = image.copy()
    blend_size = overlap // 2 if overlap > 0 else 0

    for ii, tile_data in enumerate(tiles):
        tile_img = tile_data['image']
        start = tile_data['start']
        end = tile_data['end']

        if num_splits == 1:
            if is_vertical:
                result[start:end, :, :] = tile_img
            else:
                result[:, start:end, :] = tile_img
        elif ii == 0:
            if is_vertical:
                result[start:end - blend_size, :, :] = tile_img[:-blend_size, :, :] if blend_size > 0 else tile_img
            else:
                result[:, start:end - blend_size, :] = tile_img[:, :-blend_size, :] if blend_size > 0 else tile_img
        elif ii == len(tiles) - 1:
            if is_vertical:
                result[start + blend_size:end, :, :] = tile_img[blend_size:, :, :]
            else:
                result[:, start + blend_size:end, :] = tile_img[:, blend_size:, :]
        else:
            if is_vertical:
                result[start + blend_size:end - blend_size, :, :] = tile_img[blend_size:-blend_size, :, :]
            else:
                result[:, start + blend_size:end - blend_size, :] = tile_img[:, blend_size:-blend_size, :]

    return result



# ============ Main Inpainting Interface ============

def advanced_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "lama_large",
    use_crf: bool = True,
    inpainting_size: int = 2048,
    verbose: bool = False
) -> np.ndarray:
    """
    Advanced inpainting interface
    
    Features from manga-translator-ui (MTU):
    1. Split processing - Auto-split for extreme aspect ratio images
    2. Alpha channel handling - Support RGBA images
    3. Memory cleanup - Auto cleanup GPU memory after inference
    
    Args:
        image: BGR/BGRA image
        mask: Binary mask
        method: "lama_large" (MTU default), "lama", "aot", "ns", "telea"
        use_crf: Whether to use CRF to refine mask
        inpainting_size: Maximum processing size
        verbose: Whether to output detailed logs
    
    Returns:
        Inpainted image
    """
    # Handle RGBA image
    original_alpha = None
    image_rgb = image
    if image.ndim == 3 and image.shape[2] == 4:
        image_rgb = image[:, :, :3]
        original_alpha = image[:, :, 3]

    # Normalize mask
    mask_binary = _normalize_binary_mask(mask)
    _diag_print(
        f"[INPAINT_DIAG] advanced_inpaint: method={method}, "
        f"mask_pixels={np.count_nonzero(mask_binary)}, shape={image_rgb.shape}"
    )

    if supports_mtu_inpaint(method):
        try:
            return mtu_inpaint(
                image=image,
                mask=mask_binary,
                method=method,
                inpainting_size=inpainting_size,
                verbose=verbose,
            )
        except Exception as exc:
            logger.warning(f"Embedded MTU inpaint failed for method={method}, fallback to local path: {exc}")

    # CRF refine mask
    if use_crf:
        mask_binary = refine_mask_with_crf(image_rgb, mask_binary)

    # Check if split processing is needed
    h, w = image_rgb.shape[:2]
    aspect_ratio = max(w / h, h / w)
    
    if aspect_ratio > INPAINT_SPLIT_RATIO:
        inpainted_rgb = _dispatch_inpaint_with_split(image_rgb, mask_binary, method, verbose)
    else:
        inpainted_rgb = _inpaint_single_tile(image_rgb, mask_binary, method)

    mask_3ch = mask_binary[:, :, None] if len(mask_binary.shape) == 2 else mask_binary
    mask_3ch = np.where(mask_3ch > 0, 1, 0).astype(np.float32)
    inpainted_rgb = (inpainted_rgb.astype(np.float32) * mask_3ch + image_rgb.astype(np.float32) * (1.0 - mask_3ch))
    inpainted_rgb = np.clip(inpainted_rgb, 0, 255).astype(np.uint8)

    if method != "aot_local":
        try:
            inpainted_rgb = _stabilize_simple_regions_with_ns(
                original=image_rgb,
                mask=mask_binary,
                inpainted=inpainted_rgb,
            )
        except Exception as exc:
            logger.warning(f"simple-region stabilization failed, keep raw inpaint result: {exc}")

    # Cleanup memory
    _cleanup_memory()

    # Handle alpha channel
    if original_alpha is not None:
        alpha = _inpaint_handle_alpha_channel(original_alpha, mask_binary)
        return np.concatenate([inpainted_rgb, alpha[:, :, None]], axis=2)

    return inpainted_rgb
