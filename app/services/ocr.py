import base64
import io
import os
import re
from threading import Lock

import torch
from dotenv import load_dotenv
from manga_ocr import MangaOcr
from openai import OpenAI
from ultralytics import YOLO

from app.core.logger import logger
from app.core.paths import MODELS_DIR

DET_MODEL_PATH = MODELS_DIR / "comic-text-segmenter.pt"
MOCR_MODEL_PATH = MODELS_DIR / "manga-ocr-base"
load_dotenv()

# ============ YOLO 文本检测参数 ============
# 置信度阈值：越低检测越多文字区域，但可能有误检（默认 0.25，范围 0.1-0.5）
DET_CONF_THRESHOLD = float(os.getenv("DET_CONF_THRESHOLD", "0.25"))
# IOU 阈值：用于合并重叠的检测框（默认 0.5）
DET_IOU_THRESHOLD = float(os.getenv("DET_IOU_THRESHOLD", "0.5"))
# 检测图像尺寸：越大检测越精确但越慢（默认 1024，范围 512-2048）
DET_IMG_SIZE = int(os.getenv("DET_IMG_SIZE", "1024"))
# 是否在检测前增强对比度（对低对比度图片有帮助）
DET_ENHANCE_CONTRAST = os.getenv("DET_ENHANCE_CONTRAST", "true").lower() in ("true", "1", "yes")

# ============ OCR 引擎配置 ============
# 本地 OCR 引擎细分选择（仅当 ocr_engine=local 时生效）：
# - manga_ocr: 仅支持日文，准确率高
# - paddle_ocr: 支持多语言（日/英/中/韩等）
# - auto: 自动检测语言，优先用 MangaOCR，英文时回退 PaddleOCR
OCR_ENGINE = os.getenv("OCR_ENGINE", "auto").lower()
# PaddleOCR 语言设置（默认日文，可选：en, ch, korean 等）
PADDLE_OCR_LANG = os.getenv("PADDLE_OCR_LANG", "japan")
# 是否使用 PaddleOCR 轻量级模型（推荐 CPU 环境开启，速度更快）
PADDLE_OCR_LITE = os.getenv("PADDLE_OCR_LITE", "true").lower() in ("true", "1", "yes")

# ============ OCR 高级优化参数 ============
# 是否启用 OCR 图像预处理（降噪 + 对比度增强 + 锐化），提升模糊图片识别率
OCR_PREPROCESS = os.getenv("OCR_PREPROCESS", "true").lower() in ("true", "1", "yes")
# 是否启用 OCR 后处理（清理噪声字符，如多余竖线、重复字符等）
OCR_POSTPROCESS = os.getenv("OCR_POSTPROCESS", "true").lower() in ("true", "1", "yes")
# OCR 置信度过滤阈值（仅对 PaddleOCR 生效，低于此值的结果会被过滤）
OCR_CONF_THRESHOLD = float(os.getenv("OCR_CONF_THRESHOLD", "0.5"))
# 是否启用文本方向自动校正（识别倾斜/旋转的文字）
OCR_AUTO_ROTATE = os.getenv("OCR_AUTO_ROTATE", "true").lower() in ("true", "1", "yes")

_MODEL_LOCK = Lock()
_DET_MODEL: YOLO | None = None
_MOCR: MangaOcr | None = None
_PADDLE_OCR = None


def _is_true_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_cuda_related_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "cuda",
            "cudnn",
            "no kernel image",
            "driver",
            "device-side assert",
        )
    )


def _is_cuda_runtime_usable() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() = False"
    try:
        a = torch.tensor([1, 2, 3], device="cuda")
        b = torch.tensor([2], device="cuda")
        _ = torch.isin(a, b)
        torch.cuda.synchronize()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _resolve_device() -> tuple[torch.device, bool]:
    gpu_enabled = _is_true_env("MOEGAL_USE_GPU", default=False)
    cuda_usable, cuda_fail_reason = _is_cuda_runtime_usable() if gpu_enabled else (False, "")
    use_cuda = gpu_enabled and cuda_usable

    if gpu_enabled and not use_cuda:
        logger.warning(f"检测到 GPU 已启用但 CUDA 不可用，自动回退 CPU。原因：{cuda_fail_reason}")

    return torch.device("cuda:0") if use_cuda else torch.device("cpu"), use_cuda


def warmup_models() -> tuple[YOLO, MangaOcr]:
    global _DET_MODEL, _MOCR

    with _MODEL_LOCK:
        if _DET_MODEL is not None and _MOCR is not None:
            return _DET_MODEL, _MOCR

        device, use_cuda = _resolve_device()

        if _DET_MODEL is None:
            _DET_MODEL = YOLO(str(DET_MODEL_PATH)).to(device)
            logger.info(f"气泡检测模型加载成功，使用：{_DET_MODEL.device}")

        if _MOCR is None:
            if use_cuda:
                try:
                    _MOCR = MangaOcr(pretrained_model_name_or_path=str(MOCR_MODEL_PATH), force_cpu=False)
                    logger.info("MangaOCR 加载成功，使用：cuda")
                except Exception as exc:
                    if not _is_cuda_related_error(exc):
                        raise
                    logger.warning(f"MangaOCR CUDA 初始化失败，自动回退 CPU。原因：{exc}")
                    _MOCR = MangaOcr(pretrained_model_name_or_path=str(MOCR_MODEL_PATH), force_cpu=True)
                    logger.info("MangaOCR 加载成功，使用：cpu")
            else:
                _MOCR = MangaOcr(pretrained_model_name_or_path=str(MOCR_MODEL_PATH), force_cpu=True)
                logger.info("MangaOCR 加载成功，使用：cpu")

        return _DET_MODEL, _MOCR


def get_det_model() -> YOLO:
    det_model, _ = warmup_models()
    return det_model


def get_mocr() -> MangaOcr:
    _, mocr = warmup_models()
    return mocr


def get_paddle_ocr():
    """获取 PaddleOCR 实例（懒加载）"""
    global _PADDLE_OCR

    if _PADDLE_OCR is not None:
        return _PADDLE_OCR

    try:
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x 新 API
        # 参数说明：
        # - use_doc_orientation_classify: 是否使用文档方向分类
        # - use_doc_unwarping: 是否使用文档矫正
        # - use_textline_orientation: 是否使用文本行方向分类
        # - lang: 语言设置（japan, en, ch, korean 等）
        _PADDLE_OCR = PaddleOCR(
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang=PADDLE_OCR_LANG,
            show_log=False,
        )
        logger.info(f"PaddleOCR 加载成功，语言: {PADDLE_OCR_LANG}")
        return _PADDLE_OCR

    except ImportError:
        logger.warning("PaddleOCR 未安装，请运行: pip install paddleocr paddlepaddle")
        return None


def _detect_text_language(text: str) -> str:
    """
    检测文本的主要语言类型
    
    Returns:
        'ja': 日文
        'en': 英文
        'zh': 中文
        'ko': 韩文
        'mixed': 混合
        'unknown': 未知
    """
    if not text:
        return 'unknown'

    # 统计各语言字符比例
    ja_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text))  # 平假名+片假名
    zh_chars = len(re.findall(r'[\u4E00-\u9FFF]', text))  # 汉字
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    ko_chars = len(re.findall(r'[\uAC00-\uD7AF]', text))  # 韩文
    
    total = len(text)
    if total == 0:
        return 'unknown'

    # 判断主要语言
    ja_ratio = ja_chars / total
    en_ratio = en_chars / total
    zh_ratio = zh_chars / total
    ko_ratio = ko_chars / total

    # 如果有假名，大概率是日文
    if ja_ratio > 0.1:
        return 'ja'
    
    # 英文比例高
    if en_ratio > 0.5:
        return 'en'
    
    # 韩文比例高
    if ko_ratio > 0.3:
        return 'ko'
    
    # 有汉字但无假名，可能是中文
    if zh_ratio > 0.3 and ja_ratio < 0.05:
        return 'zh'
    
    return 'mixed' if (ja_ratio + en_ratio + zh_ratio) > 0.2 else 'unknown'


def _is_mainly_english(text: str) -> bool:
    """判断文本是否主要是英文"""
    if not text:
        return False
    
    # 清理空白
    clean_text = re.sub(r'\s+', '', text)
    if not clean_text:
        return False
    
    # 统计英文字母比例
    en_chars = len(re.findall(r'[a-zA-Z0-9]', clean_text))
    # 统计日文字符
    ja_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', clean_text))
    
    # 如果英文字符占比高，且几乎没有日文字符
    total = len(clean_text)
    return (en_chars / total > 0.6) and (ja_chars / total < 0.2)


def _preprocess_ocr_image(image_pil):
    """
    OCR图像预处理：提升识别准确率
    
    处理流程：
    1. 降噪
    2. 对比度增强
    3. 锐化
    """
    import cv2
    import numpy as np
    from PIL import Image
    
    # PIL 转 OpenCV
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # 1. 轻度降噪（保留文字边缘）
    try:
        img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)
    except Exception:
        pass  # 降噪失败则跳过
    
    # 2. 对比度增强（CLAHE）
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_cv = cv2.merge([l, a, b])
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_LAB2BGR)
    
    # 3. 锐化
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    img_cv = cv2.filter2D(img_cv, -1, kernel)
    
    # 转回 PIL
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def _postprocess_ocr_text(text: str) -> str:
    """
    OCR结果后处理：清理噪声字符
    
    处理：
    1. 移除常见的OCR错误字符
    2. 规范化空白
    3. 移除重复字符
    """
    if not text:
        return text
    
    # 移除常见的OCR噪声字符
    noise_chars = ['|', '¦', '‖', '═', '║', '▌', '▍', '▎', '▏', '━', '┃']
    for char in noise_chars:
        text = text.replace(char, '')
    
    # 规范化空白
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 移除行首行尾空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    
    # 移除明显的重复错误（如 "あああああ" 超过5个）
    text = re.sub(r'(.)\1{5,}', r'\1\1\1', text)
    
    return text


def ocr_recognize(image_pil, prefer_engine: str = None) -> str:
    """
    统一OCR识别接口，自动选择最佳引擎
    
    Args:
        image_pil: PIL Image 对象
        prefer_engine: 指定引擎 ('manga_ocr', 'paddle_ocr', 'auto', 'vision', 'local')
        
    Returns:
        识别出的文本
    """
    import cv2
    import numpy as np
    
    engine = prefer_engine or OCR_ENGINE
    
    # 多模态 OCR 模式
    if engine == 'vision':
        return vision_ocr_recognize(image_pil)
    
    # local 模式：使用本地模型（auto 逻辑）
    if engine == 'local':
        engine = 'auto'
    
    # 如果指定了 PaddleOCR
    if engine == 'paddle_ocr':
        paddle_ocr = get_paddle_ocr()
        if paddle_ocr is None:
            logger.warning("PaddleOCR 不可用，回退到 MangaOCR")
            mocr = get_mocr()
            return mocr(image_pil)
        
        # PIL 转 OpenCV
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # PaddleOCR 3.x 使用 predict() 方法
        result = paddle_ocr.predict(input=img_cv)
        
        if not result:
            return ""
        
        # PaddleOCR 3.x 返回格式：每个 res 有 'rec_texts' 属性
        texts = []
        for res in result:
            if hasattr(res, 'rec_texts') and res.rec_texts:
                texts.extend(res.rec_texts)
            elif hasattr(res, 'res') and isinstance(res.res, dict):
                # 兼容可能的返回格式
                if 'rec_texts' in res.res:
                    texts.extend(res.res['rec_texts'])
        
        return '\n'.join(texts)
    
    # 如果指定了 MangaOCR
    if engine == 'manga_ocr':
        mocr = get_mocr()
        return mocr(image_pil)
    
    # auto 模式：先用 MangaOCR，检测语言后决定是否切换
    if engine == 'auto':
        mocr = get_mocr()
        text = mocr(image_pil)
        
        # 如果检测到主要是英文，用 PaddleOCR 重试
        if _is_mainly_english(text):
            paddle_ocr = get_paddle_ocr()
            if paddle_ocr is not None:
                img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                
                # PaddleOCR 3.x 使用 predict() 方法
                result = paddle_ocr.predict(input=img_cv)
                
                if result:
                    texts = []
                    for res in result:
                        if hasattr(res, 'rec_texts') and res.rec_texts:
                            texts.extend(res.rec_texts)
                        elif hasattr(res, 'res') and isinstance(res.res, dict):
                            if 'rec_texts' in res.res:
                                texts.extend(res.res['rec_texts'])
                    
                    paddle_text = '\n'.join(texts)
                    
                    # 如果 PaddleOCR 结果更合理（更长或更可信），使用它
                    if len(paddle_text) >= len(text) * 0.8:
                        logger.debug(f"检测到英文内容，使用 PaddleOCR: {paddle_text[:30]}...")
                        return paddle_text
        
        return text
    
    # 默认使用 MangaOCR
    mocr = get_mocr()
    return mocr(image_pil)


def detect_text_regions(image_cv):
    """
    检测图像中的文字区域
    
    Args:
        image_cv: OpenCV BGR 图像
        
    Returns:
        bboxes: 文字区域边界框列表 [[x1,y1,x2,y2], ...]
    """
    import cv2
    import numpy as np
    
    det_model = get_det_model()
    
    # 图像预处理：增强对比度（可选）
    if DET_ENHANCE_CONTRAST:
        # CLAHE 对比度增强
        lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        process_img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        process_img = image_cv
    
    # YOLO 检测
    results = det_model(
        process_img,
        verbose=False,
        conf=DET_CONF_THRESHOLD,
        iou=DET_IOU_THRESHOLD,
        imgsz=DET_IMG_SIZE,
    )
    
    if len(results) == 0 or results[0].boxes is None:
        return []
    
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    
    # 过滤太小的框（噪点）
    height, width = image_cv.shape[:2]
    min_box_area = max(100, (height * width) * 0.0001)
    
    filtered_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area >= min_box_area:
            filtered_bboxes.append(bbox)
    
    logger.info(f"检测到 {len(filtered_bboxes)} 个文字区域（原始 {len(bboxes)} 个）")
    
    return filtered_bboxes


# ============ 多模态 OCR 配置（仅当 ocr_engine=vision 时生效）============
# 多模态 API 供应商：openai 或 dashscope
VISION_OCR_PROVIDER = os.getenv("VISION_OCR_PROVIDER", "openai").lower()

# ===== OpenAI 兼容 API 配置 =====
# 优先使用独立配置，未配置则回退到翻译服务的 OpenAI 配置
VISION_OPENAI_API_KEY = os.getenv("VISION_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
VISION_OPENAI_BASE_URL = os.getenv("VISION_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
VISION_OCR_MODEL = os.getenv("VISION_OCR_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ===== DashScope API 配置 =====
# 优先使用独立配置，未配置则回退到翻译服务的 DashScope 配置
VISION_DASHSCOPE_API_KEY = os.getenv("VISION_DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
VISION_DASHSCOPE_BASE_URL = os.getenv("VISION_DASHSCOPE_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
VISION_DASHSCOPE_MODEL = os.getenv("VISION_DASHSCOPE_MODEL") or os.getenv("DASHSCOPE_MODEL", "qwen-vl-plus")

# OCR 提示词：要求模型仅输出识别的文字，不翻译、不解释
VISION_OCR_PROMPT = os.getenv(
    "VISION_OCR_PROMPT",
    "请识别这张图片中的所有文字，只输出识别到的文字内容，不要添加任何解释、翻译或格式化。"
    "如果有多行文字，用换行符分隔。如果没有文字，返回空字符串。"
)

_VISION_CLIENT = None


def _get_vision_client():
    """获取多模态 API 客户端（懒加载）"""
    global _VISION_CLIENT
    if _VISION_CLIENT is not None:
        return _VISION_CLIENT
    
    if VISION_OCR_PROVIDER == "openai":
        api_key = VISION_OPENAI_API_KEY
        base_url = VISION_OPENAI_BASE_URL
        model = VISION_OCR_MODEL
        if not api_key:
            raise RuntimeError("VISION_OCR_PROVIDER=openai 但 VISION_OPENAI_API_KEY 和 OPENAI_API_KEY 均未配置")
    elif VISION_OCR_PROVIDER == "dashscope":
        api_key = VISION_DASHSCOPE_API_KEY
        base_url = VISION_DASHSCOPE_BASE_URL
        model = VISION_DASHSCOPE_MODEL
        if not api_key:
            raise RuntimeError("VISION_OCR_PROVIDER=dashscope 但 VISION_DASHSCOPE_API_KEY 和 DASHSCOPE_API_KEY 均未配置")
    else:
        raise RuntimeError(f"不支持的 VISION_OCR_PROVIDER: {VISION_OCR_PROVIDER}")
    
    _VISION_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    logger.info(f"多模态 OCR 客户端初始化成功: provider={VISION_OCR_PROVIDER}, base_url={base_url}, model={model}")
    return _VISION_CLIENT


def _get_vision_model() -> str:
    """获取当前配置的多模态模型名称"""
    if VISION_OCR_PROVIDER == "openai":
        return VISION_OCR_MODEL
    elif VISION_OCR_PROVIDER == "dashscope":
        return VISION_DASHSCOPE_MODEL
    return "gpt-4o-mini"


def vision_ocr_recognize(image_pil) -> str:
    """
    使用多模态模型进行 OCR 识别
    
    Args:
        image_pil: PIL Image 对象
        
    Returns:
        识别出的文本
    """
    client = _get_vision_client()
    model = _get_vision_model()
    
    # 将 PIL Image 转为 base64
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
        )
        
        text = response.choices[0].message.content or ""
        text = text.strip()
        logger.debug(f"多模态 OCR 识别结果: {text[:50]}...")
        return text
        
    except Exception as e:
        logger.error(f"多模态 OCR 识别失败: {e}")
        return ""
