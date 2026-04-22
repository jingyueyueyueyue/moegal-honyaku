import json
import os

from app.core.logger import logger
from app.core.paths import PROJECT_ROOT

CONF_FILE = PROJECT_ROOT / "saved" / "user_conf.json"

TRANSLATE_API_TYPE_OPTIONS = ("dashscope", "openai")
TRANSLATE_MODE_OPTIONS = ("parallel", "structured", "context", "context-batch", "context-sequential")
OCR_ENGINE_OPTIONS = ("local", "vision")
LOCAL_OCR_ENGINE_OPTIONS = ("auto", "manga_ocr", "paddle_ocr", "model_48px", "model_48px_ctc")

INPAINT_METHOD_FAST = "fast"
INPAINT_METHOD_QUALITY = "quality"
INPAINT_METHOD_OPTIONS = (INPAINT_METHOD_FAST, INPAINT_METHOD_QUALITY)
_LEGACY_INPAINT_METHOD_MAP = {
    "fast": INPAINT_METHOD_FAST,
    "quality": INPAINT_METHOD_QUALITY,
    "telea": INPAINT_METHOD_FAST,
    "ns": INPAINT_METHOD_FAST,
    "aot": INPAINT_METHOD_FAST,
    "lama": INPAINT_METHOD_QUALITY,
    "lama_mpe": INPAINT_METHOD_QUALITY,
    "lama_large": INPAINT_METHOD_QUALITY,
}


def normalize_inpaint_method(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return _LEGACY_INPAINT_METHOD_MAP.get(normalized)


_DEFAULT_OCR_ENGINE = os.getenv("MOEGAL_OCR_ENGINE", "local").lower()
if _DEFAULT_OCR_ENGINE not in OCR_ENGINE_OPTIONS:
    logger.warning(f"Invalid MOEGAL_OCR_ENGINE={_DEFAULT_OCR_ENGINE}, fallback to local")
    _DEFAULT_OCR_ENGINE = "local"

_DEFAULT_INPAINT_METHOD = normalize_inpaint_method(os.getenv("INPAINT_METHOD", INPAINT_METHOD_QUALITY))
if _DEFAULT_INPAINT_METHOD not in INPAINT_METHOD_OPTIONS:
    raw_method = os.getenv("INPAINT_METHOD", INPAINT_METHOD_QUALITY)
    logger.warning(f"Invalid INPAINT_METHOD={raw_method}, fallback to {INPAINT_METHOD_QUALITY}")
    _DEFAULT_INPAINT_METHOD = INPAINT_METHOD_QUALITY

_AUTO_SAVE_IMAGE = os.getenv("AUTO_SAVE_IMAGE", "false").lower() in ("true", "1", "yes")
_ENABLE_AI_LINEBREAK = os.getenv("AI_LINEBREAK_ENABLED", "true").lower() in ("true", "1", "yes")
_AI_LINEBREAK_MIN_LENGTH = int(os.getenv("AI_LINEBREAK_MIN_LENGTH", "50"))


class CustomConf:
    def __init__(
        self,
        translate_api_type="openai",
        translate_mode="context",
        ocr_engine=None,
        auto_save_image=None,
        enable_ai_linebreak=None,
        local_ocr_engine=None,
    ):
        self.translate_api_type = translate_api_type
        self.translate_mode = translate_mode
        self.ocr_engine = ocr_engine if ocr_engine else _DEFAULT_OCR_ENGINE
        self.auto_save_image = auto_save_image if auto_save_image is not None else _AUTO_SAVE_IMAGE

        default_local_ocr_engine = os.getenv("LOCAL_OCR_ENGINE", "auto").lower()
        if default_local_ocr_engine not in LOCAL_OCR_ENGINE_OPTIONS:
            logger.warning(f"Invalid LOCAL_OCR_ENGINE={default_local_ocr_engine}, fallback to auto")
            default_local_ocr_engine = "auto"
        self.local_ocr_engine = local_ocr_engine if local_ocr_engine else default_local_ocr_engine

        self.enable_ai_linebreak = enable_ai_linebreak if enable_ai_linebreak is not None else _ENABLE_AI_LINEBREAK
        self.ai_linebreak_min_length = _AI_LINEBREAK_MIN_LENGTH

        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.dashscope_model = os.getenv("DASHSCOPE_MODEL", "qwen3-max")

        self.vision_ocr_provider = os.getenv("VISION_OCR_PROVIDER", "openai").lower()
        self.vision_openai_api_key = os.getenv("VISION_OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        self.vision_openai_base_url = os.getenv("VISION_OPENAI_BASE_URL", "") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.vision_openai_model = os.getenv("VISION_OCR_MODEL", "") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.vision_dashscope_api_key = os.getenv("VISION_DASHSCOPE_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", "")
        self.vision_dashscope_base_url = os.getenv("VISION_DASHSCOPE_BASE_URL", "") or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.vision_dashscope_model = os.getenv("VISION_DASHSCOPE_MODEL", "") or os.getenv("DASHSCOPE_MODEL", "qwen-vl-plus")

        self.inpaint_method = _DEFAULT_INPAINT_METHOD
        self.use_crf_refine = self.inpaint_method == INPAINT_METHOD_QUALITY

        self.merge_textboxes = os.getenv("MERGE_TEXTBOXES", "true").lower() in ("true", "1", "yes")
        self.use_yolo_obb = os.getenv("USE_YOLO_OBB", "true").lower() in ("true", "1", "yes")
        self.yolo_obb_conf = float(os.getenv("YOLO_OBB_CONF", "0.4"))

        self._load_from_file()
        self._sync_derived_inpaint_settings()

    def _sync_derived_inpaint_settings(self):
        normalized = normalize_inpaint_method(getattr(self, "inpaint_method", None)) or _DEFAULT_INPAINT_METHOD
        self.inpaint_method = normalized
        self.use_crf_refine = normalized == INPAINT_METHOD_QUALITY

    def _load_from_file(self):
        persist_keys = {
            "translate_api_type",
            "translate_mode",
            "ocr_engine",
            "auto_save_image",
            "enable_ai_linebreak",
            "ai_linebreak_min_length",
            "local_ocr_engine",
            "inpaint_method",
            "use_crf_refine",
            "merge_textboxes",
            "use_yolo_obb",
            "yolo_obb_conf",
            "vision_ocr_provider",
        }

        try:
            if not CONF_FILE.exists():
                return
            with open(CONF_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)

            for key in persist_keys:
                if key in saved:
                    setattr(self, key, saved[key])
            logger.debug(f"Loaded config from {CONF_FILE}")
        except Exception as exc:
            logger.warning(f"Failed to load config file: {exc}")

    def _save_to_file(self):
        persist_keys = {
            "translate_api_type",
            "translate_mode",
            "ocr_engine",
            "auto_save_image",
            "enable_ai_linebreak",
            "ai_linebreak_min_length",
            "local_ocr_engine",
            "inpaint_method",
            "use_crf_refine",
            "merge_textboxes",
            "use_yolo_obb",
            "yolo_obb_conf",
            "vision_ocr_provider",
        }

        try:
            CONF_FILE.parent.mkdir(parents=True, exist_ok=True)
            to_save = {k: v for k, v in self.__dict__.items() if k in persist_keys}
            with open(CONF_FILE, "w", encoding="utf-8") as f:
                json.dump(to_save, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to save config file: {exc}")

    def update_conf(self, attr, v):
        assert hasattr(self, attr), f"attr '{attr}' is not exists."

        if attr == "translate_api_type":
            assert v in TRANSLATE_API_TYPE_OPTIONS, f"translate_api_type must be one of {TRANSLATE_API_TYPE_OPTIONS}"
        if attr == "translate_mode":
            assert v in TRANSLATE_MODE_OPTIONS, f"translate_mode must be one of {TRANSLATE_MODE_OPTIONS}"
        if attr == "ocr_engine":
            assert v in OCR_ENGINE_OPTIONS, f"ocr_engine must be one of {OCR_ENGINE_OPTIONS}"
        if attr == "auto_save_image":
            v = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
        if attr == "enable_ai_linebreak":
            v = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
        if attr == "vision_ocr_provider":
            assert v in ("openai", "dashscope"), "vision_ocr_provider must be openai or dashscope"
        if attr == "local_ocr_engine":
            assert v in LOCAL_OCR_ENGINE_OPTIONS, f"local_ocr_engine must be one of {LOCAL_OCR_ENGINE_OPTIONS}"
        if attr == "inpaint_method":
            v = normalize_inpaint_method(v)
            assert v in INPAINT_METHOD_OPTIONS, f"inpaint_method must be one of {INPAINT_METHOD_OPTIONS}"
        if attr == "use_crf_refine":
            v = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
        if attr == "merge_textboxes":
            v = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
        if attr == "use_yolo_obb":
            v = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
        if attr == "yolo_obb_conf":
            v = float(v)
            assert 0.1 <= v <= 0.9, "yolo_obb_conf must be within 0.1-0.9"

        setattr(self, attr, v)
        if attr in {"inpaint_method", "use_crf_refine"}:
            self._sync_derived_inpaint_settings()

        logger.info(f"Set {attr}={getattr(self, attr, None)}")
        self._save_to_file()
        return {
            attr: getattr(self, attr, None),
            "status": "success",
        }

    def to_dict(self, exclude=None):
        exclude = exclude or []
        assert isinstance(exclude, list)
        return {k: v for k, v in self.__dict__.items() if k not in exclude}


custom_conf = CustomConf()
