import os

from app.core.logger import logger

TRANSLATE_API_TYPE_OPTIONS = ("dashscope", "openai")
TRANSLATE_MODE_OPTIONS = ("parallel", "structured")
OCR_ENGINE_OPTIONS = ("local", "vision")

# 从环境变量读取 OCR 引擎默认值（由 start.cmd 设置）
_DEFAULT_OCR_ENGINE = os.getenv("MOEGAL_OCR_ENGINE", "local").lower()
if _DEFAULT_OCR_ENGINE not in OCR_ENGINE_OPTIONS:
    logger.warning(f"无效的 MOEGAL_OCR_ENGINE={_DEFAULT_OCR_ENGINE}，使用默认值 local")
    _DEFAULT_OCR_ENGINE = "local"

# 从环境变量读取是否自动保存图片（默认关闭）
_AUTO_SAVE_IMAGE = os.getenv("AUTO_SAVE_IMAGE", "false").lower() in ("true", "1", "yes")

# 从环境变量读取是否启用AI智能断句（默认启用）
_ENABLE_AI_LINEBREAK = os.getenv("AI_LINEBREAK_ENABLED", "true").lower() in ("true", "1", "yes")


class CustomConf:
    def __init__(
            self,
            # 默认使用 OpenAI，前端可改为 dashscope。
            translate_api_type="openai",
            # parallel: 每句并发请求；structured: 单请求列表输入输出。
            translate_mode="parallel",
            # local: 本地 OCR 模型（MangaOCR/PaddleOCR）；vision: 多模态模型 OCR
            ocr_engine=None,
            # 是否自动保存翻译前后的图片（默认由环境变量决定）
            auto_save_image=None,
            # 是否启用AI智能断句（默认由环境变量决定）
            enable_ai_linebreak=None,
            ):
        self.translate_api_type = translate_api_type
        self.translate_mode = translate_mode
        # OCR 引擎默认值由环境变量 MOEGAL_OCR_ENGINE 决定
        self.ocr_engine = ocr_engine if ocr_engine else _DEFAULT_OCR_ENGINE
        self.auto_save_image = auto_save_image if auto_save_image is not None else _AUTO_SAVE_IMAGE
        # AI断句默认值由环境变量 AI_LINEBREAK_ENABLED 决定
        self.enable_ai_linebreak = enable_ai_linebreak if enable_ai_linebreak is not None else _ENABLE_AI_LINEBREAK

    def update_conf(self, attr, v):
        assert hasattr(self, attr), f"attr '{attr}' is not exists."
        if attr == "translate_api_type":
            assert v in TRANSLATE_API_TYPE_OPTIONS, (
                f"translate_api_type 必须是 {TRANSLATE_API_TYPE_OPTIONS}"
            )
        if attr == "translate_mode":
            assert v in TRANSLATE_MODE_OPTIONS, (
                f"translate_mode 必须是 {TRANSLATE_MODE_OPTIONS}"
            )
        if attr == "ocr_engine":
            assert v in OCR_ENGINE_OPTIONS, (
                f"ocr_engine 必须是 {OCR_ENGINE_OPTIONS}"
            )
        if attr == "auto_save_image":
            v = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
        if attr == "enable_ai_linebreak":
            v = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
        setattr(self, attr, v)
        logger.info(f"将 {attr} 设置为 {v}")
        return {
            attr: getattr(self, attr, None),
            "status": "success"
        }

    def to_dict(self, exclude=None):
        exclude = exclude or []
        assert isinstance(exclude, list)
        return {
            k: v for k, v in self.__dict__.items() if k not in exclude
        }

custom_conf = CustomConf()
