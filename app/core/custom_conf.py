import os

from app.core.logger import logger

TRANSLATE_API_TYPE_OPTIONS = ("dashscope", "openai")
# parallel: 每句并发请求
# structured: 单请求列表输入输出
# context: 上下文感知翻译（单图片内）
# context-batch: 批量等待所有图片，一次性翻译（跨图片上下文）
# context-sequential: 顺序翻译，累积上下文（跨图片上下文）
TRANSLATE_MODE_OPTIONS = ("parallel", "structured", "context", "context-batch", "context-sequential")
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
# AI断句最小文本长度（默认50字符）
_AI_LINEBREAK_MIN_LENGTH = int(os.getenv("AI_LINEBREAK_MIN_LENGTH", "50"))


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
        
        # ============ AI断句配置（运行时可由前端设置）============
        # 是否启用AI智能断句（复用翻译模型，无需单独配置模型）
        self.enable_ai_linebreak = enable_ai_linebreak if enable_ai_linebreak is not None else _ENABLE_AI_LINEBREAK
        # 触发断句的最小文本长度（短于此长度不处理）
        self.ai_linebreak_min_length = _AI_LINEBREAK_MIN_LENGTH
        
        # ============ 动态翻译配置（运行时可由前端设置）============
        # OpenAI 翻译服务配置
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")  # API密钥
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")  # API地址
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 模型名称
        
        # DashScope（阿里云）翻译服务配置
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")  # API密钥
        self.dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")  # API地址
        self.dashscope_model = os.getenv("DASHSCOPE_MODEL", "qwen3-max")  # 模型名称
        
        # ============ 动态 OCR Vision 配置（运行时可由前端设置）============
        # 多模态OCR供应商：openai 或 dashscope
        self.vision_ocr_provider = os.getenv("VISION_OCR_PROVIDER", "openai").lower()
        # OpenAI Vision OCR 配置（未配置时回退到翻译服务的OpenAI配置）
        self.vision_openai_api_key = os.getenv("VISION_OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        self.vision_openai_base_url = os.getenv("VISION_OPENAI_BASE_URL", "") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.vision_openai_model = os.getenv("VISION_OCR_MODEL", "") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # DashScope Vision OCR 配置（未配置时回退到翻译服务的DashScope配置）
        self.vision_dashscope_api_key = os.getenv("VISION_DASHSCOPE_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", "")
        self.vision_dashscope_base_url = os.getenv("VISION_DASHSCOPE_BASE_URL", "") or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.vision_dashscope_model = os.getenv("VISION_DASHSCOPE_MODEL", "") or os.getenv("DASHSCOPE_MODEL", "qwen-vl-plus")

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
        if attr == "vision_ocr_provider":
            assert v in ("openai", "dashscope"), (
                f"vision_ocr_provider 必须是 openai 或 dashscope"
            )
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