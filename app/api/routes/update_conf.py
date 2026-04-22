from fastapi import APIRouter
from app.core.custom_conf import (
    custom_conf,
    TRANSLATE_API_TYPE_OPTIONS,
    TRANSLATE_MODE_OPTIONS,
    OCR_ENGINE_OPTIONS,
    LOCAL_OCR_ENGINE_OPTIONS,
    INPAINT_METHOD_OPTIONS,
)
from pydantic import BaseModel
from typing import Union, Optional

update_conf_router = APIRouter()

class UpdateItem(BaseModel):
    attr: str
    v: Union[str, float, bool] = None

class BatchUpdateItem(BaseModel):
    """批量更新配置项"""
    translate_api_type: Optional[str] = None
    translate_mode: Optional[str] = None
    ocr_engine: Optional[str] = None
    auto_save_image: Optional[bool] = None
    # AI 断句配置
    enable_ai_linebreak: Optional[bool] = None
    ai_linebreak_min_length: Optional[int] = None
    # OpenAI 翻译配置
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: Optional[str] = None
    # DashScope 翻译配置
    dashscope_api_key: Optional[str] = None
    dashscope_base_url: Optional[str] = None
    dashscope_model: Optional[str] = None
    # Vision OCR 配置
    vision_ocr_provider: Optional[str] = None
    vision_openai_api_key: Optional[str] = None
    vision_openai_base_url: Optional[str] = None
    vision_openai_model: Optional[str] = None
    vision_dashscope_api_key: Optional[str] = None
    vision_dashscope_base_url: Optional[str] = None
    vision_dashscope_model: Optional[str] = None
    # 本地 OCR 引擎细分配置
    local_ocr_engine: Optional[str] = None
    # Inpainting 配置
    inpaint_method: Optional[str] = None
    use_crf_refine: Optional[bool] = None
    # 文本框合并配置
    merge_textboxes: Optional[bool] = None

@update_conf_router.post("/conf/init")
def init_conf():
    # 初始化默认值：OpenAI + 并行模式 + 本地 OCR。
    custom_conf.update_conf("translate_api_type", "openai")
    custom_conf.update_conf("translate_mode", "parallel")
    custom_conf.update_conf("ocr_engine", "local")
    return custom_conf.to_dict()

@update_conf_router.post("/conf/update")
def update_conf(item: UpdateItem):
    custom_conf.update_conf(item.attr, item.v)
    return custom_conf.to_dict()

@update_conf_router.post("/conf/batch-update")
def batch_update_conf(item: BatchUpdateItem):
    """批量更新配置项"""
    updated = []
    for attr, value in item.model_dump(exclude_none=True).items():
        if value is not None:
            custom_conf.update_conf(attr, value)
            updated.append(attr)
    return {
        "status": "success",
        "updated": updated,
        "conf": custom_conf.to_dict()
    }

@update_conf_router.get("/conf/query")
def query_conf():
    return custom_conf.to_dict()


@update_conf_router.get("/conf/options")
def query_conf_options():
    return {
        "translate_api_type": list(TRANSLATE_API_TYPE_OPTIONS),
        "translate_mode": list(TRANSLATE_MODE_OPTIONS),
        "ocr_engine": list(OCR_ENGINE_OPTIONS),
        "auto_save_image": [True, False],
        "enable_ai_linebreak": [True, False],
        "vision_ocr_provider": ["openai", "dashscope"],
        "local_ocr_engine": list(LOCAL_OCR_ENGINE_OPTIONS),
        "inpaint_method": list(INPAINT_METHOD_OPTIONS),
        "use_crf_refine": [True, False],
        "merge_textboxes": [True, False],
    }