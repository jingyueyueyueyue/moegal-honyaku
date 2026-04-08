import asyncio
import base64
import json
import random
import time
from typing import Literal

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from app.core.custom_conf import custom_conf
from app.core.logger import logger
from app.services.web_image_input import (
    TranslateWebInputError,
    decode_image_base64_data_url,
    ensure_body_size_within_limit,
)

manga_translate_router = APIRouter()

DOWNLOAD_RETRY_COUNT = 2
DOWNLOAD_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)


def _now_ms():
    """返回当前时间的毫秒时间戳"""
    return time.time() * 1000


def _elapsed_ms(start_ms: float) -> float:
    """计算从 start_ms 到现在的耗时（毫秒）"""
    return round(_now_ms() - start_ms, 2)


def _decode_image(file_bytes: bytes):
    if not file_bytes:
        raise TranslateWebInputError(400, "图片为空")
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr_cv is None:
        raise TranslateWebInputError(400, "图片解码失败，请确认输入为有效图片")
    return img_bgr_cv, Image.fromarray(img_bgr_cv)


async def _download_image_bytes(image_url: str, referer: str) -> bytes:
    headers = {
        "Referer": referer,
        "User-Agent": "Mozilla/5.0",
    }
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        last_error: Exception | None = None
        for attempt in range(DOWNLOAD_RETRY_COUNT + 1):
            try:
                response = await client.get(image_url, headers=headers)
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if 400 <= status_code < 500 and status_code != 429:
                    raise RuntimeError(f"图片下载失败，状态码: {status_code}") from exc
                last_error = RuntimeError(f"图片下载失败，状态码: {status_code}")
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as exc:
                last_error = exc
            if attempt < DOWNLOAD_RETRY_COUNT:
                await asyncio.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"图片下载失败：{last_error}")


async def _translate_image_bytes(
    file_bytes: bytes,
    include_res_img: bool,
    background_tasks: BackgroundTasks,
    text_direction: Literal["horizontal", "vertical"] = "horizontal",
    enable_linebreak: bool | None = None,
):
    """
    翻译图片的核心处理函数
    
    Args:
        file_bytes: 图片字节数据
        include_res_img: 是否在响应中包含结果图片的 base64
        background_tasks: FastAPI 后台任务
        text_direction: 文字方向 (horizontal/vertical)
        enable_linebreak: 是否启用 AI 智能断句
            - None: 使用全局配置 (custom_conf.enable_ai_linebreak)
            - True: 强制启用
            - False: 强制禁用
    """
    from app.services.ocr import detect_text_regions
    from app.services.pic_process import draw_text_on_boxes, get_text_masked_pic, save_img
    from app.services.translate_api import translate_with_linebreak

    timings = {}  # 记录各阶段耗时
    
    # 图片解码
    t0 = _now_ms()
    img_bgr_cv, img_pil = _decode_image(file_bytes)
    timings["decode"] = _elapsed_ms(t0)
    
    # 文本检测
    t1 = _now_ms()
    bboxes = detect_text_regions(img_bgr_cv)
    timings["detect"] = _elapsed_ms(t1)
    
    # 获取文本遮罩
    t2 = _now_ms()
    all_text, inpaint = await get_text_masked_pic(
        img_pil, img_bgr_cv, bboxes, True, custom_conf.ocr_engine
    )
    timings["mask_ocr"] = _elapsed_ms(t2)
    
    if len(all_text) == 0:
        logger.warning("未检测出文字")
        return None, None, None, None, timings

    # 确定是否启用 AI 断句（请求参数优先，否则使用全局配置）
    use_linebreak = enable_linebreak if enable_linebreak is not None else custom_conf.enable_ai_linebreak

    # 翻译（带AI断句）
    t3 = _now_ms()
    cn_text, price = await translate_with_linebreak(
        all_text,
        api_type=custom_conf.translate_api_type,
        translate_mode=custom_conf.translate_mode,
        enable_linebreak=use_linebreak,
    )
    timings["translate"] = _elapsed_ms(t3)

    # 绘制文字
    t4 = _now_ms()
    img_res = draw_text_on_boxes(inpaint, bboxes, cn_text, text_direction)
    timings["draw"] = _elapsed_ms(t4)
    
    # 编码结果图片
    t5 = _now_ms()
    ok, buffer = cv2.imencode(".jpg", img_res, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("结果图片编码失败")
    timings["encode"] = _elapsed_ms(t5)

    cn_file_bytes = buffer.tobytes()
    file_name = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}.jpg"
    
    # 根据配置决定是否保存图片
    if custom_conf.auto_save_image:
        background_tasks.add_task(save_img, cn_file_bytes, "cn", file_name)
        background_tasks.add_task(save_img, file_bytes, "raw", file_name)
    
    b64_img = base64.b64encode(cn_file_bytes).decode("utf8") if include_res_img else None
    return all_text, cn_text, price, (b64_img, file_name), timings


def _error_response(info: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        content={
            "status": "error",
            "info": info,
        },
        status_code=status_code,
    )


def _validation_error_message(exc: ValidationError) -> str:
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    if not errors:
        return "请求参数不合法"
    message = str(errors[0].get("msg", "请求参数不合法")).strip()
    if message.startswith("Value error, "):
        message = message.removeprefix("Value error, ").strip()
    return message or "请求参数不合法"


@manga_translate_router.post("/api/v1/translate/upload")
async def translate_upload(
    background_tasks: BackgroundTasks,
    img: UploadFile = File(...),
    include_res_img: bool = True,
    text_direction: Literal["horizontal", "vertical"] = "horizontal",
    enable_linebreak: bool | None = None,
):
    """
    上传图片翻译接口
    
    Args:
        img: 上传的图片文件
        include_res_img: 是否在响应中包含结果图片的 base64（默认 true）
        text_direction: 文字方向，horizontal（横排）或 vertical（竖排）
        enable_linebreak: 是否启用 AI 智能断句
            - 不传：使用全局配置
            - true：强制启用
            - false：强制禁用
    """
    start = time.time()
    try:
        file_bytes = await img.read()
        all_text, cn_text, price, img_result, timings = await _translate_image_bytes(
            file_bytes=file_bytes,
            include_res_img=include_res_img,
            background_tasks=background_tasks,
            text_direction=text_direction,
            enable_linebreak=enable_linebreak,
        )
        if all_text is None:
            return JSONResponse(content={
                "status": "error",
                "info": "未检测出文字",
            })
    except Exception as e:
        logger.error(f"翻译失败：{e}")
        return JSONResponse(content={
            "status": "error",
            "info": f"{e}",
        })
    b64_img, file_name = img_result
    duration = round(time.time() - start, 3)
    logger.info(f"翻译成功 [{file_name}] 总耗时 {duration}s | 各阶段(ms): decode={timings['decode']}, detect={timings['detect']}, mask_ocr={timings['mask_ocr']}, translate={timings['translate']}, draw={timings['draw']}, encode={timings['encode']}")
    return JSONResponse(content={
        "status": "success",
        "duration": duration,
        "price": round(price, 8),
        "cn_text": cn_text,
        "raw_text": all_text,
        "res_img": b64_img,
    })


class TranslateWebRequest(BaseModel):
    image_url: str | None = None
    image_base64: str | None = None
    referer: str
    source_type: Literal["img", "canvas"] | None = None
    include_res_img: bool = True
    text_direction: Literal["horizontal", "vertical"] = "horizontal"
    # AI 智能断句：None=使用全局配置，true=启用，false=禁用
    enable_linebreak: bool | None = None

    @field_validator("image_url", "image_base64", mode="before")
    @classmethod
    def _normalize_image_source(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("source_type", mode="before")
    @classmethod
    def _normalize_source_type(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("text_direction", mode="before")
    @classmethod
    def _normalize_text_direction(cls, value):
        if value is None:
            return "horizontal"
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in ("horizontal", "vertical"):
                return stripped
            return "horizontal"
        return "horizontal"

    @field_validator("enable_linebreak", mode="before")
    @classmethod
    def _normalize_enable_linebreak(cls, value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes"):
                return True
            if lowered in ("false", "0", "no"):
                return False
        return None

    @model_validator(mode="after")
    def validate_image_source(self):
        has_url = bool(self.image_url)
        has_base64 = bool(self.image_base64)
        if not has_url and not has_base64:
            raise ValueError("image_url 和 image_base64 不能同时为空")
        if has_url and has_base64:
            raise ValueError("image_url 和 image_base64 不能同时存在")
        return self


@manga_translate_router.post("/api/v1/translate/web")
async def translate_web(request: Request, background_tasks: BackgroundTasks):
    """
    Web 端翻译接口（JSON 请求体）
    
    请求体示例：
    {
        "image_url": "https://example.com/manga.jpg",
        "referer": "https://example.com",
        "include_res_img": true,
        "text_direction": "horizontal",
        "enable_linebreak": true
    }
    
    参数说明：
    - image_url / image_base64: 图片来源（二选一）
    - referer: 来源页面 URL（用于下载图片时携带）
    - include_res_img: 是否返回结果图片的 base64
    - text_direction: 文字方向 (horizontal/vertical)
    - enable_linebreak: AI 智能断句
        - 不传或传 null：使用全局配置
        - true：强制启用
        - false：强制禁用
    """
    start = time.time()
    timings = {}
    
    try:
        # 请求解析
        t_parse = _now_ms()
        ensure_body_size_within_limit(content_length=request.headers.get("content-length"))
        body = await request.body()
        ensure_body_size_within_limit(actual_size=len(body))
        if not body:
            raise TranslateWebInputError(400, "请求体不能为空")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise TranslateWebInputError(400, "请求体必须是 JSON 对象")
        timings["parse"] = _elapsed_ms(t_parse)

        req = TranslateWebRequest.model_validate(payload)
        
        # 图片获取
        t_fetch = _now_ms()
        if req.image_url is not None:
            file_bytes = await _download_image_bytes(req.image_url, req.referer)
        else:
            file_bytes = decode_image_base64_data_url(req.image_base64)
        timings["fetch"] = _elapsed_ms(t_fetch)
        
        all_text, cn_text, price, img_result, proc_timings = await _translate_image_bytes(
            file_bytes=file_bytes,
            include_res_img=req.include_res_img,
            background_tasks=background_tasks,
            text_direction=req.text_direction,
            enable_linebreak=req.enable_linebreak,
        )
        timings.update(proc_timings)
        
        if all_text is None:
            return JSONResponse(content={
                "status": "error",
                "info": "未检测出文字",
            })
        b64_img, file_name = img_result
        duration = round(time.time() - start, 3)
        logger.info(f"翻译成功 [{file_name}] 总耗时 {duration}s | 各阶段(ms): parse={timings.get('parse', 0)}, fetch={timings.get('fetch', 0)}, decode={timings['decode']}, detect={timings['detect']}, mask_ocr={timings['mask_ocr']}, translate={timings['translate']}, draw={timings['draw']}, encode={timings['encode']}")
        return JSONResponse(content={
            "status": "success",
            "duration": duration,
            "price": round(price, 8),
            "cn_text": cn_text,
            "raw_text": all_text,
            "res_img": b64_img,
        })
    except json.JSONDecodeError:
        return _error_response("请求体不是合法 JSON", 400)
    except ValidationError as exc:
        return _error_response(_validation_error_message(exc), 400)
    except TranslateWebInputError as exc:
        return _error_response(exc.message, exc.status_code)
    except Exception as e:
        logger.error(f"翻译失败：{e}")
        return _error_response(str(e), 500)