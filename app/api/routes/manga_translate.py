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


def _format_timing_summary(timings: dict[str, float]) -> str:
    ordered_keys = (
        "parse",
        "fetch",
        "decode",
        "detect_mask_ocr",
        "translate",
        "draw",
        "encode",
    )
    parts = []
    for key in ordered_keys:
        value = timings.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value}ms")
    return " | ".join(parts)


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
    from app.services.pic_process import draw_text_on_boxes_mit, get_text_masked_pic_auto, save_img
    from app.services.translate_api import translate_with_linebreak

    timings = {}  # 记录各阶段耗时
    
    # 图片解码
    t0 = _now_ms()
    img_bgr_cv, img_pil = _decode_image(file_bytes)
    timings["decode"] = _elapsed_ms(t0)
    
    # 文本检测 + OCR + Inpainting（使用新的 MIT 高级流程）
    t1 = _now_ms()
    all_text, inpaint, bboxes, raw_mask = await get_text_masked_pic_auto(
        img_pil,
        img_bgr_cv,
        inpaint=True,
        ocr_engine=custom_conf.ocr_engine,
        text_direction=text_direction,
    )
    timings["detect_mask_ocr"] = _elapsed_ms(t1)
    
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

    # 绘制文字（使用 MIT 高质量渲染）
    t4 = _now_ms()
    img_res = draw_text_on_boxes_mit(inpaint, bboxes, cn_text, text_direction, original_image=img_bgr_cv, raw_mask=raw_mask)
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
    timing_summary = _format_timing_summary(timings)
    logger.info(
        f"翻译成功 [{file_name}] total={duration}s | text={len(all_text)} | {timing_summary}"
    )
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
        timing_summary = _format_timing_summary(timings)
        logger.info(
            f"翻译成功 [{file_name}] total={duration}s | text={len(all_text)} | {timing_summary}"
        )
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


# ============ 批量翻译接口（context-batch 模式）============

class BatchTranslateItem(BaseModel):
    """单个图片项"""
    image_url: str | None = None
    image_base64: str | None = None
    
    @field_validator("image_url", "image_base64", mode="before")
    @classmethod
    def _normalize_image_source(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value
    
    @model_validator(mode="after")
    def validate_image_source(self):
        has_url = bool(self.image_url)
        has_base64 = bool(self.image_base64)
        if not has_url and not has_base64:
            raise ValueError("image_url 和 image_base64 不能同时为空")
        if has_url and has_base64:
            raise ValueError("image_url 和 image_base64 不能同时存在")
        return self


class BatchTranslateRequest(BaseModel):
    """批量翻译请求"""
    images: list[BatchTranslateItem]
    referer: str
    text_direction: Literal["horizontal", "vertical"] = "horizontal"
    enable_linebreak: bool | None = None


@manga_translate_router.post("/api/v1/translate/batch")
async def translate_batch(request: Request, background_tasks: BackgroundTasks):
    """
    批量翻译接口（context-batch 模式）
    
    一次性翻译多张图片，保持跨图片的上下文连贯性。
    
    请求体示例：
    {
        "images": [
            {"image_url": "https://example.com/page1.jpg"},
            {"image_url": "https://example.com/page2.jpg"}
        ],
        "referer": "https://example.com",
        "text_direction": "horizontal",
        "enable_linebreak": true
    }
    
    返回格式：
    {
        "status": "success",
        "results": [
            {"raw_text": [...], "cn_text": [...], "res_img": "base64..."},
            ...
        ]
    }
    """
    from app.services.pic_process import draw_text_on_boxes_mit, get_text_masked_pic_auto, save_img
    from app.services.translate_api import translate_batch_with_context, ai_linebreak_batch
    
    start = time.time()
    
    try:
        # 请求解析
        ensure_body_size_within_limit(content_length=request.headers.get("content-length"))
        body = await request.body()
        ensure_body_size_within_limit(actual_size=len(body))
        if not body:
            raise TranslateWebInputError(400, "请求体不能为空")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise TranslateWebInputError(400, "请求体必须是 JSON 对象")
        
        req = BatchTranslateRequest.model_validate(payload)
        
        if not req.images:
            return JSONResponse(content={
                "status": "error",
                "info": "图片列表不能为空",
            })
        
        # 1. 获取所有图片并检测文本
        pages_text = []  # 每页的文本列表
        pages_data = []  # 每页的图片数据 (inpaint_img, pil_img)
        
        for item in req.images:
            # 获取图片
            if item.image_url:
                file_bytes = await _download_image_bytes(item.image_url, req.referer)
            else:
                file_bytes = decode_image_base64_data_url(item.image_base64)
            
            img_bgr_cv, img_pil = _decode_image(file_bytes)
            
            # 使用 MIT 高级流程：检测 + OCR + Inpainting
            all_text, inpaint, bboxes, raw_mask = await get_text_masked_pic_auto(
                img_pil,
                img_bgr_cv,
                inpaint=True,
                ocr_engine=custom_conf.ocr_engine,
                text_direction=req.text_direction,
            )
            
            pages_text.append(all_text)
            pages_data.append({
                "bboxes": bboxes,
                "inpaint": inpaint,
                "pil": img_pil,
                "cv": img_bgr_cv,
                "raw_mask": raw_mask,
            })
        
        # 2. 批量翻译（跨图片上下文）
        non_empty_indices = [i for i, texts in enumerate(pages_text) if texts]
        
        if not non_empty_indices:
            return JSONResponse(content={
                "status": "success",
                "results": [{"raw_text": [], "cn_text": [], "res_img": None} for _ in req.images],
                "duration": round(time.time() - start, 3),
            })
        
        # 调用批量翻译 API
        pages_translations, price = await translate_batch_with_context(
            [pages_text[i] for i in non_empty_indices],
            api_type=custom_conf.translate_api_type
        )
        
        # AI 断句
        use_linebreak = req.enable_linebreak if req.enable_linebreak is not None else custom_conf.enable_ai_linebreak
        if use_linebreak:
            pages_translations = await asyncio.gather(*[
                ai_linebreak_batch(translations, custom_conf.translate_api_type)
                for translations in pages_translations
            ])
        
        # 3. 绘制文字并编码
        results = []
        for orig_idx in non_empty_indices:
            translations = pages_translations[non_empty_indices.index(orig_idx)]
            page_data = pages_data[orig_idx]
            
            # 绘制文字（使用 MIT 高质量渲染）
            img_res = draw_text_on_boxes_mit(
                page_data["inpaint"],
                page_data["bboxes"],
                translations,
                req.text_direction,
                original_image=page_data["cv"],
                raw_mask=page_data.get("raw_mask"),
            )
            
            # 编码
            ok, buffer = cv2.imencode(".jpg", img_res, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("结果图片编码失败")
            
            cn_file_bytes = buffer.tobytes()
            b64_img = base64.b64encode(cn_file_bytes).decode("utf8")
            
            # 保存
            file_name = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}.jpg"
            if custom_conf.auto_save_image:
                background_tasks.add_task(save_img, cn_file_bytes, "cn", file_name)
            
            results.append({
                "index": orig_idx,
                "raw_text": pages_text[orig_idx],
                "cn_text": translations,
                "res_img": b64_img,
            })
        
        # 填充空结果
        all_results = []
        result_map = {r["index"]: r for r in results}
        for i in range(len(req.images)):
            if i in result_map:
                r = result_map[i]
                all_results.append({
                    "raw_text": r["raw_text"],
                    "cn_text": r["cn_text"],
                    "res_img": r["res_img"],
                })
            else:
                all_results.append({
                    "raw_text": [],
                    "cn_text": [],
                    "res_img": None,
                })
        
        duration = round(time.time() - start, 3)
        timing_summary = _format_timing_summary(timings)
        logger.info(
            f"批量翻译成功 pages={len(non_empty_indices)} | total={duration}s"
            + (f" | {timing_summary}" if timing_summary else "")
        )
        
        return JSONResponse(content={
            "status": "success",
            "duration": duration,
            "price": round(price, 8),
            "results": all_results,
        })
        
    except json.JSONDecodeError:
        return _error_response("请求体不是合法 JSON", 400)
    except ValidationError as exc:
        return _error_response(_validation_error_message(exc), 400)
    except TranslateWebInputError as exc:
        return _error_response(exc.message, exc.status_code)
    except Exception as e:
        logger.error(f"批量翻译失败：{e}")
        return _error_response(str(e), 500)


# ============ 顺序翻译接口（context-sequential 模式）============

class SequentialTranslateRequest(BaseModel):
    """顺序翻译请求"""
    image_url: str | None = None
    image_base64: str | None = None
    previous_translations: list[str] = []  # 之前的翻译结果（上下文）
    referer: str
    text_direction: Literal["horizontal", "vertical"] = "horizontal"
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
    
    @model_validator(mode="after")
    def validate_image_source(self):
        has_url = bool(self.image_url)
        has_base64 = bool(self.image_base64)
        if not has_url and not has_base64:
            raise ValueError("image_url 和 image_base64 不能同时为空")
        if has_url and has_base64:
            raise ValueError("image_url 和 image_base64 不能同时存在")
        return self


@manga_translate_router.post("/api/v1/translate/sequential")
async def translate_sequential(request: Request, background_tasks: BackgroundTasks):
    """
    顺序翻译接口（context-sequential 模式）
    
    翻译当前图片时参考之前的翻译结果，保持上下文连贯性。
    
    请求体示例：
    {
        "image_url": "https://example.com/page1.jpg",
        "referer": "https://example.com",
        "previous_translations": ["之前的翻译1", "之前的翻译2"],
        "text_direction": "horizontal",
        "enable_linebreak": true
    }
    """
    from app.services.pic_process import draw_text_on_boxes_mit, get_text_masked_pic_auto, save_img
    from app.services.translate_api import translate_with_previous_context, ai_linebreak_batch
    
    start = time.time()
    timings = {}
    
    try:
        # 请求解析
        ensure_body_size_within_limit(content_length=request.headers.get("content-length"))
        body = await request.body()
        ensure_body_size_within_limit(actual_size=len(body))
        if not body:
            raise TranslateWebInputError(400, "请求体不能为空")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise TranslateWebInputError(400, "请求体必须是 JSON 对象")
        timings["parse"] = _elapsed_ms(_now_ms())
        
        req = SequentialTranslateRequest.model_validate(payload)
        
        # 图片获取
        t_fetch = _now_ms()
        if req.image_url is not None:
            file_bytes = await _download_image_bytes(req.image_url, req.referer)
        else:
            file_bytes = decode_image_base64_data_url(req.image_base64)
        timings["fetch"] = _elapsed_ms(t_fetch)
        
        # 图片解码
        t_decode = _now_ms()
        img_bgr_cv, img_pil = _decode_image(file_bytes)
        timings["decode"] = _elapsed_ms(t_decode)
        
        # 使用 MIT 高级流程：检测 + OCR + Inpainting
        t_detect = _now_ms()
        all_text, inpaint, bboxes, raw_mask = await get_text_masked_pic_auto(
            img_pil,
            img_bgr_cv,
            inpaint=True,
            ocr_engine=custom_conf.ocr_engine,
            text_direction=req.text_direction,
        )
        timings["detect_mask_ocr"] = _elapsed_ms(t_detect)
        
        if len(all_text) == 0:
            return JSONResponse(content={
                "status": "error",
                "info": "未检测出文字",
            })
        
        # 累积上下文翻译
        use_linebreak = req.enable_linebreak if req.enable_linebreak is not None else custom_conf.enable_ai_linebreak
        
        t_translate = _now_ms()
        cn_text, price = await translate_with_previous_context(
            all_text,
            req.previous_translations,
            api_type=custom_conf.translate_api_type
        )
        
        # AI 断句
        if use_linebreak:
            cn_text = await ai_linebreak_batch(cn_text, custom_conf.translate_api_type)
        timings["translate"] = _elapsed_ms(t_translate)
        
        # 绘制文字（使用 MIT 高质量渲染）
        t_draw = _now_ms()
        img_res = draw_text_on_boxes_mit(inpaint, bboxes, cn_text, req.text_direction, original_image=img_bgr_cv, raw_mask=raw_mask)
        timings["draw"] = _elapsed_ms(t_draw)
        
        # 编码
        t_encode = _now_ms()
        ok, buffer = cv2.imencode(".jpg", img_res, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("结果图片编码失败")
        timings["encode"] = _elapsed_ms(t_encode)
        
        cn_file_bytes = buffer.tobytes()
        file_name = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}.jpg"
        
        if custom_conf.auto_save_image:
            background_tasks.add_task(save_img, cn_file_bytes, "cn", file_name)
        
        b64_img = base64.b64encode(cn_file_bytes).decode("utf8")
        duration = round(time.time() - start, 3)
        
        timing_summary = _format_timing_summary(timings)
        logger.info(
            f"顺序翻译成功 [{file_name}] total={duration}s | text={len(all_text)} | {timing_summary}"
        )
        
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
        logger.error(f"顺序翻译失败：{e}")
        return _error_response(str(e), 500)
