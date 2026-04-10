import asyncio
import json
import os
import re
import random
from functools import wraps
from typing import Callable, TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ============ 重试机制配置 ============
# 最大重试次数（默认3次）
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
# 初始重试延迟（秒，默认1秒）
RETRY_INITIAL_DELAY = float(os.getenv("RETRY_INITIAL_DELAY", "1.0"))
# 重试延迟倍数（默认2，指数退避）
RETRY_BACKOFF_MULTIPLIER = float(os.getenv("RETRY_BACKOFF_MULTIPLIER", "2.0"))
# 重试延迟最大值（秒，默认30秒）
RETRY_MAX_DELAY = float(os.getenv("RETRY_MAX_DELAY", "30.0"))
# 重试抖动因子（0-1，默认0.1，避免惊群效应）
RETRY_JITTER = float(os.getenv("RETRY_JITTER", "0.1"))

# ============ AI 断句配置 ============
# 是否启用AI智能断句（默认启用）
AI_LINEBREAK_ENABLED = os.getenv("AI_LINEBREAK_ENABLED", "true").lower() in ("true", "1", "yes")
# AI断句最大文本长度（超过此长度才触发断句，默认50字符）
AI_LINEBREAK_MIN_LENGTH = int(os.getenv("AI_LINEBREAK_MIN_LENGTH", "50"))
# 断句模型（默认复用翻译模型）
AI_LINEBREAK_MODEL = os.getenv("AI_LINEBREAK_MODEL")  # None表示复用翻译模型

# ============ 翻译提示词配置（可通过 .env 自定义）============
# 默认提示词：统一翻译风格，仅输出翻译内容，不附加解释
_DEFAULT_TRANSLATE_PROMPT = "将句子翻译成中文（如果是符号就直接输出，不要加任何解释、注解或括号内容，仅保留自然对话或原声风格的翻译。）"
_DEFAULT_TRANSLATE_STRUCTURED_PROMPT = (
    "你会收到一个 JSON 数组，每项是待翻译句子。"
    "请返回 JSON：{\"result\": [\"翻译1\", \"翻译2\", ...]}。"
    "只输出 JSON，不要输出任何多余文本。"
)

# 从环境变量读取提示词（未配置则使用默认值）
TRANSLATE_SYSTEM_PROMPT = os.getenv("TRANSLATE_SYSTEM_PROMPT", _DEFAULT_TRANSLATE_PROMPT)
TRANSLATE_STRUCTURED_SYSTEM_PROMPT = os.getenv("TRANSLATE_STRUCTURED_PROMPT", _DEFAULT_TRANSLATE_STRUCTURED_PROMPT)

# AI断句提示词
AI_LINEBREAK_PROMPT = os.getenv(
    "AI_LINEBREAK_PROMPT",
    "你是一个专业的漫画排版助手。将输入的翻译文本进行合理断句，使其更适合漫画气泡显示。"
    "规则：1.每行不宜过长，建议10-20个字符；2.在自然的语义停顿处换行；"
    "3.保持语气完整性；4.用换行符(\\n)分隔每行。"
    "只输出断句后的文本，不要解释。"
)


# ============ 重试机制实现 ============
T = TypeVar('T')


def _calculate_retry_delay(attempt: int) -> float:
    """计算重试延迟（指数退避 + 抖动）"""
    delay = RETRY_INITIAL_DELAY * (RETRY_BACKOFF_MULTIPLIER ** (attempt - 1))
    delay = min(delay, RETRY_MAX_DELAY)
    # 添加抖动
    jitter = delay * RETRY_JITTER * random.uniform(-1, 1)
    return max(0, delay + jitter)


def with_retry(
    max_attempts: int = None,
    retryable_exceptions: tuple = None,
):
    """
    异步重试装饰器（指数退避）
    
    Args:
        max_attempts: 最大尝试次数
        retryable_exceptions: 可重试的异常类型元组
    """
    if max_attempts is None:
        max_attempts = RETRY_MAX_ATTEMPTS
    if retryable_exceptions is None:
        # 默认重试所有异常
        retryable_exceptions = (Exception,)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        delay = _calculate_retry_delay(attempt)
                        # 使用模块级logger（如果可用）或print
                        try:
                            from app.core.logger import logger
                            logger.warning(f"翻译请求失败(尝试 {attempt}/{max_attempts})，{delay:.2f}秒后重试: {e}")
                        except ImportError:
                            print(f"[WARN] 翻译请求失败(尝试 {attempt}/{max_attempts})，{delay:.2f}秒后重试: {e}")
                        await asyncio.sleep(delay)
            # 所有重试都失败
            raise last_exception
        return wrapper
    return decorator


# ============ 客户端缓存 ============
_CLIENT_CACHE = {}

def _get_client(api_type: str, api_key: str, base_url: str) -> AsyncOpenAI:
    """获取或创建 OpenAI 客户端（带缓存）"""
    cache_key = f"{api_type}:{api_key}:{base_url}"
    if cache_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[cache_key] = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    return _CLIENT_CACHE[cache_key]


def _provider_options(api_type: str):
    """
    获取翻译 API 配置（从 custom_conf 动态读取）
    
    Args:
        api_type: "dashscope" 或 "openai"
        
    Returns:
        (client, model, extra_kwargs)
    """
    from app.core.custom_conf import custom_conf
    
    if api_type == "dashscope":
        api_key = custom_conf.dashscope_api_key
        base_url = custom_conf.dashscope_base_url
        model = custom_conf.dashscope_model
        if not api_key:
            raise RuntimeError("DashScope API Key 未配置，请通过 /conf/batch-update 设置 dashscope_api_key")
        client = _get_client("dashscope", api_key, base_url)
        return client, model, {"extra_body": {"enable_thinking": False}}
    
    if api_type == "openai":
        api_key = custom_conf.openai_api_key
        base_url = custom_conf.openai_base_url
        model = custom_conf.openai_model
        if not api_key:
            raise RuntimeError("OpenAI API Key 未配置，请通过 /conf/batch-update 设置 openai_api_key")
        client = _get_client("openai", api_key, base_url)
        return client, model, {}
    
    raise RuntimeError(f"不支持的 translate_api_type: {api_type}")


def _normalize_content(content) -> str:
    # 兼容字符串或多段内容结构，统一为纯文本。
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content or "").strip()


def _extract_json_payload(raw: str):
    text = raw.strip()
    candidates = [text]

    # 兼容 ```json ... ``` 包裹。
    if text.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        stripped = re.sub(r"\n?```$", "", stripped)
        candidates.append(stripped.strip())

    # 兼容模型前后夹杂说明文本的情况。
    obj_match = re.search(r"\{[\s\S]*\}", text)
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if obj_match:
        candidates.append(obj_match.group(0))
    if arr_match:
        candidates.append(arr_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise RuntimeError("结构化输出解析失败，模型返回非 JSON")


def _parse_structured_result(raw: str, expected_count: int):
    payload = _extract_json_payload(raw)
    if isinstance(payload, dict):
        result = payload.get("result")
    elif isinstance(payload, list):
        result = payload
    else:
        result = None
    if not isinstance(result, list):
        raise RuntimeError("结构化输出格式错误，缺少 result 列表")
    normalized = [str(item).strip() for item in result]
    if len(normalized) != expected_count:
        raise RuntimeError(
            f"结构化输出数量不匹配，期望 {expected_count} 条，实际 {len(normalized)} 条"
        )
    return normalized


@with_retry()
async def _translate_single(sentence: str, api_type: str):
    """单个句子翻译（带重试）"""
    if not sentence:
        return "", 0.0
    client, model, extra_kwargs = _provider_options(api_type)
    res = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": sentence},
        ],
        **extra_kwargs,
    )
    content = _normalize_content(res.choices[0].message.content)
    return content, 0.0


async def _translate_parallel(all_text, api_type: str):
    # 并行模式：每个句子独立请求，整体用 gather 并发。
    tasks = [_translate_single(sentence, api_type) for sentence in all_text]
    if not tasks:
        return [], 0.0
    res = await asyncio.gather(*tasks)
    res_text, prices = zip(*res)
    return list(res_text), sum(prices)


@with_retry()
async def _translate_structured(all_text, api_type: str):
    """结构化翻译（带重试）：一次性输入列表，要求模型返回翻译列表 JSON"""
    if not all_text:
        return [], 0.0
    client, model, extra_kwargs = _provider_options(api_type)
    payload = json.dumps(all_text, ensure_ascii=False)
    res = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TRANSLATE_STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        **extra_kwargs,
    )
    raw = _normalize_content(res.choices[0].message.content)
    return _parse_structured_result(raw, len(all_text)), 0.0


async def translate_req(all_text, api_type: str = "dashscope", translate_mode: str = "parallel"):
    if translate_mode == "parallel":
        return await _translate_parallel(all_text, api_type)
    if translate_mode == "structured":
        return await _translate_structured(all_text, api_type)
    raise RuntimeError(f"不支持的 translate_mode: {translate_mode}")


# ============ AI 智能断句功能 ============
@with_retry(max_attempts=2)  # 断句失败影响较小，减少重试次数
async def _ai_linebreak_single(text: str, api_type: str) -> str:
    """
    对单个翻译结果进行AI智能断句
    
    Args:
        text: 待断句的翻译文本
        api_type: API类型
        
    Returns:
        断句后的文本
    """
    if not text or len(text) < AI_LINEBREAK_MIN_LENGTH:
        return text
    
    client, model, extra_kwargs = _provider_options(api_type)
    
    # 如果配置了专用断句模型，使用它
    if AI_LINEBREAK_MODEL:
        model = AI_LINEBREAK_MODEL
    
    res = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": AI_LINEBREAK_PROMPT},
            {"role": "user", "content": text},
        ],
        **extra_kwargs,
    )
    
    result = _normalize_content(res.choices[0].message.content)
    return result if result else text


async def ai_linebreak_batch(texts: list[str], api_type: str = "dashscope") -> list[str]:
    """
    批量AI智能断句
    
    Args:
        texts: 翻译文本列表
        api_type: API类型
        
    Returns:
        断句后的文本列表
    """
    if not AI_LINEBREAK_ENABLED:
        return texts
    
    results = []
    for text in texts:
        try:
            result = await _ai_linebreak_single(text, api_type)
            results.append(result)
        except Exception as e:
            # 断句失败不影响主流程，返回原文
            try:
                from app.core.logger import logger
                logger.warning(f"AI断句失败，使用原文: {e}")
            except ImportError:
                pass
            results.append(text)
    
    return results


async def translate_with_linebreak(
    all_text: list[str], 
    api_type: str = "dashscope", 
    translate_mode: str = "parallel",
    enable_linebreak: bool = True
) -> tuple[list[str], float]:
    """
    带AI断句的翻译（主入口）
    
    Args:
        all_text: 待翻译文本列表
        api_type: API类型 (dashscope/openai)
        translate_mode: 翻译模式 (parallel/structured)
        enable_linebreak: 是否启用AI断句
        
    Returns:
        (翻译结果列表, 费用)
    """
    # 1. 执行翻译
    translated, cost = await translate_req(all_text, api_type, translate_mode)
    
    # 2. AI断句（如果启用）
    if enable_linebreak and AI_LINEBREAK_ENABLED:
        translated = await ai_linebreak_batch(translated, api_type)
    
    return translated, cost