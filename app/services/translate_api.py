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

# 上下文感知翻译提示词
_CONTEXT_TRANSLATE_SYSTEM_PROMPT = os.getenv(
    "CONTEXT_TRANSLATE_SYSTEM_PROMPT",
    "你是一个专业的漫画翻译助手。你会收到一个JSON数组，包含漫画一页中所有气泡的文本。"
    "请进行翻译，注意：\n"
    "1. 结合上下文理解代词指代（他/她/它）和省略句\n"
    "2. 保持角色称呼和专有名词的一致性\n"
    "3. 理解对话语境，保持语气和情感连贯\n"
    "4. 每个气泡独立翻译，但考虑整体叙事流畅\n"
    "返回格式：{\"result\": [\"翻译1\", \"翻译2\", ...]}\n"
    "只输出JSON，不要解释。"
)

# 累积上下文翻译提示词（用于 context-sequential 模式）
_SEQUENTIAL_TRANSLATE_SYSTEM_PROMPT = os.getenv(
    "SEQUENTIAL_TRANSLATE_SYSTEM_PROMPT",
    "你是一个专业的漫画翻译助手。你会收到之前已翻译的内容和新待翻译的内容。"
    "请根据上下文连贯性进行翻译，注意：\n"
    "1. 参考已翻译内容理解代词、角色称呼和术语\n"
    "2. 保持与之前翻译的风格和术语一致性\n"
    "3. 理解对话的连续性和语境\n"
    "输入格式：{\"previous\": [\"已翻译1\", ...], \"current\": [\"待翻译1\", ...]}\n"
    "返回格式：{\"result\": [\"翻译1\", \"翻译2\", ...]}\n"
    "只输出JSON，不要解释。"
)

# 批量上下文翻译提示词（用于 context-batch 模式）
_BATCH_TRANSLATE_SYSTEM_PROMPT = os.getenv(
    "BATCH_TRANSLATE_SYSTEM_PROMPT",
    "你是漫画翻译助手。输入是JSON格式的多页文本：{\"pages\": [[...], [...]]}。\n"
    "翻译规则：保持角色称呼和术语一致，理解上下文。\n"
    "输出格式示例（假设2页，每页2个气泡）：\n"
    "{\"result\": [[\"第一页第一个\", \"第一页第二个\"], [\"第二页第一个\", \"第二页第二个\"]]}\n"
    "严格输出JSON，不要任何额外文字。"
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
        except json.JSONDecodeError:
            # 尝试修复不完整的 JSON（模型输出被截断）
            fixed = _try_fix_incomplete_json(candidate)
            if fixed:
                try:
                    return json.loads(fixed)
                except:
                    pass
        except Exception:
            continue
    # 提取失败，显示原始内容前300字符帮助调试
    preview = text[:300] + ("..." if len(text) > 300 else "")
    raise RuntimeError(f"结构化输出解析失败，模型返回非 JSON。原始内容: {preview}")


def _try_fix_incomplete_json(text: str) -> str | None:
    """尝试修复被截断的 JSON"""
    text = text.strip()
    if not text:
        return None
    
    # 统计括号数量
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    if open_braces <= 0 and open_brackets <= 0:
        return text  # 括号已匹配，不需要修复
    
    # 查找最后一个完整的字符串（闭合引号）
    last_complete = -1
    quote_positions = []
    in_escape = False
    for i, c in enumerate(text):
        if c == '\\' and not in_escape:
            in_escape = True
            continue
        if c == '"' and not in_escape:
            quote_positions.append(i)
        in_escape = False
    
    if len(quote_positions) >= 2 and len(quote_positions) % 2 == 0:
        last_complete = quote_positions[-1]
    
    if last_complete > 0:
        truncated = text[:last_complete + 1]
        # 补全闭合括号
        open_braces = truncated.count('{') - truncated.count('}')
        open_brackets = truncated.count('[') - truncated.count(']')
        truncated += ']' * open_brackets + '}' * open_braces
        return truncated
    
    # 简单地补全括号
    return text + ']' * open_brackets + '}' * open_braces


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


@with_retry()
async def _translate_context(all_text, api_type: str):
    """
    上下文感知翻译（带重试）
    
    特点：
    1. 将所有文本作为整体发送，模型可以理解上下文关系
    2. 利用上下文理解代词指代、省略句、语气等
    3. 保持角色称呼和术语的一致性
    """
    if not all_text:
        return [], 0.0
    
    client, model, extra_kwargs = _provider_options(api_type)
    
    # 构建上下文输入
    # 格式：带索引的列表，方便模型理解顺序关系
    context_input = json.dumps(all_text, ensure_ascii=False)
    
    res = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _CONTEXT_TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": context_input},
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
    if translate_mode == "context":
        return await _translate_context(all_text, api_type)
    # context-batch 和 context-sequential 需要特殊处理，这里作为 fallback
    if translate_mode in ("context-batch", "context-sequential"):
        return await _translate_context(all_text, api_type)
    raise RuntimeError(f"不支持的 translate_mode: {translate_mode}")


# ============ 累积上下文翻译（context-sequential）============
@with_retry()
async def translate_with_previous_context(
    current_text: list[str],
    previous_translations: list[str],
    api_type: str = "dashscope"
) -> tuple[list[str], float]:
    """
    累积上下文翻译：翻译当前文本时参考之前的翻译结果
    
    Args:
        current_text: 当前待翻译文本列表
        previous_translations: 之前的翻译结果（作为上下文）
        api_type: API类型
        
    Returns:
        (翻译结果列表, 费用)
    """
    if not current_text:
        return [], 0.0
    
    client, model, extra_kwargs = _provider_options(api_type)
    
    # 构建输入
    payload = json.dumps({
        "previous": previous_translations[-20:],  # 只保留最近20条，避免上下文过长
        "current": current_text
    }, ensure_ascii=False)
    
    res = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SEQUENTIAL_TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        **extra_kwargs,
    )
    
    raw = _normalize_content(res.choices[0].message.content)
    return _parse_structured_result(raw, len(current_text)), 0.0


# ============ 批量上下文翻译（context-batch）============
def _parse_batch_result(raw: str, expected_counts: list[int], original_pages: list[list[str]]) -> list[list[str]]:
    """解析批量翻译结果，允许部分结果"""
    payload = _extract_json_payload(raw)
    
    if isinstance(payload, dict):
        result = payload.get("result")
    elif isinstance(payload, list):
        result = payload
    else:
        raise RuntimeError("批量翻译输出格式错误")
    
    if not isinstance(result, list):
        raise RuntimeError("批量翻译输出缺少 result 列表")
    
    normalized = []
    
    # 如果页数不足，补齐
    while len(result) < len(expected_counts):
        result.append([])
    
    for i, (page_result, expected_count, original_texts) in enumerate(zip(result, expected_counts, original_pages)):
        if not isinstance(page_result, list):
            page_result = []
        
        page_translations = [str(item).strip() for item in page_result]
        
        # 如果翻译数量不足，用原文填充
        while len(page_translations) < expected_count:
            page_translations.append(original_texts[len(page_translations)])
        
        # 如果翻译数量过多，截断
        page_translations = page_translations[:expected_count]
        
        normalized.append(page_translations)
    
    return normalized


@with_retry()
async def translate_batch_with_context(
    pages_text: list[list[str]],
    api_type: str = "dashscope"
) -> tuple[list[list[str]], float]:
    """
    批量上下文翻译：一次性翻译多页漫画的所有文本
    
    Args:
        pages_text: 每页的文本列表，例如 [[页1气泡1, 页1气泡2], [页2气泡1, ...]]
        api_type: API类型
        
    Returns:
        (每页翻译结果列表, 费用)
    """
    if not pages_text:
        return [], 0.0
    
    # 过滤空页
    non_empty_pages = [(i, texts) for i, texts in enumerate(pages_text) if texts]
    if not non_empty_pages:
        return [[] for _ in pages_text], 0.0
    
    client, model, extra_kwargs = _provider_options(api_type)
    
    # 构建输入
    payload = json.dumps({
        "pages": [texts for _, texts in non_empty_pages]
    }, ensure_ascii=False)
    
    res = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _BATCH_TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        **extra_kwargs,
    )
    
    raw = _normalize_content(res.choices[0].message.content)
    expected_counts = [len(texts) for _, texts in non_empty_pages]
    original_pages = [texts for _, texts in non_empty_pages]
    non_empty_results = _parse_batch_result(raw, expected_counts, original_pages)
    
    # 重建完整结果（包含空页）
    results = [[] for _ in pages_text]
    for (orig_idx, _), translations in zip(non_empty_pages, non_empty_results):
        results[orig_idx] = translations
    
    return results, 0.0


# ============ AI 智能断句功能 ============
@with_retry(max_attempts=2)  # 断句失败影响较小，减少重试次数
async def _ai_linebreak_single(text: str, api_type: str, min_length: int) -> str:
    """
    对单个翻译结果进行AI智能断句
    
    Args:
        text: 待断句的翻译文本
        api_type: API类型
        min_length: 最小文本长度，短于此长度不处理
        
    Returns:
        断句后的文本
    """
    if not text or len(text) < min_length:
        return text
    
    # 复用翻译模型进行断句
    client, model, extra_kwargs = _provider_options(api_type)
    
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
    from app.core.custom_conf import custom_conf
    
    # 从 custom_conf 读取配置
    if not custom_conf.enable_ai_linebreak:
        return texts
    
    min_length = custom_conf.ai_linebreak_min_length
    
    results = []
    for text in texts:
        try:
            result = await _ai_linebreak_single(text, api_type, min_length)
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
    from app.core.custom_conf import custom_conf
    
    # 1. 执行翻译
    translated, cost = await translate_req(all_text, api_type, translate_mode)
    
    # 2. AI断句（如果启用且全局配置开启）
    if enable_linebreak and custom_conf.enable_ai_linebreak:
        translated = await ai_linebreak_batch(translated, api_type)
    
    return translated, cost


async def translate_req(all_text, api_type: str = "dashscope", translate_mode: str = "parallel"):
    if translate_mode == "parallel":
        return await _translate_parallel(all_text, api_type)

    try:
        if translate_mode == "structured":
            return await _translate_structured(all_text, api_type)
        if translate_mode == "context":
            return await _translate_context(all_text, api_type)
        if translate_mode in ("context-batch", "context-sequential"):
            return await _translate_context(all_text, api_type)
    except RuntimeError as exc:
        from app.core.logger import logger

        logger.warning(
            "Structured translation failed, falling back to parallel mode: mode=%s api_type=%s error=%s",
            translate_mode,
            api_type,
            exc,
        )
        return await _translate_parallel(all_text, api_type)

    raise RuntimeError(f"涓嶆敮鎸佺殑 translate_mode: {translate_mode}")
