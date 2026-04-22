import re
import sys
import types

from app.core.logger import logger


_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
_KATAKANA_RE = re.compile(r"[\u30a0-\u30ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _heuristic_langid_classify(text: str):
    sample = text or ""
    if _HIRAGANA_RE.search(sample) or _KATAKANA_RE.search(sample):
        return ("ja", 0.99)
    if _HANGUL_RE.search(sample):
        return ("ko", 0.99)
    if _CYRILLIC_RE.search(sample):
        return ("ru", 0.95)
    if _CJK_RE.search(sample):
        return ("zh", 0.90)
    if _LATIN_RE.search(sample):
        return ("en", 0.80)
    return ("en", 0.10)


def ensure_py3langid_shim() -> None:
    if "py3langid" in sys.modules:
        return

    try:
        __import__("py3langid")
        return
    except Exception:
        pass

    shim = types.ModuleType("py3langid")
    shim.classify = _heuristic_langid_classify
    sys.modules["py3langid"] = shim
    logger.debug("Injected lightweight py3langid shim for embedded MTU compatibility.")
