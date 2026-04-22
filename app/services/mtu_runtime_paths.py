from functools import lru_cache
from pathlib import Path

from app.core.paths import PROJECT_ROOT


@lru_cache(maxsize=1)
def resolve_mtu_project_dir() -> Path | None:
    """Legacy helper kept for compatibility.

    MTU runtime is now vendored directly into this project under
    ``PROJECT_ROOT / "manga_translator"``. This function keeps the old name so
    existing callers do not need to change.
    """
    candidate = PROJECT_ROOT / "manga_translator"
    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate
    return resolved if resolved.exists() else None


def has_mtu_project() -> bool:
    return resolve_mtu_project_dir() is not None
