import os
from dataclasses import dataclass
from typing import Optional

from .utils.openai_compat import resolve_openai_compatible_api_key


@dataclass(frozen=True)
class RuntimeAPIConfig:
    api_key: Optional[str]
    base_url: str
    model_name: str


def get_runtime_api_override(config, feature: str, provider: str) -> dict[str, str]:
    if config is None:
        return {}
    overrides = getattr(config, "_runtime_api_overrides", None) or {}
    return dict(overrides.get(feature, {}).get(provider, {}))


def resolve_runtime_api_config(
    config,
    *,
    feature: str,
    provider: str,
    api_key_env: Optional[str],
    api_base_env: Optional[str],
    model_env: Optional[str],
    fallback_api_key_env: Optional[str],
    fallback_api_base_env: Optional[str],
    fallback_model_env: Optional[str],
    default_api_base: str,
    default_model: str,
    allow_empty_local_api_key: bool = False,
) -> RuntimeAPIConfig:
    override = get_runtime_api_override(config, feature, provider)
    allow_server_api_keys = getattr(config, "_allow_server_api_keys", True)
    api_key = (
        override.get("api_key")
        or (
            os.getenv(api_key_env)
            if allow_server_api_keys and api_key_env
            else None
        )
        or (
            os.getenv(fallback_api_key_env)
            if allow_server_api_keys and fallback_api_key_env
            else None
        )
    )
    base_url = (
        override.get("api_base")
        or (os.getenv(api_base_env) if api_base_env else None)
        or (os.getenv(fallback_api_base_env) if fallback_api_base_env else None)
        or default_api_base
    )
    model_name = (
        override.get("model")
        or (os.getenv(model_env) if model_env else None)
        or (os.getenv(fallback_model_env) if fallback_model_env else None)
        or default_model
    )
    if allow_empty_local_api_key:
        api_key = resolve_openai_compatible_api_key(api_key, base_url)
    return RuntimeAPIConfig(
        api_key=api_key,
        base_url=(base_url or default_api_base).rstrip("/"),
        model_name=model_name or default_model,
    )
