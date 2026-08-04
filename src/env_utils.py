"""Repository-root environment helpers. Never log secret values."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PLACEHOLDER_VALUES = {
    "",
    "PASTE_KEY_HERE",
    "DIEN_KEY_VAO_DAY",
    "YOUR_API_KEY",
    "sk-or-v1-...",
    "pix_...",
}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_repo_env() -> Path:
    """Load <repo-root>/.env explicitly. Cached; never prints values."""
    root = get_repo_root()
    env_path = root / ".env"
    load_dotenv(env_path, override=False)
    # Ensure local HF caches stay inside the repo
    hf = root / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf))
    return env_path


def is_configured(name: str) -> bool:
    load_repo_env()
    value = (os.getenv(name) or "").strip()
    if not value:
        return False
    if value in PLACEHOLDER_VALUES:
        return False
    if value.endswith("...") and len(value) < 20:
        return False
    return True


def get_env(name: str, default: str = "") -> str:
    load_repo_env()
    return (os.getenv(name) or default).strip()


def sanitized_config_status() -> dict[str, bool]:
    return {
        "DEEPSEEK_API_KEY": is_configured("DEEPSEEK_API_KEY"),
        "PAGEINDEX_API_KEY": is_configured("PAGEINDEX_API_KEY"),
        "QWEN_API_KEY": is_configured("QWEN_API_KEY"),
        "HF_TOKEN": is_configured("HF_TOKEN"),
    }


def classify_external_error(exc: Exception) -> str:
    """Map exceptions to sanitized categories without leaking secrets."""
    text = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    if status in (401, 403) or "unauthorized" in text or "authentication" in text or "invalid api key" in text:
        return "authentication"
    if status == 404 or "not found" in text:
        return "invalid_endpoint"
    if status == 400 or "invalid model" in text or "model_not_found" in text:
        return "invalid_model"
    if status == 429 or "rate limit" in text:
        return "quota_exceeded"
    if "insufficient" in text or "balance" in text or "billing" in text or "payment" in text:
        return "insufficient_balance"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "connection" in text or "network" in text or "dns" in text:
        return "network_failure"
    if status in (500, 502, 503, 504):
        return "service_processing_failure"
    return "service_processing_failure"


if __name__ == "__main__":
    load_repo_env()
    status = sanitized_config_status()
    for key, ok in status.items():
        print(f"{key} configured: {str(ok).lower()}")
