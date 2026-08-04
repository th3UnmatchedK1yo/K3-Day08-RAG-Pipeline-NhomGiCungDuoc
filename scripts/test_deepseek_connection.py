#!/usr/bin/env python
"""Minimal DeepSeek live connection test. Never prints API keys."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env_utils import classify_external_error, get_env, is_configured, load_repo_env


def main() -> int:
    load_repo_env()
    base = get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com") or "https://api.deepseek.com"
    model = get_env("LLM_MODEL", "deepseek-v4-flash") or "deepseek-v4-flash"
    print(f"base URL: {base}")
    print(f"model: {model}")

    if not is_configured("DEEPSEEK_API_KEY"):
        print("status: failure")
        print("error: key_missing")
        return 1

    try:
        from src.task10_generation import call_deepseek

        answer = call_deepseek(
            [
                {
                    "role": "user",
                    "content": "Return exactly this text and nothing else: DEEPSEEK_CONNECTION_OK",
                }
            ]
        )
        print(f"response_preview: {answer[:80]}")
        if "DEEPSEEK_CONNECTION_OK" in answer:
            print("status: success")
            return 0
        print("status: failure")
        print("error: unexpected_response")
        return 2
    except Exception as exc:
        category = str(exc) if str(exc) in {
            "authentication",
            "invalid_model",
            "invalid_endpoint",
            "insufficient_balance",
            "quota_exceeded",
            "timeout",
            "network_failure",
            "service_processing_failure",
            "malformed_response",
            "empty_message_content",
        } else classify_external_error(exc)
        print("status: failure")
        print(f"error: {category}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
