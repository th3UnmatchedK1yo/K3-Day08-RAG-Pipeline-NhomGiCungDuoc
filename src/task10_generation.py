"""
Task 10 — Evidence-based generation with DeepSeek V4 Flash.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .env_utils import classify_external_error, get_env, is_configured, load_repo_env
from .task9_retrieval_pipeline import retrieve

load_repo_env()

TOP_K = int(get_env("TOP_K", "5") or "5")
TOP_P = float(get_env("LLM_TOP_P", "1.0") or "1.0")
TEMPERATURE = float(get_env("LLM_TEMPERATURE", "0.2") or "0.2")
MAX_TOKENS = int(get_env("LLM_MAX_TOKENS", "1800") or "1800")
LLM_TIMEOUT = float(get_env("LLM_TIMEOUT_SECONDS", "90") or "90")
LLM_MODEL = get_env("LLM_MODEL", "deepseek-v4-flash") or "deepseek-v4-flash"
DEEPSEEK_BASE_URL = get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com") or "https://api.deepseek.com"
DEEPSEEK_THINKING = get_env("DEEPSEEK_THINKING", "disabled") or "disabled"

HANDWRITING_REFUSAL = (
    "Tôi không thể phân tích trực tiếp nội dung bài viết này vì corpus hiện chỉ có "
    "đề bài, band score và nhận xét của examiner, không có phần bài viết tay ở dạng "
    "văn bản."
)

INSUFFICIENT_EVIDENCE = "Tôi không thể xác minh nội dung này từ corpus hiện có."

DEEPSEEK_UNAVAILABLE = (
    "DeepSeek hiện không khả dụng; hệ thống đang hiển thị bằng chứng truy xuất trực "
    "tiếp từ corpus."
)

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu tiêu chí IELTS Writing Band Descriptors và nhận xét examiner công khai.

Quy tắc bắt buộc:
1. Chỉ dùng thông tin từ context được cung cấp — KHÔNG bịa đặt, KHÔNG thêm kiến thức IELTS ngoài corpus.
2. Trả lời bằng tiếng Việt trừ khi người dùng hỏi bằng tiếng Anh.
3. Phân biệt rõ: official scoring descriptor / examiner comment / official teaching guidance / assistant explanation.
4. Mỗi khẳng định về descriptor, tiêu chí chấm, hoặc nhận xét examiner phải có citation liền kề dạng:
   [Source title, section or script, page] hoặc [PageIndex: Source title, section, p.X] hoặc [Source org, title] với nguồn web.
5. Không nhận mình là giám khảo IELTS chính thức.
6. Không đảm bảo band score chính xác cho bài của người dùng.
7. Không suy ra sub-score từng tiêu chí từ overall band.
8. Không khẳng định một linking word tự động đạt Band 8.
9. Không khẳng định model answer chưa được chấm là Band 8.
10. Không phân tích chữ viết tay không có trong corpus. Nếu bị hỏi về handwriting unavailable, trả lời đúng câu refusal đã cung cấp.
11. Nếu evidence không đủ: "Tôi không thể xác minh nội dung này từ corpus hiện có."
12. Không mô tả rank-based PageIndex score như độ tin cậy ngữ nghĩa.
"""

_LLM_CLIENT = None


def get_llm_client():
    """Cached OpenAI-compatible DeepSeek client."""
    global _LLM_CLIENT
    load_repo_env()
    if not is_configured("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY not configured")
    if _LLM_CLIENT is None:
        from openai import OpenAI

        _LLM_CLIENT = OpenAI(
            api_key=get_env("DEEPSEEK_API_KEY"),
            base_url=get_env("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL) or DEEPSEEK_BASE_URL,
            timeout=float(get_env("LLM_TIMEOUT_SECONDS", str(LLM_TIMEOUT)) or LLM_TIMEOUT),
        )
    return _LLM_CLIENT


def call_deepseek(messages: list[dict]) -> str:
    """Call DeepSeek chat completions. Raises RuntimeError with sanitized category on failure."""
    client = get_llm_client()
    model = get_env("LLM_MODEL", LLM_MODEL) or LLM_MODEL
    temperature = float(get_env("LLM_TEMPERATURE", str(TEMPERATURE)) or TEMPERATURE)
    top_p = float(get_env("LLM_TOP_P", str(TOP_P)) or TOP_P)
    max_tokens = int(get_env("LLM_MAX_TOKENS", str(MAX_TOKENS)) or MAX_TOKENS)
    thinking = get_env("DEEPSEEK_THINKING", DEEPSEEK_THINKING) or "disabled"

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    # Optional thinking field — retry once without if unsupported
    try_with_thinking = thinking.lower() in {"enabled", "disabled", "auto"}
    if try_with_thinking:
        kwargs["extra_body"] = {"thinking": {"type": thinking}}

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as first_exc:
        category = classify_external_error(first_exc)
        msg = str(first_exc).lower()
        unsupported = (
            "unexpected keyword" in msg
            or "extra_body" in msg
            or "thinking" in msg
            or "unsupported" in msg
            or "unknown field" in msg
        )
        if unsupported and "extra_body" in kwargs:
            kwargs.pop("extra_body", None)
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as second_exc:
                raise RuntimeError(classify_external_error(second_exc)) from second_exc
        elif category in {"authentication", "insufficient_balance"}:
            raise RuntimeError(category) from first_exc
        else:
            raise RuntimeError(category) from first_exc

    try:
        choices = response.choices or []
        if not choices:
            raise RuntimeError("malformed_response")
        content = choices[0].message.content
        if content is None or not str(content).strip():
            raise RuntimeError("empty_message_content")
        return str(content).strip()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("malformed_response") from exc


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Lost-in-the-middle mitigation: important chunks at start and end."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def _citation_label(chunk: dict) -> str:
    meta = chunk.get("metadata") or {}
    title = meta.get("source_title") or meta.get("source") or "Unknown source"
    page = meta.get("page_start") or meta.get("page")
    section_bits = []
    if meta.get("task_id"):
        section_bits.append(str(meta["task_id"]))
    if meta.get("script_id"):
        section_bits.append(str(meta["script_id"]))
    if meta.get("section"):
        section_bits.append(str(meta["section"]))
    if meta.get("content_type"):
        section_bits.append(str(meta["content_type"]).replace("_", " "))
    if meta.get("criterion"):
        section_bits.append(str(meta["criterion"]).replace("_", " "))
    if meta.get("band") is not None:
        section_bits.append(f"Band {meta['band']}")
    section = " ".join(section_bits) if section_bits else (meta.get("content_type") or "section")

    is_pageindex = (
        chunk.get("source") == "pageindex"
        or meta.get("retrieval_provider") == "pageindex"
    )
    if is_pageindex:
        if page is not None:
            return f"[PageIndex: {title}, {section}, p.{page}]"
        return f"[PageIndex: {title}, {section}]"

    if meta.get("source_url") and not page:
        org = meta.get("source_org") or "Web"
        return f"[{org}, {title}]"
    if page is not None:
        return f"[{title}, {section}, p.{page}]"
    return f"[{title}, {section}]"


def format_context(chunks: list[dict]) -> str:
    """Format chunks with source labels for citation (hybrid + PageIndex)."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata") or {}
        source = meta.get("source") or meta.get("source_file") or meta.get("source_title") or f"Source {i}"
        label = _citation_label(chunk)
        ctype = meta.get("content_type") or meta.get("type") or "unknown"
        provider = chunk.get("source") or meta.get("retrieval_provider") or "hybrid"
        parts.append(
            f"[Document {i} | Source: {source} | Type: {ctype} | Provider: {provider} | CiteAs: {label}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(parts)


def _looks_like_handwriting_analysis(query: str) -> bool:
    q = (query or "").lower()
    keys = [
        "chữ viết tay",
        "bài viết tay",
        "handwritten",
        "handwriting",
        "phân tích bài mẫu viết tay",
        "wording of the candidate",
        "exact wording",
    ]
    return any(k in q for k in keys)


def _retrieval_only_answer(query: str, chunks: list[dict], reason: Optional[str] = None) -> str:
    if _looks_like_handwriting_analysis(query):
        available = any((c.get("metadata") or {}).get("candidate_text_available") for c in chunks)
        if not available:
            return HANDWRITING_REFUSAL

    if not chunks:
        return INSUFFICIENT_EVIDENCE

    header = DEEPSEEK_UNAVAILABLE
    if reason:
        header = f"{DEEPSEEK_UNAVAILABLE} (lý do: {reason})"

    lines = [
        f"**[Retrieval-only evidence]** {header}",
        "",
        "Các đoạn nguồn liên quan nhất (không phải câu trả lời do LLM sinh):",
        "",
    ]
    for i, chunk in enumerate(chunks, 1):
        cite = _citation_label(chunk)
        excerpt = (chunk.get("content") or "")[:500].strip()
        lines.append(f"{i}. {cite}")
        lines.append(excerpt)
        lines.append("")
    lines.append(
        "_Trợ lý này tra cứu nguồn IELTS công khai; không phải giám khảo chính thức và không đảm bảo band score._"
    )
    return "\n".join(lines)


def deepseek_configured() -> bool:
    return is_configured("DEEPSEEK_API_KEY")


def generate_with_citation(query: str, top_k: int = TOP_K, **retrieve_kwargs: Any) -> dict:
    """End-to-end RAG generation with DeepSeek; retrieval-only evidence if unavailable."""
    chunks = retrieve(query, top_k=top_k, **retrieve_kwargs)

    if _looks_like_handwriting_analysis(query):
        available = any((c.get("metadata") or {}).get("candidate_text_available") for c in chunks)
        if not available:
            return {
                "answer": HANDWRITING_REFUSAL,
                "sources": chunks,
                "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
                "mode": "refusal",
            }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    if not deepseek_configured():
        return {
            "answer": _retrieval_only_answer(query, reordered, reason="key_missing"),
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
            "mode": "retrieval_only",
        }

    user_message = (
        f"Context:\n{context}\n\n---\n\nQuestion: {query}\n\n"
        "Hãy trả lời theo quy tắc hệ thống, có citation liền kề mỗi khẳng định factual."
    )

    try:
        answer = call_deepseek(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )
    except Exception as exc:
        reason = str(exc) if str(exc) in {
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
            "key_missing",
        } else "service_processing_failure"
        return {
            "answer": _retrieval_only_answer(query, reordered, reason=reason),
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
            "mode": "retrieval_only_error",
            "error_category": reason,
        }

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "mode": "generated",
    }


if __name__ == "__main__":
    for q in [
        "Sự khác biệt giữa Band 6 và Band 7 ở Lexical Resource Task 2 là gì?",
        "Task Achievement và Task Response khác nhau như thế nào?",
        "Phân tích chữ viết tay bài Task 2A Band 8.5",
    ]:
        print("=" * 70)
        print("Q:", q)
        result = generate_with_citation(q)
        print("A:", result["answer"][:800])
        print("mode:", result.get("mode"), "sources:", len(result["sources"]))
