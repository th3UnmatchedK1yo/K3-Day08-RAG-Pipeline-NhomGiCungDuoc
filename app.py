"""
IELTS Writing Band Descriptor & Examiner Feedback Assistant
Streamlit RAG chatbot (Task 9 retrieval + Task 10 generation).

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env_utils import get_env, load_repo_env, sanitized_config_status

load_repo_env()
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface"))

st.set_page_config(
    page_title="IELTS Writing Band Descriptor & Examiner Feedback Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = (
    "Trợ lý này tra cứu và giải thích các tiêu chí cùng nguồn IELTS công khai. "
    "Đây không phải là giám khảo IELTS chính thức và không đảm bảo band score."
)

SUGGESTIONS = [
    "Sự khác biệt giữa Band 6 và Band 7 ở Lexical Resource Task 2 là gì?",
    "Task Achievement và Task Response khác nhau như thế nào?",
    "Band 8 yêu cầu gì ở Coherence and Cohesion?",
    "Examiner nhận xét gì về bài Task 2A Band 8.5?",
    "Vì sao dùng quá nhiều cohesive devices có thể làm giảm chất lượng bài viết?",
    "Cho ví dụ về cách dùng reference và substitution để tạo cohesion.",
]

CRITERIA = [
    "(any)",
    "task_achievement",
    "task_response",
    "coherence_cohesion",
    "lexical_resource",
    "grammatical_range_accuracy",
]


def _safe_collection_count():
    try:
        from src.task4_chunking_indexing import get_collection

        return get_collection().count()
    except Exception:
        return "n/a"


def _web_corpus_status() -> dict:
    report_path = PROJECT_ROOT / "data" / "inspection" / "web_crawl_report.json"
    news_dir = PROJECT_ROOT / "data" / "landing" / "news"
    success = 0
    failed = 0
    latest = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            success = report.get("successful_web_articles", 0)
            failed = len(report.get("failed_urls") or [])
            latest = report.get("crawled_at")
            return {"success": success, "failed": failed, "latest": latest}
        except Exception:
            pass
    if news_dir.exists():
        for f in news_dir.glob("article_*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("status") == "success" and len(data.get("content_markdown") or "") >= 500:
                    success += 1
                else:
                    failed += 1
                latest = data.get("date_crawled") or latest
            except Exception:
                failed += 1
    return {"success": success, "failed": failed, "latest": latest}


with st.sidebar:
    st.title("IELTS Writing Assistant")
    st.caption("Band Descriptors & Examiner Feedback")
    st.info(DISCLAIMER)

    st.divider()
    st.subheader("Trạng thái hệ thống")
    cfg = sanitized_config_status()
    try:
        from src.task8_pageindex_vectorless import pageindex_status_summary

        pi = pageindex_status_summary()
    except Exception:
        pi = {
            "enabled": False,
            "key_configured": False,
            "prepared_count": 0,
            "uploaded_count": 0,
            "completed_count": 0,
        }
    web = _web_corpus_status()
    st.markdown(
        f"""
**Embedding**
- model: `{get_env('EMBEDDING_MODEL', 'BAAI/bge-m3') or 'BAAI/bge-m3'}`
- backend: `{get_env('EMBEDDING_BACKEND', 'local') or 'local'}`
- collection: `{get_env('CHROMA_COLLECTION', 'ielts_writing_docs') or 'ielts_writing_docs'}`
- count: `{_safe_collection_count()}`

**LLM**
- provider: DeepSeek
- configured: `{str(cfg.get('DEEPSEEK_API_KEY')).lower()}`
- model: `{get_env('LLM_MODEL', 'deepseek-v4-flash') or 'deepseek-v4-flash'}`

**PageIndex**
- enabled: `{str(pi.get('enabled')).lower()}`
- key configured: `{str(pi.get('key_configured')).lower()}`
- prepared: `{pi.get('prepared_count')}`
- uploaded: `{pi.get('uploaded_count')}`
- completed: `{pi.get('completed_count')}`

**Web corpus**
- successful: `{web.get('success')} / 5`
- failed: `{web.get('failed')}`
- latest crawl: `{web.get('latest') or 'n/a'}`

**Retrieval**
- mode: `{get_env('RETRIEVAL_MODE', 'hybrid') or 'hybrid'}`
- threshold: `{get_env('SCORE_THRESHOLD', '0.51') or '0.51'}`
"""
    )

    mode = st.radio(
        "Chế độ",
        ["Tra cứu tiêu chí", "So sánh band", "Tra cứu nhận xét examiner"],
        index=0,
    )

    st.divider()
    st.subheader("Câu hỏi gợi ý")
    for s in SUGGESTIONS:
        if st.button(s, use_container_width=True, key=f"sug_{hash(s)}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("Bộ lọc (tuỳ chọn)")
    task_filter = st.selectbox("Task", ["(any)", "Task 1", "Task 2"])
    criterion_filter = st.selectbox("Criterion", CRITERIA)
    band_filter = st.text_input("Band (ví dụ 7 hoặc 8.5)", "")
    examiner_only = st.checkbox("Chỉ nhận xét examiner chính thức", value=False)
    top_k = st.slider("Context top_k", 3, 10, 5)
    show_debug = st.checkbox("Hiện debug scores", value=False)

    st.divider()
    st.caption("Kiến trúc: Dense (local bge-m3) + BM25 + RRF → optional PageIndex → DeepSeek")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

st.title("IELTS Writing Band Descriptor & Examiner Feedback Assistant")
st.caption(DISCLAIMER)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Nguồn tham khảo ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata") or {}
                    st.markdown(
                        f"**[{i}] {meta.get('source_title') or meta.get('source', 'Unknown')}** "
                        f"| org: `{meta.get('source_org', 'n/a')}` "
                        f"| type: `{meta.get('content_type') or meta.get('type', 'n/a')}` "
                        f"| page: `{meta.get('page_start') or meta.get('page', 'n/a')}` "
                        f"| method: `{src.get('source') or src.get('retrieval_mode', 'hybrid')}`"
                    )
                    if show_debug or msg.get("debug"):
                        st.caption(
                            f"score={src.get('score')} dense={src.get('dense_score') or meta.get('best_dense_score')} "
                            f"fallback_attempted={meta.get('fallback_attempted')} "
                            f"fallback_succeeded={meta.get('fallback_succeeded')}"
                        )
                    st.text((src.get("content") or "")[:400])
                    st.divider()


def _build_where():
    clauses = []
    if task_filter == "Task 1":
        clauses.append({"task_number": 1})
    elif task_filter == "Task 2":
        clauses.append({"task_number": 2})
    if criterion_filter != "(any)":
        clauses.append({"criterion": criterion_filter})
    if band_filter.strip():
        try:
            clauses.append({"band": float(band_filter.strip())})
        except ValueError:
            pass
    if examiner_only or mode == "Tra cứu nhận xét examiner":
        clauses.append({"content_type": "examiner_comment"})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


user_input = st.chat_input("Nhập câu hỏi về tiêu chí IELTS Writing / nhận xét examiner...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    if mode == "So sánh band" and "band" not in query.lower():
        query = f"[So sánh band] {query}"
    elif mode == "Tra cứu nhận xét examiner" and "examiner" not in query.lower() and "nhận xét" not in query.lower():
        query = f"Examiner comment: {query}"

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất nguồn IELTS và tổng hợp câu trả lời..."):
            answer = ""
            sources = []
            try:
                from src.task10_generation import generate_with_citation

                where = _build_where()
                response = generate_with_citation(query, top_k=top_k, where=where)
                answer = response.get("answer") or "Chưa thể trả lời."
                sources = response.get("sources") or []

                if sources:
                    meta0 = sources[0].get("metadata") or {}
                    st.caption(
                        f"Retrieval source: `{sources[0].get('source', 'hybrid')}` · "
                        f"fallback_attempted=`{meta0.get('fallback_attempted')}` · "
                        f"fallback_succeeded=`{meta0.get('fallback_succeeded')}` · "
                        f"best_dense=`{meta0.get('best_dense_score')}`"
                    )

                examiner_chunks = [
                    s for s in sources
                    if (s.get("metadata") or {}).get("content_type") == "examiner_comment"
                ]
                if examiner_chunks:
                    meta0 = examiner_chunks[0].get("metadata") or {}
                    if meta0.get("candidate_text_available") is False:
                        st.warning(
                            "Phần bài viết tay của thí sinh không có dạng văn bản trong corpus. "
                            "Chỉ hiển thị đề bài (nếu có), band và nhận xét examiner."
                        )
                    if meta0.get("band") is not None:
                        st.markdown(f"**Overall band (từ nguồn):** {meta0.get('band')}")

            except NotImplementedError:
                answer = "Task 10 chưa được implement."
            except Exception as e:
                answer = (
                    "Không thể hoàn tất generation, nhưng hệ thống vẫn an toàn không crash.\n\n"
                    f"Chi tiết đã được làm sạch: `{type(e).__name__}`"
                )
                try:
                    from src.task9_retrieval_pipeline import retrieve

                    sources = retrieve(query, top_k=top_k)
                    if sources:
                        answer += "\n\n**Đoạn nguồn liên quan:**\n"
                        for i, s in enumerate(sources, 1):
                            answer += f"\n{i}. {(s.get('content') or '')[:300]}\n"
                except Exception:
                    sources = []

            st.markdown(answer)
            if sources:
                with st.expander(f"Nguồn tham khảo ({len(sources)})"):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata") or {}
                        st.markdown(
                            f"**[{i}] {meta.get('source_title') or meta.get('source', 'Unknown')}**  \n"
                            f"org=`{meta.get('source_org', 'n/a')}` · "
                            f"type=`{meta.get('content_type') or meta.get('type', 'n/a')}` · "
                            f"page=`{meta.get('page_start') or meta.get('page', 'n/a')}` · "
                            f"method=`{src.get('source') or src.get('retrieval_mode', 'hybrid')}`"
                        )
                        if show_debug:
                            with st.expander("Debug scores", expanded=False):
                                st.json(
                                    {
                                        "score": src.get("score"),
                                        "dense_score": src.get("dense_score"),
                                        "bm25_score": src.get("bm25_score"),
                                        "rrf_score": src.get("rrf_score"),
                                        "best_dense_score": meta.get("best_dense_score"),
                                        "fallback_attempted": meta.get("fallback_attempted"),
                                        "fallback_succeeded": meta.get("fallback_succeeded"),
                                        "score_type": meta.get("score_type"),
                                    }
                                )
                        st.text((src.get("content") or "")[:400])
                        st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "debug": show_debug}
    )
