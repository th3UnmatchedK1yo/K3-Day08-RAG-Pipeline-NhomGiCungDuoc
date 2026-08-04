"""
Task 9 — Hybrid retrieval: dense + BM25 + RRF, PageIndex fallback via dense cosine.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from .env_utils import get_env, load_repo_env
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

load_repo_env()

SCORE_THRESHOLD = float(get_env("SCORE_THRESHOLD", "0.51") or "0.51")
DEFAULT_TOP_K = int(get_env("TOP_K", "5") or "5")
RERANK_METHOD = "rrf"

VI_ALIASES = {
    "từ vựng": "lexical_resource",
    "vốn từ": "lexical_resource",
    "mạch lạc": "coherence_cohesion",
    "liên kết": "coherence_cohesion",
    "từ nối": "coherence_cohesion",
    "ngữ pháp": "grammatical_range_accuracy",
    "đáp ứng đề bài": "task_response",
    "hoàn thành yêu cầu": "task_achievement",
    "nhận xét giám khảo": "examiner_comment",
    "nhận xét examiner": "examiner_comment",
}

EN_ALIASES = {
    "vocabulary": "lexical_resource",
    "lexical resource": "lexical_resource",
    "coherence": "coherence_cohesion",
    "cohesion": "coherence_cohesion",
    "grammar": "grammatical_range_accuracy",
    "task response": "task_response",
    "task achievement": "task_achievement",
    "examiner comment": "examiner_comment",
}


def parse_ielts_query(query: str) -> dict[str, Any]:
    """Lightweight IELTS intent parser with VN/EN aliases and band comparisons."""
    q = query or ""
    q_lower = q.lower()
    intent: dict[str, Any] = {
        "criterion": None,
        "content_type": None,
        "task_number": None,
        "bands": [],
        "confident": False,
        "filters": {},
    }

    for phrase, crit in {**VI_ALIASES, **EN_ALIASES}.items():
        if phrase in q_lower:
            if crit == "examiner_comment":
                intent["content_type"] = "examiner_comment"
            else:
                intent["criterion"] = crit

    if re.search(r"\btask\s*1\b|writing task 1|nhiệm vụ 1", q_lower):
        intent["task_number"] = 1
    elif re.search(r"\btask\s*2\b|writing task 2|nhiệm vụ 2", q_lower):
        intent["task_number"] = 2

    band_matches = re.findall(
        r"band\s*(\d(?:\.\d)?)\s*(?:and|vs|versus|compared with|so với|với|và|/|-)\s*band\s*(\d(?:\.\d)?)",
        q_lower,
    )
    bands = []
    if band_matches:
        bands = [float(band_matches[0][0]), float(band_matches[0][1])]
    else:
        singles = re.findall(r"band\s*(\d(?:\.\d)?)", q_lower)
        bands = [float(x) for x in singles]
    intent["bands"] = bands

    # Confident only when we have clear structured signals
    signals = sum(
        [
            intent["criterion"] is not None,
            intent["content_type"] is not None,
            intent["task_number"] is not None,
            len(bands) > 0,
        ]
    )
    intent["confident"] = signals >= 2

    if intent["confident"]:
        filters: dict[str, Any] = {}
        if intent["criterion"]:
            filters["criterion"] = intent["criterion"]
        if intent["content_type"]:
            filters["content_type"] = intent["content_type"]
        if intent["task_number"] is not None:
            filters["task_number"] = intent["task_number"]
        # Chroma where with multiple keys uses $and
        if len(filters) == 1:
            intent["filters"] = filters
        elif len(filters) > 1:
            intent["filters"] = {"$and": [{k: v} for k, v in filters.items()]}
    return intent


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    mode: str = "hybrid",
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Full retrieval pipeline.

    mode:
      - hybrid (Config B): dense + BM25 + RRF
      - dense (Config A): dense only
    Fallback uses ORIGINAL dense cosine similarity, never RRF score.
    """
    intent = parse_ielts_query(query)
    filters = where if where is not None else (intent.get("filters") or None)

    candidate_k = max(top_k * 2, top_k)

    dense_results = semantic_search(query, top_k=candidate_k, where=filters)
    # If filters too strict and empty, retry without filters
    if filters and not dense_results:
        dense_results = semantic_search(query, top_k=candidate_k, where=None)

    for item in dense_results:
        item["dense_score"] = float(item.get("score") or 0.0)

    best_dense = dense_results[0]["dense_score"] if dense_results else 0.0

    if mode == "dense":
        final = []
        for item in dense_results[:top_k]:
            out = dict(item)
            out["source"] = "hybrid"  # tests expect hybrid|pageindex; dense-only still local
            out["retrieval_mode"] = "dense"
            final.append(out)
        if best_dense < score_threshold:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        return final

    sparse_results = lexical_search(query, top_k=candidate_k)
    for item in sparse_results:
        item["bm25_score"] = float(item.get("score") or 0.0)

    merged = rerank_rrf([dense_results, sparse_results], top_k=candidate_k)
    for item in merged:
        item["source"] = "hybrid"
        item["retrieval_mode"] = "hybrid"

    if use_reranking and merged and RERANK_METHOD != "rrf":
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    for item in final_results:
        item["source"] = item.get("source") or "hybrid"

    fallback_attempted = False
    fallback_succeeded = False
    if best_dense < score_threshold:
        fallback_attempted = True
        print(
            f"  [WARN] Semantic best score ({best_dense:.3f}) < threshold ({score_threshold}) "
            "— trying PageIndex fallback"
        )
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            fallback_succeeded = True
            for item in fallback:
                meta = dict(item.get("metadata") or {})
                meta["best_dense_score"] = best_dense
                meta["fallback_attempted"] = True
                meta["fallback_succeeded"] = True
                item["metadata"] = meta
            return fallback

    for item in final_results:
        meta = dict(item.get("metadata") or {})
        meta["best_dense_score"] = best_dense
        meta["fallback_attempted"] = fallback_attempted
        meta["fallback_succeeded"] = fallback_succeeded
        item["metadata"] = meta

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Sự khác biệt giữa Band 6 và Band 7 ở Lexical Resource Task 2 là gì?",
        "What does Task Response assess in Writing Task 2?",
        "Examiner nhận xét gì về bài Task 2A Band 8.5?",
        "How do I repair a car engine?",
        "xyzabc123nonsense",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(
                f"  {i}. [score={r['score']:.4f}] "
                f"[dense={r.get('dense_score')}] [src={r.get('source')}] "
                f"{r['content'][:80].replace(chr(10), ' ')}..."
            )
