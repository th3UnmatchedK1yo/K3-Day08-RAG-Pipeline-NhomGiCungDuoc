"""
Task 7 — Reranking (RRF primary; cross-encoder/MMR optional).
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional


def _item_key(item: dict) -> str:
    meta = item.get("metadata") or {}
    chunk_id = meta.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    content = item.get("content") or ""
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Optional cross-encoder rerank; fails gracefully if unavailable."""
    try:
        import os
        import requests

        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise RuntimeError("JINA_API_KEY missing")
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        reranked = response.json().get("results", [])
        out = []
        for r in reranked:
            item = dict(candidates[r["index"]])
            item["score"] = float(r["relevance_score"])
            out.append(item)
        return out[:top_k]
    except Exception as exc:
        print(f"[WARN] Cross-encoder unavailable: {exc}. Returning original top_k by score.")
        ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return ranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Optional MMR; requires embeddings on candidates."""
    try:
        import numpy as np

        def cos(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            return float(np.dot(a, b) / denom)

        selected: list[int] = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected) < top_k:
            best_idx = None
            best_score = float("-inf")
            for idx in remaining:
                emb = candidates[idx].get("embedding")
                if emb is None:
                    relevance = float(candidates[idx].get("score") or 0)
                    mmr_score = relevance
                else:
                    relevance = cos(query_embedding, emb)
                    max_sim = 0.0
                    for sel in selected:
                        sel_emb = candidates[sel].get("embedding")
                        if sel_emb is not None:
                            max_sim = max(max_sim, cos(emb, sel_emb))
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [candidates[i] for i in selected]
    except Exception as exc:
        print(f"[WARN] MMR unavailable: {exc}")
        return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion.
    RRF(d) = sum(1 / (k + rank))
    Merge by metadata.chunk_id (fallback content hash).
    Preserves dense_score, bm25_score, rrf_score.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        for rank, item in enumerate(ranked_list, start=1):
            key = _item_key(item)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map:
                merged = dict(item)
                meta = dict(item.get("metadata") or {})
                merged["metadata"] = meta
                content_map[key] = merged
            else:
                # Merge score provenance
                existing = content_map[key]
                if "dense_score" in item and "dense_score" not in existing:
                    existing["dense_score"] = item["dense_score"]
                if "bm25_score" in item and "bm25_score" not in existing:
                    existing["bm25_score"] = item["bm25_score"]
            # Infer provenance from list position if not labeled
            current = content_map[key]
            if list_idx == 0 and "dense_score" not in current:
                current["dense_score"] = float(item.get("score") or 0.0)
            if list_idx == 1 and "bm25_score" not in current:
                current["bm25_score"] = float(item.get("score") or 0.0)

    sorted_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results: list[dict] = []
    for key, score in sorted_keys[:top_k]:
        item = dict(content_map[key])
        item["rrf_score"] = score
        item["score"] = score  # final ordering score
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
    ranked_lists: Optional[list[list[dict]]] = None,
) -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        print("[WARN] MMR via unified rerank() needs query_embedding; falling back to score sort.")
        return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
    if method == "rrf":
        lists = ranked_lists if ranked_lists is not None else [candidates]
        return rerank_rrf(lists, top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dense = [
        {"content": "Band 7 Lexical Resource", "score": 0.8, "metadata": {"chunk_id": "a"}},
        {"content": "Band 6 Lexical Resource", "score": 0.7, "metadata": {"chunk_id": "b"}},
    ]
    sparse = [
        {"content": "Band 6 Lexical Resource", "score": 4.2, "metadata": {"chunk_id": "b"}},
        {"content": "Coherence and Cohesion Band 8", "score": 3.1, "metadata": {"chunk_id": "c"}},
    ]
    results = rerank_rrf([dense, sparse], top_k=3)
    for r in results:
        print(f"[{r['score']:.4f}] {r['metadata'].get('chunk_id')} {r['content']}")
