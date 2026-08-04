"""
Task 5 — Semantic search over the IELTS Chroma collection (BAAI/bge-m3).
"""

from __future__ import annotations

from typing import Any, Optional

from .task4_chunking_indexing import get_collection, get_embedding_model


def semantic_search(
    query: str,
    top_k: int = 10,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Dense retrieval using the same embedding model and Chroma collection as Task 4.

    Returns list of {content, score, metadata} sorted by cosine similarity descending.
    similarity = 1.0 - chroma_distance
    """
    collection = get_collection()
    try:
        count = collection.count()
    except Exception:
        count = 0
    if count == 0:
        return []

    model = get_embedding_model()
    query_vector = model.encode([query], normalize_embeddings=True)[0].tolist()

    n_results = min(max(top_k, 1), count)
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_vector],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception:
        # Retry without filter if filter invalid
        kwargs.pop("where", None)
        results = collection.query(**kwargs)

    output: list[dict] = []
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        score = 1.0 - float(dist)
        output.append(
            {
                "content": doc,
                "score": score,
                "metadata": meta or {},
            }
        )
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    queries = [
        "Sự khác biệt giữa Band 6 và Band 7 ở Lexical Resource Task 2 là gì?",
        "What does Task Response assess in Writing Task 2?",
        "Band 8 Coherence and Cohesion requirements",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = semantic_search(q, top_k=5)
        for r in results:
            print(f"[{r['score']:.4f}] {r['content'][:120].replace(chr(10), ' ')}...")
