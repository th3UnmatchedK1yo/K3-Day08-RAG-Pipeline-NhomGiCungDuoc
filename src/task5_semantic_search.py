"""
<<<<<<< HEAD
Task 5 — Semantic Search Module.

Dense retrieval sử dụng BGE-M3 + ChromaDB.
"""

from .task4_chunking_indexing import (
    get_collection,
    get_embedding_model,
)

=======
Task 5 — Semantic search over the IELTS Chroma collection (BAAI/bge-m3).
"""

from __future__ import annotations
>>>>>>> dev

from typing import Any, Optional

from .task4_chunking_indexing import get_collection, get_embedding_model


def semantic_search(
    query: str,
    top_k: int = 10,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Dense retrieval using the same embedding model and Chroma collection as Task 4.

<<<<<<< HEAD
    Args:
        query: Câu truy vấn.
        top_k: Số lượng kết quả tối đa.

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict
        }

        Sorted by score descending.
    """

    if not query or not query.strip():
        return []

    if top_k <= 0:
        return []

    # Không yêu cầu nhiều kết quả hơn số chunk hiện có.
    collection = get_collection()
    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    top_k = min(top_k, total_chunks)

    # ------------------------------------------------------------------
    # Bước 1: Embed query bằng chính model BGE-M3 ở Task 4
    # ------------------------------------------------------------------

    model = get_embedding_model()

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # ------------------------------------------------------------------
    # Bước 2: Query ChromaDB bằng cosine distance
    # ------------------------------------------------------------------

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    # ------------------------------------------------------------------
    # Bước 3: Chuyển cosine distance → similarity score
    # ------------------------------------------------------------------

    output = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(
        documents,
        metadatas,
        distances,
    ):
        score = max(0.0, 1.0 - float(dist))

        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": meta,
        })

    # Đảm bảo kết quả được sort giảm dần theo score.
    output.sort(
        key=lambda x: x["score"],
        reverse=True
    )

=======
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
>>>>>>> dev
    return output[:top_k]


if __name__ == "__main__":
<<<<<<< HEAD
    query = "What is the difference between Band 6 and Band 7 in Lexical Resource?"

    print("=" * 70)
    print("Task 5 — Semantic Search")
    print(f"Query: {query}")
    print("=" * 70)

    results = semantic_search(query, top_k=5)

    print(f"\nFound {len(results)} results:\n")

    for i, result in enumerate(results, start=1):
        print(
            f"[{i}] score={result['score']:.4f} "
            f"source={result['metadata'].get('source')} "
            f"chunk={result['metadata'].get('chunk_index')}"
        )

        print(result["content"][:500])
        print("-" * 70)
=======
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
>>>>>>> dev
