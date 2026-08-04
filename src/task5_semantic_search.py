"""
Task 5 — Semantic Search Module.

Dense retrieval sử dụng BGE-M3 + ChromaDB.
"""

from .task4_chunking_indexing import (
    get_collection,
    get_embedding_model,
)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

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

    return output[:top_k]


if __name__ == "__main__":
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