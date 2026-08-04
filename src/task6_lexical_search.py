"""
Task 6 — BM25 lexical search over the same IELTS chunks used in Chroma.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .task4_chunking_indexing import chunk_documents, load_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "bm25"
CACHE_CORPUS = CACHE_DIR / "corpus.pkl"
CACHE_META = CACHE_DIR / "corpus_meta.json"

CORPUS: list[dict] = []
_BM25: Optional[BM25Okapi] = None

MULTIWORD_PHRASES = [
    "task response",
    "task achievement",
    "lexical resource",
    "coherence and cohesion",
    "grammatical range and accuracy",
    "cohesive devices",
    "band score",
]


def tokenize(text: str) -> list[str]:
    """Lowercase tokenize; preserve numbers/bands and key IELTS multiword phrases."""
    if not text:
        return []
    text = text.lower()
    # Protect multiword phrases with lowercase placeholders
    for i, phrase in enumerate(MULTIWORD_PHRASES):
        text = text.replace(phrase, f" __phrase{i}__ ")
    # Keep numbers like 8.5
    text = re.sub(r"[^\w\s.]", " ", text)
    tokens: list[str] = []
    for tok in text.split():
        m = re.fullmatch(r"__phrase(\d+)__", tok)
        if m:
            idx = int(m.group(1))
            tokens.append(MULTIWORD_PHRASES[idx].replace(" ", "_"))
            continue
        # Keep band-like decimals and words
        if re.fullmatch(r"\d+(?:\.\d+)?", tok) or re.search(r"[a-zA-ZÀ-ỹ]", tok):
            tokens.append(tok.strip("."))
    return tokens


def _load_chunk_corpus() -> list[dict]:
    docs = load_documents()
    chunks = chunk_documents(docs) if docs else []
    return [{"content": c["content"], "metadata": c.get("metadata") or {}} for c in chunks]


def build_bm25_index(corpus: list[dict] | None = None):
    """Build and cache BM25 index from corpus chunks."""
    global CORPUS, _BM25
    if corpus is None:
        corpus = _load_chunk_corpus()
    CORPUS = corpus
    tokenized = [tokenize(doc["content"]) for doc in CORPUS]
    _BM25 = BM25Okapi(tokenized)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE_CORPUS.open("wb") as f:
        pickle.dump({"corpus": CORPUS, "tokenized": tokenized}, f)
    CACHE_META.write_text(
        json.dumps(
            {
                "num_docs": len(CORPUS),
                "phrases": MULTIWORD_PHRASES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return _BM25


def _ensure_index():
    global CORPUS, _BM25
    if _BM25 is not None and CORPUS:
        return
    if CACHE_CORPUS.exists():
        try:
            with CACHE_CORPUS.open("rb") as f:
                data = pickle.load(f)
            CORPUS = data["corpus"]
            _BM25 = BM25Okapi(data["tokenized"])
            return
        except Exception:
            pass
    build_bm25_index()


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """BM25 search; returns non-zero scores only, sorted descending."""
    _ensure_index()
    if not CORPUS or _BM25 is None:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []
    scores = _BM25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1]

    results: list[dict] = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append(
            {
                "content": CORPUS[idx]["content"],
                "score": score,
                "metadata": CORPUS[idx].get("metadata") or {},
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    build_bm25_index()
    for q in [
        "Band 6 and Band 7 Lexical Resource",
        "Task Response",
        "cohesive devices overuse",
    ]:
        print(f"\nQuery: {q}")
        for r in lexical_search(q, top_k=5):
            print(f"[{r['score']:.3f}] {r['content'][:100].replace(chr(10), ' ')}...")
