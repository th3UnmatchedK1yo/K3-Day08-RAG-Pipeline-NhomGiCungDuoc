"""
Task 4 — Structure-aware chunking & ChromaDB indexing for IELTS Writing corpus.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from .env_utils import load_repo_env

load_repo_env()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
CORPUS_JSONL = STANDARDIZED_DIR / "ielts" / "corpus.jsonl"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
INSPECTION_DIR = PROJECT_ROOT / "data" / "inspection"
HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE))

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "markdown_header_recursive"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024
VECTOR_STORE = "chromadb"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "ielts_writing_docs")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")

# Prefer a fully downloaded local snapshot when available (offline-friendly).
_LOCAL_BGE_CANDIDATES = [
    PROJECT_ROOT / ".cache" / "models" / "bge-m3",
    PROJECT_ROOT
    / ".cache"
    / "huggingface"
    / "models--BAAI--bge-m3"
    / "snapshots"
    / "5617a9f61b028005a4858fdac845db406aefb181",
]


def _resolve_embedding_model_name() -> str:
    for path in _LOCAL_BGE_CANDIDATES:
        weights = path / "pytorch_model.bin"
        safetensors = path / "model.safetensors"
        config = path / "config.json"
        if config.exists() and (
            (weights.exists() and weights.stat().st_size > 1_000_000_000)
            or (safetensors.exists() and safetensors.stat().st_size > 1_000_000_000)
        ):
            return str(path)
    return EMBEDDING_MODEL


_EMBEDDING_MODEL = None
_CHROMA_CLIENT = None


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        HF_CACHE.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(HF_CACHE)
        os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
        from sentence_transformers import SentenceTransformer

        model_name = _resolve_embedding_model_name()
        print(f"[OK] Loading embedding model from: {model_name}")
        _EMBEDDING_MODEL = SentenceTransformer(model_name, cache_folder=str(HF_CACHE))
    return _EMBEDDING_MODEL


def _get_chroma_client():
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _CHROMA_CLIENT


def get_collection():
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    """Delete only the local IELTS collection."""
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[OK] Deleted collection: {COLLECTION_NAME}")
    except Exception as exc:
        print(f"[WARN] Could not delete collection ({exc})")
    return get_collection()


def load_documents() -> list[dict]:
    """
    Load corpus records (preferred) or standardized markdown.
    Returns list of {'content': str, 'metadata': dict}.
    """
    documents: list[dict] = []
    if CORPUS_JSONL.exists():
        with CORPUS_JSONL.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = (rec.get("text") or "").strip()
                if not text:
                    continue
                meta = {k: v for k, v in rec.items() if k != "text"}
                meta["source"] = rec.get("source_file") or rec.get("source_title") or rec.get("doc_id")
                meta["type"] = rec.get("source_type") or rec.get("content_type")
                documents.append({"content": text, "metadata": meta})
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "source_file": md_file.name,
                    "doc_id": md_file.stem,
                    "record_id": md_file.stem,
                    "content_type": "teaching_guidance",
                },
            }
        )
    return documents


def _split_oversized(text: str, base_meta: dict, record_id: str, start_idx: int = 0) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for i, piece in enumerate(splitter.split_text(text)):
        idx = start_idx + i
        chunk_id = f"{record_id}_chunk_{idx:04d}"
        meta = {**base_meta, "chunk_id": chunk_id, "chunk_index": idx}
        chunks.append({"content": piece, "metadata": meta})
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Structure-aware chunking:
    - Keep band descriptor / examiner comment / task prompt together when possible
    - Split web guidance by markdown headers then recursive splitter
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for doc in documents:
        content = doc["content"]
        meta = dict(doc.get("metadata") or {})
        record_id = str(meta.get("record_id") or meta.get("doc_id") or meta.get("source") or "doc")
        content_type = meta.get("content_type") or ""

        keep_together_types = {
            "band_descriptor",
            "assessment_criterion",
            "examiner_comment",
            "task_prompt",
            "typed_candidate_response",
            "practice_question",
        }

        if content_type in keep_together_types:
            if len(content) <= int(CHUNK_SIZE * 1.1):
                chunk_id = f"{record_id}_chunk_0001"
                chunks.append(
                    {
                        "content": content,
                        "metadata": {**meta, "chunk_id": chunk_id, "chunk_index": 0},
                    }
                )
            else:
                chunks.extend(_split_oversized(content, meta, record_id, start_idx=1))
            continue

        # Teaching / web guidance: split by headers first
        try:
            sections = header_splitter.split_text(content)
        except Exception:
            sections = []

        if not sections:
            parts = recursive.split_text(content)
            for i, piece in enumerate(parts, start=1):
                chunk_id = f"{record_id}_chunk_{i:04d}"
                chunks.append(
                    {
                        "content": piece,
                        "metadata": {**meta, "chunk_id": chunk_id, "chunk_index": i},
                    }
                )
            continue

        local_idx = 1
        for section in sections:
            section_text = section.page_content if hasattr(section, "page_content") else str(section)
            section_meta = {**meta}
            if hasattr(section, "metadata"):
                section_meta.update(section.metadata)
            if len(section_text) <= int(CHUNK_SIZE * 1.1):
                chunk_id = f"{record_id}_chunk_{local_idx:04d}"
                chunks.append(
                    {
                        "content": section_text,
                        "metadata": {**section_meta, "chunk_id": chunk_id, "chunk_index": local_idx},
                    }
                )
                local_idx += 1
            else:
                sub = _split_oversized(section_text, section_meta, record_id, start_idx=local_idx)
                chunks.extend(sub)
                local_idx += len(sub)

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return chunks
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    collection = get_collection()
    if not chunks:
        print("[WARN] No chunks to index")
        return

    ids = []
    documents = []
    embeddings = []
    metadatas = []
    for c in chunks:
        meta = _sanitize_metadata(c.get("metadata") or {})
        chunk_id = meta.get("chunk_id") or f"chunk_{len(ids):04d}"
        ids.append(str(chunk_id))
        documents.append(c["content"])
        embeddings.append(c["embedding"])
        metadatas.append(meta)

    batch = 100
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[i : i + batch],
            documents=documents[i : i + batch],
            embeddings=embeddings[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )
    print(f"[OK] Upserted {len(ids)} chunks into '{COLLECTION_NAME}'")


def _write_indexing_report(docs: list[dict], chunks: list[dict], skipped: list[dict]):
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    by_content = Counter(c["metadata"].get("content_type") for c in chunks)
    by_criterion = Counter(c["metadata"].get("criterion") for c in chunks if c["metadata"].get("criterion"))
    by_band = Counter(str(c["metadata"].get("band")) for c in chunks if c["metadata"].get("band") is not None)
    by_source = Counter(c["metadata"].get("source_title") or c["metadata"].get("source") for c in chunks)
    report = {
        "documents_loaded": len(docs),
        "records_loaded": len(docs),
        "chunks_created": len(chunks),
        "chunks_by_content_type": dict(by_content),
        "chunks_by_criterion": dict(by_criterion),
        "chunks_by_band": dict(by_band),
        "chunks_by_source": dict(by_source),
        "records_skipped": len(skipped),
        "reasons_for_skipping": skipped,
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    path = INSPECTION_DIR / "indexing_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {path}")


def run_pipeline(reset: bool = True):
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} / {COLLECTION_NAME}")
    print("=" * 50)

    if reset:
        reset_collection()

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents/records")
    skipped = []
    kept = []
    for d in docs:
        if not (d.get("content") or "").strip():
            skipped.append({"reason": "empty_content", "metadata": d.get("metadata")})
        else:
            kept.append(d)

    chunks = chunk_documents(kept)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    _write_indexing_report(kept, chunks, skipped)
    print("[OK] Indexed to vector store")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()
    run_pipeline(reset=not args.no_reset)
