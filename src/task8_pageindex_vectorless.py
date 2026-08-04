"""
Task 8 — Optional PageIndex vectorless RAG adapter.

CLI:
  python -m src.task8_pageindex_vectorless --prepare
  python -m src.task8_pageindex_vectorless --upload
  python -m src.task8_pageindex_vectorless --status
  python -m src.task8_pageindex_vectorless --test-query "..."

Never upload on Streamlit startup. Never print API keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .env_utils import classify_external_error, get_env, is_configured, load_repo_env

load_repo_env()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
CORPUS_JSONL = STANDARDIZED_DIR / "ielts" / "corpus.jsonl"
UPLOAD_DIR = PROJECT_ROOT / "data" / "pageindex_upload"
INSPECTION_DIR = PROJECT_ROOT / "data" / "inspection"
MANIFEST_PATH = INSPECTION_DIR / "pageindex_manifest.json"

PREPARE_GROUPS = {
    "ielts_official_scoring.md": {"official_scoring", "assessment_criterion", "band_descriptor"},
    "ielts_examiner_comments.md": {"examiner_comment", "examiner_sample"},
    "ielts_teaching_guidance.md": {"teaching_guidance"},
    "ielts_web_guidance.md": {"web_guidance", "practice_question"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pageindex_enabled() -> bool:
    return get_env("PAGEINDEX_ENABLED", "true").lower() in {"1", "true", "yes"}


def _api_key() -> str:
    return get_env("PAGEINDEX_API_KEY")


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "documents" in data:
                return data
            # migrate old list format
            if isinstance(data, list):
                docs = {}
                for item in data:
                    name = item.get("file") or item.get("filename")
                    if name:
                        docs[name] = {
                            "sha256": item.get("sha256"),
                            "doc_id": item.get("doc_id"),
                            "upload_type": item.get("upload_type", "markdown"),
                            "status": item.get("status") or ("completed" if item.get("doc_id") else "failed"),
                            "retrieval_ready": bool(item.get("retrieval_ready")),
                            "uploaded_at": item.get("uploaded_at"),
                            "last_checked_at": item.get("last_checked_at"),
                        }
                return {"version": 1, "documents": docs}
        except Exception:
            pass
    return {"version": 1, "documents": {}}


def _save_manifest(manifest: dict) -> None:
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_documents() -> list[Path]:
    """Build machine-readable PageIndex upload files from corpus.jsonl. No API key required."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[str]] = {name: [] for name in PREPARE_GROUPS}

    if not CORPUS_JSONL.exists():
        print(f"[WARN] Missing corpus: {CORPUS_JSONL}")
        return []

    with CORPUS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            if rec.get("content_type") == "typed_candidate_response" and not rec.get("candidate_text_available"):
                continue
            source_type = rec.get("source_type") or ""
            content_type = rec.get("content_type") or ""
            target = None
            for fname, keys in PREPARE_GROUPS.items():
                if source_type in keys or content_type in keys:
                    target = fname
                    break
            if target is None:
                if source_type == "web_guidance" or rec.get("source_url"):
                    target = "ielts_web_guidance.md"
                else:
                    target = "ielts_teaching_guidance.md"

            title = rec.get("source_title") or rec.get("doc_id") or "Untitled"
            block = [
                f"# {title}",
                "",
                f"- Source organisation: {rec.get('source_org')}",
                f"- Content type: {content_type}",
                f"- Source type: {source_type}",
                f"- Task: {rec.get('task_number')}",
                f"- Band: {rec.get('band')}",
                f"- Criterion: {rec.get('criterion')}",
                f"- Original page: {rec.get('page_start')}",
                f"- Source URL: {rec.get('source_url')}",
                f"- Record ID: {rec.get('record_id')}",
                "",
                text,
                "",
                "---",
                "",
            ]
            buckets[target].append("\n".join(block))

    written: list[Path] = []
    for fname, parts in buckets.items():
        path = UPLOAD_DIR / fname
        if not parts:
            # keep an empty stub note so CLI is deterministic
            path.write_text(
                f"# {fname}\n\nNo eligible records for this group.\n",
                encoding="utf-8",
            )
        else:
            path.write_text("".join(parts), encoding="utf-8")
        written.append(path)
        print(f"  [OK] Prepared {path.name} ({path.stat().st_size} bytes)")
    print(f"[OK] Prepared {len(written)} files in {UPLOAD_DIR}")
    return written


def _get_client():
    if not is_configured("PAGEINDEX_API_KEY"):
        raise RuntimeError("key_missing")
    from pageindex.client import PageIndexClient

    base = get_env("PAGEINDEX_BASE_URL", "https://api.pageindex.ai") or "https://api.pageindex.ai"
    # SDK may ignore base_url; keep for future / HTTP fallback
    client = PageIndexClient(api_key=_api_key())
    client._configured_base_url = base  # type: ignore[attr-defined]
    return client


def upload_documents() -> dict:
    """Upload prepared markdown files with hash-based dedupe."""
    if not is_configured("PAGEINDEX_API_KEY"):
        print("[WARN] PAGEINDEX_API_KEY not configured — skip upload.")
        return _load_manifest()

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not any(UPLOAD_DIR.glob("*.md")):
        prepare_documents()

    manifest = _load_manifest()
    docs = manifest.setdefault("documents", {})
    try:
        client = _get_client()
    except Exception as exc:
        print(f"[WARN] PageIndex client unavailable: {classify_external_error(exc)}")
        return manifest

    for path in sorted(UPLOAD_DIR.glob("*.md")):
        digest = _sha256_file(path)
        existing = docs.get(path.name) or {}
        if existing.get("sha256") == digest and existing.get("doc_id"):
            print(f"  [SKIP] Unchanged hash reuse: {path.name} -> {existing.get('doc_id')}")
            existing["last_checked_at"] = _now()
            docs[path.name] = existing
            continue
        try:
            resp = client.submit_document(str(path))
            doc_id = None
            if isinstance(resp, dict):
                doc_id = resp.get("doc_id") or resp.get("id") or resp.get("document_id")
            docs[path.name] = {
                "sha256": digest,
                "doc_id": doc_id,
                "upload_type": "markdown",
                "status": "processing" if doc_id else "failed",
                "retrieval_ready": False,
                "uploaded_at": _now(),
                "last_checked_at": _now(),
            }
            print(f"  [OK] Uploaded: {path.name} -> {doc_id}")
        except Exception as exc:
            category = classify_external_error(exc)
            docs[path.name] = {
                "sha256": digest,
                "doc_id": existing.get("doc_id"),
                "upload_type": "markdown",
                "status": "failed",
                "retrieval_ready": False,
                "uploaded_at": existing.get("uploaded_at"),
                "last_checked_at": _now(),
                "error_category": category,
            }
            print(f"  [FAIL] {path.name}: {category}")
            if category in {"authentication"}:
                break

    _save_manifest(manifest)
    print(f"[OK] Wrote manifest: {MANIFEST_PATH}")
    return manifest


def check_status(poll: bool = True) -> dict:
    """Poll PageIndex processing status with finite timeout."""
    manifest = _load_manifest()
    if not is_configured("PAGEINDEX_API_KEY"):
        print("[WARN] PAGEINDEX_API_KEY not configured")
        return manifest

    try:
        client = _get_client()
    except Exception as exc:
        print(f"[WARN] {classify_external_error(exc)}")
        return manifest

    interval = float(get_env("PAGEINDEX_POLL_INTERVAL_SECONDS", "3") or "3")
    timeout = float(get_env("PAGEINDEX_PROCESSING_TIMEOUT_SECONDS", "300") or "300")
    deadline = time.time() + timeout

    docs = manifest.get("documents") or {}
    pending = [k for k, v in docs.items() if v.get("doc_id") and not v.get("retrieval_ready") and v.get("status") != "failed"]

    while True:
        for name in list(pending):
            info = docs[name]
            doc_id = info.get("doc_id")
            try:
                ready = False
                if hasattr(client, "is_retrieval_ready"):
                    ready = bool(client.is_retrieval_ready(doc_id))
                else:
                    meta = client.get_document(doc_id)
                    status = str((meta or {}).get("status") or "").lower()
                    ready = status in {"completed", "ready", "success"}
                info["last_checked_at"] = _now()
                if ready:
                    info["status"] = "completed"
                    info["retrieval_ready"] = True
                    print(f"  [OK] Ready: {name}")
                    pending.remove(name)
                else:
                    info["status"] = "processing"
                    print(f"  ... processing: {name}")
            except Exception as exc:
                category = classify_external_error(exc)
                info["last_checked_at"] = _now()
                if category in {"authentication", "invalid_endpoint"}:
                    info["status"] = "failed"
                    info["error_category"] = category
                    if name in pending:
                        pending.remove(name)
                    print(f"  [FAIL] {name}: {category}")
                else:
                    print(f"  [WARN] status check {name}: {category}")
            docs[name] = info

        _save_manifest(manifest)
        if not pending or not poll:
            break
        if time.time() >= deadline:
            print("[WARN] PageIndex processing timeout")
            for name in pending:
                docs[name]["status"] = docs[name].get("status") or "processing"
                docs[name]["error_category"] = "timeout"
            _save_manifest(manifest)
            break
        time.sleep(interval)

    completed = sum(1 for v in docs.values() if v.get("retrieval_ready"))
    print(f"[OK] retrieval-ready documents: {completed}/{len(docs)}")
    return manifest


def _flatten_relevant_contents(contents: Any) -> list[dict]:
    items: list[dict] = []
    if contents is None:
        return items
    if isinstance(contents, dict):
        items.append(contents)
        return items
    if isinstance(contents, list):
        for entry in contents:
            if isinstance(entry, list):
                items.extend(_flatten_relevant_contents(entry))
            elif isinstance(entry, dict):
                items.append(entry)
            elif isinstance(entry, str) and entry.strip():
                items.append({"relevant_content": entry})
    return items


def _content_hash(text: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", (text or "").strip().lower()).encode("utf-8")).hexdigest()


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Optional vectorless retrieval. Returns [] safely when unavailable."""
    if not _pageindex_enabled():
        print("[WARN] PAGEINDEX_ENABLED=false")
        return []
    if not is_configured("PAGEINDEX_API_KEY"):
        print("[WARN] PAGEINDEX_API_KEY not configured — PageIndex fallback disabled.")
        return []
    if not MANIFEST_PATH.exists():
        print("[WARN] pageindex_manifest.json missing — run --prepare/--upload first.")
        return []

    manifest = _load_manifest()
    ready_docs = [
        (name, info)
        for name, info in (manifest.get("documents") or {}).items()
        if info.get("doc_id") and info.get("retrieval_ready")
    ]
    if not ready_docs:
        print("[WARN] No retrieval-ready PageIndex documents.")
        return []

    try:
        client = _get_client()
    except Exception as exc:
        print(f"[WARN] PageIndex unavailable: {classify_external_error(exc)}")
        return []

    retrieval_timeout = float(get_env("PAGEINDEX_RETRIEVAL_TIMEOUT_SECONDS", "120") or "120")
    poll_interval = float(get_env("PAGEINDEX_POLL_INTERVAL_SECONDS", "3") or "3")
    results: list[dict] = []
    seen: set[str] = set()

    for name, info in ready_docs[:4]:
        doc_id = info["doc_id"]
        try:
            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = None
            retrieval = resp
            if isinstance(resp, dict):
                retrieval_id = resp.get("retrieval_id") or resp.get("id")
            if retrieval_id and hasattr(client, "get_retrieval"):
                deadline = time.time() + retrieval_timeout
                while time.time() < deadline:
                    retrieval = client.get_retrieval(retrieval_id)
                    status = str((retrieval or {}).get("status") or "").lower()
                    if status in {"completed", "ready", "success"} or retrieval.get("retrieved_nodes"):
                        break
                    if status in {"failed", "error"}:
                        break
                    time.sleep(poll_interval)
            nodes = []
            if isinstance(retrieval, dict):
                nodes = retrieval.get("retrieved_nodes") or retrieval.get("nodes") or []
            for node in nodes:
                flat = _flatten_relevant_contents(node.get("relevant_contents"))
                for item in flat:
                    text = (
                        item.get("relevant_content")
                        or item.get("content")
                        or item.get("text")
                        or ""
                    ).strip()
                    if not text:
                        continue
                    page = item.get("page_index") or item.get("page") or node.get("page")
                    section = item.get("section_title") or item.get("section") or name
                    node_id = str(node.get("node_id") or node.get("id") or len(results) + 1)
                    dedupe = f"{doc_id}|{node_id}|{page}|{_content_hash(text)}"
                    if dedupe in seen:
                        continue
                    seen.add(dedupe)
                    results.append(
                        {
                            "content": text,
                            "score": 0.0,  # assigned globally below as 1.0/rank
                            "metadata": {
                                "source": name,
                                "source_title": name.replace(".md", "").replace("_", " "),
                                "section": section,
                                "page": page,
                                "node_id": node_id,
                                "doc_id": doc_id,
                                "retrieval_provider": "pageindex",
                                "score_type": "rank_based",
                            },
                            "source": "pageindex",
                        }
                    )
        except Exception as exc:
            print(f"[WARN] PageIndex query failed for {name}: {classify_external_error(exc)}")
            continue

    # Deterministic display score only — never compare to SCORE_THRESHOLD
    for rank, item in enumerate(results, start=1):
        item["score"] = 1.0 / rank
        item.setdefault("metadata", {})["score_type"] = "rank_based"
    return results[:top_k]


def pageindex_status_summary() -> dict[str, Any]:
    manifest = _load_manifest()
    docs = manifest.get("documents") or {}
    prepared = len(list(UPLOAD_DIR.glob("*.md"))) if UPLOAD_DIR.exists() else 0
    uploaded = sum(1 for v in docs.values() if v.get("doc_id"))
    completed = sum(1 for v in docs.values() if v.get("retrieval_ready"))
    return {
        "enabled": _pageindex_enabled(),
        "key_configured": is_configured("PAGEINDEX_API_KEY"),
        "prepared_count": prepared,
        "uploaded_count": uploaded,
        "completed_count": completed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PageIndex prepare/upload/status/search")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--test-query", type=str, default="")
    args = parser.parse_args()

    if args.prepare:
        prepare_documents()
    if args.upload:
        upload_documents()
    if args.status:
        check_status(poll=True)
    if args.test_query:
        for r in pageindex_search(args.test_query, top_k=3):
            print(f"[{r['score']:.4f}][{r['metadata'].get('score_type')}] {r['content'][:120]}...")
    if not any([args.prepare, args.upload, args.status, args.test_query]):
        parser.print_help()
