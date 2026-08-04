"""
Task 1 — Validate existing official IELTS Writing PDF/DOCX documents.

The source PDFs are already placed under data/landing/. This module:
1. Discovers PDFs recursively under data/landing/
2. Copies them into data/landing/legal/ for test compatibility (non-destructive)
3. Validates readability (header, size, page count)
4. Writes data/inspection/source_validation.json
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_ROOT / "data" / "landing"
DATA_DIR = LANDING_DIR / "legal"
INSPECTION_DIR = PROJECT_ROOT / "data" / "inspection"

IGNORE_DIR_NAMES = {"news", "standardized", "inspection"}
VALID_EXTENSIONS = {".pdf", ".docx", ".doc"}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directory ready: {DATA_DIR}")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_source_files(landing_dir: Path = LANDING_DIR) -> list[Path]:
    """Recursively find PDF/DOCX under data/landing/, ignoring news/."""
    found: list[Path] = []
    if not landing_dir.exists():
        return found

    for path in landing_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        # Skip nested ignored directories (news/, etc.)
        rel_parts = {p.lower() for p in path.relative_to(landing_dir).parts[:-1]}
        if rel_parts & IGNORE_DIR_NAMES:
            continue
        found.append(path)

    return sorted(found, key=lambda p: str(p).lower())


def ensure_legal_compatibility_copies(source_files: list[Path] | None = None) -> list[str]:
    """
    Non-destructively copy top-level landing PDFs into data/landing/legal/
    for tests that require files specifically under legal/.
    """
    setup_directory()
    if source_files is None:
        source_files = discover_source_files()

    # Prefer originals that are directly under landing/ (not already in legal/)
    top_level = [
        p for p in source_files
        if p.parent.resolve() == LANDING_DIR.resolve()
    ]
    # If none at top level, use any discovered outside legal/
    if not top_level:
        top_level = [
            p for p in source_files
            if p.parent.resolve() != DATA_DIR.resolve()
        ]

    copied: list[str] = []
    for src in top_level:
        dest = DATA_DIR / src.name
        if dest.exists():
            if _file_sha256(src) == _file_sha256(dest):
                continue
            # Different content with same name — keep existing, do not overwrite
            print(f"  [WARN] Skip copy (different hash exists): {dest.name}")
            continue
        shutil.copy2(src, dest)
        copied.append(src.name)
        print(f"  [OK] Copied for test compatibility: {src.name} -> legal/")

    if copied:
        print(f"Compatibility copies completed: {len(copied)} file(s)")
    else:
        print("No new compatibility copies needed.")
    return copied


def _validate_pdf(path: Path) -> dict:
    result = {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "file_size": path.stat().st_size,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "empty": path.stat().st_size == 0,
        "valid_pdf_header": False,
        "page_count": None,
        "readable": False,
        "errors": [],
    }

    try:
        with path.open("rb") as f:
            header = f.read(5)
        result["valid_pdf_header"] = header == b"%PDF-"
        if not result["valid_pdf_header"]:
            result["errors"].append("Invalid PDF header")
    except Exception as exc:
        result["errors"].append(f"Header read error: {exc}")

    page_count = None
    if fitz is not None:
        try:
            doc = fitz.open(str(path))
            page_count = doc.page_count
            doc.close()
        except Exception as exc:
            result["errors"].append(f"PyMuPDF error: {exc}")

    if page_count is None and PdfReader is not None:
        try:
            reader = PdfReader(str(path))
            page_count = len(reader.pages)
        except Exception as exc:
            result["errors"].append(f"pypdf error: {exc}")

    result["page_count"] = page_count
    result["readable"] = (
        not result["empty"]
        and result["valid_pdf_header"]
        and isinstance(page_count, int)
        and page_count > 0
        and result["file_size"] > 1024
    )
    return result


def _validate_docx(path: Path) -> dict:
    size = path.stat().st_size
    result = {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "file_size": size,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "empty": size == 0,
        "valid_pdf_header": None,
        "page_count": None,
        "readable": size > 1024,
        "errors": [] if size > 1024 else ["File too small or empty"],
    }
    return result


def validate_source_file(path: Path) -> dict:
    if path.suffix.lower() == ".pdf":
        return _validate_pdf(path)
    return _validate_docx(path)


def validate_all_sources() -> dict:
    """Validate at least three readable official documents and write report."""
    setup_directory()
    copied = ensure_legal_compatibility_copies()

    all_sources = discover_source_files()
    # Prefer unique filenames: legal copies + any others not duplicated by name
    by_name: dict[str, Path] = {}
    for p in all_sources:
        by_name.setdefault(p.name, p)
    unique_sources = list(by_name.values())

    validations = [validate_source_file(p) for p in unique_sources]
    readable = [v for v in validations if v.get("readable")]

    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "topic": "IELTS Writing Band Descriptors and Examiner Feedback Assistant",
        "landing_dir": str(LANDING_DIR),
        "legal_dir": str(DATA_DIR),
        "compatibility_copies": copied,
        "total_discovered": len(unique_sources),
        "readable_count": len(readable),
        "minimum_required": 3,
        "passed": len(readable) >= 3,
        "files": validations,
    }

    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INSPECTION_DIR / "source_validation.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Task 1 — Source Validation Summary")
    print("=" * 60)
    print(f"Discovered: {len(unique_sources)}")
    print(f"Readable:   {len(readable)}")
    print(f"Passed ≥3:  {report['passed']}")
    for v in validations:
        status = "OK" if v.get("readable") else "FAIL"
        pages = v.get("page_count")
        print(
            f"  [{status}] {v['filename']} | {v['file_size']} bytes"
            + (f" | {pages} pages" if pages is not None else "")
        )
    print(f"\n[OK] Wrote: {out_path}")
    return report


if __name__ == "__main__":
    report = validate_all_sources()
    if not report["passed"]:
        raise SystemExit(1)
