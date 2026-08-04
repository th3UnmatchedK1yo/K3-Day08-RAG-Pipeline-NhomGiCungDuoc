#!/usr/bin/env python
"""
Test PDF text extraction before building a RAG pipeline.

What it does:
1. Extracts text page-by-page using both pypdf and PyMuPDF.
2. Keeps the longer extraction result for each page.
3. Marks pages with little/no extractable text.
4. Saves low-text pages as PNG images for manual inspection.
5. Writes:
   - combined.md
   - pages/page_XXX.md
   - low_text_pages/page_XXX.png
   - extraction_report.json

Usage:
    python test_pdf_extract.py "path/to/file.pdf"
    python test_pdf_extract.py "path/to/file.pdf" --output data/standardized/pdf_test
    python test_pdf_extract.py "path/to/file.pdf" --min-chars 80
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Normalize extracted text without aggressively changing content."""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_with_pypdf(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    results: list[str] = []

    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = f"[PYPDF_EXTRACTION_ERROR: {exc}]"
        results.append(clean_text(text))

    return results


def extract_with_pymupdf(pdf_path: Path) -> list[str]:
    document = fitz.open(str(pdf_path))
    results: list[str] = []

    try:
        for page in document:
            try:
                text = page.get_text("text") or ""
            except Exception as exc:
                text = f"[PYMUPDF_EXTRACTION_ERROR: {exc}]"
            results.append(clean_text(text))
    finally:
        document.close()

    return results


def render_page_as_png(
    document: fitz.Document,
    page_index: int,
    output_path: Path,
    zoom: float = 2.0,
) -> None:
    page = document.load_page(page_index)
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    pixmap.save(str(output_path))


def choose_best_text(pypdf_text: str, pymupdf_text: str) -> tuple[str, str]:
    """Select the extraction result containing more usable characters."""
    pypdf_len = len(pypdf_text.strip())
    pymupdf_len = len(pymupdf_text.strip())

    if pymupdf_len > pypdf_len:
        return pymupdf_text, "pymupdf"

    return pypdf_text, "pypdf"


def build_output_dir(pdf_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()

    return (
        Path.cwd()
        / "output_extracted"
        / pdf_path.stem
    ).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test text extraction quality from a PDF before RAG indexing."
    )
    parser.add_argument("pdf", help="Path to the PDF file.")
    parser.add_argument(
        "--output",
        help="Output directory. Default: ./output_extracted/<pdf-name>/",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="Pages with fewer characters are considered low-text. Default: 80.",
    )
    parser.add_argument(
        "--render-all",
        action="store_true",
        help="Render every page to PNG, not only low-text pages.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()

    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}", file=sys.stderr)
        return 1

    if pdf_path.suffix.lower() != ".pdf":
        print(f"ERROR: Expected a PDF file: {pdf_path}", file=sys.stderr)
        return 1

    output_dir = build_output_dir(pdf_path, args.output)
    pages_dir = output_dir / "pages"
    images_dir = output_dir / "low_text_pages"

    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"PDF: {pdf_path}")
    print(f"Output: {output_dir}")
    print("Extracting with pypdf...")
    pypdf_pages = extract_with_pypdf(pdf_path)

    print("Extracting with PyMuPDF...")
    pymupdf_pages = extract_with_pymupdf(pdf_path)

    page_count = max(len(pypdf_pages), len(pymupdf_pages))
    report: dict[str, Any] = {
        "source_pdf": str(pdf_path),
        "page_count": page_count,
        "min_chars_threshold": args.min_chars,
        "pages": [],
    }

    combined_sections: list[str] = [
        f"# Extracted text: {pdf_path.name}",
        "",
        "> Automatically extracted for inspection before RAG indexing.",
        "> Low-text pages may contain scans, handwriting, charts, or image-only content.",
        "",
    ]

    document = fitz.open(str(pdf_path))

    try:
        for index in range(page_count):
            pypdf_text = pypdf_pages[index] if index < len(pypdf_pages) else ""
            pymupdf_text = pymupdf_pages[index] if index < len(pymupdf_pages) else ""

            best_text, selected_engine = choose_best_text(
                pypdf_text,
                pymupdf_text,
            )

            page_number = index + 1
            char_count = len(best_text)
            low_text = char_count < args.min_chars

            page_header = f"# Page {page_number}"
            page_body = best_text if best_text else "[NO EXTRACTABLE TEXT]"
            page_markdown = (
                f"{page_header}\n\n"
                f"<!-- extraction_engine: {selected_engine} -->\n"
                f"<!-- character_count: {char_count} -->\n\n"
                f"{page_body}\n"
            )

            page_md_path = pages_dir / f"page_{page_number:03d}.md"
            page_md_path.write_text(page_markdown, encoding="utf-8")

            image_path: str | None = None
            if low_text or args.render_all:
                png_path = images_dir / f"page_{page_number:03d}.png"
                render_page_as_png(document, index, png_path)
                image_path = str(png_path)

            combined_sections.extend(
                [
                    f"## Page {page_number}",
                    "",
                    f"<!-- extraction_engine: {selected_engine} -->",
                    f"<!-- character_count: {char_count} -->",
                    "",
                    page_body,
                    "",
                ]
            )

            report["pages"].append(
                {
                    "page_number": page_number,
                    "selected_engine": selected_engine,
                    "selected_character_count": char_count,
                    "pypdf_character_count": len(pypdf_text),
                    "pymupdf_character_count": len(pymupdf_text),
                    "low_text": low_text,
                    "page_markdown": str(page_md_path),
                    "rendered_image": image_path,
                }
            )

            status = "LOW TEXT / IMAGE?" if low_text else "OK"
            print(
                f"Page {page_number:03d}: "
                f"{char_count:5d} chars | {selected_engine:8s} | {status}"
            )
    finally:
        document.close()

    combined_path = output_dir / "combined.md"
    combined_path.write_text(
        "\n".join(combined_sections),
        encoding="utf-8",
    )

    report["summary"] = {
        "normal_text_pages": sum(
            1 for page in report["pages"] if not page["low_text"]
        ),
        "low_text_pages": sum(
            1 for page in report["pages"] if page["low_text"]
        ),
        "combined_markdown": str(combined_path),
    }

    report_path = output_dir / "extraction_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Combined Markdown: {combined_path}")
    print(f"Report: {report_path}")
    print(f"Low-text page images: {images_dir}")
    print(
        "\nImportant: Do not index low-text pages blindly. "
        "Inspect the PNG files first."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
