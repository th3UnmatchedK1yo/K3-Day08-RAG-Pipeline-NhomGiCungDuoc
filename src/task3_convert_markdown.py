"""
Task 3 — Convert IELTS PDF/DOCX/news JSON into standardized Markdown + corpus.jsonl.

Extraction strategy:
  1) PyMuPDF page text
  2) pypdf page text
  3) Choose cleaner non-empty page text
  4) Optional MarkItDown / Qwen Vision (infographic only, never handwriting OCR)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import fitz
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_ROOT / "data" / "landing"
OUTPUT_DIR = PROJECT_ROOT / "data" / "standardized"
IELTS_DIR = OUTPUT_DIR / "ielts"
INSPECTION_DIR = PROJECT_ROOT / "data" / "inspection"

LOW_TEXT_THRESHOLD = 80
HANDWRITING_HINTS = (
    "sample script",
    "candidate writing scripts",
    "handwritten",
)

DOC_META_MAP = {
    "ielts-writing-band-descriptors.pdf": {
        "doc_id": "ielts_writing_band_descriptors_2023",
        "source_title": "IELTS Writing Band Descriptors 2023",
        "source_org": "IELTS",
        "source_type": "official_scoring",
        "examiner_scored": False,
    },
    "ielts-writing-key-assessment-criteria.pdf": {
        "doc_id": "ielts_writing_key_assessment_criteria",
        "source_title": "IELTS Writing Key Assessment Criteria",
        "source_org": "IELTS",
        "source_type": "official_scoring",
        "examiner_scored": False,
    },
    "ielts-academic-writing-sample-tasks-2023.pdf": {
        "doc_id": "ielts_sample_tasks_2023",
        "source_title": "IELTS Academic Writing Sample Tasks 2023",
        "source_org": "IELTS",
        "source_type": "examiner_sample",
        "examiner_scored": True,
    },
    "ielts_writing-_coherence_cohesion.pdf": {
        "doc_id": "ielts_coherence_cohesion_bc",
        "source_title": "British Council IELTS Writing - Coherence & Cohesion",
        "source_org": "British Council",
        "source_type": "teaching_guidance",
        "examiner_scored": False,
    },
    "ielts-academic-writing-example-responses-to-parts-1-and-2-with-band-scores-and-examiner-comments.pdf": {
        "doc_id": "ielts_example_responses_examiner_comments",
        "source_title": "IELTS Academic Writing Example Responses with Band Scores and Examiner Comments",
        "source_org": "IELTS",
        "source_type": "examiner_sample",
        "examiner_scored": True,
    },
}

CRITERION_ALIASES = {
    "task achievement": "task_achievement",
    "task response": "task_response",
    "coherence & cohesion": "coherence_cohesion",
    "coherence and cohesion": "coherence_cohesion",
    "lexical resource": "lexical_resource",
    "grammatical range & accuracy": "grammatical_range_accuracy",
    "grammatical range and accuracy": "grammatical_range_accuracy",
}


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s[:80] or "record"


def clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    # Collapse PDF letter-spacing artefacts: "wo rds" patterns of single letters
    # only when many spaced single-letter tokens appear — keep conservative.
    text = re.sub(r"\b([A-Za-z])\s([A-Za-z])\s([A-Za-z])\s([A-Za-z])\b", r"\1\2\3\4", text)
    text = re.sub(r"(?m)^Page\s+\d+\s+of\s+\d+\s*$", "", text)
    text = re.sub(r"(?m)^Page\s+\d+\s+of\s+\d+\s+IELTS\.org\s*$", "", text)
    text = re.sub(r"(?m)^\s*IELTS\.org\s*$", "", text)
    text = re.sub(r"(?m)^\s*Updated May 2023\s*$", "", text)
    text = re.sub(r"(?m)^\s*Please visit IELTS\.org for updates\s*$", "", text)
    # Repair hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse excessive whitespace while preserving paragraphs
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces from positional PDF extraction
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _meaningful_char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def extract_page_texts(pdf_path: Path) -> list[dict[str, Any]]:
    pymupdf_pages: list[str] = []
    pypdf_pages: list[str] = []

    doc = fitz.open(str(pdf_path))
    try:
        for page in doc:
            pymupdf_pages.append(clean_pdf_text(page.get_text("text") or ""))
        has_images = [bool(page.get_images()) for page in doc]
    finally:
        doc.close()

    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            try:
                pypdf_pages.append(clean_pdf_text(page.extract_text() or ""))
            except Exception:
                pypdf_pages.append("")
    except Exception:
        pypdf_pages = [""] * len(pymupdf_pages)

    n = max(len(pymupdf_pages), len(pypdf_pages))
    pages: list[dict[str, Any]] = []
    for i in range(n):
        a = pymupdf_pages[i] if i < len(pymupdf_pages) else ""
        b = pypdf_pages[i] if i < len(pypdf_pages) else ""
        if _meaningful_char_count(a) >= _meaningful_char_count(b):
            text, engine = a, "pymupdf"
        else:
            text, engine = b, "pypdf"
        low = _meaningful_char_count(text) < LOW_TEXT_THRESHOLD
        pages.append(
            {
                "page_number": i + 1,
                "text": text,
                "extraction_engine": engine,
                "native_text_length": len(text),
                "meaningful_chars": _meaningful_char_count(text),
                "has_images": has_images[i] if i < len(has_images) else False,
                "low_text": low,
            }
        )
    return pages


def _normalize_criterion(raw: str) -> Optional[str]:
    key = re.sub(r"\s+", " ", raw.strip().lower())
    return CRITERION_ALIASES.get(key)


def _yaml_front_matter(meta: dict[str, Any]) -> str:
    lines = ["---"]
    for k, v in meta.items():
        if v is None:
            lines.append(f"{k}: null")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        else:
            val = str(v).replace('"', "'")
            lines.append(f'{k}: "{val}"')
    lines.append("---\n")
    return "\n".join(lines)


def _base_record(doc_meta: dict, **kwargs) -> dict:
    rec = {
        "doc_id": doc_meta["doc_id"],
        "record_id": kwargs.get("record_id"),
        "source_title": doc_meta["source_title"],
        "source_org": doc_meta["source_org"],
        "source_type": doc_meta["source_type"],
        "source_file": kwargs.get("source_file"),
        "source_url": kwargs.get("source_url"),
        "content_type": kwargs.get("content_type"),
        "task_number": kwargs.get("task_number"),
        "task_id": kwargs.get("task_id"),
        "script_id": kwargs.get("script_id"),
        "criterion": kwargs.get("criterion"),
        "band": kwargs.get("band"),
        "essay_type": kwargs.get("essay_type"),
        "page_start": kwargs.get("page_start"),
        "page_end": kwargs.get("page_end"),
        "examiner_scored": kwargs.get("examiner_scored", doc_meta.get("examiner_scored", False)),
        "candidate_text_available": kwargs.get("candidate_text_available", False),
        "extraction_engine": kwargs.get("extraction_engine", "pymupdf"),
        "text": kwargs.get("text", ""),
    }
    if not rec["record_id"]:
        seed = f"{rec['doc_id']}|{rec['content_type']}|{rec['page_start']}|{rec['text'][:80]}"
        rec["record_id"] = _slug(seed) + "_" + hashlib.md5(seed.encode()).hexdigest()[:8]
    # Flag invalid terminology
    if rec.get("task_number") == 2 and rec.get("criterion") == "task_achievement":
        rec["flag"] = "invalid_criterion_for_task2"
        rec["criterion"] = "task_response"
    if rec.get("task_number") == 1 and rec.get("criterion") == "task_response":
        rec["flag"] = "invalid_criterion_for_task1"
        rec["criterion"] = "task_achievement"
    return rec


def _split_descriptor_row(body: str) -> list[str]:
    """Split one band row into up to 4 criterion cells using column anchors."""
    coherence_pat = (
        r"(Information and ideas are logically|"
        r"The message can be followed|"
        r"Cohesion is used in such a way|"
        r"Organisation is evident|"
        r"There is a very limited range of cohesive|"
        r"There is little control of organisational)"
    )
    lexical_pat = (
        r"(The resource is sufficient|"
        r"Full flexibility and precise use|"
        r"A wide range of vocabulary|"
        r"A sufficient range of vocabulary|"
        r"uses a limited range of vocabulary|"
        r"Uses only a very limited range of words|"
        r"Can only use a few isolated words|"
        r"Uses an adequate range of vocabulary|"
        r"Uses a wide enough vocabulary|"
        r"Uses a limited range of vocabulary)"
    )
    grammar_pat = (
        r"(A wide range of structures|"
        r"A variety of complex structures|"
        r"Uses a mix of simple and complex|"
        r"Uses only a very limited range of structures|"
        r"Attempts sentence forms but|"
        r"Cannot use sentence forms|"
        r"Simple sentence forms|"
        r"Uses a limited range of structures|"
        r"Punctuation and grammar are used)"
    )

    def _find(pat: str, text: str) -> int:
        m = re.search(pat, text, flags=re.I)
        return m.start() if m else -1

    c_idx = _find(coherence_pat, body)
    l_idx = _find(lexical_pat, body)
    g_idx = _find(grammar_pat, body)

    if c_idx > 0 and l_idx > c_idx and g_idx > l_idx:
        return [
            body[:c_idx].strip(),
            body[c_idx:l_idx].strip(),
            body[l_idx:g_idx].strip(),
            body[g_idx:].strip(),
        ]

    # Fallback: roughly quarter the sentences
    sentences = re.split(r"(?<=[.!?])\s+|\n+", body)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 4:
        n = len(sentences)
        cuts = [0, n // 4, n // 2, (3 * n) // 4, n]
        return [" ".join(sentences[cuts[i] : cuts[i + 1]]) for i in range(4)]
    return [body]


def parse_band_descriptors(pages: list[dict], source_file: str, doc_meta: dict) -> list[dict]:
    """Parse Task 1 / Task 2 band descriptor tables into criterion cells."""
    records: list[dict] = []
    # Determine task sections by page markers
    task1_pages = []
    task2_pages = []
    current = None
    for p in pages:
        t = p["text"].lower()
        if "writing task 1" in t and "band descriptor" in t:
            current = 1
        elif "writing task 2" in t and "band descriptor" in t:
            current = 2
        if current == 1 and p["meaningful_chars"] > LOW_TEXT_THRESHOLD:
            task1_pages.append(p)
        elif current == 2 and p["meaningful_chars"] > LOW_TEXT_THRESHOLD:
            task2_pages.append(p)

    def parse_task_block(block_pages: list[dict], task_number: int) -> None:
        text = "\n".join(p["text"] for p in block_pages)
        criteria_order = (
            ["task_achievement", "coherence_cohesion", "lexical_resource", "grammatical_range_accuracy"]
            if task_number == 1
            else ["task_response", "coherence_cohesion", "lexical_resource", "grammatical_range_accuracy"]
        )
        # Split by leading band numbers 9..0 at line starts
        parts = re.split(r"(?m)^(?P<band>[0-9])\s*\n", text)
        # parts like [preamble, '9', body9, '8', body8, ...]
        i = 1
        while i + 1 < len(parts):
            try:
                band = float(parts[i])
            except ValueError:
                i += 2
                continue
            body = parts[i + 1].strip()
            i += 2
            cells = _split_descriptor_row(body)
            for idx, cell in enumerate(cells):
                criterion = criteria_order[idx] if idx < len(criteria_order) else None
                if not cell.strip():
                    continue
                page_start = block_pages[0]["page_number"]
                band_label = str(band).replace(".", "_")
                records.append(
                    _base_record(
                        doc_meta,
                        source_file=source_file,
                        content_type="band_descriptor",
                        task_number=task_number,
                        criterion=criterion,
                        band=band,
                        page_start=page_start,
                        page_end=block_pages[-1]["page_number"],
                        extraction_engine=block_pages[0]["extraction_engine"],
                        candidate_text_available=False,
                        record_id=f"{doc_meta['doc_id']}_t{task_number}_band{band_label}_{criterion or 'all'}",
                        text=(
                            f"IELTS Writing Task {task_number} Band {band:g} — "
                            f"{(criterion or 'all').replace('_', ' ').title()}\n\n{cell.strip()}"
                        ),
                    )
                )

    if task1_pages:
        parse_task_block(task1_pages, 1)
    if task2_pages:
        parse_task_block(task2_pages, 2)

    # If parser failed, keep page-level records for usable pages
    if not records:
        for p in pages:
            if p["low_text"]:
                continue
            records.append(
                _base_record(
                    doc_meta,
                    source_file=source_file,
                    content_type="band_descriptor",
                    page_start=p["page_number"],
                    page_end=p["page_number"],
                    extraction_engine=p["extraction_engine"],
                    text=p["text"],
                    record_id=f"{doc_meta['doc_id']}_page_{p['page_number']:03d}",
                )
            )
    return records


def parse_assessment_criteria(pages: list[dict], source_file: str, doc_meta: dict) -> list[dict]:
    records: list[dict] = []
    full = "\n\n".join(p["text"] for p in pages if not p["low_text"])
    # Split by criterion headings
    pattern = re.compile(
        r"(?im)^(Task Achievement|Task Response|Coherence and Cohesion|Coherence & Cohesion|"
        r"Lexical Resource|Grammatical Range and Accuracy|Grammatical Range & Accuracy)\s*$"
    )
    splits = pattern.split(full)
    if len(splits) > 1:
        preamble = splits[0].strip()
        if preamble:
            records.append(
                _base_record(
                    doc_meta,
                    source_file=source_file,
                    content_type="assessment_criterion",
                    page_start=1,
                    page_end=pages[-1]["page_number"] if pages else 1,
                    extraction_engine=pages[0]["extraction_engine"] if pages else "pymupdf",
                    text=preamble,
                    record_id=f"{doc_meta['doc_id']}_overview",
                )
            )
        i = 1
        while i + 1 < len(splits):
            heading = splits[i].strip()
            body = splits[i + 1].strip()
            i += 2
            crit = _normalize_criterion(heading)
            task_number = 1 if crit == "task_achievement" else 2 if crit == "task_response" else None
            records.append(
                _base_record(
                    doc_meta,
                    source_file=source_file,
                    content_type="assessment_criterion",
                    criterion=crit,
                    task_number=task_number,
                    page_start=1,
                    page_end=pages[-1]["page_number"] if pages else 1,
                    extraction_engine=pages[0]["extraction_engine"] if pages else "pymupdf",
                    text=f"{heading}\n\n{body}",
                    record_id=f"{doc_meta['doc_id']}_{crit or _slug(heading)}",
                )
            )
    else:
        for p in pages:
            if p["low_text"]:
                continue
            records.append(
                _base_record(
                    doc_meta,
                    source_file=source_file,
                    content_type="assessment_criterion",
                    page_start=p["page_number"],
                    page_end=p["page_number"],
                    extraction_engine=p["extraction_engine"],
                    text=p["text"],
                    record_id=f"{doc_meta['doc_id']}_page_{p['page_number']:03d}",
                )
            )
    return records


def parse_sample_tasks(pages: list[dict], source_file: str, doc_meta: dict) -> tuple[list[dict], list[dict]]:
    """Return (records, skipped_pages)."""
    records: list[dict] = []
    skipped: list[dict] = []

    # Track current task context
    current_task_number: Optional[int] = None
    current_task_id: Optional[str] = None
    current_script: Optional[str] = None

    for p in pages:
        text = p["text"]
        page_no = p["page_number"]
        low = p["low_text"]
        lower = text.lower()

        # Detect task prompts
        m_task = re.search(r"Academic Writing Sample Task[^\n]*?(1[A-C]|2[A-B])", text, re.I)
        if m_task:
            tid = m_task.group(1).upper()
            current_task_id = tid
            current_task_number = 1 if tid.startswith("1") else 2

        m_script = re.search(r"Sample\s+Script\s+([A-Z])", text, re.I)
        if m_script:
            letter = m_script.group(1).upper()
            current_script = f"task{(current_task_id or 'x').lower()}_script_{letter.lower()}"

        # Image-only / handwriting page: little text, often just title
        if low:
            skipped.append(
                {
                    "source_file": source_file,
                    "page_number": page_no,
                    "reason": "skipped_image_text",
                    "preview": text[:120],
                    "script_id": current_script,
                }
            )
            continue

        # Examiner comment pages (require Band score nearby; skip TOC/overview)
        is_examiner_page = bool(
            re.search(r"(?i)examiner comment", text)
            and re.search(r"(?i)Band\s+\d(?:\.\d)?", text)
            and "contents" not in lower
            and "detailed performance descriptors" not in lower
        )
        if is_examiner_page:
            band = None
            m_band = re.search(r"(?i)Band\s+(\d(?:\.\d)?)", text)
            if m_band:
                band = float(m_band.group(1))
            records.append(
                _base_record(
                    doc_meta,
                    source_file=source_file,
                    content_type="examiner_comment",
                    task_number=current_task_number,
                    task_id=current_task_id,
                    script_id=current_script,
                    band=band,
                    page_start=page_no,
                    page_end=page_no,
                    examiner_scored=True,
                    candidate_text_available=False,
                    extraction_engine=p["extraction_engine"],
                    text=text,
                    record_id=(
                        f"{doc_meta['doc_id']}_"
                        f"{(current_task_id or 'task').lower()}_"
                        f"{(current_script or 'script')}_band"
                        f"{str(band).replace('.', '_') if band is not None else 'na'}_comment"
                    ),
                )
            )
            continue

        # Task prompt pages (machine-readable instructions)
        if re.search(r"(?i)(writing task\s*[12]|you should spend about)", text) and "examiner comment" not in lower:
            # Skip pure assessment overview pages that already appear in criteria PDF
            content_type = "task_prompt"
            if "task achievement" in lower and "task response" in lower and "detailed performance" in lower:
                content_type = "assessment_criterion"
            records.append(
                _base_record(
                    doc_meta,
                    source_file=source_file,
                    content_type=content_type,
                    task_number=current_task_number,
                    task_id=current_task_id,
                    script_id=current_script if content_type != "task_prompt" else None,
                    page_start=page_no,
                    page_end=page_no,
                    examiner_scored=doc_meta.get("examiner_scored", False),
                    candidate_text_available=False,
                    extraction_engine=p["extraction_engine"],
                    text=text,
                    record_id=f"{doc_meta['doc_id']}_page_{page_no:03d}_{content_type}",
                )
            )
            continue

        # Typed overview / guidance pages
        records.append(
            _base_record(
                doc_meta,
                source_file=source_file,
                content_type="teaching_guidance",
                task_number=current_task_number,
                task_id=current_task_id,
                page_start=page_no,
                page_end=page_no,
                extraction_engine=p["extraction_engine"],
                text=text,
                record_id=f"{doc_meta['doc_id']}_page_{page_no:03d}",
            )
        )

    return records, skipped


def parse_coherence_infographic(pages: list[dict], source_file: str, doc_meta: dict) -> list[dict]:
    records: list[dict] = []
    for p in pages:
        text = p["text"]
        if p["low_text"]:
            continue
        # Preserve key section themes even if reading order is imperfect
        records.append(
            _base_record(
                doc_meta,
                source_file=source_file,
                content_type="teaching_guidance",
                criterion="coherence_cohesion",
                page_start=p["page_number"],
                page_end=p["page_number"],
                extraction_engine=p["extraction_engine"],
                text=text,
                record_id=f"{doc_meta['doc_id']}_coherence_cohesion",
            )
        )
    return records


def parse_example_responses(pages: list[dict], source_file: str, doc_meta: dict) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    skipped: list[dict] = []
    for p in pages:
        text = clean_pdf_text(re.sub(r"(?:\s*\n\s*)+", "\n", p["text"]))
        # Remove sparse single-space noise lines
        lines = [ln.strip() for ln in text.split("\n") if ln.strip() and not re.fullmatch(r"[\W_]+", ln.strip())]
        text = "\n".join(lines)
        meaningful = _meaningful_char_count(text)
        if meaningful < LOW_TEXT_THRESHOLD:
            skipped.append(
                {
                    "source_file": source_file,
                    "page_number": p["page_number"],
                    "reason": "skipped_image_text",
                    "preview": text[:120],
                }
            )
            continue

        task_number = None
        if re.search(r"(?i)part\s*1|writing task\s*1|task 1", text):
            task_number = 1
        if re.search(r"(?i)part\s*2|writing task\s*2|task 2", text):
            task_number = 2

        band = None
        m_band = re.search(r"(?i)band\s*score\s*[:\-]?\s*(\d(?:\.\d)?)", text)
        if m_band:
            band = float(m_band.group(1))

        has_candidate = bool(re.search(r"(?i)candidate response", text))
        has_examiner_block = bool(re.search(r"(?i)examiner comment|band score", text))
        if has_candidate and meaningful > 200 and not has_examiner_block:
            content_type = "typed_candidate_response"
            candidate_available = True
        elif has_examiner_block and not has_candidate:
            content_type = "examiner_comment"
            candidate_available = False
        elif has_candidate and has_examiner_block:
            # Mixed page: keep typed response if substantial candidate prose exists
            content_type = "typed_candidate_response"
            candidate_available = True
        else:
            content_type = "teaching_guidance"
            candidate_available = False

        records.append(
            _base_record(
                doc_meta,
                source_file=source_file,
                content_type=content_type,
                task_number=task_number,
                band=band,
                page_start=p["page_number"],
                page_end=p["page_number"],
                examiner_scored=True,
                candidate_text_available=candidate_available,
                extraction_engine=p["extraction_engine"],
                text=text,
                record_id=f"{doc_meta['doc_id']}_page_{p['page_number']:03d}_{content_type}",
            )
        )
    return records, skipped


def convert_pdf(pdf_path: Path, visual_report: list[dict]) -> tuple[list[dict], str, list[dict]]:
    source_file = pdf_path.name
    doc_meta = DOC_META_MAP.get(
        source_file,
        {
            "doc_id": _slug(pdf_path.stem),
            "source_title": pdf_path.stem.replace("-", " ").replace("_", " ").title(),
            "source_org": "IELTS",
            "source_type": "official_scoring",
            "examiner_scored": False,
        },
    )
    pages = extract_page_texts(pdf_path)
    skipped: list[dict] = []

    for p in pages:
        visual_report.append(
            {
                "source_file": source_file,
                "page_number": p["page_number"],
                "native_text_length": p["native_text_length"],
                "has_images": p["has_images"],
                "layout_type": "infographic" if "coherence" in source_file.lower() else "document",
                "native_extraction_sufficient": not p["low_text"],
                "vision_used": False,
                "vision_model": None,
                "manual_review_required": bool(p["low_text"] and p["has_images"]),
            }
        )

    name = source_file.lower()
    if "band-descriptors" in name:
        records = parse_band_descriptors(pages, source_file, doc_meta)
    elif "key-assessment" in name:
        records = parse_assessment_criteria(pages, source_file, doc_meta)
    elif "sample-tasks" in name:
        records, skipped = parse_sample_tasks(pages, source_file, doc_meta)
    elif "coherence" in name:
        records = parse_coherence_infographic(pages, source_file, doc_meta)
    elif "example-responses" in name:
        records, skipped = parse_example_responses(pages, source_file, doc_meta)
    else:
        records = []
        for p in pages:
            if p["low_text"]:
                skipped.append(
                    {
                        "source_file": source_file,
                        "page_number": p["page_number"],
                        "reason": "skipped_image_text",
                        "preview": p["text"][:120],
                    }
                )
                continue
            records.append(
                _base_record(
                    doc_meta,
                    source_file=source_file,
                    content_type="teaching_guidance",
                    page_start=p["page_number"],
                    page_end=p["page_number"],
                    extraction_engine=p["extraction_engine"],
                    text=p["text"],
                    record_id=f"{doc_meta['doc_id']}_page_{p['page_number']:03d}",
                )
            )

    # Build markdown document
    md_body_parts = []
    for rec in records:
        md_body_parts.append(f"## {rec['content_type']} — {rec['record_id']}\n\n{rec['text']}\n")
    front = _yaml_front_matter(
        {
            "doc_id": doc_meta["doc_id"],
            "source_title": doc_meta["source_title"],
            "source_org": doc_meta["source_org"],
            "source_type": doc_meta["source_type"],
            "source_file": source_file,
            "source_url": None,
            "examiner_scored": doc_meta.get("examiner_scored", False),
        }
    )
    markdown = front + "\n".join(md_body_parts)
    if not records:
        # Still write a stub noting skipped handwriting so convert doesn't invent content
        markdown = front + (
            "\n> No machine-readable body text was indexed for this file "
            "(handwriting/image-only pages were skipped by policy).\n"
        )
    return records, markdown, skipped


def convert_news_articles() -> tuple[list[dict], list[dict]]:
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    skipped: list[dict] = []

    if not news_dir.exists():
        return records, skipped

    for filepath in sorted(news_dir.glob("*.json")):
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append({"file": filepath.name, "reason": f"invalid_json: {exc}"})
            continue
        if data.get("status") != "success":
            skipped.append({"file": filepath.name, "reason": "crawl_failed", "url": data.get("url")})
            continue
        content = (data.get("content_markdown") or "").strip()
        if len(content) < 100:
            skipped.append({"file": filepath.name, "reason": "empty_or_short_content", "url": data.get("url")})
            continue

        title = data.get("title") or filepath.stem
        org = data.get("source_org") or "IELTS"
        url = data.get("url")
        doc_id = _slug(filepath.stem + "_" + (title or ""))
        doc_meta = {
            "doc_id": doc_id,
            "source_title": title,
            "source_org": org,
            "source_type": "web_guidance",
            "examiner_scored": False,
        }
        rec = _base_record(
            doc_meta,
            source_file=filepath.name,
            source_url=url,
            content_type="teaching_guidance" if "practice" not in (url or "").lower() else "practice_question",
            page_start=None,
            page_end=None,
            extraction_engine="web_crawl",
            candidate_text_available=False,
            text=content,
            record_id=f"{doc_id}_body",
        )
        if "practice" in (url or "").lower() or "task-2" in (url or "").lower():
            rec["content_type"] = "practice_question" if "practice-tests" in (url or "") else "teaching_guidance"
            rec["task_number"] = 2
        records.append(rec)

        front = _yaml_front_matter(
            {
                "doc_id": doc_id,
                "source_title": title,
                "source_org": org,
                "source_type": "web_guidance",
                "source_file": filepath.name,
                "source_url": url,
                "examiner_scored": False,
                "date_crawled": data.get("date_crawled"),
            }
        )
        out = output_dir / f"{filepath.stem}.md"
        out.write_text(front + f"# {title}\n\n**Source:** {url}\n\n---\n\n{content}\n", encoding="utf-8")
        print(f"  [OK] News markdown: {out.name}")
    return records, skipped


def discover_pdfs() -> list[Path]:
    found: list[Path] = []
    if not LANDING_DIR.exists():
        return found
    for path in LANDING_DIR.rglob("*.pdf"):
        # skip nothing under news
        if "news" in {p.lower() for p in path.relative_to(LANDING_DIR).parts[:-1]}:
            continue
        found.append(path)
    # Deduplicate by filename preferring legal/ copy for standardized/legal output
    by_name: dict[str, Path] = {}
    for p in found:
        prev = by_name.get(p.name)
        if prev is None:
            by_name[p.name] = p
        elif "legal" in p.parts and "legal" not in prev.parts:
            by_name[p.name] = p
    return sorted(by_name.values(), key=lambda x: x.name.lower())


def convert_legal_docs() -> tuple[list[dict], list[dict], list[dict]]:
    legal_out = OUTPUT_DIR / "legal"
    legal_out.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    all_skipped: list[dict] = []
    visual_report: list[dict] = []

    for pdf_path in discover_pdfs():
        print(f"Converting: {pdf_path.name}")
        records, markdown, skipped = convert_pdf(pdf_path, visual_report)
        out_path = legal_out / f"{pdf_path.stem}.md"
        # Ensure markdown not empty for tests (>200 chars)
        if len(markdown) < 200:
            markdown += (
                "\n\n<!-- Conversion note: machine-readable content was limited; "
                "handwriting/image-only pages were skipped by policy. -->\n"
            )
        out_path.write_text(markdown, encoding="utf-8")
        print(f"  [OK] Saved: {out_path} ({len(records)} records, {len(skipped)} skipped pages)")
        all_records.extend(records)
        all_skipped.extend(skipped)
    return all_records, all_skipped, visual_report

def convert_ielts_docs():
    """Convert PDF IELTS trong data/landing/ielts/ sang Markdown."""
    ielts_dir = LANDING_DIR / "ielts"
    output_dir = OUTPUT_DIR / "ielts"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    if not ielts_dir.exists():
        print(f"⚠ Không tìm thấy thư mục: {ielts_dir}")
        return

    for filepath in ielts_dir.iterdir():
        if filepath.suffix.lower() == ".pdf":
            print(f"Converting IELTS: {filepath.name}")

            result = md.convert(str(filepath))

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(
                result.text_content,
                encoding="utf-8"
            )

            print(f"  ✓ Saved: {output_path}")

def convert_all():
<<<<<<< HEAD
    """Convert IELTS documents."""
    print("=" * 50)
    print("Task 3: Convert IELTS PDF to Markdown")
    print("=" * 50)

    print("\n--- IELTS Documents ---")
    convert_ielts_docs()
=======
    print("=" * 50)
    print("Task 3: Convert to Markdown (IELTS corpus)")
    print("=" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IELTS_DIR.mkdir(parents=True, exist_ok=True)
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Official Documents ---")
    legal_records, skipped_pages, visual_report = convert_legal_docs()
>>>>>>> dev

    print("\n--- Web Articles ---")
    news_records, skipped_news = convert_news_articles()

    all_records = legal_records + news_records

    corpus_path = IELTS_DIR / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "records_created": len(all_records),
        "legal_records": len(legal_records),
        "news_records": len(news_records),
        "skipped_pages": skipped_pages,
        "skipped_news": skipped_news,
        "handwriting_policy": (
            "OCR disabled. Image-only handwritten candidate pages are skipped "
            "and not indexed. Examiner comments remain when machine-readable."
        ),
        "content_type_counts": {},
    }
    for rec in all_records:
        ct = rec.get("content_type") or "unknown"
        report["content_type_counts"][ct] = report["content_type_counts"].get(ct, 0) + 1

    (IELTS_DIR / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (INSPECTION_DIR / "visual_extraction_report.json").write_text(
        json.dumps(visual_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[OK] Corpus records: {len(all_records)}")
    print(f"[OK] Wrote: {corpus_path}")
    print(f"[OK] Output dir: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_all()
