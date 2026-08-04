# Baseline Test Results (before IELTS adaptation)

**Command:** `.venv\Scripts\python.exe -m pytest tests -v`  
**Date:** 2026-08-04  
**Repository root:** `C:\Users\Legion\Documents\THUCHANH-AITHUCCHIEN-E403\K3-Day08-RAG-Pipeline-NhomGiCungDuoc`

## Summary

| Result | Count |
|--------|------:|
| Passed | 7 |
| Failed | 4 |
| Skipped | 24 |
| Total | 35 |

## Failed tests (expected — starter stubs / missing data)

1. `TestTask1::test_minimum_3_legal_files` — `data/landing/legal/` empty (PDFs only at `data/landing/` root)
2. `TestTask2::test_minimum_5_news_files` — no crawled article JSON yet
3. `TestTask3::test_has_markdown_files` — no standardized markdown yet
4. `TestTask3::test_legal_and_news_both_converted` — no converted outputs yet

## Passed / skipped

- Directory existence checks for legal/news/standardized passed (empty placeholders present).
- Tasks 4–10 function tests mostly skipped (`NotImplementedError`) or failed import soft-skip.

## Notes

- Existing IELTS PDFs were already present under `data/landing/` but not copied into `legal/`.
- Functional modules were still starter stubs for University Services topic.
