# Final Implementation Report — IELTS Writing RAG Pipeline

## Team

| Thành viên | MSSV | Vai trò |
|-----------|------|---------|
| Võ Hà Minh Huy | 2A202601373 | Full RAG pipeline (Tasks 1–3, 5–10), Streamlit, DeepSeek, PageIndex, crawl, tích hợp |
| Nguyễn Minh Thái | 2A202601619 | Task 4 ChromaDB / vector database; tìm & chuẩn bị tài liệu PDF |
| Đỗ Duy Đông | 2A202601675 | Golden dataset + evaluation A/B; quản lý Git (merge / repo nhóm) |

Nhóm: **Nhóm Gì Cũng Được** — báo cáo phân công chi tiết: `group_project/README.md`

## Repository root

`C:\Users\Legion\Documents\THUCHANH-AITHUCCHIEN-E403\K3-Day08-RAG-Pipeline-NhomGiCungDuoc`

(also mapped as `R:\` via `subst` when needed for Windows long-path installs)

## Architecture preserved

| Layer | Status |
|---|---|
| Local embedding | **BAAI/bge-m3** via `sentence-transformers` (`EMBEDDING_BACKEND=local`) |
| Hugging Face token | Optional Hub download auth only — **does not** switch to remote inference |
| Vector store | Chroma collection **`ielts_writing_docs`** (cosine) |
| Lexical | Local BM25 |
| Fusion | Local RRF |
| Generation | **DeepSeek V4 Flash** (`deepseek-v4-flash`) via OpenAI-compatible client |
| Fallback | PageIndex (vectorless), only when best **dense cosine** &lt; threshold |

## Corpus and index (current)

| Metric | Value |
|---|---:|
| Standardized corpus records | **115** |
| Chroma chunks | **344** |
| Collection | `ielts_writing_docs` |
| Embedding model | `BAAI/bge-m3` (local, 1024-d) |
| Dense fallback threshold | **0.51** (recalibrated suggestion 0.509) |

Handwriting/image-only pages remain skipped (`skipped_image_text`). Examiner comments retained. No OCR.

## Web crawl results

**Successful web articles: 5 / 5**

| # | Source | Method | Status |
|---|---|---|---|
| 1 | IDP — Why can’t I get a Band 8? | cached (reused) | success |
| 2 | IDP — 8 steps to Band 8 | cached (reused) | success |
| 3 | BC — Writing Task 2 tips | crawl4ai | success |
| 4 | BC — How to write an English essay for IELTS | crawl4ai | success |
| 5 | BC — Academic Writing practice Task 2 (`/writing/academic/task-2`) | crawl4ai | success |

Failed web sources: **none** in the latest crawl.

British Council Stage-1 `requests` timed out; Stage-3 Crawl4AI succeeded. No fabricated content.

Report: `data/inspection/web_crawl_report.json`

## DeepSeek

| Check | Result |
|---|---|
| Key configured (sanitized) | true |
| Base URL | `https://api.deepseek.com` |
| Model | `deepseek-v4-flash` |
| Live connection test | **SUCCESS** (`DEEPSEEK_CONNECTION_OK`) |
| Script | `scripts/test_deepseek_connection.py` (not run inside pytest) |

When DeepSeek is unavailable, the app returns retrieval-only evidence with citations and states clearly that DeepSeek is unavailable — it does not pretend excerpts are a generated answer.

OpenRouter is no longer required for production generation.

## PageIndex

| Step | Result |
|---|---|
| Prepare (`--prepare`, no API key required) | **4 files** under `data/pageindex_upload/` |
| Upload | 4 documents (unchanged SHA-256 reused / already uploaded) |
| Status | **4/4 retrieval-ready** (`completed`) |
| Live retrieval test | **SUCCESS** (Band 8 Coherence and Cohesion query returned ranked nodes) |

Prepared files:

- `ielts_official_scoring.md`
- `ielts_examiner_comments.md`
- `ielts_teaching_guidance.md`
- `ielts_web_guidance.md`

Manifest: `data/inspection/pageindex_manifest.json`

PageIndex scores are **rank-based** (`score = 1.0 / rank`, `score_type=rank_based`) and are **never** compared to `SCORE_THRESHOLD`. Fallback gating uses original dense cosine only.

## Evaluation (PageIndex disabled during A/B)

| Metric | Config A (dense) | Config B (hybrid) |
|---|---:|---:|
| Exact source hit rate | 0.857 | 0.857 |
| Context recall | 0.952 | 0.952 |
| Context precision | 0.524 | 0.543 |
| Citation presence rate | 0.952 | 0.952 |
| Out-of-domain rejection rate | 1.000 | 1.000 |
| Unsupported claim count | 0 | 0 |

Hybrid (Config B) remains the Streamlit production path. Difference vs dense-only is small (slightly higher context precision).

## Tests

Run: `.venv\Scripts\python.exe -m pytest tests -v -rs`

**70 passed, 2 skipped, 0 failed**

Skipped (starter `tests/test_individual.py` Task 6 university-domain queries against IELTS corpus):

1. `TestTask6::test_results_have_required_keys` — reason: `Không có kết quả` (query `scholarship eligibility`)
2. `TestTask6::test_keyword_match_scores_higher` — reason: `Không có kết quả` (query `tuition fee payment policy`)

## Environment (sanitized)

Do not commit `.env`. Template only: `.env.example`.

Configured keys (values never printed):

- `DEEPSEEK_API_KEY` configured: true/false
- `PAGEINDEX_API_KEY` configured: true/false
- `QWEN_API_KEY` configured: true/false
- `HF_TOKEN` configured: true/false

Qwen Vision remains optional (`QWEN_VISION_ENABLED=false` by default) and is not used for handwriting or normal chat generation.

## Reproduction commands

```bat
cd /d <repo-root>
.venv\Scripts\python.exe scripts\test_deepseek_connection.py
.venv\Scripts\python.exe -m src.task2_crawl_news
.venv\Scripts\python.exe -m src.task3_convert_markdown
.venv\Scripts\python.exe -m src.task4_chunking_indexing
.venv\Scripts\python.exe -m src.task8_pageindex_vectorless --prepare
.venv\Scripts\python.exe -m src.task8_pageindex_vectorless --upload
.venv\Scripts\python.exe -m src.task8_pageindex_vectorless --status
.venv\Scripts\python.exe -m src.task8_pageindex_vectorless --test-query "What does Band 8 require for Coherence and Cohesion?"
.venv\Scripts\python.exe -m group_project.evaluation.calibrate_threshold
.venv\Scripts\python.exe -m group_project.evaluation.eval_pipeline
.venv\Scripts\python.exe -m pytest tests -v -rs
streamlit run app.py
```

## Known limitations

1. Handwritten candidate pages are not OCR’d and are not indexed (by design).
2. The system is a corpus retrieval assistant — not an official IELTS examiner.
3. PageIndex is optional fallback only; production answers normally use local hybrid retrieval + DeepSeek.
4. Do not claim five web successes or live DeepSeek/PageIndex success unless the corresponding live checks above succeeded (they did in this run).
5. Qwen was not used for generation or handwriting in this run.
