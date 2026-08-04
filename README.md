---
title: IELTS Writing RAG Assistant
emoji: 📘
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# IELTS Writing Band Descriptor & Examiner Feedback Assistant

**Nhóm:** Nhóm Gì Cũng Được  
**Thành viên:** Võ Hà Minh Huy (2A202601373) · Nguyễn Minh Thái (2A202601619) · Đỗ Duy Đông (2A202601675)

Day 8 RAG Pipeline lab — adapted from University Services to **IELTS Writing** scoring guidance.

The chatbot answers Vietnamese/English questions about official Writing band descriptors, assessment criteria, examiner comments, and public teaching guidance. It is **not** an official IELTS examiner and does **not** guarantee band scores.

Phân công & báo cáo nhóm: [`group_project/README.md`](group_project/README.md) · Kết quả eval: [`group_project/evaluation/results.md`](group_project/evaluation/results.md)

---

## App Screenshots

Dưới đây là một số hình ảnh giao diện của ứng dụng:

![Screenshot 1](assets/screenshots/Screenshot%202026-08-04%20163253.png)
![Screenshot 2](assets/screenshots/Screenshot%202026-08-04%20163304.png)
![Screenshot 3](assets/screenshots/Screenshot%202026-08-04%20165043.png)

---

## Architecture

```text
PDFs (data/landing/) + 5 official webpages
        │
        ▼
Task 1 validate  →  Task 2 crawl  →  Task 3 Markdown + corpus.jsonl
        │
        ▼
Task 4 structure-aware chunking + BAAI/bge-m3 → ChromaDB (ielts_writing_docs, cosine)
        │
        ├─ Config A: dense only
        └─ Config B: dense + BM25 + RRF (k=60)
                │
                ▼
        optional PageIndex fallback (dense cosine threshold)
                │
                ▼
        Task 10 DeepSeek V4 Flash generation + citations (or retrieval-only if unavailable)
                │
                ▼
        Streamlit chatbot (app.py)
```

---

## Data pipeline

1. **Task 1** — Validate existing IELTS PDFs under `data/landing/`; copy into `data/landing/legal/` for test compatibility (non-destructive).
2. **Task 2** — Crawl five official IDP / British Council pages into `data/landing/news/article_XX.json`.
3. **Task 3** — Native PDF extraction (PyMuPDF + pypdf); skip image-only handwriting; write `data/standardized/` + `data/standardized/ielts/corpus.jsonl`.
4. **Task 4** — Chunk + embed (`BAAI/bge-m3`, 1024-d) + upsert Chroma collection `ielts_writing_docs`.
5. **Tasks 5–9** — Semantic / BM25 / RRF hybrid retrieval with query intent parsing.
6. **Task 10** — Evidence-based Vietnamese answers with DeepSeek V4 Flash + citations.

### Official web sources

1. [IDP — Why can’t I get a Band 8?](https://ielts.idp.com/prepare/article-writing-task-2-why-cant-i-get-a-band-8)
2. [IDP — 8 steps to Band 8](https://ielts.idp.com/prepare/article-ielts-writing-task-2-8-steps-to-band-8)
3. [British Council — Writing Task 2 tips](https://takeielts.britishcouncil.org/blog/ielts-writing-task-2-tips)
4. [British Council — How to write an IELTS essay](https://takeielts.britishcouncil.org/blog/how-to-write-an-english-essay-for-ielts)
5. [British Council — Academic Writing Practice Test Task 2](https://takeielts.britishcouncil.org/take-ielts/prepare/free-ielts-english-practice-tests/writing/academic/task-2)

### Selected systems

| Config | Retrieval |
|--------|-----------|
| **A baseline** | Dense only (`BAAI/bge-m3` + Chroma cosine) |
| **B proposed** | Dense + BM25Okapi + RRF (`k=60`) |

Same corpus and embedding model for fair A/B comparison. Generation: `temperature=0.2`, `top_p=0.9`, context `top_k=5`.

---

## Installation (Windows CMD)

```bat
cd /d <repo-root>
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Optional — only if you want Crawl4AI browser rendering:
playwright install chromium

copy .env.example .env
REM Edit .env and set DEEPSEEK_API_KEY / LLM_MODEL (default deepseek-v4-flash)
```

Model caches stay inside the repo:

```bat
set HF_HOME=%CD%\.cache\huggingface
set TRANSFORMERS_CACHE=%CD%\.cache\huggingface
```

> On some Windows machines with a very long repo path, `pip install torch` may hit `WinError 206`. Use `subst R: %CD%` then install via `R:\.venv\Scripts\python.exe -m pip install ...`.

---

## Run the pipeline

```bat
.venv\Scripts\python.exe -m src.task1_collect_legal_docs
.venv\Scripts\python.exe -m src.task2_crawl_news
.venv\Scripts\python.exe -m src.task3_convert_markdown
.venv\Scripts\python.exe -m src.task4_chunking_indexing
.venv\Scripts\python.exe -m src.task5_semantic_search
.venv\Scripts\python.exe -m src.task6_lexical_search
.venv\Scripts\python.exe -m src.task9_retrieval_pipeline
```

Rebuild Chroma only:

```bat
.venv\Scripts\python.exe -m src.task4_chunking_indexing
```

---

## Run the chatbot

```bat
streamlit run app.py
```

Smoke import (no server):

```bat
.venv\Scripts\python.exe -c "import app; print('app import ok')"
```

---

## Evaluation

```bat
.venv\Scripts\python.exe -m group_project.evaluation.calibrate_threshold
.venv\Scripts\python.exe -m group_project.evaluation.eval_pipeline
```

Outputs:

- `group_project/evaluation/threshold_calibration.json`
- `group_project/evaluation/results.md`
- `group_project/evaluation/eval_summary.json`

Tests:

```bat
.venv\Scripts\python.exe -m pytest tests -v
```

---

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `DEEPSEEK_API_KEY` | For generation | DeepSeek V4 Flash chat completions |
| `DEEPSEEK_BASE_URL` | No | Default `https://api.deepseek.com` |
| `LLM_MODEL` | No | Default `deepseek-v4-flash` (retrieval-only if key missing) |
| `PAGEINDEX_API_KEY` | No | Optional Task 8 fallback |
| `EMBEDDING_BACKEND` / `EMBEDDING_MODEL` | No | Keep `local` + `BAAI/bge-m3` |
| `HF_TOKEN` | No | Optional Hugging Face Hub download auth only |
| `QWEN_API_KEY` / `QWEN_BASE_URL` / `QWEN_MODEL` | No | Optional non-handwriting visual preprocess (`QWEN_VISION_ENABLED=false` by default) |

Missing PageIndex / DeepSeek keys must **not** crash the app (retrieval-only evidence mode).

---

## Citation format

- PDF: `[IELTS Writing Band Descriptors 2023, Task 2 Band 7 Lexical Resource, p.7]`
- Examiner: `[IELTS Academic Writing Sample Tasks 2023, Task 2A Script C Examiner Comment, p.22]`
- Web: `[IDP IELTS, Why can’t I get a Band 8?]`

---

## Known limitation — handwriting

Some sample-task pages are handwritten candidate scripts.

- OCR is intentionally **disabled**.
- Image-only handwriting pages are skipped (`skipped_image_text`) and **not indexed**.
- Examiner comments / prompts / descriptors remain searchable.
- The chatbot must refuse to analyse unavailable handwriting wording.
- This is a **data-availability** limit, not a ChromaDB failure.

---

## Folder structure

```text
├── app.py
├── README.md
├── LAB_GUIDE.md
├── requirements.txt
├── .env.example
├── data/
│   ├── landing/           # PDFs + legal/ + news/
│   ├── standardized/      # Markdown + ielts/corpus.jsonl
│   └── inspection/        # validation / indexing / visual reports
├── src/                   # Task 1–10 modules
├── chroma_db/             # local Chroma persistence
├── .cache/huggingface/    # local model cache
├── tests/
├── group_project/evaluation/
└── docs/                  # baseline/final reports
```
