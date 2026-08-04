# Bài Tập Nhóm — IELTS Writing RAG Chatbot

**Nhóm:** Nhóm Gì Cũng Được  
**Lab:** Day 08 — RAG Pipeline v2  
**Domain:** IELTS Writing Band Descriptors & Examiner Feedback Assistant

## Mục Tiêu

Xây dựng chatbot RAG trả lời câu hỏi về tiêu chí chấm IELTS Writing (Task 1 / Task 2), band descriptors, nhận xét examiner công khai và hướng dẫn từ IDP / British Council.

Hệ thống **không** phải giám khảo IELTS chính thức và **không** đảm bảo band score cho bài của người dùng.

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot — ✅ Hoàn thành

| Hạng mục | Trạng thái |
|----------|------------|
| Giao diện chat Streamlit (`app.py`) | ✅ |
| Trả lời có citation (Task 10 + DeepSeek V4 Flash) | ✅ |
| Hiển thị source documents | ✅ |
| Hybrid retrieval (Dense + BM25 + RRF) | ✅ |
| PageIndex fallback khi dense cosine thấp | ✅ |
| Conversation / follow-up trong session Streamlit | ✅ |

**Stack:**

```text
Streamlit (app.py)
    → Task 9 retrieve (Config B: dense + BM25 + RRF)
    → optional PageIndex (dense cosine < 0.51)
    → Task 10 DeepSeek V4 Flash + citations
```

---

## Yêu cầu 2: RAG Evaluation Pipeline — ✅ Hoàn thành

| Deliverable | File | Trạng thái |
|-------------|------|------------|
| Golden dataset (≥15 Q&A) | `group_project/evaluation/golden_dataset.json` (**21** câu) | ✅ |
| Evaluation script | `group_project/evaluation/eval_pipeline.py` | ✅ |
| Báo cáo A/B | `group_project/evaluation/results.md` | ✅ |
| Calibrate threshold | `group_project/evaluation/calibrate_threshold.py` | ✅ |

### Framework & so sánh A/B

| Config | Retrieval |
|--------|-----------|
| **A (baseline)** | Dense only — `BAAI/bge-m3` + ChromaDB |
| **B (proposed / production)** | Dense + BM25 + RRF (`k=60`) |

Cùng corpus, cùng embedding, cùng `top_k=5`. PageIndex **tắt** khi chạy A/B để so sánh công bằng.

### Kết quả tổng hợp (21 câu)

| Metric | Config A | Config B |
|--------|---------:|---------:|
| Exact source hit rate | 0.857 | 0.857 |
| Context recall | 0.952 | 0.952 |
| Context precision | 0.524 | **0.543** |
| Citation presence rate | 0.952 | 0.952 |
| Out-of-domain rejection rate | 1.000 | 1.000 |

Chi tiết phân tích: xem `group_project/evaluation/results.md`.

---

## Kiến Trúc Hệ Thống

```text
┌─────────────────────────────────────────────────────────────────┐
│  Nguồn dữ liệu                                                  │
│  • 5 PDF IELTS Writing (band descriptors, criteria, samples)    │
│  • 5 webpage IDP + British Council (crawl Task 2)               │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Task 1–3  Validate → Crawl → Markdown + corpus.jsonl (115 rec) │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Task 4  Chunking + local BAAI/bge-m3 → ChromaDB                │
│          collection: ielts_writing_docs  |  344 chunks          │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     Config A: Dense only              Config B: Dense + BM25 + RRF
              │                                 │
              └────────────────┬────────────────┘
                               ▼
              best dense cosine < 0.51 ?
                       │ yes
                       ▼
              PageIndex (vectorless fallback)
                       │
                       ▼
              Task 10 DeepSeek V4 Flash + citations
                       │
                       ▼
              Streamlit chatbot (app.py)
```

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Võ Hà Minh Huy | 2A202601373 | Lead / full RAG pipeline: Task 1–3, Task 5–10, Streamlit `app.py`, DeepSeek, PageIndex, crawl, calibrate & tích hợp end-to-end | ✅ Hoàn thành |
| Nguyễn Minh Thái | 2A202601619 | Task 4 — ChromaDB / vector database (`src/task4_chunking_indexing.py`); tìm & chuẩn bị tài liệu PDF IELTS Writing đưa vào `data/landing/` | ✅ Hoàn thành |
| Đỗ Duy Đông | 2A202601675 | Golden dataset (`golden_dataset.json`); evaluation A/B; quản lý Git (merge, repo nhóm, đồng bộ nhánh) | ✅ Hoàn thành |

### Chi tiết theo thành viên

**Võ Hà Minh Huy — RAG Pipeline Lead**

- Xây / hoàn thiện pipeline Tasks 1–3, 5–10 và `app.py`
- Hybrid retrieval (semantic + BM25 + RRF), ngưỡng fallback 0.51
- Generation DeepSeek V4 Flash; PageIndex prepare/upload/retrieval
- Crawl British Council + IDP; báo cáo kỹ thuật `docs/final_implementation_report.md`

**Nguyễn Minh Thái — Vector DB & PDF sources**

- Thu thập / xác nhận PDF IELTS Writing (band descriptors, criteria, sample tasks, examiner comments)
- Task 4: structure-aware chunking, embedding local `BAAI/bge-m3`, index Chroma `ielts_writing_docs`
- Đảm bảo collection cosine, deterministic chunk IDs, không OCR chữ viết tay

**Đỗ Duy Đông — Evaluation & Git**

- Xây `golden_dataset.json` (21 câu: in-domain + OOD + handwriting refusal)
- Chạy / duy trì `eval_pipeline.py`, cập nhật `results.md`
- Quản lý Git: merge PR/nhánh, giữ repo nhóm sạch, tránh conflict khi ghép pipeline

---

## Hướng Dẫn Chạy

```bat
cd /d <repo-root>
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
REM Điền DEEPSEEK_API_KEY (và tùy chọn PAGEINDEX_API_KEY) vào .env — không commit .env

.venv\Scripts\python.exe -m src.task1_collect_legal_docs
.venv\Scripts\python.exe -m src.task2_crawl_news
.venv\Scripts\python.exe -m src.task3_convert_markdown
.venv\Scripts\python.exe -m src.task4_chunking_indexing

streamlit run app.py
```

Evaluation:

```bat
.venv\Scripts\python.exe -m group_project.evaluation.calibrate_threshold
.venv\Scripts\python.exe -m group_project.evaluation.eval_pipeline
.venv\Scripts\python.exe -m pytest tests -v -rs
```

---

## Số liệu hiện tại (tóm tắt)

| Hạng mục | Giá trị |
|----------|---------|
| Corpus records | 115 |
| Chroma chunks | 344 |
| Collection | `ielts_writing_docs` |
| Embedding | local `BAAI/bge-m3` |
| Web articles | 5 / 5 success |
| Golden questions | 21 |
| Tests | 70 passed, 2 skipped, 0 failed |
| LLM | DeepSeek `deepseek-v4-flash` |

---

## Lưu ý

- Không OCR bài viết tay; 6 trang image-only bị skip có chủ đích.
- Giữ repo này nếu học tiếp track Knowledge Graph (giai đoạn sau).
- Không commit file `.env` (chứa API key).
