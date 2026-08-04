# IELTS RAG Evaluation Results

**Nhóm:** Nhóm Gì Cũng Được  
**Golden dataset:** Đỗ Duy Đông (MSSV 2A202601675)  
**Pipeline / retrieval configs:** Võ Hà Minh Huy (MSSV 2A202601373)  
**Vector DB / PDF corpus:** Nguyễn Minh Thái (MSSV 2A202601619)

## Configurations

- **Config A (baseline):** Dense retrieval only (`BAAI/bge-m3` + ChromaDB)
- **Config B (proposed):** Dense + BM25 + RRF (k=60)

Same corpus and embedding model for both configurations.  
PageIndex **disabled** during A/B so the comparison is fair.

## Aggregate Metrics

| Metric | Config A | Config B |
|---|---:|---:|
| Exact source hit rate | 0.857 | 0.857 |
| Context recall | 0.952 | 0.952 |
| Context precision | 0.524 | **0.543** |
| Citation presence rate | 0.952 | 0.952 |
| Out-of-domain rejection rate | 1.0 | 1.0 |
| Unsupported claim count | 0 | 0 |

### Nhận xét ngắn

- Config B (hybrid) **không làm giảm** hit rate / recall so với dense-only.
- Context precision tăng nhẹ **0.524 → 0.543** nhờ BM25 + RRF ưu tiên chunk có từ khoá band/tiêu chí khớp hơn.
- OOD rejection = 1.0: câu ngoài domain (ví dụ sửa động cơ xe) được từ chối đúng.
- Chênh lệch A/B **nhỏ** — không claim “cải thiện lớn”; Streamlit vẫn dùng Config B vì ổn định hơn với query có số band / tên tiêu chí.

## Phân tích worst performers

| ID | Vấn đề | Ghi chú |
|----|--------|---------|
| `q01`, `q10`, `q11` | `source_hit=False` dù recall cao | Retriever lấy đúng *loại* nội dung nhưng title nguồn trong top-k chưa khớp exact `expected_source_titles` |
| `q01`, `q10`, `q20` | Context precision thấp (0.0) | Top-k lẫn chunk liên quan yếu / web guidance; cần filter metadata chặt hơn theo `content_type` |
| `q21` | Context recall = 0.0 | Câu hỏi yêu cầu wording chữ viết tay — corpus cố ý **không** có text handwriting (policy); chatbot từ chối đúng |
| `q15` | Answer có thể insufficient | Một số sample chỉ còn examiner comment, thiếu script text |

## Đề xuất cải tiến

1. Siết metadata filter theo `content_type` / `criterion` / `task_number` trước khi RRF.
2. Mở rộng golden set với câu “exact title” vs “paraphrase” để đo hit rate công bằng hơn.
3. Với handwriting: giữ refusal; không thêm OCR giả.
4. Cân nhắc HyDE / query expansion cho câu hỏi tiếng Việt paraphrase mạnh.
5. Theo dõi PageIndex riêng như fallback (không đưa vào bảng A/B chính).

## Known Limitation — Handwritten Candidate Responses

PDF text extraction retrieves examiner comments but cannot retrieve handwritten candidate responses.

- OCR was intentionally excluded (project policy).
- Image-only/handwriting pages are recorded as `skipped_image_text` and not indexed.
- The chatbot can still answer using task prompts, band descriptors, criteria, and examiner comments.
- It must refuse to analyse unavailable handwriting wording.
- This is a **data-availability limitation**, not a ChromaDB failure.

Skipped pages recorded: **6**

### Sample skipped pages
- ielts-academic-writing-sample-tasks-2023.pdf p.14: skipped_image_text (Academic Writing Sample Task – 1C – Sample Script B)
- ielts-academic-writing-sample-tasks-2023.pdf p.16: skipped_image_text (Academic Writing Sample Task – 1C – Sample Script C)
- ielts-academic-writing-sample-tasks-2023.pdf p.19: skipped_image_text (Academic Writing Sample Task – 2A – Sample Script B)
- ielts-academic-writing-sample-tasks-2023.pdf p.21: skipped_image_text (Academic Writing Sample Task – 2A – Sample Script C)
- ielts-academic-writing-sample-tasks-2023.pdf p.23: skipped_image_text (Academic Writing Sample Task – 2B – Sample Script A)
- ielts-academic-writing-sample-tasks-2023.pdf p.25: skipped_image_text (Academic Writing Sample Task – 2B – Sample Script B)

## Per-question (Config B)

- `q01` hit=False recall=1.00 prec=0.00 cite=True
- `q02` hit=True recall=1.00 prec=0.20 cite=True
- `q03` hit=True recall=1.00 prec=0.40 cite=True
- `q04` hit=True recall=1.00 prec=1.00 cite=True
- `q05` hit=True recall=1.00 prec=1.00 cite=True
- `q06` hit=True recall=1.00 prec=0.60 cite=True
- `q07` hit=True recall=1.00 prec=0.20 cite=True
- `q08` hit=True recall=1.00 prec=0.60 cite=True
- `q09` hit=True recall=1.00 prec=0.60 cite=True
- `q10` hit=False recall=1.00 prec=0.00 cite=True
- `q11` hit=False recall=1.00 prec=0.20 cite=True
- `q12` hit=True recall=1.00 prec=0.40 cite=True
- `q13` hit=True recall=1.00 prec=0.60 cite=True
- `q14` hit=True recall=1.00 prec=0.60 cite=True
- `q15` hit=True recall=1.00 prec=0.80 cite=True
- `q16` hit=True recall=1.00 prec=0.80 cite=True
- `q17` hit=True recall=1.00 prec=1.00 cite=True
- `q18` hit=True recall=1.00 prec=0.60 cite=True
- `q19` hit=True recall=1.00 prec=1.00 cite=True
- `q20` hit=True recall=1.00 prec=0.00 cite=False
- `q21` hit=True recall=0.00 prec=0.80 cite=True
