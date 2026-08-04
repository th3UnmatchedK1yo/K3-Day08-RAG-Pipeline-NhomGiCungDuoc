# IELTS RAG Evaluation Results

## Configurations

- **Config A (baseline):** Dense retrieval only (`BAAI/bge-m3` + ChromaDB)
- **Config B (proposed):** Dense + BM25 + RRF (k=60)

Same corpus and embedding model for both configurations.

## Aggregate Metrics

| Metric | Config A | Config B |
|---|---:|---:|
| Exact source hit rate | 0.857 | 0.857 |
| Context recall | 0.952 | 0.952 |
| Context precision | 0.524 | 0.543 |
| Citation presence rate | 0.952 | 0.952 |
| Out-of-domain rejection rate | 1.0 | 1.0 |
| Unsupported claim count | 0 | 0 |

## Known Limitation — Handwritten Candidate Responses

PDF text extraction retrieves examiner comments but cannot retrieve handwritten candidate responses.

- OCR was intentionally excluded (project policy).
- Image-only/handwriting pages are recorded as `skipped_image_text` and not indexed.
- The chatbot can still answer using task prompts, band descriptors, criteria, and examiner comments.
- It must refuse to analyse unavailable handwriting wording.
- This is a **data-availability limitation**, not a ChromaDB failure.

Skipped pages recorded: **6**

### Sample skipped pages
- ielts-academic-writing-sample-tasks-2023.pdf p.14: skipped_image_text (Academic Writing Sample Task – 1C – Sample
Script B)
- ielts-academic-writing-sample-tasks-2023.pdf p.16: skipped_image_text (Academic Writing Sample Task – 1C – Sample
Script C)
- ielts-academic-writing-sample-tasks-2023.pdf p.19: skipped_image_text (Academic Writing Sample Task – 2A – Sample
Script B)
- ielts-academic-writing-sample-tasks-2023.pdf p.21: skipped_image_text (Academic Writing Sample Task – 2A – Sample
Script C)
- ielts-academic-writing-sample-tasks-2023.pdf p.23: skipped_image_text (Academic Writing Sample Task – 2B – Sample
Script A)
- ielts-academic-writing-sample-tasks-2023.pdf p.25: skipped_image_text (Academic Writing Sample Task – 2B – Sample
Script B)

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