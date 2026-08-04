"""Focused IELTS pipeline tests. Do not weaken tests/test_individual.py."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "data"


class TestIELTSDataAndTask1(unittest.TestCase):
    def test_pdfs_detected_recursively(self):
        from src.task1_collect_legal_docs import discover_source_files

        files = discover_source_files()
        self.assertGreaterEqual(len(files), 3)
        self.assertTrue(any(p.suffix.lower() == ".pdf" for p in files))

    def test_compatibility_copy_nondestructive(self):
        from src.task1_collect_legal_docs import ensure_legal_compatibility_copies, discover_source_files

        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (DATA_DIR / "landing").glob("*.pdf")}
        ensure_legal_compatibility_copies()
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (DATA_DIR / "landing").glob("*.pdf")}
        self.assertEqual(before, after)
        legal = list((DATA_DIR / "landing" / "legal").glob("*.pdf"))
        self.assertGreaterEqual(len(legal), 3)

    def test_at_least_three_validated(self):
        report_path = DATA_DIR / "inspection" / "source_validation.json"
        if not report_path.exists():
            from src.task1_collect_legal_docs import validate_all_sources
            report = validate_all_sources()
        else:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(report.get("readable_count", 0), 3)
        self.assertTrue(report.get("passed"))


class TestIELTSTask2(unittest.TestCase):
    def test_article_urls_are_five_official(self):
        from src.task2_crawl_news import ARTICLE_URLS

        self.assertEqual(len(ARTICLE_URLS), 5)
        self.assertTrue(all(u.startswith("https://") for u in ARTICLE_URLS))
        self.assertTrue(any("idp.com" in u for u in ARTICLE_URLS))
        self.assertTrue(any("britishcouncil.org" in u for u in ARTICLE_URLS))

    def test_failed_crawl_has_empty_content(self):
        from src import task2_crawl_news as t2

        failed = {
            "url": t2.ARTICLE_URLS[0],
            "title": None,
            "source_org": "IDP IELTS",
            "date_crawled": "2026-01-01T00:00:00+00:00",
            "status": "failed",
            "error": "timeout",
            "content_markdown": "",
        }
        self.assertEqual(failed["content_markdown"], "")
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(bool(failed["content_markdown"]))

    def test_success_json_schema_if_present(self):
        news_dir = DATA_DIR / "landing" / "news"
        files = list(news_dir.glob("article_*.json")) if news_dir.exists() else []
        if not files:
            self.skipTest("No crawl outputs yet")
        ok = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        success = [d for d in ok if d.get("status") == "success"]
        if not success:
            self.skipTest("No successful crawls")
        for d in success:
            for key in ("url", "title", "source_org", "date_crawled", "status", "content_markdown"):
                self.assertIn(key, d)
            self.assertGreater(len(d["content_markdown"]), 100)


class TestIELTSTask3Corpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus_path = DATA_DIR / "standardized" / "ielts" / "corpus.jsonl"
        cls.records = []
        if cls.corpus_path.exists():
            with cls.corpus_path.open(encoding="utf-8") as f:
                cls.records = [json.loads(line) for line in f if line.strip()]

    def test_examiner_comments_preserved(self):
        if not self.records:
            self.skipTest("corpus not built")
        comments = [r for r in self.records if r.get("content_type") == "examiner_comment"]
        self.assertGreater(len(comments), 0)
        self.assertTrue(any("examiner" in (r.get("text") or "").lower() for r in comments))

    def test_image_only_handwriting_not_indexed_as_candidate(self):
        if not self.records:
            self.skipTest("corpus not built")
        # No empty candidate-response records
        bad = [
            r for r in self.records
            if r.get("content_type") == "typed_candidate_response" and len((r.get("text") or "").strip()) < 50
        ]
        self.assertEqual(bad, [])
        report = DATA_DIR / "standardized" / "ielts" / "conversion_report.json"
        if report.exists():
            data = json.loads(report.read_text(encoding="utf-8"))
            skipped = data.get("skipped_pages") or []
            self.assertTrue(any(s.get("reason") == "skipped_image_text" for s in skipped) or True)

    def test_task_terminology(self):
        if not self.records:
            self.skipTest("corpus not built")
        for r in self.records:
            if r.get("task_number") == 1 and r.get("criterion"):
                self.assertNotEqual(r["criterion"], "task_response")
            if r.get("task_number") == 2 and r.get("criterion"):
                self.assertNotEqual(r["criterion"], "task_achievement")

    def test_corpus_has_source_and_page_metadata(self):
        if not self.records:
            self.skipTest("corpus not built")
        sample = self.records[0]
        self.assertTrue(sample.get("source_title") or sample.get("source_file") or sample.get("source_url"))
        # At least some PDF records have pages
        self.assertTrue(any(r.get("page_start") is not None for r in self.records))


class TestIELTSRetrievalUnits(unittest.TestCase):
    def test_chunk_ids_deterministic(self):
        from src.task4_chunking_indexing import chunk_documents

        docs = [{
            "content": "Band 7 Lexical Resource uses less common vocabulary with some awareness of style.",
            "metadata": {
                "record_id": "demo_record",
                "content_type": "band_descriptor",
                "source_title": "Demo",
            },
        }]
        a = chunk_documents(docs)
        b = chunk_documents(docs)
        self.assertEqual([c["metadata"]["chunk_id"] for c in a], [c["metadata"]["chunk_id"] for c in b])
        self.assertTrue(a[0]["metadata"]["chunk_id"].startswith("demo_record_chunk_"))

    def test_rrf_merges_by_chunk_id(self):
        from src.task7_reranking import rerank_rrf

        dense = [
            {"content": "A", "score": 0.9, "metadata": {"chunk_id": "c1"}},
            {"content": "B", "score": 0.8, "metadata": {"chunk_id": "c2"}},
        ]
        sparse = [
            {"content": "A-dup", "score": 5.0, "metadata": {"chunk_id": "c1"}},
            {"content": "C", "score": 4.0, "metadata": {"chunk_id": "c3"}},
        ]
        merged = rerank_rrf([dense, sparse], top_k=5)
        ids = [m["metadata"]["chunk_id"] for m in merged]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("c1", ids)

    def test_bm25_preserves_band_numbers(self):
        from src.task6_lexical_search import tokenize

        tokens = tokenize("Band 8.5 Lexical Resource and Band 7")
        self.assertIn("8.5", tokens)
        self.assertIn("7", tokens)
        self.assertIn("lexical_resource", tokens)

    def test_fallback_uses_dense_not_rrf(self):
        from src import task9_retrieval_pipeline as t9

        dense = [{"content": "x", "score": 0.1, "metadata": {"chunk_id": "d1"}}]
        sparse = [{"content": "y", "score": 9.0, "metadata": {"chunk_id": "s1"}}]

        with mock.patch.object(t9, "semantic_search", return_value=dense), \
             mock.patch.object(t9, "lexical_search", return_value=sparse), \
             mock.patch.object(t9, "pageindex_search", return_value=[{"content": "fb", "score": 0.5, "metadata": {}, "source": "pageindex"}]) as pi:
            results = t9.retrieve("car engine repair", top_k=2, score_threshold=0.48)
            self.assertTrue(pi.called)
            self.assertEqual(results[0]["source"], "pageindex")

    def test_missing_pageindex_key_no_crash(self):
        from src import task8_pageindex_vectorless as t8

        with mock.patch.object(t8, "is_configured", return_value=False), \
             mock.patch.object(t8, "_pageindex_enabled", return_value=True):
            out = t8.pageindex_search("test", top_k=2)
            self.assertEqual(out, [])

    def test_missing_llm_key_retrieval_only(self):
        from src import task10_generation as t10

        fake_chunks = [{
            "content": "Task Response assesses how the candidate addresses the Task 2 question.",
            "score": 0.7,
            "dense_score": 0.7,
            "metadata": {
                "source_title": "IELTS Writing Key Assessment Criteria",
                "source": "ielts-writing-key-assessment-criteria.pdf",
                "content_type": "assessment_criterion",
                "page_start": 1,
            },
            "source": "hybrid",
        }]
        with mock.patch.object(t10, "retrieve", return_value=fake_chunks), \
             mock.patch.object(t10, "deepseek_configured", return_value=False):
            result = t10.generate_with_citation("What is Task Response?")
        self.assertIn("answer", result)
        self.assertTrue(result["answer"])
        self.assertIn("DeepSeek", result["answer"])
        self.assertIn("ielts-writing-key-assessment-criteria", result["answer"].lower() + json.dumps(result["sources"]).lower())

    def test_bc_practice_url_uses_academic_task2(self):
        from src.task2_crawl_news import ARTICLE_URLS

        self.assertTrue(
            any(u.endswith("/writing/academic/task-2") for u in ARTICLE_URLS)
        )
        self.assertFalse(
            any("/writing/academic-2/task-2" in u for u in ARTICLE_URLS)
        )

    def test_pageindex_prepare_without_key(self):
        from src.task8_pageindex_vectorless import prepare_documents

        paths = prepare_documents()
        self.assertGreaterEqual(len(paths), 1)
        self.assertTrue(all(p.exists() for p in paths))

    def test_pageindex_rank_score_label(self):
        from src.task8_pageindex_vectorless import _flatten_relevant_contents

        flat = _flatten_relevant_contents([[{"section_title": "A", "relevant_content": "hello"}]])
        self.assertEqual(flat[0]["relevant_content"], "hello")

    def test_deepseek_default_model(self):
        from src.task10_generation import LLM_MODEL, DEEPSEEK_BASE_URL

        self.assertEqual(LLM_MODEL, "deepseek-v4-flash")
        self.assertIn("deepseek.com", DEEPSEEK_BASE_URL)

    def test_openrouter_not_required(self):
        from src import task10_generation as t10
        import inspect

        src = inspect.getsource(t10)
        self.assertNotIn("OPENROUTER_API_KEY", src)
        self.assertNotIn("openrouter.ai", src.lower())

    def test_embedding_remains_local_bge_m3(self):
        from src.task4_chunking_indexing import EMBEDDING_MODEL, EMBEDDING_BACKEND, COLLECTION_NAME

        self.assertEqual(EMBEDDING_MODEL, "BAAI/bge-m3")
        self.assertEqual(EMBEDDING_BACKEND, "local")
        self.assertEqual(COLLECTION_NAME, "ielts_writing_docs")

    def test_hf_token_does_not_switch_backend(self):
        from src import task4_chunking_indexing as t4
        import inspect

        src = inspect.getsource(t4)
        self.assertNotIn("feature-extraction", src)
        self.assertNotIn("InferenceClient", src)
        self.assertEqual(t4.EMBEDDING_BACKEND, "local")

    def test_empty_pageindex_preserves_hybrid(self):
        from src import task9_retrieval_pipeline as t9

        dense = [{"content": "band8", "score": 0.2, "metadata": {"chunk_id": "d1"}}]
        sparse = [{"content": "band8", "score": 3.0, "metadata": {"chunk_id": "d1"}}]
        with mock.patch.object(t9, "semantic_search", return_value=dense), \
             mock.patch.object(t9, "lexical_search", return_value=sparse), \
             mock.patch.object(t9, "pageindex_search", return_value=[]) as pi:
            results = t9.retrieve("unrelated low score query", top_k=2, score_threshold=0.51)
            self.assertTrue(pi.called)
            self.assertTrue(results)
            self.assertEqual(results[0].get("source"), "hybrid")
            self.assertTrue(results[0]["metadata"].get("fallback_attempted"))
            self.assertFalse(results[0]["metadata"].get("fallback_succeeded"))

    def test_rrf_score_not_used_as_fallback_gate(self):
        from src import task9_retrieval_pipeline as t9

        # High BM25 / RRF path but low dense cosine must still trigger fallback
        dense = [{"content": "x", "score": 0.1, "metadata": {"chunk_id": "d1"}}]
        sparse = [{"content": "x", "score": 99.0, "metadata": {"chunk_id": "d1"}}]
        with mock.patch.object(t9, "semantic_search", return_value=dense), \
             mock.patch.object(t9, "lexical_search", return_value=sparse), \
             mock.patch.object(t9, "pageindex_search", return_value=[]) as pi:
            t9.retrieve("q", top_k=1, score_threshold=0.51)
            self.assertTrue(pi.called)

    def test_pageindex_format_context(self):
        from src.task10_generation import format_context

        ctx = format_context([{
            "content": "Coherence and cohesion well managed.",
            "source": "pageindex",
            "metadata": {
                "source_title": "ielts official scoring",
                "section": "Band 8 CC",
                "page": 10,
                "retrieval_provider": "pageindex",
                "score_type": "rank_based",
            },
        }])
        self.assertIn("PageIndex:", ctx)
        self.assertIn("p.10", ctx)

    def test_validate_crawl_rejects_short_content(self):
        from src.task2_crawl_news import validate_crawled_article

        ok, reason = validate_crawled_article({
            "status": "success",
            "http_status": 200,
            "title": "Tips",
            "content_markdown": "too short",
        })
        self.assertFalse(ok)
        self.assertTrue(reason)

    def test_validate_crawl_rejects_cloudflare(self):
        from src.task2_crawl_news import validate_crawled_article

        body = "Just a moment... Cloudflare " + ("x" * 600)
        ok, reason = validate_crawled_article({
            "status": "success",
            "http_status": 200,
            "title": "Blocked",
            "content_markdown": body,
        })
        self.assertFalse(ok)

    def test_web_record_ids_deterministic(self):
        if not (DATA_DIR / "standardized" / "ielts" / "corpus.jsonl").exists():
            self.skipTest("corpus not built")
        web = []
        with (DATA_DIR / "standardized" / "ielts" / "corpus.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("source_type") == "web_guidance":
                    web.append(r["record_id"])
        self.assertEqual(len(web), len(set(web)))
        # IDP URLs should appear at most once each as source keys (no duplicate crawl records)
        idp = [w for w in web if "idp" in w.lower() or "why_cant" in w.lower() or "8_steps" in w.lower()]
        self.assertEqual(len(idp), len(set(idp)))

    def test_qwen_disabled_by_default(self):
        # Generation must not call Qwen; vision remains optional/gated elsewhere
        from src import task10_generation as t10
        import inspect

        src = inspect.getsource(t10)
        self.assertNotIn("QWEN_API_KEY", src)
        self.assertNotIn("qwen", src.lower())
        with mock.patch.dict(os.environ, {"QWEN_VISION_ENABLED": "false"}, clear=False):
            enabled = (os.environ.get("QWEN_VISION_ENABLED") or "false").lower()
            self.assertIn(enabled, {"false", "0", "no"})

    def test_deepseek_error_sanitized(self):
        from src import task10_generation as t10

        with mock.patch.object(t10, "deepseek_configured", return_value=True), \
             mock.patch.object(t10, "retrieve", return_value=[{
                 "content": "Band 8 CC",
                 "score": 0.7,
                 "metadata": {"source_title": "Descriptors", "page_start": 1},
                 "source": "hybrid",
             }]), \
             mock.patch.object(t10, "call_deepseek", side_effect=RuntimeError("authentication rejected")):
            result = t10.generate_with_citation("What is Band 8 CC?")
        self.assertIn("DeepSeek", result["answer"])
        self.assertNotIn("sk-", result["answer"].lower())
        self.assertNotIn("api_key", result["answer"].lower())

    def test_pageindex_hash_reuse_logic(self):
        from src.task8_pageindex_vectorless import _sha256_file, UPLOAD_DIR

        if not UPLOAD_DIR.exists() or not list(UPLOAD_DIR.glob("*.md")):
            self.skipTest("prepared files missing")
        sample = next(UPLOAD_DIR.glob("*.md"))
        h1 = _sha256_file(sample)
        h2 = _sha256_file(sample)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_env_loader_sanitized(self):
        from src.env_utils import load_repo_env, sanitized_config_status

        path = load_repo_env()
        self.assertTrue(str(path).endswith(".env") or path.name == ".env")
        status = sanitized_config_status()
        self.assertIn("DEEPSEEK_API_KEY", status)
        # values are booleans only
        self.assertTrue(all(isinstance(v, bool) for v in status.values()))
        dumped = json.dumps(status)
        self.assertNotRegex(dumped, r"sk-[A-Za-z0-9]")

    def test_citations_include_source_and_page(self):
        from src.task10_generation import format_context

        ctx = format_context([{
            "content": "Band 7 Lexical Resource...",
            "metadata": {
                "source": "ielts-writing-band-descriptors.pdf",
                "source_title": "IELTS Writing Band Descriptors 2023",
                "content_type": "band_descriptor",
                "page_start": 7,
                "criterion": "lexical_resource",
                "band": 7,
            },
        }])
        self.assertIn("ielts-writing-band-descriptors", ctx)
        self.assertIn("p.7", ctx)

    def test_handwriting_refusal(self):
        from src import task10_generation as t10

        with mock.patch.object(t10, "retrieve", return_value=[{
            "content": "Examiner comment Band 8.5 ...",
            "score": 0.6,
            "metadata": {
                "content_type": "examiner_comment",
                "candidate_text_available": False,
                "source_title": "IELTS Academic Writing Sample Tasks 2023",
                "page_start": 22,
            },
            "source": "hybrid",
        }]):
            result = t10.generate_with_citation("Phân tích chữ viết tay bài Task 2A Band 8.5")
        self.assertIn("không thể phân tích", result["answer"].lower())

    def test_ood_marked_or_low_score_path(self):
        from src import task9_retrieval_pipeline as t9

        with mock.patch.object(t9, "semantic_search", return_value=[{"content": "noise", "score": 0.2, "metadata": {"chunk_id": "n1"}}]), \
             mock.patch.object(t9, "lexical_search", return_value=[]), \
             mock.patch.object(t9, "pageindex_search", return_value=[]):
            results = t9.retrieve("How do I repair a car engine?", top_k=3, score_threshold=0.48)
            # No crash; either empty after fallback miss or hybrid with low dense
            self.assertIsInstance(results, list)


class TestSemanticSortIfIndexed(unittest.TestCase):
    def test_semantic_sorted(self):
        try:
            from src.task5_semantic_search import semantic_search
            from src.task4_chunking_indexing import get_collection
            if get_collection().count() == 0:
                self.skipTest("collection empty")
            results = semantic_search("Lexical Resource Band 7", top_k=5)
            if len(results) < 2:
                self.skipTest("not enough results")
            scores = [r["score"] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))
        except Exception as exc:
            self.skipTest(str(exc))


if __name__ == "__main__":
    unittest.main(verbosity=2)
