"""Calibrate SCORE_THRESHOLD using relevant vs unrelated semantic scores."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface"))

OUT_PATH = Path(__file__).parent / "threshold_calibration.json"

RELEVANT = [
    "What is the difference between Band 6 and Band 7 Lexical Resource?",
    "What does Task Response assess in Writing Task 2?",
    "What does Band 8 require for Coherence and Cohesion?",
    "What did the examiner say about the Task 2A Band 8.5 response?",
    "Why can overusing cohesive devices reduce writing quality?",
]

UNRELATED = [
    "How do I repair a car engine?",
    "What is the weather tomorrow?",
    "How do I cook Vietnamese pho?",
    "Explain quantum field theory.",
    "What is the tuition fee at a university?",
]


def _best_score(query: str) -> float:
    from src.task5_semantic_search import semantic_search

    results = semantic_search(query, top_k=3)
    if not results:
        return 0.0
    return float(results[0]["score"])


def main():
    relevant_scores = []
    unrelated_scores = []
    details = {"relevant": [], "unrelated": []}

    for q in RELEVANT:
        s = _best_score(q)
        relevant_scores.append(s)
        details["relevant"].append({"query": q, "best_dense_score": s})
        print(f"[RELEVANT] {s:.4f} | {q}")

    for q in UNRELATED:
        s = _best_score(q)
        unrelated_scores.append(s)
        details["unrelated"].append({"query": q, "best_dense_score": s})
        print(f"[UNRELATED] {s:.4f} | {q}")

    rel_min = min(relevant_scores) if relevant_scores else 0.0
    unr_max = max(unrelated_scores) if unrelated_scores else 0.0
    # Midpoint suggestion; keep 0.48 if gap unclear
    if rel_min > unr_max:
        suggested = round((rel_min + unr_max) / 2, 4)
    else:
        suggested = 0.48

    report = {
        "relevant_scores": relevant_scores,
        "unrelated_scores": unrelated_scores,
        "relevant_mean": statistics.mean(relevant_scores) if relevant_scores else None,
        "unrelated_mean": statistics.mean(unrelated_scores) if unrelated_scores else None,
        "relevant_min": rel_min,
        "unrelated_max": unr_max,
        "suggested_threshold": suggested,
        "default_threshold": 0.48,
        "details": details,
        "note": "Fallback must use original dense cosine similarity, never RRF score.",
    }
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSuggested threshold: {suggested}")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
