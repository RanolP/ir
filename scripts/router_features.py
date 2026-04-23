#!/usr/bin/env python3
"""Shared feature helpers for tier-2 router research."""

from __future__ import annotations

import re


def detect_lang(text: str) -> str:
    if re.search(r"[\uac00-\ud7a3]", text):
        return "ko"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "other"


def has_question(text: str) -> bool:
    if "?" in text:
        return True
    return bool(re.search(r"\b(what|why|how|when|where|who)\b", text.lower()))


def overlap_count(a: list[str], b: list[str], k: int) -> int:
    return len(set(a[:k]) & set(b[:k]))


def feature_text(dataset: str, query: str, bm25: dict, vector: dict, hybrid: dict) -> str:
    scores = bm25.get("bm25_scores") or []
    top10 = ",".join(f"{s:.3f}" for s in scores[:10]) if scores else "-"
    bm25_ranked = bm25.get("ranked") or []
    vector_ranked = vector.get("ranked") or []
    top1_same = "yes" if bm25_ranked[:1] == vector_ranked[:1] and bm25_ranked[:1] else "no"
    return "\n".join(
        [
            "task: tier2 routing",
            f"dataset: {dataset}",
            f"lang: {detect_lang(query)}",
            f"has_question: {'yes' if has_question(query) else 'no'}",
            f"chars: {len(query)}",
            f"tokens: {len(query.split())}",
            f"bm25_top: {bm25.get('bm25_top', 0.0):.6f}",
            f"bm25_gap: {bm25.get('bm25_gap', 0.0):.6f}",
            f"bm25_top10: {top10}",
            f"bm25_vector_top1_same: {top1_same}",
            f"bm25_vector_overlap3: {overlap_count(bm25_ranked, vector_ranked, 3)}",
            f"bm25_vector_overlap5: {overlap_count(bm25_ranked, vector_ranked, 5)}",
            f"bm25_vector_overlap10: {overlap_count(bm25_ranked, vector_ranked, 10)}",
            f"fused_top: {hybrid.get('fused_top', 0.0):.6f}",
            f"fused_gap: {hybrid.get('fused_gap', 0.0):.6f}",
            f"query: {query}",
        ]
    )


def tier2_label(skip_row: dict, hybrid_row: dict, ndcg_margin: float) -> str:
    ndcg_gain = float(hybrid_row.get("ndcg10", 0.0)) - float(skip_row.get("ndcg10", 0.0))
    recall_gain = float(hybrid_row.get("recall10", 0.0)) - float(skip_row.get("recall10", 0.0))
    return "run_tier2" if ndcg_gain >= ndcg_margin or recall_gain > 0 else "skip_tier2"
