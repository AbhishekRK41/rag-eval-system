"""
Retrieval metrics computed against the golden Q&A dataset.

Relevance judgment: a retrieved chunk is judged relevant to a golden question if
(a) it comes from the labeled correct doc_id, AND (b) it contains at least one of the
`answer_contains` keyword/phrase strings (case-insensitive). Condition (a) alone would
be too generous -- a doc has multiple chunks and only some actually answer the
question; condition (b) grounds relevance in whether the retrieved text could actually
support a correct answer, which is what retrieval is *for*. This also makes the
metrics comparable across the fixed vs. semantic chunking strategies, which produce
different chunk boundaries and chunk_ids for the same underlying content.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.retrieval.sparse import ScoredChunk


def is_relevant(chunk_text: str, doc_id: str, gold_doc_id: str, answer_contains: list[str]) -> bool:
    if doc_id != gold_doc_id:
        return False
    text_lower = chunk_text.lower()
    return any(phrase.lower() in text_lower for phrase in answer_contains)


@dataclass
class QueryResult:
    query_id: str
    relevance: list[bool]  # in ranked order


def hit_rate_at_k(results: list[QueryResult], k: int) -> float:
    hits = sum(1 for r in results if any(r.relevance[:k]))
    return hits / len(results) if results else 0.0


def mrr(results: list[QueryResult], k: int | None = None) -> float:
    total = 0.0
    for r in results:
        rel = r.relevance[:k] if k else r.relevance
        for rank, is_rel in enumerate(rel, 1):
            if is_rel:
                total += 1.0 / rank
                break
    return total / len(results) if results else 0.0


def ndcg_at_k(results: list[QueryResult], k: int) -> float:
    total = 0.0
    for r in results:
        rel = r.relevance[:k]
        dcg = sum((1.0 if is_rel else 0.0) / math.log2(i + 2) for i, is_rel in enumerate(rel))
        ideal_rel = sorted(rel, reverse=True)
        idcg = sum((1.0 if is_rel else 0.0) / math.log2(i + 2) for i, is_rel in enumerate(ideal_rel))
        total += (dcg / idcg) if idcg > 0 else 0.0
    return total / len(results) if results else 0.0


def evaluate_strategy(golden: list[dict], retrieve_fn, k: int = 5) -> dict:
    """
    retrieve_fn: Callable[[str, int], list[ScoredChunk]] -- takes (query, top_k)
    Returns a dict of metrics at k plus the raw per-query results for inspection.
    """
    query_results = []
    for item in golden:
        retrieved: list[ScoredChunk] = retrieve_fn(item["question"], k)
        relevance = [
            is_relevant(sc.chunk.text, sc.chunk.doc_id, item["doc_id"], item["answer_contains"])
            for sc in retrieved
        ]
        query_results.append(QueryResult(query_id=item["id"], relevance=relevance))

    return {
        "hit_rate@k": round(hit_rate_at_k(query_results, k), 4),
        "mrr@k": round(mrr(query_results, k), 4),
        "ndcg@k": round(ndcg_at_k(query_results, k), 4),
        "n_queries": len(query_results),
        "k": k,
    }
