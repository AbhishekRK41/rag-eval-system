"""Hybrid retrieval via Reciprocal Rank Fusion (RRF) of sparse + dense rankings."""
from __future__ import annotations

from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import BM25Retriever, ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    """
    Standard RRF: score(doc) = sum over lists of 1 / (k + rank_in_list).
    k=60 is the commonly used default from the original RRF paper (Cormack et al.)
    and is not sensitive to score scale differences between BM25 and cosine similarity,
    which is exactly why RRF (rather than a weighted score sum) is used to fuse them.
    """
    fused_scores: dict[str, float] = {}
    chunk_lookup = {}
    for ranked in ranked_lists:
        for rank, sc in enumerate(ranked):
            cid = sc.chunk.chunk_id
            chunk_lookup[cid] = sc.chunk
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    ordered = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        ScoredChunk(chunk=chunk_lookup[cid], score=score, source="hybrid")
        for cid, score in ordered
    ]


class HybridRetriever:
    def __init__(self, sparse: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.sparse = sparse
        self.dense = dense
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10, candidate_pool: int = 30) -> list[ScoredChunk]:
        sparse_results = self.sparse.search(query, top_k=candidate_pool)
        dense_results = self.dense.search(query, top_k=candidate_pool)
        fused = reciprocal_rank_fusion([sparse_results, dense_results], k=self.rrf_k)
        return fused[:top_k]
