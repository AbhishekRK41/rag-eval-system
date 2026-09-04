"""BM25 sparse retrieval over the chunk corpus."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    source: str  # "sparse" | "dense" | "hybrid" | "reranked"


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._corpus_tokens = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [ScoredChunk(chunk=c, score=float(s), source="sparse") for c, s in ranked]
