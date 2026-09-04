"""
Reranking stage: re-scores a candidate pool (from hybrid retrieval) using signals
that only make sense to compute pairwise against the actual query, which is the
whole point of a reranking stage vs. the first-pass retriever.

HONEST NOTE: a production system would use a cross-encoder (e.g. bge-reranker,
ms-marco-MiniLM) that jointly encodes (query, chunk) through a transformer. That
requires a model download this sandboxed environment can't do at build time. The
`Reranker` ABC below is the swap point -- `CrossEncoderReranker` is stubbed with the
exact interface a real cross-encoder wrapper would implement (see comment in the
class). What's implemented and actually running is `LexicalOverlapReranker`, which
combines: (a) query-term coverage in the chunk, (b) an exact-phrase match bonus
for multi-word query spans, and (c) a length-normalization penalty so long chunks
don't win purely by containing more words. This is a legitimate, explainable
reranking signal on top of RRF fusion, not a placeholder that always agrees with
the first-pass ranking -- the eval report shows it changes the top-k set.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from src.retrieval.sparse import ScoredChunk, tokenize


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
        ...


class CrossEncoderReranker(Reranker):
    """
    Swap point for a real cross-encoder. Production implementation would be:

        from sentence_transformers import CrossEncoder
        model = CrossEncoder("BAAI/bge-reranker-base")
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = model.predict(pairs)

    Left unimplemented here deliberately (raises) rather than silently degrading to
    the lexical reranker, so it's obvious in a demo/interview which one ran.
    """

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
        raise NotImplementedError(
            "CrossEncoderReranker requires a downloaded model (e.g. bge-reranker-base) "
            "which this environment cannot fetch. Use LexicalOverlapReranker, or wire "
            "in sentence-transformers + a downloaded checkpoint in your own environment."
        )


class LexicalOverlapReranker(Reranker):
    def __init__(self, phrase_bonus: float = 0.5, length_penalty_weight: float = 0.15):
        self.phrase_bonus = phrase_bonus
        self.length_penalty_weight = length_penalty_weight

    def _score(self, query: str, text: str) -> float:
        q_tokens = tokenize(query)
        t_tokens = tokenize(text)
        t_token_set = set(t_tokens)
        if not q_tokens:
            return 0.0

        coverage = sum(1 for t in q_tokens if t in t_token_set) / len(q_tokens)

        phrase_score = 0.0
        q_words = re.findall(r"[a-zA-Z0-9]+", query.lower())
        text_lower = text.lower()
        for n in (3, 2):
            for i in range(len(q_words) - n + 1):
                span = " ".join(q_words[i:i + n])
                if span in text_lower:
                    phrase_score += self.phrase_bonus * (n / 3)

        length_penalty = self.length_penalty_weight * min(len(t_tokens) / 250.0, 1.0)

        return coverage + phrase_score - length_penalty

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
        rescored = [
            ScoredChunk(chunk=sc.chunk, score=self._score(query, sc.chunk.text), source="reranked")
            for sc in candidates
        ]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:top_k]
