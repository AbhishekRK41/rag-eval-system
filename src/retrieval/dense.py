"""
Dense (vector) retrieval.

IMPORTANT / HONEST NOTE FOR ANYONE READING THIS FILE:
This environment has no access to a model hub (HuggingFace) or an embeddings API at
build time, so the `Embedder` interface below is implemented with TF-IDF + Truncated
SVD (i.e. classic Latent Semantic Analysis) as a genuinely dense, fixed-length vector
representation that captures co-occurrence structure beyond exact keyword match.

It is NOT a transformer sentence embedder, and the eval report is explicit about that.
The `Embedder` ABC is the actual point: swap `TfidfSvdEmbedder` for a real one
(e.g. BGE-small, E5-base, or an API-based embedder) by implementing `embed_texts` /
`embed_query` and passing it into `DenseRetriever` -- nothing else in the retrieval,
hybrid fusion, or eval code needs to change. That swap is a config change, not a
rewrite, and it's the first thing to do before using this in production.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.ingestion.chunker import Chunk
from src.retrieval.sparse import ScoredChunk


class Embedder(ABC):
    """Swap this out for a real neural embedder in production. See module docstring."""

    @abstractmethod
    def fit(self, texts: list[str]) -> None: ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray: ...


class TfidfSvdEmbedder(Embedder):
    def __init__(self, n_components: int = 128, random_state: int = 42):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.svd.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_comp = max(n_comp, 2)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        tfidf = self.vectorizer.transform(texts)
        return self.svd.transform(tfidf)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self.embedder = embedder or TfidfSvdEmbedder()
        self.embedder.fit([c.text for c in chunks])
        self.matrix = self.embedder.embed_texts([c.text for c in chunks])

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        q_vec = self.embedder.embed_query(query).reshape(1, -1)
        sims = cosine_similarity(q_vec, self.matrix)[0]
        ranked_idx = np.argsort(-sims)[:top_k]
        return [
            ScoredChunk(chunk=self.chunks[i], score=float(sims[i]), source="dense")
            for i in ranked_idx
        ]
