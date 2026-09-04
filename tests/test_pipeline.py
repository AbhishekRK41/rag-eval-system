import json
from pathlib import Path

import pytest

from src.ingestion.chunker import FixedSizeChunker, SemanticChunker, chunk_corpus, load_corpus
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.retrieval.reranker import LexicalOverlapReranker
from src.eval.retrieval_metrics import evaluate_strategy, is_relevant

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(ROOT / "data" / "raw")


@pytest.fixture(scope="module")
def golden():
    return json.loads((ROOT / "data" / "golden_qa.json").read_text())


def test_corpus_loads(corpus):
    assert len(corpus) == 10
    assert all(len(text) > 100 for text in corpus.values())


def test_fixed_chunker_respects_size():
    chunker = FixedSizeChunker(chunk_size_words=50, overlap_words=10)
    chunks = chunker.chunk("doc1", "word " * 200)
    assert all(len(c.text.split()) <= 50 for c in chunks)
    assert len(chunks) > 1


def test_semantic_chunker_preserves_sections(corpus):
    chunker = SemanticChunker()
    chunks = chunker.chunk("01_leave_policy", corpus["01_leave_policy"])
    sections_found = {c.metadata.get("section") for c in chunks}
    assert any("LEAVE CATEGORIES" in (s or "") for s in sections_found)


def test_bm25_retrieves_correct_doc(corpus):
    chunks = chunk_corpus(corpus, "semantic")
    retriever = BM25Retriever(chunks)
    results = retriever.search("how many days of earned leave do I accrue", top_k=3)
    assert results[0].chunk.doc_id == "01_leave_policy"


def test_dense_retriever_returns_scored_results(corpus):
    chunks = chunk_corpus(corpus, "semantic")
    retriever = DenseRetriever(chunks)
    results = retriever.search("group mediclaim sum insured", top_k=5)
    assert len(results) == 5
    assert all(r.source == "dense" for r in results)


def test_rrf_fusion_combines_rankings(corpus):
    chunks = chunk_corpus(corpus, "semantic")
    sparse = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    hybrid = HybridRetriever(sparse, dense)
    results = hybrid.search("notice period Band 5", top_k=5)
    assert len(results) == 5
    assert all(r.source == "hybrid" for r in results)


def test_rrf_fusion_is_deterministic():
    from src.ingestion.chunker import Chunk
    from src.retrieval.sparse import ScoredChunk

    c1 = Chunk("c1", "doc1", "text one")
    c2 = Chunk("c2", "doc1", "text two")
    list_a = [ScoredChunk(c1, 0.9, "sparse"), ScoredChunk(c2, 0.1, "sparse")]
    list_b = [ScoredChunk(c2, 0.9, "dense"), ScoredChunk(c1, 0.1, "dense")]
    fused = reciprocal_rank_fusion([list_a, list_b])
    # both docs rank 1 in one list and rank 2 in the other -> should tie
    assert fused[0].score == fused[1].score


def test_reranker_reorders_by_lexical_overlap():
    from src.ingestion.chunker import Chunk
    from src.retrieval.sparse import ScoredChunk

    relevant = Chunk("c1", "doc1", "the maximum earned leave carry forward is 45 days")
    irrelevant = Chunk("c2", "doc1", "the office wifi password is changed every quarter")
    candidates = [ScoredChunk(irrelevant, 0.5, "hybrid"), ScoredChunk(relevant, 0.4, "hybrid")]
    reranker = LexicalOverlapReranker()
    reranked = reranker.rerank("what is the maximum earned leave carry forward", candidates, top_k=2)
    assert reranked[0].chunk.chunk_id == "c1"


def test_golden_dataset_schema(golden):
    assert len(golden) >= 30
    for item in golden:
        assert {"id", "question", "doc_id", "answer_contains"} <= item.keys()
        assert isinstance(item["answer_contains"], list) and item["answer_contains"]


def test_hit_rate_reasonable_on_semantic_hybrid(corpus, golden):
    chunks = chunk_corpus(corpus, "semantic")
    sparse = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    hybrid = HybridRetriever(sparse, dense)
    metrics = evaluate_strategy(golden, lambda q, k: hybrid.search(q, top_k=k), k=5)
    # regression floor -- fails CI if a retrieval change tanks quality
    assert metrics["hit_rate@k"] >= 0.85


def test_is_relevant_requires_both_doc_and_content_match():
    assert is_relevant("the cap is 45 days", "01_leave_policy", "01_leave_policy", ["45 days"])
    assert not is_relevant("the cap is 45 days", "02_expense_reimbursement", "01_leave_policy", ["45 days"])
    assert not is_relevant("unrelated text", "01_leave_policy", "01_leave_policy", ["45 days"])


def test_api_health():
    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["n_docs"] == 10


def test_api_query_returns_sources():
    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as client:
        resp = client.post("/query", json={"question": "How much is the base sum insured?", "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["sources"]) == 3
        assert body["strategy_used"] == "hybrid_rrf_reranked"
