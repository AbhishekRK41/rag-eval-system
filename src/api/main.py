from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.eval.run_eval import build_strategies
from src.generation.generator import generate_answer
from src.ingestion.chunker import chunk_corpus, load_corpus

ROOT = Path(__file__).resolve().parents[2]

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    corpus = load_corpus(ROOT / "data" / "raw")
    chunks = chunk_corpus(corpus, strategy="semantic")
    strategies, hybrid = build_strategies(chunks)
    _state["strategies"] = strategies
    _state["n_chunks"] = len(chunks)
    _state["n_docs"] = len(corpus)
    yield
    _state.clear()


app = FastAPI(
    title="RAG Eval System API",
    description="Production-style RAG API with hybrid retrieval, reranking, and a "
                "built-in evaluation harness over a corporate policy corpus.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    strategy: str = "hybrid_rrf_reranked"  # sparse_bm25 | dense_tfidf_svd | hybrid_rrf | hybrid_rrf_reranked


class QueryResponse(BaseModel):
    answer: str
    mode: str
    sources: list[dict]
    strategy_used: str


@app.get("/health")
def health():
    return {"status": "ok", "n_docs": _state.get("n_docs"), "n_chunks": _state.get("n_chunks")}


@app.get("/strategies")
def list_strategies():
    return {"available": list(_state["strategies"].keys())}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    strategies = _state["strategies"]
    if req.strategy not in strategies:
        raise HTTPException(400, f"Unknown strategy '{req.strategy}'. Available: {list(strategies)}")

    retrieved = strategies[req.strategy](req.question, req.top_k)
    gen = generate_answer(req.question, retrieved)
    return QueryResponse(
        answer=gen["answer"],
        mode=gen["mode"],
        strategy_used=req.strategy,
        sources=[
            {"doc_id": sc.chunk.doc_id, "chunk_id": sc.chunk.chunk_id,
             "score": round(sc.score, 4), "text": sc.chunk.text[:300]}
            for sc in retrieved
        ],
    )


@app.get("/eval-report")
def eval_report():
    report_path = ROOT / "data" / "eval_report.json"
    if not report_path.exists():
        raise HTTPException(404, "No eval report found. Run `python -m src.eval.run_eval` first.")
    return json.loads(report_path.read_text())
