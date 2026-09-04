"""
Runs the full retrieval benchmark: {fixed, semantic} chunking x
{sparse, dense, hybrid, hybrid+rerank} retrieval, against the golden Q&A set.

Usage: python -m src.eval.run_eval [--k 5] [--out data/eval_report.json]

Also runs one end-to-end generation + faithfulness check using the winning
config, and writes both a JSON report (machine-readable, consumed by CI) and a
Markdown table (for pasting into the README).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.eval.retrieval_metrics import evaluate_strategy
from src.generation.generator import generate_answer
from src.eval.generation_metrics import heuristic_faithfulness
from src.ingestion.chunker import chunk_corpus, load_corpus
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import LexicalOverlapReranker
from src.retrieval.sparse import BM25Retriever

ROOT = Path(__file__).resolve().parents[2]


def build_strategies(chunks):
    sparse = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    hybrid = HybridRetriever(sparse, dense)
    reranker = LexicalOverlapReranker()

    def sparse_fn(q, k):
        return sparse.search(q, top_k=k)

    def dense_fn(q, k):
        return dense.search(q, top_k=k)

    def hybrid_fn(q, k):
        return hybrid.search(q, top_k=k)

    def hybrid_rerank_fn(q, k):
        candidates = hybrid.search(q, top_k=max(k * 3, 15))
        return reranker.rerank(q, candidates, top_k=k)

    return {
        "sparse_bm25": sparse_fn,
        "dense_tfidf_svd": dense_fn,
        "hybrid_rrf": hybrid_fn,
        "hybrid_rrf_reranked": hybrid_rerank_fn,
    }, hybrid


def run(k: int = 5, out_path: Path | None = None) -> dict:
    corpus = load_corpus(ROOT / "data" / "raw")
    golden = json.loads((ROOT / "data" / "golden_qa.json").read_text())

    report = {"k": k, "n_docs": len(corpus), "n_golden_queries": len(golden), "chunking": {}}

    for chunk_strategy in ("fixed", "semantic"):
        t0 = time.time()
        chunks = chunk_corpus(corpus, strategy=chunk_strategy)
        strategies, hybrid_retriever = build_strategies(chunks)

        strat_report = {"n_chunks": len(chunks), "retrieval": {}}
        best_fn = None
        for name, fn in strategies.items():
            metrics = evaluate_strategy(golden, fn, k=k)
            strat_report["retrieval"][name] = metrics
            if name == "hybrid_rrf_reranked":
                best_fn = fn
        strat_report["build_time_sec"] = round(time.time() - t0, 2)
        report["chunking"][chunk_strategy] = strat_report

        # One end-to-end generation sample on the semantic+hybrid_rerank config,
        # to demonstrate + smoke-test the generation and faithfulness layer.
        if chunk_strategy == "semantic" and best_fn is not None:
            sample_q = golden[0]
            retrieved = best_fn(sample_q["question"], k)
            gen = generate_answer(sample_q["question"], retrieved)
            context_text = " ".join(sc.chunk.text for sc in retrieved)
            faith = heuristic_faithfulness(gen["answer"], context_text)
            report["generation_sample"] = {
                "question": sample_q["question"],
                "mode": gen["mode"],
                "answer": gen["answer"],
                "faithfulness": faith,
            }

    if out_path:
        out_path.write_text(json.dumps(report, indent=2))
    return report


def to_markdown(report: dict) -> str:
    lines = [
        f"**Golden set:** {report['n_golden_queries']} hand-labeled questions across "
        f"{report['n_docs']} policy documents. Metrics @k={report['k']}.\n",
        "| Chunking | Retrieval strategy | Chunks | Hit Rate@k | MRR@k | NDCG@k |",
        "|---|---|---|---|---|---|",
    ]
    name_map = {
        "sparse_bm25": "BM25 (sparse only)",
        "dense_tfidf_svd": "TF-IDF+SVD (dense only)",
        "hybrid_rrf": "Hybrid (RRF fusion)",
        "hybrid_rrf_reranked": "Hybrid + lexical reranker",
    }
    for chunk_strategy, strat in report["chunking"].items():
        n_chunks = strat["n_chunks"]
        for retr_name, m in strat["retrieval"].items():
            lines.append(
                f"| {chunk_strategy} | {name_map.get(retr_name, retr_name)} | {n_chunks} | "
                f"{m['hit_rate@k']:.3f} | {m['mrr@k']:.3f} | {m['ndcg@k']:.3f} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=str, default="data/eval_report.json")
    args = parser.parse_args()

    report = run(k=args.k, out_path=ROOT / args.out)
    print(json.dumps(report, indent=2))
    print("\n\n===== MARKDOWN TABLE =====\n")
    print(to_markdown(report))
