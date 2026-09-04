# RAG Eval System

A production-shaped Retrieval-Augmented Generation pipeline with a **built-in
evaluation harness** — not just "ask the corpus a question," but *quantified,
CI-gated evidence* that the retrieval actually works, across chunking strategies
and retrieval methods, against a hand-labeled golden dataset.

The corpus is a 10-document internal policy handbook for a fictional retail
company (leave, expenses, WFH, performance, IT security, exit, code of conduct,
benefits, asset provisioning, international travel) — deliberately messy and
cross-referencing, the way real internal knowledge bases are, so retrieval has to
actually discriminate between similar-sounding clauses in different documents.

## Why this exists

Most RAG portfolio projects stop at "it answers questions." The hard, interview-
relevant problem in RAG is **knowing whether it's working** — chunking strategy,
retrieval method, and reranking all have measurable, sometimes counter-intuitive
effects on answer quality. This repo makes that measurable and repeatable.

## Results (real numbers, not illustrative)

**Golden set:** 35 hand-labeled questions across 10 policy documents. Metrics @k=5.
Relevance judged as: retrieved chunk is from the correct source doc **and** contains
the specific fact needed to answer (see `src/eval/retrieval_metrics.py`).

| Chunking | Retrieval strategy | Chunks | Hit Rate@k | MRR@k | NDCG@k |
|---|---|---|---|---|---|
| fixed | BM25 (sparse only) | 37 | 1.000 | 0.943 | 0.948 |
| fixed | TF-IDF+SVD (dense only) | 37 | 1.000 | 0.938 | 0.945 |
| fixed | Hybrid (RRF fusion) | 37 | 1.000 | 0.938 | 0.943 |
| fixed | Hybrid + lexical reranker | 37 | 0.943 | 0.844 | 0.865 |
| semantic | BM25 (sparse only) | 61 | 0.943 | 0.914 | 0.922 |
| semantic | TF-IDF+SVD (dense only) | 61 | 0.943 | 0.886 | 0.896 |
| semantic | Hybrid (RRF fusion) | 61 | 0.943 | 0.914 | 0.919 |
| semantic | Hybrid + lexical reranker | 61 | 0.914 | 0.800 | 0.829 |

**Two findings worth knowing before an interview asks about them:**

1. **BM25 alone wins on this corpus.** Policy documents are lexically precise —
   the question "what is the maximum earned leave carry-forward" and the answer
   clause share exact vocabulary ("earned leave", "carry forward"). Dense/hybrid
   retrieval earns its keep on corpora with vocabulary mismatch (synonyms,
   paraphrase, cross-lingual); it doesn't help, and can add fusion noise, when
   the corpus is this lexically clean. That's a real, generalizable finding, not
   a limitation of the demo — it's the kind of thing you'd check before defaulting
   to hybrid in production.
2. **The lexical reranker underperforms RRF alone here**, on every chunking
   strategy. My hypothesis: with a small candidate pool and a corpus this
   lexically clean, BM25/RRF have already surfaced the right chunk near the top,
   and my reranker's phrase-overlap heuristic doesn't add information beyond
   what RRF fusion already captured — it just adds noise. A real cross-encoder
   (semantic relevance, not lexical overlap) would likely behave differently;
   see the `CrossEncoderReranker` swap point in `src/retrieval/reranker.py`.
   I'm reporting this rather than hiding it because a portfolio project that
   only shows improving numbers is less credible than one that shows where an
   intervention didn't help and why.

Fixed-size chunking scores *higher* than semantic chunking on Hit Rate here —
worth digging into if extending this: likely because fixed windows with overlap
sometimes accidentally include neighboring section content that happens to
contain the answer phrase, which is a metric artifact, not necessarily
better retrieval. Semantic chunking's chunks are cleaner (see `n_chunks`: 61
tightly-scoped chunks vs 37 coarser fixed windows) which matters more once you
add a token-budget constraint on context sent to the LLM — not captured by
Hit Rate@k alone.

Regenerate this table any time: `python -m src.eval.run_eval --k 5`

## Architecture

```
Query ─┬─→ BM25 (sparse)      ─┐
       └─→ TF-IDF+SVD (dense) ─┴─→ RRF fusion ─→ Lexical reranker ─→ top-k chunks
                                                                          │
                                                                          ▼
                                                          LLM generation (Claude,
                                                          w/ extractive fallback
                                                          if no API key) ─→ Answer
                                                                          │
                                                                          ▼
                                                     Heuristic + LLM-judge
                                                     faithfulness scoring
```

Golden Q&A dataset → retrieval metrics (Hit Rate@k / MRR / NDCG) computed per
{chunking strategy} × {retrieval strategy} → CI posts the table on every PR
that touches retrieval/ingestion code (`.github/workflows/eval_on_pr.yml`), and
a pytest regression floor (`test_hit_rate_reasonable_on_semantic_hybrid`) fails
the build if a retrieval change tanks quality below 0.85 Hit Rate@5.

## Honest limitations & the swap points that fix them

This was built in a sandboxed environment with no access to a model hub or
embeddings API, so two components are deliberately-labeled substitutes for
what a production version would use — both have a documented interface so
swapping them is a config change, not a rewrite:

- **Dense retrieval** uses TF-IDF + Truncated SVD (classic LSA), not a
  transformer embedder. Swap point: `Embedder` ABC in `src/retrieval/dense.py`
  — implement `embed_texts`/`embed_query` against BGE/E5/OpenAI embeddings and
  pass it into `DenseRetriever`. Nothing else changes.
- **Reranking** uses a lexical-overlap heuristic (query-term coverage +
  phrase-match bonus + length penalty), not a cross-encoder. Swap point:
  `Reranker` ABC in `src/retrieval/reranker.py` — `CrossEncoderReranker` is
  stubbed with the exact interface a `sentence-transformers` cross-encoder
  wrapper would implement.
- **Faithfulness scoring** has both a zero-cost heuristic (checks that numeric
  claims in the answer are grounded in retrieved context — the most common
  hallucination mode in policy QA) and a real LLM-judge path
  (`llm_judge_faithfulness`) that runs when `ANTHROPIC_API_KEY` is set.

## Running it

```bash
pip install -r requirements.txt

# Run the eval suite (regenerates data/eval_report.json + the table above)
PYTHONPATH=. python -m src.eval.run_eval --k 5

# Run tests
PYTHONPATH=. pytest tests/ -v

# Run the API
export ANTHROPIC_API_KEY=sk-...   # optional — falls back to extractive answers without it
PYTHONPATH=. uvicorn src.api.main:app --reload

# Or via Docker
docker compose up --build
```

Then open `frontend/index.html` in a browser (with the API running on `:8000`),
or hit the API directly:

```bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the standard notice period for a Band 5 employee?", "strategy": "hybrid_rrf_reranked"}'
```

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Corpus/chunk counts, liveness check |
| `/strategies` | GET | Lists available retrieval strategies |
| `/query` | POST | `{question, top_k, strategy}` → answer + sources |
| `/eval-report` | GET | Returns the last-generated eval report JSON |

## What I'd do next with more time

- Swap in a real embedder + cross-encoder once I have API/model-hub access,
  and re-run the eval to see whether the "BM25 already wins" and "reranker
  underperforms" findings hold — I expect the reranker finding to flip with a
  real cross-encoder, and want the data to say so either way.
- Expand the golden set past 35 questions and add multi-hop questions that
  require synthesizing across two documents (e.g. "if I'm on a PIP, can I still
  buy out my notice period?" — spans `HR-POL-005` and `HR-POL-011`), since none
  of the current questions test that failure mode.
- Add a token-budget-aware context assembler so `/query` reports cost/latency
  tradeoffs per strategy, not just retrieval quality.

## Repo structure

```
rag-eval-system/
├── src/
│   ├── ingestion/chunker.py       # fixed + semantic chunking
│   ├── retrieval/
│   │   ├── sparse.py              # BM25
│   │   ├── dense.py               # TF-IDF+SVD (swappable Embedder ABC)
│   │   ├── hybrid.py              # Reciprocal Rank Fusion
│   │   └── reranker.py            # lexical reranker (swappable Reranker ABC)
│   ├── generation/generator.py    # Claude API + extractive fallback
│   ├── eval/
│   │   ├── retrieval_metrics.py   # hit_rate@k, MRR, NDCG
│   │   ├── generation_metrics.py  # faithfulness (heuristic + LLM-judge)
│   │   └── run_eval.py            # full benchmark runner
│   └── api/main.py                # FastAPI app
├── data/
│   ├── raw/                       # 10-doc policy corpus
│   ├── golden_qa.json             # 35 hand-labeled Q&A pairs
│   └── eval_report.json           # generated by run_eval.py
├── frontend/index.html            # minimal chat UI
├── tests/test_pipeline.py         # 13 tests incl. a CI regression floor
├── .github/workflows/eval_on_pr.yml
├── Dockerfile / docker-compose.yml
└── requirements.txt
```
