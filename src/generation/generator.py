"""
Answer generation from retrieved context.

Uses the Anthropic API if ANTHROPIC_API_KEY is set in the environment. If it isn't
(e.g. running the eval suite in CI without secrets), falls back to a deterministic
extractive answer built directly from the top retrieved chunk, so the retrieval +
eval pipeline is fully testable without any API key or network access. The
`/query` API route reports which mode was used in its response.
"""
from __future__ import annotations

import os

from src.retrieval.sparse import ScoredChunk

SYSTEM_PROMPT = """You are a company policy assistant. Answer the user's question using
ONLY the provided context chunks. If the context does not contain the answer, say so
explicitly rather than guessing. Cite which document (by its doc_id) each part of your
answer comes from. Be concise and specific -- quote exact numbers/limits from the policy
text rather than paraphrasing them vaguely."""


def _format_context(chunks: list[ScoredChunk]) -> str:
    blocks = []
    for i, sc in enumerate(chunks, 1):
        blocks.append(f"[Chunk {i} | doc_id={sc.chunk.doc_id}]\n{sc.chunk.text}")
    return "\n\n".join(blocks)


def generate_answer(query: str, context_chunks: list[ScoredChunk]) -> dict:
    context_str = _format_context(context_chunks)
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nQuestion: {query}",
                }],
            )
            answer_text = "".join(
                block.text for block in resp.content if block.type == "text"
            )
            return {"answer": answer_text, "mode": "llm", "model": "claude-sonnet-4-6"}
        except Exception as e:  # pragma: no cover - network/key failure path
            return _extractive_fallback(query, context_chunks, error=str(e))

    return _extractive_fallback(query, context_chunks)


def _extractive_fallback(query: str, context_chunks: list[ScoredChunk], error: str | None = None) -> dict:
    if not context_chunks:
        return {"answer": "No relevant context found.", "mode": "extractive_fallback", "error": error}
    top = context_chunks[0]
    answer = (
        f"[Extractive fallback -- no ANTHROPIC_API_KEY set] "
        f"Most relevant passage (doc_id={top.chunk.doc_id}):\n\n{top.chunk.text}"
    )
    return {"answer": answer, "mode": "extractive_fallback", "error": error}
