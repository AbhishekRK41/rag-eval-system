"""
Generation-quality metrics: faithfulness (is the answer grounded in retrieved
context, or is the model hallucinating beyond it?) and answer relevance.

HONEST NOTE: proper faithfulness/relevance scoring (e.g. via RAGAS) uses an LLM-as-
judge, which costs API calls per question and needs ANTHROPIC_API_KEY set. That path
is implemented in `llm_judge_faithfulness` and is what should be used for a real
eval run. `heuristic_faithfulness` is a zero-cost, no-API-key fallback used so the
eval suite (and CI) can run end-to-end without secrets: it checks what fraction of
numeric claims (amounts, day counts, percentages) in the generated answer actually
appear somewhere in the retrieved context, since numeric fabrication is the most
common and most damaging hallucination mode in a policy-QA setting.
"""
from __future__ import annotations

import os
import re

_NUM_RE = re.compile(r"\b\d[\d,]*\.?\d*%?\b")


def heuristic_faithfulness(answer: str, context_text: str) -> dict:
    answer_nums = set(_NUM_RE.findall(answer))
    context_nums = set(_NUM_RE.findall(context_text))
    if not answer_nums:
        return {"method": "heuristic", "score": 1.0, "note": "no numeric claims in answer"}
    grounded = sum(1 for n in answer_nums if n in context_nums)
    score = grounded / len(answer_nums)
    return {
        "method": "heuristic",
        "score": round(score, 3),
        "ungrounded_numbers": sorted(answer_nums - context_nums),
    }


def llm_judge_faithfulness(question: str, answer: str, context_text: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"method": "llm_judge", "score": None, "note": "ANTHROPIC_API_KEY not set"}
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""Question: {question}
Context: {context_text}
Answer: {answer}

Rate the answer's faithfulness to the context on a 0.0-1.0 scale, where 1.0 means every
claim in the answer is directly supported by the context and 0.0 means the answer is
entirely unsupported or contradicts the context. Respond with ONLY a JSON object:
{{"score": <float>, "reasoning": "<one sentence>"}}"""
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    import json
    try:
        parsed = json.loads(text.strip().strip("```json").strip("```"))
        return {"method": "llm_judge", **parsed}
    except Exception:
        return {"method": "llm_judge", "score": None, "raw": text}
