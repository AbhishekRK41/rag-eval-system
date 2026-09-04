"""
Chunking strategies for the ingestion pipeline.

Two strategies are implemented and benchmarked against each other in the eval suite
(see src/eval/run_eval.py):

1. FixedSizeChunker   - naive fixed-token-count windows with overlap. Simple, fast,
                        but can split a policy clause mid-sentence or across an
                        unrelated section boundary.
2. SemanticChunker    - splits on structural boundaries (numbered sections /
                        sub-sections, which every doc in this corpus has) and only
                        falls back to fixed-size splitting when a single section is
                        itself too long. This keeps a clause and its full context
                        together, which matters a lot for retrieval precision on
                        structured documents like policies, contracts, or SOPs.

Both produce the same Chunk dataclass so downstream code doesn't care which was used.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _word_count(text: str) -> int:
    return len(text.split())


class FixedSizeChunker:
    def __init__(self, chunk_size_words: int = 120, overlap_words: int = 25):
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def chunk(self, doc_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        words = text.split()
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(words):
            end = min(start + self.chunk_size_words, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::fixed::{idx}",
                    doc_id=doc_id,
                    text=chunk_text,
                    metadata={**metadata, "strategy": "fixed", "chunk_index": idx},
                )
            )
            idx += 1
            if end == len(words):
                break
            start = end - self.overlap_words
        return chunks


class SemanticChunker:
    """
    Splits on top-level numbered sections (e.g. "3. TRAVEL EXPENSES") and, within a
    section, on numbered sub-clauses (e.g. "3.1", "3.2") when the section is long.
    Falls back to FixedSizeChunker for any section that's still too long after that
    (rare in this corpus, but real documents can have a runaway section).
    """

    SECTION_RE = re.compile(r"^\d+\.\s+[A-Z][A-Z0-9 &/\-,()]+$", re.MULTILINE)
    SUBCLAUSE_RE = re.compile(r"(?=^\d+\.\d+\s)", re.MULTILINE)

    def __init__(self, max_chunk_words: int = 220, fallback_chunk_words: int = 140,
                 fallback_overlap_words: int = 20):
        self.max_chunk_words = max_chunk_words
        self._fixed = FixedSizeChunker(fallback_chunk_words, fallback_overlap_words)

    def chunk(self, doc_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        header_lines = text.split("\n\n", 1)
        doc_header = header_lines[0] if len(header_lines) > 1 else ""
        body = header_lines[1] if len(header_lines) > 1 else text

        boundaries = [m.start() for m in self.SECTION_RE.finditer(body)]
        boundaries.append(len(body))

        chunks: list[Chunk] = []
        idx = 0
        for i in range(len(boundaries) - 1):
            section_text = body[boundaries[i]:boundaries[i + 1]].strip()
            if not section_text:
                continue
            section_title_match = self.SECTION_RE.match(section_text)
            section_title = section_title_match.group(0) if section_title_match else ""

            if _word_count(section_text) <= self.max_chunk_words:
                full_text = f"{doc_header}\n{section_text}".strip()
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::sem::{idx}",
                        doc_id=doc_id,
                        text=full_text,
                        metadata={**metadata, "strategy": "semantic",
                                  "chunk_index": idx, "section": section_title},
                    )
                )
                idx += 1
            else:
                # Section too long -> split on sub-clauses if present, else fixed-size.
                sub_parts = self.SUBCLAUSE_RE.split(section_text)
                sub_parts = [p.strip() for p in sub_parts if p.strip()]
                if len(sub_parts) > 1:
                    for sp in sub_parts:
                        full_text = f"{doc_header}\n{section_title}\n{sp}".strip()
                        chunks.append(
                            Chunk(
                                chunk_id=f"{doc_id}::sem::{idx}",
                                doc_id=doc_id,
                                text=full_text,
                                metadata={**metadata, "strategy": "semantic",
                                          "chunk_index": idx, "section": section_title},
                            )
                        )
                        idx += 1
                else:
                    for fc in self._fixed.chunk(doc_id, section_text, metadata):
                        fc.chunk_id = f"{doc_id}::sem::{idx}"
                        fc.metadata["strategy"] = "semantic-fallback-fixed"
                        fc.metadata["section"] = section_title
                        chunks.append(fc)
                        idx += 1
        return chunks


def load_corpus(raw_dir: str | Path) -> dict[str, str]:
    """Loads all .txt files in raw_dir. doc_id = filename stem."""
    raw_dir = Path(raw_dir)
    docs = {}
    for path in sorted(raw_dir.glob("*.txt")):
        docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


def chunk_corpus(docs: dict[str, str], strategy: str = "semantic") -> list[Chunk]:
    chunker = SemanticChunker() if strategy == "semantic" else FixedSizeChunker()
    all_chunks: list[Chunk] = []
    for doc_id, text in docs.items():
        # first line of the doc is used as a lightweight title metadata field
        title = text.strip().split("\n", 1)[0]
        all_chunks.extend(chunker.chunk(doc_id, text, metadata={"title": title}))
    return all_chunks


if __name__ == "__main__":
    corpus = load_corpus(Path(__file__).resolve().parents[2] / "data" / "raw")
    for strategy in ("fixed", "semantic"):
        chunks = chunk_corpus(corpus, strategy)
        sizes = [_word_count(c.text) for c in chunks]
        print(f"{strategy:8s} -> {len(chunks):3d} chunks | "
              f"avg {sum(sizes)/len(sizes):.0f} words | max {max(sizes)} words")
