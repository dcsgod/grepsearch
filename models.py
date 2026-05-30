"""
greprag.models
--------------
Pydantic data contracts shared across all GrepRAG modules.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core document unit
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A single piece of text extracted from a larger document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    # position in original document (character offsets)
    start_char: int | None = None
    end_char: int | None = None

    # which chunk index within its source document
    chunk_index: int | None = None

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.id!r}, source={self.source!r}, text={preview!r}...)"


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """Holds a candidate chunk together with its retrieval scores."""

    chunk: Chunk
    bm25_score: float = 0.0
    rerank_score: float | None = None  # set after Stage 2; None if not reranked

    # rank within the final sorted list (1-indexed)
    rank: int | None = None

    @property
    def final_score(self) -> float:
        """Return the best available score."""
        return self.rerank_score if self.rerank_score is not None else self.bm25_score

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(rank={self.rank}, "
            f"rerank={self.rerank_score:.4f if self.rerank_score is not None else 'n/a'}, "
            f"bm25={self.bm25_score:.4f}, "
            f"source={self.chunk.source!r})"
        )


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """All tunable parameters for the GrepRAG two-stage pipeline."""

    # Stage 1 – Lexical retrieval
    top_k_lexical: int = Field(100, ge=1, description="Candidates returned by BM25 / grep.")

    # Stage 2 – Cross-Encoder reranking
    top_k_final: int = Field(5, ge=1, description="Final results after reranking.")
    reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace model id for the cross-encoder.",
    )
    reranker_batch_size: int = Field(32, ge=1)
    reranker_max_length: int = Field(512, ge=64)

    # Query expansion
    use_query_expansion: bool = False
    query_expansion_backend: str = Field(
        "wordnet",
        description="Backend for query expansion: 'wordnet', 'openai', or 'ollama'.",
    )
    query_expansion_n_synonyms: int = Field(4, ge=1)

    # LLM expander settings (only used when backend != 'wordnet')
    llm_base_url: str = "http://localhost:11434/v1"  # Ollama default
    llm_model: str = "llama3"
    llm_api_key: str = "ollama"  # ignored for Ollama, set for OpenAI

    # Chunking defaults (used by GrepRAG.index_text helpers)
    chunk_size: int = Field(512, ge=64)
    chunk_overlap: int = Field(64, ge=0)
