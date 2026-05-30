"""
greprag
-------
Two-Stage Retrieval library: BM25 lexical search + Cross-Encoder reranking.

Public API
----------
    GrepRAG          — Main pipeline class
    PipelineConfig   — Configuration dataclass
    Chunk            — Data model for a text chunk
    RetrievalResult  — Data model for a retrieval result

    chunk_text       — Chunk raw text
    chunk_file       — Chunk a file
    chunk_documents  — Chunk a list of document dicts

    BM25Retriever    — Stage 1: in-memory BM25 retriever
    GrepRetriever    — Stage 1: subprocess grep/ripgrep retriever
    CrossEncoderReranker — Stage 2: cross-encoder reranker

    WordNetExpander  — Offline query expansion (WordNet)
    LLMExpander      — LLM-based query expansion (OpenAI-compatible)
"""

from .chunker import chunk_documents, chunk_file, chunk_text
from .expander import LLMExpander, WordNetExpander
from .lexical import BM25Retriever, GrepRetriever
from .models import Chunk, PipelineConfig, RetrievalResult
from .pipeline import GrepRAG
from .reranker import CrossEncoderReranker

__all__ = [
    # Main API
    "GrepRAG",
    "PipelineConfig",
    # Data models
    "Chunk",
    "RetrievalResult",
    # Chunking
    "chunk_text",
    "chunk_file",
    "chunk_documents",
    # Retrieval backends
    "BM25Retriever",
    "GrepRetriever",
    # Reranker
    "CrossEncoderReranker",
    # Query expansion
    "WordNetExpander",
    "LLMExpander",
]

__version__ = "0.1.0"
