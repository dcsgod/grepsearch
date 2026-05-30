"""
greprag.pipeline
----------------
The GrepRAG pipeline: orchestrates chunking, Stage 1, and Stage 2.

This is the main user-facing class.  For most use cases you only need:

    from greprag import GrepRAG, PipelineConfig

    rag = GrepRAG()
    rag.index(chunks)
    results = rag.search("your question here")

Advanced usage (query expansion, file-based search, custom config):
    see class docstring and examples/basic_usage.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Sequence

from .chunker import chunk_documents, chunk_file, chunk_text
from .expander import BaseExpander, make_expander
from .lexical import BM25Retriever, GrepRetriever
from .models import Chunk, PipelineConfig, RetrievalResult
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class GrepRAG:
    """
    Two-Stage Retrieval pipeline.

    Stage 1 — Lexical filter (BM25):
        Fast, memory-efficient keyword search.  Returns top-100 candidates.

    Stage 2 — Cross-Encoder reranking:
        Semantic scoring of Stage 1 candidates.  Returns top-5 final results.

    Optionally, query expansion (WordNet or LLM) is applied before Stage 1
    to fix the vocabulary-mismatch ("automobile" vs "car") problem.

    Parameters
    ----------
    config:
        A :class:`~greprag.models.PipelineConfig` instance.  If ``None``,
        default values are used.

    Example
    -------
    >>> from greprag import GrepRAG, PipelineConfig, chunk_text
    >>>
    >>> rag = GrepRAG(PipelineConfig(use_query_expansion=True))
    >>> chunks = chunk_text(my_long_document, source="my_doc.txt")
    >>> rag.index(chunks)
    >>>
    >>> results = rag.search("how do transformers handle long sequences?")
    >>> for r in results:
    ...     print(f"[{r.rank}] score={r.rerank_score:.3f}  {r.chunk.text[:80]}")
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._retriever = BM25Retriever()
        self._reranker = CrossEncoderReranker(
            model_name=self.config.reranker_model,
            batch_size=self.config.reranker_batch_size,
            max_length=self.config.reranker_max_length,
        )
        self._expander: BaseExpander | None = None
        if self.config.use_query_expansion:
            self._expander = make_expander(
                self.config.query_expansion_backend,
                llm_base_url=self.config.llm_base_url,
                llm_model=self.config.llm_model,
                llm_api_key=self.config.llm_api_key,
            )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: Sequence[Chunk]) -> "GrepRAG":
        """
        Build the Stage 1 BM25 index from *chunks*.

        Parameters
        ----------
        chunks:
            Any sequence of :class:`~greprag.models.Chunk` objects.
            Use :func:`~greprag.chunker.chunk_text`,
            :func:`~greprag.chunker.chunk_file`, or
            :func:`~greprag.chunker.chunk_documents` to produce them.

        Returns
        -------
        GrepRAG
            Returns ``self`` for method chaining.
        """
        self._retriever.index(chunks)
        logger.info("GrepRAG indexed %d chunks.", len(chunks))
        return self

    def index_text(
        self,
        text: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "GrepRAG":
        """
        Convenience method: chunk *text* and index it in one call.

        Parameters
        ----------
        text:
            Raw text to chunk and index.
        source:
            Label for the source (e.g. a filename or URL).
        metadata:
            Extra key-value pairs stored on each chunk.

        Returns
        -------
        GrepRAG
        """
        chunks = chunk_text(
            text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            source=source,
            metadata=metadata,
        )
        return self.index(chunks)

    def index_file(self, path: str | Path) -> "GrepRAG":
        """
        Convenience method: read *path*, chunk it, and index it.

        Parameters
        ----------
        path:
            Path to a plain-text file.

        Returns
        -------
        GrepRAG
        """
        chunks = chunk_file(
            path,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        return self.index(chunks)

    def index_documents(
        self,
        documents: list[dict[str, Any]],
        text_key: str = "text",
        source_key: str = "source",
    ) -> "GrepRAG":
        """
        Convenience method: chunk a list of document dicts and index them.

        Parameters
        ----------
        documents:
            List of dicts, each containing at least *text_key*.
        text_key:
            Key for the body text (default: ``"text"``).
        source_key:
            Key used as the source label (default: ``"source"``).

        Returns
        -------
        GrepRAG
        """
        chunks = chunk_documents(
            documents,
            text_key=text_key,
            source_key=source_key,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        return self.index(chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        top_k_lexical: int | None = None,
        skip_reranking: bool = False,
    ) -> list[RetrievalResult]:
        """
        Run the full two-stage retrieval pipeline for *query*.

        Pipeline steps
        --------------
        1. *(Optional)* Expand the query with synonyms.
        2. Stage 1: BM25 search → top ``top_k_lexical`` candidates.
        3. Stage 2: Cross-Encoder reranking → top ``top_k`` final results.

        Parameters
        ----------
        query:
            The user's natural-language query.
        top_k:
            Number of final results to return.  Defaults to
            ``config.top_k_final``.
        top_k_lexical:
            Number of candidates from Stage 1.  Defaults to
            ``config.top_k_lexical``.
        skip_reranking:
            If ``True``, skip Stage 2 and return BM25 results directly.
            Useful for debugging or when ultra-low latency is required.

        Returns
        -------
        list[RetrievalResult]
            Sorted by ``rerank_score`` (or ``bm25_score`` if reranking
            was skipped).

        Raises
        ------
        RuntimeError
            If the BM25 index has not been built yet.
        """
        final_k = top_k if top_k is not None else self.config.top_k_final
        lexical_k = top_k_lexical if top_k_lexical is not None else self.config.top_k_lexical

        t0 = time.perf_counter()

        # --- Query Expansion ---
        queries = self._expand_query(query)
        t_expand = time.perf_counter() - t0

        # --- Stage 1: Lexical Retrieval ---
        t1 = time.perf_counter()
        candidates = self._stage1(queries, lexical_k)
        t_stage1 = time.perf_counter() - t1

        if not candidates:
            logger.warning("Stage 1 returned 0 candidates for query: %r", query)
            return []

        if skip_reranking:
            return candidates[:final_k]

        # --- Stage 2: Cross-Encoder Reranking ---
        t2 = time.perf_counter()
        try:
            results = self._reranker.rerank(query, candidates, top_k=final_k)
            t_stage2 = time.perf_counter() - t2
        except Exception as exc:
            logger.error(
                "Stage 2 reranking failed (%s). Falling back to BM25 results.", exc
            )
            results = candidates[:final_k]
            t_stage2 = 0.0

        total = time.perf_counter() - t0
        logger.debug(
            "search(%r): expand=%.1fms  stage1=%.1fms  stage2=%.1fms  total=%.1fms  "
            "candidates=%d  results=%d",
            query,
            t_expand * 1000,
            t_stage1 * 1000,
            t_stage2 * 1000,
            total * 1000,
            len(candidates),
            len(results),
        )
        return results

    def search_files(
        self,
        root: str | Path,
        query: str,
        top_k: int | None = None,
        file_patterns: list[str] | None = None,
        use_ripgrep: bool = True,
    ) -> list[RetrievalResult]:
        """
        Search a directory of files using the Grep backend (Stage 1) +
        Cross-Encoder reranking (Stage 2).

        No prior indexing needed — grep scans files on-the-fly.

        Parameters
        ----------
        root:
            Directory to search recursively.
        query:
            The user's query string.
        top_k:
            Number of final results.  Defaults to ``config.top_k_final``.
        file_patterns:
            Optional list of glob patterns (e.g. ``["*.py", "*.md"]``).
        use_ripgrep:
            Prefer ripgrep (``rg``) over grep when available.

        Returns
        -------
        list[RetrievalResult]
        """
        final_k = top_k if top_k is not None else self.config.top_k_final

        grep_retriever = GrepRetriever(
            root=root,
            use_ripgrep=use_ripgrep,
            file_patterns=file_patterns or [],
        )

        # Expand query terms for multi-term grep
        queries = self._expand_query(query)

        # Run grep for each expanded query, collect unique chunks
        all_candidates: list[RetrievalResult] = []
        seen_ids: set[str] = set()
        for q in queries:
            for r in grep_retriever.search(q, top_k=self.config.top_k_lexical):
                if r.chunk.id not in seen_ids:
                    all_candidates.append(r)
                    seen_ids.add(r.chunk.id)

        if not all_candidates:
            return []

        try:
            return self._reranker.rerank(query, all_candidates, top_k=final_k)
        except Exception as exc:
            logger.error("Reranking failed for search_files (%s). Returning grep results.", exc)
            return all_candidates[:final_k]

    # ------------------------------------------------------------------
    # Persistence (BM25 index)
    # ------------------------------------------------------------------

    def save_index(self, path: str | Path) -> None:
        """
        Persist the BM25 index to *path* (pickle format).

        Parameters
        ----------
        path:
            File path to write the index to (e.g. ``"my_index.pkl"``).
        """
        self._retriever.save(path)

    @classmethod
    def load_index(
        cls,
        path: str | Path,
        config: PipelineConfig | None = None,
    ) -> "GrepRAG":
        """
        Create a new :class:`GrepRAG` instance with an index loaded from *path*.

        Parameters
        ----------
        path:
            Path previously written by :meth:`save_index`.
        config:
            Optional :class:`PipelineConfig`.  If ``None``, defaults are used.

        Returns
        -------
        GrepRAG
        """
        instance = cls(config=config)
        instance._retriever = BM25Retriever.load(path)
        return instance

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self) -> "GrepRAG":
        """
        Pre-load the Cross-Encoder model weights.

        Call this once at application startup to avoid cold-start latency
        on the first real query.

        Returns
        -------
        GrepRAG
        """
        self._reranker.warmup()
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand_query(self, query: str) -> list[str]:
        """Return [query] or [query, synonym1, ...] if expansion is enabled."""
        if self._expander is None:
            return [query]
        try:
            return self._expander.expand(query, n_synonyms=self.config.query_expansion_n_synonyms)
        except Exception as exc:
            logger.warning("Query expansion failed (%s). Using original query.", exc)
            return [query]

    def _stage1(self, queries: list[str], top_k: int) -> list[RetrievalResult]:
        """Run Stage 1 BM25 retrieval, with multi-query fusion if needed."""
        if len(queries) == 1:
            return self._retriever.search(queries[0], top_k=top_k)
        return self._retriever.search_multi(queries, top_k=top_k, fusion="rrf")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        indexed = self._retriever.chunk_count if self._retriever._indexed else 0
        return (
            f"GrepRAG(chunks_indexed={indexed}, "
            f"reranker={self.config.reranker_model!r}, "
            f"query_expansion={self.config.use_query_expansion})"
        )
