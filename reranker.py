"""
greprag.reranker
----------------
Stage 2 of the Two-Stage Retrieval pipeline: Cross-Encoder reranking.

The CrossEncoderReranker wraps ``sentence_transformers.CrossEncoder``.
It is designed to be loaded **once** at process start and reused across
many queries (avoids cold-start overhead).

Batching is used internally for efficient CPU/GPU utilisation.
"""

from __future__ import annotations

import logging
from typing import Sequence

from .models import RetrievalResult

logger = logging.getLogger(__name__)

# Default model: small, fast, good quality (MS MARCO fine-tuned MiniLM-L-6)
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Rerank a list of BM25 candidates using a Cross-Encoder model.

    The model takes (query, document) pairs and returns a relevance
    score between 0 and 1 (sigmoid-activated).  Candidates are sorted
    by this score and the top *top_k* are returned.

    Parameters
    ----------
    model_name:
        HuggingFace model id.  Defaults to
        ``"cross-encoder/ms-marco-MiniLM-L-6-v2"`` (~80 MB, CPU-friendly).
        For higher accuracy at the cost of speed, try
        ``"cross-encoder/ms-marco-electra-base"``.
    batch_size:
        Number of (query, document) pairs fed to the model per inference
        call.  Larger values improve throughput on GPU; 32 works well on CPU.
    max_length:
        Maximum token length of (query + document) pairs.  Pairs longer
        than this are truncated.  Must not exceed the model's positional
        embedding limit (usually 512).
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        ``None`` lets sentence-transformers auto-detect.

    Example
    -------
    >>> reranker = CrossEncoderReranker()
    >>> ranked = reranker.rerank("what is attention?", candidates, top_k=5)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Rerank *candidates* for *query* and return the top *top_k*.

        Parameters
        ----------
        query:
            The user's original query string.
        candidates:
            The output of :class:`~greprag.lexical.BM25Retriever.search`.
        top_k:
            Number of results to return after reranking.

        Returns
        -------
        list[RetrievalResult]
            Each result has ``rerank_score`` set (0–1) and ``rank``
            updated to reflect the new ordering.

        Notes
        -----
        If *candidates* is empty, an empty list is returned without
        loading the model (avoids cold-start when Stage 1 returns nothing).
        """
        if not candidates:
            return []

        # Build (query, document) pairs
        pairs = [(query, r.chunk.text) for r in candidates]

        scores = self._predict(pairs)

        # Attach scores and sort
        reranked: list[RetrievalResult] = []
        for result, score in zip(candidates, scores):
            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    bm25_score=result.bm25_score,
                    rerank_score=float(score),
                )
            )

        reranked.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)

        # Assign new ranks
        for i, r in enumerate(reranked, start=1):
            r.rank = i

        return reranked[:top_k]

    def warmup(self) -> None:
        """
        Pre-load the model weights so that the first real query is fast.
        Call this once at application startup.
        """
        _ = self._get_model()
        logger.info("CrossEncoderReranker: model '%s' warmed up.", self.model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the CrossEncoder model (loaded once, reused always)."""
        if self._model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for reranking. "
                    "Install it with: pip install sentence-transformers"
                ) from exc

            logger.info("Loading CrossEncoder model: %s", self.model_name)
            kwargs: dict = {
                "max_length": self.max_length,
            }
            if self.device:
                kwargs["device"] = self.device

            self._model = CrossEncoder(
                self.model_name,
                activation_fn=torch.nn.Sigmoid(),
                **kwargs,
            )
            logger.info("CrossEncoder loaded successfully.")
        return self._model

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run batched inference on (query, doc) pairs."""
        model = self._get_model()
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return scores.tolist()
