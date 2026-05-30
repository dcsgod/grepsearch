"""
greprag.lexical
---------------
Stage 1 of the Two-Stage Retrieval pipeline: fast lexical search.

Two backends are provided:

BM25Retriever
    In-memory BM25 index (via rank-bm25).  Best for corpora already
    loaded as Chunk objects.

GrepRetriever
    Subprocess-based grep backend.  Best for large file-based corpora
    where you do not want to load everything into memory.
"""

from __future__ import annotations

import logging
import pickle
import re
import subprocess
from pathlib import Path
from typing import Sequence

from rank_bm25 import BM25Okapi

from .models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")
_STOPWORDS = frozenset(
    """a an the is are was were be been being have has had do does did
    will would could should may might shall can need dare ought used
    to of in on at by for with as from into through during before after
    above below between among""".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords, split on whitespace."""
    text = _PUNCT_RE.sub(" ", text.lower())
    return [w for w in text.split() if w and w not in _STOPWORDS]


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    In-memory BM25 retriever backed by ``rank_bm25.BM25Okapi``.

    Usage
    -----
    >>> retriever = BM25Retriever()
    >>> retriever.index(chunks)
    >>> results = retriever.search("what is attention in transformers", top_k=50)
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._indexed = False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: Sequence[Chunk]) -> None:
        """
        Build (or rebuild) the BM25 index from *chunks*.

        Parameters
        ----------
        chunks:
            Any sequence of :class:`~greprag.models.Chunk` objects.
        """
        if not chunks:
            raise ValueError("Cannot index an empty list of chunks.")

        self._chunks = list(chunks)
        tokenized = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._indexed = True
        logger.info("BM25Retriever indexed %d chunks.", len(self._chunks))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 100) -> list[RetrievalResult]:
        """
        Return the top-k chunks ranked by BM25 score.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Maximum number of candidates to return.

        Returns
        -------
        list[RetrievalResult]
            Sorted by descending BM25 score.  Each result has
            ``rerank_score=None`` until Stage 2 processes it.
        """
        self._require_index()
        tokens = _tokenize(query)
        if not tokens:
            logger.warning("BM25Retriever: query tokenised to empty list.")
            return []

        scores: list[float] = self._bm25.get_scores(tokens).tolist()

        # Pair (score, chunk) and sort descending
        paired = sorted(
            zip(scores, self._chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        top = paired[:top_k]

        results: list[RetrievalResult] = []
        for rank, (score, chunk) in enumerate(top, start=1):
            results.append(
                RetrievalResult(chunk=chunk, bm25_score=score, rank=rank)
            )
        return results

    def search_multi(
        self,
        queries: list[str],
        top_k: int = 100,
        fusion: str = "rrf",
    ) -> list[RetrievalResult]:
        """
        Search with multiple query strings (e.g. after query expansion) and
        fuse the ranked lists.

        Parameters
        ----------
        queries:
            List of query strings (original + expanded synonyms).
        top_k:
            Candidates to return after fusion.
        fusion:
            Fusion strategy.  ``"rrf"`` (Reciprocal Rank Fusion, default)
            or ``"max"`` (take max BM25 score per chunk).

        Returns
        -------
        list[RetrievalResult]
        """
        self._require_index()
        if not queries:
            return []

        # Collect per-query ranked lists
        all_results: list[list[RetrievalResult]] = [
            self.search(q, top_k=top_k) for q in queries
        ]

        if fusion == "rrf":
            return _reciprocal_rank_fusion(all_results, top_k=top_k)
        else:
            # "max": keep highest BM25 score per chunk id
            best: dict[str, RetrievalResult] = {}
            for result_list in all_results:
                for r in result_list:
                    cid = r.chunk.id
                    if cid not in best or r.bm25_score > best[cid].bm25_score:
                        best[cid] = r
            sorted_results = sorted(best.values(), key=lambda x: x.bm25_score, reverse=True)
            for i, r in enumerate(sorted_results[:top_k], start=1):
                r.rank = i
            return sorted_results[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the index and chunks to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, fh)
        logger.info("BM25Retriever saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        """Load a pickled index from *path* and return a ready retriever."""
        path = Path(path)
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls()
        obj._chunks = data["chunks"]
        obj._bm25 = data["bm25"]
        obj._indexed = True
        logger.info("BM25Retriever loaded from %s (%d chunks)", path, len(obj._chunks))
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_index(self) -> None:
        if not self._indexed:
            raise RuntimeError(
                "BM25Retriever has no index. Call .index(chunks) first."
            )

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# Grep Retriever (file-based)
# ---------------------------------------------------------------------------

class GrepRetriever:
    """
    Subprocess-based grep retriever for file-based corpora.

    This is the most memory-efficient option for large local codebases
    or document collections: it runs ``grep`` (or ``rg``/ripgrep if
    available) as a subprocess and streams matching lines back.

    Each matching line becomes its own single-line Chunk.  Use together
    with a :class:`~greprag.reranker.CrossEncoderReranker` to produce
    meaningful results from line-level hits.

    Parameters
    ----------
    root:
        Directory to search recursively.
    use_ripgrep:
        Prefer ``rg`` (ripgrep) over ``grep`` when available.
        Ripgrep is significantly faster for large repos.
    file_patterns:
        Glob patterns passed to ``--include`` / ``--glob``.
        E.g. ``["*.py", "*.md"]``.  Empty means all files.
    context_lines:
        Number of surrounding lines to include for context
        (``-C`` / ``--context`` flag).  Default: 2.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        use_ripgrep: bool = True,
        file_patterns: list[str] | None = None,
        context_lines: int = 2,
    ) -> None:
        self.root = Path(root)
        self.use_ripgrep = use_ripgrep and _ripgrep_available()
        self.file_patterns = file_patterns or []
        self.context_lines = context_lines

    def search(self, query: str, top_k: int = 100) -> list[RetrievalResult]:
        """
        Run a case-insensitive grep for *query* under ``self.root``.

        Parameters
        ----------
        query:
            The string to grep for.
        top_k:
            Maximum number of matching blocks to return.

        Returns
        -------
        list[RetrievalResult]
            Each result wraps one grep match block.  BM25 scores are
            not available (set to 1.0 as a placeholder).
        """
        raw_lines = self._run_grep(query)
        chunks = _lines_to_chunks(raw_lines, top_k=top_k)
        return [
            RetrievalResult(chunk=c, bm25_score=1.0, rank=i + 1)
            for i, c in enumerate(chunks)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_grep(self, query: str) -> list[str]:
        if self.use_ripgrep:
            cmd = ["rg", "--ignore-case", f"--context={self.context_lines}", query, str(self.root)]
            for pat in self.file_patterns:
                cmd += ["--glob", pat]
        else:
            cmd = ["grep", "-r", "-i", f"--context={self.context_lines}", query, str(self.root)]
            for pat in self.file_patterns:
                cmd += ["--include", pat]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.splitlines()
        except FileNotFoundError:
            logger.warning(
                "grep/rg not found. Falling back to Python string search."
            )
            return self._python_search(query)
        except subprocess.TimeoutExpired:
            logger.warning("grep timed out after 30 s.")
            return []

    def _python_search(self, query: str) -> list[str]:
        """Pure-Python fallback: walk files and find matching lines."""
        matches: list[str] = []
        q_lower = query.lower()
        for fpath in self.root.rglob("*"):
            if not fpath.is_file():
                continue
            if self.file_patterns and not any(
                fpath.match(p) for p in self.file_patterns
            ):
                continue
            try:
                lines = fpath.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines):
                if q_lower in line.lower():
                    start = max(0, i - self.context_lines)
                    end = min(len(lines), i + self.context_lines + 1)
                    block = "\n".join(lines[start:end])
                    matches.append(f"{fpath}:{i+1}:{block}")
        return matches


def _ripgrep_available() -> bool:
    try:
        subprocess.run(["rg", "--version"], capture_output=True, timeout=3)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _lines_to_chunks(lines: list[str], top_k: int) -> list[Chunk]:
    """Group contiguous grep output lines into Chunk objects."""
    chunks: list[Chunk] = []
    current_block: list[str] = []
    current_source = ""

    for line in lines:
        if line.strip() == "--":
            # grep separator between match blocks
            if current_block:
                chunks.append(
                    Chunk(
                        text="\n".join(current_block),
                        source=current_source,
                        metadata={"backend": "grep"},
                    )
                )
                current_block = []
                current_source = ""
            if len(chunks) >= top_k:
                break
        else:
            # Try to extract source path from "path:linenum:content"
            parts = line.split(":", 2)
            if len(parts) >= 3:
                if not current_source:
                    current_source = parts[0]
                current_block.append(parts[-1])
            else:
                current_block.append(line)

    if current_block and len(chunks) < top_k:
        chunks.append(
            Chunk(
                text="\n".join(current_block),
                source=current_source,
                metadata={"backend": "grep"},
            )
        )

    return chunks[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion helper
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    top_k: int = 100,
    k: int = 60,
) -> list[RetrievalResult]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank_i)) for each list where the document appears.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for r in result_list:
            cid = r.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + (r.rank or 1))
            chunk_map[cid] = r  # keep most recent copy

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    fused: list[RetrievalResult] = []
    for i, cid in enumerate(sorted_ids[:top_k], start=1):
        r = chunk_map[cid]
        fused.append(
            RetrievalResult(
                chunk=r.chunk,
                bm25_score=scores[cid],  # store RRF score here
                rank=i,
            )
        )
    return fused
