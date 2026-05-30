"""
greprag.chunker
---------------
Strategies for splitting raw text / files / document dicts into Chunk objects.

Chunking strategy: recursive sentence-aware splitting.
- Never breaks mid-sentence if avoidable.
- Supports configurable chunk_size (tokens or characters) and overlap.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from .models import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sentence boundary pattern: split on ". ", "? ", "! ", ".\n", etc.
_SENT_END_RE = re.compile(r"(?<=[.?!])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a lightweight regex."""
    parts = _SENT_END_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _char_len(text: str) -> int:
    return len(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    source: str = "",
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Split *text* into overlapping chunks of approximately *chunk_size*
    characters, preferring to break on sentence boundaries.

    Parameters
    ----------
    text:
        Raw text to chunk.
    chunk_size:
        Target maximum character length per chunk.
    overlap:
        Number of characters from the previous chunk to prepend to the
        next chunk (context window).
    source:
        An optional label (filename, URL, etc.) stored on each Chunk.
    metadata:
        Arbitrary key-value pairs attached to every chunk produced.

    Returns
    -------
    list[Chunk]
    """
    if not text or not text.strip():
        return []

    meta = metadata or {}
    sentences = _split_sentences(text)

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_len = 0
    chunk_index = 0
    char_cursor = 0

    overlap_buffer = ""

    for sent in sentences:
        sent_len = _char_len(sent)

        # If a single sentence exceeds chunk_size, hard-split it by characters
        if sent_len > chunk_size:
            # Flush whatever we have first
            if current_parts:
                body = " ".join(current_parts)
                chunks.append(
                    Chunk(
                        text=body,
                        source=source,
                        metadata=meta,
                        start_char=char_cursor - current_len,
                        end_char=char_cursor,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
                overlap_buffer = body[-overlap:] if overlap else ""
                current_parts = []
                current_len = 0

            # Hard-split the giant sentence
            for i in range(0, sent_len, chunk_size - overlap):
                slice_text = sent[i : i + chunk_size]
                if overlap_buffer:
                    slice_text = overlap_buffer + " " + slice_text
                    overlap_buffer = ""
                start = char_cursor + i
                chunks.append(
                    Chunk(
                        text=slice_text,
                        source=source,
                        metadata=meta,
                        start_char=start,
                        end_char=start + _char_len(slice_text),
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
            char_cursor += sent_len
            continue

        # Normal case: accumulate sentences until we hit chunk_size
        if current_len + sent_len + 1 > chunk_size and current_parts:
            body = " ".join(current_parts)
            chunks.append(
                Chunk(
                    text=(overlap_buffer + " " + body).strip() if overlap_buffer else body,
                    source=source,
                    metadata=meta,
                    start_char=char_cursor - current_len,
                    end_char=char_cursor,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
            overlap_buffer = body[-overlap:] if overlap else ""
            current_parts = []
            current_len = 0

        current_parts.append(sent)
        current_len += sent_len + 1  # +1 for the space
        char_cursor += sent_len + 1

    # Flush remaining sentences
    if current_parts:
        body = " ".join(current_parts)
        chunks.append(
            Chunk(
                text=(overlap_buffer + " " + body).strip() if overlap_buffer else body,
                source=source,
                metadata=meta,
                start_char=char_cursor - current_len,
                end_char=char_cursor,
                chunk_index=chunk_index,
            )
        )

    return chunks


def chunk_file(
    path: str | Path,
    chunk_size: int = 512,
    overlap: int = 64,
    encoding: str = "utf-8",
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Read a plain-text file from *path* and return its chunks.

    Parameters
    ----------
    path:
        Path to the text file.
    chunk_size, overlap:
        Forwarded to :func:`chunk_text`.
    encoding:
        File encoding (default ``"utf-8"``).
    metadata:
        Extra key-value pairs stored on every chunk. The key ``"file"``
        is automatically set to the resolved path string.

    Returns
    -------
    list[Chunk]
    """
    path = Path(path)
    text = path.read_text(encoding=encoding)
    meta = {"file": str(path.resolve()), **(metadata or {})}
    return chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
        source=str(path.name),
        metadata=meta,
    )


def chunk_documents(
    documents: list[dict[str, Any]],
    text_key: str = "text",
    source_key: str = "source",
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Chunk a list of document dicts.

    Each dict must contain at least *text_key* (default ``"text"``).
    All other keys are stored in ``Chunk.metadata``.

    Parameters
    ----------
    documents:
        List of dicts, e.g. ``[{"text": "...", "source": "doc1", ...}, ...]``
    text_key:
        Dict key that holds the body text.
    source_key:
        Dict key used as the ``Chunk.source`` label.
    chunk_size, overlap:
        Forwarded to :func:`chunk_text`.

    Returns
    -------
    list[Chunk]

    Example
    -------
    >>> docs = [{"text": "Hello world.", "source": "intro", "author": "Alice"}]
    >>> chunks = chunk_documents(docs)
    """
    all_chunks: list[Chunk] = []
    for doc in documents:
        text = doc.get(text_key, "")
        source = str(doc.get(source_key, ""))
        meta = {k: v for k, v in doc.items() if k not in (text_key, source_key)}
        all_chunks.extend(
            chunk_text(text, chunk_size=chunk_size, overlap=overlap, source=source, metadata=meta)
        )
    return all_chunks
