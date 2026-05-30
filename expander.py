"""
greprag.expander
----------------
Query Expansion: generates synonyms / related terms to widen Stage 1 recall.

Fixes the "vocabulary mismatch" problem: if a user searches for "automobile"
but documents only contain "car", a pure grep will find nothing.  By expanding
the query to ["automobile", "car", "vehicle", "auto"] *before* Stage 1, we
ensure relevant documents are not filtered out.

Two backends are supported:

WordNetExpander  (default, fully offline, zero cost)
    Uses NLTK WordNet synonym sets.  No API key required.  Fast (~1 ms).

LLMExpander  (configurable, requires a running LLM)
    Calls any OpenAI-compatible API (OpenAI, Ollama, LM Studio, etc.)
    and asks for synonym / related query variants.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseExpander(ABC):
    """Abstract base for all query-expansion backends."""

    @abstractmethod
    def expand(self, query: str, n_synonyms: int = 4) -> list[str]:
        """
        Return *query* plus up to *n_synonyms* related terms / query variants.

        The original query is always the first element of the returned list.

        Parameters
        ----------
        query:
            The user's original query string.
        n_synonyms:
            Maximum number of additional terms to generate.

        Returns
        -------
        list[str]
            At least ``[query]``; more elements when synonyms are found.
        """


# ---------------------------------------------------------------------------
# WordNet expander (offline)
# ---------------------------------------------------------------------------

class WordNetExpander(BaseExpander):
    """
    Synonym expansion via NLTK WordNet — fully offline, zero latency.

    Limitations
    -----------
    * Only expands individual keywords, not full phrases.
    * English only.
    * May produce some noisy synonyms for ambiguous words.

    Parameters
    ----------
    include_hypernyms:
        If ``True``, also include direct hypernym (parent concept) lemmas.
        E.g. "dog" → "canine", "animal".
    """

    def __init__(self, include_hypernyms: bool = False) -> None:
        self.include_hypernyms = include_hypernyms
        self._nltk_ready = False

    def expand(self, query: str, n_synonyms: int = 4) -> list[str]:
        """
        Expand *query* by adding WordNet synonyms of its keywords.

        Returns the original query as the first element, followed by
        additional query variants where individual words are substituted
        with synonyms.
        """
        self._ensure_nltk()
        from nltk.corpus import wordnet as wn  # type: ignore[import]

        words = query.lower().split()
        candidates: set[str] = set()

        for word in words:
            synsets = wn.synsets(word)
            for synset in synsets:
                lemmas = [l.name().replace("_", " ") for l in synset.lemmas()]
                candidates.update(lemmas)
                if self.include_hypernyms:
                    for hyp in synset.hypernyms():
                        candidates.update(l.name().replace("_", " ") for l in hyp.lemmas())

        # Remove the words already in the query
        query_words = set(words)
        synonyms = sorted(
            w for w in candidates if w not in query_words and w != query
        )

        results = [query]
        for syn in synonyms[:n_synonyms]:
            # Build a variant query by appending the synonym term
            results.append(syn)
        return results

    def _ensure_nltk(self) -> None:
        if self._nltk_ready:
            return
        try:
            import nltk  # type: ignore[import]
            try:
                from nltk.corpus import wordnet  # noqa: F401
                wordnet.synsets("test")  # check if data is available
            except LookupError:
                logger.info("Downloading NLTK WordNet data…")
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            self._nltk_ready = True
        except ImportError as exc:
            raise ImportError(
                "nltk is required for WordNetExpander. "
                "Install it with: pip install nltk"
            ) from exc


# ---------------------------------------------------------------------------
# LLM expander (OpenAI-compatible)
# ---------------------------------------------------------------------------

class LLMExpander(BaseExpander):
    """
    Query expansion via any OpenAI-compatible LLM API.

    Works with:
    - **OpenAI** — set ``base_url="https://api.openai.com/v1"``
    - **Ollama** — set ``base_url="http://localhost:11434/v1"``
    - **LM Studio** — set ``base_url="http://localhost:1234/v1"``
    - Any other OpenAI-API-compatible server.

    Parameters
    ----------
    base_url:
        API endpoint base URL.
    model:
        Model name (e.g. ``"llama3"``, ``"gpt-4o-mini"``).
    api_key:
        API key.  For local servers (Ollama, LM Studio) any non-empty
        string works (e.g. ``"ollama"``).
    timeout:
        HTTP request timeout in seconds.

    Example
    -------
    >>> expander = LLMExpander(
    ...     base_url="http://localhost:11434/v1",
    ...     model="llama3",
    ...     api_key="ollama",
    ... )
    >>> expander.expand("automobile engine problems", n_synonyms=4)
    ['automobile engine problems', 'car engine issues', 'vehicle motor failures', ...]
    """

    _SYSTEM_PROMPT = (
        "You are a query expansion assistant for a search engine. "
        "Given a user query, generate {n} alternative phrasings or related "
        "search queries that cover synonyms and related concepts. "
        "Return ONLY a JSON array of strings with no extra commentary. "
        "Example output: [\"query variant 1\", \"query variant 2\"]"
    )

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3",
        api_key: str = "ollama",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client = None

    def expand(self, query: str, n_synonyms: int = 4) -> list[str]:
        """
        Call the LLM to generate *n_synonyms* query variants.

        Falls back to returning ``[query]`` if the API call fails or
        times out, ensuring the pipeline never breaks due to expansion.
        """
        try:
            variants = self._call_llm(query, n_synonyms)
            # Always include the original query first
            seen: set[str] = {query}
            results = [query]
            for v in variants:
                v = v.strip()
                if v and v not in seen:
                    results.append(v)
                    seen.add(v)
            return results[:n_synonyms + 1]
        except Exception as exc:
            logger.warning(
                "LLMExpander failed (%s). Falling back to original query.", exc
            )
            return [query]

    def _call_llm(self, query: str, n: int) -> list[str]:
        client = self._get_client()
        system = self._SYSTEM_PROMPT.format(n=n)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Query: {query}"},
            ],
            temperature=0.3,
            max_tokens=256,
            timeout=self.timeout,
        )
        content = response.choices[0].message.content or ""
        return _parse_json_list(content)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for LLMExpander. "
                    "Install it with: pip install openai"
                ) from exc
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_expander(
    backend: str,
    *,
    llm_base_url: str = "http://localhost:11434/v1",
    llm_model: str = "llama3",
    llm_api_key: str = "ollama",
) -> BaseExpander:
    """
    Factory function to create an expander from a backend name string.

    Parameters
    ----------
    backend:
        One of ``"wordnet"``, ``"openai"``, or ``"ollama"``.
    llm_base_url, llm_model, llm_api_key:
        Passed to :class:`LLMExpander` when *backend* is ``"openai"``
        or ``"ollama"``.

    Returns
    -------
    BaseExpander
    """
    b = backend.lower()
    if b == "wordnet":
        return WordNetExpander()
    elif b in ("openai", "ollama", "llm"):
        return LLMExpander(
            base_url=llm_base_url,
            model=llm_model,
            api_key=llm_api_key,
        )
    else:
        raise ValueError(
            f"Unknown expander backend: {backend!r}. "
            "Choose from: 'wordnet', 'openai', 'ollama'."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JSON_LIST_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _parse_json_list(text: str) -> list[str]:
    """Extract a JSON array from LLM output, tolerating extra prose."""
    import json

    match = _JSON_LIST_RE.search(text)
    if not match:
        # Last-resort: treat each non-empty line as a variant
        return [l.strip().strip('"').strip("'") for l in text.splitlines() if l.strip()]
    try:
        parsed = json.loads(match.group())
        return [str(item) for item in parsed if isinstance(item, (str, int, float))]
    except json.JSONDecodeError:
        return []
