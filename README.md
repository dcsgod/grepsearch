# GrepRAG

> **Two-Stage Retrieval for Python — BM25 + Cross-Encoder. No vector database needed.**

GrepRAG implements the **Two-Stage Retrieval** (Hybrid Retrieval + Reranking) pattern:

| Stage | Method | Speed | Purpose |
|---|---|---|---|
| **Stage 1** | BM25 / grep | ⚡ Microseconds | Cast a wide net. High recall. |
| **Stage 2** | Cross-Encoder (BERT) | 🧠 ~50ms | Re-score top candidates. High precision. |

This gives you the best of both worlds: the speed and memory efficiency of lexical search, and the semantic accuracy of transformer-based models.

---

## Why Not Just Vector Search?

| | Vector DB (e.g. Pinecone) | GrepRAG |
|---|---|---|
| Memory | High (index must be in RAM) | Low (BM25 uses an inverted index) |
| Exact string match | ❌ Poor | ✅ Excellent |
| Semantic accuracy | ✅ Good | ✅ Good (via Cross-Encoder) |
| Dependency | Requires vector DB server | Pure Python |
| Speed (1M docs) | ~100ms | ~5ms (Stage 1) |

---

## Installation

```bash
pip install greprag
```

With optional query expansion:

```bash
# WordNet (offline, no API key)
pip install "greprag[expand-wordnet]"

# LLM-based (OpenAI / Ollama)
pip install "greprag[expand-llm]"

# Everything
pip install "greprag[all]"
```

---

## Quick Start

```python
from greprag import GrepRAG, PipelineConfig

# 1. Create the pipeline
rag = GrepRAG(PipelineConfig(top_k_lexical=100, top_k_final=5))

# 2. Index your documents (auto-chunked)
rag.index_documents([
    {"text": "Transformers use self-attention to model token relationships.", "source": "a.txt"},
    {"text": "BERT is a bidirectional transformer pre-trained on masked LM.", "source": "b.txt"},
    {"text": "Python is widely used in data science and machine learning.", "source": "c.txt"},
    # ... more documents
])

# 3. Search
results = rag.search("how does attention work in transformers?")

for r in results:
    print(f"[{r.rank}] score={r.rerank_score:.3f}  source={r.chunk.source}")
    print(f"       {r.chunk.text[:100]}\n")
```

---

## API Reference

### `GrepRAG` — Main Pipeline

```python
from greprag import GrepRAG, PipelineConfig

rag = GrepRAG(config=PipelineConfig(...))
```

#### Indexing

| Method | Description |
|---|---|
| `rag.index(chunks)` | Index a list of `Chunk` objects |
| `rag.index_text(text, source)` | Chunk a raw string and index it |
| `rag.index_file(path)` | Read a file, chunk it, and index it |
| `rag.index_documents(docs)` | Chunk a list of dicts and index them |

All indexing methods return `self` for chaining.

#### Search

```python
results: list[RetrievalResult] = rag.search(
    query="your question",
    top_k=5,              # number of final results (default: config.top_k_final)
    top_k_lexical=100,    # BM25 candidates (default: config.top_k_lexical)
    skip_reranking=False, # set True for BM25-only, ultra-fast mode
)
```

#### File Search (no pre-indexing needed)

```python
results = rag.search_files(
    root="/path/to/your/codebase",
    query="how is authentication handled?",
    file_patterns=["*.py", "*.md"],  # optional
    top_k=5,
)
```

#### Persistence

```python
rag.save_index("my_index.pkl")
rag = GrepRAG.load_index("my_index.pkl", config=config)
```

---

### `PipelineConfig`

```python
from greprag import PipelineConfig

config = PipelineConfig(
    # Stage 1
    top_k_lexical=100,

    # Stage 2
    top_k_final=5,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    reranker_batch_size=32,
    reranker_max_length=512,

    # Query expansion
    use_query_expansion=False,
    query_expansion_backend="wordnet",  # or "openai" / "ollama"
    query_expansion_n_synonyms=4,

    # LLM expander (when backend != "wordnet")
    llm_base_url="http://localhost:11434/v1",
    llm_model="llama3",
    llm_api_key="ollama",

    # Chunking
    chunk_size=512,
    chunk_overlap=64,
)
```

---

### Chunking

```python
from greprag import chunk_text, chunk_file, chunk_documents

# Raw string
chunks = chunk_text("Your long document...", chunk_size=512, overlap=64, source="doc.txt")

# File
chunks = chunk_file("/path/to/file.txt")

# List of dicts
chunks = chunk_documents([
    {"text": "...", "source": "doc1", "author": "Alice"},
])
```

---

### Query Expansion

Fix the **vocabulary trap** — if your corpus says "car" but the user searches "automobile":

```python
from greprag import GrepRAG, PipelineConfig

# Offline (WordNet) — no API key, ~1ms
config = PipelineConfig(
    use_query_expansion=True,
    query_expansion_backend="wordnet",
)

# LLM-based (Ollama) — richer synonyms, requires local Ollama
config = PipelineConfig(
    use_query_expansion=True,
    query_expansion_backend="ollama",
    llm_base_url="http://localhost:11434/v1",
    llm_model="llama3",
)

rag = GrepRAG(config)
```

---

### Using Components Directly

```python
from greprag import BM25Retriever, CrossEncoderReranker, chunk_text

# Stage 1 only
retriever = BM25Retriever()
retriever.index(chunk_text(my_text))
candidates = retriever.search("attention mechanism", top_k=100)

# Stage 2 only
reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker.warmup()  # pre-load model at startup
results = reranker.rerank("attention mechanism", candidates, top_k=5)

# Grep-based retrieval (file search, no indexing)
from greprag import GrepRetriever
grep = GrepRetriever("/path/to/repo", file_patterns=["*.py"])
candidates = grep.search("def authenticate", top_k=50)
```

---

## The Architecture in Detail

```
User Query
    │
    ▼
┌─────────────────────────┐
│  Query Expansion        │  (optional)
│  "car" → ["car",        │  WordNet or LLM
│   "automobile", "vehicle"]│
└────────────┬────────────┘
             │ expanded queries
             ▼
┌─────────────────────────┐
│  Stage 1: BM25          │  Fast lexical search
│  top_k_lexical = 100    │  Returns 100 candidates
│  ~1–5ms                 │  High recall
└────────────┬────────────┘
             │ 100 candidates
             ▼
┌─────────────────────────┐
│  Stage 2: Cross-Encoder │  Semantic reranking
│  top_k_final = 5        │  (query, doc) → score 0–1
│  ~20–80ms               │  High precision
└────────────┬────────────┘
             │ top 5 results
             ▼
        LLM Prompt
```

---

## Reranker Model Options

| Model | Size | Speed | Quality |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` *(default)* | ~80 MB | ⚡ Fast | ✅ Good |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~130 MB | Medium | ✅✅ Better |
| `cross-encoder/ms-marco-electra-base` | ~440 MB | 🐢 Slower | ✅✅✅ Best |

---

## Running Tests

```bash
pip install "greprag[all]"
pytest tests/

# Skip model download tests (for CI without internet access)
GREPRAG_SKIP_MODEL_TESTS=1 pytest tests/
```

---

## License

MIT
