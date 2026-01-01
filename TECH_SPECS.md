# RAG Question-Answering System - Technical Specifications

## Project Overview

**Project Name**: rag-research-qa  
**Objective**: Build a production-grade Retrieval-Augmented Generation system for question answering on AI research papers with hybrid retrieval, dual embedding comparison, and comprehensive source attribution.

**Source Documents**:
| Paper ID | Title | Key Topics |
|----------|-------|------------|
| `1706_03762v7.pdf` | Attention Is All You Need (Vaswani et al., 2017) | Transformer architecture, self-attention, positional encoding, multi-head attention |
| `2005_11401v4.pdf` | RAG for Knowledge-Intensive NLP Tasks (Lewis et al., 2020) | RAG architecture, retriever-generator interaction, knowledge-intensive tasks |
| `2005_14165v4.pdf` | Language Models are Few-Shot Learners (Brown et al., 2020) | GPT-3, few-shot learning, in-context learning, scaling laws |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG-QA SYSTEM ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │   PDF       │    │           DOCUMENT PREPROCESSING                  │   │
│  │   Files     │───▶│  ┌─────────┐  ┌──────────┐  ┌────────────────┐   │   │
│  │             │    │  │ PDF     │─▶│ Section  │─▶│ Hybrid Chunker │   │   │
│  └─────────────┘    │  │ Parser  │  │ Detector │  │ (Semantic +    │   │   │
│                     │  └─────────┘  └──────────┘  │  Fixed-size)   │   │   │
│                     │                             └────────────────┘   │   │
│                     └──────────────────────────────────────────────────┘   │
│                                          │                                  │
│                                          ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    DUAL EMBEDDING PIPELINE                            │  │
│  │  ┌────────────────────┐         ┌─────────────────────────────────┐  │  │
│  │  │  OpenAI Embeddings │         │  Open-Source Embeddings         │  │  │
│  │  │  text-embedding-   │         │  BAAI/bge-base-en-v1.5          │  │  │
│  │  │  3-small           │         │  (sentence-transformers)        │  │  │
│  │  └─────────┬──────────┘         └───────────────┬─────────────────┘  │  │
│  │            │                                    │                     │  │
│  │            ▼                                    ▼                     │  │
│  │  ┌────────────────────┐         ┌─────────────────────────────────┐  │  │
│  │  │  ChromaDB          │         │  ChromaDB                       │  │  │
│  │  │  Collection:       │         │  Collection:                    │  │  │
│  │  │  openai_embeddings │         │  opensource_embeddings          │  │  │
│  │  └────────────────────┘         └─────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                          │                                  │
│                                          ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    HYBRID RETRIEVAL ENGINE                            │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │  │
│  │  │  Dense Retrieval│  │  Sparse         │  │  Reciprocal Rank    │   │  │
│  │  │  (ChromaDB      │  │  Retrieval      │  │  Fusion (RRF)       │   │  │
│  │  │   Similarity)   │  │  (BM25)         │  │                     │   │  │
│  │  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘   │  │
│  │           │                    │                      │              │  │
│  │           └────────────────────┴──────────────────────┘              │  │
│  │                                │                                      │  │
│  │                                ▼                                      │  │
│  │                    ┌─────────────────────┐                           │  │
│  │                    │  Cross-Encoder      │                           │  │
│  │                    │  Reranker           │                           │  │
│  │                    │  (ms-marco-MiniLM)  │                           │  │
│  │                    └─────────────────────┘                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                          │                                  │
│                                          ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    ANSWER GENERATION                                  │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │  │
│  │  │  Context        │  │  LLM Generation │  │  Citation           │   │  │
│  │  │  Assembly       │─▶│  (gpt-4o-mini)  │─▶│  Formatter          │   │  │
│  │  │                 │  │                 │  │  (Inline + Struct)  │   │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                          │                                  │
│                                          ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    OUTPUT LAYER                                       │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │  │
│  │  │  FastAPI        │  │  CLI Interface  │  │  Jupyter Notebook   │   │  │
│  │  │  Endpoints      │  │                 │  │  Interface          │   │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Document Preprocessing Pipeline

#### 1.1 PDF Parser
**Library**: `pdfplumber` (primary) with `pypdf` fallback  
**Rationale**: `pdfplumber` excels at preserving layout and extracting text from academic papers with complex formatting (equations, tables, multi-column layouts).

**Extracted Metadata per Document**:
```python
@dataclass
class DocumentMetadata:
    doc_id: str              # Unique identifier (filename stem)
    title: str               # Paper title
    authors: List[str]       # Author list
    source_file: str         # Original filename
    total_pages: int         # Page count
    extraction_date: datetime
```

#### 1.2 Section Detector
**Approach**: Regex-based heuristics + font-size analysis

**Section Patterns for Academic Papers**:
```python
SECTION_PATTERNS = [
    r"^(\d+\.?\s+)?Abstract",
    r"^(\d+\.?\s+)?Introduction", 
    r"^(\d+\.?\s+)?Related Work",
    r"^(\d+\.?\s+)?Background",
    r"^(\d+\.?\s+)?Method(ology|s)?",
    r"^(\d+\.?\s+)?Model",
    r"^(\d+\.?\s+)?Architecture",
    r"^(\d+\.?\s+)?Experiment(s|al)?",
    r"^(\d+\.?\s+)?Results",
    r"^(\d+\.?\s+)?Discussion",
    r"^(\d+\.?\s+)?Conclusion(s)?",
    r"^(\d+\.?\s+)?References",
    r"^(\d+\.?\s+)?Appendix",
    r"^[A-Z]\.\s+",  # Appendix sections (A. , B. , etc.)
]
```

#### 1.3 Hybrid Chunker
**Strategy**: Semantic-first with fixed-size fallback

```python
class ChunkingConfig:
    # Semantic chunking parameters
    min_chunk_size: int = 100        # Minimum tokens per chunk
    max_chunk_size: int = 512        # Maximum tokens per chunk
    
    # Fixed-size fallback parameters  
    fallback_chunk_size: int = 400   # Tokens when semantic fails
    chunk_overlap: int = 50          # Overlap between chunks
    
    # Tokenizer
    tokenizer: str = "cl100k_base"   # OpenAI's tokenizer
```

**Chunking Algorithm**:
1. Attempt semantic split at section boundaries
2. If section > `max_chunk_size`: split at paragraph boundaries
3. If paragraph > `max_chunk_size`: apply fixed-size chunking with overlap
4. If chunk < `min_chunk_size`: merge with adjacent chunk

**Chunk Schema**:
```python
class Chunk(BaseModel):
    chunk_id: str                    # UUID
    doc_id: str                      # Parent document ID
    content: str                     # Chunk text
    token_count: int                 # Token count
    
    # Provenance metadata
    page_numbers: List[int]          # Pages this chunk spans
    section_title: Optional[str]     # Section header if detected
    section_hierarchy: List[str]     # e.g., ["3. Methods", "3.1 Attention"]
    
    # Position metadata
    chunk_index: int                 # Position in document
    start_char: int                  # Character offset start
    end_char: int                    # Character offset end
```

---

### 2. Dual Embedding System

#### 2.1 OpenAI Embeddings
**Model**: `text-embedding-3-small`  
**Dimensions**: 1536  
**Cost**: $0.00002 / 1K tokens  
**Max Input**: 8191 tokens

**Configuration**:
```python
class OpenAIEmbeddingConfig:
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100            # Batch API calls
    retry_attempts: int = 3
    retry_delay: float = 1.0
```

#### 2.2 Open-Source Embeddings
**Model**: `BAAI/bge-base-en-v1.5`  
**Dimensions**: 768  
**Rationale**: Top performer on MTEB benchmark, strong for retrieval tasks

**Alternative Options**:
| Model | Dimensions | MTEB Score | Notes |
|-------|------------|------------|-------|
| `BAAI/bge-base-en-v1.5` | 768 | 63.55 | **Selected** - Best balance |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 56.26 | Lightweight option |
| `BAAI/bge-large-en-v1.5` | 1024 | 64.23 | Higher quality, slower |

**Configuration**:
```python
class OpenSourceEmbeddingConfig:
    model_name: str = "BAAI/bge-base-en-v1.5"
    dimensions: int = 768
    device: str = "cpu"              # or "cuda" if available
    normalize_embeddings: bool = True
    batch_size: int = 32
```

#### 2.3 Embedding Comparison Framework
**Metrics**:
- **Retrieval Precision@K**: % of relevant chunks in top-K results
- **Mean Reciprocal Rank (MRR)**: Average reciprocal of first relevant result position
- **Latency**: Time to embed query and retrieve results

---

### 3. Vector Store (ChromaDB)

**Version**: `chromadb >= 0.4.0`  
**Persistence**: Local SQLite + Parquet files

**Collection Schema**:
```python
# Two separate collections for A/B comparison
collections = {
    "openai_embeddings": {
        "embedding_function": OpenAIEmbeddingFunction(),
        "metadata": {"hnsw:space": "cosine"}
    },
    "opensource_embeddings": {
        "embedding_function": SentenceTransformerEmbeddingFunction(),
        "metadata": {"hnsw:space": "cosine"}
    }
}
```

**Stored Metadata per Chunk**:
```python
metadata_schema = {
    "doc_id": str,
    "chunk_index": int,
    "page_numbers": str,        # JSON-encoded list
    "section_title": str,
    "section_hierarchy": str,   # JSON-encoded list
    "token_count": int,
    "source_file": str
}
```

---

### 4. Hybrid Retrieval Engine

#### 4.1 Dense Retrieval
**Method**: ChromaDB cosine similarity search  
**Top-K**: 20 candidates (before fusion)

#### 4.2 Sparse Retrieval (BM25)
**Library**: `rank_bm25`  
**Preprocessing**:
- Lowercase
- Remove stopwords
- Tokenize on whitespace + punctuation

**Parameters**:
```python
class BM25Config:
    k1: float = 1.5          # Term frequency saturation
    b: float = 0.75          # Length normalization
    top_k: int = 20          # Candidates before fusion
```

#### 4.3 Reciprocal Rank Fusion (RRF)
**Formula**:
```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```
Where `k = 60` (standard constant)

**Implementation**:
```python
def reciprocal_rank_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """Fuse rankings from dense and sparse retrieval."""
    scores = defaultdict(float)
    
    for rank, (doc_id, _) in enumerate(dense_results):
        scores[doc_id] += 1 / (k + rank + 1)
    
    for rank, (doc_id, _) in enumerate(sparse_results):
        scores[doc_id] += 1 / (k + rank + 1)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

#### 4.4 Cross-Encoder Reranker
**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`  
**Purpose**: Re-score top candidates for higher precision

**Configuration**:
```python
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_rerank: int = 10       # Rerank top 10 from RRF
    final_top_k: int = 5         # Return top 5 after reranking
    device: str = "cpu"
```

---

### 5. Answer Generation

#### 5.1 Context Assembly
**Strategy**: Concatenate top-K chunks with metadata separators

**Template**:
```
[Source 1: {paper_title}, Section: {section_title}, Pages: {pages}]
{chunk_content}

[Source 2: {paper_title}, Section: {section_title}, Pages: {pages}]
{chunk_content}
...
```

**Max Context Tokens**: 4000 (leaving room for system prompt + answer)

#### 5.2 LLM Configuration
**Model**: `gpt-4o-mini`  
**Temperature**: 0.1 (low for factual accuracy)  
**Max Output Tokens**: 1000

**System Prompt**:
```
You are a research assistant specialized in AI/ML papers. Answer questions based ONLY on the provided context from research papers.

CRITICAL INSTRUCTIONS:
1. If the context doesn't contain sufficient information to answer, say so explicitly.
2. Always cite your sources using the format [Paper Title, Section X.X, Page Y].
3. Use direct quotes sparingly, paraphrasing when possible.
4. Structure complex answers with clear organization.
5. Distinguish between what the paper explicitly states vs. your interpretation.
```

#### 5.3 Citation Formatting

**Inline Citation Format**:
```
The Transformer uses self-attention mechanisms where "queries, keys, and values" 
are computed from the input [Attention Is All You Need, Section 3.2, Page 3].
```

**Structured Attribution Schema**:
```python
class SourceAttribution(BaseModel):
    chunk_id: str
    paper_title: str
    section: Optional[str]
    page_numbers: List[int]
    relevance_score: float
    excerpt: str                    # Short excerpt used
    
class AnswerResponse(BaseModel):
    question: str
    answer: str                     # Contains inline citations
    sources: List[SourceAttribution]
    retrieval_method: str           # "openai" | "opensource" | "hybrid"
    confidence: float               # 0.0 - 1.0
    processing_time_ms: float
```

---

### 6. API Layer (FastAPI)

#### 6.1 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Full RAG pipeline |
| `/retrieve` | POST | Retrieval only (debugging) |
| `/compare` | POST | Compare OpenAI vs open-source retrieval |
| `/documents` | GET | List indexed documents |
| `/stats` | GET | System statistics |

#### 6.2 Request/Response Schemas

**Query Request**:
```python
class QueryRequest(BaseModel):
    question: str
    retrieval_method: Literal["openai", "opensource", "hybrid"] = "hybrid"
    top_k: int = 5
    include_sources: bool = True
    
class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceAttribution]
    metadata: Dict[str, Any]
```

---

### 7. Testing Framework

#### 7.1 Test Categories

| Category | Purpose | Tool |
|----------|---------|------|
| Unit Tests | Individual component testing | pytest |
| Integration Tests | Pipeline flow testing | pytest |
| Retrieval Quality | Measure retrieval accuracy | Custom metrics |
| End-to-End | Full system validation | pytest + sample questions |

#### 7.2 Sample Questions (Ground Truth)

```python
EVALUATION_QUESTIONS = [
    {
        "question": "What are the main components of a RAG model, and how do they interact?",
        "expected_sources": ["2005_11401v4.pdf"],
        "key_concepts": ["retriever", "generator", "DPR", "BART"]
    },
    {
        "question": "What are the two sub-layers in each encoder layer of the Transformer model?",
        "expected_sources": ["1706_03762v7.pdf"],
        "key_concepts": ["self-attention", "feed-forward", "residual", "layer normalization"]
    },
    {
        "question": "Explain how positional encoding is implemented in Transformers and why it is necessary.",
        "expected_sources": ["1706_03762v7.pdf"],
        "key_concepts": ["sinusoidal", "position", "sequence order", "no recurrence"]
    },
    {
        "question": "Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?",
        "expected_sources": ["1706_03762v7.pdf"],
        "key_concepts": ["multiple heads", "different subspaces", "parallel attention", "concatenate"]
    },
    {
        "question": "What is few-shot learning, and how does GPT-3 implement it during inference?",
        "expected_sources": ["2005_14165v4.pdf"],
        "key_concepts": ["in-context learning", "demonstrations", "no gradient updates", "prompt"]
    }
]
```

---

### 8. Project Structure

```
rag-research-qa/
├── README.md
├── TECH_SPECS.md
├── ATOMIC_STEPS.md
├── FutureFeatures.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py            # PDF text extraction
│   │   ├── section_detector.py      # Section boundary detection
│   │   └── chunker.py               # Hybrid chunking logic
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract embedding interface
│   │   ├── openai_embeddings.py     # OpenAI implementation
│   │   └── opensource_embeddings.py # Sentence-transformers implementation
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py          # ChromaDB wrapper
│   │   ├── bm25_retriever.py        # BM25 sparse retrieval
│   │   ├── hybrid_retriever.py      # RRF fusion logic
│   │   └── reranker.py              # Cross-encoder reranking
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── context_builder.py       # Context assembly
│   │   ├── llm_client.py            # OpenAI API wrapper
│   │   └── citation_formatter.py    # Citation injection
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py          # End-to-end orchestration
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py                  # FastAPI app
│       ├── routes.py                # Endpoint definitions
│       └── schemas.py               # Pydantic models
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── notebooks/
│   └── RAG_QA_System_Demo.ipynb     # Final Colab deliverable
│
├── data/
│   ├── papers/                      # Source PDFs
│   └── chroma_db/                   # Persistent vector store
│
└── scripts/
    ├── index_documents.py           # CLI to index PDFs
    ├── query_cli.py                 # CLI to query system
    └── evaluate.py                  # Run evaluation suite
```

---

### 9. Dependencies

```
# Core
python>=3.10
pydantic>=2.0
python-dotenv

# PDF Processing
pdfplumber>=0.10.0
pypdf>=3.0.0
tiktoken                            # Token counting

# Embeddings
openai>=1.0.0
sentence-transformers>=2.2.0
torch>=2.0.0

# Vector Store
chromadb>=0.4.0

# Retrieval
rank-bm25>=0.2.2

# API
fastapi>=0.100.0
uvicorn>=0.23.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0

# Utilities
tqdm
loguru
```

---

### 10. Configuration Management

**Environment Variables** (`.env`):
```bash
# OpenAI
OPENAI_API_KEY=sk-...

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# Model Settings
EMBEDDING_MODEL_OPENAI=text-embedding-3-small
EMBEDDING_MODEL_OPENSOURCE=BAAI/bge-base-en-v1.5
LLM_MODEL=gpt-4o-mini

# Retrieval Settings
RETRIEVAL_TOP_K=5
RERANK_TOP_K=10

# Logging
LOG_LEVEL=INFO
```

---

## Success Criteria

| Metric | Target |
|--------|--------|
| All 5 sample questions answered correctly | ✓ |
| Source attribution accuracy | >90% |
| Retrieval Precision@5 | >80% |
| End-to-end latency | <5 seconds |
| Test coverage | >80% |
| API endpoints functional | All 6 endpoints |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| PDF extraction quality varies | Multi-library fallback (pdfplumber → pypdf) |
| OpenAI API rate limits | Exponential backoff + batch processing |
| Cross-encoder slow on CPU | Cache reranked results; optional GPU flag |
| ChromaDB persistence issues | Regular backup scripts |
| Token limit exceeded | Dynamic context truncation |
