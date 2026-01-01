# rag-research-qa

Retrieval-Augmented Generation system for question answering on AI research papers. Features hybrid retrieval (BM25 + dense), cross-encoder reranking, dual embedding comparison, and source attribution.

## Features

- **Hybrid Chunking**: Semantic-first chunking with fixed-size fallback
- **Dual Embedding System**: Compare OpenAI vs open-source (BGE) embeddings
- **Hybrid Retrieval**: BM25 + Dense retrieval with Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: Fine-grained relevance scoring
- **Source Attribution**: Both inline citations and structured source metadata
- **FastAPI Interface**: RESTful API for integration

## Research Papers

The system is designed to answer questions about these seminal AI papers:

| Paper | Authors | Year | Topics |
|-------|---------|------|--------|
| Attention Is All You Need | Vaswani et al. | 2017 | Transformer architecture, self-attention |
| RAG for Knowledge-Intensive NLP | Lewis et al. | 2020 | Retrieval-augmented generation |
| Language Models are Few-Shot Learners | Brown et al. | 2020 | GPT-3, few-shot learning |

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/crowbarmassage/rag-research-qa.git
cd rag-research-qa

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Add Research Papers

```bash
# Copy PDFs to data/papers/
cp /path/to/1706_03762v7.pdf data/papers/
cp /path/to/2005_11401v4.pdf data/papers/
cp /path/to/2005_14165v4.pdf data/papers/
```

### 4. Index Documents

```bash
python scripts/index_documents.py
```

### 5. Run the API

```bash
uvicorn src.api.main:app --reload
```

### 6. Query the System

```bash
# Via API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is multi-head attention?"}'

# Or use the interactive CLI
python scripts/query_cli.py -i
```

## Project Structure

```
rag-research-qa/
├── src/
│   ├── config.py             # Pydantic settings
│   ├── preprocessing/        # PDF parsing, chunking
│   ├── embeddings/           # OpenAI & open-source embeddings
│   ├── retrieval/            # Vector store, BM25, hybrid retrieval
│   ├── generation/           # Context building, LLM generation
│   ├── pipeline/             # End-to-end RAG pipeline
│   └── api/                  # FastAPI endpoints
├── tests/                    # pytest test suite (59 tests)
├── data/
│   ├── papers/               # Source PDFs
│   └── chroma_db/            # Vector store persistence
└── scripts/
    ├── index_documents.py    # Index PDFs into ChromaDB
    ├── query_cli.py          # Interactive CLI
    └── evaluate.py           # Run evaluation suite
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Full RAG pipeline |
| `/retrieve` | POST | Retrieval only (debugging) |
| `/compare` | POST | Compare embedding providers |
| `/documents` | GET | List indexed documents |
| `/stats` | GET | System statistics |

## Sample Questions

Test the system with these questions:

1. What are the main components of a RAG model, and how do they interact?
2. What are the two sub-layers in each encoder layer of the Transformer model?
3. Explain how positional encoding is implemented in Transformers and why it is necessary.
4. Describe the concept of multi-head attention in the Transformer architecture.
5. What is few-shot learning, and how does GPT-3 implement it during inference?

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v
```

## Documentation

- [TECH_SPECS.md](TECH_SPECS.md) - Detailed technical specifications
- [ATOMIC_STEPS.md](ATOMIC_STEPS.md) - Implementation guide with validation checkpoints
- [FutureFeatures.md](FutureFeatures.md) - Enhancement roadmap

## License

MIT
