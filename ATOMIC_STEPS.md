# RAG-QA System - Atomic Implementation Steps

## Overview

**Total Phases**: 10  
**Total Steps**: 42  
**Estimated Time**: 15-20 hours

Each step includes:
- ✅ **Validation Checkpoint**: How to verify the step is complete
- 🧪 **Test Command**: pytest command to run for that step

---

## Phase 1: Project Setup & Infrastructure (Steps 1-5)

### Step 1: Initialize Project Structure
**Action**: Create directory structure and initialize Python project

```bash
mkdir -p rag-qa-system/{src/{preprocessing,embeddings,retrieval,generation,pipeline,api},tests,notebooks,data/{papers,chroma_db},scripts}
cd rag-qa-system
touch src/__init__.py src/preprocessing/__init__.py src/embeddings/__init__.py 
touch src/retrieval/__init__.py src/generation/__init__.py src/pipeline/__init__.py src/api/__init__.py
touch tests/__init__.py tests/conftest.py
```

✅ **Validation**: All directories exist, `__init__.py` files present  
🧪 **Test**: `ls -la src/` shows all subdirectories

---

### Step 2: Create pyproject.toml and requirements.txt
**Action**: Define project metadata and dependencies

**pyproject.toml**:
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "rag-qa-system"
version = "0.1.0"
description = "RAG Question-Answering System for AI Research Papers"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

**requirements.txt**: (See TECH_SPECS.md Section 9)

✅ **Validation**: `pip install -r requirements.txt` succeeds  
🧪 **Test**: `python -c "import pdfplumber, chromadb, openai; print('OK')"`

---

### Step 3: Create Configuration Module
**Action**: Build centralized configuration with Pydantic Settings

**File**: `src/config.py`

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    # Paths
    project_root: Path = Field(default=Path(__file__).parent.parent)
    data_dir: Path = Field(default=Path("data"))
    papers_dir: Path = Field(default=Path("data/papers"))
    chroma_dir: Path = Field(default=Path("data/chroma_db"))
    
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model_openai: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    
    # Open-source embeddings
    embedding_model_opensource: str = "BAAI/bge-base-en-v1.5"
    
    # Chunking
    max_chunk_size: int = 512
    min_chunk_size: int = 100
    chunk_overlap: int = 50
    
    # Retrieval
    retrieval_top_k: int = 20
    rerank_top_k: int = 10
    final_top_k: int = 5
    
    # Generation
    llm_temperature: float = 0.1
    max_output_tokens: int = 1000
    max_context_tokens: int = 4000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

✅ **Validation**: `from src.config import settings` loads without error  
🧪 **Test**: `pytest tests/test_config.py -v`

---

### Step 4: Create .env.example and .gitignore
**Action**: Environment template and git exclusions

**.env.example**:
```bash
OPENAI_API_KEY=sk-your-key-here
```

**.gitignore**:
```
# Environment
.env
.venv/
venv/

# Data
data/chroma_db/
*.pdf

# Python
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/

# Notebooks
.ipynb_checkpoints/
```

✅ **Validation**: Files created, `.env` not tracked  
🧪 **Test**: `git status` shows `.env` ignored

---

### Step 5: Copy Research Papers to Data Directory
**Action**: Move PDFs to `data/papers/`

```bash
cp /path/to/1706_03762v7.pdf data/papers/
cp /path/to/2005_11401v4.pdf data/papers/
cp /path/to/2005_14165v4.pdf data/papers/
```

✅ **Validation**: `ls data/papers/` shows 3 PDF files  
🧪 **Test**: `pytest tests/test_data_setup.py -v`

---

## Phase 2: PDF Parsing & Text Extraction (Steps 6-10)

### Step 6: Create PDF Parser Base Class
**Action**: Abstract interface for PDF text extraction

**File**: `src/preprocessing/pdf_parser.py`

```python
from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class PageContent(BaseModel):
    page_number: int
    text: str
    
class DocumentMetadata(BaseModel):
    doc_id: str
    title: str
    authors: List[str] = []
    source_file: str
    total_pages: int
    extraction_date: datetime = Field(default_factory=datetime.now)

class ExtractedDocument(BaseModel):
    metadata: DocumentMetadata
    pages: List[PageContent]
    full_text: str

class PDFParser(ABC):
    @abstractmethod
    def extract(self, pdf_path: Path) -> ExtractedDocument:
        pass
```

✅ **Validation**: Class imports without error  
🧪 **Test**: `pytest tests/test_preprocessing.py::test_pdf_parser_interface -v`

---

### Step 7: Implement pdfplumber Extraction
**Action**: Primary PDF extraction using pdfplumber

**File**: `src/preprocessing/pdf_parser.py` (add to existing)

```python
import pdfplumber
from loguru import logger

class PdfPlumberParser(PDFParser):
    def extract(self, pdf_path: Path) -> ExtractedDocument:
        pages = []
        full_text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(PageContent(page_number=i + 1, text=text))
                full_text_parts.append(text)
        
        # Extract title from first page (heuristic: first line or bold text)
        title = self._extract_title(pages[0].text if pages else "")
        
        metadata = DocumentMetadata(
            doc_id=pdf_path.stem,
            title=title,
            source_file=pdf_path.name,
            total_pages=total_pages
        )
        
        return ExtractedDocument(
            metadata=metadata,
            pages=pages,
            full_text="\n\n".join(full_text_parts)
        )
    
    def _extract_title(self, first_page_text: str) -> str:
        lines = first_page_text.strip().split("\n")
        # Heuristic: title is usually the first non-empty line
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 10 and not line.startswith("arXiv"):
                return line
        return "Unknown Title"
```

✅ **Validation**: Parse all 3 PDFs successfully  
🧪 **Test**: `pytest tests/test_preprocessing.py::test_pdfplumber_extraction -v`

---

### Step 8: Implement pypdf Fallback Parser
**Action**: Fallback extraction for problematic PDFs

```python
from pypdf import PdfReader

class PyPDFParser(PDFParser):
    def extract(self, pdf_path: Path) -> ExtractedDocument:
        reader = PdfReader(pdf_path)
        pages = []
        full_text_parts = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(PageContent(page_number=i + 1, text=text))
            full_text_parts.append(text)
        
        title = self._extract_title(pages[0].text if pages else "")
        
        metadata = DocumentMetadata(
            doc_id=pdf_path.stem,
            title=title,
            source_file=pdf_path.name,
            total_pages=len(reader.pages)
        )
        
        return ExtractedDocument(
            metadata=metadata,
            pages=pages,
            full_text="\n\n".join(full_text_parts)
        )
```

✅ **Validation**: Fallback parser works on same PDFs  
🧪 **Test**: `pytest tests/test_preprocessing.py::test_pypdf_fallback -v`

---

### Step 9: Create Composite Parser with Fallback Logic
**Action**: Smart parser that tries pdfplumber first, falls back to pypdf

```python
class CompositeParser(PDFParser):
    def __init__(self):
        self.primary = PdfPlumberParser()
        self.fallback = PyPDFParser()
    
    def extract(self, pdf_path: Path) -> ExtractedDocument:
        try:
            result = self.primary.extract(pdf_path)
            if len(result.full_text.strip()) < 100:
                logger.warning(f"Primary extraction yielded little text, trying fallback")
                return self.fallback.extract(pdf_path)
            return result
        except Exception as e:
            logger.warning(f"Primary parser failed: {e}, using fallback")
            return self.fallback.extract(pdf_path)
```

✅ **Validation**: Composite parser handles edge cases  
🧪 **Test**: `pytest tests/test_preprocessing.py::test_composite_parser -v`

---

### Step 10: Write PDF Parser Unit Tests
**Action**: Comprehensive tests for parsing

**File**: `tests/test_preprocessing.py`

```python
import pytest
from pathlib import Path
from src.preprocessing.pdf_parser import CompositeParser, ExtractedDocument

@pytest.fixture
def parser():
    return CompositeParser()

@pytest.fixture  
def sample_pdf(tmp_path):
    # Returns path to test PDF
    return Path("data/papers/1706_03762v7.pdf")

def test_extract_returns_document(parser, sample_pdf):
    result = parser.extract(sample_pdf)
    assert isinstance(result, ExtractedDocument)
    assert result.metadata.total_pages > 0
    assert len(result.full_text) > 1000

def test_metadata_populated(parser, sample_pdf):
    result = parser.extract(sample_pdf)
    assert result.metadata.doc_id == "1706_03762v7"
    assert "Attention" in result.metadata.title or len(result.metadata.title) > 0

def test_all_papers_parse(parser):
    papers_dir = Path("data/papers")
    for pdf_file in papers_dir.glob("*.pdf"):
        result = parser.extract(pdf_file)
        assert len(result.pages) > 0
```

✅ **Validation**: All tests pass  
🧪 **Test**: `pytest tests/test_preprocessing.py -v`

---

## Phase 3: Section Detection & Chunking (Steps 11-16)

### Step 11: Create Section Detector
**Action**: Regex-based section boundary detection

**File**: `src/preprocessing/section_detector.py`

```python
import re
from typing import List, Tuple, Optional
from pydantic import BaseModel

class Section(BaseModel):
    title: str
    level: int                    # 1 = main, 2 = subsection, etc.
    start_char: int
    end_char: int
    content: str
    page_start: int
    page_end: int

class SectionDetector:
    SECTION_PATTERNS = [
        (r"^(\d+)\s+([A-Z][A-Za-z\s]+)$", 1),           # "1 Introduction"
        (r"^(\d+\.\d+)\s+([A-Z][A-Za-z\s]+)$", 2),      # "3.1 Attention"
        (r"^(\d+\.\d+\.\d+)\s+([A-Za-z\s]+)$", 3),      # "3.1.1 Details"
        (r"^(Abstract)$", 1),
        (r"^(Introduction)$", 1),
        (r"^(Related Work)$", 1),
        (r"^(Conclusion)s?$", 1),
        (r"^(References)$", 1),
        (r"^(Appendix)\s*([A-Z])?", 1),
        (r"^([A-Z])\s+([A-Z][A-Za-z\s]+)$", 1),         # "A Supplementary"
    ]
    
    def detect_sections(self, text: str, pages: List[PageContent]) -> List[Section]:
        # Implementation: scan text line by line, match patterns
        ...
```

✅ **Validation**: Detects at least 5 sections per paper  
🧪 **Test**: `pytest tests/test_preprocessing.py::test_section_detection -v`

---

### Step 12: Implement Section Detection Logic
**Action**: Full implementation of section boundary detection

```python
def detect_sections(self, text: str, pages: List[PageContent]) -> List[Section]:
    sections = []
    lines = text.split("\n")
    current_pos = 0
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        for pattern, level in self.SECTION_PATTERNS:
            match = re.match(pattern, line_stripped)
            if match:
                sections.append({
                    "title": line_stripped,
                    "level": level,
                    "start_char": current_pos,
                    "line_index": i
                })
                break
        
        current_pos += len(line) + 1  # +1 for newline
    
    # Set end positions
    for i, section in enumerate(sections):
        if i + 1 < len(sections):
            section["end_char"] = sections[i + 1]["start_char"]
        else:
            section["end_char"] = len(text)
    
    # Convert to Section objects with content
    return [
        Section(
            title=s["title"],
            level=s["level"],
            start_char=s["start_char"],
            end_char=s["end_char"],
            content=text[s["start_char"]:s["end_char"]],
            page_start=self._char_to_page(s["start_char"], pages),
            page_end=self._char_to_page(s["end_char"], pages)
        )
        for s in sections
    ]
```

✅ **Validation**: Section boundaries are accurate  
🧪 **Test**: `pytest tests/test_preprocessing.py::test_section_boundaries -v`

---

### Step 13: Create Tokenizer Utility
**Action**: Token counting using tiktoken

**File**: `src/preprocessing/chunker.py`

```python
import tiktoken
from typing import List

class TokenizerUtil:
    def __init__(self, model: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model)
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])
```

✅ **Validation**: Token counts match expected values  
🧪 **Test**: `pytest tests/test_preprocessing.py::test_tokenizer -v`

---

### Step 14: Implement Hybrid Chunker - Semantic Split
**Action**: Primary chunking by section/paragraph boundaries

**File**: `src/preprocessing/chunker.py` (add)

```python
from pydantic import BaseModel
from typing import List, Optional
import uuid

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    token_count: int
    page_numbers: List[int]
    section_title: Optional[str]
    section_hierarchy: List[str]
    chunk_index: int
    start_char: int
    end_char: int

class HybridChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.tokenizer = TokenizerUtil()
    
    def chunk_document(
        self, 
        doc: ExtractedDocument,
        sections: List[Section]
    ) -> List[Chunk]:
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_chunks = self._chunk_section(section, doc.metadata.doc_id)
            for chunk in section_chunks:
                chunk.chunk_index = chunk_index
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _chunk_section(self, section: Section, doc_id: str) -> List[Chunk]:
        token_count = self.tokenizer.count_tokens(section.content)
        
        if token_count <= self.config.max_chunk_size:
            # Section fits in one chunk
            return [self._create_chunk(section.content, section, doc_id)]
        
        # Split by paragraphs first
        paragraphs = section.content.split("\n\n")
        return self._chunk_paragraphs(paragraphs, section, doc_id)
```

✅ **Validation**: Sections split into valid chunks  
🧪 **Test**: `pytest tests/test_preprocessing.py::test_semantic_chunking -v`

---

### Step 15: Implement Fixed-Size Fallback Chunking
**Action**: Fallback for oversized paragraphs

```python
def _chunk_paragraphs(
    self, 
    paragraphs: List[str], 
    section: Section, 
    doc_id: str
) -> List[Chunk]:
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = self.tokenizer.count_tokens(para)
        
        if para_tokens > self.config.max_chunk_size:
            # Paragraph too big - use fixed-size split
            if current_chunk:
                chunks.append(self._create_chunk(current_chunk, section, doc_id))
                current_chunk = ""
                current_tokens = 0
            
            chunks.extend(self._fixed_size_split(para, section, doc_id))
            continue
        
        if current_tokens + para_tokens > self.config.max_chunk_size:
            chunks.append(self._create_chunk(current_chunk, section, doc_id))
            current_chunk = para
            current_tokens = para_tokens
        else:
            current_chunk += ("\n\n" + para if current_chunk else para)
            current_tokens += para_tokens
    
    if current_chunk:
        chunks.append(self._create_chunk(current_chunk, section, doc_id))
    
    return chunks

def _fixed_size_split(
    self, 
    text: str, 
    section: Section, 
    doc_id: str
) -> List[Chunk]:
    """Split text into fixed-size chunks with overlap."""
    tokens = self.tokenizer.tokenizer.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = min(start + self.config.fallback_chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = self.tokenizer.tokenizer.decode(chunk_tokens)
        
        chunks.append(self._create_chunk(chunk_text, section, doc_id))
        
        start = end - self.config.chunk_overlap
        if start >= len(tokens) - self.config.chunk_overlap:
            break
    
    return chunks
```

✅ **Validation**: No chunk exceeds max_chunk_size  
🧪 **Test**: `pytest tests/test_preprocessing.py::test_fixed_size_fallback -v`

---

### Step 16: Write Chunking Integration Tests
**Action**: Test full chunking pipeline

```python
def test_chunk_all_documents():
    parser = CompositeParser()
    detector = SectionDetector()
    chunker = HybridChunker(ChunkingConfig())
    
    papers_dir = Path("data/papers")
    all_chunks = []
    
    for pdf_file in papers_dir.glob("*.pdf"):
        doc = parser.extract(pdf_file)
        sections = detector.detect_sections(doc.full_text, doc.pages)
        chunks = chunker.chunk_document(doc, sections)
        
        all_chunks.extend(chunks)
        
        # Verify constraints
        for chunk in chunks:
            assert chunk.token_count <= 512
            assert chunk.token_count >= 50 or chunk == chunks[-1]
            assert chunk.doc_id == pdf_file.stem
    
    assert len(all_chunks) > 50  # Should have many chunks
    print(f"Total chunks created: {len(all_chunks)}")
```

✅ **Validation**: All papers chunk successfully  
🧪 **Test**: `pytest tests/test_preprocessing.py::test_chunk_all_documents -v`

---

## Phase 4: Embedding System (Steps 17-22)

### Step 17: Create Abstract Embedding Interface
**Action**: Base class for embedding providers

**File**: `src/embeddings/base.py`

```python
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimensions(self) -> int:
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, return shape (n, dimensions)."""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query, return shape (dimensions,)."""
        pass
```

✅ **Validation**: Interface defined correctly  
🧪 **Test**: `pytest tests/test_embeddings.py::test_interface -v`

---

### Step 18: Implement OpenAI Embeddings
**Action**: OpenAI text-embedding-3-small integration

**File**: `src/embeddings/openai_embeddings.py`

```python
from openai import OpenAI
import numpy as np
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import EmbeddingProvider
from src.config import settings

class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model_openai
        self._dimensions = 1536
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self._model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
```

✅ **Validation**: Successfully embeds test text  
🧪 **Test**: `pytest tests/test_embeddings.py::test_openai_embeddings -v`

---

### Step 19: Implement Open-Source Embeddings
**Action**: Sentence-transformers BGE integration

**File**: `src/embeddings/opensource_embeddings.py`

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import torch

from .base import EmbeddingProvider
from src.config import settings

class OpenSourceEmbeddings(EmbeddingProvider):
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            settings.embedding_model_opensource,
            device=self.device
        )
        self._dimensions = self.model.get_sentence_embedding_dimension()
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return settings.embedding_model_opensource
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
```

✅ **Validation**: BGE model loads and embeds  
🧪 **Test**: `pytest tests/test_embeddings.py::test_opensource_embeddings -v`

---

### Step 20: Create Embedding Factory
**Action**: Factory pattern for embedding provider selection

**File**: `src/embeddings/__init__.py`

```python
from typing import Literal
from .base import EmbeddingProvider
from .openai_embeddings import OpenAIEmbeddings
from .opensource_embeddings import OpenSourceEmbeddings

def get_embedding_provider(
    provider: Literal["openai", "opensource"]
) -> EmbeddingProvider:
    if provider == "openai":
        return OpenAIEmbeddings()
    elif provider == "opensource":
        return OpenSourceEmbeddings()
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

✅ **Validation**: Factory returns correct instances  
🧪 **Test**: `pytest tests/test_embeddings.py::test_factory -v`

---

### Step 21: Write Embedding Comparison Tests
**Action**: Compare OpenAI vs open-source embeddings

```python
def test_embedding_similarity_comparison():
    openai_emb = OpenAIEmbeddings()
    oss_emb = OpenSourceEmbeddings()
    
    texts = [
        "The Transformer architecture uses self-attention",
        "Self-attention allows the model to attend to all positions",
        "Recurrent neural networks process sequences sequentially"
    ]
    
    # Embed with both
    openai_vectors = openai_emb.embed_texts(texts)
    oss_vectors = oss_emb.embed_texts(texts)
    
    # Check dimensions
    assert openai_vectors.shape == (3, 1536)
    assert oss_vectors.shape == (3, 768)
    
    # Check that similar texts have higher similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    openai_sim = cosine_similarity(openai_vectors)
    oss_sim = cosine_similarity(oss_vectors)
    
    # Texts 0 and 1 should be more similar than 0 and 2
    assert openai_sim[0, 1] > openai_sim[0, 2]
    assert oss_sim[0, 1] > oss_sim[0, 2]
```

✅ **Validation**: Both embeddings show semantic similarity  
🧪 **Test**: `pytest tests/test_embeddings.py::test_embedding_similarity_comparison -v`

---

### Step 22: Embedding Performance Benchmarks
**Action**: Measure latency and throughput

```python
def test_embedding_performance():
    import time
    
    openai_emb = OpenAIEmbeddings()
    oss_emb = OpenSourceEmbeddings()
    
    # Test batch of 50 chunks
    test_texts = ["Sample text for embedding benchmark."] * 50
    
    # OpenAI timing
    start = time.time()
    openai_emb.embed_texts(test_texts)
    openai_time = time.time() - start
    
    # Open-source timing  
    start = time.time()
    oss_emb.embed_texts(test_texts)
    oss_time = time.time() - start
    
    print(f"OpenAI: {openai_time:.2f}s for 50 texts")
    print(f"Open-source: {oss_time:.2f}s for 50 texts")
    
    # Open-source should be faster for local inference
    # (unless GPU is available for OpenAI-equivalent speed)
```

✅ **Validation**: Performance metrics captured  
🧪 **Test**: `pytest tests/test_embeddings.py::test_embedding_performance -v -s`

---

## Phase 5: Vector Store & Indexing (Steps 23-27)

### Step 23: Create ChromaDB Wrapper
**Action**: Abstraction over ChromaDB operations

**File**: `src/retrieval/vector_store.py`

```python
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import json

from src.config import settings
from src.embeddings import get_embedding_provider

class VectorStore:
    def __init__(self, collection_name: str, embedding_provider: str = "openai"):
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embedding_provider = get_embedding_provider(embedding_provider)
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        texts = [c.content for c in chunks]
        embeddings = self.embedding_provider.embed_texts(texts)
        
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[self._chunk_to_metadata(c) for c in chunks]
        )
    
    def _chunk_to_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        return {
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "page_numbers": json.dumps(chunk.page_numbers),
            "section_title": chunk.section_title or "",
            "section_hierarchy": json.dumps(chunk.section_hierarchy),
            "token_count": chunk.token_count
        }
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_embedding = self.embedding_provider.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        return self._format_results(results)
```

✅ **Validation**: Add and search operations work  
🧪 **Test**: `pytest tests/test_retrieval.py::test_vector_store_basic -v`

---

### Step 24: Implement Dual Collection Indexing
**Action**: Index documents in both OpenAI and open-source collections

```python
class DualVectorStore:
    def __init__(self):
        self.openai_store = VectorStore("openai_embeddings", "openai")
        self.opensource_store = VectorStore("opensource_embeddings", "opensource")
    
    def index_chunks(self, chunks: List[Chunk]) -> Dict[str, int]:
        """Index chunks in both collections."""
        self.openai_store.add_chunks(chunks)
        self.opensource_store.add_chunks(chunks)
        
        return {
            "openai_count": self.openai_store.collection.count(),
            "opensource_count": self.opensource_store.collection.count()
        }
    
    def search(
        self, 
        query: str, 
        provider: str = "openai",
        top_k: int = 10
    ) -> List[Dict]:
        store = self.openai_store if provider == "openai" else self.opensource_store
        return store.search(query, top_k)
```

✅ **Validation**: Both collections populated  
🧪 **Test**: `pytest tests/test_retrieval.py::test_dual_indexing -v`

---

### Step 25: Create Document Indexing Script
**Action**: CLI script to index all PDFs

**File**: `scripts/index_documents.py`

```python
#!/usr/bin/env python
"""Index all research papers into ChromaDB."""

from pathlib import Path
from loguru import logger
from tqdm import tqdm

from src.preprocessing.pdf_parser import CompositeParser
from src.preprocessing.section_detector import SectionDetector
from src.preprocessing.chunker import HybridChunker, ChunkingConfig
from src.retrieval.vector_store import DualVectorStore
from src.config import settings

def main():
    parser = CompositeParser()
    detector = SectionDetector()
    chunker = HybridChunker(ChunkingConfig())
    vector_store = DualVectorStore()
    
    papers_dir = settings.papers_dir
    all_chunks = []
    
    for pdf_file in tqdm(list(papers_dir.glob("*.pdf")), desc="Processing PDFs"):
        logger.info(f"Processing: {pdf_file.name}")
        
        doc = parser.extract(pdf_file)
        sections = detector.detect_sections(doc.full_text, doc.pages)
        chunks = chunker.chunk_document(doc, sections)
        
        all_chunks.extend(chunks)
        logger.info(f"  Created {len(chunks)} chunks")
    
    logger.info(f"Indexing {len(all_chunks)} total chunks...")
    stats = vector_store.index_chunks(all_chunks)
    
    logger.info(f"Indexing complete: {stats}")

if __name__ == "__main__":
    main()
```

✅ **Validation**: Script runs without error  
🧪 **Test**: `python scripts/index_documents.py`

---

### Step 26: Verify Index Contents
**Action**: Validate indexed data integrity

```python
def test_index_integrity():
    store = DualVectorStore()
    
    # Check counts match
    openai_count = store.openai_store.collection.count()
    oss_count = store.opensource_store.collection.count()
    
    assert openai_count == oss_count
    assert openai_count > 50  # Should have many chunks
    
    # Verify all 3 documents indexed
    results = store.openai_store.collection.get()
    doc_ids = set(m["doc_id"] for m in results["metadatas"])
    
    assert "1706_03762v7" in doc_ids  # Attention paper
    assert "2005_11401v4" in doc_ids  # RAG paper
    assert "2005_14165v4" in doc_ids  # GPT-3 paper
```

✅ **Validation**: All documents indexed with correct metadata  
🧪 **Test**: `pytest tests/test_retrieval.py::test_index_integrity -v`

---

### Step 27: Write Vector Store Unit Tests
**Action**: Comprehensive retrieval tests

```python
def test_semantic_search_relevance():
    store = DualVectorStore()
    
    # Query about attention mechanism
    query = "How does multi-head attention work in Transformers?"
    results = store.search(query, provider="openai", top_k=5)
    
    # Top results should be from Attention paper
    assert any("1706_03762v7" in r["metadata"]["doc_id"] for r in results[:3])
    
    # Query about few-shot learning
    query = "How does GPT-3 perform few-shot learning?"
    results = store.search(query, provider="opensource", top_k=5)
    
    # Top results should be from GPT-3 paper
    assert any("2005_14165v4" in r["metadata"]["doc_id"] for r in results[:3])
```

✅ **Validation**: Retrieval returns relevant chunks  
🧪 **Test**: `pytest tests/test_retrieval.py -v`

---

## Phase 6: Hybrid Retrieval (Steps 28-32)

### Step 28: Implement BM25 Retriever
**Action**: Sparse retrieval using rank_bm25

**File**: `src/retrieval/bm25_retriever.py`

```python
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import re

class BM25Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}
        
        # Tokenize documents
        self.tokenized_docs = [self._tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk_id = self.chunks[idx].chunk_id
            score = scores[idx]
            results.append((chunk_id, score))
        
        return results
```

✅ **Validation**: BM25 returns ranked results  
🧪 **Test**: `pytest tests/test_retrieval.py::test_bm25_retriever -v`

---

### Step 29: Implement Reciprocal Rank Fusion
**Action**: Combine dense and sparse rankings

**File**: `src/retrieval/hybrid_retriever.py`

```python
from collections import defaultdict
from typing import List, Dict, Tuple

def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Fuse multiple rankings using RRF.
    
    Args:
        rankings: List of rankings, each is [(doc_id, score), ...]
        k: RRF constant (default 60)
    
    Returns:
        Fused ranking as [(doc_id, rrf_score), ...]
    """
    rrf_scores = defaultdict(float)
    
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            rrf_scores[doc_id] += 1 / (k + rank + 1)
    
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results
```

✅ **Validation**: RRF produces valid fused ranking  
🧪 **Test**: `pytest tests/test_retrieval.py::test_rrf_fusion -v`

---

### Step 30: Implement Cross-Encoder Reranker
**Action**: Fine-grained reranking with cross-encoder

**File**: `src/retrieval/reranker.py`

```python
from sentence_transformers import CrossEncoder
from typing import List, Tuple
import torch

from src.config import settings

class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str, float]],  # (chunk_id, content, score)
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []
        
        # Create query-document pairs
        pairs = [(query, content) for _, content, _ in candidates]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Combine with chunk_ids and sort
        reranked = [
            (candidates[i][0], float(scores[i]))
            for i in range(len(candidates))
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
```

✅ **Validation**: Reranker improves result quality  
🧪 **Test**: `pytest tests/test_retrieval.py::test_cross_encoder_reranker -v`

---

### Step 31: Create Hybrid Retriever Pipeline
**Action**: Orchestrate dense + sparse + reranking

**File**: `src/retrieval/hybrid_retriever.py` (add)

```python
class HybridRetriever:
    def __init__(
        self,
        vector_store: DualVectorStore,
        chunks: List[Chunk],
        embedding_provider: str = "openai"
    ):
        self.vector_store = vector_store
        self.bm25 = BM25Retriever(chunks)
        self.reranker = CrossEncoderReranker()
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.embedding_provider = embedding_provider
    
    def retrieve(
        self,
        query: str,
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        rerank_top_k: int = 10,
        final_top_k: int = 5
    ) -> List[Dict]:
        # 1. Dense retrieval
        dense_results = self.vector_store.search(
            query, 
            provider=self.embedding_provider,
            top_k=top_k_dense
        )
        dense_ranking = [(r["chunk_id"], r["score"]) for r in dense_results]
        
        # 2. Sparse retrieval (BM25)
        sparse_ranking = self.bm25.search(query, top_k=top_k_sparse)
        
        # 3. Reciprocal Rank Fusion
        fused_ranking = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
        
        # 4. Get top candidates for reranking
        top_candidates = []
        for chunk_id, rrf_score in fused_ranking[:rerank_top_k]:
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                top_candidates.append((chunk_id, chunk.content, rrf_score))
        
        # 5. Cross-encoder reranking
        reranked = self.reranker.rerank(query, top_candidates, top_k=final_top_k)
        
        # 6. Build final results
        results = []
        for chunk_id, score in reranked:
            chunk = self.chunk_map[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "content": chunk.content,
                "score": score,
                "metadata": {
                    "doc_id": chunk.doc_id,
                    "section_title": chunk.section_title,
                    "page_numbers": chunk.page_numbers
                }
            })
        
        return results
```

✅ **Validation**: Full hybrid pipeline works  
🧪 **Test**: `pytest tests/test_retrieval.py::test_hybrid_retriever -v`

---

### Step 32: Retrieval Quality Evaluation
**Action**: Measure retrieval performance on sample questions

```python
def test_retrieval_quality():
    """Evaluate retrieval against ground truth."""
    retriever = HybridRetriever(...)  # Initialize with indexed data
    
    EVAL_QUESTIONS = [
        {
            "question": "What are the two sub-layers in each encoder layer?",
            "expected_doc": "1706_03762v7",
            "key_terms": ["self-attention", "feed-forward"]
        },
        # ... more questions
    ]
    
    precision_at_5 = []
    mrr_scores = []
    
    for eval_item in EVAL_QUESTIONS:
        results = retriever.retrieve(eval_item["question"], final_top_k=5)
        
        # Check if expected doc in top 5
        doc_ids = [r["metadata"]["doc_id"] for r in results]
        hit = eval_item["expected_doc"] in doc_ids
        precision_at_5.append(1 if hit else 0)
        
        # MRR
        try:
            rank = doc_ids.index(eval_item["expected_doc"]) + 1
            mrr_scores.append(1 / rank)
        except ValueError:
            mrr_scores.append(0)
    
    avg_precision = sum(precision_at_5) / len(precision_at_5)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)
    
    print(f"Precision@5: {avg_precision:.2%}")
    print(f"MRR: {avg_mrr:.3f}")
    
    assert avg_precision >= 0.8  # Target: 80%
```

✅ **Validation**: Precision@5 >= 80%  
🧪 **Test**: `pytest tests/test_retrieval.py::test_retrieval_quality -v`

---

## Phase 7: Answer Generation (Steps 33-37)

### Step 33: Create Context Builder
**Action**: Assemble retrieved chunks into LLM context

**File**: `src/generation/context_builder.py`

```python
from typing import List, Dict
from src.config import settings

class ContextBuilder:
    def __init__(self, max_tokens: int = None):
        self.max_tokens = max_tokens or settings.max_context_tokens
        self.tokenizer = TokenizerUtil()
    
    def build_context(self, results: List[Dict]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(results):
            source_header = self._format_source_header(i + 1, result)
            chunk_text = result["content"]
            
            part = f"{source_header}\n{chunk_text}\n"
            part_tokens = self.tokenizer.count_tokens(part)
            
            if current_tokens + part_tokens > self.max_tokens:
                break
            
            context_parts.append(part)
            current_tokens += part_tokens
        
        return "\n".join(context_parts)
    
    def _format_source_header(self, index: int, result: Dict) -> str:
        meta = result["metadata"]
        pages = meta.get("page_numbers", [])
        section = meta.get("section_title", "")
        
        parts = [f"[Source {index}"]
        if meta.get("doc_id"):
            parts.append(f": {meta['doc_id']}")
        if section:
            parts.append(f", Section: {section}")
        if pages:
            parts.append(f", Pages: {pages}")
        parts.append("]")
        
        return "".join(parts)
```

✅ **Validation**: Context fits within token limit  
🧪 **Test**: `pytest tests/test_generation.py::test_context_builder -v`

---

### Step 34: Create LLM Client
**Action**: OpenAI gpt-4o-mini wrapper for generation

**File**: `src/generation/llm_client.py`

```python
from openai import OpenAI
from typing import Optional
from src.config import settings

class LLMClient:
    SYSTEM_PROMPT = """You are a research assistant specialized in AI/ML papers. 
Answer questions based ONLY on the provided context from research papers.

CRITICAL INSTRUCTIONS:
1. If the context doesn't contain sufficient information to answer, say so explicitly.
2. Always cite your sources using the format [Source N] where N matches the source number in the context.
3. Use direct quotes sparingly, paraphrasing when possible.
4. Structure complex answers with clear organization.
5. Distinguish between what the paper explicitly states vs. your interpretation."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
    
    def generate(
        self,
        question: str,
        context: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        user_prompt = f"""Context from research papers:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above, with citations."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.max_output_tokens
        )
        
        return response.choices[0].message.content
```

✅ **Validation**: LLM generates coherent answers  
🧪 **Test**: `pytest tests/test_generation.py::test_llm_client -v`

---

### Step 35: Implement Citation Formatter
**Action**: Parse and enrich citations in generated answers

**File**: `src/generation/citation_formatter.py`

```python
import re
from typing import List, Dict
from pydantic import BaseModel

class SourceAttribution(BaseModel):
    source_index: int
    chunk_id: str
    paper_title: str
    section: Optional[str]
    page_numbers: List[int]
    relevance_score: float
    excerpt: str

class CitationFormatter:
    # Map doc_id to paper titles
    PAPER_TITLES = {
        "1706_03762v7": "Attention Is All You Need",
        "2005_11401v4": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "2005_14165v4": "Language Models are Few-Shot Learners"
    }
    
    def format_inline_citations(
        self,
        answer: str,
        sources: List[Dict]
    ) -> str:
        """Replace [Source N] with detailed inline citations."""
        def replace_citation(match):
            source_num = int(match.group(1)) - 1
            if source_num < len(sources):
                source = sources[source_num]
                doc_id = source["metadata"]["doc_id"]
                title = self.PAPER_TITLES.get(doc_id, doc_id)
                section = source["metadata"].get("section_title", "")
                pages = source["metadata"].get("page_numbers", [])
                
                citation_parts = [title]
                if section:
                    citation_parts.append(f"Section: {section}")
                if pages:
                    citation_parts.append(f"Page(s): {', '.join(map(str, pages))}")
                
                return f"[{', '.join(citation_parts)}]"
            return match.group(0)
        
        return re.sub(r'\[Source (\d+)\]', replace_citation, answer)
    
    def build_source_attributions(
        self,
        sources: List[Dict]
    ) -> List[SourceAttribution]:
        """Build structured source attribution list."""
        attributions = []
        for i, source in enumerate(sources):
            doc_id = source["metadata"]["doc_id"]
            attributions.append(SourceAttribution(
                source_index=i + 1,
                chunk_id=source["chunk_id"],
                paper_title=self.PAPER_TITLES.get(doc_id, doc_id),
                section=source["metadata"].get("section_title"),
                page_numbers=source["metadata"].get("page_numbers", []),
                relevance_score=source["score"],
                excerpt=source["content"][:200] + "..."
            ))
        return attributions
```

✅ **Validation**: Citations properly formatted  
🧪 **Test**: `pytest tests/test_generation.py::test_citation_formatter -v`

---

### Step 36: Create Answer Response Schema
**Action**: Pydantic model for complete answer response

**File**: `src/generation/schemas.py`

```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class SourceAttribution(BaseModel):
    source_index: int
    chunk_id: str
    paper_title: str
    section: Optional[str]
    page_numbers: List[int]
    relevance_score: float
    excerpt: str

class AnswerResponse(BaseModel):
    question: str
    answer: str                          # With inline citations
    sources: List[SourceAttribution]
    retrieval_method: str                # "openai" | "opensource" | "hybrid"
    confidence: float                    # Based on source relevance
    processing_time_ms: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is multi-head attention?",
                "answer": "Multi-head attention allows the model to...",
                "sources": [],
                "retrieval_method": "hybrid",
                "confidence": 0.92,
                "processing_time_ms": 1234.5
            }
        }
```

✅ **Validation**: Schema validates correctly  
🧪 **Test**: `pytest tests/test_generation.py::test_answer_schema -v`

---

### Step 37: Write Generation Unit Tests
**Action**: Test answer generation pipeline

```python
def test_full_generation_pipeline():
    context_builder = ContextBuilder()
    llm_client = LLMClient()
    citation_formatter = CitationFormatter()
    
    # Mock retrieved results
    mock_results = [
        {
            "chunk_id": "chunk_1",
            "content": "The Transformer uses self-attention to...",
            "score": 0.95,
            "metadata": {
                "doc_id": "1706_03762v7",
                "section_title": "3.2 Attention",
                "page_numbers": [3, 4]
            }
        }
    ]
    
    # Build context
    context = context_builder.build_context(mock_results)
    assert "[Source 1" in context
    
    # Generate answer
    question = "How does self-attention work?"
    answer = llm_client.generate(question, context)
    assert len(answer) > 50
    
    # Format citations
    formatted = citation_formatter.format_inline_citations(answer, mock_results)
    attributions = citation_formatter.build_source_attributions(mock_results)
    
    assert len(attributions) == 1
    assert attributions[0].paper_title == "Attention Is All You Need"
```

✅ **Validation**: End-to-end generation works  
🧪 **Test**: `pytest tests/test_generation.py -v`

---

## Phase 8: RAG Pipeline Integration (Steps 38-39)

### Step 38: Create RAG Pipeline Orchestrator
**Action**: End-to-end pipeline class

**File**: `src/pipeline/rag_pipeline.py`

```python
import time
from typing import Literal

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.context_builder import ContextBuilder
from src.generation.llm_client import LLMClient
from src.generation.citation_formatter import CitationFormatter
from src.generation.schemas import AnswerResponse, SourceAttribution

class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        embedding_provider: str = "hybrid"
    ):
        self.retriever = retriever
        self.context_builder = ContextBuilder()
        self.llm_client = LLMClient()
        self.citation_formatter = CitationFormatter()
        self.default_provider = embedding_provider
    
    def answer(
        self,
        question: str,
        retrieval_method: Literal["openai", "opensource", "hybrid"] = None,
        top_k: int = 5
    ) -> AnswerResponse:
        start_time = time.time()
        
        method = retrieval_method or self.default_provider
        
        # 1. Retrieve relevant chunks
        results = self.retriever.retrieve(
            query=question,
            final_top_k=top_k
        )
        
        # 2. Build context
        context = self.context_builder.build_context(results)
        
        # 3. Generate answer
        raw_answer = self.llm_client.generate(question, context)
        
        # 4. Format citations
        formatted_answer = self.citation_formatter.format_inline_citations(
            raw_answer, results
        )
        
        # 5. Build source attributions
        sources = self.citation_formatter.build_source_attributions(results)
        
        # 6. Calculate confidence
        avg_score = sum(r["score"] for r in results) / len(results) if results else 0
        confidence = min(avg_score, 1.0)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnswerResponse(
            question=question,
            answer=formatted_answer,
            sources=sources,
            retrieval_method=method,
            confidence=confidence,
            processing_time_ms=processing_time
        )
```

✅ **Validation**: Pipeline produces valid AnswerResponse  
🧪 **Test**: `pytest tests/test_pipeline.py -v`

---

### Step 39: End-to-End Pipeline Tests with Sample Questions
**Action**: Test all 5 sample questions

```python
SAMPLE_QUESTIONS = [
    "What are the main components of a RAG model, and how do they interact?",
    "What are the two sub-layers in each encoder layer of the Transformer model?",
    "Explain how positional encoding is implemented in Transformers and why it is necessary.",
    "Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?",
    "What is few-shot learning, and how does GPT-3 implement it during inference?"
]

def test_sample_questions():
    pipeline = RAGPipeline(...)  # Initialize with indexed data
    
    for question in SAMPLE_QUESTIONS:
        response = pipeline.answer(question)
        
        # Verify response structure
        assert response.question == question
        assert len(response.answer) > 100
        assert len(response.sources) > 0
        assert response.processing_time_ms > 0
        
        # Verify citations present
        assert "[" in response.answer and "]" in response.answer
        
        print(f"\nQ: {question}")
        print(f"A: {response.answer[:200]}...")
        print(f"Sources: {len(response.sources)}, Confidence: {response.confidence:.2f}")
```

✅ **Validation**: All 5 questions answered with citations  
🧪 **Test**: `pytest tests/test_pipeline.py::test_sample_questions -v -s`

---

## Phase 9: API Layer (Steps 40-41)

### Step 40: Create FastAPI Application
**Action**: API endpoints for RAG system

**File**: `src/api/main.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional

from src.pipeline.rag_pipeline import RAGPipeline
from src.generation.schemas import AnswerResponse

app = FastAPI(
    title="RAG-QA API",
    description="Question Answering on AI Research Papers",
    version="0.1.0"
)

# Initialize pipeline (loaded at startup)
pipeline: RAGPipeline = None

@app.on_event("startup")
async def startup():
    global pipeline
    # Load indexed data and initialize pipeline
    pipeline = RAGPipeline(...)

class QueryRequest(BaseModel):
    question: str
    retrieval_method: Literal["openai", "opensource", "hybrid"] = "hybrid"
    top_k: int = 5

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/query", response_model=AnswerResponse)
async def query(request: QueryRequest):
    try:
        response = pipeline.answer(
            question=request.question,
            retrieval_method=request.retrieval_method,
            top_k=request.top_k
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
async def retrieve_only(request: QueryRequest):
    """Retrieval without generation (for debugging)."""
    results = pipeline.retriever.retrieve(
        query=request.question,
        final_top_k=request.top_k
    )
    return {"results": results}

@app.get("/documents")
async def list_documents():
    """List indexed documents."""
    # Return document metadata
    ...

@app.get("/stats")
async def system_stats():
    """Return system statistics."""
    ...
```

✅ **Validation**: API starts and responds  
🧪 **Test**: `uvicorn src.api.main:app --reload` then `curl localhost:8000/health`

---

### Step 41: Write API Integration Tests
**Action**: Test API endpoints

**File**: `tests/test_api.py`

```python
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query_endpoint():
    response = client.post("/query", json={
        "question": "What is multi-head attention?",
        "retrieval_method": "hybrid",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0

def test_retrieve_endpoint():
    response = client.post("/retrieve", json={
        "question": "How does positional encoding work?",
        "top_k": 3
    })
    assert response.status_code == 200
    assert "results" in response.json()
```

✅ **Validation**: All API tests pass  
🧪 **Test**: `pytest tests/test_api.py -v`

---

## Phase 10: Final Deliverables (Step 42)

### Step 42: Create Jupyter Notebook Deliverable
**Action**: Comprehensive Colab notebook demonstrating the system

**File**: `notebooks/RAG_QA_System_Demo.ipynb`

**Notebook Sections**:
1. **Setup & Installation** - Install dependencies, mount Drive
2. **Configuration** - API keys, settings
3. **Document Processing Demo** - Show PDF parsing, chunking
4. **Embedding Comparison** - OpenAI vs open-source embeddings
5. **Retrieval Demo** - BM25, dense, hybrid retrieval
6. **Full RAG Pipeline** - End-to-end question answering
7. **Sample Questions** - Run all 5 sample questions
8. **Evaluation Metrics** - Show precision, MRR results
9. **API Demo** - Optional: start FastAPI server

✅ **Validation**: Notebook runs end-to-end in Colab  
🧪 **Test**: Run all cells in Google Colab

---

## Summary Checklist

| Phase | Steps | Key Validation |
|-------|-------|----------------|
| 1. Project Setup | 1-5 | Project structure, dependencies installed |
| 2. PDF Parsing | 6-10 | All 3 PDFs parse successfully |
| 3. Chunking | 11-16 | Hybrid chunking produces valid chunks |
| 4. Embeddings | 17-22 | Both embedding providers work |
| 5. Vector Store | 23-27 | ChromaDB indexed with all chunks |
| 6. Hybrid Retrieval | 28-32 | Precision@5 >= 80% |
| 7. Generation | 33-37 | Answers generated with citations |
| 8. Pipeline | 38-39 | All 5 sample questions answered |
| 9. API | 40-41 | All endpoints functional |
| 10. Deliverables | 42 | Notebook runs in Colab |

**Total Test Commands**:
```bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Run specific phase
pytest tests/test_preprocessing.py -v
pytest tests/test_embeddings.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_generation.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_api.py -v
```
