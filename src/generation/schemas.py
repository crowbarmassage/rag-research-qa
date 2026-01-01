"""Pydantic schemas for generation responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SourceAttribution(BaseModel):
    """Source attribution for an answer."""
    source_index: int
    chunk_id: str
    paper_title: str
    section: Optional[str] = None
    page_numbers: List[int] = Field(default_factory=list)
    relevance_score: float
    excerpt: str


class AnswerResponse(BaseModel):
    """Complete answer response with metadata."""
    question: str
    answer: str  # With inline citations
    sources: List[SourceAttribution]
    retrieval_method: str  # "openai" | "opensource" | "hybrid"
    confidence: float  # Based on source relevance scores
    processing_time_ms: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is multi-head attention?",
                "answer": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions [Source 1].",
                "sources": [
                    {
                        "source_index": 1,
                        "chunk_id": "abc123",
                        "paper_title": "Attention Is All You Need",
                        "section": "3.2 Multi-Head Attention",
                        "page_numbers": [3, 4],
                        "relevance_score": 0.95,
                        "excerpt": "Multi-head attention allows the model to jointly attend..."
                    }
                ],
                "retrieval_method": "hybrid",
                "confidence": 0.92,
                "processing_time_ms": 1234.5
            }
        }
    }


class RetrievalResult(BaseModel):
    """Result from retrieval without generation."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class ComparisonResult(BaseModel):
    """Result from embedding comparison."""
    query: str
    openai_results: List[RetrievalResult]
    opensource_results: List[RetrievalResult]
    agreement_score: float  # Overlap between top-k results


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    doc_id: str
    title: str
    total_chunks: int
    source_file: str
