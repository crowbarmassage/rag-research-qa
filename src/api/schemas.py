"""API request/response schemas."""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for the /query endpoint."""
    question: str = Field(..., description="The question to answer")
    retrieval_method: Literal["openai", "opensource", "hybrid"] = Field(
        default="hybrid",
        description="Retrieval method to use"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of sources to retrieve"
    )


class RetrieveRequest(BaseModel):
    """Request schema for the /retrieve endpoint."""
    question: str = Field(..., description="The query to search for")
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return"
    )
    method: Literal["hybrid", "dense", "sparse"] = Field(
        default="hybrid",
        description="Retrieval method"
    )


class CompareRequest(BaseModel):
    """Request schema for the /compare endpoint."""
    question: str = Field(..., description="The query to compare")
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results per provider"
    )


class SourceInfo(BaseModel):
    """Source information in responses."""
    source_index: int
    chunk_id: str
    paper_title: str
    section: Optional[str] = None
    page_numbers: List[int] = Field(default_factory=list)
    relevance_score: float
    excerpt: str


class QueryResponse(BaseModel):
    """Response schema for the /query endpoint."""
    question: str
    answer: str
    sources: List[SourceInfo]
    retrieval_method: str
    confidence: float
    processing_time_ms: float


class RetrievalItem(BaseModel):
    """A single retrieval result."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class RetrieveResponse(BaseModel):
    """Response schema for the /retrieve endpoint."""
    query: str
    results: List[RetrievalItem]


class CompareResponse(BaseModel):
    """Response schema for the /compare endpoint."""
    query: str
    openai_results: List[RetrievalItem]
    opensource_results: List[RetrievalItem]
    agreement_score: float


class DocumentStats(BaseModel):
    """Document statistics."""
    doc_id: str
    title: str
    total_chunks: int


class StatsResponse(BaseModel):
    """Response schema for the /stats endpoint."""
    total_chunks: int
    documents: List[DocumentStats]
    vector_store_stats: Dict[str, int]


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    status: str
    indexed_chunks: int
    message: str = ""
