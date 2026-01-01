"""FastAPI application for RAG-QA system."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import settings
from src.pipeline import RAGPipeline
from .schemas import (
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    CompareRequest,
    CompareResponse,
    StatsResponse,
    HealthResponse,
    SourceInfo,
    RetrievalItem,
    DocumentStats,
)

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    global pipeline

    logger.info("Starting RAG-QA API...")

    # Check if papers directory exists and has PDFs
    papers_dir = Path(settings.papers_dir)
    if papers_dir.exists() and list(papers_dir.glob("*.pdf")):
        try:
            pipeline = RAGPipeline.from_documents(
                papers_dir=papers_dir,
                persist_dir=settings.chroma_dir,
                embedding_provider="openai",
                use_reranker=True
            )
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            pipeline = None
    else:
        logger.warning(
            f"No PDFs found in {papers_dir}. "
            "Add papers and restart or use /index endpoint."
        )
        pipeline = None

    yield

    # Cleanup
    logger.info("Shutting down RAG-QA API...")


app = FastAPI(
    title="RAG-QA API",
    description="Question Answering on AI Research Papers using RAG",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_pipeline():
    """Check if pipeline is initialized."""
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Add PDFs to data/papers/ and restart."
        )
    return pipeline


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if pipeline is None:
        return HealthResponse(
            status="degraded",
            indexed_chunks=0,
            message="Pipeline not initialized"
        )

    stats = pipeline.get_document_stats()
    return HealthResponse(
        status="healthy",
        indexed_chunks=stats["total_chunks"],
        message=f"Ready with {len(stats['documents'])} documents"
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Full RAG pipeline: retrieve relevant chunks and generate an answer.
    """
    p = check_pipeline()

    try:
        response = p.answer(
            question=request.question,
            retrieval_method=request.retrieval_method,
            top_k=request.top_k
        )

        return QueryResponse(
            question=response.question,
            answer=response.answer,
            sources=[
                SourceInfo(
                    source_index=s.source_index,
                    chunk_id=s.chunk_id,
                    paper_title=s.paper_title,
                    section=s.section,
                    page_numbers=s.page_numbers,
                    relevance_score=s.relevance_score,
                    excerpt=s.excerpt
                )
                for s in response.sources
            ],
            retrieval_method=response.retrieval_method,
            confidence=response.confidence,
            processing_time_ms=response.processing_time_ms
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """
    Retrieval only (without generation) - useful for debugging.
    """
    p = check_pipeline()

    try:
        results = p.retrieve_only(
            question=request.question,
            top_k=request.top_k,
            method=request.method
        )

        return RetrieveResponse(
            query=request.question,
            results=[
                RetrievalItem(
                    chunk_id=r["chunk_id"],
                    content=r["content"],
                    score=r["score"],
                    metadata=r["metadata"]
                )
                for r in results
            ]
        )

    except Exception as e:
        logger.error(f"Retrieve failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """
    Compare OpenAI vs open-source embeddings for retrieval.
    """
    p = check_pipeline()

    try:
        comparison = p.compare_embeddings(
            question=request.question,
            top_k=request.top_k
        )

        return CompareResponse(
            query=comparison["query"],
            openai_results=[
                RetrievalItem(
                    chunk_id=r["chunk_id"],
                    content=r["content"],
                    score=r["score"],
                    metadata=r["metadata"]
                )
                for r in comparison["openai_results"]
            ],
            opensource_results=[
                RetrievalItem(
                    chunk_id=r["chunk_id"],
                    content=r["content"],
                    score=r["score"],
                    metadata=r["metadata"]
                )
                for r in comparison["opensource_results"]
            ],
            agreement_score=comparison["agreement_score"]
        )

    except Exception as e:
        logger.error(f"Compare failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=StatsResponse)
async def list_documents():
    """
    List indexed documents and their statistics.
    """
    p = check_pipeline()

    try:
        stats = p.get_document_stats()
        vs_stats = p.vector_store.get_stats() if p.vector_store else {}

        return StatsResponse(
            total_chunks=stats["total_chunks"],
            documents=[
                DocumentStats(
                    doc_id=d["doc_id"],
                    title=d["title"],
                    total_chunks=d["total_chunks"]
                )
                for d in stats["documents"]
            ],
            vector_store_stats=vs_stats
        )

    except Exception as e:
        logger.error(f"Documents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def system_stats():
    """
    Get system statistics (alias for /documents).
    """
    return await list_documents()
