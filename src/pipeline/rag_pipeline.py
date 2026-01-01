"""End-to-end RAG pipeline orchestrator."""

import time
from typing import Literal, List, Dict, Optional
from pathlib import Path
from loguru import logger

from src.config import settings
from src.preprocessing import (
    CompositeParser,
    SectionDetector,
    HybridChunker,
    ChunkingConfig,
    Chunk,
)
from src.retrieval import DualVectorStore, HybridRetriever
from src.generation import (
    ContextBuilder,
    LLMClient,
    CitationFormatter,
    AnswerResponse,
)


class RAGPipeline:
    """End-to-end RAG pipeline for question answering."""

    def __init__(
        self,
        chunks: List[Chunk] = None,
        vector_store: DualVectorStore = None,
        embedding_provider: Literal["openai", "opensource"] = "openai",
        use_reranker: bool = True
    ):
        """
        Initialize the RAG pipeline.

        Args:
            chunks: Pre-loaded chunks (for retriever initialization)
            vector_store: Pre-initialized vector store
            embedding_provider: Which embedding provider to use
            use_reranker: Whether to use cross-encoder reranking
        """
        self.chunks = chunks or []
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

        # Initialize retriever if we have chunks and vector store
        self.retriever = None
        if chunks and vector_store:
            self.retriever = HybridRetriever(
                vector_store=vector_store,
                chunks=chunks,
                embedding_provider=embedding_provider,
                use_reranker=use_reranker
            )

        # Initialize generation components
        self.context_builder = ContextBuilder()
        self.llm_client = LLMClient()
        self.citation_formatter = CitationFormatter()

        logger.info(
            f"Initialized RAGPipeline with {len(chunks)} chunks, "
            f"provider={embedding_provider}, reranker={use_reranker}"
        )

    @classmethod
    def from_documents(
        cls,
        papers_dir: Path = None,
        persist_dir: Path = None,
        embedding_provider: Literal["openai", "opensource"] = "openai",
        use_reranker: bool = True,
        force_reindex: bool = False
    ) -> "RAGPipeline":
        """
        Create a pipeline from PDF documents.

        Args:
            papers_dir: Directory containing PDF files
            persist_dir: Directory for vector store persistence
            embedding_provider: Which embedding provider to use
            use_reranker: Whether to use cross-encoder reranking
            force_reindex: Whether to force re-indexing

        Returns:
            Initialized RAGPipeline
        """
        papers_dir = papers_dir or settings.papers_dir
        persist_dir = persist_dir or settings.chroma_dir

        # Initialize vector store
        vector_store = DualVectorStore(persist_directory=persist_dir)

        # Check if we need to index
        stats = vector_store.get_stats()
        if stats["openai_count"] > 0 and not force_reindex:
            logger.info(f"Using existing index with {stats['openai_count']} chunks")
            # Load chunks from vector store
            chunks = cls._load_chunks_from_store(vector_store)
        else:
            # Process documents and index
            logger.info("Indexing documents...")
            chunks = cls._process_documents(papers_dir)
            vector_store.index_chunks(chunks)

        return cls(
            chunks=chunks,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            use_reranker=use_reranker
        )

    @staticmethod
    def _process_documents(papers_dir: Path) -> List[Chunk]:
        """Process PDF documents and return chunks."""
        parser = CompositeParser()
        section_detector = SectionDetector()
        chunker = HybridChunker()

        all_chunks = []
        papers_dir = Path(papers_dir)

        for pdf_file in papers_dir.glob("*.pdf"):
            logger.info(f"Processing: {pdf_file.name}")
            doc = parser.extract(pdf_file)
            sections = section_detector.detect_sections(doc.full_text, doc.pages)
            chunks = chunker.chunk_document(doc, sections)
            all_chunks.extend(chunks)
            logger.info(f"  Created {len(chunks)} chunks")

        return all_chunks

    @staticmethod
    def _load_chunks_from_store(vector_store: DualVectorStore) -> List[Chunk]:
        """Load chunks from existing vector store."""
        results = vector_store.openai_store.get_all_chunks()

        chunks = []
        for r in results:
            meta = r.get("metadata", {})
            chunks.append(Chunk(
                chunk_id=r["chunk_id"],
                doc_id=meta.get("doc_id", ""),
                content=r["content"],
                token_count=meta.get("token_count", 0),
                page_numbers=meta.get("page_numbers", []),
                section_title=meta.get("section_title"),
                section_hierarchy=meta.get("section_hierarchy", []),
                chunk_index=meta.get("chunk_index", 0)
            ))

        return chunks

    def answer(
        self,
        question: str,
        retrieval_method: Literal["openai", "opensource", "hybrid"] = "hybrid",
        top_k: int = None
    ) -> AnswerResponse:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: The question to answer
            retrieval_method: Which retrieval method to use
            top_k: Number of chunks to retrieve

        Returns:
            AnswerResponse with answer and sources
        """
        start_time = time.time()
        top_k = top_k or settings.final_top_k

        if not self.retriever:
            raise RuntimeError("Pipeline not initialized with retriever")

        # 1. Retrieve relevant chunks
        if retrieval_method == "hybrid":
            results = self.retriever.retrieve(
                query=question,
                final_top_k=top_k,
                method="hybrid"
            )
        elif retrieval_method in ["openai", "opensource"]:
            # Switch provider temporarily
            original_provider = self.retriever.embedding_provider
            self.retriever.embedding_provider = retrieval_method
            results = self.retriever.retrieve(
                query=question,
                final_top_k=top_k,
                method="dense"
            )
            self.retriever.embedding_provider = original_provider
        else:
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
        if results:
            avg_score = sum(r.get("score", 0) for r in results) / len(results)
            confidence = min(max(avg_score, 0.0), 1.0)
        else:
            confidence = 0.0

        processing_time = (time.time() - start_time) * 1000

        return AnswerResponse(
            question=question,
            answer=formatted_answer,
            sources=sources,
            retrieval_method=retrieval_method,
            confidence=confidence,
            processing_time_ms=processing_time
        )

    def retrieve_only(
        self,
        question: str,
        top_k: int = None,
        method: Literal["hybrid", "dense", "sparse"] = "hybrid"
    ) -> List[Dict]:
        """Retrieve without generation (for debugging)."""
        if not self.retriever:
            raise RuntimeError("Pipeline not initialized with retriever")

        return self.retriever.retrieve(
            query=question,
            final_top_k=top_k or settings.final_top_k,
            method=method
        )

    def compare_embeddings(
        self,
        question: str,
        top_k: int = 5
    ) -> Dict:
        """Compare OpenAI vs open-source embeddings."""
        if not self.vector_store:
            raise RuntimeError("Pipeline not initialized with vector store")

        openai_results = self.vector_store.search(question, "openai", top_k)
        opensource_results = self.vector_store.search(question, "opensource", top_k)

        # Calculate agreement
        openai_ids = {r["chunk_id"] for r in openai_results}
        opensource_ids = {r["chunk_id"] for r in opensource_results}
        overlap = len(openai_ids & opensource_ids)
        agreement = overlap / top_k if top_k > 0 else 0

        return {
            "query": question,
            "openai_results": openai_results,
            "opensource_results": opensource_results,
            "agreement_score": agreement
        }

    def get_document_stats(self) -> Dict:
        """Get statistics about indexed documents."""
        if not self.chunks:
            return {"total_chunks": 0, "documents": []}

        doc_stats = {}
        for chunk in self.chunks:
            doc_id = chunk.doc_id
            if doc_id not in doc_stats:
                doc_stats[doc_id] = {
                    "doc_id": doc_id,
                    "title": self.citation_formatter.get_paper_title(doc_id),
                    "total_chunks": 0
                }
            doc_stats[doc_id]["total_chunks"] += 1

        return {
            "total_chunks": len(self.chunks),
            "documents": list(doc_stats.values())
        }
