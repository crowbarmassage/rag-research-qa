"""Hybrid retrieval with dense + sparse + reranking."""

from collections import defaultdict
from typing import List, Dict, Tuple, Literal
from loguru import logger

from src.preprocessing import Chunk
from src.config import settings
from .vector_store import VectorStore, DualVectorStore
from .bm25_retriever import BM25Retriever
from .reranker import CrossEncoderReranker


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Fuse multiple rankings using Reciprocal Rank Fusion.

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


class HybridRetriever:
    """Hybrid retriever combining dense, sparse, and reranking."""

    def __init__(
        self,
        vector_store: DualVectorStore,
        chunks: List[Chunk],
        embedding_provider: Literal["openai", "opensource"] = "openai",
        use_reranker: bool = True
    ):
        self.vector_store = vector_store
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.embedding_provider = embedding_provider

        # Initialize BM25
        self.bm25 = BM25Retriever(chunks)

        # Initialize reranker if enabled
        self.reranker = CrossEncoderReranker() if use_reranker else None
        self.use_reranker = use_reranker

        logger.info(
            f"Initialized HybridRetriever with {len(chunks)} chunks, "
            f"provider={embedding_provider}, reranker={use_reranker}"
        )

    def retrieve(
        self,
        query: str,
        top_k_dense: int = None,
        top_k_sparse: int = None,
        rerank_top_k: int = None,
        final_top_k: int = None,
        method: Literal["hybrid", "dense", "sparse"] = "hybrid"
    ) -> List[Dict]:
        """
        Retrieve relevant chunks.

        Args:
            query: Query text
            top_k_dense: Number of dense retrieval results
            top_k_sparse: Number of sparse (BM25) results
            rerank_top_k: Number of candidates to rerank
            final_top_k: Final number of results to return
            method: Retrieval method ("hybrid", "dense", or "sparse")

        Returns:
            List of result dictionaries
        """
        top_k_dense = top_k_dense or settings.retrieval_top_k
        top_k_sparse = top_k_sparse or settings.retrieval_top_k
        rerank_top_k = rerank_top_k or settings.rerank_top_k
        final_top_k = final_top_k or settings.final_top_k

        if method == "dense":
            return self._dense_only(query, final_top_k)
        elif method == "sparse":
            return self._sparse_only(query, final_top_k)

        # Hybrid retrieval
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
        fused_ranking = reciprocal_rank_fusion(
            [dense_ranking, sparse_ranking],
            k=settings.rrf_k
        )

        # 4. Get top candidates for reranking
        top_candidates = []
        for chunk_id, rrf_score in fused_ranking[:rerank_top_k]:
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                top_candidates.append({
                    "chunk_id": chunk_id,
                    "content": chunk.content,
                    "score": rrf_score,
                    "metadata": {
                        "doc_id": chunk.doc_id,
                        "section_title": chunk.section_title,
                        "page_numbers": chunk.page_numbers,
                        "section_hierarchy": chunk.section_hierarchy
                    }
                })

        # 5. Cross-encoder reranking (if enabled)
        if self.use_reranker and self.reranker:
            reranked = self.reranker.rerank_dicts(
                query,
                top_candidates,
                top_k=final_top_k
            )
            # Use rerank_score as the final score
            for r in reranked:
                r["score"] = r.pop("rerank_score", r["score"])
            return reranked

        return top_candidates[:final_top_k]

    def _dense_only(self, query: str, top_k: int) -> List[Dict]:
        """Dense retrieval only."""
        results = self.vector_store.search(
            query,
            provider=self.embedding_provider,
            top_k=top_k
        )

        # Enrich with full metadata
        enriched = []
        for r in results:
            chunk = self.chunk_map.get(r["chunk_id"])
            if chunk:
                enriched.append({
                    "chunk_id": r["chunk_id"],
                    "content": chunk.content,
                    "score": r["score"],
                    "metadata": {
                        "doc_id": chunk.doc_id,
                        "section_title": chunk.section_title,
                        "page_numbers": chunk.page_numbers,
                        "section_hierarchy": chunk.section_hierarchy
                    }
                })

        return enriched

    def _sparse_only(self, query: str, top_k: int) -> List[Dict]:
        """Sparse (BM25) retrieval only."""
        return self.bm25.search_with_content(query, top_k)

    def compare_methods(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """Compare different retrieval methods."""
        return {
            "dense": self._dense_only(query, top_k),
            "sparse": self._sparse_only(query, top_k),
            "hybrid": self.retrieve(query, final_top_k=top_k)
        }
