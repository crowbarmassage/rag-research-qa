"""Tests for retrieval module."""

import pytest
from pathlib import Path
import tempfile

from src.preprocessing import Chunk
from src.retrieval import (
    BM25Retriever,
    reciprocal_rank_fusion,
)


class TestBM25Retriever:
    """Tests for BM25 retriever."""

    def test_index_chunks(self, sample_chunks):
        retriever = BM25Retriever(sample_chunks)
        assert len(retriever.chunks) == 3

    def test_search_returns_results(self, sample_chunks):
        retriever = BM25Retriever(sample_chunks)
        results = retriever.search("attention", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    def test_search_relevance(self, sample_chunks):
        retriever = BM25Retriever(sample_chunks)

        # Query about attention should rank attention chunks higher
        results = retriever.search("self-attention mechanisms", top_k=3)
        chunk_ids = [r[0] for r in results]

        # chunk_1 mentions self-attention, should be in results
        assert "chunk_1" in chunk_ids or "chunk_2" in chunk_ids

    def test_search_with_content(self, sample_chunks):
        retriever = BM25Retriever(sample_chunks)
        results = retriever.search_with_content("attention", top_k=2)

        assert len(results) <= 2
        for r in results:
            assert "chunk_id" in r
            assert "content" in r
            assert "score" in r
            assert "metadata" in r


class TestReciprocalRankFusion:
    """Tests for RRF algorithm."""

    def test_rrf_single_ranking(self):
        ranking = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        fused = reciprocal_rank_fusion([ranking], k=60)

        assert len(fused) == 3
        # Order should be preserved
        assert fused[0][0] == "doc1"
        assert fused[1][0] == "doc2"
        assert fused[2][0] == "doc3"

    def test_rrf_multiple_rankings(self):
        ranking1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        ranking2 = [("doc2", 0.95), ("doc1", 0.85), ("doc4", 0.75)]

        fused = reciprocal_rank_fusion([ranking1, ranking2], k=60)

        # doc2 is ranked 1st and 2nd, should have highest RRF score
        # doc1 is ranked 1st and 2nd, should be close
        assert len(fused) == 4
        top_two = [fused[0][0], fused[1][0]]
        assert "doc1" in top_two
        assert "doc2" in top_two

    def test_rrf_empty_ranking(self):
        fused = reciprocal_rank_fusion([], k=60)
        assert fused == []

    def test_rrf_k_parameter(self):
        ranking = [("doc1", 0.9), ("doc2", 0.8)]

        fused_k10 = reciprocal_rank_fusion([ranking], k=10)
        fused_k100 = reciprocal_rank_fusion([ranking], k=100)

        # With lower k, the difference between ranks is larger
        diff_k10 = fused_k10[0][1] - fused_k10[1][1]
        diff_k100 = fused_k100[0][1] - fused_k100[1][1]

        assert diff_k10 > diff_k100


class TestVectorStoreIntegration:
    """Integration tests for vector store (requires ChromaDB)."""

    def test_add_and_search(self, sample_chunks):
        # Use temporary directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.retrieval import VectorStore

            store = VectorStore(
                collection_name="test_collection",
                embedding_provider="opensource",  # Faster for tests
                persist_directory=Path(tmpdir)
            )

            # Add chunks
            count = store.add_chunks(sample_chunks)
            assert count == len(sample_chunks)

            # Search
            results = store.search("attention", top_k=2)
            assert len(results) <= 2

            for r in results:
                assert "chunk_id" in r
                assert "content" in r
                assert "score" in r

    def test_count(self, sample_chunks):
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.retrieval import VectorStore

            store = VectorStore(
                collection_name="test_count",
                embedding_provider="opensource",
                persist_directory=Path(tmpdir)
            )

            store.add_chunks(sample_chunks)
            assert store.count() == len(sample_chunks)
