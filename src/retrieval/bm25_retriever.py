"""BM25 sparse retrieval."""

import re
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from loguru import logger

from src.preprocessing import Chunk
from src.config import settings


class BM25Retriever:
    """Sparse retrieval using BM25."""

    # Common English stopwords
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "the", "this", "but", "they",
        "have", "had", "what", "when", "where", "who", "which", "why", "how"
    }

    def __init__(self, chunks: List[Chunk] = None):
        self.chunks = []
        self.chunk_map = {}
        self.tokenized_docs = []
        self.bm25 = None

        if chunks:
            self.index(chunks)

    def index(self, chunks: List[Chunk]):
        """Index chunks for BM25 retrieval."""
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}

        # Tokenize documents
        self.tokenized_docs = [self._tokenize(c.content) for c in chunks]

        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_docs,
            k1=settings.bm25_k1,
            b=settings.bm25_b
        )

        logger.info(f"BM25 indexed {len(chunks)} chunks")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)

        # Remove stopwords and short tokens
        tokens = [
            t for t in tokens
            if t not in self.STOPWORDS and len(t) > 2
        ]

        return tokens

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search for relevant chunks.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (chunk_id, score) tuples
        """
        if not self.bm25:
            logger.warning("BM25 index not built")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                chunk_id = self.chunks[idx].chunk_id
                results.append((chunk_id, float(scores[idx])))

        return results

    def get_chunk(self, chunk_id: str) -> Chunk:
        """Get a chunk by ID."""
        return self.chunk_map.get(chunk_id)

    def search_with_content(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict]:
        """Search and return full chunk information."""
        results = self.search(query, top_k)

        return [
            {
                "chunk_id": chunk_id,
                "content": self.chunk_map[chunk_id].content,
                "score": score,
                "metadata": {
                    "doc_id": self.chunk_map[chunk_id].doc_id,
                    "section_title": self.chunk_map[chunk_id].section_title,
                    "page_numbers": self.chunk_map[chunk_id].page_numbers
                }
            }
            for chunk_id, score in results
            if chunk_id in self.chunk_map
        ]
