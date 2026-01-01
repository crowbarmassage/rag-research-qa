"""Cross-encoder reranking."""

from typing import List, Tuple, Dict
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from loguru import logger

from src.config import settings


def sigmoid(x):
    """Apply sigmoid to convert logits to probabilities."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class CrossEncoderReranker:
    """Rerank candidates using a cross-encoder model."""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading cross-encoder {self.model_name} on {self.device}")
        self.model = CrossEncoder(
            self.model_name,
            max_length=512,
            device=self.device
        )

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str, float]],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Query text
            candidates: List of (chunk_id, content, original_score) tuples
            top_k: Number of top results to return

        Returns:
            List of (chunk_id, cross_encoder_score) tuples
        """
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

    def rerank_dicts(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank candidate dicts.

        Args:
            query: Query text
            candidates: List of result dicts with 'chunk_id' and 'content' keys
            top_k: Number of top results to return

        Returns:
            Reranked list of result dicts
        """
        if not candidates:
            return []

        # Create query-document pairs
        pairs = [(query, c["content"]) for c in candidates]

        # Score with cross-encoder
        raw_scores = self.model.predict(pairs)

        # Normalize scores to 0-1 using sigmoid
        normalized_scores = sigmoid(np.array(raw_scores))

        # Update scores and sort
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(normalized_scores[i])

        reranked = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]
