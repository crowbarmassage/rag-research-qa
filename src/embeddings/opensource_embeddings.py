"""Open-source embedding provider using sentence-transformers."""

from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger

from .base import EmbeddingProvider
from src.config import settings


class OpenSourceEmbeddings(EmbeddingProvider):
    """Open-source embeddings using BAAI/bge-base-en-v1.5."""

    def __init__(self, model_name: str = None, device: str = None):
        self._model_name = model_name or settings.embedding_model_opensource
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading embedding model {self._model_name} on {self.device}")
        self.model = SentenceTransformer(
            self._model_name,
            device=self.device
        )
        self._dimensions = self.model.get_sentence_embedding_dimension()
        self._batch_size = 32

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            batch_size=self._batch_size
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embed_texts([query])[0]
