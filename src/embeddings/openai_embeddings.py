"""OpenAI embedding provider."""

from typing import List
import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from .base import EmbeddingProvider
from src.config import settings


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI text-embedding-3-small embeddings."""

    def __init__(self, api_key: str = None, model: str = None):
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self._model = model or settings.embedding_model_openai
        self._dimensions = 1536
        self._batch_size = 100

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts with retry logic."""
        response = self.client.embeddings.create(
            model=self._model,
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, batching if necessary."""
        if not texts:
            return np.array([])

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            logger.debug(f"Embedding batch {i // self._batch_size + 1}")
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embed_texts([query])[0]
