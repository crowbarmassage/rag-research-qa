"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimension size."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (n, dimensions)
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.

        Args:
            query: Query text to embed.

        Returns:
            numpy array of shape (dimensions,)
        """
        pass

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Alias for embed_texts for compatibility."""
        return self.embed_texts(texts)
