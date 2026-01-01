"""Embedding providers for the RAG system."""

from typing import Literal

from .base import EmbeddingProvider
from .openai_embeddings import OpenAIEmbeddings
from .opensource_embeddings import OpenSourceEmbeddings


def get_embedding_provider(
    provider: Literal["openai", "opensource"] = "openai"
) -> EmbeddingProvider:
    """
    Factory function to get an embedding provider.

    Args:
        provider: Either "openai" or "opensource"

    Returns:
        An embedding provider instance
    """
    if provider == "openai":
        return OpenAIEmbeddings()
    elif provider == "opensource":
        return OpenSourceEmbeddings()
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'opensource'.")


__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddings",
    "OpenSourceEmbeddings",
    "get_embedding_provider",
]
