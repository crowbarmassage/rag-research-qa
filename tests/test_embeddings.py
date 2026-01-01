"""Tests for embeddings module."""

import pytest
import numpy as np

from src.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    OpenSourceEmbeddings,
    get_embedding_provider,
)


class TestEmbeddingFactory:
    """Tests for embedding provider factory."""

    def test_get_openai_provider(self):
        provider = get_embedding_provider("openai")
        assert isinstance(provider, OpenAIEmbeddings)

    def test_get_opensource_provider(self):
        provider = get_embedding_provider("opensource")
        assert isinstance(provider, OpenSourceEmbeddings)

    def test_invalid_provider(self):
        with pytest.raises(ValueError):
            get_embedding_provider("invalid")


class TestOpenAIEmbeddings:
    """Tests for OpenAI embeddings."""

    @pytest.fixture
    def provider(self):
        return OpenAIEmbeddings()

    def test_dimensions(self, provider):
        assert provider.dimensions == 1536

    def test_model_name(self, provider):
        assert "embedding" in provider.model_name.lower()

    def test_embed_single_text(self, provider):
        embeddings = provider.embed_texts(["Hello, world!"])
        assert embeddings.shape == (1, 1536)

    def test_embed_multiple_texts(self, provider):
        texts = ["First text", "Second text", "Third text"]
        embeddings = provider.embed_texts(texts)
        assert embeddings.shape == (3, 1536)

    def test_embed_query(self, provider):
        embedding = provider.embed_query("What is attention?")
        assert embedding.shape == (1536,)

    def test_embeddings_normalized(self, provider):
        embedding = provider.embed_query("Test text")
        norm = np.linalg.norm(embedding)
        # OpenAI embeddings should be roughly normalized
        assert 0.9 < norm < 1.1


class TestOpenSourceEmbeddings:
    """Tests for open-source embeddings."""

    @pytest.fixture
    def provider(self):
        return OpenSourceEmbeddings()

    def test_dimensions(self, provider):
        # BGE-base has 768 dimensions
        assert provider.dimensions == 768

    def test_model_name(self, provider):
        assert "bge" in provider.model_name.lower()

    def test_embed_single_text(self, provider):
        embeddings = provider.embed_texts(["Hello, world!"])
        assert embeddings.shape == (1, 768)

    def test_embed_multiple_texts(self, provider):
        texts = ["First text", "Second text", "Third text"]
        embeddings = provider.embed_texts(texts)
        assert embeddings.shape == (3, 768)

    def test_embed_query(self, provider):
        embedding = provider.embed_query("What is attention?")
        assert embedding.shape == (768,)

    def test_embeddings_normalized(self, provider):
        embedding = provider.embed_query("Test text")
        norm = np.linalg.norm(embedding)
        # Should be normalized
        assert 0.99 < norm < 1.01


class TestEmbeddingSimilarity:
    """Tests for embedding similarity."""

    def test_similar_texts_higher_similarity(self):
        provider = OpenSourceEmbeddings()

        texts = [
            "The Transformer uses self-attention",
            "Self-attention is used in Transformers",
            "I like pizza and ice cream"
        ]

        embeddings = provider.embed_texts(texts)

        # Cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_01 = cosine_sim(embeddings[0], embeddings[1])
        sim_02 = cosine_sim(embeddings[0], embeddings[2])

        # Similar texts should have higher similarity
        assert sim_01 > sim_02
