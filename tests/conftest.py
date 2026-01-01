"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    1 Introduction

    The Transformer architecture has revolutionized natural language processing.
    It relies entirely on self-attention mechanisms without using recurrence.

    2 Background

    Previous approaches used recurrent neural networks (RNNs) and LSTMs.
    These models process sequences sequentially, limiting parallelization.

    3 Model Architecture

    3.1 Encoder-Decoder Structure

    The Transformer follows an encoder-decoder structure. The encoder maps
    an input sequence to a sequence of continuous representations.

    3.2 Attention

    Attention allows the model to focus on different parts of the input.
    Multi-head attention enables attending to multiple positions simultaneously.

    4 Conclusion

    We have presented the Transformer, a new architecture for sequence modeling.
    """


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    from src.preprocessing import Chunk

    return [
        Chunk(
            chunk_id="chunk_1",
            doc_id="test_doc",
            content="The Transformer uses self-attention mechanisms.",
            token_count=8,
            page_numbers=[1],
            section_title="Introduction",
            section_hierarchy=["Introduction"],
            chunk_index=0
        ),
        Chunk(
            chunk_id="chunk_2",
            doc_id="test_doc",
            content="Multi-head attention enables parallel processing.",
            token_count=7,
            page_numbers=[2],
            section_title="Attention",
            section_hierarchy=["Model Architecture", "Attention"],
            chunk_index=1
        ),
        Chunk(
            chunk_id="chunk_3",
            doc_id="test_doc",
            content="RNNs process sequences sequentially.",
            token_count=5,
            page_numbers=[1],
            section_title="Background",
            section_hierarchy=["Background"],
            chunk_index=2
        ),
    ]


@pytest.fixture
def mock_results():
    """Mock retrieval results for testing."""
    return [
        {
            "chunk_id": "chunk_1",
            "content": "The Transformer uses self-attention mechanisms to process input sequences.",
            "score": 0.95,
            "metadata": {
                "doc_id": "1706_03762v7",
                "section_title": "3.2 Attention",
                "page_numbers": [3, 4],
                "section_hierarchy": ["3 Model Architecture", "3.2 Attention"]
            }
        },
        {
            "chunk_id": "chunk_2",
            "content": "Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
            "score": 0.88,
            "metadata": {
                "doc_id": "1706_03762v7",
                "section_title": "3.2.2 Multi-Head Attention",
                "page_numbers": [4],
                "section_hierarchy": ["3 Model Architecture", "3.2 Attention", "3.2.2 Multi-Head Attention"]
            }
        }
    ]
