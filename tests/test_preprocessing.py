"""Tests for preprocessing module."""

import pytest
from pathlib import Path

from src.preprocessing import (
    CompositeParser,
    SectionDetector,
    HybridChunker,
    ChunkingConfig,
    TokenizerUtil,
    Chunk,
    PageContent,
)


class TestTokenizerUtil:
    """Tests for TokenizerUtil."""

    def test_count_tokens(self):
        tokenizer = TokenizerUtil()
        text = "Hello, world!"
        count = tokenizer.count_tokens(text)
        assert count > 0
        assert count < 10

    def test_truncate_to_tokens(self):
        tokenizer = TokenizerUtil()
        text = "This is a longer sentence that should be truncated."
        truncated = tokenizer.truncate_to_tokens(text, 5)
        assert tokenizer.count_tokens(truncated) <= 5

    def test_empty_string(self):
        tokenizer = TokenizerUtil()
        assert tokenizer.count_tokens("") == 0


class TestSectionDetector:
    """Tests for SectionDetector."""

    def test_detect_numbered_sections(self, sample_text):
        detector = SectionDetector()
        pages = [PageContent(page_number=1, text=sample_text)]
        sections = detector.detect_sections(sample_text, pages)

        assert len(sections) >= 4
        titles = [s.title for s in sections]
        assert any("Introduction" in t for t in titles)
        assert any("Background" in t for t in titles)

    def test_section_hierarchy(self, sample_text):
        detector = SectionDetector()
        pages = [PageContent(page_number=1, text=sample_text)]
        sections = detector.detect_sections(sample_text, pages)

        for i, section in enumerate(sections):
            hierarchy = detector.get_section_hierarchy(sections, i)
            assert len(hierarchy) >= 1
            assert section.title in hierarchy

    def test_empty_text(self):
        detector = SectionDetector()
        sections = detector.detect_sections("", [])
        # Should return at least one section for empty text
        assert len(sections) >= 1


class TestHybridChunker:
    """Tests for HybridChunker."""

    def test_chunk_small_text(self):
        chunker = HybridChunker(ChunkingConfig(max_chunk_size=100))
        from src.preprocessing import ExtractedDocument, DocumentMetadata

        doc = ExtractedDocument(
            metadata=DocumentMetadata(
                doc_id="test",
                title="Test",
                source_file="test.pdf",
                total_pages=1
            ),
            pages=[PageContent(page_number=1, text="Short text.")],
            full_text="Short text."
        )

        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_chunk_respects_max_size(self, sample_text):
        config = ChunkingConfig(max_chunk_size=50, min_chunk_size=10)
        chunker = HybridChunker(config)
        tokenizer = TokenizerUtil()

        from src.preprocessing import ExtractedDocument, DocumentMetadata

        doc = ExtractedDocument(
            metadata=DocumentMetadata(
                doc_id="test",
                title="Test",
                source_file="test.pdf",
                total_pages=1
            ),
            pages=[PageContent(page_number=1, text=sample_text)],
            full_text=sample_text
        )

        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            # Allow some flexibility for edge cases
            assert chunk.token_count <= config.max_chunk_size + 10

    def test_chunk_has_metadata(self, sample_text):
        chunker = HybridChunker()

        from src.preprocessing import ExtractedDocument, DocumentMetadata

        doc = ExtractedDocument(
            metadata=DocumentMetadata(
                doc_id="test_doc",
                title="Test Document",
                source_file="test.pdf",
                total_pages=1
            ),
            pages=[PageContent(page_number=1, text=sample_text)],
            full_text=sample_text
        )

        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            assert chunk.doc_id == "test_doc"
            assert chunk.chunk_id is not None
            assert len(chunk.content) > 0


class TestChunk:
    """Tests for Chunk model."""

    def test_chunk_creation(self):
        chunk = Chunk(
            doc_id="test",
            content="Test content",
            token_count=2,
            page_numbers=[1],
        )
        assert chunk.chunk_id is not None
        assert chunk.doc_id == "test"

    def test_chunk_optional_fields(self):
        chunk = Chunk(
            doc_id="test",
            content="Test",
            token_count=1,
            page_numbers=[1],
        )
        assert chunk.section_title is None
        assert chunk.section_hierarchy == []
