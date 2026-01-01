"""Tests for generation module."""

import pytest

from src.generation import (
    ContextBuilder,
    CitationFormatter,
    LLMClient,
    AnswerResponse,
    SourceAttribution,
)


class TestContextBuilder:
    """Tests for ContextBuilder."""

    def test_build_context(self, mock_results):
        builder = ContextBuilder(max_tokens=1000)
        context = builder.build_context(mock_results)

        assert len(context) > 0
        assert "[Source 1" in context
        assert "[Source 2" in context

    def test_context_includes_content(self, mock_results):
        builder = ContextBuilder()
        context = builder.build_context(mock_results)

        assert "self-attention" in context.lower()
        assert "multi-head" in context.lower()

    def test_empty_results(self):
        builder = ContextBuilder()
        context = builder.build_context([])
        assert context == ""

    def test_source_header_format(self, mock_results):
        builder = ContextBuilder()
        context = builder.build_context(mock_results)

        assert "1706_03762v7" in context
        assert "Section:" in context or "Attention" in context


class TestCitationFormatter:
    """Tests for CitationFormatter."""

    def test_get_paper_title(self):
        formatter = CitationFormatter()

        title = formatter.get_paper_title("1706_03762v7")
        assert title == "Attention Is All You Need"

        title = formatter.get_paper_title("unknown_id")
        assert title == "unknown_id"

    def test_format_inline_citations(self, mock_results):
        formatter = CitationFormatter()

        answer = "The model uses [Source 1] for attention and [Source 2] for multi-head."
        formatted = formatter.format_inline_citations(answer, mock_results)

        assert "[Source 1]" not in formatted
        assert "Attention Is All You Need" in formatted

    def test_build_source_attributions(self, mock_results):
        formatter = CitationFormatter()
        attributions = formatter.build_source_attributions(mock_results)

        assert len(attributions) == 2
        assert all(isinstance(a, SourceAttribution) for a in attributions)

        assert attributions[0].source_index == 1
        assert attributions[0].paper_title == "Attention Is All You Need"
        assert attributions[0].relevance_score == 0.95

    def test_format_sources_section(self, mock_results):
        formatter = CitationFormatter()
        attributions = formatter.build_source_attributions(mock_results)
        sources_section = formatter.format_sources_section(attributions)

        assert "Sources:" in sources_section
        assert "[1]" in sources_section
        assert "[2]" in sources_section


class TestLLMClient:
    """Tests for LLMClient."""

    @pytest.fixture
    def client(self):
        return LLMClient()

    def test_generate_answer(self, client, mock_results):
        from src.generation import ContextBuilder

        builder = ContextBuilder()
        context = builder.build_context(mock_results)

        answer = client.generate(
            question="What is self-attention?",
            context=context
        )

        assert len(answer) > 50
        # Should reference sources
        assert "[Source" in answer or "attention" in answer.lower()

    def test_generate_with_custom_temperature(self, client, mock_results):
        from src.generation import ContextBuilder

        builder = ContextBuilder()
        context = builder.build_context(mock_results)

        answer = client.generate(
            question="What is attention?",
            context=context,
            temperature=0.0
        )

        assert len(answer) > 0


class TestAnswerResponse:
    """Tests for AnswerResponse schema."""

    def test_create_response(self):
        response = AnswerResponse(
            question="What is attention?",
            answer="Attention is a mechanism...",
            sources=[
                SourceAttribution(
                    source_index=1,
                    chunk_id="chunk_1",
                    paper_title="Test Paper",
                    section="Introduction",
                    page_numbers=[1, 2],
                    relevance_score=0.95,
                    excerpt="Test excerpt..."
                )
            ],
            retrieval_method="hybrid",
            confidence=0.9,
            processing_time_ms=1234.5
        )

        assert response.question == "What is attention?"
        assert len(response.sources) == 1
        assert response.confidence == 0.9

    def test_response_json(self):
        response = AnswerResponse(
            question="Test?",
            answer="Answer.",
            sources=[],
            retrieval_method="hybrid",
            confidence=0.5,
            processing_time_ms=100.0
        )

        json_str = response.model_dump_json()
        assert "question" in json_str
        assert "answer" in json_str
