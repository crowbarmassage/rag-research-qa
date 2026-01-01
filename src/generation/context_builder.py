"""Context assembly for LLM generation."""

from typing import List, Dict

from src.config import settings
from src.preprocessing import TokenizerUtil


class ContextBuilder:
    """Build context string from retrieved chunks."""

    def __init__(self, max_tokens: int = None):
        self.max_tokens = max_tokens or settings.max_context_tokens
        self.tokenizer = TokenizerUtil()

    def build_context(self, results: List[Dict]) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            results: List of retrieval result dicts

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_parts = []
        current_tokens = 0

        for i, result in enumerate(results):
            source_header = self._format_source_header(i + 1, result)
            chunk_text = result.get("content", "")

            part = f"{source_header}\n{chunk_text}\n"
            part_tokens = self.tokenizer.count_tokens(part)

            if current_tokens + part_tokens > self.max_tokens:
                # Try to fit a truncated version
                remaining = self.max_tokens - current_tokens - 50  # Buffer
                if remaining > 100:
                    truncated = self.tokenizer.truncate_to_tokens(chunk_text, remaining)
                    part = f"{source_header}\n{truncated}...\n"
                    context_parts.append(part)
                break

            context_parts.append(part)
            current_tokens += part_tokens

        return "\n".join(context_parts)

    def _format_source_header(self, index: int, result: Dict) -> str:
        """Format the source header for a chunk."""
        meta = result.get("metadata", {})
        pages = meta.get("page_numbers", [])
        section = meta.get("section_title", "")
        doc_id = meta.get("doc_id", "")

        parts = [f"[Source {index}"]

        if doc_id:
            parts.append(f": {doc_id}")
        if section:
            parts.append(f", Section: {section}")
        if pages:
            if isinstance(pages, list):
                page_str = ", ".join(map(str, pages[:3]))
                if len(pages) > 3:
                    page_str += "..."
            else:
                page_str = str(pages)
            parts.append(f", Pages: {page_str}")

        parts.append("]")
        return "".join(parts)

    def get_source_summary(self, results: List[Dict]) -> str:
        """Get a summary of sources used."""
        sources = []
        for i, result in enumerate(results):
            meta = result.get("metadata", {})
            doc_id = meta.get("doc_id", "unknown")
            section = meta.get("section_title", "")
            sources.append(f"[{i + 1}] {doc_id}: {section}")
        return "\n".join(sources)
