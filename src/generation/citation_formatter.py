"""Citation formatting and source attribution."""

import re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class SourceAttribution(BaseModel):
    """Structured source attribution."""
    source_index: int
    chunk_id: str
    paper_title: str
    section: Optional[str] = None
    page_numbers: List[int] = Field(default_factory=list)
    relevance_score: float
    excerpt: str


class CitationFormatter:
    """Format citations and build source attributions."""

    # Map document IDs to paper titles
    PAPER_TITLES = {
        "1706.03762v7": "Attention Is All You Need",
        "1706_03762v7": "Attention Is All You Need",
        "2005.11401v4": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "2005_11401v4": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "2005.14165v4": "Language Models are Few-Shot Learners",
        "2005_14165v4": "Language Models are Few-Shot Learners",
    }

    def get_paper_title(self, doc_id: str) -> str:
        """Get the paper title for a document ID."""
        return self.PAPER_TITLES.get(doc_id, doc_id)

    def format_inline_citations(
        self,
        answer: str,
        sources: List[Dict]
    ) -> str:
        """
        Replace [Source N] with detailed inline citations.

        Args:
            answer: Generated answer with [Source N] citations
            sources: List of source dictionaries

        Returns:
            Answer with expanded citations
        """
        def replace_citation(match):
            source_num = int(match.group(1)) - 1
            if source_num < len(sources):
                source = sources[source_num]
                doc_id = source.get("metadata", {}).get("doc_id", "")
                title = self.get_paper_title(doc_id)
                section = source.get("metadata", {}).get("section_title", "")
                pages = source.get("metadata", {}).get("page_numbers", [])

                citation_parts = [title]
                if section:
                    citation_parts.append(f"Section: {section}")
                if pages:
                    if isinstance(pages, list) and pages:
                        citation_parts.append(f"Page(s): {', '.join(map(str, pages[:2]))}")
                    elif pages:
                        citation_parts.append(f"Page: {pages}")

                return f"[{', '.join(citation_parts)}]"
            return match.group(0)

        return re.sub(r'\[Source (\d+)\]', replace_citation, answer)

    def build_source_attributions(
        self,
        sources: List[Dict]
    ) -> List[SourceAttribution]:
        """
        Build structured source attribution list.

        Args:
            sources: List of source dictionaries

        Returns:
            List of SourceAttribution objects
        """
        attributions = []

        for i, source in enumerate(sources):
            metadata = source.get("metadata", {})
            doc_id = metadata.get("doc_id", "")
            content = source.get("content", "")

            # Create excerpt (first 200 chars)
            excerpt = content[:200].strip()
            if len(content) > 200:
                excerpt += "..."

            page_numbers = metadata.get("page_numbers", [])
            if isinstance(page_numbers, str):
                try:
                    import json
                    page_numbers = json.loads(page_numbers)
                except:
                    page_numbers = []

            attributions.append(SourceAttribution(
                source_index=i + 1,
                chunk_id=source.get("chunk_id", ""),
                paper_title=self.get_paper_title(doc_id),
                section=metadata.get("section_title"),
                page_numbers=page_numbers if isinstance(page_numbers, list) else [],
                relevance_score=source.get("score", 0.0),
                excerpt=excerpt
            ))

        return attributions

    def format_sources_section(
        self,
        attributions: List[SourceAttribution]
    ) -> str:
        """Format a sources section for the response."""
        if not attributions:
            return ""

        lines = ["\n---\n**Sources:**\n"]
        for attr in attributions:
            line = f"[{attr.source_index}] {attr.paper_title}"
            if attr.section:
                line += f", {attr.section}"
            if attr.page_numbers:
                line += f" (pp. {', '.join(map(str, attr.page_numbers[:2]))})"
            lines.append(line)

        return "\n".join(lines)
