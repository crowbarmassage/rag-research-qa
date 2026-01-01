"""Hybrid chunking with semantic-first and fixed-size fallback."""

import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
import tiktoken

from .pdf_parser import ExtractedDocument
from .section_detector import Section, SectionDetector
from src.config import settings


class Chunk(BaseModel):
    """A chunk of text from a document."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    content: str
    token_count: int
    page_numbers: List[int]
    section_title: Optional[str] = None
    section_hierarchy: List[str] = Field(default_factory=list)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0


class ChunkingConfig(BaseModel):
    """Configuration for chunking."""
    min_chunk_size: int = Field(default=100)
    max_chunk_size: int = Field(default=512)
    fallback_chunk_size: int = Field(default=400)
    chunk_overlap: int = Field(default=50)
    tokenizer: str = Field(default="cl100k_base")


class TokenizerUtil:
    """Utility for token counting and truncation."""

    def __init__(self, model: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to a maximum number of tokens."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])


class HybridChunker:
    """Hybrid chunker with semantic-first and fixed-size fallback."""

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig(
            min_chunk_size=settings.min_chunk_size,
            max_chunk_size=settings.max_chunk_size,
            fallback_chunk_size=settings.fallback_chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.tokenizer = TokenizerUtil(self.config.tokenizer)
        self.section_detector = SectionDetector()

    def chunk_document(
        self,
        doc: ExtractedDocument,
        sections: Optional[List[Section]] = None
    ) -> List[Chunk]:
        """Chunk a document into smaller pieces."""
        if sections is None:
            sections = self.section_detector.detect_sections(
                doc.full_text,
                doc.pages
            )

        chunks = []
        chunk_index = 0

        for i, section in enumerate(sections):
            hierarchy = self.section_detector.get_section_hierarchy(sections, i)
            section_chunks = self._chunk_section(
                section,
                doc.metadata.doc_id,
                hierarchy
            )

            for chunk in section_chunks:
                chunk.chunk_index = chunk_index
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _chunk_section(
        self,
        section: Section,
        doc_id: str,
        hierarchy: List[str]
    ) -> List[Chunk]:
        """Chunk a single section."""
        content = section.content.strip()
        token_count = self.tokenizer.count_tokens(content)

        # Section fits in one chunk
        if token_count <= self.config.max_chunk_size:
            if token_count >= self.config.min_chunk_size:
                return [self._create_chunk(
                    content,
                    section,
                    doc_id,
                    hierarchy,
                    section.start_char,
                    section.end_char
                )]
            else:
                # Too small - will be merged with adjacent chunks
                return [self._create_chunk(
                    content,
                    section,
                    doc_id,
                    hierarchy,
                    section.start_char,
                    section.end_char
                )]

        # Split by paragraphs first
        paragraphs = content.split("\n\n")
        return self._chunk_paragraphs(paragraphs, section, doc_id, hierarchy)

    def _chunk_paragraphs(
        self,
        paragraphs: List[str],
        section: Section,
        doc_id: str,
        hierarchy: List[str]
    ) -> List[Chunk]:
        """Chunk paragraphs, merging small ones and splitting large ones."""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_start = section.start_char

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.tokenizer.count_tokens(para)

            # Paragraph too big - use fixed-size split
            if para_tokens > self.config.max_chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk.strip(),
                        section,
                        doc_id,
                        hierarchy,
                        chunk_start,
                        chunk_start + len(current_chunk)
                    ))
                    chunk_start += len(current_chunk)
                    current_chunk = ""
                    current_tokens = 0

                fixed_chunks = self._fixed_size_split(
                    para,
                    section,
                    doc_id,
                    hierarchy,
                    chunk_start
                )
                chunks.extend(fixed_chunks)
                chunk_start += len(para)
                continue

            # Would exceed max - save current and start new
            if current_tokens + para_tokens > self.config.max_chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk.strip(),
                        section,
                        doc_id,
                        hierarchy,
                        chunk_start,
                        chunk_start + len(current_chunk)
                    ))
                    chunk_start += len(current_chunk)
                current_chunk = para
                current_tokens = para_tokens
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk.strip(),
                section,
                doc_id,
                hierarchy,
                chunk_start,
                section.end_char
            ))

        return chunks

    def _fixed_size_split(
        self,
        text: str,
        section: Section,
        doc_id: str,
        hierarchy: List[str],
        start_offset: int
    ) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        tokens = self.tokenizer.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + self.config.fallback_chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.tokenizer.decode(chunk_tokens)

            chunks.append(self._create_chunk(
                chunk_text,
                section,
                doc_id,
                hierarchy,
                start_offset + start,
                start_offset + end
            ))

            # Move forward with overlap
            start = end - self.config.chunk_overlap
            if start >= len(tokens) - self.config.chunk_overlap:
                break

        return chunks

    def _create_chunk(
        self,
        content: str,
        section: Section,
        doc_id: str,
        hierarchy: List[str],
        start_char: int,
        end_char: int
    ) -> Chunk:
        """Create a Chunk object."""
        # Determine page numbers
        page_numbers = list(range(section.page_start, section.page_end + 1))

        return Chunk(
            doc_id=doc_id,
            content=content,
            token_count=self.tokenizer.count_tokens(content),
            page_numbers=page_numbers,
            section_title=section.title,
            section_hierarchy=hierarchy,
            start_char=start_char,
            end_char=end_char
        )
