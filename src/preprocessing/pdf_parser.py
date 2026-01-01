"""PDF text extraction with fallback support."""

from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from loguru import logger

import pdfplumber
from pypdf import PdfReader


class PageContent(BaseModel):
    """Content extracted from a single PDF page."""
    page_number: int
    text: str


class DocumentMetadata(BaseModel):
    """Metadata for an extracted document."""
    doc_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    source_file: str
    total_pages: int
    extraction_date: datetime = Field(default_factory=datetime.now)


class ExtractedDocument(BaseModel):
    """Complete extracted document with metadata and content."""
    metadata: DocumentMetadata
    pages: List[PageContent]
    full_text: str


class PDFParser(ABC):
    """Abstract base class for PDF parsers."""

    @abstractmethod
    def extract(self, pdf_path: Path) -> ExtractedDocument:
        """Extract text and metadata from a PDF file."""
        pass

    def _extract_title(self, first_page_text: str) -> str:
        """Heuristic title extraction from first page."""
        lines = first_page_text.strip().split("\n")
        for line in lines[:10]:
            line = line.strip()
            # Skip arXiv headers, short lines, and common non-title patterns
            if len(line) > 15 and not line.startswith("arXiv"):
                if not line.startswith("http") and not line.startswith("www"):
                    return line
        return "Unknown Title"


class PdfPlumberParser(PDFParser):
    """Primary PDF parser using pdfplumber."""

    def extract(self, pdf_path: Path) -> ExtractedDocument:
        """Extract text using pdfplumber."""
        pages = []
        full_text_parts = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(PageContent(page_number=i + 1, text=text))
                full_text_parts.append(text)

        title = self._extract_title(pages[0].text if pages else "")

        metadata = DocumentMetadata(
            doc_id=pdf_path.stem,
            title=title,
            source_file=pdf_path.name,
            total_pages=total_pages
        )

        return ExtractedDocument(
            metadata=metadata,
            pages=pages,
            full_text="\n\n".join(full_text_parts)
        )


class PyPDFParser(PDFParser):
    """Fallback PDF parser using pypdf."""

    def extract(self, pdf_path: Path) -> ExtractedDocument:
        """Extract text using pypdf."""
        reader = PdfReader(pdf_path)
        pages = []
        full_text_parts = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(PageContent(page_number=i + 1, text=text))
            full_text_parts.append(text)

        title = self._extract_title(pages[0].text if pages else "")

        metadata = DocumentMetadata(
            doc_id=pdf_path.stem,
            title=title,
            source_file=pdf_path.name,
            total_pages=len(reader.pages)
        )

        return ExtractedDocument(
            metadata=metadata,
            pages=pages,
            full_text="\n\n".join(full_text_parts)
        )


class CompositeParser(PDFParser):
    """Smart parser that tries pdfplumber first, falls back to pypdf."""

    def __init__(self):
        self.primary = PdfPlumberParser()
        self.fallback = PyPDFParser()

    def extract(self, pdf_path: Path) -> ExtractedDocument:
        """Extract with fallback logic."""
        try:
            result = self.primary.extract(pdf_path)
            if len(result.full_text.strip()) < 100:
                logger.warning(
                    f"Primary extraction yielded little text for {pdf_path.name}, trying fallback"
                )
                return self.fallback.extract(pdf_path)
            logger.info(f"Successfully extracted {pdf_path.name} with pdfplumber")
            return result
        except Exception as e:
            logger.warning(f"Primary parser failed for {pdf_path.name}: {e}, using fallback")
            return self.fallback.extract(pdf_path)
