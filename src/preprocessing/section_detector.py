"""Section boundary detection for academic papers."""

import re
from typing import List, Tuple, Optional
from pydantic import BaseModel

from .pdf_parser import PageContent


class Section(BaseModel):
    """A detected section in a document."""
    title: str
    level: int  # 1 = main, 2 = subsection, etc.
    start_char: int
    end_char: int
    content: str
    page_start: int
    page_end: int


class SectionDetector:
    """Detect section boundaries in academic papers using regex patterns."""

    SECTION_PATTERNS: List[Tuple[str, int]] = [
        # Numbered sections: "1 Introduction", "2 Related Work"
        (r"^(\d+)\s+([A-Z][A-Za-z\s\-]+)$", 1),
        # Subsections: "3.1 Attention", "3.2.1 Details"
        (r"^(\d+\.\d+)\s+([A-Z][A-Za-z\s\-]+)$", 2),
        (r"^(\d+\.\d+\.\d+)\s+([A-Za-z\s\-]+)$", 3),
        # Common section headers without numbers
        (r"^(Abstract)\s*$", 1),
        (r"^(Introduction)\s*$", 1),
        (r"^(Related\s+Work)\s*$", 1),
        (r"^(Background)\s*$", 1),
        (r"^(Methods?|Methodology)\s*$", 1),
        (r"^(Model|Architecture)\s*$", 1),
        (r"^(Experiments?|Experimental\s+Setup)\s*$", 1),
        (r"^(Results?)\s*$", 1),
        (r"^(Discussion)\s*$", 1),
        (r"^(Conclusions?)\s*$", 1),
        (r"^(References)\s*$", 1),
        (r"^(Appendix)\s*([A-Z])?\s*$", 1),
        # Appendix sections: "A Supplementary"
        (r"^([A-Z])\s+([A-Z][A-Za-z\s]+)$", 1),
    ]

    def detect_sections(
        self,
        text: str,
        pages: List[PageContent]
    ) -> List[Section]:
        """Detect sections in the document text."""
        section_markers = []
        lines = text.split("\n")
        current_pos = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            for pattern, level in self.SECTION_PATTERNS:
                match = re.match(pattern, line_stripped, re.IGNORECASE)
                if match:
                    section_markers.append({
                        "title": line_stripped,
                        "level": level,
                        "start_char": current_pos,
                        "line_index": i
                    })
                    break

            current_pos += len(line) + 1  # +1 for newline

        # If no sections found, create one section for entire document
        if not section_markers:
            return [Section(
                title="Document",
                level=1,
                start_char=0,
                end_char=len(text),
                content=text,
                page_start=1,
                page_end=len(pages) if pages else 1
            )]

        # Set end positions
        for i, section in enumerate(section_markers):
            if i + 1 < len(section_markers):
                section["end_char"] = section_markers[i + 1]["start_char"]
            else:
                section["end_char"] = len(text)

        # Build page mapping
        page_char_ranges = self._build_page_ranges(pages)

        # Convert to Section objects
        sections = []
        for s in section_markers:
            content = text[s["start_char"]:s["end_char"]]
            page_start = self._char_to_page(s["start_char"], page_char_ranges)
            page_end = self._char_to_page(s["end_char"] - 1, page_char_ranges)

            sections.append(Section(
                title=s["title"],
                level=s["level"],
                start_char=s["start_char"],
                end_char=s["end_char"],
                content=content,
                page_start=page_start,
                page_end=page_end
            ))

        return sections

    def _build_page_ranges(
        self,
        pages: List[PageContent]
    ) -> List[Tuple[int, int, int]]:
        """Build list of (page_number, start_char, end_char) tuples."""
        ranges = []
        current_pos = 0

        for page in pages:
            page_len = len(page.text) + 2  # +2 for paragraph separator
            ranges.append((page.page_number, current_pos, current_pos + page_len))
            current_pos += page_len

        return ranges

    def _char_to_page(
        self,
        char_pos: int,
        page_ranges: List[Tuple[int, int, int]]
    ) -> int:
        """Convert character position to page number."""
        for page_num, start, end in page_ranges:
            if start <= char_pos < end:
                return page_num
        return page_ranges[-1][0] if page_ranges else 1

    def get_section_hierarchy(
        self,
        sections: List[Section],
        current_index: int
    ) -> List[str]:
        """Get the hierarchy of parent sections for a given section."""
        current = sections[current_index]
        hierarchy = [current.title]

        # Look backwards for parent sections
        for i in range(current_index - 1, -1, -1):
            if sections[i].level < current.level:
                hierarchy.insert(0, sections[i].title)
                current = sections[i]
                if current.level == 1:
                    break

        return hierarchy
