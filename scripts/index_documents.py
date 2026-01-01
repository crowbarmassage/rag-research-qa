#!/usr/bin/env python
"""Index all research papers into ChromaDB."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from tqdm import tqdm

from src.config import settings
from src.preprocessing import CompositeParser, SectionDetector, HybridChunker
from src.retrieval import DualVectorStore


def main():
    """Index all PDFs in the papers directory."""
    papers_dir = Path(settings.papers_dir)

    if not papers_dir.exists():
        papers_dir.mkdir(parents=True)
        logger.warning(f"Created papers directory: {papers_dir}")
        logger.info("Please add PDF files to this directory and run again.")
        return

    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {papers_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files")

    # Initialize components
    parser = CompositeParser()
    detector = SectionDetector()
    chunker = HybridChunker()
    vector_store = DualVectorStore()

    all_chunks = []

    # Process each PDF
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        logger.info(f"Processing: {pdf_file.name}")

        try:
            doc = parser.extract(pdf_file)
            sections = detector.detect_sections(doc.full_text, doc.pages)
            chunks = chunker.chunk_document(doc, sections)

            all_chunks.extend(chunks)
            logger.info(f"  Title: {doc.metadata.title[:50]}...")
            logger.info(f"  Pages: {doc.metadata.total_pages}")
            logger.info(f"  Sections: {len(sections)}")
            logger.info(f"  Chunks: {len(chunks)}")

        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")

    if not all_chunks:
        logger.error("No chunks created. Check PDF files.")
        return

    # Index chunks
    logger.info(f"\nIndexing {len(all_chunks)} chunks into dual vector stores...")
    stats = vector_store.index_chunks(all_chunks)

    logger.info("\n" + "=" * 50)
    logger.info("Indexing Complete!")
    logger.info(f"  OpenAI embeddings: {stats['openai_count']} chunks")
    logger.info(f"  Open-source embeddings: {stats['opensource_count']} chunks")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
