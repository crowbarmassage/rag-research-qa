#!/usr/bin/env python
"""CLI to query the RAG system."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from loguru import logger

from src.pipeline import RAGPipeline
from src.config import settings


def main():
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (or use --interactive)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode"
    )
    parser.add_argument(
        "-m", "--method",
        choices=["hybrid", "openai", "opensource"],
        default="hybrid",
        help="Retrieval method"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of sources to retrieve"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking"
    )

    args = parser.parse_args()

    # Initialize pipeline
    logger.info("Loading RAG pipeline...")
    try:
        pipeline = RAGPipeline.from_documents(
            use_reranker=not args.no_rerank
        )
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        logger.info("Run 'python scripts/index_documents.py' first.")
        sys.exit(1)

    def ask(question: str):
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("=" * 60)

        response = pipeline.answer(
            question=question,
            retrieval_method=args.method,
            top_k=args.top_k
        )

        print(f"\nAnswer:\n{response.answer}")
        print(f"\n{'─'*60}")
        print(f"Confidence: {response.confidence:.2%}")
        print(f"Method: {response.retrieval_method}")
        print(f"Time: {response.processing_time_ms:.0f}ms")
        print(f"\nSources:")
        for s in response.sources:
            print(f"  [{s.source_index}] {s.paper_title}")
            if s.section:
                print(f"      Section: {s.section}")
            if s.page_numbers:
                print(f"      Pages: {s.page_numbers}")

    if args.interactive:
        print("\nRAG-QA Interactive Mode")
        print("Type 'quit' or 'exit' to stop\n")

        while True:
            try:
                question = input("\nYour question: ").strip()
                if question.lower() in ["quit", "exit", "q"]:
                    break
                if question:
                    ask(question)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")

        print("\nGoodbye!")

    elif args.question:
        ask(args.question)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
