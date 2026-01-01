#!/usr/bin/env python
"""Evaluate the RAG system with sample questions."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.pipeline import RAGPipeline


# Sample evaluation questions from TECH_SPECS
EVALUATION_QUESTIONS = [
    {
        "question": "What are the main components of a RAG model, and how do they interact?",
        "expected_sources": ["2005_11401v4", "2005.11401v4"],
        "key_concepts": ["retriever", "generator", "DPR", "BART"]
    },
    {
        "question": "What are the two sub-layers in each encoder layer of the Transformer model?",
        "expected_sources": ["1706_03762v7", "1706.03762v7"],
        "key_concepts": ["self-attention", "feed-forward", "residual", "layer normalization"]
    },
    {
        "question": "Explain how positional encoding is implemented in Transformers and why it is necessary.",
        "expected_sources": ["1706_03762v7", "1706.03762v7"],
        "key_concepts": ["sinusoidal", "position", "sequence order", "no recurrence"]
    },
    {
        "question": "Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?",
        "expected_sources": ["1706_03762v7", "1706.03762v7"],
        "key_concepts": ["multiple heads", "different subspaces", "parallel attention", "concatenate"]
    },
    {
        "question": "What is few-shot learning, and how does GPT-3 implement it during inference?",
        "expected_sources": ["2005_14165v4", "2005.14165v4"],
        "key_concepts": ["in-context learning", "demonstrations", "no gradient updates", "prompt"]
    }
]


def evaluate_retrieval(pipeline: RAGPipeline, question: dict, top_k: int = 5):
    """Evaluate retrieval quality for a question."""
    results = pipeline.retrieve_only(question["question"], top_k=top_k)

    # Check if expected source is in results
    doc_ids = [r["metadata"]["doc_id"] for r in results]
    expected = question["expected_sources"]

    hit = any(
        any(exp in doc_id for exp in expected)
        for doc_id in doc_ids
    )

    # Calculate MRR
    mrr = 0
    for rank, doc_id in enumerate(doc_ids, 1):
        if any(exp in doc_id for exp in expected):
            mrr = 1 / rank
            break

    return {
        "hit": hit,
        "mrr": mrr,
        "retrieved_docs": doc_ids
    }


def evaluate_answer(pipeline: RAGPipeline, question: dict):
    """Evaluate answer generation quality."""
    response = pipeline.answer(question["question"])

    # Check if key concepts are mentioned
    answer_lower = response.answer.lower()
    concepts_found = sum(
        1 for concept in question["key_concepts"]
        if concept.lower() in answer_lower
    )
    concept_coverage = concepts_found / len(question["key_concepts"])

    return {
        "answer": response.answer,
        "confidence": response.confidence,
        "concept_coverage": concept_coverage,
        "concepts_found": concepts_found,
        "total_concepts": len(question["key_concepts"]),
        "processing_time_ms": response.processing_time_ms
    }


def main():
    logger.info("Loading RAG pipeline for evaluation...")

    try:
        pipeline = RAGPipeline.from_documents()
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        logger.info("Run 'python scripts/index_documents.py' first.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("RAG-QA System Evaluation")
    print("=" * 70)

    # Retrieval evaluation
    print("\n--- Retrieval Evaluation ---")
    retrieval_hits = []
    mrr_scores = []

    for q in EVALUATION_QUESTIONS:
        result = evaluate_retrieval(pipeline, q)
        retrieval_hits.append(1 if result["hit"] else 0)
        mrr_scores.append(result["mrr"])

        status = "✓" if result["hit"] else "✗"
        print(f"{status} {q['question'][:50]}...")
        print(f"   MRR: {result['mrr']:.3f}, Docs: {result['retrieved_docs'][:3]}")

    precision_at_5 = sum(retrieval_hits) / len(retrieval_hits)
    mean_mrr = sum(mrr_scores) / len(mrr_scores)

    print(f"\nRetrieval Precision@5: {precision_at_5:.1%}")
    print(f"Mean Reciprocal Rank: {mean_mrr:.3f}")

    # Answer generation evaluation
    print("\n--- Answer Generation Evaluation ---")
    total_coverage = 0
    total_time = 0

    for q in EVALUATION_QUESTIONS:
        result = evaluate_answer(pipeline, q)
        total_coverage += result["concept_coverage"]
        total_time += result["processing_time_ms"]

        print(f"\nQ: {q['question'][:60]}...")
        print(f"   Concept Coverage: {result['concept_coverage']:.0%} ({result['concepts_found']}/{result['total_concepts']})")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Time: {result['processing_time_ms']:.0f}ms")

    avg_coverage = total_coverage / len(EVALUATION_QUESTIONS)
    avg_time = total_time / len(EVALUATION_QUESTIONS)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Retrieval Precision@5: {precision_at_5:.1%} (target: >80%)")
    print(f"  Mean Reciprocal Rank: {mean_mrr:.3f}")
    print(f"  Concept Coverage: {avg_coverage:.1%}")
    print(f"  Avg Processing Time: {avg_time:.0f}ms")

    # Pass/fail
    if precision_at_5 >= 0.8:
        print("\n✓ Retrieval quality PASSED")
    else:
        print("\n✗ Retrieval quality BELOW target")


if __name__ == "__main__":
    main()
