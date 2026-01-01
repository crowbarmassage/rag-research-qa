# RAG-QA System - Future Features & Enhancement Roadmap

## Overview

This document outlines potential enhancements beyond the core deliverable. Features are prioritized by impact and implementation complexity.

---

## Tier 1: High-Impact, Low Complexity

### 1.1 Query Expansion & Reformulation
**Description**: Automatically expand user queries with synonyms and related terms to improve retrieval recall.

**Implementation**:
```python
class QueryExpander:
    def expand(self, query: str) -> List[str]:
        # Use LLM to generate query variants
        prompt = f"""Generate 3 alternative phrasings for this question:
        Original: {query}
        
        Return only the alternative questions, one per line."""
        
        variants = self.llm.generate(prompt).split("\n")
        return [query] + variants
```

**Benefit**: 10-20% improvement in recall for ambiguous queries  
**Effort**: 2-3 hours

---

### 1.2 Streaming Response Generation
**Description**: Stream LLM responses token-by-token for better UX.

**Implementation**:
```python
async def generate_stream(self, question: str, context: str):
    stream = self.client.chat.completions.create(
        model=self.model,
        messages=[...],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

**Benefit**: Perceived latency reduction from 3-5s to <500ms first token  
**Effort**: 2-3 hours

---

### 1.3 Caching Layer
**Description**: Cache embeddings, retrieval results, and generated answers for repeated queries.

**Implementation**:
```python
from functools import lru_cache
import hashlib

class CachedRetriever:
    def __init__(self, retriever):
        self.retriever = retriever
        self.cache = {}
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        cache_key = hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = self.retriever.retrieve(query, top_k)
        self.cache[cache_key] = results
        return results
```

**Benefit**: 90%+ latency reduction for repeated queries  
**Effort**: 1-2 hours

---

### 1.4 Confidence Calibration
**Description**: Improve confidence scoring using retrieval score distribution and answer-context alignment.

**Implementation**:
```python
def calculate_confidence(
    self,
    retrieval_scores: List[float],
    answer: str,
    context: str
) -> float:
    # 1. Retrieval score component
    avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
    
    # 2. Score distribution (high variance = less confident)
    score_variance = np.var(retrieval_scores)
    distribution_penalty = min(score_variance * 2, 0.3)
    
    # 3. Answer-context alignment (check if answer uses context)
    alignment = self._calculate_alignment(answer, context)
    
    return min(avg_retrieval - distribution_penalty + alignment * 0.2, 1.0)
```

**Benefit**: More reliable confidence scores for user decision-making  
**Effort**: 3-4 hours

---

## Tier 2: High-Impact, Medium Complexity

### 2.1 Multi-Document Synthesis
**Description**: When a question requires information from multiple papers, explicitly synthesize across sources.

**Implementation**:
```python
SYNTHESIS_PROMPT = """Based on the following sources from different papers, 
synthesize a comprehensive answer that:
1. Identifies common themes across papers
2. Notes differences in approaches or findings
3. Provides a unified understanding

Sources:
{context}

Question: {question}"""
```

**Benefit**: Better answers for cross-paper questions  
**Effort**: 4-6 hours

---

### 2.2 Evaluation Dashboard
**Description**: Interactive dashboard showing retrieval quality, answer accuracy, and system metrics.

**Features**:
- Retrieval metrics (Precision@K, MRR, NDCG)
- Answer quality indicators
- Response time distribution
- Embedding comparison visualizations

**Implementation**: Streamlit or Gradio app

**Benefit**: Ongoing quality monitoring  
**Effort**: 6-8 hours

---

### 2.3 Adaptive Chunk Size
**Description**: Dynamically adjust chunk retrieval based on query complexity.

**Logic**:
```python
def adaptive_top_k(self, query: str) -> int:
    # Simple queries need fewer chunks
    word_count = len(query.split())
    
    if word_count < 5:  # Simple factual query
        return 3
    elif word_count < 15:  # Standard query
        return 5
    else:  # Complex multi-part query
        return 8
```

**Benefit**: Reduced noise for simple queries, better coverage for complex ones  
**Effort**: 3-4 hours

---

### 2.4 Table & Figure Extraction
**Description**: Extract and index tables/figures from PDFs separately for structured data queries.

**Implementation**:
```python
def extract_tables(self, pdf_path: Path) -> List[Table]:
    with pdfplumber.open(pdf_path) as pdf:
        tables = []
        for page_num, page in enumerate(pdf.pages):
            for table in page.extract_tables():
                tables.append(Table(
                    content=table,
                    page=page_num,
                    caption=self._find_caption(page, table)
                ))
        return tables
```

**Benefit**: Accurate answers for data-specific questions  
**Effort**: 6-8 hours

---

### 2.5 Conversational Memory
**Description**: Track conversation history for follow-up questions.

**Implementation**:
```python
class ConversationalRAG:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.history: List[Dict] = []
    
    def answer(self, question: str) -> AnswerResponse:
        # Rewrite question with context
        if self.history:
            rewritten = self._rewrite_with_context(question)
        else:
            rewritten = question
        
        response = self.pipeline.answer(rewritten)
        
        self.history.append({
            "question": question,
            "answer": response.answer
        })
        
        return response
    
    def _rewrite_with_context(self, question: str) -> str:
        prompt = f"""Previous conversation:
        {self._format_history()}
        
        Current question: {question}
        
        Rewrite the current question to be self-contained:"""
        
        return self.llm.generate(prompt)
```

**Benefit**: Natural follow-up question handling  
**Effort**: 4-6 hours

---

## Tier 3: High-Impact, High Complexity

### 3.1 Fine-Tuned Retrieval Model
**Description**: Fine-tune embedding model on domain-specific query-document pairs.

**Approach**:
1. Generate synthetic query-document pairs from papers
2. Use contrastive learning to fine-tune BGE model
3. Replace open-source embeddings with fine-tuned version

**Benefit**: 15-30% retrieval improvement on domain-specific queries  
**Effort**: 15-20 hours

---

### 3.2 Agentic RAG with Tool Use
**Description**: Agent that can decide when to search, when to ask for clarification, and when to answer.

**Architecture**:
```
User Query → Agent → [Search | Clarify | Answer | Compare]
                         ↓
                    Tool Execution
                         ↓
                    Agent Decision
                         ↓
                    Final Response
```

**Tools**:
- `search(query)`: Retrieve from vector store
- `compare_papers(topic)`: Compare treatment across papers
- `get_definition(term)`: Find specific term definition
- `clarify(question)`: Ask user for clarification

**Benefit**: More sophisticated query handling  
**Effort**: 20-30 hours

---

### 3.3 PDF Visual Understanding
**Description**: Use vision models to understand figures, diagrams, and equations in papers.

**Implementation**:
1. Extract images from PDFs
2. Use GPT-4V or similar to generate descriptions
3. Index image descriptions as searchable content
4. Link image chunks to text chunks

**Benefit**: Answer questions about visual content  
**Effort**: 15-20 hours

---

### 3.4 Citation Graph Navigation
**Description**: Build and navigate citation relationships between papers.

**Features**:
- Extract citations from papers
- Build citation graph
- Answer questions like "What papers cite the attention mechanism?"
- Navigate "papers cited by" and "papers citing" relationships

**Benefit**: Research discovery capabilities  
**Effort**: 15-20 hours

---

### 3.5 Incremental Indexing
**Description**: Add new papers without re-indexing entire corpus.

**Implementation**:
```python
class IncrementalIndexer:
    def add_document(self, pdf_path: Path) -> IndexStats:
        # Check if already indexed
        if self._is_indexed(pdf_path):
            return IndexStats(added=0, skipped=1)
        
        # Process and add
        doc = self.parser.extract(pdf_path)
        chunks = self.chunker.chunk_document(doc)
        self.vector_store.add_chunks(chunks)
        
        # Update BM25 index incrementally
        self.bm25.add_documents(chunks)
        
        return IndexStats(added=len(chunks), skipped=0)
```

**Benefit**: Scalability for growing corpus  
**Effort**: 6-8 hours

---

## Tier 4: Research Extensions

### 4.1 Hallucination Detection
**Description**: Detect when LLM generates content not supported by retrieved context.

**Approach**:
- NLI-based fact checking against source chunks
- Highlight potentially hallucinated claims
- Confidence penalties for unsupported statements

**Benefit**: Improved answer reliability  
**Effort**: 10-15 hours

---

### 4.2 Retrieval-Augmented Fine-Tuning (RAFT)
**Description**: Fine-tune the generator on retrieval-augmented examples.

**Approach**:
- Generate synthetic QA pairs from papers
- Include retrieved context in training
- Fine-tune smaller model for domain-specific generation

**Benefit**: Better generation quality, lower API costs  
**Effort**: 20-30 hours

---

### 4.3 Learned Sparse Retrieval (SPLADE)
**Description**: Replace BM25 with learned sparse representations.

**Implementation**:
- Use pre-trained SPLADE model
- Generate sparse vectors for documents and queries
- Combine with dense retrieval

**Benefit**: Better sparse retrieval than BM25  
**Effort**: 8-10 hours

---

### 4.4 Query-Adaptive Retrieval
**Description**: Automatically select retrieval strategy based on query characteristics.

**Strategies**:
- Factual queries → BM25 heavy
- Conceptual queries → Dense heavy
- Comparison queries → Multi-document retrieval
- Definition queries → Exact match + dense

**Benefit**: Optimal retrieval per query type  
**Effort**: 10-15 hours

---

## Implementation Priority Matrix

| Feature | Impact | Complexity | Priority |
|---------|--------|------------|----------|
| Query Expansion | High | Low | P1 |
| Streaming Responses | High | Low | P1 |
| Caching Layer | High | Low | P1 |
| Confidence Calibration | Medium | Low | P1 |
| Multi-Document Synthesis | High | Medium | P2 |
| Evaluation Dashboard | Medium | Medium | P2 |
| Conversational Memory | High | Medium | P2 |
| Table Extraction | Medium | Medium | P2 |
| Fine-Tuned Retrieval | Very High | High | P3 |
| Agentic RAG | Very High | High | P3 |
| PDF Visual Understanding | High | High | P3 |
| Hallucination Detection | High | High | P4 |

---

## Quick Wins for Demo Enhancement

If you want to enhance the demo quickly, focus on:

1. **Streaming** - Immediate UX improvement
2. **Query Expansion** - Better recall with minimal code
3. **Caching** - Makes repeated demos faster
4. **Better prompts** - Iterate on system prompts for citation quality

These can all be implemented in a single day and significantly improve the demo experience.

---

## Notes for Future Implementation

1. **Start with evaluation** - Before implementing features, establish baseline metrics
2. **A/B test changes** - Compare retrieval quality before/after
3. **Monitor costs** - Track API usage for OpenAI-dependent features
4. **Consider latency** - Some features (reranking, query expansion) add latency
5. **Document everything** - Update TECH_SPECS.md as architecture evolves
