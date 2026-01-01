"""ChromaDB vector store wrapper."""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from src.config import settings
from src.embeddings import get_embedding_provider, EmbeddingProvider
from src.preprocessing import Chunk


class VectorStore:
    """Wrapper for ChromaDB operations."""

    def __init__(
        self,
        collection_name: str,
        embedding_provider: str = "openai",
        persist_directory: Path = None
    ):
        persist_dir = persist_directory or settings.chroma_dir
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embedding_provider = get_embedding_provider(embedding_provider)
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
        logger.info(
            f"Initialized VectorStore: {collection_name} "
            f"with {embedding_provider} embeddings"
        )

    def _get_or_create_collection(self):
        """Get or create the ChromaDB collection."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> int:
        """
        Add chunks to the vector store.

        Args:
            chunks: List of Chunk objects to add
            batch_size: Number of chunks to process at a time

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.content for c in batch]
            embeddings = self.embedding_provider.embed_texts(texts)

            self.collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[self._chunk_to_metadata(c) for c in batch]
            )
            total_added += len(batch)
            logger.debug(f"Added batch {i // batch_size + 1}, total: {total_added}")

        logger.info(f"Added {total_added} chunks to {self.collection_name}")
        return total_added

    def _chunk_to_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Convert a Chunk to ChromaDB metadata."""
        return {
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "page_numbers": json.dumps(chunk.page_numbers),
            "section_title": chunk.section_title or "",
            "section_hierarchy": json.dumps(chunk.section_hierarchy),
            "token_count": chunk.token_count
        }

    def _metadata_to_dict(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ChromaDB metadata back to usable dict."""
        return {
            "doc_id": metadata.get("doc_id", ""),
            "chunk_index": metadata.get("chunk_index", 0),
            "page_numbers": json.loads(metadata.get("page_numbers", "[]")),
            "section_title": metadata.get("section_title", ""),
            "section_hierarchy": json.loads(metadata.get("section_hierarchy", "[]")),
            "token_count": metadata.get("token_count", 0)
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Search for similar chunks.

        Args:
            query: Query text
            top_k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of result dictionaries
        """
        query_embedding = self.embedding_provider.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=filter_dict
        )

        return self._format_results(results)

    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results into a list of dicts."""
        formatted = []

        if not results or not results.get("ids"):
            return formatted

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, chunk_id in enumerate(ids):
            # Convert distance to similarity score (1 - distance for cosine)
            score = 1 - distances[i] if distances else 0.0

            formatted.append({
                "chunk_id": chunk_id,
                "content": documents[i] if documents else "",
                "score": score,
                "metadata": self._metadata_to_dict(metadatas[i]) if metadatas else {}
            })

        return formatted

    def get_all_chunks(self) -> List[Dict]:
        """Get all chunks from the collection."""
        results = self.collection.get(
            include=["documents", "metadatas"]
        )

        formatted = []
        for i, chunk_id in enumerate(results["ids"]):
            formatted.append({
                "chunk_id": chunk_id,
                "content": results["documents"][i],
                "metadata": self._metadata_to_dict(results["metadatas"][i])
            })

        return formatted

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self.collection.count()

    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")


class DualVectorStore:
    """Manager for dual OpenAI and open-source embedding collections."""

    def __init__(self, persist_directory: Path = None):
        self.openai_store = VectorStore(
            "openai_embeddings",
            "openai",
            persist_directory
        )
        self.opensource_store = VectorStore(
            "opensource_embeddings",
            "opensource",
            persist_directory
        )

    def index_chunks(self, chunks: List[Chunk]) -> Dict[str, int]:
        """Index chunks in both collections."""
        logger.info(f"Indexing {len(chunks)} chunks in dual stores...")

        openai_count = self.openai_store.add_chunks(chunks)
        opensource_count = self.opensource_store.add_chunks(chunks)

        return {
            "openai_count": openai_count,
            "opensource_count": opensource_count
        }

    def search(
        self,
        query: str,
        provider: str = "openai",
        top_k: int = 10
    ) -> List[Dict]:
        """Search using specified provider."""
        store = self.openai_store if provider == "openai" else self.opensource_store
        return store.search(query, top_k)

    def get_stats(self) -> Dict[str, int]:
        """Get collection statistics."""
        return {
            "openai_count": self.openai_store.count(),
            "opensource_count": self.opensource_store.count()
        }
