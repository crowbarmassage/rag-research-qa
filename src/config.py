"""Configuration management for the RAG-QA system."""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default=Path("data"))
    papers_dir: Path = Field(default=Path("data/papers"))
    chroma_dir: Path = Field(default=Path("data/chroma_db"))

    # OpenAI
    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")
    embedding_model_openai: str = Field(
        default="text-embedding-3-small",
        validation_alias="EMBEDDING_MODEL_OPENAI"
    )
    llm_model: str = Field(default="gpt-4o-mini", validation_alias="LLM_MODEL")

    # Open-source embeddings
    embedding_model_opensource: str = Field(
        default="BAAI/bge-base-en-v1.5",
        validation_alias="EMBEDDING_MODEL_OPENSOURCE"
    )

    # Chunking
    max_chunk_size: int = Field(default=512)
    min_chunk_size: int = Field(default=100)
    chunk_overlap: int = Field(default=50)
    fallback_chunk_size: int = Field(default=400)

    # Retrieval
    retrieval_top_k: int = Field(default=20, validation_alias="RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(default=10, validation_alias="RERANK_TOP_K")
    final_top_k: int = Field(default=5)
    bm25_k1: float = Field(default=1.5)
    bm25_b: float = Field(default=0.75)
    rrf_k: int = Field(default=60)

    # Generation
    llm_temperature: float = Field(default=0.1)
    max_output_tokens: int = Field(default=1000)
    max_context_tokens: int = Field(default=4000)

    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


# Global settings instance
settings = Settings()
