from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings loaded from environment or .env file."""

    # Application Settings
    APP_NAME: str = "TableRAG-Core"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = BASE_DIR / "data" / "indexes"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"

    # Models & Retrieval Parameters (CPU optimized)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Hybrid Retrieval (BM25 + FAISS + RRF)
    BM25_TOP_K: int = 10
    DENSE_TOP_K: int = 10
    RRF_K_CONSTANT: int = 60
    RERANK_TOP_N: int = 3

    # Ollama Generation Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT_SECONDS: float = 60.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def setup_directories(self) -> None:
        """Ensures that all required local data and index directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton instance of the Settings object."""
    settings = Settings()
    settings.setup_directories()
    return settings
