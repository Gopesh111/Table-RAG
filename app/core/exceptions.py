"""Explicit, production-grade exception hierarchy for TableRAG-Core."""

from typing import Any, Dict, Optional


class TableRAGException(Exception):
    """Base exception for all domain-specific errors in TableRAG."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentParsingError(TableRAGException):
    """Raised when parsing or extracting text/tables from a document fails."""
    pass


class TableExtractionError(DocumentParsingError):
    """Raised specifically when tabular structure detection or markdown formatting fails."""
    pass


class RetrievalError(TableRAGException):
    """Raised when sparse, dense, or hybrid retrieval encounters an error."""
    pass


class IndexNotFoundError(RetrievalError):
    """Raised when query is executed against uninitialized or missing indexes."""
    pass


class RerankingError(RetrievalError):
    """Raised when cross-encoder reranking encounters a failure."""
    pass


class LLMGenerationError(TableRAGException):
    """Raised when language model generation or streaming fails."""
    pass


class OllamaConnectionError(LLMGenerationError):
    """Raised when local Ollama instance is unreachable or times out."""
    pass
