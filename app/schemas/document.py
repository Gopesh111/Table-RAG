"""Document parsing and ingestion schemas."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Identifies whether an extracted chunk is layout-prose or structured tabular markdown."""
    TEXT = "text"
    TABLE = "table"


class ChunkMetadata(BaseModel):
    """Complete metadata tracking provenance and layout location of an indexed chunk."""
    document_name: str = Field(..., description="Original filename of the ingested document")
    page_number: int = Field(..., ge=1, description="1-indexed page number where chunk was found")
    content_type: ContentType = Field(..., description="Whether chunk is text or table")
    table_id: Optional[str] = Field(default=None, description="Unique table identifier if table")
    section_hint: Optional[str] = Field(default=None, description="Nearest preceding section header")
    chunk_index: int = Field(..., ge=0, description="Sequential index of chunk in document")


class ProcessedChunk(BaseModel):
    """An atomic, searchable piece of context ready for dual sparse-dense indexing."""
    chunk_id: str = Field(..., description="Globally unique UUID or hashed chunk identifier")
    content: str = Field(..., description="Markdown table or text content")
    metadata: ChunkMetadata = Field(..., description="Provenance metadata")


class DocumentIngestResponse(BaseModel):
    """Response returned upon successful document parsing and indexing."""
    document_name: str
    total_pages: int
    total_chunks: int
    total_tables: int
    total_text_chunks: int
    ingestion_latency_ms: float
    status: str = "success"
