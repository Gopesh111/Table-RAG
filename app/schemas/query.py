"""Query request, scoring, citations, and debug response schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.document import ChunkMetadata


class QueryRequest(BaseModel):
    """User query payload."""
    query: str = Field(..., min_length=2, max_length=1000, description="User search query or question")
    stream: bool = Field(default=False, description="Whether to stream LLM response via SSE")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of reranked chunks to feed to LLM")


class ScoredChunk(BaseModel):
    """A chunk enriched with retrieval ranks and cross-encoder scores."""
    chunk_id: str
    content: str
    metadata: ChunkMetadata
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None


class LatencyBreakdown(BaseModel):
    """Detailed timing metrics across every stage of the pipeline in milliseconds."""
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    rrf_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class QueryResponse(BaseModel):
    """Grounded question answering response with citations and latency breakdown."""
    query: str
    answer: str
    citations: List[ScoredChunk]
    latency_breakdown: LatencyBreakdown


class DebugRetrievalResponse(BaseModel):
    """Diagnostic response showing multi-stage candidate scoring without LLM invocation."""
    query: str
    dense_candidates: List[ScoredChunk]
    sparse_candidates: List[ScoredChunk]
    rrf_fused_candidates: List[ScoredChunk]
    final_reranked_candidates: List[ScoredChunk]
    latency_breakdown: LatencyBreakdown
