"""Main FastAPI application for TableRAG-Core."""

from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import time
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import Settings, get_settings
from app.core.exceptions import IndexNotFoundError, TableRAGException
from app.core.logger import configure_logging, get_logger
from app.generation.llm_client import OllamaClient
from app.generation.prompt_templates import GroundedPromptBuilder
from app.ingestion.chunker import LayoutAwareChunker
from app.ingestion.pdf_parser import PDFTableParser
from app.retrieval.bm25_index import BM25IndexWrapper
from app.retrieval.hybrid_engine import HybridRetrievalEngine
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import FAISSVectorStore
from app.schemas.document import DocumentIngestResponse
from app.schemas.query import (
    DebugRetrievalResponse,
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
)

# Global singleton storage for indexes & models
_vector_store: FAISSVectorStore = None
_bm25_index: BM25IndexWrapper = None
_hybrid_engine: HybridRetrievalEngine = None
_reranker: CrossEncoderReranker = None
_llm_client: OllamaClient = None

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models, directories, and pre-existing indexes upon server startup."""
    global _vector_store, _bm25_index, _hybrid_engine, _reranker, _llm_client
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)

    logger.info(f"Initializing {settings.APP_NAME} in [{settings.APP_ENV}] mode...")

    # Initialize models
    _vector_store = FAISSVectorStore(
        model_name=settings.EMBEDDING_MODEL_NAME,
        dimension=settings.EMBEDDING_DIMENSION,
    )
    _bm25_index = BM25IndexWrapper()
    _hybrid_engine = HybridRetrievalEngine(
        vector_store=_vector_store,
        bm25_index=_bm25_index,
        default_rrf_k=settings.RRF_K_CONSTANT,
    )
    _reranker = CrossEncoderReranker(model_name=settings.RERANKER_MODEL_NAME)
    _llm_client = OllamaClient(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )

    # Attempt to load pre-existing indexes from disk
    if settings.INDEX_DIR.exists():
        _vector_store.load(settings.INDEX_DIR)
        _bm25_index.load(settings.INDEX_DIR)

    logger.info("TableRAG-Core components successfully initialized.")
    yield
    logger.info("Shutting down TableRAG-Core server...")


app = FastAPI(
    title="TableRAG-Core API",
    description="Production-grade hybrid table & document retrieval-augmented generation engine.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler for domain-specific errors
@app.exception_handler(TableRAGException)
async def tablerag_exception_handler(request, exc: TableRAGException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": exc.__class__.__name__, "message": exc.message, "details": exc.details},
    )


# Dependency Injection Helpers
def get_hybrid() -> HybridRetrievalEngine:
    return _hybrid_engine


def get_rerank() -> CrossEncoderReranker:
    return _reranker


def get_llm() -> OllamaClient:
    return _llm_client


def get_cfg() -> Settings:
    return get_settings()


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/api/v1/health")
async def health_check(
    llm: OllamaClient = Depends(get_llm),
    cfg: Settings = Depends(get_cfg),
):
    """System health check, index status, and Ollama daemon connectivity."""
    ollama_ok = await llm.check_health()
    indexed_chunks = len(_vector_store.chunks) if _vector_store else 0

    return {
        "status": "healthy",
        "app_name": cfg.APP_NAME,
        "environment": cfg.APP_ENV,
        "indexes": {
            "is_initialized": _vector_store.is_initialized if _vector_store else False,
            "total_chunks": indexed_chunks,
        },
        "models": {
            "embedding": cfg.EMBEDDING_MODEL_NAME,
            "reranker": cfg.RERANKER_MODEL_NAME,
            "llm": cfg.OLLAMA_MODEL,
        },
        "ollama_connected": ollama_ok,
    }


@app.post("/api/v1/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    cfg: Settings = Depends(get_cfg),
):
    """Uploads a PDF document, extracts layout tables and text, and updates sparse/dense indexes."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents (.pdf) are supported for ingestion.",
        )

    t0 = time.perf_counter()
    save_path = cfg.UPLOAD_DIR / file.filename

    # Save uploaded bytes to local storage
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Parse PDF for tables and prose
    parser = PDFTableParser()
    extracted_blocks = parser.parse_pdf(save_path)

    # 2. Chunk while preserving tables atomically
    chunker = LayoutAwareChunker()
    chunks = chunker.chunk_blocks(document_name=file.filename, blocks=extracted_blocks)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable text or tabular structures could be extracted from this PDF.",
        )

    # 3. Index into FAISS and BM25
    _vector_store.add_chunks(chunks)
    _bm25_index.add_chunks(chunks)

    # 4. Persist updated indexes to disk
    _vector_store.save(cfg.INDEX_DIR)
    _bm25_index.save(cfg.INDEX_DIR)

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    table_count = sum(1 for c in chunks if c.metadata.content_type.value == "table")
    text_count = len(chunks) - table_count

    return DocumentIngestResponse(
        document_name=file.filename,
        total_pages=max((c.metadata.page_number for c in chunks), default=1),
        total_chunks=len(chunks),
        total_tables=table_count,
        total_text_chunks=text_count,
        ingestion_latency_ms=elapsed_ms,
        status="success",
    )


@app.post("/api/v1/chat/query")
async def query_pipeline(
    req: QueryRequest,
    hybrid: HybridRetrievalEngine = Depends(get_hybrid),
    rerank: CrossEncoderReranker = Depends(get_rerank),
    llm: OllamaClient = Depends(get_llm),
    cfg: Settings = Depends(get_cfg),
):
    """Full hybrid retrieval, cross-encoder reranking, and grounded neural generation."""
    if not _vector_store.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Please ingest a PDF via /api/v1/documents/ingest first.",
        )

    total_start = time.perf_counter()

    # 1. Hybrid Retrieval (BM25 + FAISS + RRF)
    dense_cand, sparse_cand, fused_cand, ret_lat = hybrid.retrieve(
        query=req.query,
        dense_k=cfg.DENSE_TOP_K,
        sparse_k=cfg.BM25_TOP_K,
        rrf_k=cfg.RRF_K_CONSTANT,
    )

    # 2. Cross-Encoder Reranking
    reranked_chunks, rerank_ms = rerank.rerank(
        query=req.query,
        candidates=fused_cand,
        top_n=req.top_k,
    )

    # 3. Grounded Prompt Construction
    prompt = GroundedPromptBuilder.build_prompt(query=req.query, chunks=reranked_chunks)

    # 4. Handle Streaming Response if requested
    if req.stream:
        async def stream_generator() -> AsyncGenerator[str, None]:
            async for token in llm.generate_stream(prompt):
                yield token

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # 5. Full non-streaming Generation
    gen_start = time.perf_counter()
    answer_text = await llm.generate(prompt)
    gen_ms = round((time.perf_counter() - gen_start) * 1000.0, 3)
    total_ms = round((time.perf_counter() - total_start) * 1000.0, 3)

    latency = LatencyBreakdown(
        dense_ms=ret_lat.get("dense_ms", 0.0),
        sparse_ms=ret_lat.get("sparse_ms", 0.0),
        rrf_ms=ret_lat.get("rrf_ms", 0.0),
        rerank_ms=rerank_ms,
        generation_ms=gen_ms,
        total_ms=total_ms,
    )

    return QueryResponse(
        query=req.query,
        answer=answer_text,
        citations=reranked_chunks,
        latency_breakdown=latency,
    )


@app.post("/api/v1/retrieval/debug", response_model=DebugRetrievalResponse)
async def debug_retrieval(
    req: QueryRequest,
    hybrid: HybridRetrievalEngine = Depends(get_hybrid),
    rerank: CrossEncoderReranker = Depends(get_rerank),
    cfg: Settings = Depends(get_cfg),
):
    """Diagnostic endpoint displaying multi-stage candidate rankings without running the LLM."""
    if not _vector_store.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Indexes are empty. Ingest a document first.",
        )

    total_start = time.perf_counter()

    dense_cand, sparse_cand, fused_cand, ret_lat = hybrid.retrieve(
        query=req.query,
        dense_k=cfg.DENSE_TOP_K,
        sparse_k=cfg.BM25_TOP_K,
        rrf_k=cfg.RRF_K_CONSTANT,
    )

    reranked_chunks, rerank_ms = rerank.rerank(
        query=req.query,
        candidates=fused_cand,
        top_n=req.top_k,
    )
    total_ms = round((time.perf_counter() - total_start) * 1000.0, 3)

    latency = LatencyBreakdown(
        dense_ms=ret_lat.get("dense_ms", 0.0),
        sparse_ms=ret_lat.get("sparse_ms", 0.0),
        rrf_ms=ret_lat.get("rrf_ms", 0.0),
        rerank_ms=rerank_ms,
        generation_ms=0.0,
        total_ms=total_ms,
    )

    return DebugRetrievalResponse(
        query=req.query,
        dense_candidates=dense_cand,
        sparse_candidates=sparse_cand,
        rrf_fused_candidates=fused_cand,
        final_reranked_candidates=reranked_chunks,
        latency_breakdown=latency,
    )
