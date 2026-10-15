"""Hybrid Retrieval Engine combining dense FAISS and sparse BM25 via Reciprocal Rank Fusion."""

import time
from typing import Dict, List, Tuple

from app.core.logger import get_logger
from app.retrieval.bm25_index import BM25IndexWrapper
from app.retrieval.vector_store import FAISSVectorStore
from app.schemas.query import ScoredChunk

logger = get_logger("hybrid_engine")


class HybridRetrievalEngine:
    """Combines semantic vector search and exact lexical BM25 using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_index: BM25IndexWrapper,
        default_rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.default_rrf_k = default_rrf_k

    def retrieve(
        self,
        query: str,
        dense_k: int = 10,
        sparse_k: int = 10,
        rrf_k: int = 60,
    ) -> Tuple[List[ScoredChunk], List[ScoredChunk], List[ScoredChunk], Dict[str, float]]:
        """Executes parallel sparse & dense retrieval and computes Reciprocal Rank Fusion scores.

        RRF Formula:
            RRF_Score(d) = 1 / (rrf_k + rank_dense) + 1 / (rrf_k + rank_sparse)
        """
        latencies: Dict[str, float] = {}

        # 1. Dense FAISS Vector Search
        t0 = time.perf_counter()
        dense_results = self.vector_store.search(query, top_k=dense_k)
        latencies["dense_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

        # 2. Sparse BM25 Keyword Search
        t1 = time.perf_counter()
        sparse_results = self.bm25_index.search(query, top_k=sparse_k)
        latencies["sparse_ms"] = round((time.perf_counter() - t1) * 1000.0, 3)

        # 3. Reciprocal Rank Fusion
        t2 = time.perf_counter()
        fused_map: Dict[str, ScoredChunk] = {}

        # Process dense candidates
        dense_candidates: List[ScoredChunk] = []
        for chunk, score, rank in dense_results:
            rrf_contrib = 1.0 / (rrf_k + rank)
            sc = ScoredChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                dense_rank=rank,
                sparse_rank=None,
                rrf_score=rrf_contrib,
            )
            dense_candidates.append(sc)
            fused_map[chunk.chunk_id] = sc

        # Process sparse candidates
        sparse_candidates: List[ScoredChunk] = []
        for chunk, score, rank in sparse_results:
            rrf_contrib = 1.0 / (rrf_k + rank)
            sc = ScoredChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                dense_rank=None,
                sparse_rank=rank,
                rrf_score=rrf_contrib,
            )
            sparse_candidates.append(sc)

            if chunk.chunk_id in fused_map:
                fused_map[chunk.chunk_id].sparse_rank = rank
                fused_map[chunk.chunk_id].rrf_score += rrf_contrib
            else:
                fused_map[chunk.chunk_id] = sc

        # Sort all candidates by descending RRF score
        fused_candidates = sorted(
            fused_map.values(), key=lambda x: x.rrf_score, reverse=True
        )
        latencies["rrf_ms"] = round((time.perf_counter() - t2) * 1000.0, 3)

        logger.info(
            f"Hybrid retrieval for query '{query[:40]}...': "
            f"Dense={len(dense_candidates)} ({latencies['dense_ms']}ms), "
            f"Sparse={len(sparse_candidates)} ({latencies['sparse_ms']}ms), "
            f"Fused={len(fused_candidates)} ({latencies['rrf_ms']}ms)"
        )

        return dense_candidates, sparse_candidates, fused_candidates, latencies
