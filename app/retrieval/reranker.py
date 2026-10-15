"""Cross-encoder reranking engine for high-precision final candidate scoring."""

import time
from typing import List, Tuple
from sentence_transformers import CrossEncoder

from app.core.exceptions import RerankingError
from app.core.logger import get_logger
from app.schemas.query import ScoredChunk

logger = get_logger("reranker")


class CrossEncoderReranker:
    """Reranks candidate chunks by evaluating full cross-attention query-document pairs."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        logger.info(f"Loading CrossEncoder model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: List[ScoredChunk], top_n: int = 3
    ) -> Tuple[List[ScoredChunk], float]:
        """Scores (query, document) pairs using the cross-encoder and returns top N chunks."""
        if not candidates:
            return [], 0.0

        t0 = time.perf_counter()
        pairs = [[query, chunk.content] for chunk in candidates]

        try:
            raw_scores = self.model.predict(pairs)
            for idx, score in enumerate(raw_scores):
                candidates[idx].rerank_score = float(score)

            # Sort descending by cross-encoder logit score
            reranked = sorted(
                candidates, key=lambda x: x.rerank_score or -9999.0, reverse=True
            )
            top_results = reranked[:top_n]
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

            logger.info(
                f"Reranked {len(candidates)} candidates down to {len(top_results)} in {elapsed_ms}ms"
            )
            return top_results, elapsed_ms

        except Exception as exc:
            logger.error(f"Cross-encoder reranking failed: {exc}", exc_info=True)
            raise RerankingError(f"Reranking failure: {str(exc)}") from exc
