"""Unit tests for cross-encoder reranking behavior."""

import pytest
from app.retrieval.reranker import CrossEncoderReranker
from app.schemas.query import ScoredChunk


class DummyModel:
    def predict(self, pairs):
        # Predicts higher score if keyword 'Revenue' is present
        scores = []
        for query, text in pairs:
            if "Revenue" in text:
                scores.append(4.5)
            elif "Uptime" in text:
                scores.append(2.1)
            else:
                scores.append(-1.2)
        return scores


def test_reranker_sorting_and_truncation(sample_processed_chunks):
    """Verify that reranker scores candidates, sorts descending, and returns top N."""
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = DummyModel()
    reranker.model_name = "dummy-cross-encoder"

    candidates = [
        ScoredChunk(
            chunk_id=c.chunk_id,
            content=c.content,
            metadata=c.metadata,
            rrf_score=0.01,
        )
        for c in sample_processed_chunks
    ]

    top_results, elapsed_ms = reranker.rerank(
        query="What is the total revenue?",
        candidates=candidates,
        top_n=2,
    )

    # Must return exactly 2 items
    assert len(top_results) == 2

    # Top item must be the Revenue chunk with score 4.5
    assert top_results[0].rerank_score == 4.5
    assert "Revenue" in top_results[0].content

    # Second item must be Uptime chunk with score 2.1
    assert top_results[1].rerank_score == 2.1
    assert elapsed_ms >= 0.0
