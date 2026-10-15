"""Unit tests verifying mathematical correctness of Reciprocal Rank Fusion (RRF)."""

import pytest
from app.retrieval.hybrid_engine import HybridRetrievalEngine
from app.schemas.document import ChunkMetadata, ContentType, ProcessedChunk


class DummyDenseStore:
    def __init__(self, items):
        self.items = items

    def search(self, query, top_k=10):
        # Returns (chunk, score, rank)
        return [(chunk, 0.9 - idx * 0.1, idx + 1) for idx, chunk in enumerate(self.items[:top_k])]


class DummyBM25Index:
    def __init__(self, items):
        self.items = items

    def search(self, query, top_k=10):
        # Returns (chunk, score, rank)
        return [(chunk, 10.0 - idx * 1.0, idx + 1) for idx, chunk in enumerate(self.items[:top_k])]


def test_rrf_scoring_math(sample_processed_chunks):
    """Verify that RRF score matches formula: 1 / (60 + r_dense) + 1 / (60 + r_sparse)."""
    c0, c1, c2 = sample_processed_chunks

    # Dense returns [c0 (rank 1), c1 (rank 2)]
    dense_store = DummyDenseStore([c0, c1])
    # Sparse returns [c1 (rank 1), c2 (rank 2)]
    bm25_store = DummyBM25Index([c1, c2])

    hybrid = HybridRetrievalEngine(
        vector_store=dense_store,
        bm25_index=bm25_store,
        default_rrf_k=60,
    )

    _, _, fused, _ = hybrid.retrieve("test query", dense_k=2, sparse_k=2, rrf_k=60)

    # c1 appears in BOTH dense (rank 2) and sparse (rank 1)
    # Expected c1 score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    expected_c1_score = (1.0 / 62.0) + (1.0 / 61.0)

    # c0 appears in dense only (rank 1)
    # Expected c0 score = 1/(60+1) = 1/61 = 0.016393
    expected_c0_score = 1.0 / 61.0

    # c2 appears in sparse only (rank 2)
    # Expected c2 score = 1/(60+2) = 1/62 = 0.016129
    expected_c2_score = 1.0 / 62.0

    # c1 must rank first because it is reinforced by both sparse and dense channels!
    assert fused[0].chunk_id == c1.chunk_id
    assert pytest.approx(fused[0].rrf_score, rel=1e-4) == expected_c1_score

    assert fused[1].chunk_id == c0.chunk_id
    assert pytest.approx(fused[1].rrf_score, rel=1e-4) == expected_c0_score

    assert fused[2].chunk_id == c2.chunk_id
    assert pytest.approx(fused[2].rrf_score, rel=1e-4) == expected_c2_score
