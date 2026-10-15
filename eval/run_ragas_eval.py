"""Quantitative evaluation and benchmark script for TableRAG-Core."""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.ingestion.chunker import LayoutAwareChunker
from app.ingestion.pdf_parser import PDFTableParser
from app.retrieval.bm25_index import BM25IndexWrapper
from app.retrieval.hybrid_engine import HybridRetrievalEngine
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import FAISSVectorStore


def evaluate_pipeline():
    print("=" * 80)
    print(" " * 22 + "TABLERAG-CORE BENCHMARK EVALUATION")
    print("=" * 80)

    dataset_path = ROOT_DIR / "eval" / "test_dataset.json"
    sample_pdf_path = ROOT_DIR / "data" / "sample_docs" / "techcorp_q3_report.pdf"

    if not dataset_path.exists():
        print(f"Error: Evaluation dataset missing at {dataset_path}")
        return

    # Ensure sample PDF exists; if not, generate it
    if not sample_pdf_path.exists():
        print("Sample PDF not found. Generating sample PDF first...")
        from data.generate_sample_pdf import generate_sample_report
        generate_sample_report(sample_pdf_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} ground-truth QA evaluation cases.\n")

    # Ingest document
    print("Ingesting sample financial document into TableRAG indexes...")
    parser = PDFTableParser()
    blocks = parser.parse_pdf(sample_pdf_path)

    chunker = LayoutAwareChunker()
    chunks = chunker.chunk_blocks("techcorp_q3_report.pdf", blocks)

    vector_store = FAISSVectorStore()
    bm25_index = BM25IndexWrapper()

    vector_store.add_chunks(chunks)
    bm25_index.add_chunks(chunks)

    hybrid_engine = HybridRetrievalEngine(vector_store, bm25_index)
    reranker = CrossEncoderReranker()

    print("Index initialization complete. Running retrieval benchmarks...\n")

    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []
    latencies = []

    print(f"{'ID':<6} | {'Question':<45} | {'Hit@1':<6} | {'Hit@3':<6} | {'Latency':<8}")
    print("-" * 80)

    for case in test_cases:
        query = case["question"]
        expected_page = case["expected_page"]
        key_tokens = [tok.lower() for tok in case["key_tokens"]]

        t0 = time.perf_counter()
        _, _, fused, _ = hybrid_engine.retrieve(query, dense_k=10, sparse_k=10, rrf_k=60)
        reranked, _ = reranker.rerank(query, fused, top_n=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        # Check for presence of ground truth tokens in top retrieved candidates
        found_rank = 0
        for rank, chunk in enumerate(reranked, start=1):
            content_lower = chunk.content.lower()
            if chunk.metadata.page_number == expected_page and any(tok in content_lower for tok in key_tokens):
                found_rank = rank
                break

        if found_rank == 1:
            hits_at_1 += 1
            hits_at_3 += 1
            reciprocal_ranks.append(1.0)
            h1_str, h3_str = "YES", "YES"
        elif 1 < found_rank <= 3:
            hits_at_3 += 1
            reciprocal_ranks.append(1.0 / found_rank)
            h1_str, h3_str = "NO", "YES"
        else:
            reciprocal_ranks.append(0.0)
            h1_str, h3_str = "NO", "NO"

        q_short = query[:42] + "..." if len(query) > 42 else query
        print(f"{case['id']:<6} | {q_short:<45} | {h1_str:<6} | {h3_str:<6} | {elapsed_ms:>6.2f}ms")

    n = len(test_cases)
    mrr = np.mean(reciprocal_ranks)
    hit_rate_1 = (hits_at_1 / n) * 100.0
    hit_rate_3 = (hits_at_3 / n) * 100.0
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    mean_lat = np.mean(latencies)

    print("=" * 80)
    print("BENCHMARK SUMMARY METRICS:")
    print(f"  • Total Evaluation Cases:     {n}")
    print(f"  • Hit Rate @ 1:               {hit_rate_1:.1f}% ({hits_at_1}/{n})")
    print(f"  • Hit Rate @ 3:               {hit_rate_3:.1f}% ({hits_at_3}/{n})")
    print(f"  • Mean Reciprocal Rank (MRR): {mrr:.3f}")
    print(f"  • Retrieval Latency (p50):    {p50_lat:.2f} ms")
    print(f"  • Retrieval Latency (p95):    {p95_lat:.2f} ms")
    print(f"  • Mean Pipeline Latency:      {mean_lat:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_pipeline()
