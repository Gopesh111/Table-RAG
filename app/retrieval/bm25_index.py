"""Rank-BM25 sparse search wrapper with tokenization and persistence."""

import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple
from rank_bm25 import BM25Okapi

from app.core.exceptions import IndexNotFoundError, RetrievalError
from app.core.logger import get_logger
from app.schemas.document import ProcessedChunk

logger = get_logger("bm25_index")


class BM25IndexWrapper:
    """Manages exact keyword and lexical sparse search using BM25Okapi."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[ProcessedChunk] = []
        self.tokenized_corpus: List[List[str]] = []

    @property
    def is_initialized(self) -> bool:
        """Returns True if the BM25 index contains indexed documents."""
        return self.bm25 is not None and len(self.chunks) > 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenizes text by lowercasing and extracting alphanumeric tokens."""
        return re.findall(r"\w+", text.lower())

    def add_chunks(self, chunks: List[ProcessedChunk]) -> None:
        """Tokenizes new chunks and builds the BM25Okapi inverted index."""
        if not chunks:
            return

        self.chunks.extend(chunks)
        new_tokens = [self.tokenize(c.content) for c in chunks]
        self.tokenized_corpus.extend(new_tokens)

        logger.info(f"Building BM25Okapi index over {len(self.tokenized_corpus)} total documents")
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[ProcessedChunk, float, int]]:
        """Executes sparse keyword search, returning (chunk, bm25_score, 1-indexed rank)."""
        if not self.is_initialized:
            raise IndexNotFoundError("BM25 index is empty. Ingest documents before querying.")

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        k = min(top_k, len(self.chunks))
        results: List[Tuple[ProcessedChunk, float, int]] = []

        for rank_idx, idx in enumerate(top_indices[:k], start=1):
            results.append((self.chunks[idx], float(scores[idx]), rank_idx))

        return results

    def save(self, directory: Path) -> None:
        """Serializes BM25 corpus and chunk mapping to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        dump_path = directory / "bm25_data.pkl"

        state = {
            "chunks": self.chunks,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(dump_path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved BM25 index ({len(self.chunks)} items) to {directory}")

    def load(self, directory: Path) -> bool:
        """Loads and rebuilds BM25 index from persisted data."""
        dump_path = directory / "bm25_data.pkl"
        if not dump_path.exists():
            logger.warning(f"No BM25 data found at {directory}")
            return False

        try:
            with open(dump_path, "rb") as f:
                state = pickle.load(f)
            self.chunks = state["chunks"]
            self.tokenized_corpus = state["tokenized_corpus"]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"Loaded BM25 index with {len(self.chunks)} documents from {directory}")
            return True
        except Exception as exc:
            logger.error(f"Failed to load BM25 index from {directory}: {exc}")
            raise RetrievalError(f"Corrupted BM25 data at {directory}") from exc
