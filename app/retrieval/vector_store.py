"""FAISS dense vector store wrapper with persistence and cosine similarity."""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.exceptions import IndexNotFoundError, RetrievalError
from app.core.logger import get_logger
from app.schemas.document import ProcessedChunk

logger = get_logger("vector_store")


class FAISSVectorStore:
    """Manages exact cosine similarity dense vector retrieval using FAISS IndexFlatIP."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.embedder = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[ProcessedChunk] = []

    @property
    def is_initialized(self) -> bool:
        """Returns True if the FAISS index contains indexed vectors."""
        return self.index.ntotal > 0 and len(self.chunks) > 0

    def add_chunks(self, chunks: List[ProcessedChunk]) -> None:
        """Encodes chunk contents, normalizes vectors to unit sphere, and indexes into FAISS."""
        if not chunks:
            return

        texts = [chunk.content for chunk in chunks]
        logger.info(f"Generating dense embeddings for {len(texts)} chunks using {self.model_name}")

        # Compute embeddings as float32
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization converts Inner Product to Cosine Similarity
        ).astype(np.float32)

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"FAISS index now contains {self.index.ntotal} total vectors")

    def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[ProcessedChunk, float, int]]:
        """Performs dense vector search, returning (chunk, cosine_score, 1-indexed rank)."""
        if not self.is_initialized:
            raise IndexNotFoundError("FAISS vector store is empty. Ingest documents before querying.")

        query_vector = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results: List[Tuple[ProcessedChunk, float, int]] = []
        for rank_idx, (score, chunk_idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if chunk_idx != -1 and chunk_idx < len(self.chunks):
                results.append((self.chunks[chunk_idx], float(score), rank_idx))

        return results

    def save(self, directory: Path) -> None:
        """Serializes the FAISS index and metadata chunks to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / "faiss_index.bin"
        chunks_path = directory / "faiss_chunks.pkl"

        faiss.write_index(self.index, str(index_path))
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Saved FAISS index ({self.index.ntotal} items) to {directory}")

    def load(self, directory: Path) -> bool:
        """Loads a persisted FAISS index and metadata from disk."""
        index_path = directory / "faiss_index.bin"
        chunks_path = directory / "faiss_chunks.pkl"

        if not index_path.exists() or not chunks_path.exists():
            logger.warning(f"No FAISS index found at {directory}")
            return False

        try:
            self.index = faiss.read_index(str(index_path))
            with open(chunks_path, "wb") as f:
                self.chunks = pickle.load(f)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors from {directory}")
            return True
        except Exception as exc:
            logger.error(f"Failed to load FAISS index from {directory}: {exc}")
            raise RetrievalError(f"Corrupted FAISS index at {directory}") from exc
