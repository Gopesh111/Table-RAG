"""Batch ingestion script for sample documents."""

from pathlib import Path
import sys

# Add project root to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.ingestion.chunker import LayoutAwareChunker
from app.ingestion.pdf_parser import PDFTableParser
from app.retrieval.bm25_index import BM25IndexWrapper
from app.retrieval.vector_store import FAISSVectorStore


def ingest_all_samples():
    settings = get_settings()
    sample_dir = settings.DATA_DIR / "sample_docs"
    sample_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(sample_dir.glob("*.pdf"))

    if not pdf_files:
        print("No sample PDFs found. Generating default sample financial report...")
        from data.generate_sample_pdf import generate_sample_report
        default_pdf = sample_dir / "techcorp_q3_report.pdf"
        generate_sample_report(default_pdf)
        pdf_files = [default_pdf]

    parser = PDFTableParser()
    chunker = LayoutAwareChunker()

    vector_store = FAISSVectorStore(
        model_name=settings.EMBEDDING_MODEL_NAME,
        dimension=settings.EMBEDDING_DIMENSION,
    )
    bm25_index = BM25IndexWrapper()

    total_chunks = 0
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        blocks = parser.parse_pdf(pdf_path)
        chunks = chunker.chunk_blocks(pdf_path.name, blocks)

        vector_store.add_chunks(chunks)
        bm25_index.add_chunks(chunks)
        total_chunks += len(chunks)

        table_count = sum(1 for c in chunks if c.metadata.content_type.value == "table")
        print(f" -> Extracted {len(chunks)} chunks ({table_count} tables, {len(chunks) - table_count} prose)")

    # Save to disk
    vector_store.save(settings.INDEX_DIR)
    bm25_index.save(settings.INDEX_DIR)
    print(f"\nSuccessfully indexed {total_chunks} total chunks into {settings.INDEX_DIR}")


if __name__ == "__main__":
    ingest_all_samples()
