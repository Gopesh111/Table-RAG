"""Unit tests for layout-aware table markdown parsing and chunking."""

import pytest
from app.ingestion.chunker import LayoutAwareChunker
from app.ingestion.pdf_parser import ExtractedBlock, PDFTableParser
from app.schemas.document import ContentType


def test_table_to_markdown_formatting(sample_table_data):
    """Verify that 2D table cells are converted into valid GitHub-Flavored Markdown."""
    md = PDFTableParser.table_to_markdown(sample_table_data)

    lines = md.strip().split("\n")
    # Must have header + separator + 3 data rows = 5 lines
    assert len(lines) == 5

    # Check header row
    assert "| Quarter | Revenue ($M) | Growth (%) | Notes |" in lines[0]

    # Check separator row
    assert "| --- | --- | --- | --- |" in lines[1]

    # Check sanitization of None -> N/A
    assert "| Q2 2024 | $138.2 | 26.6% | N/A |" in lines[3]

    # Check newline replacement inside cell
    assert "\n" not in lines[4].replace("\n", "")
    assert "Record high New records" in lines[4]


def test_table_to_markdown_empty():
    """Verify empty tables return an empty string without raising an exception."""
    assert PDFTableParser.table_to_markdown([]) == ""
    assert PDFTableParser.table_to_markdown([[]]) == ""


def test_chunker_preserves_tables_atomically():
    """Verify that small/medium tables are not split into pieces."""
    md_table = (
        "| Col1 | Col2 |\n"
        "|---|---|\n"
        "| Val1 | Val2 |\n"
        "| Val3 | Val4 |\n"
    )
    blocks = [
        ExtractedBlock(
            content=md_table,
            content_type=ContentType.TABLE,
            page_number=1,
            table_id="table_p1_1",
            section_hint="Sample Section",
        ),
        ExtractedBlock(
            content="This is surrounding narrative text.",
            content_type=ContentType.TEXT,
            page_number=1,
            table_id=None,
            section_hint="Sample Section",
        ),
    ]

    chunker = LayoutAwareChunker()
    chunks = chunker.chunk_blocks("test_doc.pdf", blocks)

    assert len(chunks) == 2
    table_chunk = next(c for c in chunks if c.metadata.content_type == ContentType.TABLE)
    assert table_chunk.content == md_table
    assert table_chunk.metadata.table_id == "table_p1_1"
    assert table_chunk.metadata.page_number == 1
