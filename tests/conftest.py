"""Pytest fixtures for TableRAG-Core tests."""

import pytest
from app.schemas.document import ChunkMetadata, ContentType, ProcessedChunk


@pytest.fixture
def sample_table_data():
    """Returns a raw 2D list of table cells with edge cases."""
    return [
        ["Quarter", "Revenue ($M)", "Growth (%)", "Notes"],
        ["Q1 2024", "$124.5", "25.1%", "Preliminary"],
        ["Q2 2024", "$138.2", "26.6%", None],  # Contains None to test sanitization
        ["Q3 2024", "$156.8", "28.4%", "Record high\nNew records"],  # Contains internal newline
    ]


@pytest.fixture
def sample_processed_chunks():
    """Returns a list of sample processed chunks for retrieval and reranking testing."""
    return [
        ProcessedChunk(
            chunk_id="doc1-p1-tbl-0",
            content="| Metric | Q3 2024 |\n|---|---|\n| Total Revenue | $156.8M |\n| Net Income | $35.2M |",
            metadata=ChunkMetadata(
                document_name="test_report.pdf",
                page_number=1,
                content_type=ContentType.TABLE,
                table_id="table_p1_1",
                section_hint="Financial Highlights",
                chunk_index=0,
            ),
        ),
        ProcessedChunk(
            chunk_id="doc1-p2-tbl-1",
            content="| Region | Uptime |\n|---|---|\n| US-East | 99.985% |\n| EU-Central | 99.940% |",
            metadata=ChunkMetadata(
                document_name="test_report.pdf",
                page_number=2,
                content_type=ContentType.TABLE,
                table_id="table_p2_1",
                section_hint="Infrastructure SLA",
                chunk_index=1,
            ),
        ),
        ProcessedChunk(
            chunk_id="doc1-p1-txt-2",
            content="TechCorp experienced significant revenue expansion during Q3 2024 driven by enterprise adoption.",
            metadata=ChunkMetadata(
                document_name="test_report.pdf",
                page_number=1,
                content_type=ContentType.TEXT,
                table_id=None,
                section_hint="Executive Summary",
                chunk_index=2,
            ),
        ),
    ]
