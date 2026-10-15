"""Table-preserving and sliding-window prose chunking engine."""

import re
import uuid
from typing import List, Optional

from app.core.logger import get_logger
from app.ingestion.pdf_parser import ExtractedBlock
from app.schemas.document import ChunkMetadata, ContentType, ProcessedChunk

logger = get_logger("chunker")


class LayoutAwareChunker:
    """Chunks documents while preserving markdown tables as atomic units."""

    def __init__(
        self,
        max_chunk_chars: int = 1800,
        chunk_overlap_chars: int = 200,
        max_table_rows_per_chunk: int = 25,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.max_table_rows_per_chunk = max_table_rows_per_chunk

    def chunk_blocks(
        self, document_name: str, blocks: List[ExtractedBlock]
    ) -> List[ProcessedChunk]:
        """Processes extracted blocks into a flat list of ProcessedChunks."""
        processed_chunks: List[ProcessedChunk] = []
        chunk_counter = 0

        for block in blocks:
            if block.content_type == ContentType.TABLE:
                table_chunks = self._chunk_table(
                    document_name=document_name,
                    block=block,
                    start_index=chunk_counter,
                )
                processed_chunks.extend(table_chunks)
                chunk_counter += len(table_chunks)
            else:
                prose_chunks = self._chunk_prose(
                    document_name=document_name,
                    block=block,
                    start_index=chunk_counter,
                )
                processed_chunks.extend(prose_chunks)
                chunk_counter += len(prose_chunks)

        logger.info(
            f"Chunked document '{document_name}' into {len(processed_chunks)} searchable chunks "
            f"({sum(1 for c in processed_chunks if c.metadata.content_type == ContentType.TABLE)} tables, "
            f"{sum(1 for c in processed_chunks if c.metadata.content_type == ContentType.TEXT)} prose)"
        )
        return processed_chunks

    def _chunk_table(
        self, document_name: str, block: ExtractedBlock, start_index: int
    ) -> List[ProcessedChunk]:
        """Preserves markdown tables atomically, or splits long tables by rows keeping the header."""
        lines = block.content.strip().split("\n")
        if len(lines) <= 2:
            return []

        header = lines[0]
        separator = lines[1]
        data_rows = lines[2:]

        # If table fits within character and row limits, keep it as single atomic chunk
        if len(block.content) <= self.max_chunk_chars and len(data_rows) <= self.max_table_rows_per_chunk:
            chunk_id = f"{document_name}-p{block.page_number}-tbl-{start_index}"
            return [
                ProcessedChunk(
                    chunk_id=chunk_id,
                    content=block.content,
                    metadata=ChunkMetadata(
                        document_name=document_name,
                        page_number=block.page_number,
                        content_type=ContentType.TABLE,
                        table_id=block.table_id,
                        section_hint=block.section_hint,
                        chunk_index=start_index,
                    ),
                )
            ]

        # Split large table across row boundaries, duplicating header in each chunk
        table_chunks: List[ProcessedChunk] = []
        step = self.max_table_rows_per_chunk
        split_part = 0

        for idx in range(0, len(data_rows), step):
            subset_rows = data_rows[idx : idx + step]
            chunk_markdown = "\n".join([header, separator] + subset_rows)
            chunk_id = f"{document_name}-p{block.page_number}-tbl-{start_index + split_part}"

            table_chunks.append(
                ProcessedChunk(
                    chunk_id=chunk_id,
                    content=chunk_markdown,
                    metadata=ChunkMetadata(
                        document_name=document_name,
                        page_number=block.page_number,
                        content_type=ContentType.TABLE,
                        table_id=f"{block.table_id}_part{split_part + 1}",
                        section_hint=block.section_hint,
                        chunk_index=start_index + split_part,
                    ),
                )
            )
            split_part += 1

        return table_chunks

    def _chunk_prose(
        self, document_name: str, block: ExtractedBlock, start_index: int
    ) -> List[ProcessedChunk]:
        """Splits prose text using sliding window on paragraph/sentence boundaries."""
        text = block.content.strip()
        if not text:
            return []

        # If text fits in single chunk
        if len(text) <= self.max_chunk_chars:
            chunk_id = f"{document_name}-p{block.page_number}-txt-{start_index}"
            return [
                ProcessedChunk(
                    chunk_id=chunk_id,
                    content=text,
                    metadata=ChunkMetadata(
                        document_name=document_name,
                        page_number=block.page_number,
                        content_type=ContentType.TEXT,
                        table_id=None,
                        section_hint=block.section_hint,
                        chunk_index=start_index,
                    ),
                )
            ]

        # Break by paragraphs or sentences
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)
            if current_len + para_len > self.max_chunk_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Retain last item for overlap if applicable
                overlap_text = current_chunk[-1] if len(current_chunk[-1]) < self.chunk_overlap_chars else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            current_chunk.append(para)
            current_len += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        processed: List[ProcessedChunk] = []
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{document_name}-p{block.page_number}-txt-{start_index + idx}"
            processed.append(
                ProcessedChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    metadata=ChunkMetadata(
                        document_name=document_name,
                        page_number=block.page_number,
                        content_type=ContentType.TEXT,
                        table_id=None,
                        section_hint=block.section_hint,
                        chunk_index=start_index + idx,
                    ),
                )
            )

        return processed
