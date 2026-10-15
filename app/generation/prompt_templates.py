"""Grounded prompt templates with strict citation and anti-hallucination constraints."""

from typing import List
from app.schemas.document import ContentType
from app.schemas.query import ScoredChunk


class GroundedPromptBuilder:
    """Builds strictly grounded system and user prompts with provenance headers."""

    SYSTEM_PROMPT = (
        "You are an accurate, analytical financial and technical document assistant. "
        "Your mission is to answer user questions using ONLY the facts and data provided in the CONTEXT below.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. Answer based strictly on the provided context. Do NOT extrapolate or hallucinate numbers.\n"
        "2. When citing numbers, metrics, or table values, ALWAYS cite the exact Page Number and Table ID (e.g. '[Page 2, table_p2_1]').\n"
        "3. If a table is relevant, preserve the numerical precision verbatim from the markdown table.\n"
        "4. If the provided context does not contain enough information to answer the question, state: "
        "'I cannot verify this from the provided documents.' Do not attempt to guess.\n"
    )

    @classmethod
    def format_context(cls, chunks: List[ScoredChunk]) -> str:
        """Formats retrieved chunks into clear, demarcated context blocks."""
        formatted_blocks: List[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            if meta.content_type == ContentType.TABLE:
                header = f"--- [DOCUMENT: {meta.document_name} | PAGE: {meta.page_number} | TABLE: {meta.table_id or 'Table'}] ---"
            else:
                header = f"--- [DOCUMENT: {meta.document_name} | PAGE: {meta.page_number} | PROSE (Section: {meta.section_hint or 'General'})] ---"

            formatted_blocks.append(f"{header}\n{chunk.content}\n")

        return "\n".join(formatted_blocks)

    @classmethod
    def build_prompt(cls, query: str, chunks: List[ScoredChunk]) -> str:
        """Constructs the complete prompt ready for Ollama inference."""
        context_str = cls.format_context(chunks)

        user_content = (
            f"CONTEXT DOCUMENTS:\n"
            f"{context_str}\n"
            f"-----------------------------------------\n"
            f"USER QUESTION: {query}\n\n"
            f"GROUNDED ANSWER (Include page and table citations where applicable):"
        )

        return f"{cls.SYSTEM_PROMPT}\n\n{user_content}"
