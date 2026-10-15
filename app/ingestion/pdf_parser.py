"""Layout-aware PDF and tabular parser using pdfplumber."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import pdfplumber

from app.core.exceptions import DocumentParsingError, TableExtractionError
from app.core.logger import get_logger
from app.schemas.document import ContentType

logger = get_logger("pdf_parser")


@dataclass
class ExtractedBlock:
    """Intermediate representation of extracted layout elements."""
    content: str
    content_type: ContentType
    page_number: int
    table_id: Optional[str] = None
    section_hint: Optional[str] = None


class PDFTableParser:
    """Extracts tables as GitHub-flavored Markdown and isolates prose text with spatial awareness."""

    @staticmethod
    def table_to_markdown(raw_table: List[List[Optional[str]]]) -> str:
        """Converts raw 2D extracted table cells into sanitized GitHub-Flavored Markdown."""
        if not raw_table or not any(raw_table):
            return ""

        # Normalize rows to match max column count
        max_cols = max(len(row) for row in raw_table if row)
        sanitized_rows: List[List[str]] = []

        for row in raw_table:
            clean_row = []
            for col_idx in range(max_cols):
                cell_val = row[col_idx] if col_idx < len(row) else ""
                if cell_val is None:
                    clean_cell = "N/A"
                else:
                    # Remove internal newlines and escape pipe characters
                    clean_cell = str(cell_val).replace("\n", " ").replace("|", "\\|").strip()
                    if not clean_cell:
                        clean_cell = "N/A"
                clean_row.append(clean_cell)
            sanitized_rows.append(clean_row)

        if not sanitized_rows:
            return ""

        # Header row & separator
        header = sanitized_rows[0]
        markdown_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]

        # Data rows
        for row in sanitized_rows[1:]:
            markdown_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(markdown_lines)

    def parse_pdf(self, file_path: Path) -> List[ExtractedBlock]:
        """Parses PDF pages, extracting Markdown tables and surrounding prose text."""
        if not file_path.exists():
            raise DocumentParsingError(f"File not found: {file_path}")

        extracted_blocks: List[ExtractedBlock] = []
        doc_name = file_path.name

        try:
            with pdfplumber.open(file_path) as pdf:
                logger.info(f"Parsing document '{doc_name}' ({len(pdf.pages)} pages)")

                for page_idx, page in enumerate(pdf.pages, start=1):
                    # 1. Identify and extract tables
                    table_bboxes = []
                    found_tables = page.find_tables()
                    current_section_hint = None

                    for table_idx, table_obj in enumerate(found_tables, start=1):
                        table_bboxes.append(table_obj.bbox)
                        table_id = f"table_p{page_idx}_{table_idx}"
                        raw_data = table_obj.extract()

                        if raw_data:
                            md_table = self.table_to_markdown(raw_data)
                            if md_table.strip():
                                extracted_blocks.append(
                                    ExtractedBlock(
                                        content=md_table,
                                        content_type=ContentType.TABLE,
                                        page_number=page_idx,
                                        table_id=table_id,
                                        section_hint=current_section_hint,
                                    )
                                )

                    # 2. Extract prose text outside table bounding boxes
                    filtered_page = page
                    for bbox in table_bboxes:
                        try:
                            # Crop out table areas from page view
                            filtered_page = filtered_page.filter(
                                lambda obj, b=bbox: not (
                                    obj["x0"] >= b[0] and obj["x1"] <= b[2] and
                                    obj["top"] >= b[1] and obj["bottom"] <= b[3]
                                )
                            )
                        except Exception:
                            # Fallback if cropping fails
                            pass

                    prose_text = filtered_page.extract_text() or ""
                    clean_prose = prose_text.strip()

                    # Infer section hint from first line if it looks like a header
                    lines = [ln.strip() for ln in clean_prose.split("\n") if ln.strip()]
                    if lines and len(lines[0]) < 80:
                        current_section_hint = lines[0]

                    if clean_prose:
                        extracted_blocks.append(
                            ExtractedBlock(
                                content=clean_prose,
                                content_type=ContentType.TEXT,
                                page_number=page_idx,
                                table_id=None,
                                section_hint=current_section_hint,
                            )
                        )

            logger.info(f"Extracted {len(extracted_blocks)} total blocks from '{doc_name}'")
            return extracted_blocks

        except Exception as exc:
            logger.error(f"Failed to parse PDF '{file_path}': {exc}", exc_info=True)
            raise DocumentParsingError(f"Failed to parse PDF '{doc_name}': {str(exc)}") from exc
