import logging
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import pymupdf

logger = logging.getLogger(__name__)

MIN_TEXT_WORDS = 3


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving meaningful line boundaries."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines).strip()


def normalize_for_repeat_detection(text: str) -> str:
    """Normalize text so repeated headers/footers can be detected."""
    text = text.lower().strip()

    text = re.sub(r"\bpage\s+\d+\b", "page", text)
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"\s+", " ", text)

    return text


def looks_like_page_number(text: str) -> bool:
    text = text.strip().lower()

    return bool(
        re.fullmatch(r"page\s+\d+", text)
        or re.fullmatch(r"\d+", text)
    )


def looks_like_source_footer(text: str) -> bool:
    lower = text.lower().strip()

    footer_patterns = [
        "sources:",
        "living with diabetes in the uk",
        "patient information report",
    ]

    return any(lower.startswith(pattern) for pattern in footer_patterns)


def looks_like_list(text: str) -> bool:
    """
    Detect bullet or numbered lists.

    Handles PDFs where bullets may appear on separate lines.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return False

    bullet_markers = {
        "•",
        "●",
        "▪",
        "◦",
        "-",
        "–",
        "—",
    }

    bullet_count = 0
    numbered_count = 0

    for line in lines:
        if line in bullet_markers:
            bullet_count += 1
            continue

        if any(line.startswith(marker + " ") for marker in bullet_markers):
            bullet_count += 1
            continue

        if re.match(r"^\d+[\.\)]\s+", line):
            numbered_count += 1

    return bullet_count >= 2 or numbered_count >= 2


def looks_like_stat_block(text: str) -> bool:
    """
    Identify compact statistics / infographic-like blocks.

    Examples:
        ~90%
        £10.7bn
        4.4 million
        10%
    """

    stat_patterns = [
        r"\b\d+(\.\d+)?%",
        r"£\s?\d",
        r"\b\d+(\.\d+)?\s?(million|billion|bn|m)\b",
        r"~\s?\d+%",
    ]

    matches = 0

    for pattern in stat_patterns:
        matches += len(re.findall(pattern, text, flags=re.IGNORECASE))

    return matches >= 2


def is_safety_critical(text: str) -> bool:
    """
    Conservative safety-critical tagging.

    Important:
    This is metadata classification only.
    It should NOT redefine the block's content type.
    """

    lower = text.lower()

    strong_terms = [
        "medical emergency",
        "go to a&e straight away",
        "call 999",
        "seek urgent medical",
        "diabetic ketoacidosis",
        "suspected dka",
        "severe hypoglycaemia",
        "severe hypoglycemia",
        "deep or laboured breathing",
        "confusion",
        "unconscious",
    ]

    return any(term in lower for term in strong_terms)


def get_block_text(block: dict) -> tuple[str, list[float], bool]:
    """Extract text/font metadata from a PyMuPDF text block."""

    lines_out: list[str] = []
    font_sizes: list[float] = []
    bold_detected = False

    for line in block.get("lines", []):
        spans_out: list[str] = []

        for span in line.get("spans", []):
            span_text = span.get("text", "")

            if not span_text.strip():
                continue

            spans_out.append(span_text)

            font_sizes.append(
                float(span.get("size", 0))
            )

            font_name = span.get("font", "").lower()

            if "bold" in font_name:
                bold_detected = True

        if spans_out:
            lines_out.append("".join(spans_out))

    return (
        clean_text("\n".join(lines_out)),
        font_sizes,
        bold_detected,
    )


def is_heading(
    text: str,
    font_size: float,
    body_font_size: float,
    bold: bool,
) -> bool:
    """
    Conservative heading detection.

    Avoid treating short statistics as headings.
    """

    stripped = text.strip()
    word_count = len(stripped.split())

    if not stripped:
        return False

    if word_count > 16:
        return False

    # Prevent stats such as "~90%" becoming headings.
    if looks_like_stat_block(stripped):
        return False

    if re.fullmatch(r"[\d\s%£.,~]+", stripped):
        return False

    # Numbered section:
    # 3 Symptoms and Getting Diagnosed
    if re.match(
        r"^\d+(\.\d+)*[\.\s]+[A-Za-z]",
        stripped,
    ):
        return True

    # Large font.
    if (
        body_font_size > 0
        and font_size >= body_font_size * 1.30
    ):
        return True

    # Bold + slightly larger.
    if (
        bold
        and body_font_size > 0
        and font_size >= body_font_size * 1.08
        and word_count <= 12
    ):
        return True

    return False


def extract_table_text(table: Any) -> str:
    """Extract table while preserving row/column relationships."""

    rows = table.extract()

    if not rows:
        return ""

    output_rows = []

    for row in rows:
        cells = []

        for cell in row:
            if cell is None:
                cells.append("")
            else:
                cells.append(
                    clean_text(str(cell))
                )

        output_rows.append(
            " | ".join(cells)
        )

    return "\n".join(output_rows).strip()


def bbox_overlap_ratio(
    block_bbox: tuple,
    table_bbox: tuple,
) -> float:
    bx0, by0, bx1, by1 = block_bbox
    tx0, ty0, tx1, ty1 = table_bbox

    overlap_x = max(
        0,
        min(bx1, tx1) - max(bx0, tx0)
    )

    overlap_y = max(
        0,
        min(by1, ty1) - max(by0, ty0)
    )

    overlap_area = overlap_x * overlap_y

    block_area = max(
        (bx1 - bx0) * (by1 - by0),
        1,
    )

    return overlap_area / block_area


def collect_repeated_page_text(
    document: pymupdf.Document,
) -> set[str]:
    """
    Detect recurring short blocks across pages.

    Useful for:
        headers
        footers
        report titles
        page markers
    """

    counter: Counter[str] = Counter()

    for page in document:

        page_dict = page.get_text("dict")

        seen_this_page = set()

        for block in page_dict.get("blocks", []):

            if "lines" not in block:
                continue

            text, _, _ = get_block_text(block)

            if not text:
                continue

            # Only short text is considered header/footer candidate.
            if len(text.split()) > 15:
                continue

            normalized = normalize_for_repeat_detection(text)

            if normalized:
                seen_this_page.add(normalized)

        for item in seen_this_page:
            counter[item] += 1

    page_count = len(document)

    threshold = max(
        3,
        round(page_count * 0.35),
    )

    return {
        text
        for text, count in counter.items()
        if count >= threshold
    }


def extract_pdf(file_path: str) -> list[dict]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    try:
        document = pymupdf.open(path)
    except pymupdf.FileDataError as exc:
        raise ValueError(
            f"Unable to open PDF: {file_path}"
        ) from exc

    extracted_blocks: list[dict] = []

    try:

        doc_title = (
            document.metadata.get("title")
            or path.stem
        )

        repeated_text = collect_repeated_page_text(
            document
        )

        for page_number, page in enumerate(
            document,
            start=1,
        ):

            page_dict = page.get_text("dict")

            # --------------------------------------------------
            # Determine body font size
            # --------------------------------------------------

            all_font_sizes: list[float] = []

            for block in page_dict.get("blocks", []):

                if "lines" not in block:
                    continue

                _, font_sizes, _ = get_block_text(block)

                all_font_sizes.extend(
                    size
                    for size in font_sizes
                    if size > 0
                )

            body_font_size = (
                median(all_font_sizes)
                if all_font_sizes
                else 10.0
            )

            # --------------------------------------------------
            # TABLES
            # --------------------------------------------------

            table_bboxes: list[tuple] = []

            try:
                table_finder = page.find_tables()

                for table_index, table in enumerate(
                    table_finder.tables
                ):

                    table_text = extract_table_text(table)

                    if not table_text:
                        continue

                    bbox = tuple(table.bbox)

                    table_bboxes.append(bbox)

                    extracted_blocks.append(
                        {
                            "page": page_number,
                            "source": path.name,
                            "doc_title": doc_title,
                            "block_index": f"table_{table_index}",
                            "content_type": "table",
                            "text": table_text,
                            "bbox": bbox,
                            "safety_critical": is_safety_critical(
                                table_text
                            ),
                        }
                    )

            except Exception as exc:
                logger.warning(
                    "Table extraction failed on page %s: %s",
                    page_number,
                    exc,
                )

            # --------------------------------------------------
            # TEXT BLOCKS
            # --------------------------------------------------

            text_index = 0

            for block in page_dict.get("blocks", []):

                if "lines" not in block:
                    continue

                bbox = tuple(
                    block.get(
                        "bbox",
                        (0, 0, 0, 0),
                    )
                )

                # Skip text already represented by table.
                if any(
                    bbox_overlap_ratio(
                        bbox,
                        table_bbox,
                    )
                    > 0.60
                    for table_bbox in table_bboxes
                ):
                    continue

                text, font_sizes, bold = get_block_text(
                    block
                )

                if not text:
                    continue

                normalized = normalize_for_repeat_detection(
                    text
                )

                # --------------------------------------------------
                # REMOVE HEADERS / FOOTERS
                # --------------------------------------------------

                if normalized in repeated_text:
                    continue

                if looks_like_page_number(text):
                    continue

                if looks_like_source_footer(text):
                    continue

                # Avoid useless tiny fragments.
                if len(text.split()) < MIN_TEXT_WORDS:
                    continue

                max_font_size = (
                    max(font_sizes)
                    if font_sizes
                    else body_font_size
                )

                # --------------------------------------------------
                # CLASSIFICATION
                # --------------------------------------------------

                if looks_like_stat_block(text):
                    content_type = "stat_block"

                elif looks_like_list(text):
                    content_type = "list"

                elif is_heading(
                    text,
                    max_font_size,
                    body_font_size,
                    bold,
                ):
                    content_type = "heading"

                else:
                    content_type = "paragraph"

                extracted_blocks.append(
                    {
                        "page": page_number,
                        "source": path.name,
                        "doc_title": doc_title,
                        "block_index": text_index,
                        "content_type": content_type,
                        "text": text,
                        "bbox": bbox,
                        "safety_critical": is_safety_critical(
                            text
                        ),
                    }
                )

                text_index += 1

        # --------------------------------------------------
        # Reading order
        # --------------------------------------------------

        extracted_blocks.sort(
            key=lambda item: (
                item["page"],
                item["bbox"][1],
                item["bbox"][0],
            )
        )

        return extracted_blocks

    finally:
        document.close()


if __name__ == "__main__":

    pdf_path = (
        "data/Living_with_Diabetes_UK_Report.PDF"
    )

    blocks = extract_pdf(pdf_path)

    print(
        f"\nTotal semantic blocks extracted: "
        f"{len(blocks)}"
    )

    for block in blocks:

        print(
            "\n========================================"
        )

        print("Page:", block["page"])
        print("Type:", block["content_type"])
        print(
            "Safety:",
            block["safety_critical"],
        )

        print("Text:")
        print(block["text"][:600])