import re

from .pdf_extractor import extract_pdf

DEFAULT_MAX_WORDS = 350
DEFAULT_OVERLAP_WORDS = 50

MIN_CHUNK_WORDS = 15


def validate_chunk_parameters(
    max_words: int,
    overlap_words: int,
) -> None:
    if max_words <= 0:
        raise ValueError("max_words must be greater than 0")

    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError(
            "overlap_words must be greater than or equal to 0 "
            "and less than max_words"
        )


def word_count(text: str) -> int:
    return len(text.split())


def clean_section_name(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def split_large_prose(
    text: str,
    max_words: int,
    overlap_words: int,
) -> list[str]:

    validate_chunk_parameters(max_words, overlap_words)

    words = text.split()

    if len(words) <= max_words:
        return [text]

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + max_words,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(
            0,
            end - overlap_words,
        )

    return chunks


def create_chunk(
    *,
    text: str,
    block: dict,
    section: str | None,
    chunk_number: int,
    content_type: str | None = None,
) -> dict:

    return {
        "chunk_id": (
            f"{block['source']}"
            f"_p{block['page']}"
            f"_c{chunk_number}"
        ),
        "text": text,
        "word_count": word_count(text),
        "page": block["page"],
        "source": block["source"],
        "doc_title": block["doc_title"],
        "section": section,
        "content_type": (
            content_type
            or block["content_type"]
        ),
        "safety_critical": block.get(
            "safety_critical",
            False,
        ),
        "chunk_index": chunk_number,
    }


def meaningful_chunk(text: str) -> bool:
    """
    Prevent tiny fragments from becoming embeddings.
    """

    words = text.split()

    if len(words) >= MIN_CHUNK_WORDS:
        return True

    # Allow short but meaningful safety warnings.
    lower = text.lower()

    safety_terms = [
        "emergency",
        "call 999",
        "urgent",
        "dka",
        "ketoacidosis",
    ]

    return any(
        term in lower
        for term in safety_terms
    )


def chunk_blocks(
    blocks: list[dict],
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[dict]:

    validate_chunk_parameters(max_words, overlap_words)

    chunks: list[dict] = []

    current_section: str | None = None

    prose_buffer: list[str] = []

    prose_metadata: dict | None = None

    chunk_number = 0

    # ------------------------------------------------------
    # BUFFER FLUSH
    # ------------------------------------------------------

    def flush_prose():

        nonlocal prose_buffer
        nonlocal prose_metadata
        nonlocal chunk_number

        if (
            not prose_buffer
            or prose_metadata is None
        ):
            return

        combined = "\n\n".join(
            prose_buffer
        ).strip()

        pieces = split_large_prose(
            combined,
            max_words,
            overlap_words,
        )

        for piece in pieces:

            if not meaningful_chunk(piece):
                continue

            chunks.append(
                create_chunk(
                    text=piece,
                    block=prose_metadata,
                    section=current_section,
                    chunk_number=chunk_number,
                    content_type="prose",
                )
            )

            chunk_number += 1

        prose_buffer = []
        prose_metadata = None

    # ------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------

    for block in blocks:

        text = block["text"].strip()

        if not text:
            continue

        block_type = block["content_type"]

        # --------------------------------------------------
        # HEADING
        # --------------------------------------------------

        if block_type == "heading":

            flush_prose()

            current_section = clean_section_name(
                text
            )

            continue

        # --------------------------------------------------
        # TABLE
        # --------------------------------------------------

        if block_type == "table":

            flush_prose()

            if meaningful_chunk(text):

                chunks.append(
                    create_chunk(
                        text=text,
                        block=block,
                        section=current_section,
                        chunk_number=chunk_number,
                        content_type="table",
                    )
                )

                chunk_number += 1

            continue

        # --------------------------------------------------
        # SAFETY WARNING
        # --------------------------------------------------

        if block.get(
            "safety_critical",
            False,
        ) and block_type != "table":

            flush_prose()

            chunks.append(
                create_chunk(
                    text=text,
                    block=block,
                    section=current_section,
                    chunk_number=chunk_number,
                    content_type="safety_warning",
                )
            )

            chunk_number += 1

            continue

        # --------------------------------------------------
        # STAT BLOCK
        # --------------------------------------------------

        if block_type == "stat_block":

            flush_prose()

            if meaningful_chunk(text):

                chunks.append(
                    create_chunk(
                        text=text,
                        block=block,
                        section=current_section,
                        chunk_number=chunk_number,
                        content_type="stat_block",
                    )
                )

                chunk_number += 1

            continue

        # --------------------------------------------------
        # LIST
        # --------------------------------------------------

        if block_type == "list":

            flush_prose()

            if word_count(text) <= max_words:

                if meaningful_chunk(text):

                    chunks.append(
                        create_chunk(
                            text=text,
                            block=block,
                            section=current_section,
                            chunk_number=chunk_number,
                            content_type="list",
                        )
                    )

                    chunk_number += 1

            else:

                pieces = split_large_prose(
                    text,
                    max_words,
                    overlap_words,
                )

                for piece in pieces:

                    if not meaningful_chunk(piece):
                        continue

                    chunks.append(
                        create_chunk(
                            text=piece,
                            block=block,
                            section=current_section,
                            chunk_number=chunk_number,
                            content_type="list",
                        )
                    )

                    chunk_number += 1

            continue

        # --------------------------------------------------
        # PARAGRAPH
        # --------------------------------------------------

        if block_type == "paragraph":

            proposed_text = "\n\n".join(
                prose_buffer + [text]
            )

            if (
                prose_buffer
                and word_count(
                    proposed_text
                ) > max_words
            ):
                flush_prose()

            if prose_metadata is None:
                prose_metadata = block

            prose_buffer.append(text)

    flush_prose()

    return chunks


def chunk_pdf(
    pdf_path: str,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[dict]:

    blocks = extract_pdf(pdf_path)

    return chunk_blocks(
        blocks,
        max_words=max_words,
        overlap_words=overlap_words,
    )


if __name__ == "__main__":

    pdf_path = (
        "data/Living_with_Diabetes_UK_Report.PDF"
    )

    chunks = chunk_pdf(pdf_path)

    print(
        f"\nTotal chunks created: "
        f"{len(chunks)}"
    )

    for chunk in chunks:

        print(
            "\n========================================"
        )

        print(
            "Chunk ID:",
            chunk["chunk_id"],
        )

        print(
            "Page:",
            chunk["page"],
        )

        print(
            "Section:",
            chunk["section"],
        )

        print(
            "Type:",
            chunk["content_type"],
        )

        print(
            "Safety Critical:",
            chunk["safety_critical"],
        )

        print(
            "Word Count:",
            chunk["word_count"],
        )

        print(
            "Source:",
            chunk["source"],
        )

        print()

        print(
            chunk["text"][:800]
        )