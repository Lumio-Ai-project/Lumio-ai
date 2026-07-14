import re
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 80

PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    metadata: dict


def chunk_article(
    text: str,
    metadata: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split article text into semantic chunks by merging paragraphs
    until the size limit is reached, with optional overlap.
    """
    if not text or not text.strip():
        return []

    paragraphs = [
        p.strip()
        for p in PARAGRAPH_SPLIT_RE.split(text.strip())
        if p.strip() and len(p.strip()) >= MIN_CHUNK_SIZE
    ]

    if not paragraphs:
        normalized = text.strip()
        if len(normalized) >= MIN_CHUNK_SIZE:
            return [TextChunk(text=normalized, chunk_index=0, metadata=metadata)]
        return []

    raw_chunks = _merge_paragraphs(paragraphs, chunk_size, chunk_overlap)
    return [
        TextChunk(text=chunk_text, chunk_index=index, metadata=metadata)
        for index, chunk_text in enumerate(raw_chunks)
    ]


def _merge_paragraphs(
    paragraphs: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)

        if paragraph_len > chunk_size:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            chunks.extend(_split_long_paragraph(paragraph, chunk_size, chunk_overlap))
            continue

        projected = current_len + paragraph_len + (2 if current else 0)
        if projected <= chunk_size:
            current.append(paragraph)
            current_len = projected
            continue

        chunks.append("\n\n".join(current))
        overlap_text = _tail_overlap("\n\n".join(current), chunk_overlap)
        current = [overlap_text, paragraph] if overlap_text else [paragraph]
        current_len = sum(len(part) for part in current) + 2 * max(0, len(current) - 1)

    if current:
        chunks.append("\n\n".join(current))

    return [chunk for chunk in chunks if len(chunk.strip()) >= MIN_CHUNK_SIZE]


def _split_long_paragraph(
    paragraph: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    return _merge_paragraphs(sentences, chunk_size, chunk_overlap)


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 and len(text) <= overlap * 2 else ""
    return text[-overlap:].lstrip()
