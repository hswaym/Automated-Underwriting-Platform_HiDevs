"""Text preprocessing — clean and normalize extracted document text."""

from __future__ import annotations

import re

from schema import TextChunk

# Lines matching these patterns are noise (page numbers, repeated dividers)
_NOISE_RE = re.compile(r"^\s*(page\s+\d{1,4}|-\s*\d{1,4}\s*-|\d{1,4}|[-=_]{3,})\s*$", re.IGNORECASE)

# Section header heuristic: ALL CAPS or ends with ":" and short
_HEADER_RE = re.compile(r"^([A-Z][A-Z\s\d/,&'-]{2,59}|.{1,60}:)\s*$")

# Approximate chars per token (conservative)
_CHARS_PER_TOKEN = 4
_MAX_CHUNK_CHARS = 3000 * _CHARS_PER_TOKEN  # ~12 000


def _clean_lines(text: str) -> tuple[list[str], list[str]]:
    """Return (content_lines, detected_headers) after noise removal."""
    headers: list[str] = []
    content: list[str] = []
    seen: set[str] = set()

    for line in text.splitlines():
        normalized = " ".join(line.split())  # collapse internal whitespace
        if not normalized:
            continue
        if _NOISE_RE.match(normalized):
            continue
        if normalized in seen:  # duplicate header/footer across pages
            continue
        seen.add(normalized)

        if _HEADER_RE.match(normalized) and len(normalized) < 60:
            headers.append(normalized)
        content.append(normalized)

    return content, headers


def _split_at_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split text into chunks no larger than max_chars, breaking at blank lines."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in re.split(r"\n{2,}", text):
        if current_len + len(paragraph) > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def preprocess(chunks: list[TextChunk]) -> list[TextChunk]:
    """Clean and possibly split each TextChunk; return new list."""
    result: list[TextChunk] = []

    for chunk in chunks:
        lines, headers = _clean_lines(chunk.content)
        cleaned = "\n".join(lines)

        for part in _split_at_paragraphs(cleaned, _MAX_CHUNK_CHARS):
            if part.strip():
                result.append(
                    TextChunk(
                        source=chunk.source,
                        section_headers=headers,
                        content=part.strip(),
                    )
                )

    return result
