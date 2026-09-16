"""Guidelines loader — read once at startup, cache as string."""

from __future__ import annotations

import logging
from pathlib import Path

import pypdf

from config import settings

logger = logging.getLogger(__name__)

_cache: str | None = None


def load() -> str:
    """Return the guideline document as a plain string. Cached after first call."""
    global _cache
    if _cache is not None:
        return _cache

    path: Path = settings.guidelines_path
    if not path.exists():
        logger.warning(
            "Guidelines file not found at %s — using empty guidelines. "
            "Set GUIDELINES_PATH in .env to point to your guideline document.",
            path,
        )
        _cache = "(No underwriting guidelines provided.)"
        return _cache

    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        _cache = path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        _cache = "\n\n".join(pages)
    else:
        raise ValueError(
            f"Unsupported guidelines format: {suffix}. Use .txt, .md, or .pdf"
        )

    logger.info("Loaded guidelines from %s (%d chars)", path, len(_cache))
    return _cache
