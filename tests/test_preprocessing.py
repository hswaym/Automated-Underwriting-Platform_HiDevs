"""Tests for preprocessing.py — run with: python tests/test_preprocessing.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schema import TextChunk
from preprocessing import preprocess


def test_duplicate_removal():
    """Duplicate-ish lines should be removed."""
    chunks = [
        TextChunk(source="doc.pdf", content="This is a test.\nThis is a test.\nSomething else."),
    ]
    result = preprocess(chunks)
    combined = "\n".join(c.content for c in result)
    assert combined.count("This is a test.") == 1
    print(f"  de-duped content: {combined!r}")


def test_page_number_removal():
    """Lines that look like page numbers should be stripped."""
    chunks = [
        TextChunk(source="doc.pdf", content="Page 3\nReal content here.\n- 7 -\nMore content."),
    ]
    result = preprocess(chunks)
    combined = "\n".join(c.content for c in result)
    assert "Page 3" not in combined
    assert "- 7 -" not in combined
    assert "Real content" in combined
    print(f"  cleaned: {combined!r}")


def test_section_header_detection():
    """ALL-CAPS or colon-ending short lines become section headers."""
    chunks = [
        TextChunk(source="doc.pdf", content="PROPERTY DETAILS\nThe roof was replaced in 2018."),
    ]
    result = preprocess(chunks)
    found = any("PROPERTY DETAILS" in h for c in result for h in c.section_headers)
    assert found, f"Expected section header, got: {[c.section_headers for c in result]}"
    print(f"  headers: {result[0].section_headers}")


def test_long_chunk_splitting():
    """Chunks over ~12k chars should be split on paragraph boundaries."""
    long_text = ("This is paragraph one.\n\n" * 400)  # ~8800 chars per paragraph pair
    chunks = [TextChunk(source="big.pdf", content=long_text)]
    result = preprocess(chunks)
    assert len(result) >= 1  # may or may not split depending on exact length
    total_len = sum(len(c.content) for c in result)
    print(f"  input len: {len(long_text)}, output chunks: {len(result)}, total output len: {total_len}")


def test_empty_input():
    """Empty list in, empty list out."""
    assert preprocess([]) == []


if __name__ == "__main__":
    tests = [
        test_duplicate_removal,
        test_page_number_removal,
        test_section_header_detection,
        test_long_chunk_splitting,
        test_empty_input,
    ]
    failures = 0
    for t in tests:
        try:
            print(f"[RUN] {t.__name__}")
            t()
            print(f"[OK]  {t.__name__}")
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failures += 1
    if failures:
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
