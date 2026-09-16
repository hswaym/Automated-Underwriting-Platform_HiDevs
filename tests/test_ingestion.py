"""Tests for ingestion.py — run with: python tests/test_ingestion.py"""

import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Minimal PDF bytes (hand-crafted valid PDF with one text page)
_MINIMAL_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>
stream
BT /F1 12 Tf 100 700 Td (Hello underwriter) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
441
%%EOF"""


def test_ingest_pdf():
    from ingestion import ingest

    result = ingest(
        pdf_files=[("report.pdf", _MINIMAL_PDF)],
        image_files=[],
        json_data=None,
    )
    assert result.submission_id
    assert len(result.images) == 0
    # pypdf may or may not extract the text from this minimal PDF
    # — just assert we got a ProcessedSubmission back without crashing
    print(f"  text_chunks: {len(result.text_chunks)}")


def test_ingest_image():
    from ingestion import ingest
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (400, 300), color=(120, 80, 40))
    img.save(buf, format="JPEG")
    buf.seek(0)

    result = ingest(
        pdf_files=[],
        image_files=[("property.jpg", buf.read())],
        json_data=None,
    )
    assert len(result.images) == 1
    assert result.images[0].media_type == "image/jpeg"
    assert result.images[0].source == "property.jpg"
    assert len(result.images[0].data) > 0
    print(f"  base64 length: {len(result.images[0].data)}")


def test_ingest_json():
    from ingestion import ingest

    result = ingest(
        pdf_files=[],
        image_files=[],
        json_data={"address": "123 Main St", "year_built": 1990},
    )
    assert result.structured_data is not None
    assert result.structured_data.get("address") == "123 Main St"


def test_ingest_large_image_resized():
    from ingestion import ingest
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (3000, 2000)).save(buf, format="JPEG")
    buf.seek(0)

    result = ingest(
        pdf_files=[],
        image_files=[("big.jpg", buf.read())],
        json_data=None,
    )
    assert len(result.images) == 1
    import base64
    raw = base64.b64decode(result.images[0].data)
    reopened = Image.open(io.BytesIO(raw))
    assert max(reopened.size) <= 1568
    print(f"  resized to: {reopened.size}")


if __name__ == "__main__":
    tests = [test_ingest_pdf, test_ingest_image, test_ingest_json, test_ingest_large_image_resized]
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
