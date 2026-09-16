"""Stage 1: Data Ingestion Layer.

Handles ingestion of mixed inputs:
- Inspection reports, appraisal PDFs (native text with automatic OCR fallback)
- Property photos, aerial/drone imagery (Pillow normalization, EXIF extraction)
- Structured metadata payloads
"""

from __future__ import annotations

import base64
import io
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ExifTags
import pypdf

from schema import IngestedDocument, IngestedDocumentPage, IngestedImage, RawSubmissionPayload

logger = logging.getLogger(__name__)

# Minimum characters on a PDF page before treating as a scanned image
OCR_CHAR_THRESHOLD = 50
MAX_IMAGE_DIMENSION = 1280


def parse_pdf_document(raw_bytes: bytes, filename: str) -> IngestedDocument:
    """Ingests a PDF file, extracting text per page and falling back to OCR if scanned."""
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    pages: List[IngestedDocumentPage] = []
    total_chars = 0

    # If plain text file or non-PDF bytes, decode directly as UTF-8
    if filename.lower().endswith((".txt", ".md", ".csv")) or not raw_bytes.startswith(b"%PDF"):
        try:
            text_content = raw_bytes.decode("utf-8", errors="replace").strip()
            if text_content:
                pages.append(
                    IngestedDocumentPage(
                        page_number=1,
                        text=text_content,
                        is_scanned=False,
                        ocr_applied=False,
                    )
                )
                return IngestedDocument(
                    document_id=doc_id,
                    filename=filename,
                    pages=pages,
                    raw_character_count=len(text_content),
                )
        except Exception:
            pass

    try:
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        for page_idx, page in enumerate(reader.pages, start=1):
            extracted_text = page.extract_text() or ""
            is_scanned = len(extracted_text.strip()) < OCR_CHAR_THRESHOLD
            ocr_applied = False

            if is_scanned:
                # Attempt OCR on embedded images
                ocr_text = _attempt_page_ocr(page, page_idx, filename)
                if ocr_text:
                    extracted_text = ocr_text
                    ocr_applied = True

            clean_text = extracted_text.strip()
            total_chars += len(clean_text)
            pages.append(
                IngestedDocumentPage(
                    page_number=page_idx,
                    text=clean_text,
                    is_scanned=is_scanned,
                    ocr_applied=ocr_applied,
                )
            )
    except Exception as exc:
        logger.error("Failed to parse PDF %s: %s", filename, exc)
        pages.append(
            IngestedDocumentPage(
                page_number=1,
                text=f"[Error parsing PDF: {exc}]",
                is_scanned=True,
                ocr_applied=False,
            )
        )

    return IngestedDocument(
        document_id=doc_id,
        filename=filename,
        pages=pages,
        raw_character_count=total_chars,
    )


def _attempt_page_ocr(page: Any, page_idx: int, filename: str) -> Optional[str]:
    """Extract embedded images from a page and run Tesseract OCR."""
    try:
        import pytesseract

        ocr_parts: List[str] = []
        for img_obj in page.images:
            try:
                pil_img = Image.open(io.BytesIO(img_obj.data))
                text = pytesseract.image_to_string(pil_img)
                if text.strip():
                    ocr_parts.append(text.strip())
            except Exception as e:
                logger.debug("OCR extraction failed on image in %s p%d: %s", filename, page_idx, e)

        if ocr_parts:
            return "\n".join(ocr_parts)
    except Exception as exc:
        logger.debug("pytesseract OCR unavailable or failed for %s p%d: %s", filename, page_idx, exc)
    return None


def parse_image_file(raw_bytes: bytes, filename: str) -> Optional[IngestedImage]:
    """Normalizes image to RGB JPEG, resizes if oversized, and extracts EXIF metadata."""
    img_id = f"img_{uuid.uuid4().hex[:8]}"
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img = pil_img.convert("RGB")

        # Extract EXIF tags
        exif_dict: Dict[str, Any] = {}
        try:
            raw_exif = pil_img.getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, (str, int, float)):
                        exif_dict[tag_name] = value
        except Exception:
            pass

        # Resize if dimensions exceed threshold
        orig_width, orig_height = pil_img.size
        if max(orig_width, orig_height) > MAX_IMAGE_DIMENSION:
            pil_img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        encoded_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return IngestedImage(
            image_id=img_id,
            filename=filename,
            image_base64=encoded_b64,
            width=pil_img.width,
            height=pil_img.height,
            format="JPEG",
            exif_metadata=exif_dict,
        )
    except Exception as exc:
        logger.error("Failed to parse image %s: %s", filename, exc)
        return None


def ingest_submission(
    pdf_files: List[Tuple[str, bytes]],
    image_files: List[Tuple[str, bytes]],
    metadata: Optional[Dict[str, Any]] = None,
    submission_id: Optional[str] = None,
) -> RawSubmissionPayload:
    """Entrypoint orchestrating ingestion across documents, photos, and declared data."""
    sub_id = submission_id or f"sub_{uuid.uuid4().hex[:8]}"

    ingested_docs = [parse_pdf_document(b, fn) for fn, b in pdf_files]
    ingested_imgs = [res for fn, b in image_files if (res := parse_image_file(b, fn)) is not None]

    return RawSubmissionPayload(
        submission_id=sub_id,
        documents=ingested_docs,
        images=ingested_imgs,
        declared_metadata=metadata or {},
    )


def ingest(
    pdf_files: List[Tuple[str, bytes]],
    image_files: List[Tuple[str, bytes]],
    json_data: Optional[Dict[str, Any]] = None,
) -> Any:
    """Convenience adapter returning ProcessedSubmission for backward compatibility."""
    from schema import ImageData, ProcessedSubmission, TextChunk
    raw = ingest_submission(pdf_files, image_files, metadata=json_data)
    text_chunks = [
        TextChunk(source=doc.filename, content=p.text)
        for doc in raw.documents
        for p in doc.pages
    ]
    images = [
        ImageData(source=img.filename, media_type="image/jpeg", data=img.image_base64)
        for img in raw.images
    ]
    return ProcessedSubmission(
        submission_id=raw.submission_id,
        text_chunks=text_chunks,
        images=images,
        structured_data=raw.declared_metadata or None,
    )

