"""Stage 2: Document Analysis & Structuring Layer.

Extracts structured risk parameters from unstructured inspection narratives,
appraisal forms, and loss run documents. Computes field-level confidence scores
and triggers Human-In-The-Loop (HITL) review when confidence is low or mandatory
fields are missing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from schema import DocumentExtractionResult, ExtractedField, RawSubmissionPayload


class DocumentAnalysisEngine:
    """Extracts property attributes using regex/pattern parsers and vendor layout normalizers."""

    def __init__(self, hitl_confidence_threshold: float = 0.75):
        self.hitl_confidence_threshold = hitl_confidence_threshold

    def analyze_submission_documents(self, payload: RawSubmissionPayload) -> DocumentExtractionResult:
        # Combine all document texts into a unified corpus
        full_text_corpus: List[Dict[str, Any]] = []
        for doc in payload.documents:
            for page in doc.pages:
                full_text_corpus.append({
                    "doc_name": doc.filename,
                    "page_num": page.page_number,
                    "text": page.text,
                })

        combined_text = "\n\n".join([item["text"] for item in full_text_corpus])
        # Also check declared metadata for fallback
        metadata = payload.declared_metadata

        # Perform extraction
        year_built = self._extract_year_built(combined_text, metadata, full_text_corpus)
        roof_installed_year = self._extract_roof_year(combined_text, metadata, full_text_corpus)
        roof_type = self._extract_roof_type(combined_text, metadata, full_text_corpus)
        square_footage = self._extract_square_footage(combined_text, metadata, full_text_corpus)
        construction_type = self._extract_construction_type(combined_text, metadata, full_text_corpus)
        occupancy_type = self._extract_occupancy_type(combined_text, metadata, full_text_corpus)
        prior_claims_count, prior_claims_amount = self._extract_claims_history(combined_text, metadata, full_text_corpus)
        flood_zone = self._extract_flood_zone(combined_text, metadata, full_text_corpus)
        electrical_system = self._extract_electrical(combined_text, metadata, full_text_corpus)
        plumbing_system = self._extract_plumbing(combined_text, metadata, full_text_corpus)
        property_address = self._extract_address(combined_text, metadata, full_text_corpus)

        # Calculate overall confidence
        extracted_fields = [
            year_built, roof_installed_year, roof_type, square_footage,
            construction_type, prior_claims_count, flood_zone
        ]
        confidences = [f.confidence for f in extracted_fields if f is not None]
        overall_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Evaluate HITL requirements
        hitl_reasons: List[str] = []
        if overall_conf < self.hitl_confidence_threshold:
            hitl_reasons.append(f"Overall document extraction confidence ({overall_conf:.2f}) is below threshold ({self.hitl_confidence_threshold:.2f})")

        if year_built is None:
            hitl_reasons.append("Mandatory field 'year_built' could not be identified in submitted documentation.")
        elif year_built.confidence < self.hitl_confidence_threshold:
            hitl_reasons.append(f"Low confidence ({year_built.confidence:.2f}) on year built.")

        if roof_installed_year is None:
            hitl_reasons.append("Mandatory field 'roof_installed_year' could not be resolved from inspection report.")

        if electrical_system and "knob_and_tube" in str(electrical_system.value).lower():
            hitl_reasons.append("Potential hazardous electrical system (Knob & Tube) referenced in notes.")

        hitl_required = len(hitl_reasons) > 0

        return DocumentExtractionResult(
            property_address=property_address,
            year_built=year_built,
            roof_installed_year=roof_installed_year,
            roof_type=roof_type,
            square_footage=square_footage,
            construction_type=construction_type,
            occupancy_type=occupancy_type,
            prior_claims_count=prior_claims_count,
            prior_claims_loss_total=prior_claims_amount,
            flood_zone=flood_zone,
            electrical_system=electrical_system,
            plumbing_system=plumbing_system,
            overall_confidence=round(overall_conf, 2),
            hitl_review_required=hitl_required,
            hitl_review_reasons=hitl_reasons,
        )

    # -----------------------------------------------------------------------
    # Heuristic & Regex Parsers with Source Attribution
    # -----------------------------------------------------------------------

    def _extract_year_built(self, text: str, metadata: dict, corpus: list) -> Optional[ExtractedField]:
        if "year_built" in metadata:
            return ExtractedField(value=int(metadata["year_built"]), confidence=0.95, extraction_method="metadata")

        match = re.search(r"(?:year built|constructed in|construction year|built:?)\s*[:\-]?\s*(\b1[89]\d\d|20[0-2]\d\b)", text, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            doc_name, page_num = self._find_source(match.group(0), corpus)
            return ExtractedField(value=year, confidence=0.90, source_document=doc_name, source_page=page_num)
        return None

    def _extract_roof_year(self, text: str, metadata: dict, corpus: list) -> Optional[ExtractedField]:
        if "roof_year" in metadata:
            return ExtractedField(value=int(metadata["roof_year"]), confidence=0.95, extraction_method="metadata")

        # Pattern: roof replaced / installed / new roof in YYYY or Roof Age: X years
        match_yr = re.search(r"(?:roof replaced|roof installed|new roof|roof updated|roof age:?\s*(\d+)\s*years?)\s*(?:in|dated)?\s*[:\-]?\s*(\b1[89]\d\d|20[0-2]\d\b)?", text, re.IGNORECASE)
        if match_yr:
            if match_yr.group(2):
                year = int(match_yr.group(2))
            elif match_yr.group(1):
                age = int(match_yr.group(1))
                year = 2026 - age
            else:
                year = None

            if year:
                doc_name, page_num = self._find_source(match_yr.group(0), corpus)
                return ExtractedField(value=year, confidence=0.88, source_document=doc_name, source_page=page_num)

        # Check for simple roof year mentions
        match_simple = re.search(r"roof(?:\s+\w+){0,3}\s+(?:19\d\d|20[0-2]\d)", text, re.IGNORECASE)
        if match_simple:
            yr_str = re.search(r"(19\d\d|20[0-2]\d)", match_simple.group(0))
            if yr_str:
                doc_name, page_num = self._find_source(match_simple.group(0), corpus)
                return ExtractedField(value=int(yr_str.group(1)), confidence=0.75, source_document=doc_name, source_page=page_num)

        return None

    def _extract_roof_type(self, text: str, metadata: dict, corpus: list) -> Optional[ExtractedField]:
        types = ["asphalt shingle", "architectural shingle", "metal", "slate", "tile", "flat membrane", "wood shake"]
        for t in types:
            if re.search(rf"\b{t}\b", text, re.IGNORECASE):
                doc_name, page_num = self._find_source(t, corpus)
                return ExtractedField(value=t.title(), confidence=0.85, source_document=doc_name, source_page=page_num)
        if "roof_type" in metadata:
            return ExtractedField(value=str(metadata["roof_type"]), confidence=0.90, extraction_method="metadata")
        return None

    def _extract_square_footage(self, text: str, metadata: dict, corpus: list) -> Optional[ExtractedField]:
        if "square_footage" in metadata:
            return ExtractedField(value=float(metadata["square_footage"]), confidence=0.95, extraction_method="metadata")

        match = re.search(r"(\b\d{1,2}[,\.]?\d{3}\b)\s*(?:sq(?:uare)?\s*\.?\s*f(?:ee)?t|\.sq\.?ft|\.gla)", text, re.IGNORECASE)
        if match:
            val_str = match.group(1).replace(",", "")
            doc_name, page_num = self._find_source(match.group(0), corpus)
            return ExtractedField(value=float(val_str), confidence=0.88, source_document=doc_name, source_page=page_num)
        return None

    def _extract_construction_type(self, text: str, metadata: dict, corpus: list) -> Optional[ExtractedField]:
        c_types = ["frame", "joisted masonry", "masonry non-combustible", "brick veneer", "stucco", "log"]
        for c in c_types:
            if re.search(rf"\b{c}\b", text, re.IGNORECASE):
                doc_name, page_num = self._find_source(c, corpus)
                return ExtractedField(value=c.title(), confidence=0.85, source_document=doc_name, source_page=page_num)
        if "construction_type" in metadata:
            return ExtractedField(value=str(metadata["construction_type"]), confidence=0.90, extraction_method="metadata")
        return ExtractedField(value="Frame", confidence=0.60, extraction_method="default_assumption")

    def _extract_occupancy_type(self, text: str, metadata: dict, corpus: list) -> Optional[ExtractedField]:
        occupancies = ["single family", "multi-family", "tenant occupied", "owner occupied", "vacant", "seasonal"]
        for occ in occupancies:
            if re.search(rf"\b{occ}\b", text, re.IGNORECASE):
                doc_name, page_num = self._find_source(occ, corpus)
                return ExtractedField(value=occ.title(), confidence=0.85, source_document=doc_name, source_page=page_num)
        return ExtractedField(value="Owner Occupied", confidence=0.70, extraction_method="default_assumption")

    def _extract_claims_history(self, text: str, metadata: dict, corpus: list) -> Tuple[Optional[ExtractedField], Optional[ExtractedField]]:
        claims_match = re.search(r"(?:prior claims?|loss history|claims in past 5 years?)\s*[:\-]?\s*(\d+|none|zero)", text, re.IGNORECASE)
        count = 0
        conf = 0.85
        if claims_match:
            raw_val = claims_match.group(1).lower()
            count = 0 if raw_val in ["none", "zero"] else int(raw_val)
            doc_name, page_num = self._find_source(claims_match.group(0), corpus)
            field_count = ExtractedField(value=count, confidence=conf, source_document=doc_name, source_page=page_num)
        elif "prior_claims" in metadata:
            field_count = ExtractedField(value=int(metadata["prior_claims"]), confidence=0.95, extraction_method="metadata")
        else:
            field_count = ExtractedField(value=0, confidence=0.65, extraction_method="inferred_zero")

        # Check total loss amount
        loss_match = re.search(r"(?:total paid|loss amount|claim payout)\s*[:\-]?\s*\$?([\d,]+)", text, re.IGNORECASE)
        amount = 0.0
        if loss_match:
            amount = float(loss_match.group(1).replace(",", ""))
            field_amount = ExtractedField(value=amount, confidence=0.85)
        else:
            field_amount = ExtractedField(value=0.0, confidence=0.70)

        return field_count, field_amount

    def _extract_flood_zone(self, text: str, metadata: dict, corpus: list) -> Optional[ExtractedField]:
        if "flood_zone" in metadata:
            return ExtractedField(value=str(metadata["flood_zone"]).upper(), confidence=0.95, extraction_method="metadata")

        match = re.search(r"flood zone\s*[:\-]?\s*(?:zone\s+)?([A-Z0-9]+)", text, re.IGNORECASE)
        if match:
            doc_name, page_num = self._find_source(match.group(0), corpus)
            return ExtractedField(value=match.group(1).upper(), confidence=0.88, source_document=doc_name, source_page=page_num)
        return ExtractedField(value="X", confidence=0.65, extraction_method="default_minimal_flood_zone")

    def _extract_electrical(self, text: str, metadata: dict, corpus: list) -> Optional[ExtractedField]:
        if re.search(r"knob\s*(?:and|&)\s*tube", text, re.IGNORECASE):
            doc_name, page_num = self._find_source("knob and tube", corpus)
            return ExtractedField(value="Knob and Tube", confidence=0.90, source_document=doc_name, source_page=page_num)
        if re.search(r"circuit breakers?|modern copper", text, re.IGNORECASE):
            return ExtractedField(value="Circuit Breakers", confidence=0.85)
        return None

    def _extract_plumbing(self, text: str, metadata: dict, corpus: list) -> Optional[ExtractedField]:
        if re.search(r"polybutylene|galvanized", text, re.IGNORECASE):
            return ExtractedField(value="Polybutylene / Galvanized", confidence=0.85)
        if re.search(r"copper|ptex|pvc", text, re.IGNORECASE):
            return ExtractedField(value="Modern Copper/PEX", confidence=0.85)
        return None

    def _extract_address(self, text: str, metadata: dict, corpus: list) -> Optional[ExtractedField]:
        if "address" in metadata:
            return ExtractedField(value=str(metadata["address"]), confidence=0.95, extraction_method="metadata")
        addr_match = re.search(r"(?:property address|location|insured property)\s*[:\-]?\s*([^\n\r]+)", text, re.IGNORECASE)
        if addr_match:
            return ExtractedField(value=addr_match.group(1).strip(), confidence=0.80)
        return None

    def _find_source(self, needle: str, corpus: list) -> Tuple[Optional[str], Optional[int]]:
        needle_clean = needle.lower().strip()
        for item in corpus:
            if needle_clean in item["text"].lower():
                return item["doc_name"], item["page_num"]
        return None, None
