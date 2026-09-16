"""Data contracts and schemas for the Automated Underwriting Platform."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Ingestion Models
# ---------------------------------------------------------------------------

class IngestedDocumentPage(BaseModel):
    page_number: int
    text: str
    is_scanned: bool = False
    ocr_applied: bool = False


class IngestedDocument(BaseModel):
    document_id: str
    filename: str
    pages: List[IngestedDocumentPage] = Field(default_factory=list)
    raw_character_count: int = 0


class IngestedImage(BaseModel):
    image_id: str
    filename: str
    image_base64: str
    width: int
    height: int
    format: str = "JPEG"
    exif_metadata: Dict[str, Any] = Field(default_factory=dict)


class RawSubmissionPayload(BaseModel):
    submission_id: str
    documents: List[IngestedDocument] = Field(default_factory=list)
    images: List[IngestedImage] = Field(default_factory=list)
    declared_metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. Document Analysis & Extraction Models
# ---------------------------------------------------------------------------

class ExtractedField(BaseModel):
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    source_document: Optional[str] = None
    source_page: Optional[int] = None
    extraction_method: str = "heuristic_regex"


class DocumentExtractionResult(BaseModel):
    property_address: Optional[ExtractedField] = None
    year_built: Optional[ExtractedField] = None
    roof_installed_year: Optional[ExtractedField] = None
    roof_type: Optional[ExtractedField] = None
    square_footage: Optional[ExtractedField] = None
    construction_type: Optional[ExtractedField] = None
    occupancy_type: Optional[ExtractedField] = None
    prior_claims_count: Optional[ExtractedField] = None
    prior_claims_loss_total: Optional[ExtractedField] = None
    flood_zone: Optional[ExtractedField] = None
    electrical_system: Optional[ExtractedField] = None
    plumbing_system: Optional[ExtractedField] = None
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    hitl_review_required: bool = False
    hitl_review_reasons: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 3. Computer Vision Models
# ---------------------------------------------------------------------------

class DetectedHazard(BaseModel):
    hazard_type: str  # e.g., "roof_damage_shingle_loss", "swimming_pool_unfenced", "overgrown_vegetation"
    category: Literal["roof", "structural", "liability", "environmental", "maintenance"]
    severity: Literal["low", "medium", "high", "critical"]
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax]
    source_image: str
    description: str


class VisionAnalysisResult(BaseModel):
    hazards: List[DetectedHazard] = Field(default_factory=list)
    roof_condition_score: float = Field(default=85.0, ge=0.0, le=100.0, description="100 = pristine, 0 = failed")
    brush_clearance_feet: Optional[float] = None
    has_swimming_pool: bool = False
    has_trampoline: bool = False
    has_boarded_windows: bool = False


# ---------------------------------------------------------------------------
# 4. Multimodal Fusion Models
# ---------------------------------------------------------------------------

class ContradictionFlag(BaseModel):
    flag_id: str
    field_name: str
    document_claim: str
    visual_evidence: str
    discrepancy_severity: Literal["minor", "major", "critical"]
    explanation: str


class UnifiedRiskProfile(BaseModel):
    submission_id: str
    year_built: int
    effective_building_age: int
    roof_installed_year: int
    roof_age_years: int
    roof_condition_visual: str
    square_footage: float
    construction_type: str
    flood_zone: str
    prior_claims_count: int
    prior_claims_amount: float
    visual_hazards: List[DetectedHazard] = Field(default_factory=list)
    contradictions: List[ContradictionFlag] = Field(default_factory=list)
    hitl_required: bool = False
    hitl_reasons: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 5. Risk Assessment Models
# ---------------------------------------------------------------------------

class FactorContribution(BaseModel):
    factor_name: str
    category: str
    points: int
    description: str


class RiskScoreBreakdown(BaseModel):
    risk_score: int = Field(ge=0, le=100, description="0 = minimal risk, 100 = maximum risk")
    risk_tier: Literal["Preferred", "Standard", "Non-Standard", "Ineligible"]
    factor_contributions: List[FactorContribution] = Field(default_factory=list)
    structural_score: int
    environmental_score: int
    maintenance_score: int
    claims_score: int
    liability_score: int
    contradiction_penalty: int


# ---------------------------------------------------------------------------
# 6. Guideline Compliance Models
# ---------------------------------------------------------------------------

class GuidelineRuleViolation(BaseModel):
    rule_id: str
    citation: str
    rule_category: str
    action: Literal["DECLINE", "REFER", "SURCHARGE"]
    reason: str


class ComplianceCheckResult(BaseModel):
    passed: bool
    hard_stop_triggered: bool
    violations: List[GuidelineRuleViolation] = Field(default_factory=list)
    rule_citations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 7. Final Underwriting Decision Models
# ---------------------------------------------------------------------------

class HazardFlag(BaseModel):
    name: str
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    source: str = "document"


class SectionReasoning(BaseModel):
    section: str
    finding: str


class PropertyData(BaseModel):
    address: Optional[str] = None
    year_built: Optional[int] = None
    construction_type: Optional[str] = None
    square_footage: Optional[float] = None
    occupancy_type: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    def to_text(self) -> str:
        parts = []
        if self.address:
            parts.append(f"Address: {self.address}")
        if self.year_built:
            parts.append(f"Year Built: {self.year_built}")
        if self.construction_type:
            parts.append(f"Construction Type: {self.construction_type}")
        if self.square_footage:
            parts.append(f"Square Footage: {self.square_footage}")
        if self.occupancy_type:
            parts.append(f"Occupancy: {self.occupancy_type}")
        for k, v in self.extra.items():
            parts.append(f"{k}: {v}")
        return "\n".join(parts)


class TextChunk(BaseModel):
    source: str
    section_headers: List[str] = Field(default_factory=list)
    content: str


class ImageData(BaseModel):
    source: str
    media_type: str = "image/jpeg"
    data: str


class ProcessedSubmission(BaseModel):
    submission_id: str
    text_chunks: List[TextChunk] = Field(default_factory=list)
    images: List[ImageData] = Field(default_factory=list)
    structured_data: Optional[Dict[str, Any]] = None


class UnderwritingDecision(BaseModel):
    submission_id: str
    decision: str
    confidence: Literal["low", "medium", "high"] = "high"
    risk_score: int = Field(ge=0, le=100)
    risk_tier: str = "Standard"
    summary_rationale: str = ""
    risk_breakdown: Optional[RiskScoreBreakdown] = None
    compliance_result: Optional[ComplianceCheckResult] = None
    contradictions: List[ContradictionFlag] = Field(default_factory=list)
    detected_hazards: List[DetectedHazard] = Field(default_factory=list)
    required_underwriter_actions: List[str] = Field(default_factory=list)
    can_underwriter_override: bool = True
    flags: List[HazardFlag] = Field(default_factory=list)
    section_reasoning: List[SectionReasoning] = Field(default_factory=list)
    summary: Optional[str] = None
    guideline_references: List[str] = Field(default_factory=list)

    @classmethod
    def json_schema_for_claude(cls) -> Dict[str, Any]:
        return cls.model_json_schema()

