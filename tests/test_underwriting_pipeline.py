"""Stage 8: Testing & Validation Test Suite.

Evaluates each of the four core underwriting criteria independently:
1. Document Analysis: Field-level extraction precision, confidence, and HITL flags.
2. Computer Vision / Hazard Detection: Hazard recognition, roof wear, defensible space.
3. Multimodal Processing: Text/image agreement vs contradiction detection.
4. Guideline Compliance & Risk Assessment: Hard stops, referrals, and score fidelity.
"""

from __future__ import annotations

import io
import pytest
from PIL import Image

from doc_analysis import DocumentAnalysisEngine
from decision_engine import UnderwritingDecisionEngine
from guidelines_engine import UnderwritingGuidelinesEngine
from multimodal_fusion import MultimodalFusionEngine
from pipeline import AutomatedUnderwritingPipeline
from risk_engine import ActuarialRiskEngine
from schema import (
    IngestedDocument,
    IngestedDocumentPage,
    IngestedImage,
    RawSubmissionPayload,
)
from vision_pipeline import PropertyVisionAnalyzer


# ---------------------------------------------------------------------------
# Helper Fixtures
# ---------------------------------------------------------------------------

def create_mock_image(color=(120, 120, 120), size=(640, 480)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Criterion 1: Document Analysis & Structuring Tests
# ---------------------------------------------------------------------------

class TestDocumentAnalysis:
    """Evaluates field-level extraction precision and confidence scoring."""

    def test_structured_report_extraction(self):
        engine = DocumentAnalysisEngine()
        sample_text = """
        INSPECTION REPORT
        Property Address: 742 Evergreen Terrace, Springfield, IL
        Year Built: 1998
        Construction Type: Joisted Masonry
        Gross Living Area: 2,450 sq ft
        Roof System: Architectural Shingle
        Roof Replaced: 2018
        Prior Claims in Past 5 Years: 0
        Flood Zone: Zone X
        Electrical: 200 Amp Circuit Breakers
        """
        doc = IngestedDocument(
            document_id="doc_test_1",
            filename="inspection_report.pdf",
            pages=[IngestedDocumentPage(page_number=1, text=sample_text)],
            raw_character_count=len(sample_text),
        )
        payload = RawSubmissionPayload(submission_id="sub_test_1", documents=[doc])

        result = engine.analyze_submission_documents(payload)

        # Field-level verification
        assert result.year_built is not None and result.year_built.value == 1998
        assert result.roof_installed_year is not None and result.roof_installed_year.value == 2018
        assert result.square_footage is not None and result.square_footage.value == 2450.0
        assert result.construction_type is not None and "Masonry" in result.construction_type.value
        assert result.prior_claims_count is not None and result.prior_claims_count.value == 0
        assert result.flood_zone is not None and result.flood_zone.value == "X"

        # Confidence & HITL verification
        assert result.overall_confidence >= 0.75
        assert result.hitl_review_required is False

    def test_incomplete_document_triggers_hitl(self):
        engine = DocumentAnalysisEngine(hitl_confidence_threshold=0.80)
        incomplete_text = "Brief summary note. No year built or roof information included."
        doc = IngestedDocument(
            document_id="doc_test_2",
            filename="sparse_note.pdf",
            pages=[IngestedDocumentPage(page_number=1, text=incomplete_text)],
            raw_character_count=len(incomplete_text),
        )
        payload = RawSubmissionPayload(submission_id="sub_test_2", documents=[doc])

        result = engine.analyze_submission_documents(payload)

        # Missing critical fields must force Human-In-The-Loop review
        assert result.hitl_review_required is True
        assert any("year_built" in r for r in result.hitl_review_reasons)


# ---------------------------------------------------------------------------
# Criterion 2: Computer Vision / Risk Hazard Detection Tests
# ---------------------------------------------------------------------------

class TestComputerVisionHazards:
    """Evaluates visual hazard detection, severity mapping, and roof scoring."""

    def test_roof_damage_detection(self):
        vision_engine = PropertyVisionAnalyzer()
        # Mock high-variance roof photo
        roof_bytes = create_mock_image(color=(50, 50, 50))
        from ingestion import parse_image_file

        ingested = parse_image_file(roof_bytes, "drone_aerial_roof_damage.jpg")
        assert ingested is not None

        result = vision_engine.analyze_images([ingested])
        assert result.roof_condition_score <= 50.0
        assert any(h.category == "roof" for h in result.hazards)

    def test_pool_hazard_detection(self):
        vision_engine = PropertyVisionAnalyzer()
        # Mock image with high blue ratio (pool)
        pool_bytes = create_mock_image(color=(30, 100, 240))
        from ingestion import parse_image_file

        ingested = parse_image_file(pool_bytes, "backyard_pool_exterior.jpg")
        assert ingested is not None

        result = vision_engine.analyze_images([ingested])
        assert result.has_swimming_pool is True
        assert any(h.category == "liability" for h in result.hazards)


# ---------------------------------------------------------------------------
# Criterion 3: Multimodal Fusion & Contradiction Reasoning Tests
# ---------------------------------------------------------------------------

class TestMultimodalFusion:
    """Evaluates cross-modal verification between document declarations and photos."""

    def test_detects_claimed_new_roof_with_visible_damage(self):
        pipeline = AutomatedUnderwritingPipeline()

        # Text claims roof replaced in 2024 (2 years old)
        pdf_text = "Year Built: 2005. Roof replaced in 2024. Prior claims: 0. Flood Zone: X."
        # Image clearly shows damaged roof
        damaged_roof_bytes = create_mock_image(color=(40, 40, 40))

        # We can simulate PDF bytes by passing raw text via ingestion
        from ingestion import IngestedDocument, IngestedDocumentPage, parse_image_file
        doc = IngestedDocument(
            document_id="d1",
            filename="report.pdf",
            pages=[IngestedDocumentPage(page_number=1, text=pdf_text)],
            raw_character_count=len(pdf_text),
        )
        img = parse_image_file(damaged_roof_bytes, "roof_damage_shingle_loss.jpg")

        payload = RawSubmissionPayload(
            submission_id="sub_contra_1",
            documents=[doc],
            images=[img],
        )

        doc_res = pipeline.doc_engine.analyze_submission_documents(payload)
        vis_res = pipeline.vision_engine.analyze_images(payload.images)
        fused = pipeline.fusion_engine.fuse("sub_contra_1", doc_res, vis_res)

        # Must flag critical contradiction
        assert len(fused.contradictions) >= 1
        roof_flag = [c for c in fused.contradictions if c.field_name == "roof_condition_and_age"]
        assert len(roof_flag) == 1
        assert roof_flag[0].discrepancy_severity == "critical"


# ---------------------------------------------------------------------------
# Criterion 4: Guideline Compliance & Auditable Scoring Tests
# ---------------------------------------------------------------------------

class TestGuidelineComplianceAndScoring:
    """Evaluates deterministic compliance checks, hard stops, and explainability."""

    def test_hard_decline_on_roof_exceeding_twenty_years(self):
        pipeline = AutomatedUnderwritingPipeline()
        # Roof installed in 2000 (26 years old)
        pdf_text = "Year Built: 1990. Roof replaced: 2000. Prior claims: 0. Flood Zone: X."

        from ingestion import IngestedDocument, IngestedDocumentPage
        doc = IngestedDocument(
            document_id="d2",
            filename="report.pdf",
            pages=[IngestedDocumentPage(page_number=1, text=pdf_text)],
            raw_character_count=len(pdf_text),
        )
        payload = RawSubmissionPayload(submission_id="sub_roof_age", documents=[doc])

        doc_res = pipeline.doc_engine.analyze_submission_documents(payload)
        vis_res = pipeline.vision_engine.analyze_images([])
        fused = pipeline.fusion_engine.fuse("sub_roof_age", doc_res, vis_res)
        risk = pipeline.risk_engine.calculate_risk(fused)
        comp = pipeline.guidelines_engine.evaluate_compliance(fused)
        decision = pipeline.decision_engine.make_decision(fused, risk, comp)

        # Guideline RULE-ROOF-202 must trigger DECLINE
        assert comp.hard_stop_triggered is True
        assert any(v.rule_id == "RULE-ROOF-202" and v.action == "DECLINE" for v in comp.violations)
        assert decision.decision == "DECLINE"
        assert any("HO-UW-202" in c for c in decision.compliance_result.rule_citations)

    def test_referral_on_special_flood_hazard_area(self):
        pipeline = AutomatedUnderwritingPipeline()
        # Clean roof, clean loss history, but Zone AE (100yr floodplain)
        pdf_text = "Year Built: 2015. Roof replaced: 2020. Prior claims: 0. Flood Zone: AE."

        from ingestion import IngestedDocument, IngestedDocumentPage
        doc = IngestedDocument(
            document_id="d3",
            filename="coastal_report.pdf",
            pages=[IngestedDocumentPage(page_number=1, text=pdf_text)],
            raw_character_count=len(pdf_text),
        )
        payload = RawSubmissionPayload(submission_id="sub_flood_refer", documents=[doc])

        doc_res = pipeline.doc_engine.analyze_submission_documents(payload)
        vis_res = pipeline.vision_engine.analyze_images([])
        fused = pipeline.fusion_engine.fuse("sub_flood_refer", doc_res, vis_res)
        risk = pipeline.risk_engine.calculate_risk(fused)
        comp = pipeline.guidelines_engine.evaluate_compliance(fused)
        decision = pipeline.decision_engine.make_decision(fused, risk, comp)

        # Should NOT decline, but MUST refer to human for flood endorsement
        assert comp.hard_stop_triggered is False
        assert decision.decision == "REFER_TO_HUMAN"
        assert any(v.rule_id == "RULE-FLOOD-104" and v.action == "REFER" for v in comp.violations)

    def test_straight_through_approval_for_pristine_risk(self):
        pipeline = AutomatedUnderwritingPipeline()
        # Pristine home: 2022 build, 2022 roof, 0 claims, Zone X, Masonry
        pdf_text = "Year Built: 2022. Roof replaced: 2022. Construction: Masonry Non-Combustible. Prior claims: 0. Flood Zone: X."

        from ingestion import IngestedDocument, IngestedDocumentPage
        doc = IngestedDocument(
            document_id="d4",
            filename="new_build.pdf",
            pages=[IngestedDocumentPage(page_number=1, text=pdf_text)],
            raw_character_count=len(pdf_text),
        )
        payload = RawSubmissionPayload(submission_id="sub_pristine", documents=[doc])

        doc_res = pipeline.doc_engine.analyze_submission_documents(payload)
        vis_res = pipeline.vision_engine.analyze_images([])
        fused = pipeline.fusion_engine.fuse("sub_pristine", doc_res, vis_res)
        risk = pipeline.risk_engine.calculate_risk(fused)
        comp = pipeline.guidelines_engine.evaluate_compliance(fused)
        decision = pipeline.decision_engine.make_decision(fused, risk, comp)

        assert comp.passed is True
        assert decision.decision == "APPROVE"
        assert decision.risk_score < 35
        assert decision.risk_tier == "Preferred"
