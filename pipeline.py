"""End-to-end Automated Underwriting Pipeline Orchestrator.

Wires all 7 stages into a single unified callable pipeline:
Ingestion -> Doc Analysis -> Vision Analysis -> Multimodal Fusion ->
Risk Scoring -> Guideline Compliance -> Decision & Audit Package.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from doc_analysis import DocumentAnalysisEngine
from decision_engine import UnderwritingDecisionEngine
from guidelines_engine import UnderwritingGuidelinesEngine
from ingestion import ingest_submission
from multimodal_fusion import MultimodalFusionEngine
from risk_engine import ActuarialRiskEngine
from schema import (
    DocumentExtractionResult,
    RawSubmissionPayload,
    UnderwritingDecision,
    UnifiedRiskProfile,
    VisionAnalysisResult,
)
from vision_pipeline import PropertyVisionAnalyzer
import claude_client

logger = logging.getLogger(__name__)


class AutomatedUnderwritingPipeline:
    """Production orchestrator for the AI-Powered Underwriting Platform."""

    def __init__(self, hitl_confidence_threshold: float = 0.75):
        self.doc_engine = DocumentAnalysisEngine(hitl_confidence_threshold=hitl_confidence_threshold)
        self.vision_engine = PropertyVisionAnalyzer()
        self.fusion_engine = MultimodalFusionEngine()
        self.risk_engine = ActuarialRiskEngine()
        self.guidelines_engine = UnderwritingGuidelinesEngine()
        self.decision_engine = UnderwritingDecisionEngine()

    def process_submission(
        self,
        pdf_files: List[Tuple[str, bytes]],
        image_files: List[Tuple[str, bytes]],
        metadata: Optional[Dict[str, Any]] = None,
        submission_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the end-to-end pipeline returning the decision package and intermediate state."""

        # -------------------------------------------------------------------
        # Stage 1: Data Ingestion (Documents + Images)
        # -------------------------------------------------------------------
        raw_payload: RawSubmissionPayload = ingest_submission(
            pdf_files=pdf_files,
            image_files=image_files,
            metadata=metadata,
            submission_id=submission_id,
        )
        sub_id = raw_payload.submission_id
        logger.info("Stage 1 Ingestion complete for %s (%d docs, %d images)", sub_id, len(raw_payload.documents), len(raw_payload.images))

        # -------------------------------------------------------------------
        # Stage 2: Document Analysis (Text Extraction & Structuring)
        # -------------------------------------------------------------------
        doc_result: DocumentExtractionResult = self.doc_engine.analyze_submission_documents(raw_payload)
        logger.info("Stage 2 Doc Analysis complete (confidence: %.2f, HITL: %s)", doc_result.overall_confidence, doc_result.hitl_review_required)

        # -------------------------------------------------------------------
        # Stage 3: Computer Vision / Image-Based Risk Detection
        # -------------------------------------------------------------------
        vision_result: VisionAnalysisResult = self.vision_engine.analyze_images(raw_payload.images)
        logger.info("Stage 3 Vision complete (%d hazards detected, roof condition: %.1f)", len(vision_result.hazards), vision_result.roof_condition_score)

        # -------------------------------------------------------------------
        # Stage 4: Multimodal Fusion (Text + Image Reasoning)
        # -------------------------------------------------------------------
        fused_profile: UnifiedRiskProfile = self.fusion_engine.fuse(
            submission_id=sub_id,
            doc_result=doc_result,
            vision_result=vision_result,
        )
        logger.info("Stage 4 Fusion complete (%d contradictions flagged)", len(fused_profile.contradictions))

        # -------------------------------------------------------------------
        # Stage 5: Risk Assessment Engine (Actuarial Multi-Pillar Scoring)
        # -------------------------------------------------------------------
        risk_breakdown = self.risk_engine.calculate_risk(fused_profile)
        logger.info("Stage 5 Risk Scoring complete (Score: %d, Tier: %s)", risk_breakdown.risk_score, risk_breakdown.risk_tier)

        # -------------------------------------------------------------------
        # Stage 6: Guideline Compliance Layer
        # -------------------------------------------------------------------
        compliance_result = self.guidelines_engine.evaluate_compliance(fused_profile)
        logger.info("Stage 6 Compliance complete (Passed: %s, Hard Stop: %s)", compliance_result.passed, compliance_result.hard_stop_triggered)

        # -------------------------------------------------------------------
        # Stage 7: Decision Output & Explainability
        # -------------------------------------------------------------------
        final_decision: UnderwritingDecision = self.decision_engine.make_decision(
            profile=fused_profile,
            risk_breakdown=risk_breakdown,
            compliance=compliance_result,
        )
        logger.info("Stage 7 Decision rendered: %s", final_decision.decision)

        # -------------------------------------------------------------------
        # Stage 7b: Open-Source LLM Narrative & Risk Enrichment
        # -------------------------------------------------------------------
        import llm_client
        if llm_client.is_available():
            doc_text = "\n\n".join(
                page.text
                for doc in raw_payload.documents
                for page in doc.pages
                if page.text
            )
            try:
                final_decision = llm_client.enrich_decision(
                    profile=fused_profile,
                    risk_breakdown=risk_breakdown,
                    compliance=compliance_result,
                    base_decision=final_decision,
                    document_text=doc_text,
                )
                logger.info("Stage 7b Open-Source LLM (%s) enrichment applied for %s", llm_client.get_active_model_name(), sub_id)
            except Exception as exc:
                logger.warning("Stage 7b LLM enrichment skipped: %s", exc)


        return {
            "submission_id": sub_id,
            "decision": final_decision,
            "document_extractions": doc_result,
            "vision_findings": vision_result,
            "fused_profile": fused_profile,
            "risk_breakdown": risk_breakdown,
            "compliance_result": compliance_result,
            "image_count": len(raw_payload.images),
            "doc_count": len(raw_payload.documents),
        }
