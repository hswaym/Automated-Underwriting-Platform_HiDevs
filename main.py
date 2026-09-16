"""FastAPI Application for the Automated Underwriting Platform.

Provides:
- Web dashboard (/dashboard) for interactive underwriter triage
- /submit multipart endpoint for PDF inspection documents, property photos, and metadata
- /result/{submission_id} endpoint for decision audit package
- /api/underwrite JSON endpoint for programmatic rating
- /health monitoring check
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pipeline import AutomatedUnderwritingPipeline
from schema import UnderwritingDecision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app and core pipeline
app = FastAPI(
    title="AI-Powered Automated Property Underwriting Platform",
    description="Multimodal P&C Underwriting Engine combining document intelligence, computer vision, and guideline compliance.",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
pipeline = AutomatedUnderwritingPipeline()

from config import settings  # noqa: E402 — late import avoids circular deps at startup

# In-memory storage for evaluated submissions
_DECISION_STORE: Dict[str, Dict[str, Any]] = {}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "automated-underwriting-platform",
        "pipeline_ready": True,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    import llm_client
    recent_ids = list(_DECISION_STORE.keys())[-10:]
    active_model = llm_client.get_active_model_name()
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"recent_ids": recent_ids, "active_model": active_model},
    )


@app.post("/submit")
async def submit(
    pdf_files: List[UploadFile] = File(default=[]),
    image_files: List[UploadFile] = File(default=[]),
    json_data: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    """Accepts multipart submission of inspection PDFs, property photos, and metadata."""
    if not pdf_files and not image_files and not json_data:
        raise HTTPException(status_code=422, detail="No submission files or metadata provided.")

    pdf_pairs = [(f.filename or "report.pdf", await f.read()) for f in pdf_files if f.filename]
    img_pairs = [(f.filename or "photo.jpg", await f.read()) for f in image_files if f.filename]

    meta_dict: Dict[str, Any] = {}
    if json_data:
        try:
            meta_dict = json.loads(json_data)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSON metadata payload: {exc}") from exc

    try:
        execution_result = pipeline.process_submission(
            pdf_files=pdf_pairs,
            image_files=img_pairs,
            metadata=meta_dict,
        )
        sub_id = execution_result["submission_id"]
        _DECISION_STORE[sub_id] = execution_result
        dec = execution_result["decision"]
        return {
            "submission_id": sub_id,
            "decision": dec.decision,
            "risk_score": dec.risk_score,
            "risk_tier": dec.risk_tier,
        }
    except Exception as exc:
        logger.error("Underwriting pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {exc}") from exc


@app.get("/result/{submission_id}")
def get_result(submission_id: str):
    """Returns the decision payload formatted for both the underwriter dashboard and API."""
    if submission_id not in _DECISION_STORE:
        raise HTTPException(status_code=404, detail="Submission record not found.")

    res = _DECISION_STORE[submission_id]
    decision: UnderwritingDecision = res["decision"]
    doc_res = res["document_extractions"]
    vis_res = res["vision_findings"]
    fused = res["fused_profile"]
    comp = res["compliance_result"]

    # Transform to format consumed by dashboard UI
    dashboard_flags = []
    for h in decision.detected_hazards:
        dashboard_flags.append({
            "name": h.hazard_type.replace("_", " ").title(),
            "severity": h.severity,
            "description": h.description,
            "source": f"Vision: {h.source_image}",
        })
    for c in decision.contradictions:
        dashboard_flags.append({
            "name": f"Contradiction: {c.field_name}",
            "severity": c.discrepancy_severity,
            "description": c.explanation,
            "source": "Multimodal Cross-Check",
        })

    # Format clean, human-readable section reasoning (no raw 'None' values)
    if doc_res.year_built and doc_res.square_footage:
        doc_finding = f"Extracted build year {doc_res.year_built.value}, {doc_res.square_footage.value:,.0f} sqft (Overall confidence: {doc_res.overall_confidence:.0%})."
    elif doc_res.year_built:
        doc_finding = f"Extracted build year {doc_res.year_built.value} (Overall confidence: {doc_res.overall_confidence:.0%}). Square footage not specified in documents."
    elif doc_res.square_footage:
        doc_finding = f"Extracted {doc_res.square_footage.value:,.0f} sqft (Overall confidence: {doc_res.overall_confidence:.0%}). Build year unverified in documents."
    else:
        doc_finding = f"No build year or square footage identified in uploaded document text (Overall confidence: {doc_res.overall_confidence:.0%}). Mandatory fields flagged for intake review."

    img_count = res.get("image_count", 0)
    if img_count == 0 and len(vis_res.hazards) == 0:
        vis_finding = "No property photos uploaded. Visual hazard inspection skipped (Standard nominal baseline applied)."
    else:
        vis_finding = f"Analyzed {max(1, img_count)} property photo(s); identified {len(vis_res.hazards)} visual hazard(s). Estimated roof condition index: {vis_res.roof_condition_score:.1f}/100."

    if fused.contradictions:
        contra_finding = f"Cross-modal audit identified {len(fused.contradictions)} material discrepancy(ies) between document declarations and inspection photos."
    else:
        contra_finding = "Multimodal verification complete: 0 contradictions detected between application declarations and photographic evidence."

    risk_b = res.get("risk_breakdown")
    risk_finding = (
        f"Actuarial risk score evaluated at {decision.risk_score}/100 ({decision.risk_tier}) "
        f"across structural ({risk_b.structural_score if risk_b else 0} pts), "
        f"environmental ({risk_b.environmental_score if risk_b else 0} pts), "
        f"and liability ({risk_b.liability_score if risk_b else 0} pts) pillars."
    )

    section_reasoning = [
        {"section": "Document Structuring", "finding": doc_finding},
        {"section": "Visual Assessment", "finding": vis_finding},
        {"section": "Multimodal Integrity", "finding": contra_finding},
        {"section": "Actuarial Risk Scoring", "finding": risk_finding},
    ]

    import llm_client
    return {
        "submission_id": submission_id,
        "decision": decision.decision.lower(),
        "risk_score": decision.risk_score,
        "risk_tier": decision.risk_tier,
        "confidence": decision.confidence,
        "summary": decision.summary_rationale,
        "active_model": llm_client.get_active_model_name(),
        "flags": dashboard_flags,
        "section_reasoning": section_reasoning,
        "guideline_references": comp.rule_citations,
        "required_actions": decision.required_underwriter_actions,
        "can_override": decision.can_underwriter_override,
    }
