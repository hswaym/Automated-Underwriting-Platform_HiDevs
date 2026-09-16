"""Demo mode Claude client — returns mock decisions for testing without API key."""

from __future__ import annotations

import logging
import time
from schema import ProcessedSubmission, UnderwritingDecision, HazardFlag, SectionReasoning

logger = logging.getLogger(__name__)


def analyze(submission: ProcessedSubmission, guidelines: str) -> UnderwritingDecision:
    """
    DEMO MODE: Returns realistic mock underwriting decision.
    Replace this with real claude_client.py when API key is working.
    """
    logger.info(f"[DEMO MODE] Analyzing submission {submission.submission_id}")

    # Simulate processing time
    time.sleep(2)

    # Determine decision based on simple heuristics from structured data
    decision = "approve"
    risk_score = 30
    confidence = "high"
    flags = []

    if submission.structured_data:
        year_built = submission.structured_data.get("year_built")
        roof_year = submission.structured_data.get("roof_year")
        prior_claims = submission.structured_data.get("prior_claims", 0)
        flood_zone = submission.structured_data.get("flood_zone", "X")

        # Age-based risk
        if year_built and year_built < 1970:
            risk_score += 15
            flags.append(HazardFlag(
                name="Older Construction",
                severity="medium",
                description=f"Property built in {year_built}, may require system updates",
                source="structured_data"
            ))

        # Roof age
        if roof_year:
            roof_age = 2026 - roof_year
            if roof_age > 20:
                risk_score += 25
                decision = "refer"
                confidence = "medium"
                flags.append(HazardFlag(
                    name="Aging Roof",
                    severity="high",
                    description=f"Roof is {roof_age} years old, exceeds 20-year threshold",
                    source="structured_data"
                ))
            elif roof_age > 15:
                risk_score += 10
                flags.append(HazardFlag(
                    name="Roof Approaching Age Limit",
                    severity="medium",
                    description=f"Roof is {roof_age} years old, monitor condition",
                    source="structured_data"
                ))

        # Prior claims
        if prior_claims >= 2:
            risk_score += 20
            decision = "refer"
            flags.append(HazardFlag(
                name="Multiple Prior Claims",
                severity="high",
                description=f"{prior_claims} claims in recent history",
                source="structured_data"
            ))

        # Flood zone
        if flood_zone in ["A", "AE", "V", "VE"]:
            risk_score += 15
            flags.append(HazardFlag(
                name="FEMA Special Flood Hazard Area",
                severity="medium",
                description=f"Property in flood zone {flood_zone}, NFIP required",
                source="structured_data"
            ))

    # Check text content for red flags
    all_text = " ".join(chunk.content.lower() for chunk in submission.text_chunks)

    if "foundation crack" in all_text or "structural" in all_text:
        risk_score += 20
        decision = "refer"
        flags.append(HazardFlag(
            name="Structural Concerns",
            severity="high",
            description="Inspection report mentions foundation or structural issues",
            source="inspection_report"
        ))

    if "mold" in all_text or "water damage" in all_text:
        risk_score += 15
        flags.append(HazardFlag(
            name="Water Damage/Mold",
            severity="medium",
            description="Evidence of moisture issues",
            source="inspection_report"
        ))

    # Cap risk score
    risk_score = min(risk_score, 100)

    # Adjust decision based on final score
    if risk_score >= 70:
        decision = "decline"
        confidence = "high"
    elif risk_score >= 50:
        decision = "refer"
        confidence = "medium"

    # Generate reasoning
    section_reasoning = [
        SectionReasoning(
            section="Property Overview",
            finding=f"Property assessment completed. {len(submission.text_chunks)} document(s) and {len(submission.images)} image(s) analyzed. Structured data {'provided' if submission.structured_data else 'not provided'}."
        ),
        SectionReasoning(
            section="Risk Assessment",
            finding=f"Overall risk score of {risk_score}/100 based on property age, system conditions, and claims history. {len(flags)} hazard flag(s) identified."
        ),
    ]

    if decision == "approve":
        summary = f"Property meets underwriting guidelines for standard acceptance. Risk score {risk_score}/100 falls within acceptable range. {'No significant hazards identified.' if not flags else f'{len(flags)} minor flag(s) noted but do not preclude approval.'}"
        section_reasoning.append(SectionReasoning(
            section="Recommendation",
            finding="Approve for standard homeowner's policy with normal rating."
        ))
    elif decision == "refer":
        summary = f"Property requires senior underwriter review. Risk score {risk_score}/100 indicates discretionary factors present. {len(flags)} flag(s) require evaluation before final decision."
        section_reasoning.append(SectionReasoning(
            section="Recommendation",
            finding="Refer to senior underwriting team for manual review and additional documentation."
        ))
    else:
        summary = f"Property does not meet underwriting guidelines. Risk score {risk_score}/100 exceeds acceptable threshold. {len(flags)} significant hazard(s) identified."
        section_reasoning.append(SectionReasoning(
            section="Recommendation",
            finding="Decline coverage due to unacceptable risk factors."
        ))

    guideline_refs = [
        "§2.2 Age & Condition",
        "§3.1 Roof Age Thresholds",
        "§6.1 Prior Claims",
    ]

    logger.info(f"[DEMO MODE] Decision: {decision}, Score: {risk_score}")

    return UnderwritingDecision(
        submission_id=submission.submission_id,
        decision=decision,
        risk_score=risk_score,
        confidence=confidence,
        flags=flags,
        section_reasoning=section_reasoning,
        summary=summary,
        guideline_references=guideline_refs,
    )
