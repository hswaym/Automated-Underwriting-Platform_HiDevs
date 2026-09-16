"""Stage 7: Decision Output & Explainability Layer.

Orchestrates the final underwriting recommendation (APPROVE, DECLINE, REFER_TO_HUMAN)
with a complete, immutable audit trail connecting document extractions,
computer vision hazards, risk score factor weights, and underwriting guideline citations.
"""

from __future__ import annotations

import logging
from typing import List, Literal
from schema import (
    ComplianceCheckResult,
    RiskScoreBreakdown,
    UnderwritingDecision,
    UnifiedRiskProfile,
)

logger = logging.getLogger(__name__)


class UnderwritingDecisionEngine:
    """Combines risk analytics and regulatory compliance into an explainable decision package."""

    def make_decision(
        self,
        profile: UnifiedRiskProfile,
        risk_breakdown: RiskScoreBreakdown,
        compliance: ComplianceCheckResult,
    ) -> UnderwritingDecision:
        score = risk_breakdown.risk_score
        actions = [v.action for v in compliance.violations]

        # -------------------------------------------------------------------
        # Decision Triage Logic
        # -------------------------------------------------------------------
        decision: Literal["APPROVE", "DECLINE", "REFER_TO_HUMAN"]
        confidence: Literal["low", "medium", "high"]
        action_items: List[str] = []

        if compliance.hard_stop_triggered or score >= 80:
            decision = "DECLINE"
            confidence = "high"
            decline_reasons = [v.reason for v in compliance.violations if v.action == "DECLINE"]
            if score >= 80 and not decline_reasons:
                decline_reasons.append(f"Aggregate risk score ({score}/100) exceeds maximum acceptable carrier threshold.")
            action_items.extend(decline_reasons)
            can_override = False if len(decline_reasons) >= 2 else True

        elif "REFER" in actions or score >= 45 or profile.hitl_required:
            decision = "REFER_TO_HUMAN"
            confidence = "medium"
            refer_reasons = [v.reason for v in compliance.violations if v.action == "REFER"]
            if profile.hitl_required:
                refer_reasons.extend(profile.hitl_reasons)
            if score >= 45:
                refer_reasons.append(f"Moderate risk score ({score}/100) requires underwriter pricing review.")
            action_items.extend(refer_reasons)
            can_override = True

        else:
            decision = "APPROVE"
            confidence = "high"
            action_items.append("Eligible for Straight-Through-Processing (STP). No mandatory endorsements required.")
            can_override = True

        # Determine human-friendly risk tier for output
        tier = risk_breakdown.risk_tier
        if decision == "REFER_TO_HUMAN" and profile.hitl_required and tier == "Preferred":
            display_tier = "Preferred (Pending Verification)"
        else:
            display_tier = tier

        # Generate summary rationale
        summary = self._build_summary(profile, decision, score, compliance, display_tier)

        return UnderwritingDecision(
            submission_id=profile.submission_id,
            decision=decision,
            confidence=confidence,
            risk_score=score,
            risk_tier=display_tier,
            summary_rationale=summary,
            risk_breakdown=risk_breakdown,
            compliance_result=compliance,
            contradictions=profile.contradictions,
            detected_hazards=profile.visual_hazards,
            required_underwriter_actions=action_items,
            can_underwriter_override=can_override,
        )

    def _build_summary(
        self,
        profile: UnifiedRiskProfile,
        decision: str,
        score: int,
        compliance: ComplianceCheckResult,
        tier: str,
    ) -> str:
        lines = [
            f"Automated Underwriting Recommendation: {decision} (Risk Score: {score}/100, Tier: {tier}, Construction: {profile.construction_type})."
        ]

        if compliance.violations:
            lines.append(f"Triggered {len(compliance.violations)} underwriting guideline constraints:")
            for v in compliance.violations:
                lines.append(f" • [{v.action}] {v.citation}: {v.reason}")

        if profile.contradictions:
            lines.append(f"Identified {len(profile.contradictions)} document-versus-photo discrepancies:")
            for c in profile.contradictions:
                lines.append(f" • [{c.discrepancy_severity.upper()}] {c.explanation}")

        if decision == "APPROVE":
            lines.append("Property demonstrates clean loss history, adequate roof condition, and full compliance with underwriting appetite.")

        return "\n".join(lines)

