"""Stage 6: Guideline Compliance Layer.

Encodes deterministic business rules, hard underwriting stops, and mandatory
referral criteria completely decoupled from machine learning weights.
Every violation references an exact section in the Underwriting Manual.
"""

from __future__ import annotations

import logging
from typing import List
from schema import ComplianceCheckResult, GuidelineRuleViolation, UnifiedRiskProfile

logger = logging.getLogger(__name__)


class UnderwritingGuidelinesEngine:
    """Deterministic, auditable compliance rules engine for insurance guidelines."""

    def evaluate_compliance(self, profile: UnifiedRiskProfile) -> ComplianceCheckResult:
        violations: List[GuidelineRuleViolation] = []

        # -------------------------------------------------------------------
        # Guideline 1: Roof Age Threshold (HO-UW-202)
        # -------------------------------------------------------------------
        if profile.roof_age_years > 20:
            violations.append(
                GuidelineRuleViolation(
                    rule_id="RULE-ROOF-202",
                    citation="Underwriting Guidelines Manual § 3.2 (HO-UW-202)",
                    rule_category="Property Condition",
                    action="DECLINE",
                    reason=f"Roof age of {profile.roof_age_years} years exceeds the maximum carrier limit of 20 years without certified replacement.",
                )
            )

        # -------------------------------------------------------------------
        # Guideline 2: Severe Visual Roof Damage (HO-ROOF-205)
        # -------------------------------------------------------------------
        critical_roof_hazards = [
            h for h in profile.visual_hazards if h.category == "roof" and h.severity == "critical"
        ]
        if critical_roof_hazards:
            violations.append(
                GuidelineRuleViolation(
                    rule_id="RULE-ROOF-205",
                    citation="Underwriting Guidelines Manual § 3.5 (HO-ROOF-205)",
                    rule_category="Roof Integrity",
                    action="DECLINE",
                    reason="Active physical roof damage (missing shingles/severe granular loss) detected in inspection photos.",
                )
            )

        # -------------------------------------------------------------------
        # Guideline 3: Special Flood Hazard Area (HO-ENV-104)
        # -------------------------------------------------------------------
        if profile.flood_zone in ["A", "AE", "AH", "AO", "V", "VE"]:
            violations.append(
                GuidelineRuleViolation(
                    rule_id="RULE-FLOOD-104",
                    citation="Coastal & Flood Underwriting Standard § 1.4 (HO-ENV-104)",
                    rule_category="Environmental Exposure",
                    action="REFER",
                    reason=f"Property is situated in FEMA 100-year Special Flood Hazard Area (Zone {profile.flood_zone}). Mandatory flood endorsement and elevation certificate required.",
                )
            )

        # -------------------------------------------------------------------
        # Guideline 4: Prior Claim Frequency (HO-CLM-301)
        # -------------------------------------------------------------------
        if profile.prior_claims_count >= 3:
            violations.append(
                GuidelineRuleViolation(
                    rule_id="RULE-LOSS-301",
                    citation="Loss History & Eligibility Manual § 4.1 (HO-CLM-301)",
                    rule_category="Loss History",
                    action="DECLINE",
                    reason=f"Loss history reflects {profile.prior_claims_count} claims within 5-year lookback period (maximum allowable: 2).",
                )
            )
        elif profile.prior_claims_count == 2:
            violations.append(
                GuidelineRuleViolation(
                    rule_id="RULE-LOSS-302",
                    citation="Loss History & Eligibility Manual § 4.2 (HO-CLM-302)",
                    rule_category="Loss History",
                    action="REFER",
                    reason="2 prior claims within lookback period requires underwriter verification of prior loss remediation.",
                )
            )

        # -------------------------------------------------------------------
        # Guideline 5: Unfenced Swimming Pool / Attractive Nuisance (HO-LIA-405)
        # -------------------------------------------------------------------
        pool_hazards = [h for h in profile.visual_hazards if "pool" in h.hazard_type]
        if pool_hazards:
            violations.append(
                GuidelineRuleViolation(
                    rule_id="RULE-LIA-405",
                    citation="Premises Liability Standards § 6.3 (HO-LIA-405)",
                    rule_category="Liability",
                    action="REFER",
                    reason="Swimming pool identified on premises. Underwriter must verify presence of self-locking 4-foot safety fence.",
                )
            )

        # -------------------------------------------------------------------
        # Guideline 6: Critical Multimodal Discrepancies / Material Fraud (HO-FRD-901)
        # -------------------------------------------------------------------
        critical_contradictions = [c for c in profile.contradictions if c.discrepancy_severity == "critical"]
        if critical_contradictions:
            violations.append(
                GuidelineRuleViolation(
                    rule_id="RULE-FRD-901",
                    citation="Fraud Prevention & Integrity Policy § 9.1 (HO-FRD-901)",
                    rule_category="Fraud & Misrepresentation",
                    action="DECLINE",
                    reason=f"Material contradiction between application declarations and photographic evidence: {critical_contradictions[0].explanation}",
                )
            )

        # -------------------------------------------------------------------
        # Guideline 7: Boarded Openings / Unsecured Envelope (HO-STR-102)
        # -------------------------------------------------------------------
        boarded_hazards = [h for h in profile.visual_hazards if "boarded" in h.hazard_type or "unsecured" in h.hazard_type]
        if boarded_hazards:
            violations.append(
                GuidelineRuleViolation(
                    rule_id="RULE-STR-102",
                    citation="Property Eligibility Guidelines § 2.1 (HO-STR-102)",
                    rule_category="Structure Security",
                    action="DECLINE",
                    reason="Structure has boarded windows or unsecured envelope indicating abandonment or severe blight.",
                )
            )

        # -------------------------------------------------------------------
        # Guideline 8: Mandatory Document Intake Verification (HO-DOC-101)
        # -------------------------------------------------------------------
        if profile.hitl_required:
            reasons_summary = "; ".join(profile.hitl_reasons[:2]) if profile.hitl_reasons else "Unverified property attributes"
            violations.append(
                GuidelineRuleViolation(
                    rule_id="RULE-DOC-101",
                    citation="Submission Intake & Verification Policy § 1.1 (HO-DOC-101)",
                    rule_category="Document Completeness",
                    action="REFER",
                    reason=f"Mandatory intake verification required: {reasons_summary}.",
                )
            )

        # Determine pass/fail state
        hard_stop = any(v.action == "DECLINE" for v in violations)
        passed = len(violations) == 0

        citations = [v.citation for v in violations]

        return ComplianceCheckResult(
            passed=passed,
            hard_stop_triggered=hard_stop,
            violations=violations,
            rule_citations=citations,
        )
