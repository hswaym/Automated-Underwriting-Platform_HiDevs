"""Stage 5: Risk Assessment Engine.

Calculates an explainable, auditable property risk score (0-100) across 5 core
underwriting pillars:
1. Structural Integrity & Construction
2. Environmental & Geographic Exposure
3. Property Age & Maintenance Deterioration
4. Prior Claims & Loss History
5. Liability Exposures & Multimodal Contradictions
"""

from __future__ import annotations

import logging
from typing import List
from schema import FactorContribution, RiskScoreBreakdown, UnifiedRiskProfile

logger = logging.getLogger(__name__)


class ActuarialRiskEngine:
    """Transparent, factor-weighted property risk scoring model."""

    # Baseline risk points for standard residential risk
    BASE_POINTS = 10

    def calculate_risk(self, profile: UnifiedRiskProfile) -> RiskScoreBreakdown:
        contributions: List[FactorContribution] = []

        # -------------------------------------------------------------------
        # Pillar 1: Structural Integrity (Max 25 pts)
        # -------------------------------------------------------------------
        structural_pts = 0
        c_type = profile.construction_type.lower()
        if "frame" in c_type:
            structural_pts += 10
            contributions.append(FactorContribution(
                factor_name="construction_combustibility",
                category="Structural",
                points=10,
                description="Frame construction is combustible, higher fire vulnerability than masonry."
            ))
        elif "masonry" in c_type:
            structural_pts += 3
            contributions.append(FactorContribution(
                factor_name="construction_durability",
                category="Structural",
                points=3,
                description="Masonry non-combustible exterior reduces structural fire risk."
            ))

        # Check visual structural hazards
        struct_hazards = [h for h in profile.visual_hazards if h.category == "structural"]
        if struct_hazards:
            pts = 15 if any(h.severity == "critical" for h in struct_hazards) else 8
            structural_pts += pts
            contributions.append(FactorContribution(
                factor_name="visual_structural_distress",
                category="Structural",
                points=pts,
                description=f"Visual evidence of structural compromise: {struct_hazards[0].description}"
            ))

        # -------------------------------------------------------------------
        # Pillar 2: Environmental Exposure (Max 25 pts)
        # -------------------------------------------------------------------
        environmental_pts = 0
        fz = profile.flood_zone.upper()
        if fz in ["A", "AE", "AH", "AO", "V", "VE"]:
            environmental_pts += 20
            contributions.append(FactorContribution(
                factor_name="special_flood_hazard_area",
                category="Environmental",
                points=20,
                description=f"Property located in FEMA high-risk 100-year flood zone (Zone {fz})."
            ))
        elif fz in ["B", "C", "X500"]:
            environmental_pts += 5
            contributions.append(FactorContribution(
                factor_name="moderate_flood_zone",
                category="Environmental",
                points=5,
                description="Moderate flood hazard area (500-year flood zone)."
            ))
        else:
            contributions.append(FactorContribution(
                factor_name="minimal_flood_risk",
                category="Environmental",
                points=0,
                description="Zone X indicates minimal flood risk outside the 500-year floodplain."
            ))

        # Vegetation / Wildfire defensible space
        veg_hazards = [h for h in profile.visual_hazards if "vegetation" in h.hazard_type]
        if veg_hazards:
            environmental_pts += 5
            contributions.append(FactorContribution(
                factor_name="inadequate_defensible_space",
                category="Environmental",
                points=5,
                description="Dense dry brush or overhanging tree limbs detected within 10ft perimeter."
            ))

        # -------------------------------------------------------------------
        # Pillar 3: Age & Maintenance Deterioration (Max 25 pts)
        # -------------------------------------------------------------------
        maintenance_pts = 0
        # Roof age points
        if profile.roof_age_years > 20:
            maintenance_pts += 15
            contributions.append(FactorContribution(
                factor_name="aged_roof_exceeds_threshold",
                category="Maintenance",
                points=15,
                description=f"Roof age is {profile.roof_age_years} years (exceeds standard 20-year underwriting benchmark)."
            ))
        elif profile.roof_age_years > 12:
            maintenance_pts += 8
            contributions.append(FactorContribution(
                factor_name="mid_life_roof_wear",
                category="Maintenance",
                points=8,
                description=f"Roof age is {profile.roof_age_years} years (approaching replacement cycle)."
            ))

        # Roof visual condition
        roof_hazards = [h for h in profile.visual_hazards if h.category == "roof"]
        if roof_hazards:
            pts = 10 if any(h.severity == "critical" for h in roof_hazards) else 5
            maintenance_pts += pts
            contributions.append(FactorContribution(
                factor_name="active_roof_damage",
                category="Maintenance",
                points=pts,
                description=f"Visible roof defects observed: {roof_hazards[0].description}"
            ))

        # Building age points
        if profile.effective_building_age > 40:
            maintenance_pts += 5
            contributions.append(FactorContribution(
                factor_name="vintage_structure_age",
                category="Maintenance",
                points=5,
                description=f"Structure built {profile.year_built} ({profile.effective_building_age} years old) without full modern retrofit."
            ))

        # -------------------------------------------------------------------
        # Pillar 4: Prior Claims & Loss History (Max 25 pts)
        # -------------------------------------------------------------------
        claims_pts = 0
        if profile.prior_claims_count >= 3:
            claims_pts += 20
            contributions.append(FactorContribution(
                factor_name="chronic_claim_frequency",
                category="Claims",
                points=20,
                description=f"{profile.prior_claims_count} prior claims logged within 5-year lookback."
            ))
        elif profile.prior_claims_count == 2:
            claims_pts += 12
            contributions.append(FactorContribution(
                factor_name="elevated_claim_frequency",
                category="Claims",
                points=12,
                description="2 prior claims logged within 5-year lookback."
            ))
        elif profile.prior_claims_count == 1:
            claims_pts += 6
            contributions.append(FactorContribution(
                factor_name="isolated_prior_claim",
                category="Claims",
                points=6,
                description="1 prior claim logged within 5-year lookback."
            ))

        # -------------------------------------------------------------------
        # Pillar 5: Liability & Multimodal Contradiction Penalties
        # -------------------------------------------------------------------
        liability_pts = 0
        pool_hazards = [h for h in profile.visual_hazards if "pool" in h.hazard_type]
        if pool_hazards:
            liability_pts += 8
            contributions.append(FactorContribution(
                factor_name="attractive_nuisance_pool",
                category="Liability",
                points=8,
                description="Swimming pool presents increased premises liability and drowning exposure."
            ))

        # Contradiction Penalties (Crucial anti-fraud scoring)
        contradiction_penalty = 0
        for c in profile.contradictions:
            penalty = 15 if c.discrepancy_severity == "critical" else 8
            contradiction_penalty += penalty
            contributions.append(FactorContribution(
                factor_name="multimodal_discrepancy_penalty",
                category="Integrity & Contradictions",
                points=penalty,
                description=f"{c.field_name}: {c.explanation}"
            ))

        # -------------------------------------------------------------------
        # Final Score Aggregation & Tiering
        # -------------------------------------------------------------------
        raw_total = (
            self.BASE_POINTS
            + structural_pts
            + environmental_pts
            + maintenance_pts
            + claims_pts
            + liability_pts
            + contradiction_penalty
        )
        final_score = min(100, max(0, raw_total))

        if final_score < 35:
            tier = "Preferred"
        elif final_score < 65:
            tier = "Standard"
        elif final_score < 85:
            tier = "Non-Standard"
        else:
            tier = "Ineligible"

        return RiskScoreBreakdown(
            risk_score=final_score,
            risk_tier=tier,
            factor_contributions=contributions,
            structural_score=structural_pts,
            environmental_score=environmental_pts,
            maintenance_score=maintenance_pts,
            claims_score=claims_pts,
            liability_score=liability_pts,
            contradiction_penalty=contradiction_penalty,
        )


# In-memory result cache
_results: dict[str, Any] = {}


def store(decision: Any) -> None:
    """Store an UnderwritingDecision by its submission_id."""
    _results[decision.submission_id] = decision


def get(submission_id: str) -> Any:
    """Retrieve an UnderwritingDecision by submission_id."""
    return _results.get(submission_id)


def all_ids() -> list[str]:
    """List all stored submission IDs."""
    return list(_results.keys())

