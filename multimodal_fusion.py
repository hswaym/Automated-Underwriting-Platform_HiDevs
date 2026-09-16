"""Stage 4: Multimodal Fusion Engine.

Cross-verifies document text claims against computer vision findings.
Detects contradictions (e.g. claimed new roof vs. visible missing shingles,
undisclosed swimming pools, unmaintained brush vs clean disclosure)
and constructs a UnifiedRiskProfile.
"""

from __future__ import annotations

import logging
import uuid
from typing import List
from schema import (
    ContradictionFlag,
    DocumentExtractionResult,
    UnifiedRiskProfile,
    VisionAnalysisResult,
)

logger = logging.getLogger(__name__)


class MultimodalFusionEngine:
    """Fuses structured document facts and visual observations into a consolidated risk profile."""

    CURRENT_YEAR = 2026

    def fuse(
        self,
        submission_id: str,
        doc_result: DocumentExtractionResult,
        vision_result: VisionAnalysisResult,
    ) -> UnifiedRiskProfile:
        contradictions: List[ContradictionFlag] = []

        # 1. Resolve Core Numeric Features
        year_built = int(doc_result.year_built.value) if doc_result.year_built else 1985
        building_age = max(0, self.CURRENT_YEAR - year_built)

        roof_year = int(doc_result.roof_installed_year.value) if doc_result.roof_installed_year else (self.CURRENT_YEAR - 15)
        roof_age = max(0, self.CURRENT_YEAR - roof_year)

        sq_ft = float(doc_result.square_footage.value) if doc_result.square_footage else 2200.0
        construction = str(doc_result.construction_type.value) if doc_result.construction_type else "Frame"
        flood_zone = str(doc_result.flood_zone.value) if doc_result.flood_zone else "X"

        claims_count = int(doc_result.prior_claims_count.value) if doc_result.prior_claims_count else 0
        claims_amt = float(doc_result.prior_claims_loss_total.value) if doc_result.prior_claims_loss_total else 0.0

        # Visual Roof Condition Description
        visual_roof_cond = (
            "Severe Damage / Shingle Loss"
            if vision_result.roof_condition_score < 50
            else "Moderate Wear"
            if vision_result.roof_condition_score < 75
            else "Good / Maintained"
        )

        # -------------------------------------------------------------------
        # Contradiction Detection Checks
        # -------------------------------------------------------------------

        # Contradiction 1: Roof Replacement Age vs. Visual Roof Distress
        roof_damage_hazards = [h for h in vision_result.hazards if h.category == "roof" and h.severity in ["high", "critical"]]
        if roof_age <= 6 and (roof_damage_hazards or vision_result.roof_condition_score < 60):
            contradictions.append(
                ContradictionFlag(
                    flag_id=f"contra_{uuid.uuid4().hex[:6]}",
                    field_name="roof_condition_and_age",
                    document_claim=f"Report documents roof replaced in {roof_year} ({roof_age} years old - claimed new/recent).",
                    visual_evidence=f"Exterior photo reveals {visual_roof_cond} (score {vision_result.roof_condition_score}/100) with active granular/shingle loss.",
                    discrepancy_severity="critical",
                    explanation="Material contradiction: Insured claims a recently replaced roof, but visual evidence shows advanced degradation or storm damage.",
                )
            )

        # Contradiction 2: Undisclosed Attractive Nuisances (Swimming Pool)
        pool_hazards = [h for h in vision_result.hazards if "swimming_pool" in h.hazard_type]
        if pool_hazards:
            # Check if pool was mentioned in document text
            pool_documented = False
            if doc_result.property_address and "pool" in str(doc_result.property_address.value).lower():
                pool_documented = True

            if not pool_documented:
                contradictions.append(
                    ContradictionFlag(
                        flag_id=f"contra_{uuid.uuid4().hex[:6]}",
                        field_name="undisclosed_pool",
                        document_claim="No swimming pool or attractive nuisance disclosed in underwriting documentation.",
                        visual_evidence=f"Computer vision detected an in-ground swimming pool in {pool_hazards[0].source_image} (confidence {pool_hazards[0].confidence:.2f}).",
                        discrepancy_severity="major",
                        explanation="Liability exposure omitted: Unfenced swimming pools represent significant third-party liability without appropriate safety fencing.",
                    )
                )

        # Contradiction 3: Brush Clearance / Defensible Space
        brush_hazards = [h for h in vision_result.hazards if "vegetation" in h.hazard_type]
        if brush_hazards and vision_result.brush_clearance_feet and vision_result.brush_clearance_feet < 15.0:
            contradictions.append(
                ContradictionFlag(
                    flag_id=f"contra_{uuid.uuid4().hex[:6]}",
                    field_name="defensible_space_vegetation",
                    document_claim="Standard residential lot maintenance presumed.",
                    visual_evidence=f"Dense brush/tree overhang detected with only {vision_result.brush_clearance_feet}ft clearance from siding.",
                    discrepancy_severity="major",
                    explanation="Wildfire / Wind hazard: Inadequate defensible space creates high embers-to-structure ignition risk.",
                )
            )

        # Compile HITL reasons
        hitl_reasons = list(doc_result.hitl_review_reasons)
        if any(c.discrepancy_severity == "critical" for c in contradictions):
            hitl_reasons.append("Critical contradiction between documentary claims and physical image evidence.")

        return UnifiedRiskProfile(
            submission_id=submission_id,
            year_built=year_built,
            effective_building_age=building_age,
            roof_installed_year=roof_year,
            roof_age_years=roof_age,
            roof_condition_visual=visual_roof_cond,
            square_footage=sq_ft,
            construction_type=construction,
            flood_zone=flood_zone,
            prior_claims_count=claims_count,
            prior_claims_amount=claims_amt,
            visual_hazards=vision_result.hazards,
            contradictions=contradictions,
            hitl_required=(len(hitl_reasons) > 0),
            hitl_reasons=hitl_reasons,
        )
