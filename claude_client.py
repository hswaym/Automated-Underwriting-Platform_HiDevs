"""Claude Haiku client — structured underwriting decision via JSON-mode."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import anthropic

from config import settings
from schema import (
    ComplianceCheckResult,
    DetectedHazard,
    RiskScoreBreakdown,
    UnderwritingDecision,
    UnifiedRiskProfile,
)

logger = logging.getLogger(__name__)

import os

def _get_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        key = getattr(settings, "anthropic_api_key", "")
    return key.strip()


import llm_client

def is_available() -> bool:
    """Always available via open-source AI engine."""
    return True


def enrich_decision(
    profile: UnifiedRiskProfile,
    risk_breakdown: RiskScoreBreakdown,
    compliance: ComplianceCheckResult,
    base_decision: UnderwritingDecision,
    document_text: str = "",
) -> UnderwritingDecision:
    """Forwards to the free open-source LLM client."""
    return llm_client.enrich_decision(
        profile=profile,
        risk_breakdown=risk_breakdown,
        compliance=compliance,
        base_decision=base_decision,
        document_text=document_text,
    )


# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM = """\
You are an expert commercial property underwriter. You receive structured risk data
that has already been extracted and scored by a deterministic pipeline.

Your job is to:
1. Write a clear, professional 3-5 sentence SUMMARY that explains the decision rationale
   in plain English — suitable for a broker or senior underwriter.
2. Produce a list of REQUIRED_ACTIONS the underwriter must complete before binding.
3. Flag any additional NARRATIVE_HAZARDS you observe in the text that the rule engine
   may have missed (e.g. unusual occupancy, maintenance language, liability wording).

Return ONLY a JSON object with these exact keys:
{
  "summary": "<string>",
  "required_actions": ["<string>", ...],
  "narrative_hazards": [
    {"name": "<string>", "severity": "low|medium|high|critical", "description": "<string>"}
  ]
}

Be concise. Do not repeat information already captured in violations or hazard flags.
Do not refuse or add preamble — return the JSON object immediately.\
"""


def enrich_decision(
    profile: UnifiedRiskProfile,
    risk_breakdown: RiskScoreBreakdown,
    compliance: ComplianceCheckResult,
    base_decision: UnderwritingDecision,
    document_text: str = "",
) -> UnderwritingDecision:
    """
    Calls claude-haiku-3-5-latest to enrich the deterministic decision with:
    - Professional narrative summary
    - LLM-spotted narrative hazards
    - Refined required actions list

    Falls back to the base deterministic decision on any API error.
    """
    client = _get_client()

    # Build a compact JSON context block for Haiku — keeps tokens low
    ctx: Dict[str, Any] = {
        "decision": base_decision.decision,
        "risk_score": base_decision.risk_score,
        "risk_tier": base_decision.risk_tier,
        "confidence": base_decision.confidence,
        "violations": [
            {"citation": v.citation, "reason": v.reason, "action": v.action}
            for v in compliance.violations
        ],
        "visual_hazards": [
            {"type": h.hazard_type, "severity": h.severity, "description": h.description}
            for h in profile.visual_hazards
        ],
        "contradictions": [
            {"field": c.field_name, "severity": c.discrepancy_severity, "explanation": c.explanation}
            for c in profile.contradictions
        ],
        "risk_pillars": {
            "structural": getattr(risk_breakdown, "structural_score", 0),
            "environmental": getattr(risk_breakdown, "environmental_score", 0),
            "maintenance": getattr(risk_breakdown, "maintenance_score", 0),
            "claims": getattr(risk_breakdown, "claims_score", 0),
            "liability": getattr(risk_breakdown, "liability_score", 0),
            "contradiction_penalty": getattr(risk_breakdown, "contradiction_penalty", 0),
        },
        "construction_type": profile.construction_type,
        "year_built": profile.year_built,
        "flood_zone": profile.flood_zone,
        "prior_claims": profile.prior_claims_count,
    }

    # Add a trimmed excerpt of raw document text (max 2000 chars) for narrative context
    doc_excerpt = (document_text[:2000] + "…") if len(document_text) > 2000 else document_text

    user_message = f"""PIPELINE OUTPUT:
{json.dumps(ctx, indent=2)}

DOCUMENT EXCERPT:
{doc_excerpt}

Provide the JSON enrichment now."""

    logger.info(
        "Calling %s for enrichment of submission %s (score=%d, decision=%s)",
        settings.model,
        base_decision.submission_id,
        base_decision.risk_score,
        base_decision.decision,
    )

    try:
        response = client.messages.create(
            model=settings.model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        enrichment = json.loads(raw_text)

    except anthropic.APIStatusError as exc:
        logger.warning("Haiku API error (%s) — using deterministic decision: %s", exc.status_code, exc.message)
        return base_decision
    except (json.JSONDecodeError, IndexError, KeyError) as exc:
        logger.warning("Haiku returned malformed response — using deterministic decision: %s", exc)
        return base_decision
    except Exception as exc:
        logger.warning("Unexpected error during Haiku enrichment (%s) — using deterministic decision: %s", type(exc).__name__, exc)
        return base_decision

    # ── Merge enrichment into the base decision ────────────────────────────────
    # 1. Replace summary with Haiku's narrative
    enriched_summary = enrichment.get("summary", base_decision.summary_rationale)
    # Append the deterministic violations log below the narrative
    if compliance.violations:
        enriched_summary += "\n\nGuideline violations flagged:\n"
        for v in compliance.violations:
            enriched_summary += f" • [{v.action}] {v.citation}: {v.reason}\n"

    # 2. Merge required actions (deduplicated)
    haiku_actions = enrichment.get("required_actions", [])
    base_actions = base_decision.required_underwriter_actions or []
    merged_actions = list({a: None for a in (haiku_actions + base_actions)}.keys())

    # 3. Append narrative hazards Haiku found as additional DetectedHazard entries
    existing_hazards = list(base_decision.detected_hazards or [])
    for nh in enrichment.get("narrative_hazards", []):
        existing_hazards.append(
            DetectedHazard(
                hazard_type=nh.get("name", "narrative_flag").lower().replace(" ", "_"),
                severity=nh.get("severity", "medium"),
                description=nh.get("description", ""),
                source_image="document_text",
                confidence=0.80,
            )
        )

    logger.info(
        "Haiku enriched submission %s: %d narrative hazards added, summary updated.",
        base_decision.submission_id,
        len(enrichment.get("narrative_hazards", [])),
    )

    # Return updated decision (Pydantic model_copy with overrides)
    return base_decision.model_copy(update={
        "summary_rationale": enriched_summary,
        "required_underwriter_actions": merged_actions,
        "detected_hazards": existing_hazards,
    })
