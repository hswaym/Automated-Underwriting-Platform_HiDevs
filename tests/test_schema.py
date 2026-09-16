"""Tests for schema.py — run with: python tests/test_schema.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schema import (
    PropertyData,
    UnderwritingDecision,
    HazardFlag,
    SectionReasoning,
)


def test_property_data_to_text():
    pd = PropertyData(address="456 Oak Ave", year_built=1975, construction_type="Wood Frame")
    text = pd.to_text()
    assert "456 Oak Ave" in text
    assert "1975" in text
    assert "Wood Frame" in text
    print(f"  to_text sample: {text[:80]!r}")


def test_underwriting_decision_schema():
    """json_schema_for_claude() must return a dict with required fields."""
    schema = UnderwritingDecision.json_schema_for_claude()
    assert isinstance(schema, dict)
    # Must have the key fields for structured output
    props = schema.get("properties", {})
    for field in ("decision", "risk_score", "confidence", "flags", "summary"):
        assert field in props, f"Missing field in schema: {field}"
    print(f"  schema keys: {list(props.keys())}")


def test_valid_decision_roundtrip():
    d = UnderwritingDecision(
        submission_id="abc-123",
        decision="approve",
        risk_score=42,
        confidence="high",
        flags=[
            HazardFlag(
                name="Aging Roof",
                severity="medium",
                description="Roof is 25 years old",
                source="inspection.pdf",
            )
        ],
        section_reasoning=[
            SectionReasoning(section="Roof Condition", finding="Original roof, past expected lifespan.")
        ],
        summary="Low-risk property with one moderate flag.",
        guideline_references=["§3.2 Roof Age Threshold"],
    )
    dumped = d.model_dump()
    restored = UnderwritingDecision(**dumped)
    assert restored.submission_id == "abc-123"
    assert restored.risk_score == 42
    assert restored.flags[0].severity == "medium"
    print(f"  roundtrip OK: decision={restored.decision}, score={restored.risk_score}")


def test_risk_score_bounds():
    """risk_score outside 0-100 should raise ValidationError."""
    from pydantic import ValidationError

    try:
        UnderwritingDecision(
            submission_id="x",
            decision="refer",
            risk_score=150,
            confidence="low",
            flags=[],
            section_reasoning=[],
            summary="test",
            guideline_references=[],
        )
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("  ValidationError raised as expected ✓")


if __name__ == "__main__":
    tests = [
        test_property_data_to_text,
        test_underwriting_decision_schema,
        test_valid_decision_roundtrip,
        test_risk_score_bounds,
    ]
    failures = 0
    for t in tests:
        try:
            print(f"[RUN] {t.__name__}")
            t()
            print(f"[OK]  {t.__name__}")
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failures += 1
    if failures:
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
