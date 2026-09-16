"""Tests for risk_engine.py — run with: python tests/test_risk_engine.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_decision(sid: str, score: int = 30):
    from schema import UnderwritingDecision
    return UnderwritingDecision(
        submission_id=sid,
        decision="approve",
        risk_score=score,
        confidence="medium",
        flags=[],
        section_reasoning=[],
        summary="Test summary.",
        guideline_references=[],
    )


def test_store_and_get():
    import risk_engine
    # clear state (module-level dict, re-import is same object in same process)
    risk_engine._results.clear()

    d = _make_decision("test-001")
    risk_engine.store(d)
    retrieved = risk_engine.get("test-001")
    assert retrieved is not None
    assert retrieved.submission_id == "test-001"
    assert retrieved.risk_score == 30
    print("  store/get OK")


def test_get_missing():
    import risk_engine
    risk_engine._results.clear()
    assert risk_engine.get("nonexistent") is None
    print("  get missing → None OK")


def test_all_ids():
    import risk_engine
    risk_engine._results.clear()

    for i in range(3):
        risk_engine.store(_make_decision(f"id-{i}"))

    ids = risk_engine.all_ids()
    assert len(ids) == 3
    assert set(ids) == {"id-0", "id-1", "id-2"}
    print(f"  all_ids: {ids}")


if __name__ == "__main__":
    tests = [test_store_and_get, test_get_missing, test_all_ids]
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
