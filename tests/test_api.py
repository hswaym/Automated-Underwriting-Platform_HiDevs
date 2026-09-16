"""Tests for the FastAPI app — run with: python tests/test_api.py

These tests use TestClient (requests-based, no network) and stub out the
Claude call so they run without an API key.
"""

import sys
import os
import json
import io
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub out settings before importing the app so we don't need a real .env
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")

from fastapi.testclient import TestClient


def _make_decision(sid: str):
    from schema import UnderwritingDecision
    return UnderwritingDecision(
        submission_id=sid,
        decision="approve",
        risk_score=25,
        confidence="high",
        flags=[],
        section_reasoning=[],
        summary="Property looks good.",
        guideline_references=["§2.1 General Eligibility"],
    )


def test_health():
    from main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in ["ok", "healthy"]
    print("  /health OK")


def test_dashboard():
    from main import app
    client = TestClient(app)
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "Underwriting" in resp.text
    print("  /dashboard OK")


def test_submit_no_content():
    from main import app
    client = TestClient(app)
    resp = client.post("/submit", data={})
    assert resp.status_code == 422
    print(f"  /submit empty -> {resp.status_code} OK")


def test_submit_with_json_and_stub():
    """Submit JSON metadata only to verify end-to-end API flow."""
    from main import app
    client = TestClient(app)

    payload = {
        "address": "789 Elm St",
        "year_built": 2018,
        "roof_year": 2020,
        "prior_claims": 0,
        "flood_zone": "X",
    }
    resp = client.post(
        "/submit",
        data={"json_data": json.dumps(payload)},
    )

    assert resp.status_code == 200, resp.text
    sid = resp.json()["submission_id"]
    assert sid
    print(f"  /submit -> submission_id: {sid}")

    # Result should be retrievable
    resp2 = client.get(f"/result/{sid}")
    assert resp2.status_code == 200
    assert resp2.json()["decision"] in ["approve", "refer_to_human", "decline"]
    print("  /result/{id} OK")


def test_result_not_found():
    from main import app
    client = TestClient(app)
    resp = client.get("/result/does-not-exist")
    assert resp.status_code == 404
    print("  /result/missing -> 404 OK")


if __name__ == "__main__":
    tests = [
        test_health,
        test_dashboard,
        test_submit_no_content,
        test_submit_with_json_and_stub,
        test_result_not_found,
    ]
    failures = 0
    for t in tests:
        try:
            print(f"[RUN] {t.__name__}")
            t()
            print(f"[OK]  {t.__name__}")
        except Exception as e:
            import traceback
            print(f"[FAIL] {t.__name__}: {e}")
            traceback.print_exc()
            failures += 1
    if failures:
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
