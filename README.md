# AegisIQ · AI-Powered Automated Property Underwriting Platform

Multimodal P&C Commercial & Residential Property Underwriting Platform combining:
- **Document Intelligence**: Field extraction, OCR, confidence calibration, and human-in-the-loop triggers.
- **Computer Vision**: Hazard recognition, visual roof condition scoring, and defensible space assessment.
- **Multimodal Cross-Modal Fusion**: Text-versus-photo contradiction audits.
- **Actuarial Risk Scoring**: Multi-factor quantitative scoring (0–100) and tiering (Preferred, Standard, Non-Standard).
- **Compliance & Guideline Engine**: Hard-stop and referral enforcement against underwriting appetite guidelines.
- **Zero-Cost Open-Source AI**: Enriched decision rationales powered by Groq (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`) with offline fallback.

---

## Quick Start

### 1. Environment Configuration
Copy the template and provide your Groq API key:
```bash
cp .env.example .env
# Edit .env and paste your free Groq API key (from https://console.groq.com/keys)
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000/dashboard` to access the interactive underwriting cockpit.

---

## Testing
Run the comprehensive test suite (29 tests across all 8 pipeline stages):
```bash
pytest
```
