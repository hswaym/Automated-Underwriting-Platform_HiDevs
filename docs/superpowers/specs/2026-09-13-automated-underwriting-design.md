# Automated Underwriting Platform — Design Spec

**Date:** 2026-09-13  
**Industry:** Insurance (Property)  
**Stack:** Python 3.13, FastAPI, Claude Opus 5 (Anthropic SDK 0.97.0)

---

## 1. Problem Statement

Build an AI-powered system that accepts a property insurance submission
(PDF inspection reports, property photos, scanned/handwritten forms, and
structured JSON data) and produces an underwriting decision (Approve /
Decline / Refer) with a risk score, flagged hazards, and per-section
reasoning — presented in a browser dashboard.

The system must:
- Extract relevant data from all four input types
- Apply a user-supplied underwriting guideline document as the rule source
- Perform multimodal analysis (text + images) in a single Claude call
- Display results in a web dashboard with upload form + results panel

---

## 2. Architecture

```
Submission (PDF + images + scanned forms + JSON)
         │
         ▼
  [Ingestion & Preprocessing Layer]
  ┌─────────────────────────────────────────┐
  │  ingestion.py                           │
  │  • PDF  → pypdf text extraction         │
  │  • Scanned/image PDF → pytesseract OCR  │
  │  • Images → base64 (Pillow)             │
  │  • JSON → validated Pydantic model      │
  │  preprocessing.py                       │
  │  • Noise removal (whitespace, headers)  │
  │  • Text normalization                   │
  │  • Section detection                    │
  └─────────────────────────────────────────┘
         │
         ▼
  [Claude Opus 5 — Single Streaming Call]
  ┌─────────────────────────────────────────┐
  │  claude_client.py                       │
  │  System prompt: guideline document      │
  │  User message:                          │
  │    • All extracted + preprocessed text  │
  │    • Base64 images inline               │
  │  output_config.format: json_schema      │
  └─────────────────────────────────────────┘
         │
         ▼
  [Risk Engine + Decision Layer]
  ┌─────────────────────────────────────────┐
  │  schema.py  — Pydantic output models    │
  │  risk_engine.py — parse + validate      │
  │  Decision: Approve / Decline / Refer    │
  │  Risk score: 0–100                      │
  │  Flags: list of named hazards           │
  │  Reasoning: per-section explanation     │
  └─────────────────────────────────────────┘
         │
         ▼
  [FastAPI + Jinja2 Dashboard]
  ┌─────────────────────────────────────────┐
  │  main.py — FastAPI app                  │
  │  POST /submit — multipart upload        │
  │  GET  /result/{id} — JSON result        │
  │  GET  /dashboard — HTML dashboard       │
  │  templates/dashboard.html               │
  │  static/style.css                       │
  └─────────────────────────────────────────┘
```

---

## 3. Components

### 3.1 Ingestion (`ingestion.py`)

Accepts a `Submission` — a collection of uploaded files plus optional
JSON metadata — and converts everything to a normalized `ProcessedSubmission`.

**Inputs:**
- `pdf_files: list[UploadFile]` — inspection reports, forms
- `image_files: list[UploadFile]` — property photos
- `json_data: dict | None` — structured policy/property data

**Outputs (`ProcessedSubmission`):**
```python
class ProcessedSubmission:
    submission_id: str        # UUID
    text_chunks: list[TextChunk]   # extracted + preprocessed text
    images: list[ImageData]        # base64-encoded images
    structured_data: dict | None   # parsed JSON
```

**PDF strategy:**
1. Extract text with `pypdf.PdfReader` — if a page yields < 50 chars of
   text, treat it as a scanned page and fall back to pytesseract OCR on
   that page only.
2. Never OCR pages that already have extractable text.

**Image strategy:**
- Accept JPG, PNG, WEBP, GIF
- Re-encode via Pillow to JPEG at quality=85 to reduce base64 payload
- Max dimension: 1568px (Claude vision native limit) — resize if larger

**JSON strategy:**
- Validate against `PropertyData` Pydantic model
- Convert to a human-readable key-value string for inclusion in the
  text portion of the Claude message

### 3.2 Preprocessing (`preprocessing.py`)

Cleans and normalizes text from ingested documents before sending to Claude.

**Operations (applied per TextChunk):**
1. **Whitespace normalization** — collapse multiple spaces/newlines, strip
   leading/trailing whitespace per line
2. **Header/footer noise removal** — drop lines that are purely page
   numbers (`^\s*\d+\s*$`), repeated headers (duplicate across pages)
3. **Section detection** — heuristic: lines that are ALL CAPS or end with
   `:` with < 60 chars → mark as section header
4. **Length guard** — if a single chunk exceeds 3000 tokens (estimated
   ~12,000 chars), split at paragraph boundaries; the Claude prompt
   can handle the full context but splitting keeps structure clear

**Output:** cleaned `TextChunk` objects with `source`, `section_headers`,
and `content` fields.

### 3.3 Guidelines Loader (`guidelines.py`)

Loads the underwriting guideline document once at startup. Supports:
- `.txt` — read as-is
- `.md` — read as-is
- `.pdf` — extract via pypdf (guidelines assumed to be text-based PDFs,
  no OCR needed)

Cached as a module-level string. Injected into the Claude system prompt.
Path configured via `GUIDELINES_PATH` env var.

### 3.4 Claude Client (`claude_client.py`)

Single streaming call per submission. Produces a structured JSON decision.

**Model:** `claude-opus-5`  
**Thinking:** `{type: "adaptive"}` (on by default for Opus 5, not configured explicitly)  
**Output format:** `output_config.format` with JSON schema matching `UnderwritingDecision`  
**Streaming:** yes — `client.messages.stream()` + `.get_final_message()`

**System prompt structure:**
```
You are an expert insurance underwriter. Apply the following guidelines
exactly when assessing each submission.

=== UNDERWRITING GUIDELINES ===
{guidelines_text}
=== END GUIDELINES ===

Analyze the submission and return a JSON decision conforming exactly to
the provided schema. Be precise and cite specific document sections when
flagging hazards.
```

**User message structure:**
```
[Text block 1: structured property data]
[Text block 2: document section 1 text]
...
[Text block N: document section N text]
[Image block 1: base64 image]
...
[Image block M: base64 image]
[Text block: instruction to produce the JSON decision]
```

**Refusal handling:** check `stop_reason` before reading content; if
`stop_reason == "refusal"`, surface an error result with
`decision: "refer"` and a flag named `"claude_refusal"`.

### 3.5 Schema (`schema.py`)

Pydantic models for both input and output.

**Input:**
```python
class PropertyData(BaseModel):
    address: str | None
    year_built: int | None
    construction_type: str | None
    square_footage: float | None
    occupancy_type: str | None
    # ...additional fields optional
    extra: dict = {}

class TextChunk(BaseModel):
    source: str            # filename
    section_headers: list[str]
    content: str

class ImageData(BaseModel):
    source: str            # filename
    media_type: str        # image/jpeg
    data: str              # base64

class ProcessedSubmission(BaseModel):
    submission_id: str
    text_chunks: list[TextChunk]
    images: list[ImageData]
    structured_data: dict | None
```

**Output:**
```python
class HazardFlag(BaseModel):
    name: str
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    source: str            # which document/image triggered this

class SectionReasoning(BaseModel):
    section: str
    finding: str

class UnderwritingDecision(BaseModel):
    submission_id: str
    decision: Literal["approve", "decline", "refer"]
    risk_score: int        # 0–100 (100 = maximum risk)
    confidence: Literal["low", "medium", "high"]
    flags: list[HazardFlag]
    section_reasoning: list[SectionReasoning]
    summary: str           # 2–3 sentence overall summary
    guideline_references: list[str]  # which guideline clauses applied
```

### 3.6 Risk Engine (`risk_engine.py`)

Parses Claude's structured JSON output into an `UnderwritingDecision`.
Validates the schema. Stores result in the in-memory result store.

```python
# ponytail: in-memory dict, swap to SQLite when audit trail needed
_results: dict[str, UnderwritingDecision] = {}
```

### 3.7 API (`main.py`)

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Redirect to `/dashboard` |
| `GET` | `/dashboard` | Serve html dashboard (Jinja2) |
| `POST` | `/submit` | Accept multipart upload, return `{submission_id}` |
| `GET` | `/result/{submission_id}` | Return `UnderwritingDecision` JSON |
| `GET` | `/health` | `{"status": "ok"}` |

`POST /submit` is async. It runs ingestion + preprocessing + Claude call
in the request handler. Returns immediately with the submission ID only
after Claude responds (streaming, so the connection stays open through Claude).

For the MVP, this is synchronous from the client's perspective — the HTTP
response is not returned until Claude is done. This is fine for demo
scale. A task queue (Celery/ARQ) is the upgrade path when concurrent
submissions matter.

### 3.8 Dashboard (`templates/dashboard.html`)

Single-page Jinja2 template (no JS framework). Two panels:

**Upload panel:**
- File inputs: PDFs (multiple), Images (multiple), JSON (optional)
- Submit button → POST to `/submit` via fetch()
- Shows spinner while waiting

**Results panel** (populated via fetch to `/result/{id}`):
- Decision badge: green (Approve) / red (Decline) / amber (Refer)
- Risk score gauge (CSS-only, 0–100)
- Confidence indicator
- Flags table: name, severity, source, description
- Section reasoning accordion
- Summary text
- Guideline references cited

### 3.9 Config (`.env` + `config.py`)

```
ANTHROPIC_API_KEY=...
GUIDELINES_PATH=./guidelines/underwriting_guidelines.txt
MODEL=claude-opus-5
MAX_IMAGE_DIMENSION=1568
```

---

## 4. File Map

```
Automated Underwriting Platform/
├── main.py                    # FastAPI app, routes
├── ingestion.py               # PDF, image, JSON, OCR ingestion
├── preprocessing.py           # Text cleaning and normalization
├── guidelines.py              # Guideline loader (startup)
├── claude_client.py           # Anthropic API call
├── schema.py                  # All Pydantic models
├── risk_engine.py             # Result store, parse Claude output
├── config.py                  # Settings from .env
├── templates/
│   └── dashboard.html         # Jinja2 single-page dashboard
├── static/
│   └── style.css              # Dashboard styling
├── guidelines/
│   └── underwriting_guidelines.txt  # Your guideline document (gitignored)
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_schema.py
│   ├── test_risk_engine.py
│   └── test_api.py
├── .env                       # gitignored
├── .env.example
├── requirements.txt
└── docs/
    └── superpowers/
        ├── specs/
        │   └── 2026-09-13-automated-underwriting-design.md
        └── plans/
            └── 2026-09-13-automated-underwriting-plan.md
```

---

## 5. Dependencies

**Already installed (verified):**
- `anthropic==0.97.0`
- `fastapi==0.116.2`
- `uvicorn==0.36.0`
- `pydantic==2.11.9`
- `pypdf==6.16.2`
- `pillow==11.3.0`
- `python-multipart==0.0.20`
- `jinja2==3.1.6`
- `python-dotenv==1.2.1`
- `httpx==0.28.1`

**Needs install:**
- `pytesseract` — OCR fallback for scanned pages
- `aiofiles` — async file I/O in FastAPI

**System dependency for pytesseract:**
- Tesseract OCR binary (Windows: `choco install tesseract` or
  download installer from UB Mannheim)

---

## 6. Error Handling

| Error | Handling |
|-------|----------|
| Unsupported file type | 422 response with message |
| PDF with no extractable text and Tesseract not installed | Skip OCR, include warning in result flags |
| Claude refusal | `decision: "refer"`, flag `"claude_refusal"` |
| Claude API error (4xx/5xx) | 502 response with error detail |
| Invalid JSON guidelines file | Startup failure with clear message |
| Image too large after resize | Log warning, proceed (Pillow handles it) |

---

## 7. Not In Scope (MVP)

- Authentication / user accounts
- Database persistence (in-memory only)
- Async task queue
- PDF report export
- Batch submission processing
- Custom guideline rule editor UI
