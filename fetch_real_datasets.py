"""Downloads authentic, real-life datasets for property underwriting."""

import json
import urllib.request
from pathlib import Path

sample_dir = Path("sample_data")
sample_dir.mkdir(exist_ok=True)

print("Fetching authentic real-world underwriting datasets...\n")

# 1. Fetch Real FEMA Flood Insurance Claims
print("1. Querying FEMA National Flood Insurance Program (NFIP) Open API...")
fema_url = "https://www.fema.gov/api/open/v2/FimaNfipClaims?%24top=5&%24format=json"
req = urllib.request.Request(fema_url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=12) as response:
        fema_data = json.loads(response.read().decode())
        claims = fema_data.get("FimaNfipClaims", [])
        out_fema = sample_dir / "real_fema_claims.json"
        out_fema.write_text(json.dumps(claims, indent=2))
        print(f"   [SUCCESS] Downloaded {len(claims)} authentic FEMA claims -> {out_fema}")
        if claims:
            c0 = claims[0]
            print(f"   Example: Year {c0.get('yearOfLoss')}, State: {c0.get('state')}, Damage: ${c0.get('buildingDamageAmount')}")
except Exception as e:
    print(f"   [ERROR] FEMA query failed: {e}")

# 2. Fetch Real City Building & Roofing Permits
print("\n2. Querying City of Austin Open Data (Building & Roofing Permits)...")
austin_url = "https://data.austintexas.gov/resource/3syk-w9eu.json?%24limit=5"
req = urllib.request.Request(austin_url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=12) as response:
        permits = json.loads(response.read().decode())
        out_permits = sample_dir / "real_austin_permits.json"
        out_permits.write_text(json.dumps(permits, indent=2))
        print(f"   [SUCCESS] Downloaded {len(permits)} authentic city permits -> {out_permits}")
        if permits:
            p0 = permits[0]
            print(f"   Example: Permit #{p0.get('permit_number')}, Type: {p0.get('permit_type_desc')}, Address: {p0.get('original_address1')}")
except Exception as e:
    print(f"   [ERROR] Austin Open Data query failed: {e}")

print("\nAll datasets saved in sample_data/.")
