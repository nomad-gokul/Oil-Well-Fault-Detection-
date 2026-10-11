#!/usr/bin/env bash
# Smoke test a deployed API (and optionally the dashboard).
#   scripts/smoke_test.sh https://owfd-api.onrender.com [https://<app>.streamlit.app]
set -euo pipefail
API="${1:?usage: smoke_test.sh API_URL [DASHBOARD_URL]}"; API="${API%/}"
DASH="${2:-}"
pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; exit 1; }

echo "API: $API (free tiers may need up to a minute to wake up)"
for i in $(seq 1 30); do curl -sf "$API/health" >/dev/null && break; sleep 3; done
health=$(curl -sf "$API/health") || fail "GET /health"
pass "GET /health -> $health"
curl -sf "$API/model" | grep -q '"classes"' && pass "GET /model" || fail "GET /model"
curl -sf -o /dev/null "$API/docs" && pass "GET /docs (OpenAPI UI)" || fail "GET /docs"

payload=$(python3 - <<'PY'
import json, math
t0 = 1_900_000_000
rows = [{"timestamp": f"2030-03-17T{(i // 3600):02d}:{(i // 60) % 60:02d}:{i % 60:02d}",
         "values": {"P-PDG": 2.2e7 + 1e3 * math.sin(i / 30), "P-TPT": 1.4e7, "T-TPT": 101.0,
                    "P-MON-CKP": 1.44e6, "T-JUS-CKP": 76.7, "P-JUS-CKGL": None, "QGL": None}}
        for i in range(300)]
print(json.dumps({"well_id": "smoke", "readings": rows}))
PY
)
code=$(curl -s -o /tmp/owfd_predict.json -w '%{http_code}' -X POST "$API/predict" \
  -H 'content-type: application/json' -d "$payload")
[ "$code" = 200 ] && grep -q '"p_fault"' /tmp/owfd_predict.json && pass "POST /predict (300 readings)" || fail "POST /predict -> $code"

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API/ingest" -H 'content-type: application/json' \
  -d '{"well_id":"smoke","readings":[{"timestamp":"2030-01-01T00:00:00","values":{}}]}')
[ "$code" = 401 ] && pass "POST /ingest without API key is rejected (401)" || fail "POST /ingest without key -> $code (expected 401)"

curl -sf "$API/demo" | grep -q '"running":true' && pass "demo replay running" || echo "  INFO demo replay not running (OWFD_DEMO_MODE off?)"
curl -sf "$API/metrics" | grep -q owfd_model_info && pass "GET /metrics" || fail "GET /metrics"

if [ -n "$DASH" ]; then
  code=$(curl -s -o /dev/null -w '%{http_code}' -L "${DASH%/}")
  [ "$code" = 200 ] && pass "dashboard $DASH" || fail "dashboard -> $code"
fi
echo "All smoke tests passed."
