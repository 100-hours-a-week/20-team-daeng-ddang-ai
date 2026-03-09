#!/usr/bin/env bash
set -euo pipefail

# Sends 2 curl requests per feature endpoint:
# missions, face, healthcare, chatbot
#
# Usage:
#   bash scripts/curl_smoke_twice.sh
#   BASE_URL=http://localhost:8000 bash scripts/curl_smoke_twice.sh

BASE_URL="${BASE_URL:-http://localhost:8000}"

post_json() {
  local name="$1"
  local url="$2"
  local payload="$3"

  echo ""
  echo "== ${name} -> ${url}"
  for i in 1 2; do
    echo "-- request ${i}"
    curl -sS -X POST "${url}" \
      -H "Content-Type: application/json" \
      -d "${payload}" \
      -w "\n[http_status=%{http_code} total=%{time_total}s]\n"
  done
}

MISSIONS_PAYLOAD='{
  "analysis_id": "curl-missions-001",
  "walk_id": 1,
  "missions": [
    {
      "mission_id": 1,
      "mission_type": "PAW",
      "video_url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/PAW_03.mp4"
    }
  ]
}'

FACE_PAYLOAD='{
  "analysis_id": "curl-face-001",
  "video_url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/face_test/testVideo01.mp4"
}'

HEALTHCARE_PAYLOAD='{
  "analysis_id": "curl-health-001",
  "dog_id": 123,
  "video_url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/healthcare_test/testVideo02.mp4"
}'

CHATBOT_PAYLOAD='{
  "dog_id": 1,
  "conversation_id": "curl-chat-001",
  "message": { "role": "user", "content": "강아지가 물을 많이 마셔요. 병원 가야 하나요?" },
  "image_url": null,
  "user_context": null,
  "history": []
}'

post_json "missions"   "${BASE_URL}/api/missions/judge"     "${MISSIONS_PAYLOAD}"
post_json "face"       "${BASE_URL}/api/face/analyze"       "${FACE_PAYLOAD}"
post_json "healthcare" "${BASE_URL}/api/healthcare/analyze" "${HEALTHCARE_PAYLOAD}"
post_json "chatbot"    "${BASE_URL}/api/vet/chat"           "${CHATBOT_PAYLOAD}"

echo ""
echo "done."
