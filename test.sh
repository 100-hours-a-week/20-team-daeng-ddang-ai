#!/bin/bash
# 분석 서버 주소
API_URL="http://localhost:8000/api/face/analyze"

# 테스트할 비디오 URL 리스트
URLS=(
    "https://daeng-map.s3.ap-northeast-2.amazonaws.com/face_test/angry05.mp4"
)

echo "🚀  강아지 감정 분석 테스트 시작..."
echo "--------------------------------------"

for url in "${URLS[@]}"; do
    # 확장자에 관계없이 파일명만 추출 (.mp4, .mov 등 모두 처리)
    filename=$(basename "$url")
    filename="${filename%.*}"  # 모든 확장자 제거

    analysis_id="face-test-$filename"

    echo "▶️ 분석 중: $filename ($url)"

    # CURL 요청 실행
    curl -s -X POST "$API_URL" \
         -H "Content-Type: application/json" \
         -d "{\"analysis_id\": \"$analysis_id\", \"video_url\": \"$url\", \"options\": {}}" | json_pp

    echo "--------------------------------------"
    sleep 1
done

echo "✅  모든 테스트가 완료되었습니다."
