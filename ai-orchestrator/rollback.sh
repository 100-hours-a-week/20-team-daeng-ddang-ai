#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/app/ai"
COMPOSE_FILE="${APP_DIR}/docker-compose.yml"
ENV_FILE="${APP_DIR}/.env"
BACKUP_FILE="${APP_DIR}/.backup_image" # 롤백용 이미지:태그가 저장되는 파일

# GitHub Actions에서 넘겨주는 값
ENV_FILE_B64="${ENV_FILE_B64:-}"
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-}"
DOCKERHUB_TOKEN="${DOCKERHUB_TOKEN:-}"

echo "🌐 AI Orchestrator 롤백을 시작합니다."

# 1. 파일 존재 확인
if [ ! -f "${COMPOSE_FILE}" ] || [ ! -f "${BACKUP_FILE}" ]; then
  echo "❌ 롤백에 필요한 파일이 서버에 없습니다."
  exit 1
fi

# 2. 롤백 대상 이미지 정보 읽기
if [ -f "${BACKUP_FILE}" ]; then
  ROLLBACK_IMAGE=$(cat "${BACKUP_FILE}" | tr -d '\r\n[:space:]')
fi

if [ -z "${ROLLBACK_IMAGE:-}" ]; then
  echo "❌ 롤백 이미지 정보가 비어있습니다."
  exit 1
fi

echo "🔙 복구 대상 이미지: [${ROLLBACK_IMAGE}]"
echo "🔍 [Debug] Raw string length: ${#ROLLBACK_IMAGE}"

cd "${APP_DIR}"

# 3. 중복 롤백 방지 및 현재 이미지 기록
CURRENT_IMAGE=""
if docker inspect ai-orchestrator >/dev/null 2>&1; then
  CURRENT_IMAGE="$(docker inspect -f '{{.Config.Image}}' ai-orchestrator 2>/dev/null || true)"

  if [ "${CURRENT_IMAGE}" == "${ROLLBACK_IMAGE}" ]; then
    echo "⏩ 현재 실행 중인 이미지와 롤백하려는 이미지가 동일합니다. 중단합니다."
    exit 0
  fi
fi

# 4. .env 생성 및 교체
umask 077
printf "%s" "${ENV_FILE_B64}" | base64 -d > "${ENV_FILE}"
echo "🔐 ${ENV_FILE} 작성 (mode 600)"

# 5. Docker Hub 로그인
if [ -n "${DOCKERHUB_USERNAME}" ] && [ -n "${DOCKERHUB_TOKEN}" ]; then
  echo "🔑 docker login to Docker Hub..."
  echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USERNAME}" --password-stdin
else
  echo "ℹ️ DOCKERHUB_USERNAME 혹은 TOKEN가 빈 값입니다. 이미 로그인된 상태라고 가정합니다."
fi

# 6. 실행
echo "📦 docker compose pull..."
DOCKER_IMAGE="${ROLLBACK_IMAGE}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" pull

echo "🚀 docker compose up -d..."
DOCKER_IMAGE="${ROLLBACK_IMAGE}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d

echo "📋 docker compose ps"
docker compose -f "${COMPOSE_FILE}" ps

# 7. 헬스 체크
echo "🏥 헬스 체크 시작..."

ok=0
for i in {1..10}; do
  sleep 5
  if wget -qO- http://localhost:8000/health >/dev/null 2>&1; then
    ok=1
    echo "✅ 헬스 체크 성공!"
    break
  fi
  echo "⏳ 시작 대기 중.... ($i/10)"
done

# 8. 사후 처리
if [ "$ok" -eq 1 ]; then
  # 롤백 성공 시: 이제 '롤백된 현재 이미지'를 나중을 위해 다시 백업으로 둘지 결정
  # (보통 롤백 스크립트에서는 성공 시 별도 처리를 안 하거나, 
  #  실패했던 이미지를 백업에 넣지 않도록 주의해야 합니다.)
  echo "🎉 롤백 배포 성공"
else
  echo "🚨 롤백 버전조차 헬스 체크에 실패했습니다!"
  # 롤백의 롤백은 위험하므로 여기서 중단하거나 수동 개입 필요
  exit 1
fi

echo "🧹 prune old images"
docker image prune -f >/dev/null 2>&1 || true
