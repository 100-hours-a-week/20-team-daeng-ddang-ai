#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/app/ai"
COMPOSE_FILE="${APP_DIR}/docker-compose.yml"
ENV_FILE="${APP_DIR}/.env"
BACKUP_FILE="${APP_DIR}/.backup_image" # 롤백용 이미지:태그가 저장되는 파일

# GitHub Actions에서 넘겨주는 값
IMAGE="${IMAGE:-}"
RELEASE_ID="${RELEASE_ID:-}"
ENV_FILE_B64="${ENV_FILE_B64:-}"
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-}"
DOCKERHUB_TOKEN="${DOCKERHUB_TOKEN:-}"

if [ -z "${IMAGE}" ] || [ -z "${RELEASE_ID}" ] || [ -z "${ENV_FILE_B64}" ]; then
  echo "❌ IMAGE, RELEASE_ID, ENV_FILE_B64은 필수 값입니다."
  exit 1
fi

echo "🌐 AI Orchestrator deploy (docker compose)"
echo "🏷️ RELEASE_ID=${RELEASE_ID}"

cd "${APP_DIR}"
test -f "${COMPOSE_FILE}"

DOCKER_IMAGE="${IMAGE}:${RELEASE_ID}"

# 1. 롤백용 현재 이미지 기록
CURRENT_IMAGE=""
if docker inspect ai-orchestrator >/dev/null 2>&1; then
  CURRENT_IMAGE="$(docker inspect -f '{{.Config.Image}}' ai-orchestrator 2>/dev/null || true)"

  if [ "${CURRENT_IMAGE}" == "${DOCKER_IMAGE}" ]; then
    echo "⏩ 현재 실행 중인 이미지와 배포하려는 이미지가 동일합니다. 배포를 중단합니다."
    exit 0 
  fi
fi

# 2. .env 생성 및 교체
umask 077
printf "%s" "${ENV_FILE_B64}" | base64 -d > "${ENV_FILE}"
echo "🔐 ${ENV_FILE} 작성 (mode 600)"

# 3. Docker Hub 로그인
if [ -n "${DOCKERHUB_USERNAME}" ] && [ -n "${DOCKERHUB_TOKEN}" ]; then
  echo "🔑 docker login to Docker Hub..."
  echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USERNAME}" --password-stdin
else
  echo "ℹ️ DOCKERHUB_USERNAME 혹은 TOKEN가 빈 값입니다. 이미 로그인된 상태라고 가정합니다."
fi

# 4. 배포
echo "📦 docker compose pull..."
DOCKER_IMAGE="${DOCKER_IMAGE}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" pull

echo "🚀 docker compose up -d..."
DOCKER_IMAGE="${DOCKER_IMAGE}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d

echo "📋 docker compose ps"
docker compose -f "${COMPOSE_FILE}" ps

# 5. 헬스 체크
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

if [ "$ok" -eq 1 ]; then
  # 신규 이미지가 제대로 배포되면 직전 이미지를 백업용 이미지로 저장
  if [ -n "${CURRENT_IMAGE}" ]; then
    echo "${CURRENT_IMAGE}" > "${BACKUP_FILE}"
    echo "💾 백업 이미지 업데이트 완료: ${CURRENT_IMAGE}"
  fi
else
  echo "🚨 헬스 체크 실패. 배포를 중단하고 자동 롤백을 시도합니다."

  if [ -n "${CURRENT_IMAGE}" ]; then
    echo "🔙 직전 버전(${CURRENT_IMAGE})으로 롤백합니다."

    DOCKER_IMAGE="${CURRENT_IMAGE}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" pull || true
    DOCKER_IMAGE="${CURRENT_IMAGE}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d
    echo "⚠️ 롤백 완료. 제대로 롤백됐는지 확인해주세요."
  else
    echo "❌ 롤백할 직전 버전을 찾을 수 없습니다."
  fi
  exit 1
fi

echo "🧹 prune old images"
docker image prune -f >/dev/null 2>&1 || true

echo "🎉 배포 성공"