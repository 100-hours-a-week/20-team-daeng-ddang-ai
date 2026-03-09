#!/bin/bash
set -e

echo "=== vLLM Service 환경 설정 ==="

if [ ! -d ".venv" ]; then
    echo "[1/3] 가상환경 생성 중..."
    python3 -m venv .venv
fi

echo "[2/3] 가상환경 활성화 중..."
source .venv/bin/activate

echo "[3/3] 의존성 설치 중..."
pip install --upgrade pip
pip install "vllm>=0.8.0"

echo ""
echo "✅ 환경 설정 완료!"
echo ""
echo "🚀 서버 실행 방법:"
echo "   source .venv/bin/activate"
echo "   python run.py"
echo ""
echo "🩺 헬스 체크:"
echo "   source .venv/bin/activate"
echo "   python healthcheck.py"
