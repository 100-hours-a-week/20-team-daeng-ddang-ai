#!/bin/bash
set -e

echo "=== Chatbot Service 환경 설정 ==="

# Python 가상환경 생성 (없으면)
if [ ! -d ".venv" ]; then
    echo "[1/3] 가상환경 생성 중..."
    python3 -m venv .venv
fi

# 가상환경 활성화
echo "[2/3] 가상환경 활성화 중..."
source .venv/bin/activate

# 의존성 설치
echo "[3/3] 의존성 설치 중 (requirements.txt)..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ 환경 설정 완료!"
echo ""
echo "🚀 서버 실행 방법:"
echo "   source .venv/bin/activate"
echo "   python run.py"
echo ""
echo "📦 모델 다운로드 방법:"
echo "   1. LoRA Adapter: huggingface.co/20-team-daeng-ddang-ai/vet-chat"
echo "      → models/lora-qwen-7b-final/ 에 배치"
echo "   2. Vector DB:    huggingface.co/20-team-daeng-ddang-ai/vet-chat"
echo "      → models/chroma_db/ 에 배치"
