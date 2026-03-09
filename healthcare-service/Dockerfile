# Healthcare Service Dockerfile
FROM python:3.11-slim

# NVIDIA GPU/CUDA 지원을 위한 필수 환경 변수 설정
# 클라우드 배포 시 GPU 드라이버 마운트를 위해 필요합니다.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Create non-root user
RUN addgroup --system app \
    && adduser --system --ingroup app --home /app app

# System deps (OpenCV-headless를 쓰지만 안정성을 위해 추가 기능 허용)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# GPU 서버에서 PyTorch CUDA 버전을 제대로 당겨오도록 설정하고 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/models /app/output \
    && chown -R app:app /app

EXPOSE 8200

USER app

# Container-level health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8200/health', timeout=3)"

CMD ["python", "run.py"]
