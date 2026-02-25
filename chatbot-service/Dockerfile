# Chatbot Service Dockerfile
FROM python:3.11-slim

# NVIDIA GPU/CUDA 지원을 위한 환경 변수 (LLM GPU 가속 시 필수)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# LLM 구동 시 시스템에서 종종 필요한 기본 C/C++ 런타임 설치 지원
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (GPU용 PyTorch가 우선 설치되도록 .txt내용이 수정됨)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# models/ 디렉토리는 별도 볼륨 마운트 또는 HuggingFace Hub에서 다운로드
# (lora-qwen-7b-final, chroma_db 포함)
VOLUME ["/app/models"]

# Expose port (default 8300)
EXPOSE 8300

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8300"]
