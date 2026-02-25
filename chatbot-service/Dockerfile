# Chatbot Service Dockerfile
# NOTE: GPU 서버에서 실행 시 아래 베이스 이미지를 변경하세요.
#   예) FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
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
