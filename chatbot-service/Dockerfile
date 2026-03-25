FROM python:3.10-slim

WORKDIR /app

# Create non-root user
RUN addgroup --system app \
    && adduser --system --ingroup app --home /app app

# Build dependencies for some pip packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN mkdir -p /app/models \
    && chown -R app:app /app

EXPOSE 8300

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8300/health', timeout=3)"

# 시작 시 볼륨 권한 설정 후 app 유저로 전환하여 실행
CMD chown -R app:app /app/models && exec su -s /bin/sh app -c "uvicorn app.main:app --host 0.0.0.0 --port 8300"
