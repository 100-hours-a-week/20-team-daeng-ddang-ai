# Healthcare Service Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Create non-root user
RUN addgroup --system app \
    && adduser --system --ingroup app --home /app app

# System deps
RUN apt-get update && apt-get install -y \
    python3-pip \
    ffmpeg \
    libavcodec-extra \
    libx264-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

COPY . .
RUN mkdir -p /app/models /app/output \
    && chown -R app:app /app

EXPOSE 8200

USER app

# Container-level health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8200/health', timeout=3)"

CMD ["python", "run.py"]
