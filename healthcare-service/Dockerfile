# Healthcare Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps (FFmpeg for video processing, OpenCV runtime)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8200

CMD ["python", "run.py"]
