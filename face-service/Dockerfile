# Face Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (FFmpeg is required)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (default 8100)
EXPOSE 8100

# Command to run the application
CMD ["python", "run.py"]
