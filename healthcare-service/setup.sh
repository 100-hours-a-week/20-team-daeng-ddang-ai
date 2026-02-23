#!/bin/bash

echo "=== 1. System Dependency Check (ffmpeg) ==="
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Please install (apt-get install ffmpeg or brew install ffmpeg)."
fi

echo "=== 2. Python Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Run with: python run.py"
