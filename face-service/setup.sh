#!/bin/bash

echo "=== 1. System Dependency Check ==="
if ! command -v ffmpeg &> /dev/null
then
    echo "FFmpeg could not be found. Attempting to install..."
    if [ -x "$(command -v apt-get)" ]; then
        echo "Detected Ubuntu/Debian."
        sudo apt-get update
        sudo apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0
    elif [ -x "$(command -v brew)" ]; then
        echo "Detected macOS."
        brew install ffmpeg
    else
        echo "Error: Package manager not found. Please install 'ffmpeg' manually."
        exit 1
    fi
else
    echo "FFmpeg is already installed."
fi

echo "=== 2. Python Dependency Installation ==="
# Ensure pip is up to date
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo "You can now run the server with: python run.py"
