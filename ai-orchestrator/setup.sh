#!/bin/bash

echo "=== 1. Python Dependency Installation ==="
# Ensure pip is up to date
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo "You can now run the server with: python run.py"
