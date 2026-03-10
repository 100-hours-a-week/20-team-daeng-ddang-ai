#!/bin/bash
set -euo pipefail

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python3 run.py
