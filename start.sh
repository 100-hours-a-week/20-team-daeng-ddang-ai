#!/bin/bash
set -e

# 환경변수 기본값 설정 — vllm 기본
export VLLM_PORT=${VLLM_PORT:-8400}
export VLLM_MODEL=${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
export VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
export VLLM_DTYPE=${VLLM_DTYPE:-auto}

# LoRA 관련
export VLLM_ENABLE_LORA=${VLLM_ENABLE_LORA:-false}
export VLLM_LORA_NAME=${VLLM_LORA_NAME:-}
export VLLM_LORA_PATH=${VLLM_LORA_PATH:-}
export VLLM_LORA_REPO_ID=${VLLM_LORA_REPO_ID:-}
export VLLM_LORA_SUBDIR=${VLLM_LORA_SUBDIR:-}
export VLLM_LORA_REVISION=${VLLM_LORA_REVISION:-}
export VLLM_LORA_MODULES=${VLLM_LORA_MODULES:-}
export HF_TOKEN=${HF_TOKEN:-}
export VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

# chatbot-service가 vllm을 localhost로 접근하도록 설정
export VLLM_BASE_URL=${VLLM_BASE_URL:-http://localhost:${VLLM_PORT}}

echo "=== RunPod Combined Service Starting ==="
echo "vllm model:      ${VLLM_MODEL}"
echo "vllm port:       ${VLLM_PORT}"
echo "chatbot port:    8300"
echo "multimodal port: 8500"
echo "========================================="

exec supervisord -c /etc/supervisord.conf
