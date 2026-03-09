from __future__ import annotations

import os
import shlex
import subprocess
import sys


def _append_if_value(args: list[str], flag: str, value: str | None) -> None:
    if value:
        args.extend([flag, value])


def build_command() -> list[str]:
    host = os.getenv("VLLM_HOST", "0.0.0.0")
    port = os.getenv("VLLM_PORT", "8400")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    gpu_memory_utilization = os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")
    max_model_len = os.getenv("VLLM_MAX_MODEL_LEN", "4096")
    tensor_parallel_size = os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")
    dtype = os.getenv("VLLM_DTYPE", "auto")
    api_key = os.getenv("VLLM_API_KEY")
    extra_args = os.getenv("VLLM_EXTRA_ARGS", "")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        port,
        "--model",
        model,
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--max-model-len",
        max_model_len,
        "--tensor-parallel-size",
        tensor_parallel_size,
        "--dtype",
        dtype,
    ]
    _append_if_value(cmd, "--api-key", api_key)
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return cmd


if __name__ == "__main__":
    raise SystemExit(subprocess.call(build_command()))
