from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


load_dotenv(Path(__file__).resolve().parent / ".env")


def _append_if_value(args: list[str], flag: str, value: str | None) -> None:
    if value:
        args.extend([flag, value])


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _default_cache_dir() -> Path:
    return Path(os.getenv("VLLM_MODEL_CACHE_DIR", Path(__file__).resolve().parent / "models"))


def _split_hf_repo_subpath(value: str) -> tuple[str, str] | None:
    raw = (value or "").strip().strip("/")
    if not raw:
        return None

    candidate = Path(raw)
    if candidate.is_absolute() or raw.startswith(".") or candidate.exists():
        return None

    parts = [part for part in raw.split("/") if part]
    if len(parts) < 3:
        return None
    return "/".join(parts[:2]), "/".join(parts[2:])


def _download_lora_dir(repo_id: str, subdir: str, lora_name: str, revision: str | None) -> str:
    cache_root = _default_cache_dir() / "lora" / lora_name
    cache_root.mkdir(parents=True, exist_ok=True)

    allow_patterns = [f"{subdir}/*"] if subdir else None
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(cache_root),
        allow_patterns=allow_patterns,
        revision=revision,
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None,
    )

    resolved = cache_root / subdir if subdir else cache_root
    return str(resolved)


def _resolve_lora_path(lora_name: str, lora_path: str) -> str:
    repo_id = os.getenv("VLLM_LORA_REPO_ID", "").strip()
    repo_subdir = os.getenv("VLLM_LORA_SUBDIR", "").strip().strip("/")
    revision = os.getenv("VLLM_LORA_REVISION") or None

    if repo_id:
        return _download_lora_dir(repo_id, repo_subdir, lora_name, revision)

    parsed = _split_hf_repo_subpath(lora_path)
    if parsed is not None:
        parsed_repo_id, parsed_subdir = parsed
        return _download_lora_dir(parsed_repo_id, parsed_subdir, lora_name, revision)

    return lora_path


def _has_explicit_lora_source(lora_path: str) -> bool:
    return bool(
        lora_path
        or os.getenv("VLLM_LORA_REPO_ID", "").strip()
        or os.getenv("VLLM_LORA_SUBDIR", "").strip()
    )


def build_command() -> list[str]:
    host = os.getenv("VLLM_HOST", "0.0.0.0")
    port = os.getenv("VLLM_PORT", "8400")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    gpu_memory_utilization = os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")
    max_model_len = os.getenv("VLLM_MAX_MODEL_LEN", "4096")
    tensor_parallel_size = os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")
    dtype = os.getenv("VLLM_DTYPE", "auto")
    api_key = os.getenv("VLLM_API_KEY")
    enable_lora = _env_bool("VLLM_ENABLE_LORA", False)
    lora_modules = os.getenv("VLLM_LORA_MODULES", "").strip()
    lora_name = os.getenv("VLLM_LORA_NAME", "").strip()
    lora_path = os.getenv("VLLM_LORA_PATH", "").strip()
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

    if enable_lora:
        cmd.append("--enable-lora")
        if lora_modules:
            cmd.extend(["--lora-modules", lora_modules])
        elif lora_name and _has_explicit_lora_source(lora_path):
            resolved_lora_path = _resolve_lora_path(lora_name, lora_path)
            cmd.extend(["--lora-modules", f"{lora_name}={resolved_lora_path}"])

    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return cmd


if __name__ == "__main__":
    raise SystemExit(subprocess.call(build_command()))
