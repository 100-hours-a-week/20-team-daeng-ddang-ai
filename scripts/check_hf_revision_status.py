#!/usr/bin/env python3
"""Check Hugging Face revision/update status across services.

Usage:
  python3 scripts/check_hf_revision_status.py
  python3 scripts/check_hf_revision_status.py --offline
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi
except Exception:  # pragma: no cover
    HfApi = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckTarget:
    service: str
    label: str
    repo_id: str
    token: Optional[str]
    revision_file: Path
    local_hint: Optional[Path] = None


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        # drop inline comments when not quoted
        val = val.strip()
        if "#" in val and not (val.startswith('"') or val.startswith("'")):
            val = val.split("#", 1)[0].strip()
        val = val.strip('"').strip("'")
        env[key] = val
    return env


def read_local_revision(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        val = path.read_text(encoding="utf-8").strip()
        return val or None
    except OSError:
        return None


def fetch_remote_revision(repo_id: str, token: Optional[str], offline: bool) -> tuple[Optional[str], Optional[str]]:
    if offline:
        return None, "offline mode"
    if HfApi is None:
        return None, "huggingface_hub not installed"
    if not token:
        return None, "token missing"
    try:
        sha = HfApi(token=token).model_info(repo_id=repo_id, revision="main").sha
        return sha, None
    except Exception as exc:  # pragma: no cover
        return None, f"remote fetch failed: {exc}"


def model_cache_snapshot_exists(repo_id: str, revision: Optional[str]) -> bool:
    if not revision:
        return False
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_key = f"models--{repo_id.replace('/', '--')}"
    snap_dir = cache_root / repo_key / "snapshots" / revision
    return snap_dir.exists()


def build_targets() -> list[CheckTarget]:
    # Load shell env first, then let service .env override only when key is absent.
    merged = dict(os.environ)

    face_env = parse_env_file(ROOT / "face-service" / ".env")
    health_env = parse_env_file(ROOT / "healthcare-service" / ".env")
    chat_env = parse_env_file(ROOT / "chatbot-service" / ".env")

    def pick(key: str, fallback: str = "") -> str:
        if key in merged:
            return merged[key]
        if key in face_env:
            return face_env[key]
        if key in health_env:
            return health_env[key]
        if key in chat_env:
            return chat_env[key]
        return fallback

    face_token = pick("HF_TOKEN", "")
    health_token = pick("HF_TOKEN", "")
    chat_token = pick("HUGGING_FACE_HUB_TOKEN", "")

    face_det_repo = pick("FACE_DETECTION_MODEL_ID", "20-team-daeng-ddang-ai/dog-detection")
    face_emo_repo = pick("FACE_EMOTION_MODEL_ID", "20-team-daeng-ddang-ai/dog-emotion-classification")
    health_repo = pick("HEALTH_MODEL_ID", "20-team-daeng-ddang-ai/dog-pose-estimation")
    chat_repo = pick("CHATBOT_ASSETS_REPO_ID", "20-team-daeng-ddang-ai/vet-chat")

    face_det_rev = Path(pick("FACE_DETECTION_REVISION_FILE", "models/.face_detection_revision"))
    face_emo_rev = Path(pick("FACE_EMOTION_REVISION_FILE", "models/.face_emotion_revision"))
    health_rev = Path(pick("HEALTH_MODEL_REVISION_FILE", "models/.health_model_revision"))
    chat_local_dir = Path(pick("CHATBOT_ASSETS_LOCAL_DIR", "models"))
    chat_rev = Path(pick("MODEL_REVISION_FILE", str(chat_local_dir / ".vet_chat_revision")))

    face_dir = ROOT / "face-service"
    health_dir = ROOT / "healthcare-service"
    chat_dir = ROOT / "chatbot-service"

    return [
        CheckTarget(
            service="face-service",
            label="detection",
            repo_id=face_det_repo,
            token=face_token or None,
            revision_file=(face_det_rev if face_det_rev.is_absolute() else face_dir / face_det_rev),
        ),
        CheckTarget(
            service="face-service",
            label="emotion",
            repo_id=face_emo_repo,
            token=face_token or None,
            revision_file=(face_emo_rev if face_emo_rev.is_absolute() else face_dir / face_emo_rev),
        ),
        CheckTarget(
            service="healthcare-service",
            label="pose-model",
            repo_id=health_repo,
            token=health_token or None,
            revision_file=(health_rev if health_rev.is_absolute() else health_dir / health_rev),
            local_hint=health_dir / pick("MODEL_CACHE_DIR", "models"),
        ),
        CheckTarget(
            service="chatbot-service",
            label="assets",
            repo_id=chat_repo,
            token=chat_token or None,
            revision_file=(chat_rev if chat_rev.is_absolute() else chat_dir / chat_rev),
            local_hint=(chat_local_dir if chat_local_dir.is_absolute() else chat_dir / chat_local_dir),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check HF revision/update status for each service")
    parser.add_argument("--offline", action="store_true", help="skip remote SHA fetch")
    args = parser.parse_args()

    targets = build_targets()

    print("HF revision status")
    print("=" * 72)

    for t in targets:
        local_rev = read_local_revision(t.revision_file)
        remote_rev, remote_err = fetch_remote_revision(t.repo_id, t.token, args.offline)
        cache_hit = model_cache_snapshot_exists(t.repo_id, remote_rev or local_rev)

        if remote_rev and local_rev:
            if remote_rev == local_rev:
                status = "UP_TO_DATE"
            else:
                status = "OUTDATED"
        elif remote_rev and not local_rev:
            status = "NO_LOCAL_REVISION"
        elif local_rev and not remote_rev:
            status = "REMOTE_UNKNOWN"
        else:
            status = "UNKNOWN"

        print(f"[{t.service} / {t.label}] {status}")
        print(f"  repo_id        : {t.repo_id}")
        print(f"  revision_file  : {t.revision_file}")
        print(f"  local_revision : {local_rev or '-'}")
        print(f"  remote_revision: {remote_rev or '-'}")
        print(f"  cache_snapshot : {'yes' if cache_hit else 'no'}")
        if t.local_hint:
            print(f"  local_hint     : {t.local_hint} ({'exists' if t.local_hint.exists() else 'missing'})")
        if remote_err:
            print(f"  note           : {remote_err}")
        print("-" * 72)

    print("Tips:")
    print("  - 'Downloaded ... /root/.cache/huggingface/.../snapshots/<sha>/file' log can be a cache hit path.")
    print("  - Real update is determined by local_revision vs remote_revision mismatch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
