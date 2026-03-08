#!/usr/bin/env python3
"""Verify recent async/refactor changes and run basic endpoint smoke tests.

Checks:
1) Static code assertions for recent fixes:
   - Async HTTP client reuse in orchestrator adapters.
   - Chatbot upstream errors are propagated (not swallowed into 200).
   - Face/Healthcare startup persists revision files.
2) Optional runtime smoke tests against ai-orchestrator endpoints.
3) Optional revision-file existence checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib import request, error


ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_static_checks() -> list[str]:
    errors: list[str] = []

    face_adapter = ROOT / "ai-orchestrator/app/services/adapters/face_http_adapter.py"
    health_adapter = ROOT / "ai-orchestrator/app/services/adapters/healthcare_http_adapter.py"
    chat_adapter = ROOT / "ai-orchestrator/app/services/adapters/vetchat_http_adapter.py"
    chat_service = ROOT / "ai-orchestrator/app/services/vetchat_service.py"
    face_main = ROOT / "face-service/app/main.py"
    health_main = ROOT / "healthcare-service/app/main.py"

    face_text = _read_text(face_adapter)
    health_text = _read_text(health_adapter)
    chat_text = _read_text(chat_adapter)
    service_text = _read_text(chat_service)
    face_main_text = _read_text(face_main)
    health_main_text = _read_text(health_main)

    for name, text in [
        ("face adapter", face_text),
        ("healthcare adapter", health_text),
        ("chatbot adapter", chat_text),
    ]:
        if "_async_client = httpx.AsyncClient(" not in text:
            errors.append(f"{name}: missing reusable AsyncClient field")
        if "async with httpx.AsyncClient(" in text:
            errors.append(f"{name}: still creates AsyncClient per-request")

    if "HTTP_ADAPTER_ERROR" in chat_text:
        errors.append("chatbot adapter: still returns wrapped HTTP_ADAPTER_ERROR response")
    if "except httpx.HTTPStatusError" not in service_text:
        errors.append("vetchat_service: missing HTTPStatusError propagation mapping")
    if "_persist_current_revisions_after_startup()" not in face_main_text:
        errors.append("face main: missing startup revision persist call")
    if "_persist_current_revision_after_startup()" not in health_main_text:
        errors.append("healthcare main: missing startup revision persist call")

    return errors


def build_payload(endpoint: str) -> dict[str, Any]:
    if endpoint == "face":
        return {
            "analysis_id": "verify-face-001",
            "video_url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/face_test/testVideo01.mp4",
        }
    if endpoint == "healthcare":
        return {
            "analysis_id": "verify-health-001",
            "dog_id": 123,
            "video_url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/healthcare_test/testVideo02.mp4",
        }
    if endpoint == "chatbot":
        return {
            "dog_id": 1,
            "conversation_id": "verify-conv-001",
            "message": {"role": "user", "content": "강아지가 물을 많이 마셔요. 병원 가야 하나요?"},
            "image_url": None,
            "user_context": None,
            "history": [],
        }
    raise ValueError(endpoint)


def run_smoke_tests(base_url: str, timeout: int, allow_429: bool) -> list[str]:
    errors: list[str] = []
    targets = [
        ("face", f"{base_url.rstrip('/')}/api/face/analyze"),
        ("healthcare", f"{base_url.rstrip('/')}/api/healthcare/analyze"),
        ("chatbot", f"{base_url.rstrip('/')}/api/vet/chat"),
    ]

    print("\n[Smoke Tests]")
    for ep, url in targets:
        payload = build_payload(ep)
        try:
            req = request.Request(
                url=url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=timeout) as r:
                status_code = r.getcode()
                body_text = r.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            status_code = exc.code
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception as exc:
            errors.append(f"{ep}: request failed ({type(exc).__name__}: {exc})")
            continue

        ok_codes = {200}
        if allow_429:
            ok_codes.add(429)
        if status_code not in ok_codes:
            snippet = body_text[:400]
            errors.append(f"{ep}: unexpected status={status_code}, body={snippet}")
            continue

        detail = ""
        if status_code == 200:
            try:
                data = json.loads(body_text)
                if ep == "chatbot" and "answer" not in data and "error_code" not in data:
                    errors.append("chatbot: 200 response missing both answer and error_code")
                    continue
                detail = "ok"
            except json.JSONDecodeError:
                errors.append(f"{ep}: 200 but invalid JSON")
                continue
        else:
            detail = "overloaded(429)"

        print(f"- {ep}: {status_code} ({detail})")

    return errors


def check_revision_files() -> list[str]:
    errors: list[str] = []
    paths = [
        ROOT / "face-service/models/.face_detection_revision",
        ROOT / "face-service/models/.face_emotion_revision",
        ROOT / "healthcare-service/models/.health_model_revision",
    ]

    print("\n[Revision Files]")
    for p in paths:
        if not p.exists():
            errors.append(f"missing revision file: {p}")
            continue
        content = p.read_text(encoding="utf-8").strip()
        if not content:
            errors.append(f"empty revision file: {p}")
            continue
        print(f"- {p}: {content[:12]}...")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify async refactor changes and run smoke tests.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="ai-orchestrator base URL")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--skip-smoke", action="store_true", help="skip HTTP smoke tests")
    parser.add_argument("--check-revisions", action="store_true", help="check revision files exist and non-empty")
    parser.add_argument("--allow-429", action="store_true", help="treat 429 as acceptable in smoke tests")
    args = parser.parse_args()

    all_errors: list[str] = []

    print("[Static Checks]")
    static_errors = run_static_checks()
    if static_errors:
        all_errors.extend(static_errors)
    else:
        print("- passed")

    if not args.skip_smoke:
        all_errors.extend(
            run_smoke_tests(
                base_url=args.base_url,
                timeout=args.timeout_seconds,
                allow_429=args.allow_429,
            )
        )

    if args.check_revisions:
        all_errors.extend(check_revision_files())

    if all_errors:
        print("\n[FAILED]")
        for err in all_errors:
            print(f"- {err}")
        return 1

    print("\n[OK] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
