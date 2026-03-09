from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    port = os.getenv("VLLM_PORT", "8400")
    base_url = os.getenv("VLLM_BASE_URL", f"http://127.0.0.1:{port}").rstrip("/")
    url = f"{base_url}/health"

    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            if response.status == 200:
                return 0
            print(f"Unexpected status code: {response.status}", file=sys.stderr)
            return 1
    except urllib.error.URLError as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
