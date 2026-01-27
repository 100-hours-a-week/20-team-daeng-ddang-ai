import sys
import os
import requests
import json
import time

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/test_mission_local.py <MISSION_TYPE> <VIDEO_URL> [API_URL]")
        print("Example: python scripts/test_mission_local.py SIT https://example.com/sit.mp4")
        sys.exit(1)

    mission_type = sys.argv[1].upper()
    video_url = sys.argv[2]
    api_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000/api/missions/judge"

    print(f"Testing Mission: {mission_type}")
    print(f"Video URL: {video_url}")
    print(f"Target API: {api_url}")

    payload = {
        "analysis_id": "manual-test-001",
        "walk_id": "test-walk",
        "missions": [
            {
                "mission_id": "m_test_01",
                "mission_type": mission_type,
                "video_url": video_url
            }
        ]
    }

    try:
        start_time = time.time()
        print("\nSending request...", end=" ", flush=True)
        
        response = requests.post(api_url, json=payload)
        elapsed = time.time() - start_time
        print(f"Done ({elapsed:.2f}s)\n")

        if response.status_code == 200:
            print("=== Response ===")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(f"Failed (HTTP {response.status_code})")
            print(response.text)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
