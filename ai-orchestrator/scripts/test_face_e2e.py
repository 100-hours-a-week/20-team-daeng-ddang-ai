import sys
import os
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Endpoint Configuration
# Local: http://localhost:8000/api/face/analyze
# Remote: http://<SERVER_IP>:8000/api/face/analyze
DEFAULT_API_URL = "http://localhost:8000/api/face/analyze"
API_URL = os.getenv("TEST_FACE_API_URL", DEFAULT_API_URL)

def test_face_analysis(video_url: str):
    print(f"[Test] Starting Face Analysis E2E Test")
    print(f"[Target] {API_URL}")
    print(f"[Video] {video_url}\n")

    payload = {
        "analysis_id": f"test-face-{int(time.time())}",
        "video_url": video_url,
        "options": {}
    }

    print(f"Sending Request...", end=" ", flush=True)
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        elapsed = time.time() - start_time
        
        # Parse Response
        result = response.json()
        
        print(f"DONE ({elapsed:.2f}s)")
        print("\n=== Analysis Result ===")
        print(f"Emotion: {result.get('predicted_emotion', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"Summary: {result.get('summary', 'N/A')}")
        
        # Verify video_url
        returned_url = result.get('video_url')
        if returned_url == video_url:
            print(f"Video URL Verified: {returned_url}")
        else:
            print(f"WARNING: Video URL Mismatch! Expected: {video_url}, Got: {returned_url}")
        
        print("\n--- Full Response ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except requests.exceptions.HTTPError as e:
        print(f"FAILED (HTTP {response.status_code})")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_face_e2e.py <video_url>")
        print("Example: python scripts/test_face_e2e.py https://example.com/face.mp4")
        sys.exit(1)
        
    video_url = sys.argv[1]
    test_face_analysis(video_url)
