import os
import time
import json
import threading
import http.server
import socketserver
import requests

API_URL = "http://localhost:8000/api/missions/judge"
VIDEO_PORT = 9000
VIDEO_FILENAME = "test_data/sample_sit.mp4"
LOCAL_IP = "localhost"

def run_video_server():
    handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", VIDEO_PORT), handler) as httpd:
        print(f"[Video Server] Serving directory at port {VIDEO_PORT}")
        httpd.serve_forever()

def test_judge():
    test_cases = [
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/down.mp4", "mission": "DOWN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/jump.mp4", "mission": "JUMP"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/paw.mp4", "mission": "PAW"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/sit_small.mp4", "mission": "SIT"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/spin.mp4", "mission": "TURN"},
    ]

    print(f"[Test] Starting sequential test for {len(test_cases)} videos...\n")

    for i, case in enumerate(test_cases):
        video_url = case["url"]
        mission_type = case["mission"]
        
        print(f"=== [Test Case {i+1}] Mission: {mission_type} ===")
        print(f"URL: {video_url}")

        payload = {
            "analysis_id": f"test-analysis-{i+1}",
            "walk_id": f"test-walk-{i+1}",
            "missions": [
                {
                    "mission_id": f"m_{i+1}",
                    "mission_type": mission_type,
                    "video_url": video_url
                }
            ]
        }

        print(f"Sending Request to {API_URL}...", end=" ", flush=True)
        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            elapsed = time.time() - start_time
            
            result = response.json()
            mission_res = result["missions"][0]
            status = "SUCCESS" if mission_res["success"] else "FAIL"
            
            print(f"DONE ({elapsed:.2f}s)")
            print(f"Result: {status} (Confidence: {mission_res['confidence']:.2f})")
            
        except requests.exceptions.HTTPError as e:
            print(f"FAILED (HTTP {response.status_code})")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("-" * 50)
        time.sleep(1)

if __name__ == "__main__":
    test_judge()
