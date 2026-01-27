import sys
import os
import logging
import json
from dotenv import load_dotenv

# Adjust path to find app module
sys.path.append(os.getcwd())

from app.services.adapters.face_local_adapter import FaceLocalAdapter
from app.schemas.face_schema import FaceAnalyzeRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
load_dotenv()

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_face_local.py <video_url>")
        print("Example: python scripts/test_face_local.py https://example.com/dog.mp4")
        sys.exit(1)

    video_url = sys.argv[1]
    
    print(f"Initializing FaceLocalAdapter (Model ID: {os.getenv('FACE_DETECTION_MODEL_ID')})...")
    try:
        adapter = FaceLocalAdapter()
    except Exception as e:
        print(f"Failed to initialize adapter: {e}")
        sys.exit(1)

    print(f"Analyzing video: {video_url}")
    req = FaceAnalyzeRequest(
        analysis_id="local_test",
        video_url=video_url
    )
    
    try:
        response = adapter.analyze("req_local_test", req)
        
        print("\n=== Analysis Result ===")
        # Dump model to json string
        print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()
