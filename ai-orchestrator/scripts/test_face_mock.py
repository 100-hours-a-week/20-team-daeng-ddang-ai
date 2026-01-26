import sys
import os
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add app to path
sys.path.append(os.getcwd())

try:
    from app.services.adapters.face_mock_adapter import FaceMockAdapter
    from app.schemas.face_schema import FaceAnalyzeRequest
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

def test_mock_face_analysis():
    print("--- Starting Mock Face Analysis Test ---")
    
    adapter = FaceMockAdapter()
    
    # Test Case 1: Default (Relaxed)
    print("\n[Test 1] Default Request (No options)")
    req1 = FaceAnalyzeRequest(
        analysis_id="test_req_001",
        video_url="http://test.com/video1.mp4"
    )
    res1 = adapter.analyze("req_id_001", req1)
    pprint(res1.dict())
    
    # Test Case 2: Force Happy
    print("\n[Test 2] Force Emotion: 'happy'")
    req2 = FaceAnalyzeRequest(
        analysis_id="test_req_002",
        video_url="http://test.com/video2.mp4",
        options={"force_emotion": "happy"}
    )
    res2 = adapter.analyze("req_id_002", req2)
    pprint(res2.dict())

    # Test Case 3: Force Angry
    print("\n[Test 3] Force Emotion: 'angry'")
    req3 = FaceAnalyzeRequest(
        analysis_id="test_req_003",
        video_url="http://test.com/video3.mp4",
        options={"force_emotion": "angry"}
    )
    res3 = adapter.analyze("req_id_003", req3)
    pprint(res3.dict())
    
    print("\n--- Test Completed ---")

if __name__ == "__main__":
    test_mock_face_analysis()
