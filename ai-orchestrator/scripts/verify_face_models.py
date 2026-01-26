import sys
import os
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load env
load_dotenv()

print("Verifying Face Analysis Dependencies and Models...")

# Mock settings just in case
if not os.getenv("FACE_DETECTION_MODEL_ID"):
    os.environ["FACE_DETECTION_MODEL_ID"] = "jameslahm/yolov10n"
if not os.getenv("FACE_EMOTION_MODEL_ID"):
    os.environ["FACE_EMOTION_MODEL_ID"] = "HSE-motion/hse-emotion-recognizer"

try:
    # Adjust path to find app module
    sys.path.append(os.getcwd())
    from app.services.adapters.face_local_adapter import FaceLocalAdapter
    print("Successfully imported FaceLocalAdapter.")
except ImportError as e:
    print(f"Failed to import FaceLocalAdapter: {e}")
    sys.exit(1)

print("Initializing Adapter (this may download models)...")
try:
    adapter = FaceLocalAdapter()
    print("FaceLocalAdapter initialized successfully!")
    print(f"Detector loaded: {adapter.detector}")
    print(f"Classifier loaded: {adapter.classifier}")
except Exception as e:
    print(f"Failed to initialize adapter: {e}")
    sys.exit(1)
