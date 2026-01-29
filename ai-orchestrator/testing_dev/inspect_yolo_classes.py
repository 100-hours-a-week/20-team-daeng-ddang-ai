
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from app.core.config import FACE_DETECTION_MODEL_ID, HF_TOKEN

def inspect_model():
    print(f"Inspecting model: {FACE_DETECTION_MODEL_ID}")
    
    model_path = FACE_DETECTION_MODEL_ID
    if "/" in FACE_DETECTION_MODEL_ID and not os.path.exists(FACE_DETECTION_MODEL_ID):
        try:
            print("Downloading model...")
            model_path = hf_hub_download(repo_id=FACE_DETECTION_MODEL_ID, filename="best.pt", token=HF_TOKEN)
            print(f"Downloaded to: {model_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            return

    try:
        model = YOLO(model_path)
        print("\nModel Classes:")
        print(model.names)
    except Exception as e:
        print(f"Failed to load model: {e}")

if __name__ == "__main__":
    inspect_model()
