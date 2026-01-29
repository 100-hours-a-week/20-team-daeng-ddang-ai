
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

MODEL_ID = "wuhp/dog-yolo"

def inspect_model():
    print(f"Inspecting model: {MODEL_ID}")
    
    model_path = MODEL_ID
    if "/" in MODEL_ID and not os.path.exists(MODEL_ID):
        try:
            print("Downloading model...")
            model_path = hf_hub_download(repo_id=MODEL_ID, filename="dog-75e-11n.pt")
            print(f"Downloaded to: {model_path}")
        except Exception as e:
            print(f"Download failed (might need auth or different filename): {e}")
            # Try 'model.pt' or check if it downloads automatically via YOLO
            model_path = MODEL_ID 

    try:
        model = YOLO(model_path)
        print("\nModel Classes:")
        print(model.names)
    except Exception as e:
        print(f"Failed to load model: {e}")

if __name__ == "__main__":
    inspect_model()
