
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

def load_config(model_dir):
    """Loads class mapping and inference configuration."""
    class_path = model_dir / "class.json"
    config_path = model_dir / "inference_config.json"
    
    if not class_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Config files not found in {model_dir}")

    with open(class_path, "r") as f:
        class_map = json.load(f)
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    return class_map, config

def download_model_if_needed(repo_id, model_dir):
    """Downloads model files from Hugging Face Hub if not present."""
    model_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = ["best.pt", "class.json", "inference_config.json"]
    
    print(f"Checking/Downloading model files from {repo_id} to {model_dir}...")
    for filename in files_to_download:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
    print("Model files ready.")

def main():
    parser = argparse.ArgumentParser(description="Test Dog Detection Model")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--repo_id", default="20-team-daeng-ddang-ai/dog-detection", help="Hugging Face Repo ID")
    parser.add_argument("--model_dir", default="model_files", help="Directory to store/load model files")
    parser.add_argument("--output", default="detected_output.jpg", help="Path to save the output image")
    parser.add_argument("--expand_ratio", type=float, default=0.0, help="Ratio to expand the bounding box (e.g., 0.1 for 10% margin)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Error: Image '{image_path}' not found.")
        return

    # 1. Download/Check Model
    try:
        download_model_if_needed(args.repo_id, model_dir)
    except Exception as e:
        print(f"Error downloading model: {e}")
        return

    # 2. Load Config & Model
    try:
        class_map, config = load_config(model_dir)
        # Convert string keys to int if necessary, though JSON keys are strings
        # Assuming class_map is {"0": "class_name", ...}
        
        print(f"Loading model from {model_dir / 'best.pt'}...")
        model = YOLO(str(model_dir / "best.pt"))
    except Exception as e:
        print(f"Error loading model/config: {e}")
        return

    # 3. Inference
    # 결과 시각화 및 저장
    print(f"Running inference on {image_path}...")
    conf_threshold = config.get("confidence_threshold", 0.5)
    iou_threshold = config.get("iou_threshold", 0.5) # Not directly used in YOLO call below unless custom non-max suppression is needed, but YOLO handles it. 
    # Validating args for predict: https://docs.ultralytics.com/modes/predict/#inference-arguments
    
    results = model.predict(source=str(image_path), conf=conf_threshold, iou=iou_threshold, verbose=False)

    # 탐지 결과 로그 출력
    for r in results:
        for box in r.boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            cls_name = class_map.get(str(cls), str(cls)) # Using class_map for consistency
            print(f" -> Detected '{cls_name}' with confidence: {conf*100:.2f}%")

    # 4. Visualization
    image = cv2.imread(str(image_path))
    if image is None:
        print("Error: Could not read image with OpenCV.")
        return
        
    img_h, img_w = image.shape[:2]

    result = results[0] # Single image inference
    
    for box in result.boxes:
        # Bounding Box
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Original Box (Blue)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Expand Box
        if args.expand_ratio > 0:
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, img_w, img_h, args.expand_ratio)

        # Confidence & Class
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        
        class_name = class_map.get(str(cls_id), str(cls_id))
        conf_percent = conf * 100
        
        label = f"{class_name}: {conf_percent:.1f}%"
        
        # Color: Green (Final Box)
        color = (0, 255, 0)
        
        # Draw Box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 5. Save Output
    output_path = Path(args.output)
    
    # Check if output is a directory or looks like one (no suffix)
    if output_path.is_dir() or not (output_path.name.lower().endswith('.jpg') or output_path.name.lower().endswith('.png')):
        # Create directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        # Construct filename: detected_<original_name>
        output_path = output_path / f"detected_{image_path.name}"
    else:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), image)
    print(f"Result saved to {output_path}")

def expand_box(x1, y1, x2, y2, img_w, img_h, ratio):
    """Expands the bounding box by a ratio while keeping it within image bounds."""
    w = x2 - x1
    h = y2 - y1
    
    dw = int(w * ratio)
    dh = int(h * ratio)
    
    x1 = max(0, x1 - dw)
    y1 = max(0, y1 - dh)
    x2 = min(img_w, x2 + dw)
    y2 = min(img_h, y2 + dh)
    
    return x1, y1, x2, y2

if __name__ == "__main__":
    main()
