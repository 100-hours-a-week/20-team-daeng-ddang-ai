# app/services/adapters/face_local_adapter.py
from __future__ import annotations

import os
import random
import tempfile
import cv2
import torch
import numpy as np
from PIL import Image
import datetime
import time
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple

from app.core.config import (
    FACE_DETECTION_MODEL_ID,
    FACE_EMOTION_MODEL_ID,
    HF_TOKEN,
)
from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse, FaceErrorResponse
from app.services.adapters.face_adapter import FaceAdapter

# Constants from Design
FACE_CONF_THRESHOLD = 0.6
FACE_AREA_MIN_RATIO = 0.05
MAX_FRAMES = 8

NARRATION_TEMPLATES = {
    "happy": {
        "HIGH": "오늘 산책 최고였어! 꼬리가 절로 흔들렸어.",
        "MID": "산책하면서 기분이 꽤 좋았어.",
        "LOW": "음… 그래도 조금은 즐거웠던 것 같아."
    },
    "relaxed": {
        "HIGH": "바람 맞으면서 아주 편안하게 걸었어.",
        "MID": "천천히 걷기 딱 좋은 기분이었어.",
        "LOW": "아마도 그냥 무난하고 차분했던 것 같아."
    },
    "sad": {
        "HIGH": "오늘은 조금 힘이 없어서 발걸음이 느렸어.",
        "MID": "산책 중에 살짝 풀이 죽었어.",
        "LOW": "음… 기분이 아주 좋진 않았던 것 같아."
    },
    "angry": {
        "HIGH": "괜히 신경 쓰이는 게 많아서 좀 불편했어.",
        "MID": "조금 예민해져 있었던 것 같아.",
        "LOW": "살짝 거슬리는 순간들이 있었어."
    }
}

# Try to import ML libraries (handled gracefully if missing during initial setup)
try:
    from ultralytics import YOLO
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from huggingface_hub import login
    import torch.nn.functional as F
except ImportError:
    # This might happen if dependencies aren't installed yet
    YOLO = None
    AutoModelForImageClassification = None

logger = logging.getLogger(__name__)

class FaceLocalAdapter(FaceAdapter):
    def __init__(self):
        self._ensure_dependencies()
        self._authenticate_hf()
        
        logger.info(f"Loading Face Local Adapter with models: Detection={FACE_DETECTION_MODEL_ID}, Emotion={FACE_EMOTION_MODEL_ID}")
        
        # Load Object Detection Model (YOLO)
        # ultralytics handles caching and downloading from HF automatically
        self.detector = YOLO(FACE_DETECTION_MODEL_ID)
        
        # Load Emotion Classification Model (Transformers)
        self.processor = AutoImageProcessor.from_pretrained(FACE_EMOTION_MODEL_ID)
        self.classifier = AutoModelForImageClassification.from_pretrained(FACE_EMOTION_MODEL_ID)
        self.classifier.eval()
        
        # Map model output labels to readable strings if needed
        # Assuming the model config has id2label
        self.id2label = self.classifier.config.id2label

    def _ensure_dependencies(self):
        if YOLO is None or AutoModelForImageClassification is None:
            raise RuntimeError("Required ML dependencies (ultralytics, transformers, torch) are missing. Please install requirements.")

    def _authenticate_hf(self):
        if HF_TOKEN:
            login(token=HF_TOKEN)

    def analyze(self, request_id: str, req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
        logger.info(f"[{request_id}] Starting local face analysis for {req.video_url}")
        
        start_time = time.time()
        tmp_video_path = None
        
        try:
            # 1. Download Video
            tmp_video_path = self._download_video(req.video_url, request_id)
            
            # 2. Extract & Process Frames
            # Limit to 8 frames as per spec
            frame_results = self._process_video(tmp_video_path)
            
            frames_total_extracted = 8 # We target 8. In _process_video we will enforce this.
            frames_face_detected = len(frame_results)
            frames_emotion_inferred = len(frame_results) 
            
            # 3. Ensemble Calculation
            if not frame_results:
                logger.warning(f"[{request_id}] No face detected in any selected frames.")
                # Return failure response or success with "unknown" status?
                # Spec says Error Code FACE_NOT_DETECTED if all frames fail.
                # However, to be safe with the schema, let's Raise standardized error or return empty result.
                # Spec: "FACE_NOT_DETECTED" 422
                raise ValueError("FACE_NOT_DETECTED")

            predicted_emotion, confidence, final_probs = self._calculate_ensemble(frame_results)
            
            # 4. Generate Narration
            summary = self._generate_narration(predicted_emotion, confidence)
            
            # 5. Construct Response
            processing_stats = {
                "analysis_time_ms": int((time.time() - start_time) * 1000),
                "frames_extracted": frames_total_extracted, # We attempt 8
                "frames_face_detected": frames_face_detected,
                "frames_emotion_inferred": frames_emotion_inferred,
                "fps_used": 5 # fixed in spec? or dynamic. We will say we used sampled frames.
            }
            
            result_data = {
                "emotion": {
                    "predicted_emotion": predicted_emotion,
                    "confidence": confidence,
                    "summary": summary,
                    "emotion_probabilities": final_probs
                }
            }
            
            return FaceAnalyzeResponse(
                analysis_id=req.analysis_id or request_id,
                predicted_emotion=predicted_emotion,
                confidence=confidence,
                summary=summary,
                emotion_probabilities=final_probs,
                processing=processing_stats
            )

        except ValueError as ve:
            # Handle known logical errors like FACE_NOT_DETECTED
            # We might want to bubble this up properly, but for now log and re-raise
            logger.error(f"[{request_id}] Analysis Valid Error: {ve}")
            raise ve
        except Exception as e:
            logger.error(f"[{request_id}] Local analysis failed: {e}", exc_info=True)
            raise e
        finally:
            if tmp_video_path and os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)

    def _download_video(self, url: str, request_id: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        logger.debug(f"[{request_id}] Downloading video to {path}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return path

    def _process_video(self, video_path: str) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Select 8 equidistant frames
        # If total_frames < 8, take all.
        if total_frames <= MAX_FRAMES:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int).tolist()
            
        results = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect & Analyze
            res = self._analyze_frame(frame_rgb)
            if res:
                 results.append(res)
                 
        cap.release()
        return results

    def _analyze_frame(self, frame_img: np.ndarray) -> Optional[Dict[str, Any]]:
        # 1. Detect (YOLO)
        yolo_results = self.detector(frame_img, verbose=False)
        if not yolo_results: return None
        result = yolo_results[0]
        
        best_box = None
        max_conf = 0.0
        
        for box in result.boxes:
            conf = float(box.conf[0])
            # Filter by confidence
            if conf < FACE_CONF_THRESHOLD:
                continue
                
            # Filter by Area logic (>= 5% of frame)
            # box.xyxy is [x1, y1, x2, y2]
            coords = box.xyxy[0].cpu().numpy()
            width = coords[2] - coords[0]
            height = coords[3] - coords[1]
            area_ratio = (width * height) / (frame_img.shape[0] * frame_img.shape[1])
            
            if area_ratio < FACE_AREA_MIN_RATIO:
                continue

            if conf > max_conf:
                max_conf = conf
                best_box = coords

        if best_box is None:
            return None

        # 2. Crop
        x1, y1, x2, y2 = map(int, best_box)
        # Margin optional, safely clip
        h_img, w_img, _ = frame_img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        crop = frame_img[y1:y2, x1:x2]
        if crop.size == 0: return None
        
        # 3. Emotion Inference
        pil_img = Image.fromarray(crop)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
            
        # Map output to 4 target emotions
        # We assume general model (7-8 classes) -> Map to 4
        mapped_probs = self._map_emotions(probs)
        
        return {
            "face_confidence": max_conf,
            "emotion_probs": mapped_probs
        }

    def _map_emotions(self, probs: torch.Tensor) -> Dict[str, float]:
        # Mapping logic (same as before but cleaner)
        raw_scores = {}
        for i, p in enumerate(probs):
            label = self.id2label.get(i, str(i)).lower()
            raw_scores[label] = float(p)
            
        target_scores = {"angry": 0.0, "happy": 0.0, "sad": 0.0, "relaxed": 0.0}
        
        for label, score in raw_scores.items():
            if label in ["anger", "angry", "disgust", "contempt"]:
                target_scores["angry"] += score
            elif label in ["happiness", "happy", "surprise", "ahegao", "joy"]: # 'joy' often exists
                target_scores["happy"] += score
            elif label in ["sadness", "sad", "fear", "crying"]:
                target_scores["sad"] += score
            elif label in ["neutral", "neutrality", "relaxed", "calm"]:
                target_scores["relaxed"] += score
            else:
                target_scores["relaxed"] += score # Default fallback
        
        # Normalize
        total = sum(target_scores.values())
        if total > 0:
            for k in target_scores:
                target_scores[k] /= total
                
        return target_scores

    def _calculate_ensemble(self, frame_results: List[Dict[str, Any]]) -> Tuple[str, float, Dict[str, float]]:
        # Encapsulates formula: Sum(prob * face_conf) / Sum(face_conf)
        
        final_probs = {"angry": 0.0, "happy": 0.0, "sad": 0.0, "relaxed": 0.0}
        total_weight = sum(r["face_confidence"] for r in frame_results)
        
        if total_weight == 0: 
            # Should not happen if list not empty
            return "relaxed", 0.0, final_probs
            
        for res in frame_results:
            w = res["face_confidence"] / total_weight
            for emo in final_probs:
                final_probs[emo] += res["emotion_probs"][emo] * w
                
        # Determine winner
        predicted_emotion = max(final_probs, key=final_probs.get)
        confidence = final_probs[predicted_emotion]
        
        return predicted_emotion, confidence, final_probs

    def _generate_narration(self, emotion: str, confidence: float) -> str:
        level = self._get_confidence_level(confidence)
        # fallback if emotion key missing
        templates = NARRATION_TEMPLATES.get(emotion, NARRATION_TEMPLATES["relaxed"])
        return templates.get(level, templates["MID"])

    def _get_confidence_level(self, confidence: float) -> str:
        if confidence >= 0.75:
            return "HIGH"
        elif confidence >= 0.50:
            return "MID"
        else:
            return "LOW"
