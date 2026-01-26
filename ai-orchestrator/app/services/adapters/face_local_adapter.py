# app/services/adapters/face_local_adapter.py
from __future__ import annotations

import os
import random
import tempfile
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List
import logging
import requests

from app.core.config import (
    FACE_DETECTION_MODEL_ID,
    FACE_EMOTION_MODEL_ID,
    HF_TOKEN,
)
from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse, FaceErrorResponse
from app.services.adapters.face_adapter import FaceAdapter

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
        
        tmp_video_path = None
        try:
            # 1. Download Video
            tmp_video_path = self._download_video(req.video_url, request_id)
            
            # 2. Process Video
            emotion_probs = self._process_video(tmp_video_path)
            
            # 3. Aggregate Results
            if not emotion_probs:
                logger.warning(f"[{request_id}] No face/dog detected in video.")
                return FaceAnalyzeResponse(
                    analysis_id=req.analysis_id,
                    request_id=request_id,
                    predicted_emotion="unknown",
                    confidence=0.0,
                    summary="강아지를 찾을 수 없습니다.",
                    emotion_probabilities={}
                )

            # 8-emotion dict
            # keys: anger, contempt, disgust, fear, happiness, neutral, sadness, surprise
            
            # 4-target-emotion map
            # angry: anger, disgust, contempt
            # happy: happiness, surprise
            # sad: sadness, fear
            # relaxed: neutral
            
            mapped_scores = {
                "angry": 0.0,
                "happy": 0.0,
                "sad": 0.0,
                "relaxed": 0.0
            }
            
            # Aggregate frame probabilities first
            avg_probs_8 = {}
            total_frames = len(emotion_probs)
            for prob_dict in emotion_probs:
                for emo, score in prob_dict.items():
                    avg_probs_8[emo] = avg_probs_8.get(emo, 0.0) + score
            
            for emo in avg_probs_8:
                avg_probs_8[emo] /= total_frames

            # Map various model labels to 4 target emotions
            # dima806 labels: Ahegao, Angry, Happy, Neutral, Sad, Surprise
            # HSE labels: anger, contempt, disgust, fear, happiness, neutral, sadness, surprise
            
            for emo, score in avg_probs_8.items():
                e = emo.lower()
                
                # Angry group
                if e in ["anger", "angry", "disgust", "contempt"]:
                    mapped_scores["angry"] += score
                
                # Happy group
                elif e in ["happiness", "happy", "surprise", "ahegao"]:
                    mapped_scores["happy"] += score
                
                # Sad group
                elif e in ["sadness", "sad", "fear"]:
                    mapped_scores["sad"] += score
                
                # Relaxed group
                elif e in ["neutral", "neutrality", "relaxed"]:
                    mapped_scores["relaxed"] += score
                
                else:
                    # Fallback: add to relaxed
                    mapped_scores["relaxed"] += score

            # Normalize just in case (though sum should be ~1.0)
            total_score = sum(mapped_scores.values())
            if total_score > 0:
                for k in mapped_scores:
                    mapped_scores[k] /= total_score
            
            # Find top emotion
            top_emotion = max(mapped_scores, key=mapped_scores.get)
            confidence = mapped_scores[top_emotion]

            # Generate Summary
            # Generate Summary
            summary_options = {
                "angry": [
                    "강아지가 현재 불만이 있거나 화가 난 상태로 보입니다.",
                    "지금은 강아지가 예민해 보여요. 주의가 필요합니다.",
                    "으르렁거리거나 화가 난 표정이 감지되었습니다."
                ],
                "happy": [
                    "강아지가 즐겁고 행복해 보입니다!",
                    "산책이 정말 즐거운가 봐요! 표정이 아주 밝습니다.",
                    "강아지가 신나 있어요! 꼬리를 흔들고 있을지도 몰라요."
                ],
                "sad": [
                    "강아지가 다소 우울하거나 겁을 먹은 것 같아요.",
                    "혹시 무서운 게 있었나요? 강아지가 위축되어 보입니다.",
                    "표정이 조금 슬퍼 보입니다. 컨디션을 확인해 주세요."
                ],
                "relaxed": [
                    "강아지가 편안하고 평온한 상태입니다.",
                    "아주 여유로운 표정이네요. 산책을 즐기고 있어요.",
                    "긴장하지 않고 편안하게 쉬거나 걷고 있는 모습입니다."
                ]
            }
            # top_emotion이 목록에 없으면(unknown 등) relaxed로 처리하거나 기본값 사용
            options = summary_options.get(top_emotion, ["강아지의 상태를 명확히 알기 어렵습니다."])
            summary = random.choice(options)

            return FaceAnalyzeResponse(
                analysis_id=req.analysis_id,
                request_id=request_id,
                predicted_emotion=top_emotion,
                confidence=float(confidence),
                summary=summary,
                emotion_probabilities=mapped_scores
            )

        except Exception as e:
            logger.error(f"[{request_id}] Local analysis failed: {e}", exc_info=True)
            # We return a fallback response or raise depending on requirement. 
            # For now, let's bubble up as error or return unknown
            # But the signature expects FaceAnalyzeResponse. 
            # Ideally we might raise an HTTPException in service if strictly failed.
            raise e
        finally:
            # Cleanup
            if tmp_video_path and os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)

    def _download_video(self, url: str, request_id: str) -> str:
        # Create temp file
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        
        logger.debug(f"[{request_id}] Downloading video to {path}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return path

    def _process_video(self, video_path: str) -> List[Dict[str, float]]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps) # 1 frame per second
        if frame_interval < 1: frame_interval = 1
        
        results_list = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                probs = self._analyze_frame(frame_rgb)
                if probs:
                    results_list.append(probs)
            
            frame_idx += 1
            
        cap.release()
        return results_list

    def _analyze_frame(self, frame_img: np.ndarray) -> Dict[str, float] | None:
        # 1. Detect Dog (YOLO)
        # class=16 is dog in COCO. 
        # But if using fine-tuned model, it might be class 0 or 1.
        # We will assume fine-tuned model has proper class names or we just take the best detection.
        # Let's try to find "dog" in names or just take the highest confidence object if detection model is specific.
        
        results = self.detector(frame_img, verbose=False) 
        # results is a list of Result objects
        result = results[0]
        
        best_box = None
        max_conf = 0.0
        
        for box in result.boxes:
            # Check class. If generic YOLO, check for dog (cls 16). 
            # If custom dog model, maybe any class is fine.
            # Using class name check to be safe if model has metadata
            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, "").lower()
            conf = float(box.conf[0])
            
            # Simple heuristic: if "dog" in name OR it is a custom model (we assume custom model detects dogs)
            # If we strictly want dogs and model is generic:
            if "dog" in cls_name or "yolo" in FACE_DETECTION_MODEL_ID.lower(): # Loose check
                # For standard YOLOCOCO, dog is 16.
                # If custom model, it might be the only class.
                pass
            
            if conf > max_conf:
                max_conf = conf
                best_box = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]

        if best_box is None:
            return None

        # 2. Crop
        x1, y1, x2, y2 = map(int, best_box)
        h, w, _ = frame_img.shape
        # Add small margin?
        margin = 0 
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        crop = frame_img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
            
        # 3. Classify Emotion
        # Convert to PIL for transformers
        pil_img = Image.fromarray(crop)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
            
        # Map to labels
        scores = {}
        for i, p in enumerate(probs):
            label = self.id2label.get(i, str(i))
            scores[label] = float(p)
            
        return scores
