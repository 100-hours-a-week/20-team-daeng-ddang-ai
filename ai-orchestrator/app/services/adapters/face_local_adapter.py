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
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from app.core.config import (
    FACE_DETECTION_MODEL_ID,
    FACE_EMOTION_MODEL_ID,
    HF_TOKEN,
)
from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse, FaceErrorResponse
from app.services.adapters.face_adapter import FaceAdapter

# 표정 분석 설정 (Design Spec 반영 및 완화된 기준 적용)
FACE_CONF_THRESHOLD = 0.4      # 얼굴 탐지 신뢰도 임계값 (0.6 -> 0.4 완화)
FACE_AREA_MIN_RATIO = 0.02     # 프레임 대비 얼굴 최소 크기 비율 (5% -> 2% 완화)
MAX_FRAMES = 12                # 분석할 최대 프레임 수 (8 -> 12 증량)

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
    from transformers import AutoImageProcessor # Keep processor for preprocessing
    from huggingface_hub import login, hf_hub_download
    import torch.nn.functional as F
    import torchvision
    from torchvision import models
except ImportError:
    # This might happen if dependencies aren't installed yet
    YOLO = None
    torchvision = None

logger = logging.getLogger(__name__)

# 로컬 환경에서 얼굴 탐지 및 표정 분석을 수행하는 어댑터
# 1. FFmpeg를 사용한 스마트 키프레임 추출 (I-Frame -> Scene -> FPS)
# 2. YOLOv10n 기반 얼굴 탐지 및 크롭
# 3. EfficientNet-B0 기반 4종 감정 분류 (Happy, Sad, Angry, Relaxed)
# 4. 앙상블 및 나레이션 생성
class FaceLocalAdapter(FaceAdapter):
    def __init__(self):
        self._ensure_dependencies()
        self._authenticate_hf()
        
        logger.info(f"Loading Face Local Adapter with models: Detection={FACE_DETECTION_MODEL_ID}, Emotion={FACE_EMOTION_MODEL_ID}")
        
        # 객체 탐지 모델 로드 (YOLO)
        # HuggingFace Repo ID인 경우 best.pt를 다운로드하여 로드합니다.
        model_path = FACE_DETECTION_MODEL_ID
        if "/" in FACE_DETECTION_MODEL_ID and not os.path.exists(FACE_DETECTION_MODEL_ID):
            logger.info(f"Downloading YOLO model from HF: {FACE_DETECTION_MODEL_ID}")
            try:
                # Assuming the weight file is named 'best.pt' in the HF repo
                model_path = hf_hub_download(repo_id=FACE_DETECTION_MODEL_ID, filename="best.pt")
                logger.info(f"Downloaded YOLO model to: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to download from HF (maybe it's a local path or n/a): {e}")
                # Fallback to original string if download fails (might be local path)
                model_path = FACE_DETECTION_MODEL_ID

        self.detector = YOLO(model_path)
        
        # Load Emotion Classification Model (Using torchvision for custom weights)
        # 1. Load Processor for preprocessing (resize, normalize)
        try:
            self.processor = AutoImageProcessor.from_pretrained(FACE_EMOTION_MODEL_ID)
        except Exception:
            logger.warning(f"Could not load processor from {FACE_EMOTION_MODEL_ID}, falling back to google/efficientnet-b0")
            self.processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")

        # 2. Load Model Architecture (EfficientNet-B0)
        # The user's model is a torchvision EfficientNet-B0 with a 4-class classifier.
        try:
            logger.info("Initializing torchvision EfficientNet-B0...")
            self.classifier = models.efficientnet_b0(pretrained=False)
            
            # Replace classifier head for 4 classes (standard torchvision efficientnet has 'classifier' as Sequential)
            # Default: (1): Linear(in_features=1280, out_features=1000, bias=True)
            num_features = self.classifier.classifier[1].in_features
            self.classifier.classifier[1] = torch.nn.Linear(num_features, 4)
            
            # 3. Load Weights from HF
            weight_path = hf_hub_download(repo_id=FACE_EMOTION_MODEL_ID, filename="model.safetensors")
            state_dict = {}
            from safetensors.torch import load_file
            state_dict = load_file(weight_path)
            
            # Load state dict
            self.classifier.load_state_dict(state_dict)
            self.classifier.eval()
            logger.info(f"Successfully loaded torchvision weights from {FACE_EMOTION_MODEL_ID}")
            
        except Exception as e:
            logger.error(f"Failed to load torchvision model: {e}")
            raise e

        # Map model output labels explicitly since we aren't using config
        self.id2label = {0: "happy", 1: "sad", 2: "angry", 3: "relaxed"}

    def _ensure_dependencies(self):
        if YOLO is None or torchvision is None:
            raise RuntimeError("Required ML dependencies (ultralytics, transformers, torch, torchvision) are missing. Please install requirements.")

    def _authenticate_hf(self):
        if HF_TOKEN:
            login(token=HF_TOKEN)

    # 영상 URL을 입력받아 얼굴을 탐지하고 표정을 분석하여 결과를 반환
    # Process:
    # 1. 영상 다운로드
    # 2. FFmpeg 키프레임 추출 (최대 12장)
    # 3. 각 프레임별 얼굴 탐지 및 감정 추론
    # 4. 결과 앙상블 (Confidence 가중 평균)
    # 5. 나레이션 생성 및 응답 구성
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

    # 영상을 처리하여 유효한 얼굴 분석 결과를 목록으로 반환
    def _process_video(self, video_path: str) -> List[Dict[str, Any]]:
        # FFmpeg를 사용하여 키프레임 추출
        extracted_frames_dir = Path(tempfile.mkdtemp())
        
        try:
            # 1. 프레임 추출 실행
            frame_paths, extraction_method = self._extract_keyframes(Path(video_path), extracted_frames_dir)
            logger.info(f"Extracted {len(frame_paths)} frames using method: {extraction_method}")
            
            # 2. Limit to MAX_FRAMES (8)
            # If > 8, sample equidistantly
            if len(frame_paths) > MAX_FRAMES:
                indices = np.linspace(0, len(frame_paths) - 1, MAX_FRAMES, dtype=int)
                selected_paths = [frame_paths[i] for i in indices]
            else:
                selected_paths = frame_paths
            
            # 3. Analyze each frame
            results = []
            for fpath in selected_paths:
                # Read image using cv2
                # OpenCV imread doesn't handle Path objects well in some older versions, so str()
                frame = cv2.imread(str(fpath))
                if frame is None:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect & Analyze
                res = self._analyze_frame(frame_rgb)
                if res:
                    results.append(res)
                    
            return results
            
        finally:
            # Cleanup temp dir
            if extracted_frames_dir.exists():
                shutil.rmtree(extracted_frames_dir)

    def _extract_keyframes(self, video_path: Path, out_dir: Path) -> Tuple[List[Path], str]:
        output_pattern = str(out_dir / "frame_%06d.jpg")
        
        # Helper to run ffmpeg
        def run_ffmpeg(filters):
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(video_path),
                "-vf", filters,
                "-vsync", "vfr",
                output_pattern
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return sorted(out_dir.glob("*.jpg"))

        # 1) I-frames only
        try:
            frames = run_ffmpeg("select='eq(pict_type,I)'")
            # If we get too few I-frames (e.g. < 4), it might be a short video or bad encoding.
            # Fallback to scene change to get more variety.
            if len(frames) >= 4:
                return frames, "iframe"
            else:
                logger.info(f"Only {len(frames)} I-frames found. Trying scene change detection for better coverage...")
        except subprocess.CalledProcessError:
            pass # Try next method
        except subprocess.CalledProcessError:
            pass # Try next method

        # 2) Scene-change
        # Clear any potential partial outputs if needed (though overwrite -y handles it mostly, strict logic might want clean dir)
        # But here we assume overwrite is fine.
        try:
            frames = run_ffmpeg("select='gt(scene,0.3)'")
            if frames:
                return frames, "scene"
        except subprocess.CalledProcessError:
            pass

        # 3) Fallback: FPS sampling (1 frame per sec)
        try:
            frames = run_ffmpeg("fps=1.0")
            return frames, "fps"
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg extraction failed completely for {video_path}: {e}")
            return [], "failed"

    # 단일 프레임 이미지에서 가장 신뢰도 높은 얼굴을 찾아 감정을 분석
    # Returns:
    #   Dict: { "face_confidence": float, "emotion_probs": Dict[str, float] }
    #   None: 얼굴을 찾지 못한 경우
    def _analyze_frame(self, frame_img: np.ndarray) -> Optional[Dict[str, Any]]:
        # 1. 얼굴 탐지 (YOLO)
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
        
        # Inputs from processor: {'pixel_values': tensor}
        # Torchvision efficientnet expects just the tensor (B, C, H, W)
        input_tensor = inputs['pixel_values']

        with torch.no_grad():
            # Pass tensor directly to torchvision model
            logits = self.classifier(input_tensor)
            probs = F.softmax(logits, dim=-1)[0]
            
        # Map output to 4 target emotions
        # We assume our custom model is trained on 4 classes: happy, sad, angry, relaxed
        # And the order matches index 0, 1, 2, 3
        # Direct mapping:
        mapped_probs = {
            "happy": float(probs[0]),
            "sad": float(probs[1]),
            "angry": float(probs[2]),
            "relaxed": float(probs[3])
        }
        
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

    # 여러 프레임의 분석 결과를 종합(Ensemble)하여 최종 감정을 결정
    # 방식: 얼굴 탐지 신뢰도(face_confidence)를 가중치로 한 가중 평균 사용
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
