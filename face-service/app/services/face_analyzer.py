# app/services/face_analyzer.py
from __future__ import annotations

import os
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
import timm
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from app.core.config import (
    FACE_DETECTION_MODEL_ID,
    FACE_EMOTION_MODEL_ID,
    HF_TOKEN,
    DEBUG,
    TORCH_DEVICE,
)
from app.schemas.face_schema import FaceAnalyzeRequest, FaceAnalyzeResponse

# 표정 분석 설정
FACE_CONF_THRESHOLD = 0.65     # 얼굴 탐지 신뢰도 임계값
FACE_AREA_MIN_RATIO = 0.02     # 프레임 대비 얼굴 최소 크기 비율 (2%)
MAX_FRAMES = 12                # 분석할 최대 프레임 수
# DOG_CLASS_ID는 모델 로드 시 동적으로 결정되지만 기본값으로 설정
DEFAULT_DOG_CLASS_ID = 16

# 나레이션 템플릿 (감정과 신뢰도에 따른 문구)
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

# ML 라이브러리 임포트 (초기 설정 시 없어도 에러 처리됨)
try:
    from ultralytics import YOLO
    from transformers import AutoImageProcessor # 전처리용
    from huggingface_hub import login, hf_hub_download
    import torch.nn.functional as F
    import torchvision
    from torchvision import models
except ImportError:
    YOLO = None
    torchvision = None

logger = logging.getLogger(__name__)

# Face Analysis Service Logic
# 1. FFmpeg를 사용한 균등 FPS 샘플링
# 2. YOLOv10n 기반 얼굴 탐지 (강아지 얼굴) 및 마진 포함 크롭
# 3. EfficientNet-B0 기반 4종 감정 분류 (Happy, Sad, Angry, Relaxed)
# 4. 앙상블 (Confidence Weighted) 및 나레이션 생성
class FaceAnalyzer:
    def __init__(self):
        self._ensure_dependencies()
        self._authenticate_hf()
        
        logger.info(f"FaceAnalyzer 로딩 시작: Detection={FACE_DETECTION_MODEL_ID}, Emotion={FACE_EMOTION_MODEL_ID}, Device={TORCH_DEVICE}")
        
        # 1. 객체 탐지 모델 로드 (YOLO)
        # HuggingFace Repo ID인 경우 best.pt를 다운로드하여 로드합니다.
        model_path = FACE_DETECTION_MODEL_ID
        
        # 로컬 경로가 아니고 HF 저장소 ID라면 다운로드 시도
        if "/" in FACE_DETECTION_MODEL_ID and not os.path.exists(FACE_DETECTION_MODEL_ID):
            logger.info(f"HF에서 YOLO 모델 다운로드 시도: {FACE_DETECTION_MODEL_ID}")
            download_successful = False
            try:
                # 'best.pt' 우선 시도
                model_path = hf_hub_download(repo_id=FACE_DETECTION_MODEL_ID, filename="best.pt")
                download_successful = True
                logger.info(f"Downloaded YOLO model 'best.pt' to: {model_path}")
            except Exception:
                logger.warning(f"Failed to download 'best.pt' from HF repo {FACE_DETECTION_MODEL_ID}. Trying 'dog-75e-11n.pt'...")
                try:
                    # Fallback (이전 모델명)
                    model_path = hf_hub_download(repo_id=FACE_DETECTION_MODEL_ID, filename="dog-75e-11n.pt")
                    download_successful = True
                    logger.info(f"Downloaded YOLO model 'dog-75e-11n.pt' to: {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to download 'dog-75e-11n.pt' from HF repo {FACE_DETECTION_MODEL_ID}: {e}")
            
            if not download_successful:
                # 다운로드 실패 시 원본 ID 문자열 사용 (라이브러리가 처리하도록)
                logger.warning(f"모델 파일 다운로드 실패. 원본 ID '{FACE_DETECTION_MODEL_ID}'를 사용하여 로드합니다.")
                model_path = FACE_DETECTION_MODEL_ID 

        self.detector = YOLO(model_path)
        
        # 모델의 클래스 이름(names)을 확인하여 강아지 클래스 ID 결정
        names = self.detector.names
        self.target_class_id = 16 # 기본값: COCO 'dog' class ID
        
        if 'Head' in names.values():
             for k, v in names.items():
                 if v == 'Head':
                     self.target_class_id = k
                     break
        elif 'dog' in names.values():
             for k, v in names.items():
                 if v == 'dog':
                     self.target_class_id = k
                     break
        elif 'dog_face' in names.values():
             for k, v in names.items():
                 if v == 'dog_face':
                     self.target_class_id = k
                     break
        
        logger.info(f"Target Dog Class ID determined: {self.target_class_id} (Model names: {names})")

        # 2. 감정 분류 모델 로드 (Custom EfficientNet-B0)
        try:
            # 설정 파일 및 가중치 다운로드
            labels_path = hf_hub_download(repo_id=FACE_EMOTION_MODEL_ID, filename="labels.json")
            preprocess_path = hf_hub_download(repo_id=FACE_EMOTION_MODEL_ID, filename="preprocess.json")
            weight_path = hf_hub_download(repo_id=FACE_EMOTION_MODEL_ID, filename="best.pt")

            # 라벨 로드
            with open(labels_path, "r") as f:
                labels_data = json.load(f)
                self.id2label = {int(k): v for k, v in labels_data["id2label"].items()}
                self.label2id = labels_data["label2id"]
            
            num_classes = len(self.id2label)
            logger.info(f"Loaded {num_classes} classes from labels.json: {self.id2label}")

            # 전처리 설정 로드
            with open(preprocess_path, "r") as f:
                preprocess_config = json.load(f)
            
            img_size = preprocess_config.get("img_size", 224)
            mean = preprocess_config.get("normalize_mean", [0.485, 0.456, 0.406])
            std = preprocess_config.get("normalize_std", [0.229, 0.224, 0.225])

            # 전처리 파이프라인 정의
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((img_size, img_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)
            ])
            logger.info(f"Defined transform with size={img_size}, mean={mean}, std={std}")

            # 모델 아키텍처 초기화 (timm EfficientNet-B0)
            logger.info("timm EfficientNet-B0 초기화 중...")
            self.classifier = timm.create_model(
                'efficientnet_b0',
                pretrained=False,
                num_classes=num_classes
            )
            
            # 모델 가중치 로드
            checkpoint = torch.load(weight_path, map_location=torch.device(TORCH_DEVICE)) 
            
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
                
            self.classifier.load_state_dict(state_dict)
            self.classifier.to(TORCH_DEVICE) # GPU/CPU 이동
            self.classifier.eval()
            logger.info(f"Successfully loaded model weights from {weight_path}")

        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            raise e

    def _ensure_dependencies(self):
        if YOLO is None or torchvision is None:
            raise RuntimeError("Required ML dependencies are missing.")

    def _authenticate_hf(self):
        if HF_TOKEN:
            login(token=HF_TOKEN)

    # 영상 URL을 입력받아 얼굴을 탐지하고 표정을 분석하여 결과를 반환
    # Process:
    # 1. 영상 다운로드
    # 2. FFmpeg 프레임 추출 (fps=2)
    # 3. 각 프레임별 얼굴 탐지 및 감정 추론 (Padding 적용)
    # 4. 결과 앙상블 (Margin Weighted Voting)
    # 5. 나레이션 생성 및 응답 구성
    def analyze(self, request_id: str, req: FaceAnalyzeRequest) -> FaceAnalyzeResponse:
        logger.info(f"[{request_id}] 분석 시작: {req.video_url}")
        
        start_time = time.time()
        tmp_video_path = None
        
        try:
            # 1. Download Video
            tmp_video_path = self._download_video(req.video_url, request_id)
            
            # 2. Extract & Process Frames
            # 최대 12프레임까지 처리 가능
            frame_results = self._process_video(tmp_video_path, request_id)
            
            # 3. 앙상블 계산
            if not frame_results:
                logger.warning(f"[{request_id}] 선택된 프레임에서 강아지 얼굴을 찾지 못했습니다.")
                # 실패 응답 반환
                include_processing = DEBUG or req.options.get("include_processing", False)
                
                processing_stats = None
                if include_processing:
                    processing_stats = {
                        "analysis_time_ms": int((time.time() - start_time) * 1000),
                        "frames_extracted": 0,
                        "frames_face_detected": 0,
                        "frames_emotion_inferred": 0,
                        "fps_used": 2
                    }
                return FaceAnalyzeResponse(
                     analysis_id=req.analysis_id or request_id,
                     video_url=req.video_url,
                     error_code="FACE_NOT_DETECTED",
                     processing=processing_stats
                )

            predicted_emotion, confidence, final_probs = self._calculate_ensemble(frame_results)
            
            # 4. 나레이션 생성
            summary = self._generate_narration(predicted_emotion, confidence)
            
            # 5. 응답 구성
            include_processing = DEBUG or req.options.get("include_processing", False)
            
            processing_stats = None
            if include_processing:
                processing_stats = {
                    "analysis_time_ms": int((time.time() - start_time) * 1000),
                    "frames_extracted": 8, # 근사값
                    "frames_face_detected": len(frame_results),
                    "frames_emotion_inferred": len(frame_results),
                    "fps_used": 2
                }
            
            return FaceAnalyzeResponse(
                analysis_id=req.analysis_id or request_id,
                predicted_emotion=predicted_emotion,
                confidence=confidence,
                summary=summary,
                emotion_probabilities=final_probs,
                video_url=req.video_url,
                processing=processing_stats
            )

        except ValueError as ve:
                # 논리적 에러 (예: 영상 다운로드 실패 등)
                logger.error(f"[{request_id}] 분석 유효성 에러: {ve}")
                raise ve
        except Exception as e:
            logger.error(f"[{request_id}] 로컬 분석 실패: {e}", exc_info=True)
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

    def _process_video(self, video_path: str, request_id: str) -> List[Dict[str, Any]]:
        extracted_frames_dir = Path(tempfile.mkdtemp())
        try:
            # 1. 프레임 추출 실행 (FPS Sampling)
            frame_paths, extraction_method = self._extract_frames(Path(video_path), extracted_frames_dir)
            logger.info(f"[{request_id}] Extracted {len(frame_paths)} frames via {extraction_method}")
            
            # 2. MAX_FRAMES 제한 (균등 샘플링)
            if len(frame_paths) > MAX_FRAMES:
                indices = np.linspace(0, len(frame_paths) - 1, MAX_FRAMES, dtype=int)
                selected_paths = [frame_paths[i] for i in indices]
            else:
                selected_paths = frame_paths
            
            # 3. 각 프레임 분석
            results = []
            for i, fpath in enumerate(selected_paths):
                # Read image using cv2 (BGR)
                frame = cv2.imread(str(fpath))
                if frame is None:
                    continue
                
                # 얼굴 탐지 및 감정 분석
                res = self._analyze_frame(frame, request_id, i)
                if res:
                    results.append(res)
                    
            return results
            
        finally:
            # Clean up frames
            if extracted_frames_dir.exists():
                shutil.rmtree(extracted_frames_dir)

    def _extract_frames(self, video_path: Path, out_dir: Path) -> Tuple[List[Path], str]:
        output_pattern = str(out_dir / "frame_%06d.jpg")
        
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

        # FPS 2.0 시도
        try:
            frames = run_ffmpeg("fps=2.0")
            if len(frames) > 0:
                return frames, "fps=2.0"
        except subprocess.CalledProcessError:
            pass

        # 실패 시 FPS 1.0 Fallback
        try:
            frames = run_ffmpeg("fps=1.0")
            return frames, "fps=1.0_fallback"
        except subprocess.CalledProcessError:
            return [], "failed"

    # 단일 프레임 이미지에서 가장 신뢰도 높은 얼굴을 찾아 감정을 분석
    def _analyze_frame(self, frame_bgr: np.ndarray, request_id: str, frame_idx: int) -> Optional[Dict[str, Any]]:
        # Debug: Save Raw Frame (Only if DEBUG is On)
        if DEBUG:
            debug_dir = Path("../ai-orchestrator/testing_dev/debug_crops")
            debug_dir.mkdir(parents=True, exist_ok=True)
            raw_filename = f"{request_id}_{frame_idx}_00_raw.jpg"
            cv2.imwrite(str(debug_dir / raw_filename), frame_bgr)

        # 1. 얼굴 탐지 (YOLO)
        yolo_results = self.detector(frame_bgr, verbose=False, device=TORCH_DEVICE)
        if not yolo_results: return None
        result = yolo_results[0]
        
        best_box = None
        max_conf = 0.0
        
        h_img, w_img, _ = frame_bgr.shape
        
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Filter by Class (자동으로 찾은 target_class_id 사용)
            if cls_id != self.target_class_id:
                continue

            # Filter by confidence
            if conf < FACE_CONF_THRESHOLD:
                continue
                
            # Filter by Area logic (너무 작은 얼굴 제외)
            coords = box.xyxy[0].cpu().numpy()
            width = coords[2] - coords[0]
            height = coords[3] - coords[1]
            area_ratio = (width * height) / (h_img * w_img)
            
            if area_ratio < FACE_AREA_MIN_RATIO:
                continue

            if conf > max_conf:
                max_conf = conf
                best_box = coords

        if best_box is None:
            return None

        # 2. Crop with Padding (20%)
        # 학습 환경과 동일한 조건을 위해 패딩 추가 (FACE_PAD_RATIO)
        x1, y1, x2, y2 = map(int, best_box)
        w = x2 - x1
        h = y2 - y1
        
        FACE_PAD_RATIO = 0.20
        pw, ph = int(w * FACE_PAD_RATIO), int(h * FACE_PAD_RATIO)
        
        # Debug: Save BBox Frame
        if DEBUG:
            bbox_frame = frame_bgr.copy()
            cv2.rectangle(bbox_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(bbox_frame, f"Dog {max_conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            bbox_filename = f"{request_id}_{frame_idx}_01_bbox.jpg"
            cv2.imwrite(str(debug_dir / bbox_filename), bbox_frame)
        
        # 패딩 적용
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w_img, x2 + pw)
        y2 = min(h_img, y2 + ph)
        
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0: return None
        
        # 3. Emotion Inference (EfficientNet)
        # BGR -> RGB 변환
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        if DEBUG:
            crop_filename = f"{request_id}_{frame_idx}_02_crop.jpg"
            cv2.imwrite(str(debug_dir / crop_filename), crop_bgr) 
            
        pil_img = Image.fromarray(crop_rgb)
        
        # Transform 적용 및 Batch 차원 추가
        input_tensor = self.transform(pil_img).unsqueeze(0).to(TORCH_DEVICE)

        with torch.no_grad():
            logits = self.classifier(input_tensor)
            probs = F.softmax(logits, dim=-1)[0]
            
        # 결과 매핑
        mapped_probs = {}
        for i, p in enumerate(probs):
            label = self.id2label.get(i, f"unknown_{i}")
            mapped_probs[label] = round(float(p), 4)
        
        return {
            "face_confidence": max_conf,
            "emotion_probs": mapped_probs
        }

    # 여러 프레임의 분석 결과를 종합(Ensemble)하여 최종 감정을 결정
    # 방식: Margin Weighted Voting (Top1 - Top2 차이를 가중치로 사용)
    # 확실한 프레임(Margin이 큰 프레임)에 더 큰 투표권 부여
    def _calculate_ensemble(self, frame_results: List[Dict[str, Any]]) -> Tuple[str, float, Dict[str, float]]:
        final_probs = {"angry": 0.0, "happy": 0.0, "sad": 0.0, "relaxed": 0.0}
        
        records = []
        for res in frame_results:
            probs = res["emotion_probs"]
            sorted_probs = sorted(probs.values(), reverse=True)
            if len(sorted_probs) >= 2:
                margin = sorted_probs[0] - sorted_probs[1]
            else:
                margin = sorted_probs[0] if sorted_probs else 0.0
                
            records.append({
                "probs": probs,
                "weight": margin # 가중치로 Margin 사용
            })
            
        total_weight = sum(r["weight"] for r in records)
        
        # 가중치 합이 0이면 (모든 프레임이 불확실하면) 기본값 relaxed 반환
        if total_weight == 0: 
            return "relaxed", 0.0, final_probs
            
        for r in records:
            w = r["weight"] / total_weight
            for emo in final_probs:
                if emo in r["probs"]:
                    final_probs[emo] += r["probs"][emo] * w
                
        # 승자 결정
        predicted_emotion = max(final_probs, key=final_probs.get)
        confidence = final_probs[predicted_emotion]
        
        # 소수점 반올림
        confidence = round(confidence, 4)
        for k in final_probs:
            final_probs[k] = round(final_probs[k], 4)
        
        return predicted_emotion, confidence, final_probs

    def _generate_narration(self, emotion: str, confidence: float) -> str:
        level = self._get_confidence_level(confidence)
        templates = NARRATION_TEMPLATES.get(emotion, NARRATION_TEMPLATES["relaxed"])
        return templates.get(level, templates["MID"])

    def _get_confidence_level(self, confidence: float) -> str:
        if confidence >= 0.75:
            return "HIGH"
        elif confidence >= 0.50:
            return "MID"
        else:
            return "LOW"

