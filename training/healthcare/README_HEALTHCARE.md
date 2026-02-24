# 🐶 Healthcare Analysis Service Guide

This document describes how to implement the `healthcare-service` for dog pose estimation and metrics analysis, and how to integrate it with the `ai-orchestrator`.

## 1. Architecture Overview (Proposed)

Similar to the existing `face-service`, the `healthcare-service` should be implemented as a separate FastAPI server that handles heavy model inference tasks. The `ai-orchestrator` will route requests to this service.

### Recommended Folder Structure
```bash
/healthcare-service
├── run.py                 # Entry point (uvicorn)
├── app/
│   ├── main.py            # FastAPI app & S3 logic
│   ├── core/
│   │   └── config.py      # App configuration (S3 credentials, paths)
│   ├── schemas/
│   │   └── health_schema.py # Request/Response models
│   └── services/
│       └── health_analyzer.py # Python wrapper for analyze_health.py
├── scripts/
│   └── analyze_health.py  # [IMPORTANT] Core logic script (from this repo)
├── models/
│   └── best.pt            # Downloaded from Hugging Face
└── requirements.txt
```

### Integration Flow
1. **User Request** -> `ai-orchestrator`
2. `ai-orchestrator` checks task type (`healthcare`).
3. `ai-orchestrator` calls `healthcare-service` (HTTP POST) with video URL.
4. **`healthcare-service`**:
   - Downloads video.
   - Runs `analyze_health.py`.
   - Uploads result video (overlay) to S3.
   - Returns JSON analysis results with S3 URL.
5. `ai-orchestrator` -> **User Response**.

---

## 2. Core Implementation Guide

### A. The Analysis Script (`scripts/analyze_health.py`)
This script must be included in your service repo. It performs:
- **YOLO Pose Estimation**: Detects 17+ keypoints.
- **Metrics Calculation**: Rhythm, Balance, Mobility, Stability, Patella Risk.
- **Overlay Generation**: Creates a visual feedback video.

### B. Environment Variables (`.env`)
The service needs these variables to function correctly, especially for S3 uploads and debug modes.

```ini
# Service Config
PORT=8002
DEBUG_MODE=false  # Set to 'true' for detailed processing stats & debug info

# AWS S3 Credentials (for uploading overlay videos)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=wJalr...
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=your-bucket-name
```

### C. S3 Upload Logic (Python Snippet)
Since `analyze_health.py` only saves locally, the **Service Layer (`app/main.py` or `services/health_analyzer.py`)** must handle the upload.

```python
import boto3
import os

def upload_to_s3(local_path, bucket, s3_key):
    s3 = boto3.client('s3', ...)
    try:
        s3.upload_file(local_path, bucket, s3_key)
        # Return Public URL (or Presigned URL)
        return f"https://{bucket}.s3.amazonaws.com/{s3_key}"
    except Exception as e:
        print(f"Upload failed: {e}")
        return None
    finally:
        # Crucial: Remove local temp file after upload
        if os.path.exists(local_path):
            os.remove(local_path)
```

---

## 3. API Response Schema

### Production Mode (`DEBUG_MODE=false`)
Standard response format for the frontend app.

```json
{
  "analysis_id": "uuid-string",
  "dog_id": 123,
  "analyze_at": "ISO8601-timestamp",
  "result": {
    "overall_score": 53,
    "overall_risk_level": "medium",
    "summary": "종합점수 53점. 보행 패턴에서 미세한 이상 신호가 감지되었습니다..."
  },
  "metrics": {
    "patella_risk_signal": { "level": "medium", "score": 75, "description": "..." },
    "gait_balance":        { "level": "fair", "score": 61, "description": "..." },
    "knee_mobility":       { "level": "stiff", "score": 35, "description": "..." },
    "gait_stability":      { "level": "unstable", "score": 15, "description": "..." },
    "gait_rhythm":         { "level": "consistent", "score": 81, "description": "..." }
  },
  "artifacts": {
    "keypoint_overlay_video_url": "https://s3.../overlay_uuid.mp4"
  },
  "error_code": null
}
```

### Debug Mode (`DEBUG_MODE=true`)
Includes detailed internal calculations and performance metrics.

```json
{
  "analysis_id": "...",
  // [Added in Debug] Processing stats at root level
  "processing": {
    "analysis_time_ms": 1240,
    "video_duration_sec": 5.4,
    "frames_sampled": 162,
    "fps_used": 30
  },
  "result": { ... },
  "metrics": { ... },
  "artifacts": {
    "keypoint_overlay_video_url": "...",
    // [Added in Debug] detailed math logs
    "debug": {
      "missing_ratio": { "tail": 0.0, ... },
      "height_scale": 45.2,
      "stance_debug": { "retry_triggered": true, ... },
      "rear_asym": 0.04
    }
  }
}
```

---

## 4. Error Handling `DOG_NOT_DETECTED`

If the video quality is too poor or no dog is found, the service returns a minimal error payload:

```json
{
  "analysis_id": "uuid-string",
  "dog_id": 123,
  "analyze_at": "ISO8601-timestamp",
  "error_code": "DOG_NOT_DETECTED"
}
```

---

## 5. Model Weights
Download the `best.pt` file from the Hugging Face repository and place it in the `models/` directory.

- **Repo**: `huggingface.co/20-team-daeng-ddang-ai/dog-pose-estimation`
- **File**: `best.pt`
