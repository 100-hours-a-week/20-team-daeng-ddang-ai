from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
import csv
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
import timm


# =========================
# Paths
# =========================
OUT_ROOT = Path("/root/medical_AI/hkh/ellin/dogface_test/result/v4/3_e2e_video_ffmpeg_s3_2")
FRAME_ROOT = OUT_ROOT / "frames"

FACE_MODEL = "/root/medical_AI/hkh/ellin/dogfacev4_detection/runs/detect/runs/detect/yolov10n_dogface_v4/weights/best.pt"
EMOTION_MODEL = "/root/medical_AI/hkh/ellin/dogfacev4_classification/runs_cls/effnet_b0_emotion_ft_lr1e6/best_ft.pt"

VIDEO_URLS = [
    "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set2/DogOrCat01.mp4",
    "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set2/DogOrRhino.mp4",
    "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set2/DogOrSquirrel.mp4",
    "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set2/human_face_02.mp4",
    "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set2/human_face01.mp4",
]

# =========================
# Params
# =========================
FPS = 2                 # fps uniform sampling
TOP_K = 3
FACE_CLASS_ID = 0
FACE_CONF = 0.2
FACE_PAD_RATIO = 0.20
IMG_SIZE = 224
DOG_DETECT_RATE_TH = 0.40
DOG_MEDIAN_CONF_TH = 0.70

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IDX_TO_LABEL = {
    0: "angry",
    1: "happy",
    2: "relaxed",
    3: "sad",
}


# =========================
# Utils
# =========================
def url_to_filename(url: str) -> str:
    return Path(urlparse(url).path).name


def extract_gt_label_from_stem(stem: str) -> str:
    parts = stem.split("_")
    return parts[0].lower() if len(parts) > 1 else "unknown"


def extract_frames_fps(video_source: str, out_dir: Path, fps: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", str(video_source),
        "-vf", f"fps={fps}",
        str(out_dir / "frame_%04d.jpg"),
        "-hide_banner",
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)


def crop_with_padding(img, box, ratio):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pw, ph = int(bw * ratio), int(bh * ratio)
    return img[
        max(0, y1 - ph):min(h, y2 + ph),
        max(0, x1 - pw):min(w, x2 + pw),
    ]

def draw_bbox_with_conf(img, box, conf, color=(0, 255, 0)):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"dog_face {conf:.2f}"
    y_text = max(0, y1 - 8)
    cv2.putText(
        img,
        label,
        (x1, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def load_emotion_model(path):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=len(IDX_TO_LABEL),
    )
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def aggregate_margin_weighted_topk(records, k):
    topk = sorted(records, key=lambda x: x["margin"], reverse=True)[:k]
    weights = np.array([r["margin"] for r in topk])
    probs = np.stack([r["prob"] for r in topk])
    return np.sum(probs * weights[:, None], axis=0) / np.sum(weights)


# =========================
# Main
# =========================
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FRAME_ROOT.mkdir(parents=True, exist_ok=True)

    frame_csv = OUT_ROOT / "emotion_results_frame.csv"
    video_csv = OUT_ROOT / "emotion_results_video.csv"

    print("🚀 Loading models...")
    face_model = YOLO(FACE_MODEL)
    emotion_model = load_emotion_model(EMOTION_MODEL)

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    frame_rows = []
    video_rows = []

    for url in VIDEO_URLS:
        filename = url_to_filename(url)
        if not filename:
            print("⚠️  Skip invalid url:", url)
            continue

        video_stem = Path(filename).stem
        gt_label = extract_gt_label_from_stem(video_stem)
        print(f"\n▶ Processing {filename} | GT={gt_label}")

        video_frame_dir = FRAME_ROOT / video_stem
        if video_frame_dir.exists():
            shutil.rmtree(video_frame_dir)

        extract_frames_fps(url, video_frame_dir, FPS)
        frames = sorted(video_frame_dir.glob("*.jpg"))

        frame_records = []
        frame_rows_video = []
        face_confs = []

        for img_path in frames:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            res = face_model.predict(
                source=frame,
                conf=FACE_CONF,
                classes=[FACE_CLASS_ID],
                verbose=False,
            )
            if not res[0].boxes:
                continue

            b = max(res[0].boxes, key=lambda x: float(x.conf[0]))
            face_conf = float(b.conf[0]) if b.conf is not None else 0.0
            x1, y1, x2, y2 = map(int, b.xyxy[0])

            annotated = frame.copy()
            draw_bbox_with_conf(annotated, (x1, y1, x2, y2), face_conf)
            cv2.imwrite(str(img_path), annotated)

            face_confs.append(face_conf)

            face = crop_with_padding(frame, (x1, y1, x2, y2), FACE_PAD_RATIO)
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_t = tfm(face_rgb).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                prob = F.softmax(emotion_model(face_t), dim=1)[0].cpu().numpy()

            sorted_prob = np.sort(prob)[::-1]
            margin = float(sorted_prob[0] - sorted_prob[1])
            pred_label = IDX_TO_LABEL[int(np.argmax(prob))]

            frame_records.append({"prob": prob, "margin": margin})

            frame_rows_video.append({
                "video_name": video_stem,
                "frame": img_path.name,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "dog_face_conf": f"{face_conf:.4f}",
                "margin": f"{margin:.4f}",
                "angry_prob": f"{prob[0]:.4f}",
                "happy_prob": f"{prob[1]:.4f}",
                "relaxed_prob": f"{prob[2]:.4f}",
                "sad_prob": f"{prob[3]:.4f}",
            })

        total_frames = len(frames)
        detected_frames = len(face_confs)
        detect_rate = (detected_frames / total_frames) if total_frames > 0 else 0.0
        median_conf = float(np.median(face_confs)) if face_confs else 0.0
        dog_present = (detect_rate >= DOG_DETECT_RATE_TH) and (median_conf >= DOG_MEDIAN_CONF_TH)

        print(
            f"🐶 DOG_GATE | rate:{detect_rate:.3f} "
            f"(n={detected_frames}/{total_frames}) "
            f"median_conf:{median_conf:.3f} -> {dog_present}"
        )

        if not dog_present:
            print("❌ DOG_GATE FILTERED → NO_RESULT")
            video_rows.append({
                "video_name": video_stem,
                "gt_label": gt_label,
                "final_pred_label": "",
                "happy_pct": "0.00",
                "angry_pct": "0.00",
                "relaxed_pct": "0.00",
                "sad_pct": "0.00",
                "num_used_frames": 0,
                "detect_rate": f"{detect_rate:.4f}",
                "median_conf": f"{median_conf:.4f}",
                "detected_frames": detected_frames,
                "total_frames": total_frames,
                "dog_present": 0,
                "correct": "",
            })
            continue

        frame_rows.extend(frame_rows_video)

        if not frame_records:
            print("❌ NO VALID FRAMES → NO_RESULT")
            video_rows.append({
                "video_name": video_stem,
                "gt_label": gt_label,
                "final_pred_label": "",
                "happy_pct": "0.00",
                "angry_pct": "0.00",
                "relaxed_pct": "0.00",
                "sad_pct": "0.00",
                "num_used_frames": 0,
                "detect_rate": f"{detect_rate:.4f}",
                "median_conf": f"{median_conf:.4f}",
                "detected_frames": detected_frames,
                "total_frames": total_frames,
                "dog_present": 1,
                "correct": "",
            })
            continue

        final_prob = aggregate_margin_weighted_topk(frame_records, TOP_K)
        final_pct = final_prob * 100
        final_label = IDX_TO_LABEL[int(np.argmax(final_prob))]

        print(
            f"🎯 FINAL | "
            f"happy:{final_pct[1]:.1f}% "
            f"angry:{final_pct[0]:.1f}% "
            f"relaxed:{final_pct[2]:.1f}% "
            f"sad:{final_pct[3]:.1f}%"
        )

        video_rows.append({
            "video_name": video_stem,
            "gt_label": gt_label,
            "final_pred_label": final_label,
            "happy_pct": f"{final_pct[1]:.2f}",
            "angry_pct": f"{final_pct[0]:.2f}",
            "relaxed_pct": f"{final_pct[2]:.2f}",
            "sad_pct": f"{final_pct[3]:.2f}",
            "num_used_frames": min(TOP_K, len(frame_records)),
            "detect_rate": f"{detect_rate:.4f}",
            "median_conf": f"{median_conf:.4f}",
            "detected_frames": detected_frames,
            "total_frames": total_frames,
            "dog_present": 1,
            "correct": "" if gt_label == "unknown" else int(final_label == gt_label),
        })

    if frame_rows:
        with open(frame_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=frame_rows[0].keys())
            writer.writeheader()
            writer.writerows(frame_rows)
    else:
        print("⚠️ No frame-level rows to write.")

    if video_rows:
        with open(video_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=video_rows[0].keys())
            writer.writeheader()
            writer.writerows(video_rows)

    print("\n" + "=" * 80)
    print(f"✅ Frame CSV : {frame_csv}")
    print(f"✅ Video CSV : {video_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()