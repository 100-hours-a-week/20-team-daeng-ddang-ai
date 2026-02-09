from ultralytics import YOLO
import torch

def main():
    # =========================
    # 설정
    # =========================
    MODEL_NAME = "yolov10n.pt"  # COCO pretrained
    DATA_YAML = "/root/medical_AI/hkh/ellin/dogfacev4_detection/dataset/data.yaml"

    IMG_SIZE = 640
    EPOCHS = 50
    BATCH = 16
    DEVICE = 0  # GPU: 0, CPU: "cpu"
    PROJECT = "runs/detect"
    NAME = "yolov10n_dogface_v4"

    # =========================
    # GPU 확인 (안전 체크)
    # =========================
    print("PyTorch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ GPU 사용 불가, CPU로 실행됩니다.")

    # =========================
    # 모델 로드
    # =========================
    model = YOLO(MODEL_NAME)

    # =========================
    # 학습 시작
    # =========================
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,

        lr0=0.001,
        lrf=0.01,
        optimizer="SGD",

        verbose=True
    )

    print("✅ 학습 완료")

if __name__ == "__main__":
    main()
