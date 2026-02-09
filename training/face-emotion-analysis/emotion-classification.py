import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import timm

# =====================
# CONFIG
# =====================
DATASET_DIR = Path("/root/medical_AI/hkh/ellin/dogfacev4_classification/dataset_crop")
OUT_DIR = Path("/root/medical_AI/hkh/ellin/dogfacev4_classification/runs_cls/effnet_b0_emotion")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = OUT_DIR / "best.pt"
LAST_MODEL_PATH = OUT_DIR / "last.pt"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 30
EPOCHS_FINE = 15
LR_HEAD = 1e-4
LR_FINE = 1e-5
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# DATA TRANSFORM
# =====================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# DATASET & DATALOADER
# =====================
train_dir = DATASET_DIR / "train"
val_dir = DATASET_DIR / "val"
test_dir = DATASET_DIR / "test"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

num_classes = len(train_dataset.classes)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("Classes:", train_dataset.classes)
print("Model outputs will be saved to:", str(OUT_DIR))

# =====================
# MODEL
# =====================
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=num_classes
)

model = model.to(DEVICE)

# =====================
# LOSS / OPTIMIZER
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR_HEAD)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-6
)

# =====================
# TRAIN / VALIDATE
# =====================
def train_one_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def validate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total


def evaluate_metrics(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total if total > 0 else 0.0
    f1 = f1_macro(all_targets, all_preds, num_classes)
    return acc, f1


def f1_macro(y_true, y_pred, num_classes):
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    f1s = []
    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / num_classes if num_classes > 0 else 0.0


def get_gflops(model):
    gflops = None
    cfg = getattr(model, "default_cfg", {}) or {}
    gflops = cfg.get("gflops")
    if gflops is None:
        cfg = getattr(model, "pretrained_cfg", {}) or {}
        gflops = cfg.get("gflops")
    if gflops is None:
        try:
            from thop import profile
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
            gflops = flops / 1e9
        except Exception:
            gflops = None
    return gflops


def model_size_mb(model):
    total_bytes = 0
    for t in model.state_dict().values():
        total_bytes += t.numel() * t.element_size()
    return total_bytes / (1024 ** 2)


def measure_inference_latency_ms(model, loader, warmup_batches=2):
    model.eval()
    total_time = 0.0
    total_images = 0
    batch_idx = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)

            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(x)

            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()

            if batch_idx >= warmup_batches:
                total_time += (end - start)
                total_images += x.size(0)

            batch_idx += 1

    if total_images == 0:
        return 0.0
    return (total_time / total_images) * 1000.0


# =====================
# STAGE 1: HEAD TRAIN
# =====================
print("Stage 1: Head Training")

best_val_acc = -1.0
for epoch in range(EPOCHS_HEAD):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    scheduler.step(val_loss)

    print(f"[{epoch+1}/{EPOCHS_HEAD}] "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)

# =====================
# STAGE 2: FINE TUNING
# =====================
print("Stage 2: Fine Tuning")

optimizer = Adam(model.parameters(), lr=LR_FINE)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-6
)

for epoch in range(EPOCHS_FINE):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    scheduler.step(val_loss)

    print(f"[Fine {epoch+1}/{EPOCHS_FINE}] "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)

torch.save(model.state_dict(), LAST_MODEL_PATH)

print("Training Complete")
print("Best model path:", str(BEST_MODEL_PATH))
print("Last model path:", str(LAST_MODEL_PATH))

# =====================
# FINAL METRICS (Test)
# =====================
if BEST_MODEL_PATH.exists():
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

test_acc, test_f1 = evaluate_metrics(model, test_loader)
params = sum(p.numel() for p in model.parameters())
gflops = get_gflops(model)
latency_ms = measure_inference_latency_ms(model, test_loader)
size_mb = model_size_mb(model)

print("\nFinal Model Report")
print(f"Parameters           : {params:,}")
print(f"GFLOPs               : {gflops:.4f}" if gflops is not None else "GFLOPs               : n/a")
print(f"Accuracy (ref)       : {test_acc:.4f}")
print(f"F1-Score (macro)     : {test_f1:.4f}")
print(f"Inference Latency (ms): {latency_ms:.2f}")
print(f"Size (MB)            : {size_mb:.2f}")
