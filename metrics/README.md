# Metrics & Model Results

This directory tracks the performance (metrics) and configurations of your AI models.
The actual model weights (e.g., `.pt`, `.onnx`) should be uploaded to **HuggingFace Hub**.

## � Project Overview & File Guidelines

### 1. 😠 Emotion Analysis (표정 분석)

This project has two parts: detecting the dog's face and classifying its emotion.

| Component | Model | HuggingFace Hub Uploads (Weights) | GitHub Metrics Uploads (Here) |
| :--- | :--- | :--- | :--- |
| **Dog Face Detection** | **YOLOv10n** | `best.pt`<br>`class.json`<br>`preprocess.json` | `config.yaml` (Training Args)<br>`results.csv` (mAP, Precision, Recall)<br>`plots/confusion_matrix.png` |
| **Emotion Classification** | **EfficientNet-B0** | `best.pt`<br>`class.json`<br>`inference_config.json` | `config.yaml` (Learning Rate, Epochs)<br>`results.csv` (Accuracy, F1-Score)<br>`plots/loss_curve.png` |

---

### 2. 🏥 Healthcare (헬스케어 / 관절 진단)

| Component | Model | HuggingFace Hub Uploads | GitHub Metrics Uploads |
| :--- | :--- | :--- | :--- |
| **Pose Estimation** | **YOLO-Pose** (v8/v11 etc.) | `best.pt`<br>`data.yaml` | `config.yaml`<br>`results.csv` (PCK, mAP)<br>`plots/val_batch0_labels.jpg` |

---

### 3. 💬 Vet Chatbot (수의사 챗봇)

| Component | Model | HuggingFace Hub Uploads | GitHub Metrics Uploads |
| :--- | :--- | :--- | :--- |
| **LLM** | **Llama 3 / Qwen 2.5 (8B)** | `adapter_model.bin`<br>`tokenizer.json`<br>`config.json` | `training_args.json`<br>`eval_results.json` (Perplexity, ROUGE)<br>`sample_generations.txt` |

---

### 4. ⚡ Urgent Mission (돌발 미션)

| Component | Service | Management Strategy |
| :--- | :--- | :--- |
| **General AI** | **Gemini 2.5 Flash** (API) | Store prompts and test cases here.<br>`prompts/v1_system_prompt.txt`<br>`test_cases/example_inputs.json` |

## 📂 Recommended Directory Structure

```text
metrics/
├── emotion-classification/
│   ├── v1_yolov10n_baseline/
│   │   ├── config.yaml
│   │   ├── results.csv
│   │   └── README.md (Link to HuggingFace model)
│   └── v1_efficientnet_b0/
│       ├── config.yaml
│       ├── results.csv
│       └── README.md
├── dog-pose-estimation/
│   └── v1_yolo_pose/
│       ├── config.yaml
│       └── results.csv
├── llm/
│   └── v1_finetune_8b/
│       ├── training_args.json
│       └── eval_results.json
└── urgent-mission/ (Optional)
    └── prompts/
        └── v1_prompt.txt
```

## 🚀 How to Automate?

When training models, add a few lines of code to save your configuration and results to this folder.
For example, in Python:

```python
# Save Config
import yaml
with open("metrics/emotion-classification/v1/config.yaml", "w") as f:
    yaml.dump(args, f)

# Save Results
import json
with open("metrics/emotion-classification/v1/results.json", "w") as f:
    json.dump({"accuracy": 0.95, "f1": 0.92}, f)
```
