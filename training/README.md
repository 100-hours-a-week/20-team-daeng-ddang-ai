# Training & Validation Scripts

This directory contains the **source code** for training and validating your AI models.
Do **NOT** upload heavy model weights (`.pt`) or full datasets here.

## 📂 Structure

```text
training/
├── emotion-classification/
│   ├── train.py
│   ├── dataset.py
│   └── requirements.txt
├── dog-pose-estimation/
│   └── train_yolo.py
├── llm/
│   └── finetune.py
└── urgent-mission/
    └── experiment.py
```

## ⚠️ Important
*   **Datasets**: Keep them local or in a cloud bucket. Do not commit `data/` folders.
*   **Model Weights**: Checkpoints (`runs/`, `checkpoints/`) are ignored by `.gitignore`.
*   **Metrics**: After training, move your `config.yaml` and `result.json` to the `../metrics` folder!
