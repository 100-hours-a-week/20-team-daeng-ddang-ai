# Training & Validation Scripts

This directory contains the **source code** for training and validating your AI models.
Do **NOT** upload heavy model weights (`.pt`) or full datasets here.

## 📂 Structure (Current)

```text
training/
├── face-emotion-analysis/
│   ├── dog-detection.py
│   ├── emotion-classification.py
│   └── model-test.py
├── healthcare/
│   ├── analyze_health.py
│   └── requirements.txt
├── chatbot/
│   ├── chatbot_core.py
│   └── requirements.txt
└── requirements.txt
```

## ⚠️ Important
*   **Datasets**: Keep them local or in a cloud bucket. Do not commit `data/` folders.
*   **Model Weights**: Checkpoints (`runs/`, `checkpoints/`) are ignored by `.gitignore`.
*   **Metrics**: After training, move your `config.yaml` and `result.json` to the `../metrics` folder!
