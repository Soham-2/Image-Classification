# Dataset

## Download Instructions

1. Download the AI vs Human image classification dataset from Kaggle
2. Extract the dataset into this `data/` directory
3. Ensure the following folder structure:

```
data/
├── train/
│   ├── ai/       # AI-generated images
│   └── real/     # Real/human-made images
├── test/
│   ├── ai/       # AI-generated images (for evaluation)
│   └── real/     # Real/human-made images (for evaluation)
└── README.md     # This file
```

> **Note**: The class folder names (`ai` and `real`) are detected automatically.
> If your dataset uses different names (e.g., `FAKE` / `REAL` or `ai_generated` / `human`),
> update `CLASS_NAMES` in `src/config.py` accordingly.
