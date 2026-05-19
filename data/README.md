# Demonstration Dataset

This directory contains a small subset of the Kaggle CIFAKE dataset used exclusively for live demonstration and real-world evaluation.

## Structure

These images are completely independent of the training and validation sets used in the Colab notebook. We deliberately held back 200 unseen images to ensure an unbiased, live test during the project presentation.

```
data/
├── ai/       # 100 AI-generated images (0.jpg to 99.jpg)
├── real/     # 100 Authentic/Human-made images (0.jpg to 99.jpg)
└── README.md # This file
```

## Usage
Simply drag and drop any of these 200 images directly into the Gradio Web UI (`http://127.0.0.1:7860`) to observe the Vision Transformer inferencing in real-time.
