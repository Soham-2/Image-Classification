<div align="center">
  
# 🧠 Neural Image Forensics
**A High-Accuracy AI Image Detection System**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art classifier designed to detect subtle AI generated artifacts and diffusion patterns invisible to the human eye, featuring a slick dark-mode UI.

</div>

---

## ✨ Features

- **High Precision:** Achieves **98.18%** accuracy detecting CIFAKE artificial vs. real images.
- **Advanced Architecture:** Powered by our custom **Vision Transformer (ViT-Base)** model, trained exclusively via our Colab pipeline.
- **Cyberpunk UI:** A fully custom, sleek, futuristic web interface built with pure CSS and Gradio Blocks.
- **Zero-Config Local Inference:** Runs entirely on your local machine to preserve privacy. The application leverages the weights generated from our Colab training to perform immediate offline inference!

---

## 📁 Project Structure

```text
├── app.py                         # 🚀 Main entry point - the Gradio Web Application
├── Colab_ViT_Training_120k.ipynb  # 📓 Complete training pipeline and experimentation notebook
├── requirements.txt               # 📦 All Python dependencies
├── data/                          # 📂 Reserved directory for local datasets (if needed)
└── README.md                      # 📖 Project documentation
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
*(Skip this step if you have already downloaded the source code)*
```bash
git clone https://github.com/yourusername/neural-image-forensics.git
cd neural-image-forensics
```

### 2. Set Up a Virtual Environment (Recommended)
Isolating dependencies ensures you do not conflict with other system Python packages.
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
Install all necessary ML, PyTorch, and Gradio packages:
```bash
pip install -r requirements.txt
```

### 4. Launch the AI Detection System
Start the local Gradio server. *(Note: The application will seamlessly load our ~340MB fine-tuned Vision Transformer model into memory).*
```bash
python app.py
```

Click the resulting local URL (usually `http://127.0.0.1:7860`) in your terminal to open the stylized Image Forensics interface.

---

## 🔬 Training Pipeline (Optional)

If you are interested in reproducing the model training from scratch instead of just running inference:

1. Open `Colab_ViT_Training_120k.ipynb` in **Google Colab** (recommended for access to free T4 GPUs).
2. The notebook covers everything from data loading and augmentation for the **120k CIFAKE dataset** to fine-tuning the ViT model.
3. The resulting `.safetensors` model weights were then exported and bound directly to our UI inference pipeline.

---

## 📝 Specifications at a Glance

| Spec | Details |
|------|---------|
| **Base Model** | Vision Transformer (ViT-Base) |
| **Dataset** | CIFAKE (120,000 Images at 32x32 Resolution) |
| **Accuracy** | ~98.18% |
| **Hardware Used** | NVIDIA Tesla T4 |

---

## 📜 License
This project is open-source and intended for educational & research purposes.
