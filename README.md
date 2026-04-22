# Chest CT Cancer Classifier

AI-powered lung cancer classification from chest CT scan images using deep learning with explainability through Grad-CAM visualizations.

## Overview

This project implements a Convolutional Neural Network (CNN) based classifier for detecting and classifying lung cancer types from chest CT scan slices. It uses transfer learning with ResNet50 pretrained on ImageNet, fine-tuned for pulmonary lesion classification. The model provides Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps to visually explain which regions of the CT scan influenced the prediction.

**Disclaimer**: This is a research/educational tool only. Not for clinical diagnosis. Always consult a licensed radiologist for medical decisions.

## Classes

- Adenocarcinoma
- Large Cell Carcinoma
- Normal
- Squamous Cell Carcinoma

## Project Structure

```
CT-Scan_Disease/
├── app.py                    # Gradio web interface
├── src/
│   ├── __init__.py
│   ├── model.py              # ResNet50-based classifier
│   ├── dataset.py           # Data loading and augmentation
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation with metrics & Grad-CAM
│   └── grad_cam.py           # Grad-CAM implementation
└── models/                   # Saved model checkpoints (created after training)
    └── best_model.pth
```

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Deep Learning Framework | PyTorch | 2.1+ |
| Computer Vision | TorchVision | 0.16+ |
| Model Architecture | ResNet50 (Transfer Learning) | - |
| Loss Function | CrossEntropyLoss | - |
| Optimizer | AdamW | - |
| Image Augmentation | Albumentations | - |
| Web UI | Gradio | - |
| Evaluation | scikit-learn | - |

## Installation

```bash
pip install torch torchvision gradio albumentations opencv-python pillow scikit-learn seaborn matplotlib pandas numpy
```

## Data Preparation

Organize your dataset as ImageFolder structure:
```
data_dir/
├── adenocarcinoma/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── large.cell.carcinoma/
├── normal/
└── squamous.cell.carcinoma/
```

The dataset is split 80/20 for training/validation automatically.

## Usage

### Training

```bash
cd CT-Scan_Disease
python -m src.train
```

Training configuration (in `src/train.py`):
- Data directory: Update `DATA_DIR`
- Device: Auto-detects MPS (Apple Silicon) / CUDA / CPU
- Epochs: 5 (default)
- Batch size: 16
- Learning rate: 1e-4
- Weight decay: 1e-2

### Evaluation

```bash
python -m src.evaluate
```

Evaluation generates:
- Confusion matrix
- ROC curves (one-vs-rest)
- Per-class precision, recall, F1 scores
- Grad-CAM visualizations for sample images
- JSON report with all metrics

### Web Interface

Launch the Gradio app:
```bash
python app.py
```

Access at: http://localhost:7860

Features:
- Upload CT scan slice
- Get prediction + confidence scores
- View Grad-CAM heatmap overlay

### Importing in Python

```python
from src.model import ChestCTClassifier
from src.grad_cam import GradCAM
from src.dataset import get_dataloaders

# Load model
model = ChestCTClassifier(num_classes=4)
model.load_state_dict(torch.load("models/best_model.pth"))

# Get predictions
grad_cam = GradCAM(model)
heatmap = grad_cam.generate_cam(input_tensor)
```

## Model Architecture

ResNet50 backbone with custom classification head:
1. Load pretrained ResNet50 (ImageNet weights)
2. Freeze early layers (feature extraction)
3. Unfreeze layer4 for fine-tuning
4. Replace final FC layer:
   - 2048 → 512 (Linear + ReLU + Dropout)
   - 512 → 4 (Linear output)

## Data Augmentation

Training transforms:
- Resize to 224x224
- Random horizontal flip
- Random rotation (±10°)
- ImageNet normalization

Validation transforms:
- Resize to 224x224
- ImageNet normalization

## Evaluation Metrics

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Confusion Matrix
- ROC-AUC (per class)
- Grad-CAM heatmaps

## Requirements

- Python 3.8+
- PyTorch 2.1+
- TorchVision 0.16+
- Gradio
- Albumentations
- OpenCV
- scikit-learn
- Matplotlib/Seaborn
- Pandas
- NumPy
- Pillow

## License

MIT License
