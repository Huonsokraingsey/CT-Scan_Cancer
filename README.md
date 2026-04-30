# Chest CT Lung Cancer Classifier

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
├── app.py                    # Gradio web interface with dark theme
├── src/
│   ├── __init__.py
│   ├── model.py              # ResNet50-based classifier with custom head
│   ├── dataset.py           # Data loading with augmentation
│   ├── train.py             # Training pipeline with early stopping
│   ├── evaluate.py          # Evaluation with Grad-CAM & metrics
│   └── grad_cam.py          # Grad-CAM implementation
└── models/                   # Saved model checkpoints (created after training)
    └── best_model.pth
```

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Deep Learning Framework | PyTorch | 2.1+ |
| Computer Vision | TorchVision | 0.16+ |
| Model Architecture | ResNet50 (Transfer Learning) | - |
| Loss Function | CrossEntropyLoss + Label Smoothing | - |
| Optimizer | AdamW | - |
| Learning Rate Scheduler | ReduceLROnPlateau | - |
| Image Augmentation | torchvision.transforms | - |
| Web UI | Gradio | - |
| Evaluation | scikit-learn | - |

## Installation

```bash
pip install torch torchvision gradio opencv-python pillow scikit-learn seaborn matplotlib pandas numpy albumentations
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
- Epochs: 20 (with early stopping, patience=4)
- Batch size: 16
- Learning rate: 0.001 with ReduceLROnPlateau
- Weight decay: 0.01
- Class weights for imbalanced data
- Label smoothing: 0.1

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
- Custom dark theme UI

### Importing in Python

```python
import torch
from src.model import ChestCTClassifier
from src.grad_cam import GradCAM
from src.dataset import get_dataloaders

# Load model
model = ChestCTClassifier(num_classes=4)
model.load_state_dict(torch.load("models/best_model.pth", weights_only=True))

# Get predictions
grad_cam = GradCAM(model)
heatmap = grad_cam.generate_cam(input_tensor)
```

## Model Architecture

ResNet50 backbone with custom classification head:

**Understanding the Architecture Numbers:**

The numbers `2048 → 512 → 256 → 4` represent the **fully connected (FC) layers** - the "classifier" part that sits at the end of ResNet50:

| Layer | Input Dim | Output Dim | Description |
|-------|-----------|-------------|-------------|
| FC1 | 2048 | 512 | First hidden layer (ResNet50's output) |
| FC2 | 512 | 256 | Second hidden layer |
| FC3 | 256 | 4 | Output layer (4 lung cancer classes) |

**Explanation:**
- **2048**: ResNet50's final convolutional layer outputs a 2048-dimensional feature vector (the "deep features" it learned from ImageNet)
- **512**: First linear layer compresses 2048 features into 512, followed by ReLU activation and 60% dropout to prevent overfitting
- **256**: Second linear layer further compresses to 256 features, with ReLU and 40% dropout
- **4**: Final layer maps 256 features to 4 classes (Adenocarcinoma, Large Cell Carcinoma, Normal, Squamous Cell Carcinoma)

**Why this architecture?**
1. Progressive dimensionality reduction (2048→512→256→4) allows the model to learn increasingly abstract representations
2. Dropout (0.6, 0.4) prevents overfitting by randomly "dropping" neurons during training
3. ReLU introduces non-linearity to help learn complex patterns in CT scans

**Full architecture:**
1. Load pretrained ResNet50 (ImageNet weights)
2. Freeze early layers (feature extraction)
3. Unfreeze layer3 and layer4 for fine-tuning
4. Replace final FC layer with custom head:
   - 2048 → 512 (Linear + ReLU + Dropout 0.6)
   - 512 → 256 (Linear + ReLU + Dropout 0.4)
   - 256 → 4 (Linear output)

## Data Augmentation

Training transforms:
- Resize to 224x224
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- ColorJitter (brightness=0.2, contrast=0.2)
- RandomAffine (translate=0.1)
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
- ROC-AUC (per class, one-vs-rest)
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
