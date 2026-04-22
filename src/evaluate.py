import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from src.dataset import get_dataloaders
from src.model import ChestCTClassifier
from src.grad_cam import GradCAM

def evaluate_with_gradcam(model_path, data_dir, test_dir=None, num_samples=10, output_dir="results"):
    """
    Comprehensive evaluation with Grad-CAM visualizations.
    
    Args:
        model_path: Path to trained .pth model
        data_dir: Path to dataset (with train/val folders)
        test_dir: Path to test dataset (optional, uses val if not provided)
        num_samples: Number of test images to visualize with Grad-CAM
        output_dir: Directory to save results
    """
    
    # Setup
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Evaluating model: {model_path}")
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Test directory: {test_dir}")
    print(f"🖥️  Device: {DEVICE}")
    
    # Load data
    train_loader, val_loader, test_loader, classes = get_dataloaders(data_dir, batch_size=16, test_dir=test_dir)
    if test_loader is None:
        test_loader = val_loader
        print("⚠️  Using validation set as test set (no test_dir provided)")
    print(f"📂 Classes: {classes}")
    print(f"📊 Test samples: {len(test_loader.dataset)}")
    
    # Load model
    model = ChestCTClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE).eval()
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model)
    
    # === PART 1: Standard Metrics ===
    print("\n" + "="*60)
    print("📊 PART 1: Standard Classification Metrics")
    print("="*60)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'per_class': {}
    }
    
    print(f"✅ Accuracy:  {metrics['accuracy']:.4f}")
    print(f"✅ Precision: {metrics['precision']:.4f}")
    print(f"✅ Recall:    {metrics['recall']:.4f}")
    print(f"✅ F1-Score:  {metrics['f1']:.4f}")
    
    # Per-class metrics
    print("\n📋 Per-Class Performance:")
    for i, cls in enumerate(classes):
        tp = ((y_true == i) & (y_pred == i)).sum()
        fp = ((y_true != i) & (y_pred == i)).sum()
        fn = ((y_true == i) & (y_pred != i)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        metrics['per_class'][cls] = {'precision': prec, 'recall': rec, 'f1': f1}
        print(f"  {cls:25s} | P: {prec:.3f} | R: {rec:.3f} | F1: {f1:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = f"{output_dir}/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"💾 Confusion matrix saved: {cm_path}")
    
    # === PART 2: Grad-CAM Visualizations ===
    print("\n" + "="*60)
    print("🎨 PART 2: Grad-CAM Explainability")
    print("="*60)
    
    # Get sample images from test set
    test_dataset = test_loader.dataset
    
    if hasattr(test_dataset, 'images'):
        test_images = test_dataset.images
        test_labels = test_dataset.labels
    else:
        test_images = [test_dataset.dataset.samples[i][0] for i in range(len(test_dataset))]
        test_labels = [test_dataset.dataset.samples[i][1] for i in range(len(test_dataset))]
    
    sample_indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    gradcam_results = []
    
    for idx in sample_indices:
        img_path = test_images[idx]
        true_label = test_labels[idx]
        true_class = classes[true_label]
        
        # Generate Grad-CAM
        overlay, heatmap = grad_cam.overlay_heatmap(img_path, target_class=None)
        
        # Get prediction
        from PIL import Image
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_np = np.array(Image.open(img_path).convert("RGB"))
        input_tensor = transform(Image.fromarray(img_np)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            pred_class = classes[np.argmax(probs)]
            confidence = np.max(probs)
        
        # Save visualization
        filename = Path(img_path).stem
        overlay_path = f"{output_dir}/gradcam_{filename}_{true_class}_pred_{pred_class}.png"
        overlay.save(overlay_path)
        
        result = {
            'image': filename,
            'true_class': true_class,
            'predicted_class': pred_class,
            'confidence': float(confidence),
            'probabilities': {classes[i]: float(probs[i]) for i in range(len(classes))},
            'correct': true_class == pred_class,
            'gradcam_path': overlay_path
        }
        gradcam_results.append(result)
        
        status = "✅" if result['correct'] else "❌"
        print(f"{status} {filename:20s} | True: {true_class:25s} | Pred: {pred_class:25s} | Conf: {confidence:.2%}")
        print(f"   💾 Saved: {overlay_path}")
    
    # === PART 3: ROC Curves (One-vs-Rest) ===
    print("\n" + "="*60)
    print("📈 PART 3: ROC Analysis")
    print("="*60)
    
    plt.figure(figsize=(10, 8))
    
    for i, cls in enumerate(classes):
        # Binary: class i vs rest
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - One-vs-Rest')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = f"{output_dir}/roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"💾 ROC curves saved: {roc_path}")
    
    # === PART 4: Save Summary Report ===
    report = {
        'model_path': model_path,
        'dataset': data_dir,
        'num_test_samples': len(y_true),
        'classes': classes,
        'overall_metrics': metrics,
        'gradcam_samples': gradcam_results,
        'output_files': {
            'confusion_matrix': cm_path,
            'roc_curves': roc_path,
            'gradcam_samples': [r['gradcam_path'] for r in gradcam_results]
        }
    }
    
    report_path = f"{output_dir}/evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"💾 Full report saved: {report_path}")
    
    print("\n" + "🎉 Evaluation Complete!".center(60, "="))
    print(f"📁 All results saved to: {output_dir}/")
    print("="*60)
    
    return report

if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    MODEL_PATH = "models/best_model.pth"
    DATA_DIR = "/Users/jayden/Desktop/CT-Scan_Disease/Data/train"
    TEST_DIR = "/Users/jayden/Desktop/CT-Scan_Disease/Data/test"
    OUTPUT_DIR = "results"
    NUM_SAMPLES = 8  # Number of Grad-CAM visualizations to generate
    
    evaluate_with_gradcam(MODEL_PATH, DATA_DIR, test_dir=TEST_DIR, num_samples=NUM_SAMPLES, output_dir=OUTPUT_DIR)