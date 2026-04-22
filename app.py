import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

from src.model import ChestCTClassifier
from src.grad_cam import GradCAM

# ================= CONFIGURATION =================
MODEL_PATH = "models/best_model.pth"
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLASSES = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
IMG_SIZE = 224
# =================================================

print(f"🚀 Loading model on {DEVICE}...")

# Load model
model = ChestCTClassifier(num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE).eval()

# Initialize Grad-CAM
cam = GradCAM(model)

# Preprocessing pipeline (matches training)
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def analyze_ct(pil_image):
    """Core inference + Grad-CAM pipeline"""
    if pil_image is None:
        return None, "⚠️ Please upload a Chest CT scan image.", None

    # Convert to numpy RGB
    img_np = np.array(pil_image.convert("RGB"))
    
    # Preprocess
    aug = transform(image=img_np)
    input_tensor = aug['image'].unsqueeze(0).to(DEVICE)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    
    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx]
    
    # Generate Grad-CAM
    cam_map = cam.generate_cam(input_tensor)  # [H, W] normalized 0-1
    
    # Colorize & overlay
    cam_color = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    
    # Resize to match original for clean overlay
    orig_h, orig_w = img_np.shape[:2]
    cam_resized = cv2.resize(cam_color, (orig_w, orig_h))
    overlay = cv2.addWeighted(img_np.astype(float), 0.65, cam_resized.astype(float), 0.35, 0)
    overlay_pil = Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))
    
    # Format outputs
    prob_dict = {cls.replace('.', ' ').title(): float(p) for cls, p in zip(CLASSES, probs)}
    status_text = (
        f"🔍 **Prediction:** {pred_class.replace('.', ' ').title()}\n"
        f"📊 **Confidence:** {confidence:.2%}\n"
        f"⚠️ *Research tool only. Not for clinical diagnosis. Always consult a licensed radiologist.*"
    )
    
    return prob_dict, status_text, overlay_pil

# ================= GRADIO UI =================
with gr.Blocks(title="Chest CT Cancer Classifier", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🩺 AI Chest CT Cancer Classifier")
    gr.Markdown("Upload a single axial chest CT scan slice to detect and classify potential lung cancer types. **For educational/research purposes only.**")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="📤 Upload CT Slice", height=300)
            submit_btn = gr.Button("🔍 Analyze Scan", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", size="sm")
            
        with gr.Column(scale=1):
            output_img = gr.Image(label="🌡️ Grad-CAM Heatmap Overlay", height=300)
            output_probs = gr.Label(label="📈 Class Probabilities")
            output_text = gr.Markdown()
            
    submit_btn.click(analyze_ct, inputs=[input_img], outputs=[output_probs, output_text, output_img])
    clear_btn.click(lambda: (None, "⬆️ Upload a CT scan to begin.", None), outputs=[input_img, output_text, output_img])
    
    gr.Markdown("---\n*Built with PyTorch & ResNet50 | Optimized for Apple Silicon (MPS)*")

if __name__ == "__main__":
    print("💻 Launching local interface at: http://localhost:7860")
    app.launch()