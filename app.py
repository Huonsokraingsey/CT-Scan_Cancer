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
        return None, "Please upload a Chest CT scan image.", None

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
        f"**Prediction:** {pred_class.replace('.', ' ').title()}\n\n"
        f"**Confidence:** {confidence:.2%}"
    )
    
    return prob_dict, status_text, overlay_pil

# ================= GRADIO UI =================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container {
    background: #0F172A !important;
    font-family: 'Inter', sans-serif !important;
}

.main-title {
    text-align: center;
    font-size: 2rem !important;
    font-weight: 600;
    color: #F8FAFC !important;
    margin-bottom: 0.25rem !important;
}

.subtitle {
    text-align: center;
    font-size: 0.95rem;
    color: #94A3B8 !important;
    margin-bottom: 2.5rem !important;
}

.gr-group, .gr-box {
    background: #1E293B !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    border: none !important;
}

.gr-image {
    border-radius: 8px !important;
}

.upload-zone {
    border: none !important;
}

.analyze-btn {
    background: #3B82F6 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.analyze-btn:hover {
    background: #2563EB !important;
    transform: translateY(-1px) !important;
}

.clear-btn {
    background: #475569 !important;
    color: #CBD5E1 !important;
    border: none !important;
    border-radius: 8px !important;
}

.warning-box {
    text-align: center;
    font-size: 0.75rem;
    color: #64748B !important;
    margin-top: 1.5rem;
    padding: 0 1rem;
}

.section-header {
    font-size: 0.9rem !important;
    font-weight: 500;
    color: #94A3B8 !important;
    margin-bottom: 0.75rem !important;
    padding-left: 0.25rem;
}

.output-section {
    min-height: 320px;
}

.row-spacing {
    gap: 2.5rem !important;
}
"""

theme = gr.themes.Soft(primary_hue="slate", secondary_hue="slate")

with gr.Blocks(title="Chest CT Cancer Classifier") as app:
    gr.Markdown('<p class="main-title">AI Chest CT Cancer Classifier</p>')
    gr.Markdown('<p class="subtitle">Upload a chest CT scan to detect and classify lung cancer types</p>')
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown('<p class="section-header">Upload CT Scan</p>')
            input_img = gr.Image(type="pil", label="", height=280)
            with gr.Row():
                submit_btn = gr.Button("Analyze Scan", size="lg", scale=2, variant="primary")
                clear_btn = gr.Button("Clear", size="md")
            
        with gr.Column(scale=1):
            gr.Markdown('<p class="section-header">Analysis Results</p>')
            output_img = gr.Image(label="Grad-CAM Heatmap", height=180)
            output_probs = gr.Label(label="Probability Distribution", show_label=True)
            output_text = gr.Markdown()
    
    gr.Markdown('<p class="warning-box">For educational/research purposes only. Not for clinical diagnosis. Consult a licensed radiologist.</p>')
    
    submit_btn.click(analyze_ct, inputs=[input_img], outputs=[output_probs, output_text, output_img])
    clear_btn.click(lambda: (None, "", None), outputs=[input_img, output_text, output_img])

if __name__ == "__main__":
    print("Launching local interface at: http://localhost:7860")
    app.launch(theme=theme, css=custom_css)