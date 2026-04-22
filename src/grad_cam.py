import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from src.model import ChestCTClassifier

class GradCAM:
    """
    Grad-CAM implementation for ResNet50-based CT classifier.
    Shows which regions of the image influenced the prediction.
    """
    
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        # For ResNet50, the last conv layer is in layer4[-1].conv3
        self.target_layer = target_layer or self.model.backbone.layer4[-1].conv3
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for a given input.
        
        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W]
            target_class: Class index to visualize (None = predicted class)
        
        Returns:
            heatmap: np.array [H, W] with values 0-1
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass on target class score
        target_score = output[0, target_class]
        target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # [1, 512, 7, 7] for ResNet50
        activations = self.activations  # [1, 512, 7, 7]
        
        # Global average pool gradients across spatial dimensions
        pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=False)  # [1, 512]
        
        # Weight activations by pooled gradients
        pooled_gradients = pooled_gradients.view(1, -1, 1, 1)
        cam = torch.mean(activations * pooled_gradients, dim=1, keepdim=True)  # [1, 1, 7, 7]
        
        # ReLU to keep only features that positively influence the class
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        # Upsample to original image size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()
    
    def overlay_heatmap(self, image_path, target_class=None, alpha=0.4):
        """
        Generate and overlay Grad-CAM heatmap on original image.
        
        Args:
            image_path: Path to original CT slice image
            target_class: Class index to visualize
            alpha: Transparency of overlay (0=transparent, 1=opaque)
        
        Returns:
            overlay_img: PIL Image with heatmap overlay
            heatmap: Raw heatmap array
        """
        # Load and preprocess image
        import numpy as np
        from PIL import Image
        
        # Same transforms as training (without augmentation)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load image
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)
        
        # Apply transforms
        input_tensor = transform(img_pil).unsqueeze(0)
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Generate CAM
        heatmap = self.generate_cam(input_tensor, target_class)
        
        # Convert heatmap to color map (JET)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Resize original image to 224x224 for overlay
        img_resized = cv2.resize(img_np, (224, 224))
        
        # Create overlay
        overlay = cv2.addWeighted(
            img_resized.astype(np.float32), 
            1 - alpha, 
            heatmap_color.astype(np.float32), 
            alpha, 
            0
        )
        overlay = np.uint8(np.clip(overlay, 0, 255))
        
        return Image.fromarray(overlay), heatmap