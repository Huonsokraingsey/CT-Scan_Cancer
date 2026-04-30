import torch
import torch.nn as nn
import torchvision.models as models

class ChestCTClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ChestCTClassifier, self).__init__()
        
        # 1. Load Pre-trained ResNet50 (trained on ImageNet)
        # weights=DEFAULT downloads the pre-trained weights automatically
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 2. Freeze early layers (they already know how to detect features)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze last 2 blocks for fine-tuning
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
            
        # 3. Replace the final classification layer
        # ResNet50 ends with a 2048-dim vector, we map that to our 4 classes
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.6),  # Increased dropout to prevent overfitting
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes) # Final output layer
        )
        
    def forward(self, x):
        return self.backbone(x)