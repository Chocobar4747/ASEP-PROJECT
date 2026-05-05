"""
CNN Model — ResNet18 with Regression Head
===============================================
Using ResNet18 for more stable colorimetric regression.
"""

import torch
import torch.nn as nn
from torchvision import models


class DyeConcentrationCNN(nn.Module):
    """
    ResNet18-based regression model.
    ResNet is often more robust for simple regression on small datasets.
    """
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Get input features for the final layer
        num_features = self.backbone.fc.in_features  # 512
        
        # Remove original fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Regression head — deeper with BatchNorm for better gradient flow
        # NO Sigmoid: model outputs raw ppm values directly
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def unfreeze_last_n_layers(self, n: int = 2):
        """Unfreeze last few layers of ResNet (layer4)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # ResNet18 has layer1, layer2, layer3, layer4. Unfreeze layer4.
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        if n > 1:
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        features = self.backbone(x)
        ppm = self.regressor(features)
        return ppm.squeeze(-1)
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def get_model(pretrained=True, freeze_backbone=True):
    model = DyeConcentrationCNN(pretrained=pretrained, freeze_backbone=freeze_backbone)
    params = model.count_parameters()
    print(f"ResNet18 Model created — Total: {params['total']:,} params, Trainable: {params['trainable']:,}")
    return model
