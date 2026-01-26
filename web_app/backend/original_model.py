import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class DualViewClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DualViewClassifier, self).__init__()
        
        # Use simple ResNet18 backbones
        # "DEFAULT" means best available weights (ImageNet)
        self.backbone_ap = torchvision.models.resnet18(weights="DEFAULT")
        self.backbone_lat = torchvision.models.resnet18(weights="DEFAULT")
        
        # Feature dimension for ResNet18 is 512 (after the final pooling)
        num_features = self.backbone_ap.fc.in_features
        
        # Remove original classifier
        self.backbone_ap.fc = nn.Identity()
        self.backbone_lat.fc = nn.Identity()
        
        # Fusion Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x_ap, x_lat):
        # x_ap, x_lat are [B, 3, 224, 224] (assuming Grayscale(3) transform used)
        
        # AP View processing
        feat_ap = self.backbone_ap(x_ap) # [B, 512] (since fc is Identity)
        
        # Lateral View processing
        feat_lat = self.backbone_lat(x_lat) # [B, 512]
        
        # Concatenate features
        combined = torch.cat((feat_ap, feat_lat), dim=1)
        
        # Classify
        out = self.classifier(combined)
        return out
