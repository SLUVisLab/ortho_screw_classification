import torch
import torch.nn as nn
from original_model import DualViewClassifier
import os
from torchvision import transforms
from PIL import Image

class ClassifierWrapper:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.num_classes = 4 # Hardcoded based on known classes
        
        # Initialize model
        self.model = DualViewClassifier(num_classes=self.num_classes)
        
        # Load weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}. Running with random weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Define Transforms (Same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # Canonical is just resizing/normalizing, logic might be slightly different if we want consistent crop? 
            # Actually, training used Resize(400) -> CenterCrop(224). 
            # Current inference used Resize(224). Let's stick to current inference for 'canonical' to not break behavior 
            # but maybe we should align them? The prompt implies "different screw crops".
            # Let's keep self.transform simple for canonical and add self.augment_transform for variations.
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Augmentation Transforms (Mirrors train.py)
        self.augment_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.RandomRotation(15), 
            transforms.CenterCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), 
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess(self, image_bytes_ap, image_bytes_lat, augment=False):
        """Converts raw bytes to tensors"""
        import io
        img_ap = Image.open(io.BytesIO(image_bytes_ap)).convert('RGB')
        img_lat = Image.open(io.BytesIO(image_bytes_lat)).convert('RGB')
        
        transform = self.augment_transform if augment else self.transform
        
        t_ap = transform(img_ap).unsqueeze(0).to(self.device)
        t_lat = transform(img_lat).unsqueeze(0).to(self.device)
        
        return t_ap, t_lat

    def predict(self, image_bytes_ap, image_bytes_lat, augment=False):
        """
        Returns:
            - probabilities: Dict of class -> prob
            - embedding: 1024-dim numpy array
        """
        t_ap, t_lat = self.preprocess(image_bytes_ap, image_bytes_lat, augment=augment)
        
        with torch.no_grad():
            # We need to manually forward pass to intercept the embeddings
            # The original model forward() just does classification.
            # We'll hook into the components.
            
            # 1. Feature Extraction
            feat_ap = self.model.backbone_ap(t_ap) # [1, 512]
            feat_lat = self.model.backbone_lat(t_lat) # [1, 512]
            
            # 2. Fusion
            combined = torch.cat((feat_ap, feat_lat), dim=1) # [1, 1024]
            
            # 3. Classification
            logits = self.model.classifier(combined)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            embedding = combined.cpu().numpy()[0]
            
        return probs, embedding
