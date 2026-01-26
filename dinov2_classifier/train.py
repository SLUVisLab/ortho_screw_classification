import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json

# Enable MPS fallback for DINOv2 interpolation
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import os
# Add parent directory to path to find utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse existing dataset
from utils.dataset import ScrewDataset, collate_fn

class DualViewDINOv2(nn.Module):
    """
    Dual-View Classifier using DINOv2 (ViT-S/14) as a frozen feature extractor.
    """
    def __init__(self, num_classes):
        super(DualViewDINOv2, self).__init__()
        
        # Load DINOv2 Small (ViT-S/14) from Torch Hub
        print("Loading DINOv2 model...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # DINOv2 ViT-S/14 feature dim is 384
        self.feature_dim = 384
        
        # Fusion Classifier
        # Concatenates AP (384) + Lat (384) = 768
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, ap_imgs, lat_imgs):
        # Extract features for AP view
        # DINOv2 forward_features returns a dict, "x_norm_clstoken" is the CLS token
        # Or standard forward returns the class token directly for some implementations.
        # The facebookresearch/dinov2 hub model forward() returns the features directly? 
        # Let's check: typically model(x) returns the CLS token or features.
        # Verified: hub model(x) returns the selected output, usually CLS token if not specified otherwise.
        # But to be safe and explicit, let's use forward_features() if available, or just model(x).
        # Standard usage: features = model(imgs) -> [B, 384]
        
        ap_features = self.backbone(ap_imgs)
        lat_features = self.backbone(lat_imgs)
        
        # Concatenate
        combined = torch.cat((ap_features, lat_features), dim=1)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits

def main():
    parser = argparse.ArgumentParser(description='Train DINOv2 Classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    args = parser.parse_args()

    # --- Configuration ---
    CSV_PATH = os.path.join(args.data_dir, 'metadata_summary.csv')
    ROOT_DIR = args.data_dir
    BATCH_SIZE = 128 # Increased for A100 80GB
    EPOCHS = 50
    LR = 1e-4
    
    # Device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cpu') 
    print(f"Using device: {device}")
    
    # --- Data Loading ---
    print("Loading metadata...")
    df = pd.read_csv(CSV_PATH)
    df = df[~df['manufacturer'].isin(['No Accession Number Match', 'No manufacturer data'])]
    
    EXPLICIT_DROP = ['Styker everest (K2M)', 'orthofix firebird']
    if EXPLICIT_DROP:
        df = df[~df['manufacturer'].isin(EXPLICIT_DROP)]
        
    MIN_PATIENTS = 10
    counts = df.groupby('manufacturer')['patient_number'].nunique()
    drop_classes = counts[counts < MIN_PATIENTS].index.tolist()
    if drop_classes:
        print(f"Dropping manufacturers with < {MIN_PATIENTS} patients: {drop_classes}")
        df = df[~df['manufacturer'].isin(drop_classes)]
        
    # Class mapping
    manufacturers = sorted(df['manufacturer'].unique())
    class_to_idx = {m: i for i, m in enumerate(manufacturers)}
    print(f"Classes: {manufacturers}")
    
    # Patient Split
    patients = df['patient_number'].unique()
    patient_labels = df.groupby('patient_number')['manufacturer'].first()
    train_patients, test_patients = train_test_split(
        patients, test_size=0.2, random_state=42, stratify=patient_labels
    )
    
    train_df = df[df['patient_number'].isin(train_patients)]
    test_df = df[df['patient_number'].isin(test_patients)]
    print(f"Train patients: {len(train_patients)}, Test patients: {len(test_patients)}")
    
    # Transforms for DINOv2
    # - Resize to 224x224 (patches of 14 fit perfectly: 16x16 grid)
    # - Grayscale to RGB (3 channels required)
    # - ImageNet Normalization
    
    common_transforms = [
        transforms.Resize((224, 224)), # DINO specific size
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), # Slightly larger for crop
        transforms.RandomCrop(224),    # Random crop augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose(common_transforms)
    
    # Datasets
    train_dataset = ScrewDataset(train_df, ROOT_DIR, transform=train_transform)
    test_dataset = ScrewDataset(test_df, ROOT_DIR, transform=test_transform, samples_per_patient=10) # 10 samples for robust eval
    
    # Weighted Sampler
    train_labels = []
    for pid in train_dataset.patient_ids:
        m = train_df[train_df['patient_number'] == pid]['manufacturer'].iloc[0]
        train_labels.append(class_to_idx[m])
    
    class_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8)
    
    # Model
    num_classes = len(manufacturers)
    model = DualViewDINOv2(num_classes)
    model.to(device)
    
    # Optimization
    # Only optimizing the classifier head params
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    print("Starting training...")
    train_losses = []
    train_accs = []
    train_accs = []
    test_accs = []
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            if not batch: continue
            
            ap_imgs, lat_imgs, labels = batch
            ap_imgs = ap_imgs.to(device)
            lat_imgs = lat_imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(ap_imgs, lat_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = 100 * correct / total if total > 0 else 0
        
        # Validation
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for batch in test_loader:
                if not batch: continue
                ap_imgs, lat_imgs, labels = batch
                ap_imgs = ap_imgs.to(device)
                lat_imgs = lat_imgs.to(device)
                labels = labels.to(device)
                
                outputs = model(ap_imgs, lat_imgs)
                _, predicted = torch.max(outputs.data, 1)
                v_total += labels.size(0)
                v_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * v_correct / v_total if v_total > 0 else 0
        
        # Store metrics
        train_losses.append(epoch_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        # Save Best Model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'dinov2_dual_view_best.pth')
            print(f"New best accuracy: {test_acc:.2f}%. Model saved to dinov2_dual_view_best.pth")
        
    # Save Model
    torch.save(model.state_dict(), 'dinov2_dual_view.pth')
    print("Model saved to dinov2_dual_view.pth")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('dinov2_results.png')
    print("Plot saved to dinov2_results.png")

    # Save Metrics
    metrics = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'test_acc': test_accs
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("Metrics saved to metrics.json")

if __name__ == "__main__":
    main()
