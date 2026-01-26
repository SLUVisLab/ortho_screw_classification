import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import argparse
import json

class DualViewMaskRCNNDataset(Dataset):
    """Dataset that provides crop pairs with mask annotations for Mask R-CNN training"""
    
    def __init__(self, df, root_dir, class_to_idx, transform=None, samples_per_patient=1, img_size=400):
        # Group by patient
        self.patient_groups = df.groupby('patient_number')
        self.patient_ids = list(self.patient_groups.groups.keys())
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples_per_patient = samples_per_patient
        self.img_size = img_size
        
    def __len__(self):
        return len(self.patient_ids) * self.samples_per_patient
    
    def __getitem__(self, idx):
        # Map virtual index to actual patient
        patient_idx = idx // self.samples_per_patient
        patient_id = self.patient_ids[patient_idx]
        patient_df = self.patient_groups.get_group(patient_id)
        
        # Get manufacturer label
        manufacturer = patient_df['manufacturer'].iloc[0]
        label = self.class_to_idx[manufacturer]
        
        # Get AP and Lateral rows
        ap_rows = patient_df[patient_df['view_position'] == 'AP']
        lat_rows = patient_df[patient_df['view_position'] == 'LATERAL']
        
        if len(ap_rows) == 0 or len(lat_rows) == 0:
            return self._get_empty_sample()
        
        # Randomly sample one row from each
        ap_row = ap_rows.sample(1).iloc[0]
        lat_row = lat_rows.sample(1).iloc[0]
        
        # Get AP crop and mask
        ap_crop, ap_box, ap_mask = self._get_screw_crop_with_mask(ap_row)
        lat_crop, lat_box, lat_mask = self._get_screw_crop_with_mask(lat_row)
        
        if ap_crop is None or lat_crop is None:
            return self._get_empty_sample()
        
        # Apply transforms
        if self.transform:
            ap_crop = self.transform(ap_crop)
            lat_crop = self.transform(lat_crop)
        else:
            ap_crop = transforms.ToTensor()(ap_crop)
            lat_crop = transforms.ToTensor()(lat_crop)
        
        # Create targets for Mask R-CNN format
        ap_target = {
            "boxes": torch.as_tensor([ap_box], dtype=torch.float32),
            "labels": torch.as_tensor([label], dtype=torch.int64),
            "masks": torch.from_numpy(np.array([ap_mask], dtype=np.uint8))
        }
        
        lat_target = {
            "boxes": torch.as_tensor([lat_box], dtype=torch.float32),
            "labels": torch.as_tensor([label], dtype=torch.int64),
            "masks": torch.from_numpy(np.array([lat_mask], dtype=np.uint8))
        }
        
        return ap_crop, lat_crop, ap_target, lat_target, label
    
    def _get_screw_crop_with_mask(self, row):
        """Extract crop with corresponding mask and bounding box"""
        img_path = os.path.join(self.root_dir, row['relative_file_path'])
        mask_path = self._get_mask_path(row['relative_file_path'])
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            return None, None, None
        
        # Load image (grayscale)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            return None, None, None
        
        # Find screw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if len(valid_contours) == 0:
            return None, None, None
        
        # Randomly pick one screw
        contour = valid_contours[np.random.randint(len(valid_contours))]
        
        # Get bounding box with padding (super-crop)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add jitter and padding for super-crop
        jitter_x = int(np.random.uniform(-w * 0.1, w * 0.1))
        jitter_y = int(np.random.uniform(-h * 0.1, h * 0.1))
        
        cx = x + w // 2 + jitter_x
        cy = y + h // 2 + jitter_y
        
        # Super-crop size (2x)
        crop_size = int(max(w, h) * 2.0)
        half_size = crop_size // 2
        
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(image.shape[1], cx + half_size)
        y2 = min(image.shape[0], cy + half_size)
        
        # Handle padding if needed
        pad_l = max(0, half_size - cx)
        pad_t = max(0, half_size - cy)
        pad_r = max(0, (cx + half_size) - image.shape[1])
        pad_b = max(0, (cy + half_size) - image.shape[0])
        
        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            image = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
            mask = cv2.copyMakeBorder(mask, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            x1 += pad_l
            y1 += pad_t
            x2 += pad_l
            y2 += pad_t
        
        # Extract crop
        crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]
        
        # Find the screw in the cropped mask
        contours_crop, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours_crop) == 0:
            return None, None, None
        
        # Get the largest contour (should be our screw)
        main_contour = max(contours_crop, key=cv2.contourArea)
        
        # Create instance mask
        instance_mask = np.zeros(mask_crop.shape, dtype=np.uint8)
        cv2.drawContours(instance_mask, [main_contour], -1, 255, -1)
        
        # Resize image and mask to target size
        crop_resized = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(instance_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask is binary (0 or 1)
        mask_binary = (mask_resized > 127).astype(np.uint8)
        
        # Recalculate bounding box on resized mask
        contours_resized, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_resized) == 0:
            # Fallback if resizing destroyed the mask (unlikely but possible)
            bbox_resized = [0, 0, self.img_size, self.img_size]
        else:
            br_cnt = max(contours_resized, key=cv2.contourArea)
            brx, bry, brw, brh = cv2.boundingRect(br_cnt)
            bbox_resized = [brx, bry, brx + brw, bry + brh]
        
        # Convert to PIL RGB (3 channels for model)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        return crop_pil, bbox_resized, mask_binary
    
    def _get_mask_path(self, img_rel_path):
        parts = img_rel_path.split('/')
        if 'images' in parts:
            idx = parts.index('images')
            parts[idx] = 'masks'
            filename = parts[-1]
            name, ext = os.path.splitext(filename)
            parts[-1] = f"{name}_mask{ext}"
            mask_rel_path = "/".join(parts)
            return os.path.join(self.root_dir, mask_rel_path)
        return ""
    
    def _get_empty_sample(self):
        """Return empty sample for failed loads"""
        empty_img = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
        empty_target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, self.img_size, self.img_size), dtype=torch.uint8)
        }
        return empty_img, empty_img, empty_target, empty_target, 0


class DualViewMaskRCNN(nn.Module):
    """Dual-view model using Mask R-CNN for feature extraction with segmentation"""
    
    def __init__(self, num_classes):
        super(DualViewMaskRCNN, self).__init__()
        
        # Two Mask R-CNN models (shared weights possible, but separate for flexibility)
        self.maskrcnn_ap = maskrcnn_resnet50_fpn(pretrained=True)
        self.maskrcnn_lat = maskrcnn_resnet50_fpn(pretrained=True)
        
        # Get feature dimension from RoI head
        self.feature_dim = self.maskrcnn_ap.roi_heads.box_predictor.cls_score.in_features
        
        # Fusion classifier (concatenates features from both views)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, ap_images, lat_images, ap_targets=None, lat_targets=None):
        """
        Forward pass with two modes:
        - Training: Returns detection losses + fusion classification loss
        - Inference: Returns fused predictions
        """
        if self.training:
            # Training mode: get detection losses
            ap_losses = self.maskrcnn_ap(ap_images, ap_targets)
            lat_losses = self.maskrcnn_lat(lat_images, lat_targets)
            
            # Extract features for fusion (keep gradients flowing)
            ap_features = self._extract_roi_features(self.maskrcnn_ap, ap_images, ap_targets)
            lat_features = self._extract_roi_features(self.maskrcnn_lat, lat_images, lat_targets)
            
            # Fused classification
            fused = torch.cat([ap_features.contiguous(), lat_features.contiguous()], dim=1)
            fused = fused.contiguous()
            logits = self.fusion_classifier(fused)
            
            # Combine losses
            total_loss = sum(ap_losses.values()) + sum(lat_losses.values())
            
            return {
                'total_detection_loss': total_loss,
                'logits': logits,
                'ap_losses': ap_losses,
                'lat_losses': lat_losses
            }
        else:
            # Inference mode
            ap_features = self._extract_roi_features(self.maskrcnn_ap, ap_images, None)
            lat_features = self._extract_roi_features(self.maskrcnn_lat, lat_images, None)
            
            fused = torch.cat([ap_features, lat_features], dim=1)
            fused = fused.contiguous()
            logits = self.fusion_classifier(fused)
            
            return logits
    
    def _extract_roi_features(self, maskrcnn_model, images, targets):
        """Extract RoI-pooled features from Mask R-CNN backbone"""
        # Stack images into a batch tensor
        images_batch = torch.stack(images)
        device = images_batch.device
        
        # Get backbone features
        features = maskrcnn_model.backbone(images_batch)
        features = {k: v.contiguous() for k, v in features.items()}
        
        if targets is not None:
            # Training: use ground truth boxes
            proposals = [t['boxes'] for t in targets]
        else:
            # Inference: use RPN proposals
            from torchvision.models.detection.image_list import ImageList
            image_sizes = [img.shape[-2:] for img in images]
            image_list = ImageList(images_batch, image_sizes)
            proposals, _ = maskrcnn_model.rpn(image_list, features)
        
        # Ensure exactly ONE proposal per image (take first or add dummy)
        processed_proposals = []
        valid_mask = [] # To keep track of which are real detections
        
        for prop in proposals:
            if prop.shape[0] > 0:
                processed_proposals.append(prop[:1].contiguous())
                valid_mask.append(1.0)
            else:
                # Dummy box for images with no proposals
                processed_proposals.append(torch.tensor([[0, 0, 1, 1]], dtype=torch.float32, device=device))
                valid_mask.append(0.0)
        
        # RoI pooling
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        box_features = maskrcnn_model.roi_heads.box_roi_pool(features, processed_proposals, image_sizes)
        box_features = box_features.contiguous()
        box_features = maskrcnn_model.roi_heads.box_head(box_features)
        
        # Zero out features for dummy boxes
        valid_mask = torch.tensor(valid_mask, dtype=torch.float32, device=device).unsqueeze(1)
        box_features = box_features * valid_mask
        
        return box_features


def collate_fn(batch):
    """Custom collate for Mask R-CNN format"""
    ap_imgs, lat_imgs, ap_targets, lat_targets, labels = zip(*batch)
    return list(ap_imgs), list(lat_imgs), list(ap_targets), list(lat_targets), torch.tensor(labels)


def main():
    parser = argparse.ArgumentParser(description='Train MaskRCNN Classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    args = parser.parse_args()

    # Configuration
    CSV_PATH = os.path.join(args.data_dir, 'metadata_summary.csv')
    ROOT_DIR = args.data_dir
    BATCH_SIZE = 16  # Increased for A100 80GB
    EPOCHS = 50
    LR = 1e-4
    
    # Device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load and filter data (same as train.py)
    print("Loading metadata...")
    df = pd.read_csv(CSV_PATH)
    df = df[~df['manufacturer'].isin(['No Accession Number Match', 'No manufacturer data'])]
    
    EXPLICIT_DROP = ['Styker everest (K2M)', 'orthofix firebird']
    if EXPLICIT_DROP:
        print(f"Explicitly dropping: {EXPLICIT_DROP}")
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
    num_classes = len(manufacturers)
    
    print(f"Classes: {manufacturers}")
    
    # Patient split
    patients = df['patient_number'].unique()
    patient_labels = df.groupby('patient_number')['manufacturer'].first()
    train_patients, test_patients = train_test_split(
        patients, test_size=0.2, random_state=42, stratify=patient_labels
    )
    
    train_df = df[df['patient_number'].isin(train_patients)]
    test_df = df[df['patient_number'].isin(test_patients)]
    
    print(f"Train patients: {len(train_patients)}, Test patients: {len(test_patients)}")
    
    # Transforms (Resize is now handled in dataset)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Datasets (using 400x400)
    IMG_SIZE = 400
    train_dataset = DualViewMaskRCNNDataset(train_df, ROOT_DIR, class_to_idx, train_transform, img_size=IMG_SIZE)
    test_dataset = DualViewMaskRCNNDataset(test_df, ROOT_DIR, class_to_idx, train_transform, samples_per_patient=10, img_size=IMG_SIZE)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, collate_fn=collate_fn)  
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, collate_fn=collate_fn)
    
    # Model
    model = DualViewMaskRCNN(num_classes)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print("Starting training...")
    
    # Metrics tracking
    train_losses = []
    train_accs = []
    train_accs = []
    test_accs = []
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (ap_imgs, lat_imgs, ap_targets, lat_targets, labels) in enumerate(train_loader):
            ap_imgs = [img.to(device) for img in ap_imgs]
            lat_imgs = [img.to(device) for img in lat_imgs]
            ap_targets = [{k: v.to(device) for k, v in t.items()} for t in ap_targets]
            lat_targets = [{k: v.to(device) for k, v in t.items()} for t in lat_targets]
            labels = labels.to(device)
            
            # Forward
            outputs = model(ap_imgs, lat_imgs, ap_targets, lat_targets)
            
            # Combined loss: detection + classification
            detection_loss = outputs['total_detection_loss']
            classification_loss = criterion(outputs['logits'], labels)
            total_loss = detection_loss + classification_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Stats
            epoch_loss += total_loss.item()
            _, predicted = torch.max(outputs['logits'], 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: " 
                      f"Det Loss {detection_loss.item():.4f}, "
                      f"Cls Loss {classification_loss.item():.4f}")
        
        # Stats
        avg_loss = epoch_loss/len(train_loader)
        train_acc = 100 * correct / total
        
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validation (Every epoch for better plotting)
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for ap_imgs, lat_imgs, _, _, labels in test_loader:
                ap_imgs = [img.to(device) for img in ap_imgs]
                lat_imgs = [img.to(device) for img in lat_imgs]
                labels = labels.to(device)
                
                logits = model(ap_imgs, lat_imgs)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        test_acc = 100 * correct / total
        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        # Save Best Model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'maskrcnn_dual_view_best.pth')
            print(f"New best accuracy: {test_acc:.2f}%. Model saved to maskrcnn_dual_view_best.pth")
    
    # Save
    torch.save(model.state_dict(), 'maskrcnn_dual_view.pth')
    print("Model saved!")
    
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
    plt.savefig('maskrcnn_results.png')
    print("Plot saved to maskrcnn_results.png")
    
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
