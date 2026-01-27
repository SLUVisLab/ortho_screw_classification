import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import argparse
import json
import sys
import os
# Add parent directory to path to find utils regardless of CWD
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import ScrewDataset, collate_fn
from classifier_model import DualViewClassifier
import matplotlib.pyplot as plt



# --- Advanced Training Helpers ---
def mixup_data(x1, x2, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = torch.nn.functional.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum(dim=-1)
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_preds, target, reduction=self.reduction)


def main():
    parser = argparse.ArgumentParser(description='Train Original Classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    args = parser.parse_args()

    # --- Configuration (Best Params from Optuna) ---
    CSV_PATH = os.path.join(args.data_dir, 'metadata_summary.csv')
    ROOT_DIR = args.data_dir
    BATCH_SIZE = 128 
    EPOCHS = 100
    
    # Best Hyperparameters
    LR = 0.000387403
    WEIGHT_DECAY = 6.7452e-05
    STEP_SIZE = 8
    GAMMA = 0.1
    LABEL_SMOOTHING = 0.042239
    MIXUP_ALPHA = 0.239336
    
    # ... (device check) ...
    
    # ... (data loading - skipped in this replace block) ...
    
    # ... (model setup) ...
    model = DualViewClassifier(num_classes=num_classes).to(device)
    
    # Loss with Label Smoothing
    if LABEL_SMOOTHING > 0.0:
        print(f"Using Label Smoothing: {LABEL_SMOOTHING}")
        criterion = LabelSmoothingCrossEntropy(eps=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # --- Training Loop ---
    train_losses = []
    train_accs = []
    test_losses = []
    test_losses = []
    test_accs = []
    
    best_acc = 0.0

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0 # Track total samples for accuracy
        
        for batch_idx, batch in enumerate(train_loader):
            # collate_fn might return empty batch if all samples were None
            if not batch: 
                continue
                
            ap_imgs, lat_imgs, labels = batch
            ap_imgs, lat_imgs, labels = ap_imgs.to(device), lat_imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixup Logic
            if MIXUP_ALPHA > 0.0:
                # We need to mix both AP and Lat images coherently
                # Note: mixup_data now requires 'device'
                ap_mixed, _, y_a, y_b, lam = mixup_data(ap_imgs, ap_imgs, labels, device, MIXUP_ALPHA)
                
                # To ensure LAT images are mixed with the SAME indices as AP, we can't call mixup_data again randomly.
                # However, mixup_data uses randperm. We should ideally control the permutation.
                # But for now, let's reuse the logic:
                batch_size = ap_imgs.size(0)
                # Recover index from y_b if possible? No.
                # We simply replicate the logic inline or update mixup_data to return index.
                # Easier: Modify mixup implementations to be simpler or invoke manually.
                # Actually, in `train_optuna.py` we solved this by re-generating the permutation? 
                # Wait, in the snippet I wrote for `train_optuna.py`, I did:
                # index = torch.randperm(batch_size).to(device)
                # ...
                # Let's just do it manually here to be safe and clear.
                
                index = torch.randperm(batch_size).to(device)
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                
                mixed_ap = lam * ap_imgs + (1 - lam) * ap_imgs[index, :]
                mixed_lat = lam * lat_imgs + (1 - lam) * lat_imgs[index, :]
                
                y_a, y_b = labels, labels[index]
                
                outputs = model(mixed_ap, mixed_lat)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(ap_imgs, lat_imgs)
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss {loss.item():.4f}")

        if total > 0:
            train_acc = 100 * correct / total
            epoch_loss = running_loss/len(train_loader)
            train_losses.append(epoch_loss)
            train_accs.append(train_acc)
            print(f"Epoch {epoch+1} Results: Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        scheduler.step() # Step the scheduler
        
        # --- Balanced Evaluation ---
        test_acc, test_loss = evaluate_balanced(model, test_dataset, device, epoch, criterion)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        
        # Save Best Model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'original_classifier_best.pth')
            print(f"New best accuracy: {best_acc:.2f}%. Model saved to original_classifier_best.pth")

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("Training plot saved to training_results.png")
    
    # Save Metrics for Benchmark
    metrics = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'test_loss': test_losses,
        'test_acc': test_accs
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("Metrics saved to metrics.json")

    # Save Model
    torch.save(model.state_dict(), 'original_classifier.pth')
    print("Model saved to original_classifier.pth")

def evaluate_balanced(model, test_dataset, device, epoch, criterion):
    """
    Evaluates the model on a balanced subset of the test set.
    Returns (accuracy, avg_loss)
    """
    model.eval()
    
    # Group test indices by class
    # Group test indices by class
    class_indices = {}
    
    # We iterate range(len(test_dataset)) because indices form 0..N are now virtual expanded indices
    for idx in range(len(test_dataset)):
        # Determine class for this sample
        # Since we modified dataset logic: patient_idx = idx // samples_per_patient
        patient_idx = idx // test_dataset.samples_per_patient
        pid = test_dataset.patient_ids[patient_idx]
        group = test_dataset.patient_groups.get_group(pid)
        manufacturer = group['manufacturer'].iloc[0]
        label = test_dataset.class_to_idx[manufacturer]
        
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Find minimum count
    if not class_indices:
        print("Test set is empty.")
        return 0.0, 0.0

    min_count = min(len(idxs) for idxs in class_indices.values())
    print(f"Evaluating on balanced subset: {min_count} samples per class.")
    
    if min_count == 0:
        print("Warning: One or more classes have 0 samples in test set. Skipping balanced eval.")
        return 0.0, 0.0

    # Select indices
    balanced_indices = []
    for label in class_indices:
        # Deterministic sampling for reproducibility within an epoch, but maybe we want random?
        # Let's use random sample
        balanced_indices.extend(np.random.choice(class_indices[label], min_count, replace=False))
    
    subset = Subset(test_dataset, balanced_indices)
    loader = DataLoader(subset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            if not batch: continue
            ap_imgs, lat_imgs, labels = batch
            ap_imgs, lat_imgs, labels = ap_imgs.to(device), lat_imgs.to(device), labels.to(device)
            
            outputs = model(ap_imgs, lat_imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0.0
    
    if total > 0:
        acc = 100 * correct / total
        print(f"Epoch {epoch+1} Balanced Test Acc: {acc:.2f}%, Loss: {avg_loss:.4f}")
        return acc, avg_loss
    else:
        print("Epoch {epoch+1} Balanced Test Acc: N/A (No valid pairs found)")
        return 0.0, 0.0

if __name__ == '__main__':
    main()
