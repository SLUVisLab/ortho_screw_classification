import os
import argparse
import json
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import sys

# Add parent directory to path to find utils
deployment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(deployment_dir)
# Add original_classifier to path
sys.path.append(os.path.join(deployment_dir, 'original_classifier'))

from utils.dataset import ScrewDataset, collate_fn
from classifier_model import DualViewClassifier

# --- Helpers for Advanced Training ---
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


def evaluate_balanced(model, test_dataset, device, criterion, samples_per_class=5):
    """
    Quick evaluation on a small balanced subset for pruning/reporting.
    """
    model.eval()
    
    # Simple random balanced sampling
    indices = []
    classes = list(test_dataset.class_to_idx.values())
    
    # We need to map class idx back to patient indices that have that class
    # This is expensive to do every time, so we do it once roughly or just sample
    # standardized way:
    # Use the existing logic from train.py but simplified
    
    # For speed in Optuna, let's just use the full test loader but keep it fast?
    # Actually, a fixed subset is better for stability.
    # Let's rely on the RandomSampler with replacement for simplicity in this optimization loop
    # or just run on entire test set (it's not THAT big yet)
    
    # Re-using the test_loader passed from main is safest
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
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
    acc = 100 * correct / total if total > 0 else 0.0
    return acc, avg_loss

def objective(trial, args, train_dataset, test_dataset, num_classes, device):
    # Hyperparameters to Sample
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    step_size = trial.suggest_categorical("step_size", [5, 8, 10])
    gamma = trial.suggest_categorical("gamma", [0.1, 0.2, 0.5])
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.4)
    
    # Model
    model = DualViewClassifier(num_classes=num_classes).to(device)
    
    # Loss
    if label_smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(eps=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Data Loaders (Re-create to ensure freshness or strictness if needed, but we can reuse)
    # We need a sampler for training
    train_labels = [train_dataset.class_to_idx[train_dataset.patient_groups.get_group(pid)['manufacturer'].iloc[0]] for pid in train_dataset.patient_ids]
    class_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, collate_fn=collate_fn, num_workers=4)
    
    # Training Loop (Shortened for Optimization, e.g., 10-15 epochs)
    # Ideally we train enough to see convergence, maybe 15 epochs
    N_EPOCHS = 15 
    
    for epoch in range(N_EPOCHS):
        model.train()
        for batch in train_loader:
            if not batch: continue
            ap, lat, targets = batch
            ap, lat, targets = ap.to(device), lat.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Mixup
            if mixup_alpha > 0.0:
                ap_mixed, _, y_a, y_b, lam = mixup_data(ap, ap, targets, device, mixup_alpha)
                # Note: Mixup usually mixes inputs. Dual view makes it tricky. 
                # Simplification: Mix AP with AP, Lat with Lat using same lambda? Yes.
                lat_mixed, _, _, _, _ = mixup_data(lat, lat, targets, device, 0) # Dummy call, actually we need same indices
                
                # Correct Mixup for Dual View:
                # We need to manually permute to ensure AP and Lat are permuted identically
                batch_size = ap.size(0)
                index = torch.randperm(batch_size).to(device)
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                
                ap_mixed = lam * ap + (1 - lam) * ap[index, :]
                lat_mixed = lam * lat + (1 - lam) * lat[index, :]
                y_a, y_b = targets, targets[index]
                
                outputs = model(ap_mixed, lat_mixed)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(ap, lat)
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validation
        val_acc, val_loss = evaluate_balanced(model, test_dataset, device, nn.CrossEntropyLoss()) # Use standard CE for eval metrics
        
        # Report to Optuna
        trial.report(val_acc, epoch)
        
        # Pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of GPU to use if CUDA is available')
    args = parser.parse_args()
    
    # ... (data setup) ... (unchanged lines will be matched by context, wait, replace_file_content needs exact target)
    
    # I should target the block where device is defined.
    # The tool doesn't support multiple disjoint edits in one go easily unless I match a large block or use multi_replace.
    # I will use multi_replace_file_content if I need to edit two places.
    # Actually, args are parsed at top of main, device is defined later.
    # Let's use multi_replace.
    
    # --- Data Setup ---
    root_dir = args.data_dir
    csv_path = os.path.join(root_dir, 'metadata_summary.csv')
    df = pd.read_csv(csv_path)
    
    # same filtering as train.py
    df = df[~df['manufacturer'].isin(['No Accession Number Match', 'No manufacturer data'])]
    
    # Explicitly drop classes (if needed, copying from train.py)
    EXPLICIT_DROP = ['Styker everest (K2M)', 'orthofix firebird']
    if EXPLICIT_DROP:
        df = df[~df['manufacturer'].isin(EXPLICIT_DROP)]
        
    MIN_PATIENTS = 10
    counts = df.groupby('manufacturer')['patient_number'].nunique()
    drop_classes = counts[counts < MIN_PATIENTS].index.tolist()
    if drop_classes:
        df = df[~df['manufacturer'].isin(drop_classes)]
        
    patients = df['patient_number'].unique()
    train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)
    
    train_df = df[df['patient_number'].isin(train_patients)]
    test_df = df[df['patient_number'].isin(test_patients)]
    
    # Transforms
    train_transform = transforms.Compose([
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
    
    test_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.CenterCrop((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ScrewDataset(train_df, root_dir, transform=train_transform)
    test_dataset = ScrewDataset(test_df, root_dir, transform=test_transform, samples_per_patient=10)
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)} (ID: {args.gpu_id})")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        
    num_classes = len(train_dataset.manufacturers)
    
    # --- Optuna Study ---
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, args, train_dataset, test_dataset, num_classes, device), n_trials=args.trials)
    
    # ... (rest of main) ...
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best params
    with open('best_params.json', 'w') as f:
        json.dump(trial.params, f, indent=4)

if __name__ == '__main__':
    main()
