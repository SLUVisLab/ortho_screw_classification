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



def main():
    parser = argparse.ArgumentParser(description='Train Original Classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    args = parser.parse_args()

    # --- Configuration ---
    CSV_PATH = os.path.join(args.data_dir, 'metadata_summary.csv')
    ROOT_DIR = args.data_dir
    BATCH_SIZE = 16 
    EPOCHS = 100
    LR = 1e-4
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading metadata...")
    df = pd.read_csv(CSV_PATH)
    
    # Filter irrelevant rows
    # The prompt said: "I don't care about 'No Accession Number Match' or 'No manufacturer data'"
    df = df[~df['manufacturer'].isin(['No Accession Number Match', 'No manufacturer data'])]
    
    # Explicitly drop classes requested by user
    EXPLICIT_DROP = ['Styker everest (K2M)', 'orthofix firebird']
    if EXPLICIT_DROP:
        print(f"Explicitly dropping: {EXPLICIT_DROP}")
        df = df[~df['manufacturer'].isin(EXPLICIT_DROP)]

    # Get unique patient numbers
    patients = df['patient_number'].unique()
    print(f"Total valid patients before filtering small classes: {len(patients)}")

    # Filter out manufacturers with too few patients to support a robust train/test split
    MIN_PATIENTS = 10
    counts = df.groupby('manufacturer')['patient_number'].nunique()
    drop_classes = counts[counts < MIN_PATIENTS].index.tolist()
    
    if drop_classes:
        print(f"Dropping manufacturers with < {MIN_PATIENTS} patients: {drop_classes}")
        df = df[~df['manufacturer'].isin(drop_classes)]
        # Re-fetch patients after filtering
        patients = df['patient_number'].unique()

    print(f"Total valid patients after filtering: {len(patients)}")
    
    # Stratified Split
    # We need one label per patient for stratification
    # Group by patient and take the first manufacturer (assuming patient only has one manufacturer)
    patient_labels = df.groupby('patient_number')['manufacturer'].first()
    
    train_patients, test_patients = train_test_split(
        patients, 
        test_size=0.2, 
        random_state=42,
        stratify=patient_labels
    )
    print(f"Train patients: {len(train_patients)}, Test patients: {len(test_patients)}")
    
    # Create DataFrames for splits
    train_df = df[df['patient_number'].isin(train_patients)]
    test_df = df[df['patient_number'].isin(test_patients)]
    
    # Transforms
    # We now receive a ~2.0x context crop (GRAYSCALE). 
    # Resize to 400x400 (screw becomes ~200px), Rotate, then CenterCrop to 224x224 to eliminate borders.
    # Note: X-ray models expect 1 channel. torchxrayvision normalization is usually specific, 
    # but we will use standard [0.5, 0.5] for robust scaling or look up xrv defaults.
    # xrv typically expects values between -1024 and 1024. But since we lack Hounsfield units (these are JPEGs), 
    # we'll stick to 0-1 range and let the model handle it or use simple normalization.
    # Let's use mean=0.5, std=0.5 for now to get -1 to 1 range which is safe.
    
    train_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.RandomRotation(15), 
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Grayscale(num_output_channels=3), # Convert to 3 channels for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((400, 400)), # Maintain same scale as Train
        transforms.CenterCrop((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = ScrewDataset(train_df, ROOT_DIR, transform=train_transform)
    # Increase test set size by sampling multiple pairs per patient
    test_dataset = ScrewDataset(test_df, ROOT_DIR, transform=test_transform, samples_per_patient=50)
    
    num_classes = len(train_dataset.manufacturers)
    print(f"Classes: {train_dataset.manufacturers}")
    
    # --- Weighted Random Sampler for Training ---
    # Calculate weights based on patient count per manufacturer in Train
    # Get manufacturer for each patient in the dataset
    train_labels = []
    for pid in train_dataset.patient_ids:
        # We can look up the manufacturer from the dataframe directly to be faster
        # Assuming one manufacturer per patient
        m = train_df[train_df['patient_number'] == pid]['manufacturer'].iloc[0]
        train_labels.append(train_dataset.class_to_idx[m])
    
    class_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn, num_workers=2)
    # Test loader without shuffling, we will manually sample for balanced eval
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # --- Model ---
    model = DualViewClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4) # Added weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # Added learning rate scheduler

    # --- Training Loop ---
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

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
