"""Reference reconstruction of the dual-view recipe, not an exact historical rerun.

Based on ea2f49d:dinov2_classifier/train.py and utils/dataset.py. Preserves
the historical split by default; records its patient IDs for future auditing.
New behavior: explicit seed/device/output directory, empty-batch handling,
verified backbone initialization, and checkpoint provenance. The recovered
checkpoint is supported; a new training run is not an exact historical rerun.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import hashlib
import json
import random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
if __package__:
    from .model import DualViewDINOv2, CLASSES
    from .dataset import ScrewDataset
else:
    from model import DualViewDINOv2, CLASSES
    from dataset import ScrewDataset


def collate_valid(batch):
    batch = [x for x in batch if x is not None]
    return torch.utils.data.default_collate(batch) if batch else None


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_transforms(training=False):
    steps = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()] if training else [transforms.Resize((224, 224))]
    return transforms.Compose(steps + [transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def prepare_data(root, split_mode="historical"):
    df = pd.read_csv(Path(root) / "metadata_summary.csv")
    required = {"manufacturer", "patient_number", "view_position", "relative_file_path"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(df.columns))}")
    df = df[~df.manufacturer.isin(["No Accession Number Match", "No manufacturer data", "Styker everest (K2M)", "orthofix firebird"])]
    counts = df.groupby("manufacturer").patient_number.nunique()
    df = df[df.manufacturer.isin(counts[counts >= 10].index)]
    if sorted(df.manufacturer.unique()) != CLASSES:
        raise ValueError("Dataset does not contain exactly the four documented classes after filtering.")
    if df.groupby("patient_number").manufacturer.nunique().max() != 1:
        raise ValueError("A patient has multiple manufacturer labels.")
    patients = df.patient_number.unique()
    labels = df.groupby("patient_number").manufacturer.first()
    # Historical source passes sorted group labels against first-seen patient order.
    # Keep that behavior only for comparison; aligned is a different split.
    if split_mode == "aligned":
        labels = labels.reindex(patients)
    train_ids, evaluation_ids = train_test_split(patients, test_size=.2, random_state=42, stratify=labels)
    train = ScrewDataset(df[df.patient_number.isin(train_ids)], root, transform=make_transforms(True))
    evaluation = ScrewDataset(df[df.patient_number.isin(evaluation_ids)], root, transform=make_transforms(False), samples_per_patient=10)
    return train, evaluation, train_ids, evaluation_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", "--data_dir", required=True, type=Path)
    parser.add_argument("--backbone-weights", type=Path, help="Optional local DINOv2 pretrained initialization; otherwise use the verified download cache.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Must not already exist; prevents overwriting a run.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["historical", "aligned"], default="historical")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new run directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train, evaluation, train_ids, evaluation_ids = prepare_data(args.data_dir, args.split_mode)
    labels = [train.class_to_idx[train.patient_groups.get_group(p).manufacturer.iloc[0]] for p in train.patient_ids]
    counts = Counter(labels)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler([1 / counts[y] for y in labels], len(labels), replacement=True, generator=generator)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_valid, worker_init_fn=seed_worker, generator=generator, pin_memory=(device == "cuda"))
    train_loader = DataLoader(train, sampler=sampler, **loader_args)
    eval_loader = DataLoader(evaluation, shuffle=False, **loader_args)
    model = DualViewDINOv2(backbone_weights=args.backbone_weights).to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=device, classes=CLASSES, learning_rate=1e-4, weight_decay=1e-4, dropout=.5,
                  provenance="2026-11-18 reference recipe; new training run",
                  metadata_sha256=hashlib.sha256((args.data_dir / "metadata_summary.csv").read_bytes()).hexdigest(),
                  train_patients=train_ids.tolist(), evaluation_patients=evaluation_ids.tolist(),
                  torch_version=str(torch.__version__))
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum, batches = 0, 0, 0., 0
        for batch in train_loader:
            if batch is None:
                continue
            ap, lat, labels = (x.to(device) for x in batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ap, lat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total += labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            loss_sum += loss.item()
            batches += 1
        if not total:
            raise RuntimeError("No valid training pairs; verify image and mask paths.")
        model.eval()
        eval_total, eval_correct = 0, 0
        with torch.inference_mode():
            for batch in eval_loader:
                if batch is None:
                    continue
                ap, lat, labels = (x.to(device) for x in batch)
                logits = model(ap, lat)
                eval_total += labels.numel()
                eval_correct += (logits.argmax(1) == labels).sum().item()
        if not eval_total:
            raise RuntimeError("No valid evaluation pairs; verify image and mask paths.")
        record = dict(epoch=epoch, train_loss=loss_sum / batches, train_accuracy=correct / total,
                      selection_accuracy=eval_correct / eval_total, evaluated_pairs=eval_total,
                      skipped_virtual_pairs=len(evaluation) - eval_total)
        history.append(record)
        print(json.dumps(record), flush=True)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        checkpoint = dict(model_state_dict=state, epoch=epoch, metrics=record, config=config)
        if record["selection_accuracy"] > best:
            best = record["selection_accuracy"]
            torch.save(checkpoint, args.output_dir / "best_dual_view.pth")
        torch.save(checkpoint, args.output_dir / "last_dual_view.pth")
        (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
