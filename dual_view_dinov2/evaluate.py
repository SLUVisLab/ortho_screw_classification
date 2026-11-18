"""Diagnostic evaluation on the reconstructed patient selection split."""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__:
    from .model import CLASSES, load_checkpoint
    from .train import prepare_data, collate_valid, seed_worker
else:
    from model import CLASSES, load_checkpoint
    from train import prepare_data, collate_valid, seed_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.workers < 0 or args.batch_size < 1:
        parser.error("workers must be nonnegative and batch-size must be positive")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    network = load_checkpoint(args.checkpoint, device)
    _, dataset, _, evaluation_ids = prepare_data(args.data_dir, "historical")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_valid, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(args.seed))
    confusion = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int64)
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            ap, lateral, labels = batch
            logits = network(ap.to(device), lateral.to(device)).cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model produced nonfinite scores.")
            for label, prediction in zip(labels, logits.argmax(1)):
                confusion[label, prediction] += 1
    total, correct = confusion.sum().item(), confusion.diag().sum().item()
    if not total:
        raise RuntimeError("No valid evaluation pairs; verify image and mask paths.")
    result = dict(checkpoint=args.checkpoint.name,
                  sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  device=device, seed=args.seed, workers=args.workers, batch_size=args.batch_size,
                  split="published historical patient split; random_state 42",
                  evaluation_patients=len(evaluation_ids), valid_pairs=total, correct_pairs=correct,
                  accuracy_percent=100 * correct / total,
                  confusion_matrix_true_rows_pred_columns=confusion.tolist(), classes=CLASSES,
                  limitations="Diagnostic on reconstructed selection split; not an independent final test.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
