"""Run a recovered or newly trained dual-view checkpoint on two context crops."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import json
from PIL import Image
import torch
if __package__:
    from .model import CLASSES, load_checkpoint
    from .train import make_transforms
else:
    from model import CLASSES, load_checkpoint
    from train import make_transforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ap", required=True, help="AP screw context crop, approximately twice the screw's largest dimension")
    parser.add_argument("--lateral", required=True, help="Lateral screw context crop from the same patient")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, args.device)
    transform = make_transforms(False)
    ap = transform(Image.open(args.ap).convert("RGB")).unsqueeze(0).to(args.device)
    lat = transform(Image.open(args.lateral).convert("RGB")).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        probabilities = model(ap, lat).softmax(1)[0].cpu().tolist()
    print(json.dumps(dict(zip(CLASSES, probabilities)), indent=2))


if __name__ == "__main__":
    main()
