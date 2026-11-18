"""Dual-view DINOv2 architecture compatible with the recovered screw weights."""

import hashlib
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "dinov2"))
from dinov2.hub.backbones import dinov2_vits14

CLASSES = [
    "Depuy expedium (Synthes)",
    "medtronic solera",
    "nuvasive reline",
    "seaspine mariner",
]
BACKBONE_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
BACKBONE_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def pretrained_backbone_path(path=None):
    """Resolve a local initialization or download and verify the known backbone."""
    if path is None:
        path = Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vits14_pretrain.pth"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(BACKBONE_URL, str(path), hash_prefix=BACKBONE_SHA256)
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != BACKBONE_SHA256:
        raise ValueError(f"DINOv2 pretrained backbone checksum mismatch: {path}")
    return path


class DualViewDINOv2(nn.Module):
    def __init__(self, initialize_backbone=True, backbone_weights=None):
        super().__init__()
        self.backbone = dinov2_vits14(pretrained=False)
        if initialize_backbone:
            weights = torch.load(pretrained_backbone_path(backbone_weights), map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(weights, strict=True)
        self.backbone.requires_grad_(False)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(CLASSES)),
        )

    def forward(self, ap_imgs, lat_imgs):
        with torch.no_grad():
            ap = self.backbone(ap_imgs)
            lat = self.backbone(lat_imgs)
        return self.classifier(torch.cat((ap, lat), dim=1))


def load_checkpoint(path, device="cpu"):
    """Load legacy state dictionaries or the trainer's annotated checkpoints."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    class_order = checkpoint.get("config", {}).get("classes", CLASSES)
    if class_order != CLASSES:
        raise ValueError("Checkpoint class order does not match the documented manufacturer order.")
    if "classifier.0.weight" not in state or tuple(state["classifier.0.weight"].shape) != (512, 768):
        raise ValueError("Expected dual-view DINOv2 fusion weights with shape (512, 768).")
    model = DualViewDINOv2(initialize_backbone=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
