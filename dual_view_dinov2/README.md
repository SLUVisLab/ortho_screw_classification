# Dual-view DINOv2 screw classification

This package contains the focused dual-view training and inference code for the project handoff. It uses a shared, frozen DINOv2 ViT-S/14 backbone to encode AP and lateral screw crops, concatenates the two 384-dimensional features, and applies a 768 → 512 → 4 classifier with ReLU and dropout 0.5.

The recovered March 2026 training log records **58.75% evaluation accuracy at epochs 49 and 50** of a 50-epoch run. Rounded epoch metrics are preserved in [evidence/dual_view_training_metrics.json](evidence/dual_view_training_metrics.json). The evaluation set was used to choose the checkpoint; this is not an independent final-test result.

The recovered `dinov2_dual_view.pth` checkpoint is supported by this code. Its SHA-256 is `73ddd94b7ec9bc55eead519e4b1f1680c1542357cbfcf24ed1a09494416ee4f5`. A fresh diagnostic evaluation on November 18, 2026 scored **59.58% over 240 sampled pairs** on the reconstructed evaluation split. This January checkpoint is separate from the March 58.75% run, and the fresh result is not an independent final test. Obtain the trained checkpoint separately; model weights and patient data are not added by this change. Austin manages data handoff.

## Install

From the repository root, using Python 3.10 (the verified version):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r dual_view_dinov2/requirements.txt
```

The pinned versions are the locally verified environment. For CUDA, install the matching CUDA-enabled PyTorch and torchvision builds. Training selects CUDA, then MPS, then CPU with `--device auto`.

The small DINOv2 architecture dependency is vendored with its Apache 2.0 license and source-file checksums. Training downloads the official pretrained backbone once and verifies its SHA-256. For offline training, supply `--backbone-weights /path/to/dinov2_vits14_pretrain.pth`. Loading a complete trained screw checkpoint does not require a network connection or pretrained initialization file.

## Predict with the recovered checkpoint

```bash
python -m dual_view_dinov2.predict \
  --checkpoint /path/to/dinov2_dual_view.pth \
  --ap /path/to/ap_crop.png \
  --lateral /path/to/lateral_crop.png
```

Inputs are screw-centered context crops from AP and lateral images of the same patient, approximately twice the screw bounding box's largest dimension. The training sampler selects a screw independently in each view; it does not establish anatomical correspondence between individual screws. Inference resizes to 224 × 224, converts to three-channel grayscale, and applies ImageNet normalization.

The script returns softmax scores in this fixed class order:

| Index | Manufacturer |
| --- | --- |
| 0 | Depuy expedium (Synthes) |
| 1 | medtronic solera |
| 2 | nuvasive reline |
| 3 | seaspine mariner |

The loader checks the dual-view fusion dimensions, strictly loads all tensors, and checks class order when a checkpoint records it. Missing or incompatible weights fail explicitly.

## Evaluate the checkpoint

```bash
python -m dual_view_dinov2.evaluate \
  --checkpoint /path/to/dinov2_dual_view.pth \
  --data-dir /path/to/dataset
```

This reproduces the diagnostic procedure: historical patient split, seed 42, four data-loader workers, batch size 32, and 10 sampled pairs per evaluation patient. The recorded 59.58% diagnostic used MPS. Aggregate results and its confusion matrix are in [evidence/local_dual_evaluation.json](evidence/local_dual_evaluation.json). Sampling and hardware settings can affect a fresh result; this is not a new independent test split.

## Train a new run

```bash
python -m dual_view_dinov2.train \
  --data-dir /path/to/dataset \
  --output-dir /path/to/new_run
```

The output directory must not already exist. Defaults are 50 epochs, batch size 128, Adam with learning rate `1e-4` and weight decay `1e-4`, cross-entropy loss, and classifier-only optimization. Patient sampling is weighted by inverse manufacturer frequency. The dataset uses masks to select screw contours larger than 100 pixels in area and produces context crops with center jitter. Training adds random crops and horizontal/vertical flips; evaluation samples 10 pairs per patient and skips invalid pairs.

The dataset root must contain `metadata_summary.csv` with `manufacturer`, `patient_number`, `view_position`, and `relative_file_path` columns. Views are `AP` and `LATERAL`. For an image at `.../images/example.jpg`, its mask is `.../masks/example_mask.jpg`.

The historical filtering removes missing manufacturer labels, `Styker everest (K2M)`, `orthofix firebird`, and manufacturers with fewer than 10 patients. The source data produces 104 training patients and 26 evaluation patients. The verified metadata SHA-256 is `5966754d7ddb6e6faef2c254632a029e7775e3ab8e4c8442f47cfad9b3be4528`.

Outputs are `best_dual_view.pth`, `last_dual_view.pth`, `metrics.json`, and `run_config.json`. Checkpoints record model state, epoch, class order, metrics, and configuration. The configuration contains patient split IDs and should travel with the controlled data handoff.

## Reproduction limits

This implementation adapts the dual-view recipe preserved in commit [`ea2f49d`](https://github.com/SLUVisLab/ortho_screw_classification/tree/ea2f49d55b2872a1e7ebf245e1615b5213da0a22). The exact source snapshot and RNG state for the March run were not retained. The added seed, explicit device/output selection, empty-batch handling, checksum verification, and checkpoint metadata make a new run inspectable; they do not guarantee the historical score.

`--split-mode historical` preserves the original patient/stratification-label ordering mismatch for comparison. `--split-mode aligned` corrects that ordering and creates a different split. Both use an 80/20 split with `random_state=42` and use evaluation accuracy for checkpoint selection. For a new final performance claim, establish separate patient-disjoint training, validation, and untouched test sets first.

## Verification

Run the focused regression checks from the repository root:

```bash
python -m unittest discover -s dual_view_dinov2/tests -v
```

The handoff code is also checked against the recovered checkpoint with real AP/lateral crops. This compatibility check is separate from evaluating its historical accuracy.
