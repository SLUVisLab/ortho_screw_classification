# Orthopedic Screw Classification Benchmark

This repository contains code to train and benchmark three different approaches for classifying orthopedic screws from dual-view X-rays (AP and Lateral).

## Approaches

1.  **Original Classifier (ResNet18)**:
    *   Uses `torchvision.models.resnet18` pretrained on ImageNet.
    *   Standard concatenation fusion of view features.
    *   Code: `original_classifier/`

2.  **Mask R-CNN Classifier**:
    *   Uses `maskrcnn_resnet50_fpn`.
    *   Trains with segmentation masks (bounding box + segmentation).
    *   Code: `maskrcnn_classifier/`

3.  **DINOv2 Classifier**:
    *   Uses **DINOv2 ViT-S/14** (frozen backbone) as a feature extractor.
    *   Trains a fusion MLP head on top of the [CLS] tokens.
    *   Code: `dinov2_classifier/`

## Setup

1.  **Create and Activate Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Data**:
    *   You need the path to the directory containing `metadata_summary.csv` and the image subfolders (e.g., `full_data`).
    *   The scripts expect the CSV to have columns: `manufacturer`, `patient_number`, `view_position` ('AP'/'LATERAL'), and `relative_file_path`.

## Running the Benchmark

To run all three models sequentially and generate a comparison plot:

```bash
python run_benchmark.py --data_dir /path/to/your/dataset
```

### Output
The script will produce:
*   `original_classifier/metrics.json`
*   `maskrcnn_classifier/metrics.json`
*   `dinov2_classifier/metrics.json`
*   **`benchmark_comparison.png`**: A combined plot of Training Loss and Test Accuracy for all approaches.

## Running Individual Models

You can also run models individually:

```bash
python original_classifier/train.py --data_dir /path/to/dataset
python maskrcnn_classifier/train.py --data_dir /path/to/dataset
python dinov2_classifier/train.py --data_dir /path/to/dataset
```
