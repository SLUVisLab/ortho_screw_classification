#!/bin/bash

# Configuration
# Point this to where the 'original_classifier_best.pth' is located
export MODEL_PATH="../../original_classifier/original_classifier_best.pth"

# Point this to the root of your dataset (where metadata_summary.csv is)
export DATA_DIR="../../../full_images_with_masks_batch_1"

# Point this to where you want the index file saved/loaded from
export INDEX_PATH="embeddings.pkl"

if [ "$1" == "index" ]; then
    echo "Running Indexer..."
    # Generate index
    # We pass data_dir explicitly as argument to indexer
    python indexer.py --data_dir "$DATA_DIR" --model_path "$MODEL_PATH" --output_path "$INDEX_PATH"
else
    echo "Starting Backend Server..."
    python main.py
fi
