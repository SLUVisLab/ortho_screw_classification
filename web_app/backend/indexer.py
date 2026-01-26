import os
import argparse
import torch
import numpy as np
import pickle
import pandas as pd
from PIL import Image
from tqdm import tqdm
from model_wrapper import ClassifierWrapper
import sys

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# We need to import dataset utils. Since we are in web_app/backend, and utils is in ../../deployment/utils
# But we should probably copy utils to backend for self-containment.
# For now, let's assume utils is importable or just recreate simple dataset logic here since we just need paths.

def index_dataset(data_dir, model_path, output_path):
    print(f"Indexing data from {data_dir}...")
    
    # Load Metadata
    csv_path = os.path.join(data_dir, 'metadata_summary.csv')
    df = pd.read_csv(csv_path)
    
    # Initialize Model
    wrapper = ClassifierWrapper(model_path, device='cpu')
    
    embeddings = []
    metadata = []
    
    # Group by patient to get pairs
    # Logic similar to ScrewDataset but simplified for indexing loop
    patient_groups = df.groupby('patient_number')
    
    for pid, group in tqdm(patient_groups):
        manufacturer = group['manufacturer'].iloc[0]
        
        # Get all AP and Lat images for this patient
        ap_rows = group[group['view_position'] == 'AP']
        lat_rows = group[group['view_position'] == 'LATERAL']
        
        if ap_rows.empty or lat_rows.empty:
            continue
            
        # For indexing, we ideally want high quality samples. 
        # Let's take the first available pair for each patient to represent that patient/class.
        # Or maybe index multiple? Let's index ONE pair per patient to keep index small and diverse.
        
        # Index ALL valid combinations of AP/Lateral for this patient
        for _, ap_row in ap_rows.iterrows():
            for _, lat_row in lat_rows.iterrows():
                ap_path = os.path.join(data_dir, ap_row['relative_file_path'])
                lat_path = os.path.join(data_dir, lat_row['relative_file_path'])
                
                if not os.path.exists(ap_path) or not os.path.exists(lat_path):
                    continue
                    
                try:
                    # We need to pass BYTES to the wrapper
                    with open(ap_path, 'rb') as f:
                        ap_bytes = f.read()
                    with open(lat_path, 'rb') as f:
                        lat_bytes = f.read()
                        
                    # 1. Canonical Embedding (No augmentation) - Good for exact matches
                    _, embedding = wrapper.predict(ap_bytes, lat_bytes, augment=False)
                    embeddings.append(embedding)
                    metadata.append({
                        'patient_number': pid,
                        'manufacturer': manufacturer,
                        'ap_path': ap_row['relative_file_path'],
                        'lat_path': lat_row['relative_file_path'],
                        'augmented': False
                    })

                    # 2. Augmented Embeddings (Simulate different crops/rotations)
                    # Generate 10 variations to cover the visual space better
                    for _ in range(10):
                        _, embedding = wrapper.predict(ap_bytes, lat_bytes, augment=True)
                        embeddings.append(embedding)
                        metadata.append({
                            'patient_number': pid,
                            'manufacturer': manufacturer,
                            'ap_path': ap_row['relative_file_path'],
                            'lat_path': lat_row['relative_file_path'],
                            'augmented': True
                        })
                    
                except Exception as e:
                    print(f"Error processing patient {pid}: {e}")
                    continue

    # Save Index
    index_data = {
        'embeddings': np.array(embeddings),
        'metadata': metadata
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(index_data, f)
        
    print(f"Indexed {len(embeddings)} items to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_path', default='embeddings.pkl')
    args = parser.parse_args()
    
    index_dataset(args.data_dir, args.model_path, args.output_path)
