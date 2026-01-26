import os
import argparse
import subprocess
import json
import matplotlib.pyplot as plt
import sys
import time

QUESTIONS = [
    {
        "name": "Original Classifier (ResNet18)",
        "dir": "original_classifier",
        "script": "train.py"
    },
    {
        "name": "Mask R-CNN Classifier",
        "dir": "maskrcnn_classifier",
        "script": "train.py"
    },
    {
        "name": "DINOv2 Classifier",
        "dir": "dinov2_classifier",
        "script": "train.py"
    }
]

def run_training(approach, data_dir):
    print(f"\n{'='*50}")
    print(f"Running {approach['name']}...")
    print(f"{'='*50}\n")
    
    script_path = os.path.join(approach['dir'], approach['script'])
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return None

    # Construct command
    # python [dir]/train.py --data_dir [DATA_DIR]
    cmd = [sys.executable, script_path, '--data_dir', data_dir]
    
    start_time = time.time()
    try:
        # Run process and stream output
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error running {approach['name']}: {e}")
        return None
    
    duration = time.time() - start_time
    print(f"\nFinished {approach['name']} in {duration:.2f} seconds.")
    
    # Load metrics
    metrics_path = os.path.join(approach['dir'], 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    else:
        print(f"Warning: metrics.json not found for {approach['name']}")
        return None

def plot_comparison(results):
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Training Loss
    plt.subplot(1, 2, 1)
    for name, metrics in results.items():
        if metrics and 'train_loss' in metrics:
            plt.plot(metrics['train_loss'], label=name)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Test Accuracy
    plt.subplot(1, 2, 2)
    for name, metrics in results.items():
        if metrics and 'test_acc' in metrics:
            plt.plot(metrics['test_acc'], label=name)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png')
    print("\nBenchmark comparison saved to benchmark_comparison.png")

def main():
    parser = argparse.ArgumentParser(description='Run Orthopedic Screw Classification Benchmark')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    args = parser.parse_args()
    
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    results = {}
    
    for approach in QUESTIONS:
        metrics = run_training(approach, data_dir)
        results[approach['name']] = metrics
        
    print("\nAll training runs completed.")
    plot_comparison(results)

if __name__ == "__main__":
    main()
