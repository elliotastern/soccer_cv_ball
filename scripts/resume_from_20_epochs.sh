#!/usr/bin/env python3
"""
Resume training from epoch 20
Checkpoint: /workspace/soccer_cv_ball/models/soccer ball/checkpoint_20_soccer_ball.pth

This script extracts all training configuration from the checkpoint and
sets up the environment to continue training from epoch 20.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Resume training from 20-epoch checkpoint."""
    # Load training info
    info_path = Path(__file__).parent.parent / "training_info_20_epochs.json"
    with open(info_path, 'r') as f:
        training_info = json.load(f)
    
    checkpoint_path = "/workspace/soccer_cv_ball/models/soccer ball/checkpoint_20_soccer_ball.pth"
    args_dict = training_info.get('training_args', {})
    
    print("=" * 60)
    print("RESUMING TRAINING FROM 20-EPOCH CHECKPOINT")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Completed epochs: {training_info['epoch'] + 1} (0-{training_info['epoch']})")
    print(f"Starting from epoch: {training_info['epoch'] + 1}")
    print(f"Dataset: {training_info.get('dataset_dir', 'N/A')}")
    print(f"Batch size: {training_info.get('batch_size', 'N/A')}")
    print(f"Learning rate: {training_info.get('learning_rate', 'N/A')}")
    print(f"Resolution: {training_info.get('resolution', 'N/A')}")
    print("=" * 60)
    
    # Import and run training
    from scripts.train_ball import main as train_main
    import argparse
    
    # Create args object
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training.yaml')
    parser.add_argument('--output-dir', type=str, default='models')
    args = parser.parse_args(['--config', 'configs/resume_20_epochs.yaml', '--output-dir', 'models'])
    
    # Note: The train_ball.py script will automatically detect and resume from checkpoint
    # if it's in the output directory. For explicit resume, you may need to modify
    # the script to accept a --resume parameter.
    
    print("\n⚠️  Note: You may need to update the dataset path in configs/resume_20_epochs.yaml")
    print("   Original dataset: {}".format(training_info.get('dataset_dir', 'N/A')))
    print("\n   To resume, ensure:")
    print("   1. Dataset path is correct in config")
    print("   2. Checkpoint is accessible")
    print("   3. Run: python scripts/train_ball.py --config configs/resume_20_epochs.yaml")
    
    # Uncomment to actually run training:
    # train_main()

if __name__ == "__main__":
    main()
