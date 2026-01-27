#!/usr/bin/env python3
"""
Resume training from the 20-epoch checkpoint
Extracts all training configuration and sets up for continuation
"""
import sys
from pathlib import Path
import torch
import json
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def extract_training_info(checkpoint_path: str):
    """Extract all training information from checkpoint."""
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'checkpoint_path': checkpoint_path,
        'model_state': 'model' in checkpoint,
        'optimizer_state': 'optimizer' in checkpoint,
        'scheduler_state': 'lr_scheduler' in checkpoint,
        'ema_model': 'ema_model' in checkpoint,
    }
    
    # Extract args if available
    if 'args' in checkpoint:
        args = checkpoint['args']
        if hasattr(args, '__dict__'):
            args_dict = vars(args)
        else:
            args_dict = args
        
        info['training_args'] = args_dict
        info['dataset_dir'] = args_dict.get('dataset_dir', '')
        info['output_dir'] = args_dict.get('output_dir', '')
        info['num_classes'] = args_dict.get('num_classes', 2)
        info['class_names'] = args_dict.get('class_names', ['ball'])
        info['batch_size'] = args_dict.get('batch_size', 2)
        info['learning_rate'] = args_dict.get('lr', 0.0002)
        info['epochs'] = args_dict.get('epochs', 20)
        info['resolution'] = args_dict.get('resolution', 1288)
        info['encoder'] = args_dict.get('encoder', 'dinov2_windowed_small')
    
    return info, checkpoint


def create_resume_config(training_info: dict, output_path: str):
    """Create a YAML config file for resuming training."""
    config = {
        'model': {
            'type': 'rf-detr-base',
            'class_names': training_info.get('class_names', ['ball']),
            'num_classes': training_info.get('num_classes', 2),
            'encoder': training_info.get('encoder', 'dinov2_windowed_small'),
            'resolution': training_info.get('resolution', 1288),
        },
        'training': {
            'batch_size': training_info.get('batch_size', 2),
            'learning_rate': training_info.get('learning_rate', 0.0002),
            'epochs': training_info.get('epochs', 20),
            'weight_decay': 0.0001,
            'gradient_clip': 0.1,
        },
        'dataset': {
            'train_path': training_info.get('dataset_dir', ''),
            'val_path': training_info.get('dataset_dir', '').replace('/train', '/val') if '/train' in training_info.get('dataset_dir', '') else training_info.get('dataset_dir', ''),
        },
        'checkpoint': {
            'resume_from': training_info.get('checkpoint_path', ''),
            'start_epoch': training_info.get('epoch', 0) + 1,
            'save_dir': 'models/checkpoints',
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created resume config: {output_path}")
    return config


def create_resume_script(training_info: dict, checkpoint_path: str, output_path: str):
    """Create a Python script to resume training."""
    script_content = f'''#!/usr/bin/env python3
"""
Resume training from epoch {training_info.get('epoch', 0) + 1}
Checkpoint: {checkpoint_path}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_ball import main
import argparse

if __name__ == "__main__":
    # Training arguments from checkpoint
    dataset_dir = "{training_info.get('dataset_dir', '')}"
    output_dir = "{training_info.get('output_dir', 'models')}"
    checkpoint_path = "{checkpoint_path}"
    start_epoch = {training_info.get('epoch', 0) + 1}
    total_epochs = {training_info.get('epochs', 20)}
    
    print("=" * 60)
    print("RESUMING TRAINING FROM 20-EPOCH CHECKPOINT")
    print("=" * 60)
    print(f"Checkpoint: {{checkpoint_path}}")
    print(f"Starting from epoch: {{start_epoch}}")
    print(f"Total epochs: {{total_epochs}}")
    print(f"Dataset: {{dataset_dir}}")
    print("=" * 60)
    
    # Create args namespace
    class Args:
        pass
    
    args = Args()
    args.config = "configs/training.yaml"
    args.train_dir = dataset_dir
    args.val_dir = dataset_dir.replace("/train", "/val") if "/train" in dataset_dir else dataset_dir
    args.output_dir = output_dir
    args.resume = checkpoint_path
    args.epochs = total_epochs
    
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    Path(output_path).chmod(0o755)
    print(f"✅ Created resume script: {output_path}")
    return output_path


def main():
    """Main function."""
    checkpoint_path = "/workspace/soccer_cv_ball/models/soccer ball/checkpoint_20_soccer_ball.pth"
    
    print("=" * 60)
    print("EXTRACTING TRAINING INFORMATION FROM CHECKPOINT")
    print("=" * 60)
    
    # Extract info
    training_info, checkpoint = extract_training_info(checkpoint_path)
    
    print(f"\n📊 Training Information:")
    print(f"   Epoch: {training_info['epoch']} (completed {training_info['epoch'] + 1} epochs)")
    print(f"   Next epoch: {training_info['epoch'] + 1}")
    print(f"   Total epochs planned: {training_info.get('epochs', 20)}")
    print(f"   Dataset: {training_info.get('dataset_dir', 'N/A')}")
    print(f"   Output: {training_info.get('output_dir', 'N/A')}")
    print(f"   Batch size: {training_info.get('batch_size', 'N/A')}")
    print(f"   Learning rate: {training_info.get('learning_rate', 'N/A')}")
    print(f"   Resolution: {training_info.get('resolution', 'N/A')}")
    print(f"   Encoder: {training_info.get('encoder', 'N/A')}")
    print(f"   Class names: {training_info.get('class_names', [])}")
    
    print(f"\n📦 Checkpoint Contents:")
    print(f"   Model state: {'✅' if training_info['model_state'] else '❌'}")
    print(f"   Optimizer state: {'✅' if training_info['optimizer_state'] else '❌'}")
    print(f"   Scheduler state: {'✅' if training_info['scheduler_state'] else '❌'}")
    print(f"   EMA model: {'✅' if training_info['ema_model'] else '❌'}")
    
    # Save training info as JSON
    info_path = Path(__file__).parent.parent / "training_info_20_epochs.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2, default=str)
    print(f"\n✅ Saved training info: {info_path}")
    
    # Create resume config
    config_path = Path(__file__).parent.parent / "configs" / "resume_20_epochs.yaml"
    config_path.parent.mkdir(exist_ok=True)
    create_resume_config(training_info, str(config_path))
    
    # Create resume script
    script_path = Path(__file__).parent / "resume_from_20_epochs.sh"
    create_resume_script(training_info, checkpoint_path, str(script_path))
    
    print(f"\n✅ All files created! To resume training:")
    print(f"   python {script_path}")
    print(f"\n   Or use the config:")
    print(f"   python scripts/train_ball.py --config {config_path} --resume {checkpoint_path}")


if __name__ == "__main__":
    main()
