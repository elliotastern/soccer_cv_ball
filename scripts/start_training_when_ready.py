#!/usr/bin/env python3
"""
Wait for combined dataset to be ready, then start optimized training.
"""

import time
import subprocess
import sys
from pathlib import Path

def check_dataset_ready():
    """Check if combined dataset is ready."""
    dataset_file = Path("/workspace/datasets/combined_ball_only/_annotations.coco.json")
    return dataset_file.exists()

def wait_for_dataset(max_wait_minutes=30):
    """Wait for dataset to be ready."""
    print("=" * 70)
    print("WAITING FOR DATASET COMBINATION TO COMPLETE")
    print("=" * 70)
    
    wait_seconds = max_wait_minutes * 60
    check_interval = 10  # Check every 10 seconds
    elapsed = 0
    
    while elapsed < wait_seconds:
        if check_dataset_ready():
            print(f"\n✅ Dataset is ready!")
            return True
        
        if elapsed % 60 == 0:  # Print every minute
            print(f"⏳ Waiting... ({elapsed // 60} minutes elapsed)")
        
        time.sleep(check_interval)
        elapsed += check_interval
    
    print(f"\n⚠️  Timeout after {max_wait_minutes} minutes")
    return False

def get_dataset_stats():
    """Get statistics about the combined dataset."""
    import json
    dataset_file = Path("/workspace/datasets/combined_ball_only/_annotations.coco.json")
    if dataset_file.exists():
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        return len(data['images']), len(data['annotations'])
    return 0, 0

def split_combined_dataset():
    """Split combined dataset into train/val."""
    import json
    import shutil
    import random
    
    print("\n📊 Splitting combined dataset into train/val...")
    
    combined_file = Path("/workspace/datasets/combined_ball_only/_annotations.coco.json")
    combined_dir = Path("/workspace/datasets/combined_ball_only")
    
    train_dir = Path("/workspace/datasets/combined_ball_only/train")
    val_dir = Path("/workspace/datasets/combined_ball_only/val")
    
    # Check if already split
    if (train_dir / "_annotations.coco.json").exists() and (val_dir / "_annotations.coco.json").exists():
        print("✅ Dataset already split")
        return str(train_dir), str(val_dir)
    
    # Load combined dataset
    with open(combined_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    random.shuffle(images)
    
    # Split 80/20
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Create ID maps
    train_id_map = {img['id']: idx + 1 for idx, img in enumerate(train_images)}
    val_id_map = {img['id']: idx + 1 for idx, img in enumerate(val_images)}
    
    # Process train
    train_annos = []
    train_ann_id = 1
    for img in train_images:
        src = combined_dir / img['file_name']
        dst = train_dir / img['file_name']
        if src.exists():
            shutil.copy2(src, dst)
        
        for ann in coco_data['annotations']:
            if ann['image_id'] == img['id']:
                new_ann = ann.copy()
                new_ann['id'] = train_ann_id
                new_ann['image_id'] = train_id_map[img['id']]
                train_annos.append(new_ann)
                train_ann_id += 1
    
    # Process val
    val_annos = []
    val_ann_id = 1
    for img in val_images:
        src = combined_dir / img['file_name']
        dst = val_dir / img['file_name']
        if src.exists():
            shutil.copy2(src, dst)
        
        for ann in coco_data['annotations']:
            if ann['image_id'] == img['id']:
                new_ann = ann.copy()
                new_ann['id'] = val_ann_id
                new_ann['image_id'] = val_id_map[img['id']]
                val_annos.append(new_ann)
                val_ann_id += 1
    
    # Update image IDs
    train_images_updated = [{**img, 'id': train_id_map[img['id']]} for img in train_images]
    val_images_updated = [{**img, 'id': val_id_map[img['id']]} for img in val_images]
    
    # Save train
    train_coco = {
        "info": coco_data['info'],
        "licenses": coco_data['licenses'],
        "images": train_images_updated,
        "annotations": train_annos,
        "categories": coco_data['categories']
    }
    with open(train_dir / "_annotations.coco.json", 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    # Save val
    val_coco = {
        "info": coco_data['info'],
        "licenses": coco_data['licenses'],
        "images": val_images_updated,
        "annotations": val_annos,
        "categories": coco_data['categories']
    }
    with open(val_dir / "_annotations.coco.json", 'w') as f:
        json.dump(val_coco, f, indent=2)
    
    print(f"✅ Split complete: {len(train_images)} train, {len(val_images)} val")
    return str(train_dir), str(val_dir)

def start_training():
    """Start optimized training."""
    print("\n" + "=" * 70)
    print("STARTING OPTIMIZED TRAINING")
    print("=" * 70)
    
    # Split dataset first
    train_dir, val_dir = split_combined_dataset()
    
    # Get dataset stats
    num_images, num_annos = get_dataset_stats()
    print(f"\n📊 Dataset: {num_images} images, {num_annos} annotations")
    
    # Check GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"🎮 GPU: {result.stdout.strip()}")
    except:
        print("⚠️  Could not detect GPU")
    
    # Start training
    config_path = Path(__file__).parent.parent / "configs" / "rtdetr_r50vd_ball_combined_optimized.yml"
    
    print(f"\n🚀 Starting training with config: {config_path}")
    print(f"   Optimizations enabled:")
    print(f"   • Mixed precision (FP16/FP32)")
    print(f"   • TF32 acceleration")
    print(f"   • Channels-last memory format")
    print(f"   • Large batch size (16) with gradient accumulation (4)")
    print(f"   • 8 data loading workers")
    print(f"   • Persistent workers")
    print("\n" + "=" * 70)
    
    # Run training
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "scripts" / "train_ball.py"),
        "--config",
        str(config_path)
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # Start training (this will run in foreground)
    subprocess.run(cmd)

def main():
    # Wait for dataset
    if not wait_for_dataset(max_wait_minutes=30):
        print("\n❌ Dataset not ready. Please check the combination process.")
        print("   Monitor with: tail -f /tmp/combine_datasets.log")
        return 1
    
    # Start training
    start_training()
    return 0

if __name__ == "__main__":
    sys.exit(main())
