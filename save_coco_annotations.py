#!/usr/bin/env python3
"""
Save COCO annotation files (without images) to a local directory.
"""
import shutil
from pathlib import Path

def save_coco_annotations():
    """Copy COCO annotation files to a local directory."""
    source_train = Path("/workspace/datasets/train/annotations/annotations.json")
    source_val = Path("/workspace/datasets/val/annotations/annotations.json")
    
    # Create output directory structure
    output_dir = Path("/workspace/coco_annotations_only")
    output_dir.mkdir(exist_ok=True)
    
    train_output = output_dir / "train" / "annotations"
    val_output = output_dir / "val" / "annotations"
    
    train_output.mkdir(parents=True, exist_ok=True)
    val_output.mkdir(parents=True, exist_ok=True)
    
    # Copy annotation files
    print(f"📋 Copying train annotations...")
    shutil.copy2(source_train, train_output / "annotations.json")
    print(f"   ✅ Saved to {train_output / 'annotations.json'}")
    
    print(f"📋 Copying val annotations...")
    shutil.copy2(source_val, val_output / "annotations.json")
    print(f"   ✅ Saved to {val_output / 'annotations.json'}")
    
    # Create a tar archive for easy transfer
    print(f"\n📦 Creating archive...")
    archive_path = output_dir.parent / "soccer_synth_coco_annotations.tar.gz"
    shutil.make_archive(
        str(archive_path).replace('.tar.gz', ''),
        'gztar',
        root_dir=output_dir.parent,
        base_dir=output_dir.name
    )
    print(f"   ✅ Created archive: {archive_path}")
    
    # Show file sizes
    train_size = (train_output / "annotations.json").stat().st_size / (1024 * 1024)
    val_size = (val_output / "annotations.json").stat().st_size / (1024 * 1024)
    archive_size = archive_path.stat().st_size / (1024 * 1024)
    
    print(f"\n📊 File sizes:")
    print(f"   Train annotations: {train_size:.2f} MB")
    print(f"   Val annotations:   {val_size:.2f} MB")
    print(f"   Archive:           {archive_size:.2f} MB")
    print(f"\n✅ Done! Files ready at:")
    print(f"   Directory: {output_dir}")
    print(f"   Archive:   {archive_path}")
    print(f"\n💡 To download to your local computer:")
    print(f"   - If using scp: scp user@host:{archive_path} .")
    print(f"   - Or copy the directory: {output_dir}")

if __name__ == "__main__":
    save_coco_annotations()
