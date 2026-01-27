#!/usr/bin/env python3
"""
Combine Open Soccer Ball Dataset and soccersynth_sub_sub into a single ball-only dataset.
"""

import json
import shutil
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from scripts.voc_to_coco import convert_voc_to_coco_ball_only


def merge_coco_datasets(
    dataset1_path: Path,
    dataset2_path: Path,
    output_path: Path,
    id_offset: int = 10000
) -> Tuple[int, int]:
    """
    Merge two COCO format datasets (both should be ball-only).
    
    Args:
        dataset1_path: Path to first COCO dataset
        dataset2_path: Path to second COCO dataset
        output_path: Path to save merged dataset
        id_offset: Offset for image IDs from dataset2 to ensure uniqueness
    
    Returns:
        Tuple of (total_images, total_annotations)
    """
    print(f"\n🔄 Merging COCO datasets...")
    print(f"   Dataset 1: {dataset1_path}")
    print(f"   Dataset 2: {dataset2_path}")
    print(f"   Output: {output_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset 1
    dataset1_ann_file = dataset1_path / "_annotations.coco.json"
    if not dataset1_ann_file.exists():
        raise FileNotFoundError(f"Dataset 1 annotation file not found: {dataset1_ann_file}")
    
    with open(dataset1_ann_file, 'r') as f:
        dataset1_data = json.load(f)
    
    dataset1_images = dataset1_data.get('images', [])
    dataset1_annotations = dataset1_data.get('annotations', [])
    
    print(f"   Dataset 1: {len(dataset1_images)} images, {len(dataset1_annotations)} annotations")
    
    # Load dataset 2
    dataset2_ann_file = dataset2_path / "_annotations.coco.json"
    if not dataset2_ann_file.exists():
        raise FileNotFoundError(f"Dataset 2 annotation file not found: {dataset2_ann_file}")
    
    with open(dataset2_ann_file, 'r') as f:
        dataset2_data = json.load(f)
    
    dataset2_images = dataset2_data.get('images', [])
    dataset2_annotations = dataset2_data.get('annotations', [])
    
    print(f"   Dataset 2: {len(dataset2_images)} images, {len(dataset2_annotations)} annotations")
    
    # Create image ID maps
    dataset2_image_id_map = {img['id']: img['id'] + id_offset for img in dataset2_images}
    
    # Update dataset2 image IDs
    dataset2_images_updated = []
    for img in dataset2_images:
        new_img = img.copy()
        new_img['id'] = dataset2_image_id_map[img['id']]
        dataset2_images_updated.append(new_img)
    
    # Update dataset2 annotation image IDs
    dataset2_annotations_updated = []
    new_ann_id = len(dataset1_annotations) + 1
    for ann in dataset2_annotations:
        new_ann = ann.copy()
        new_ann['id'] = new_ann_id
        new_ann['image_id'] = dataset2_image_id_map[ann['image_id']]
        dataset2_annotations_updated.append(new_ann)
        new_ann_id += 1
    
    # Merge datasets
    merged_images = dataset1_images + dataset2_images_updated
    merged_annotations = dataset1_annotations + dataset2_annotations_updated
    
    # Copy images from dataset1
    for img in dataset1_images:
        src_img = dataset1_path / img['file_name']
        dst_img = output_path / img['file_name']
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)
    
    # Copy images from dataset2
    for img in dataset2_images:
        src_img = dataset2_path / img['file_name']
        dst_img = output_path / img['file_name']
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)
    
    # Create merged COCO data
    merged_data = {
        "info": {
            "description": "Combined Open Soccer Ball Dataset + SoccerSynth Sub Sub - Ball only",
            "version": "1.0"
        },
        "licenses": [],
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": [
            {
                "id": 0,
                "name": "ball",
                "supercategory": "object"
            }
        ]
    }
    
    # Save merged annotations
    annotation_file = output_path / "_annotations.coco.json"
    with open(annotation_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"✅ Merged: {len(merged_images)} images, {len(merged_annotations)} annotations")
    print(f"✅ Saved to {annotation_file}")
    
    return len(merged_images), len(merged_annotations)


def split_dataset(
    source_dir: Path,
    train_dir: Path,
    val_dir: Path,
    split_ratio: float = 0.8
) -> Tuple[int, int]:
    """
    Split COCO dataset into train/val splits.
    """
    print(f"\n📊 Splitting dataset (train: {split_ratio:.0%}, val: {1-split_ratio:.0%})...")
    
    # Load COCO annotations
    coco_file = source_dir / "_annotations.coco.json"
    if not coco_file.exists():
        raise FileNotFoundError(f"COCO annotations not found: {coco_file}")
    
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get all images
    images = coco_data['images']
    random.shuffle(images)
    
    # Split
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image ID maps
    train_image_id_map = {img['id']: idx + 1 for idx, img in enumerate(train_images)}
    val_image_id_map = {img['id']: idx + 1 for idx, img in enumerate(val_images)}
    
    # Process train split
    train_annotations = []
    train_ann_id = 1
    for img in train_images:
        # Copy image
        src_img = source_dir / img['file_name']
        dst_img = train_dir / img['file_name']
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        
        # Get annotations for this image
        for ann in coco_data['annotations']:
            if ann['image_id'] == img['id']:
                new_ann = ann.copy()
                new_ann['id'] = train_ann_id
                new_ann['image_id'] = train_image_id_map[img['id']]
                train_annotations.append(new_ann)
                train_ann_id += 1
    
    # Process val split
    val_annotations = []
    val_ann_id = 1
    for img in val_images:
        # Copy image
        src_img = source_dir / img['file_name']
        dst_img = val_dir / img['file_name']
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        
        # Get annotations for this image
        for ann in coco_data['annotations']:
            if ann['image_id'] == img['id']:
                new_ann = ann.copy()
                new_ann['id'] = val_ann_id
                new_ann['image_id'] = val_image_id_map[img['id']]
                val_annotations.append(new_ann)
                val_ann_id += 1
    
    # Update image IDs in train/val
    train_images_updated = []
    for img in train_images:
        new_img = img.copy()
        new_img['id'] = train_image_id_map[img['id']]
        train_images_updated.append(new_img)
    
    val_images_updated = []
    for img in val_images:
        new_img = img.copy()
        new_img['id'] = val_image_id_map[img['id']]
        val_images_updated.append(new_img)
    
    # Save train COCO file
    train_coco = {
        "info": coco_data['info'],
        "licenses": coco_data['licenses'],
        "images": train_images_updated,
        "annotations": train_annotations,
        "categories": coco_data['categories']
    }
    with open(train_dir / "_annotations.coco.json", 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    # Save val COCO file
    val_coco = {
        "info": coco_data['info'],
        "licenses": coco_data['licenses'],
        "images": val_images_updated,
        "annotations": val_annotations,
        "categories": coco_data['categories']
    }
    with open(val_dir / "_annotations.coco.json", 'w') as f:
        json.dump(val_coco, f, indent=2)
    
    print(f"✅ Train: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"✅ Val: {len(val_images)} images, {len(val_annotations)} annotations")
    
    return len(train_images), len(val_images)


def main():
    print("=" * 70)
    print("COMBINING DATASETS FOR BALL-ONLY TRAINING")
    print("=" * 70)
    
    # Paths
    open_soccer_ball_dir = Path("/workspace/soccer_cv_ball/data/raw/Open Soccer Ball Dataset")
    soccersynth_dir = Path("/workspace/datasets/soccersynth_sub_sub")
    
    # Step 1: Convert Open Soccer Ball Dataset to COCO (ball-only) if needed
    print("\n" + "=" * 70)
    print("STEP 1: Converting Open Soccer Ball Dataset to COCO (ball-only)")
    print("=" * 70)
    
    train_voc_dir = open_soccer_ball_dir / "training" / "training"
    train_annotations_dir = train_voc_dir / "annotations"
    train_images_dir = train_voc_dir / "images"
    train_coco_dir = open_soccer_ball_dir / "training" / "training_coco_ball_only"
    
    test_voc_dir = open_soccer_ball_dir / "test" / "ball"
    test_annotations_dir = test_voc_dir / "annotations"
    test_images_dir = test_voc_dir / "img"
    test_coco_dir = open_soccer_ball_dir / "test" / "ball_coco_ball_only"
    
    # Convert training split
    if not (train_coco_dir / "_annotations.coco.json").exists():
        if train_annotations_dir.exists() and train_images_dir.exists():
            print("\n📦 Converting training split...")
            convert_voc_to_coco_ball_only(
                voc_dir=train_voc_dir,
                annotations_dir=train_annotations_dir,
                images_dir=train_images_dir,
                output_dir=train_coco_dir,
                split_name="train",
                category_name="ball",
                category_id=0
            )
        else:
            raise FileNotFoundError(f"Training directory not found: {train_voc_dir}")
    else:
        print(f"✅ Training COCO already exists: {train_coco_dir}")
    
    # Convert test split
    if not (test_coco_dir / "_annotations.coco.json").exists():
        if test_annotations_dir.exists() and test_images_dir.exists():
            print("\n📦 Converting test split...")
            convert_voc_to_coco_ball_only(
                voc_dir=test_voc_dir,
                annotations_dir=test_annotations_dir,
                images_dir=test_images_dir,
                output_dir=test_coco_dir,
                split_name="test",
                category_name="ball",
                category_id=0
            )
        else:
            print(f"⚠️  Test directory not found, skipping")
    else:
        print(f"✅ Test COCO already exists: {test_coco_dir}")
    
    # Step 2: Merge datasets
    print("\n" + "=" * 70)
    print("STEP 2: Merging datasets")
    print("=" * 70)
    
    # Merge training datasets
    soccersynth_train = soccersynth_dir / "train"
    combined_train_temp = Path("/workspace/datasets/combined_ball_only/temp_train")
    
    if not (combined_train_temp / "_annotations.coco.json").exists():
        merge_coco_datasets(
            dataset1_path=train_coco_dir,
            dataset2_path=soccersynth_train,
            output_path=combined_train_temp,
            id_offset=10000
        )
    else:
        print(f"✅ Merged training dataset already exists: {combined_train_temp}")
    
    # Merge test/val datasets
    soccersynth_val = soccersynth_dir / "val"
    combined_val_temp = Path("/workspace/datasets/combined_ball_only/temp_val")
    
    if test_coco_dir.exists() and (test_coco_dir / "_annotations.coco.json").exists():
        if not (combined_val_temp / "_annotations.coco.json").exists():
            merge_coco_datasets(
                dataset1_path=test_coco_dir,
                dataset2_path=soccersynth_val,
                output_path=combined_val_temp,
                id_offset=10000
            )
        else:
            print(f"✅ Merged validation dataset already exists: {combined_val_temp}")
    else:
        # If no test split, just use soccersynth val
        print(f"⚠️  No test split from Open Soccer Ball, using only soccersynth val")
        combined_val_temp = soccersynth_val
    
    # Step 3: Create final train/val split
    print("\n" + "=" * 70)
    print("STEP 3: Creating final train/val split")
    print("=" * 70)
    
    # Combine all training data and split
    combined_all = Path("/workspace/datasets/combined_ball_only/all")
    
    if not (combined_all / "_annotations.coco.json").exists():
        # Merge temp train and temp val into one big dataset
        if combined_val_temp.exists() and (combined_val_temp / "_annotations.coco.json").exists():
            merge_coco_datasets(
                dataset1_path=combined_train_temp,
                dataset2_path=combined_val_temp,
                output_path=combined_all,
                id_offset=50000
            )
        else:
            # Just copy train if no val
            shutil.copytree(combined_train_temp, combined_all, dirs_exist_ok=True)
    else:
        print(f"✅ Combined dataset already exists: {combined_all}")
    
    # Split into final train/val
    final_train_dir = Path("/workspace/datasets/combined_ball_only/train")
    final_val_dir = Path("/workspace/datasets/combined_ball_only/val")
    
    if not (final_train_dir / "_annotations.coco.json").exists():
        train_count, val_count = split_dataset(
            source_dir=combined_all,
            train_dir=final_train_dir,
            val_dir=final_val_dir,
            split_ratio=0.8
        )
    else:
        print(f"✅ Final train/val split already exists")
        with open(final_train_dir / "_annotations.coco.json", 'r') as f:
            train_data = json.load(f)
        with open(final_val_dir / "_annotations.coco.json", 'r') as f:
            val_data = json.load(f)
        train_count = len(train_data['images'])
        val_count = len(val_data['images'])
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ DATASET COMBINATION COMPLETE")
    print("=" * 70)
    print(f"\n📁 Final dataset location:")
    print(f"   Train: {final_train_dir}")
    print(f"   Val:   {final_val_dir}")
    print(f"\n📊 Statistics:")
    print(f"   Train: {train_count} images")
    print(f"   Val:   {val_count} images")
    print(f"   Total: {train_count + val_count} images")
    print(f"\n✅ All annotations are ball-only (category_id: 0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
