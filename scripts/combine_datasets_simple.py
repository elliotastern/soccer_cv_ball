#!/usr/bin/env python3
"""
Combine Open Soccer Ball Dataset and soccersynth_sub_sub - simplified version.
Converts and merges in steps with progress tracking.
"""

import json
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from scripts.voc_to_coco import convert_voc_to_coco_ball_only


def main():
    print("=" * 70)
    print("COMBINING DATASETS FOR BALL-ONLY TRAINING")
    print("=" * 70)
    
    # Paths
    open_soccer_ball_dir = Path("/workspace/soccer_cv_ball/data/raw/Open Soccer Ball Dataset")
    soccersynth_train = Path("/workspace/datasets/soccersynth_sub_sub/train")
    soccersynth_val = Path("/workspace/datasets/soccersynth_sub_sub/val")
    
    # Step 1: Convert Open Soccer Ball Dataset (if needed)
    train_coco_dir = open_soccer_ball_dir / "training" / "training_coco_ball_only"
    test_coco_dir = open_soccer_ball_dir / "test" / "ball_coco_ball_only"
    
    if not (train_coco_dir / "_annotations.coco.json").exists():
        print("\n📦 Converting Open Soccer Ball training to COCO (this may take a while)...")
        train_voc_dir = open_soccer_ball_dir / "training" / "training"
        convert_voc_to_coco_ball_only(
            voc_dir=train_voc_dir,
            annotations_dir=train_voc_dir / "annotations",
            images_dir=train_voc_dir / "images",
            output_dir=train_coco_dir,
            split_name="train",
            category_name="ball",
            category_id=0
        )
    else:
        print(f"✅ Training COCO already exists")
    
    if not (test_coco_dir / "_annotations.coco.json").exists():
        print("\n📦 Converting Open Soccer Ball test to COCO...")
        test_voc_dir = open_soccer_ball_dir / "test" / "ball"
        if (test_voc_dir / "annotations").exists():
            convert_voc_to_coco_ball_only(
                voc_dir=test_voc_dir,
                annotations_dir=test_voc_dir / "annotations",
                images_dir=test_voc_dir / "img",
                output_dir=test_coco_dir,
                split_name="test",
                category_name="ball",
                category_id=0
            )
    else:
        print(f"✅ Test COCO already exists")
    
    # Step 2: Load all datasets
    print("\n📊 Loading datasets...")
    
    with open(train_coco_dir / "_annotations.coco.json", 'r') as f:
        open_train_data = json.load(f)
    
    with open(soccersynth_train / "_annotations.coco.json", 'r') as f:
        synth_train_data = json.load(f)
    
    # Load test/val
    if (test_coco_dir / "_annotations.coco.json").exists():
        with open(test_coco_dir / "_annotations.coco.json", 'r') as f:
            open_test_data = json.load(f)
    else:
        open_test_data = {"images": [], "annotations": []}
    
    with open(soccersynth_val / "_annotations.coco.json", 'r') as f:
        synth_val_data = json.load(f)
    
    print(f"Open Soccer Ball train: {len(open_train_data['images'])} images")
    print(f"Open Soccer Ball test:  {len(open_test_data['images'])} images")
    print(f"SoccerSynth train:      {len(synth_train_data['images'])} images")
    print(f"SoccerSynth val:        {len(synth_val_data['images'])} images")
    
    # Step 3: Merge all into one dataset
    print("\n🔄 Merging all datasets...")
    
    all_images = []
    all_annotations = []
    image_id = 1
    ann_id = 1
    
    # Add Open Soccer Ball train
    id_map_open_train = {}
    for img in open_train_data['images']:
        id_map_open_train[img['id']] = image_id
        new_img = img.copy()
        new_img['id'] = image_id
        all_images.append(new_img)
        image_id += 1
    
    for ann in open_train_data['annotations']:
        new_ann = ann.copy()
        new_ann['id'] = ann_id
        new_ann['image_id'] = id_map_open_train[ann['image_id']]
        all_annotations.append(new_ann)
        ann_id += 1
    
    # Add Open Soccer Ball test
    id_map_open_test = {}
    for img in open_test_data['images']:
        id_map_open_test[img['id']] = image_id
        new_img = img.copy()
        new_img['id'] = image_id
        all_images.append(new_img)
        image_id += 1
    
    for ann in open_test_data['annotations']:
        new_ann = ann.copy()
        new_ann['id'] = ann_id
        new_ann['image_id'] = id_map_open_test[ann['image_id']]
        all_annotations.append(new_ann)
        ann_id += 1
    
    # Add SoccerSynth train
    id_map_synth_train = {}
    for img in synth_train_data['images']:
        id_map_synth_train[img['id']] = image_id
        new_img = img.copy()
        new_img['id'] = image_id
        all_images.append(new_img)
        image_id += 1
    
    for ann in synth_train_data['annotations']:
        new_ann = ann.copy()
        new_ann['id'] = ann_id
        new_ann['image_id'] = id_map_synth_train[ann['image_id']]
        all_annotations.append(new_ann)
        ann_id += 1
    
    # Add SoccerSynth val
    id_map_synth_val = {}
    for img in synth_val_data['images']:
        id_map_synth_val[img['id']] = image_id
        new_img = img.copy()
        new_img['id'] = image_id
        all_images.append(new_img)
        image_id += 1
    
    for ann in synth_val_data['annotations']:
        new_ann = ann.copy()
        new_ann['id'] = ann_id
        new_ann['image_id'] = id_map_synth_val[ann['image_id']]
        all_annotations.append(new_ann)
        ann_id += 1
    
    print(f"✅ Merged: {len(all_images)} images, {len(all_annotations)} annotations")
    
    # Step 4: Create output directory and copy images
    output_dir = Path("/workspace/datasets/combined_ball_only")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Copying images to {output_dir}...")
    
    # Copy from Open Soccer Ball train
    for img in tqdm(open_train_data['images'], desc="Copying Open Soccer Ball train"):
        src = train_coco_dir / img['file_name']
        dst = output_dir / img['file_name']
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    
    # Copy from Open Soccer Ball test
    for img in tqdm(open_test_data['images'], desc="Copying Open Soccer Ball test"):
        src = test_coco_dir / img['file_name']
        dst = output_dir / img['file_name']
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    
    # Copy from SoccerSynth train
    for img in tqdm(synth_train_data['images'], desc="Copying SoccerSynth train"):
        src = soccersynth_train / img['file_name']
        dst = output_dir / img['file_name']
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    
    # Copy from SoccerSynth val
    for img in tqdm(synth_val_data['images'], desc="Copying SoccerSynth val"):
        src = soccersynth_val / img['file_name']
        dst = output_dir / img['file_name']
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    
    # Save merged COCO file
    merged_data = {
        "info": {
            "description": "Combined Open Soccer Ball Dataset + SoccerSynth Sub Sub - Ball only",
            "version": "1.0"
        },
        "licenses": [],
        "images": all_images,
        "annotations": all_annotations,
        "categories": [{"id": 0, "name": "ball", "supercategory": "object"}]
    }
    
    merged_file = output_dir / "_annotations.coco.json"
    with open(merged_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\n✅ Saved merged dataset to {merged_file}")
    print(f"   Total: {len(all_images)} images, {len(all_annotations)} annotations")
    print("\n" + "=" * 70)
    print("✅ DATASET COMBINATION COMPLETE")
    print("=" * 70)
    print(f"\n📁 Location: {output_dir}")
    print(f"📄 File: {merged_file}")
    print(f"\n💡 Next step: Split into train/val using prepare_soccersynth_sub_sub.py or manually")


if __name__ == "__main__":
    main()
