#!/usr/bin/env python3
"""
Convert Open Soccer Ball Dataset from Pascal VOC to COCO format (ball-only).
Count total images including soccersynth_sub_sub.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.voc_to_coco import convert_voc_to_coco_ball_only


def main():
    base_dir = Path("/workspace/soccer_cv_ball/data/raw/Open Soccer Ball Dataset")
    
    # Training split
    train_voc_dir = base_dir / "training" / "training"
    train_annotations_dir = train_voc_dir / "annotations"
    train_images_dir = train_voc_dir / "images"
    train_output_dir = base_dir / "training" / "training_coco_ball_only"
    
    # Test split
    test_voc_dir = base_dir / "test" / "ball"
    test_annotations_dir = test_voc_dir / "annotations"
    test_images_dir = test_voc_dir / "img"
    test_output_dir = base_dir / "test" / "ball_coco_ball_only"
    
    print("=" * 70)
    print("CONVERTING OPEN SOCCER BALL DATASET TO COCO (BALL-ONLY)")
    print("=" * 70)
    
    # Convert training
    if train_annotations_dir.exists() and train_images_dir.exists():
        print("\n📦 Converting training split...")
        train_images, train_annos = convert_voc_to_coco_ball_only(
            voc_dir=train_voc_dir,
            annotations_dir=train_annotations_dir,
            images_dir=train_images_dir,
            output_dir=train_output_dir,
            split_name="train",
            category_name="ball",
            category_id=0
        )
    else:
        print(f"⚠️  Training directory not found or incomplete")
        train_images, train_annos = 0, 0
    
    # Convert test
    if test_annotations_dir.exists() and test_images_dir.exists():
        print("\n📦 Converting test split...")
        test_images, test_annos = convert_voc_to_coco_ball_only(
            voc_dir=test_voc_dir,
            annotations_dir=test_annotations_dir,
            images_dir=test_images_dir,
            output_dir=test_output_dir,
            split_name="test",
            category_name="ball",
            category_id=0
        )
    else:
        print(f"⚠️  Test directory not found or incomplete")
        test_images, test_annos = 0, 0
    
    # Count soccersynth_sub_sub
    import json
    soccersynth_train = Path("/workspace/datasets/soccersynth_sub_sub/train/_annotations.coco.json")
    soccersynth_val = Path("/workspace/datasets/soccersynth_sub_sub/val/_annotations.coco.json")
    
    soccersynth_train_count = 0
    soccersynth_val_count = 0
    
    if soccersynth_train.exists():
        with open(soccersynth_train, 'r') as f:
            data = json.load(f)
            soccersynth_train_count = len(data['images'])
    
    if soccersynth_val.exists():
        with open(soccersynth_val, 'r') as f:
            data = json.load(f)
            soccersynth_val_count = len(data['images'])
    
    # Summary
    print("\n" + "=" * 70)
    print("TOTAL IMAGE COUNT SUMMARY")
    print("=" * 70)
    print(f"\n📦 Open Soccer Ball Dataset (ball-only):")
    print(f"   Train: {train_images:>5} images ({train_annos:>5} annotations)")
    print(f"   Test:  {test_images:>5} images ({test_annos:>5} annotations)")
    print(f"   Total: {train_images + test_images:>5} images")
    
    print(f"\n📦 soccersynth_sub_sub (ball-only):")
    print(f"   Train: {soccersynth_train_count:>5} images")
    print(f"   Val:   {soccersynth_val_count:>5} images")
    print(f"   Total: {soccersynth_train_count + soccersynth_val_count:>5} images")
    
    print(f"\n{'─' * 70}")
    print(f"🎯 COMBINED TOTAL:")
    print(f"   Total images: {train_images + test_images + soccersynth_train_count + soccersynth_val_count:>5}")
    print(f"   Total annotations: {train_annos + test_annos + soccersynth_train_count + soccersynth_val_count:>5}")
    print("=" * 70)
    
    print(f"\n✅ COCO annotations saved to:")
    print(f"   Train: {train_output_dir}")
    print(f"   Test:  {test_output_dir}")


if __name__ == "__main__":
    main()
