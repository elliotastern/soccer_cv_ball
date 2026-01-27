#!/usr/bin/env python3
"""
Prepare soccersynth_sub_sub dataset for training:
1. Convert YOLO format to COCO format (ball-only)
2. Split into train/val (80/20)
3. Visualize 10 random frames with labels
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))


def yolo_to_coco_bbox(yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """Convert YOLO bbox to COCO bbox format [x_min, y_min, width, height]."""
    class_id, x_center, y_center, width, height = yolo_bbox
    
    x_center_abs = float(x_center) * img_width
    y_center_abs = float(y_center) * img_height
    width_abs = float(width) * img_width
    height_abs = float(height) * img_height
    
    x_min = x_center_abs - (width_abs / 2)
    y_min = y_center_abs - (height_abs / 2)
    
    return [x_min, y_min, width_abs, height_abs]


def load_yolo_annotation(txt_path: Path) -> List[List[float]]:
    """Load YOLO format annotation file."""
    annotations = []
    if not txt_path.exists():
        return annotations
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    annotations.append([float(p) for p in parts])
    
    return annotations


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions."""
    img = Image.open(image_path)
    return img.width, img.height


def convert_yolo_to_coco_ball_only(
    yolo_dir: Path,
    output_dir: Path,
    split_name: str,
    ball_class_id: int,
    category_name: str,
    category_id: int
) -> Tuple[int, int]:
    """
    Convert YOLO format to COCO format, filtering for ball-only annotations.
    
    Args:
        yolo_dir: Directory containing YOLO format images and annotations
        output_dir: Output directory for COCO format dataset
        split_name: Name of the split (train/valid/test)
        ball_class_id: Class ID for ball in YOLO format (typically 1)
        category_name: Category name for COCO format (e.g., "ball")
        category_id: Category ID for COCO format (0 for single class)
    
    Returns:
        Tuple of (num_images, num_annotations)
    """
    print(f"\n🔄 Converting {split_name} split (ball-only)...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # RF-DETR expects images directly in the split directory and _annotations.coco.json
    images_dir = output_path
    images_dir.mkdir(exist_ok=True)
    
    coco_data = {
        "info": {
            "description": f"SoccerSynth {split_name} dataset - Ball only",
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": category_id,
                "name": category_name,
                "supercategory": "object"
            }
        ]
    }
    
    image_files = sorted(yolo_dir.glob("*.png"))
    if not image_files:
        # Try other image formats
        image_files = sorted(list(yolo_dir.glob("*.jpg")) + list(yolo_dir.glob("*.jpeg")))
    
    image_id = 1
    annotation_id = 1
    images_with_balls = 0
    total_ball_annotations = 0
    
    for img_file in image_files:
        txt_file = yolo_dir / f"{img_file.stem}.txt"
        
        if not txt_file.exists():
            # Skip images without annotations
            continue
        
        img_width, img_height = get_image_size(img_file)
        
        # Copy image to output directory
        dest_image = images_dir / img_file.name
        shutil.copy2(img_file, dest_image)
        
        image_info = {
            "id": image_id,
            "width": img_width,
            "height": img_height,
            "file_name": img_file.name
        }
        coco_data["images"].append(image_info)
        
        # Load YOLO annotations and filter for ball only
        yolo_annotations = load_yolo_annotation(txt_file)
        has_ball = False
        
        for yolo_bbox in yolo_annotations:
            class_id = int(yolo_bbox[0])
            
            # Only keep ball annotations (class_id == ball_class_id)
            if class_id == ball_class_id:
                coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
                
                # Skip if bbox is too small (likely annotation error)
                if coco_bbox[2] < 1.0 or coco_bbox[3] < 1.0:
                    continue
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,  # Always 0 for single class
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
                total_ball_annotations += 1
                has_ball = True
        
        if has_ball:
            images_with_balls += 1
        else:
            # Remove image entry if it has no ball annotations
            coco_data["images"].pop()
            # Remove copied image if no ball annotations
            if dest_image.exists():
                dest_image.unlink()
            continue
        
        image_id += 1
    
    # Save COCO annotations file (RF-DETR expects _annotations.coco.json)
    annotation_file = output_path / "_annotations.coco.json"
    with open(annotation_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ Converted {images_with_balls} images with ball annotations")
    print(f"✅ Created {total_ball_annotations} ball annotations")
    print(f"✅ Saved to {annotation_file}")
    
    return images_with_balls, total_ball_annotations


def split_dataset(
    source_dir: Path,
    train_dir: Path,
    val_dir: Path,
    split_ratio: float = 0.8
) -> Tuple[int, int]:
    """
    Split COCO dataset into train/val splits.
    
    Args:
        source_dir: Source directory with images and _annotations.coco.json
        train_dir: Output directory for training split
        val_dir: Output directory for validation split
        split_ratio: Ratio for training split (default 0.8)
    
    Returns:
        Tuple of (train_images, val_images)
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


def visualize_labels(
    dataset_dir: Path,
    output_dir: Path,
    num_samples: int = 10
):
    """
    Visualize random frames with bounding box labels.
    
    Args:
        dataset_dir: Directory with images and _annotations.coco.json
        output_dir: Output directory for visualization images
        num_samples: Number of random frames to visualize
    """
    print(f"\n🎨 Visualizing {num_samples} random frames...")
    
    # Load COCO annotations
    coco_file = dataset_dir / "_annotations.coco.json"
    if not coco_file.exists():
        raise FileNotFoundError(f"COCO annotations not found: {coco_file}")
    
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get images with annotations
    images_with_anns = []
    for img in coco_data['images']:
        anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img['id']]
        if anns:
            images_with_anns.append((img, anns))
    
    # Random sample
    if len(images_with_anns) < num_samples:
        print(f"⚠️  Only {len(images_with_anns)} images with annotations, showing all")
        sampled = images_with_anns
    else:
        sampled = random.sample(images_with_anns, num_samples)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (img_info, anns) in enumerate(sampled):
        # Load image
        img_path = dataset_dir / img_info['file_name']
        if not img_path.exists():
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        
        # Draw bounding boxes
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # Draw rectangle
            draw.rectangle(
                [x, y, x + w, y + h],
                outline="red",
                width=3
            )
            
            # Draw label
            category_id = ann['category_id']
            category_name = next(
                (cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id),
                "ball"
            )
            
            # Try to use a font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw text background
            text = f"{category_name}"
            bbox_text = draw.textbbox((x, y - 20), text, font=font)
            draw.rectangle(bbox_text, fill="red")
            draw.text((x, y - 20), text, fill="white", font=font)
        
        # Save visualization
        output_path = output_dir / f"visualization_{idx+1:02d}_{img_info['file_name']}"
        img.save(output_path)
        print(f"✅ Saved: {output_path}")
    
    print(f"✅ Created {len(sampled)} visualizations in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare soccersynth_sub_sub dataset for training"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/workspace/soccer_cv_ball/data/raw/soccersynth_sub_sub/test",
        help="Source directory with YOLO format data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/datasets/soccersynth_sub_sub",
        help="Output directory for COCO format dataset"
    )
    parser.add_argument(
        "--ball-class-id",
        type=int,
        default=1,
        help="YOLO class ID for ball (default: 1)"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=10,
        help="Number of random frames to visualize (default: 10)"
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip YOLO to COCO conversion (use existing COCO data)"
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Step 1: Convert YOLO to COCO (ball-only)
    if not args.skip_conversion:
        print("=" * 70)
        print("STEP 1: Converting YOLO to COCO format (ball-only)")
        print("=" * 70)
        
        temp_dir = output_dir / "temp_coco"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        num_images, num_annos = convert_yolo_to_coco_ball_only(
            yolo_dir=source_dir,
            output_dir=temp_dir,
            split_name="all",
            ball_class_id=args.ball_class_id,
            category_name="ball",
            category_id=0  # Single class, use 0
        )
        
        if num_images == 0:
            raise ValueError("No ball annotations found in dataset!")
    else:
        temp_dir = source_dir
        print("⏭️  Skipping conversion (using existing COCO data)")
    
    # Step 2: Split into train/val
    print("\n" + "=" * 70)
    print("STEP 2: Splitting into train/val")
    print("=" * 70)
    
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    train_images, val_images = split_dataset(
        source_dir=temp_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        split_ratio=args.split_ratio
    )
    
    # Step 3: Visualize random frames
    print("\n" + "=" * 70)
    print("STEP 3: Visualizing random frames")
    print("=" * 70)
    
    vis_output_dir = output_dir / "visualizations"
    visualize_labels(
        dataset_dir=train_dir,
        output_dir=vis_output_dir,
        num_samples=args.num_visualizations
    )
    
    # Cleanup temp directory
    if not args.skip_conversion and temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")
    
    print("\n" + "=" * 70)
    print("✅ DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"📁 Train dataset: {train_dir}")
    print(f"📁 Val dataset: {val_dir}")
    print(f"📊 Train images: {train_images}")
    print(f"📊 Val images: {val_images}")
    print(f"🎨 Visualizations: {vis_output_dir}")


if __name__ == "__main__":
    main()
