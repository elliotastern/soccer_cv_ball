#!/usr/bin/env python3
"""
Add COCO format annotations (ball-only) to the raw soccersynth_sub_sub directory.
Converts YOLO format to COCO format and organizes it in the raw data location.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import List, Tuple
from PIL import Image

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


def convert_to_coco_ball_only(
    yolo_dir: Path,
    output_dir: Path,
    ball_class_id: int = 1
) -> Tuple[int, int]:
    """
    Convert YOLO format to COCO format, filtering for ball-only annotations.
    
    Args:
        yolo_dir: Directory containing YOLO format images and annotations
        output_dir: Output directory for COCO format dataset
        ball_class_id: Class ID for ball in YOLO format (default: 1)
    
    Returns:
        Tuple of (num_images, num_annotations)
    """
    print(f"\n🔄 Converting to COCO format (ball-only)...")
    print(f"   Source: {yolo_dir}")
    print(f"   Output: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    coco_data = {
        "info": {
            "description": "SoccerSynth Sub Sub - Ball only",
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 0,
                "name": "ball",
                "supercategory": "object"
            }
        ]
    }
    
    image_files = sorted(yolo_dir.glob("*.png"))
    if not image_files:
        image_files = sorted(list(yolo_dir.glob("*.jpg")) + list(yolo_dir.glob("*.jpeg")))
    
    image_id = 1
    annotation_id = 1
    images_with_balls = 0
    total_ball_annotations = 0
    
    for img_file in image_files:
        txt_file = yolo_dir / f"{img_file.stem}.txt"
        
        if not txt_file.exists():
            continue
        
        img_width, img_height = get_image_size(img_file)
        
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
                    "category_id": 0,  # Ball category
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
            continue
        
        image_id += 1
    
    # Save COCO annotations file
    annotation_file = output_path / "_annotations.coco.json"
    with open(annotation_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ Converted {images_with_balls} images with ball annotations")
    print(f"✅ Created {total_ball_annotations} ball annotations")
    print(f"✅ Saved to {annotation_file}")
    
    return images_with_balls, total_ball_annotations


def main():
    raw_dir = Path("/workspace/soccer_cv_ball/data/raw/soccersynth_sub_sub")
    test_dir = raw_dir / "test"
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Create COCO annotations directory structure
    coco_dir = raw_dir / "coco_ball_only"
    coco_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("ADDING COCO ANNOTATIONS TO RAW DATA DIRECTORY")
    print("=" * 70)
    print(f"Source: {test_dir}")
    print(f"Output: {coco_dir}")
    
    # Convert YOLO to COCO (ball-only)
    num_images, num_annos = convert_to_coco_ball_only(
        yolo_dir=test_dir,
        output_dir=coco_dir,
        ball_class_id=1
    )
    
    if num_images == 0:
        raise ValueError("No ball annotations found!")
    
    print("\n" + "=" * 70)
    print("✅ COCO ANNOTATIONS ADDED SUCCESSFULLY")
    print("=" * 70)
    print(f"📁 Location: {coco_dir}")
    print(f"📄 Annotation file: {coco_dir / '_annotations.coco.json'}")
    print(f"📊 Images: {num_images}")
    print(f"📊 Annotations: {num_annos}")
    print(f"\n💡 Note: Images remain in {test_dir}")
    print(f"   COCO annotations reference images by filename")


if __name__ == "__main__":
    main()
