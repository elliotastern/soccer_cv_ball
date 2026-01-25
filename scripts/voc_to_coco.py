#!/usr/bin/env python3
"""
Convert Pascal VOC XML format to COCO format for ball-only detection.
Handles the Open Soccer Ball Dataset format.
"""
import xml.etree.ElementTree as ET
import json
import shutil
import re
from pathlib import Path
from typing import List, Tuple
from PIL import Image


def parse_voc_xml(xml_path: Path) -> dict:
    """Parse Pascal VOC XML annotation file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        # If parsing fails, try to fix common XML issues
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Fix common issues: malformed <path> tags
        content = re.sub(r'<path><path></path>', '<path>Unknown</path>', content)
        content = re.sub(r'<path></path>', '<path>Unknown</path>', content)
        # Fix any remaining nested path tags
        content = re.sub(r'<path>.*?<path>.*?</path>.*?</path>', '<path>Unknown</path>', content, flags=re.DOTALL)
        
        try:
            root = ET.fromstring(content)
        except Exception as parse_err:
            raise ValueError(f"Could not parse XML file: {parse_err}")
    
    filename = root.find('filename')
    if filename is None or filename.text is None:
        raise ValueError("No filename found in XML")
    filename = filename.text
    
    size = root.find('size')
    if size is None:
        raise ValueError("No size found in XML")
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name')
        if name is None or name.text is None:
            continue
        if name.text.lower() == 'ball':
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            bboxes.append([xmin, ymin, xmax, ymax])
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'bboxes': bboxes
    }


def voc_bbox_to_coco(xmin: int, ymin: int, xmax: int, ymax: int) -> List[float]:
    """Convert Pascal VOC bbox (xmin, ymin, xmax, ymax) to COCO format [x_min, y_min, width, height]."""
    x_min = float(xmin)
    y_min = float(ymin)
    width = float(xmax - xmin)
    height = float(ymax - ymin)
    return [x_min, y_min, width, height]


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions."""
    img = Image.open(image_path)
    return img.width, img.height


def convert_voc_to_coco_ball_only(
    voc_dir: Path,
    annotations_dir: Path,
    images_dir: Path,
    output_dir: Path,
    split_name: str,
    category_name: str = "ball",
    category_id: int = 0
) -> Tuple[int, int]:
    """
    Convert Pascal VOC format to COCO format, filtering for ball-only annotations.
    
    Args:
        voc_dir: Directory containing Pascal VOC XML annotations
        annotations_dir: Directory containing XML annotation files
        images_dir: Directory containing image files
        output_dir: Output directory for COCO format dataset
        split_name: Name of the split (train/valid/test)
        category_name: Category name for COCO format (e.g., "ball")
        category_id: Category ID for COCO format (0 for single class)
    
    Returns:
        Tuple of (num_images, num_annotations)
    """
    print(f"\n🔄 Converting {split_name} split from Pascal VOC to COCO format (ball-only)...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # RF-DETR expects images directly in the split directory and _annotations.coco.json
    images_output_dir = output_path
    images_output_dir.mkdir(exist_ok=True)
    
    coco_data = {
        "info": {
            "description": f"Open Soccer Ball Dataset {split_name} - Ball only",
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
    
    # Get all XML files
    xml_files = sorted(annotations_dir.glob("*.xml"))
    
    if not xml_files:
        raise ValueError(f"No XML annotation files found in {annotations_dir}")
    
    image_id = 1
    annotation_id = 1
    images_with_balls = 0
    total_ball_annotations = 0
    skipped = 0
    
    for xml_file in xml_files:
        try:
            annotation_data = parse_voc_xml(xml_file)
            
            if len(annotation_data['bboxes']) == 0:
                # Skip images without ball annotations
                skipped += 1
                continue
            
            # Find corresponding image
            image_path = images_dir / annotation_data['filename']
            
            if not image_path.exists():
                # Try alternative paths or skip
                skipped += 1
                continue
            
            # Verify image dimensions match XML
            img_width, img_height = get_image_size(image_path)
            if img_width != annotation_data['width'] or img_height != annotation_data['height']:
                print(f"⚠️  Warning: Image {annotation_data['filename']} dimensions mismatch (XML: {annotation_data['width']}x{annotation_data['height']}, actual: {img_width}x{img_height})")
                # Use actual image dimensions
                annotation_data['width'] = img_width
                annotation_data['height'] = img_height
            
            # Copy image to output directory
            dest_image = images_output_dir / annotation_data['filename']
            shutil.copy2(image_path, dest_image)
            
            image_info = {
                "id": image_id,
                "width": annotation_data['width'],
                "height": annotation_data['height'],
                "file_name": annotation_data['filename']
            }
            coco_data["images"].append(image_info)
            
            # Convert ball annotations to COCO format
            for bbox in annotation_data['bboxes']:
                xmin, ymin, xmax, ymax = bbox
                coco_bbox = voc_bbox_to_coco(xmin, ymin, xmax, ymax)
                
                # Skip if bbox is too small (likely annotation error)
                if coco_bbox[2] < 1.0 or coco_bbox[3] < 1.0:
                    continue
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
                total_ball_annotations += 1
            
            images_with_balls += 1
            image_id += 1
            
        except Exception as e:
            print(f"⚠️  Warning: Error processing {xml_file.name}: {e}")
            skipped += 1
            continue
    
    # Save COCO JSON
    annotation_file = output_path / "_annotations.coco.json"
    with open(annotation_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ Converted {images_with_balls} images with {total_ball_annotations} ball annotations")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} files (no balls or missing images)")
    print(f"✅ Saved to {annotation_file}")
    
    return images_with_balls, total_ball_annotations


def main():
    """Main function for standalone conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML to COCO format")
    parser.add_argument("--voc-dir", type=str, required=True, help="Directory containing VOC dataset")
    parser.add_argument("--annotations-dir", type=str, required=True, help="Directory containing XML files")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory containing image files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for COCO format")
    parser.add_argument("--split-name", type=str, default="train", help="Split name (train/val/test)")
    
    args = parser.parse_args()
    
    convert_voc_to_coco_ball_only(
        voc_dir=Path(args.voc_dir),
        annotations_dir=Path(args.annotations_dir),
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        split_name=args.split_name
    )


if __name__ == "__main__":
    main()
