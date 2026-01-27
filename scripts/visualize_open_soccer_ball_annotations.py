#!/usr/bin/env python3
"""
Visualize 10 random annotations from Open Soccer Ball Dataset to verify correctness.
"""

import xml.etree.ElementTree as ET
import random
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re

sys.path.append(str(Path(__file__).parent.parent))


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


def visualize_annotations(
    dataset_dir: Path,
    annotations_dir: Path,
    images_dir: Path,
    output_dir: Path,
    num_samples: int = 10
):
    """Visualize random annotations from the dataset."""
    print(f"\n🎨 Visualizing {num_samples} random annotations...")
    
    # Get all XML files with ball annotations
    xml_files = []
    for xml_file in annotations_dir.glob("*.xml"):
        try:
            annotation_data = parse_voc_xml(xml_file)
            if len(annotation_data['bboxes']) > 0:
                xml_files.append((xml_file, annotation_data))
        except Exception as e:
            print(f"⚠️  Skipping {xml_file.name}: {e}")
            continue
    
    if len(xml_files) == 0:
        raise ValueError("No XML files with ball annotations found!")
    
    # Random sample
    if len(xml_files) < num_samples:
        print(f"⚠️  Only {len(xml_files)} files with annotations, showing all")
        sampled = xml_files
    else:
        sampled = random.sample(xml_files, num_samples)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (xml_file, annotation_data) in enumerate(sampled):
        # Load image
        image_path = images_dir / annotation_data['filename']
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            continue
        
        try:
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
        except Exception as e:
            print(f"⚠️  Could not load image {image_path}: {e}")
            continue
        
        # Draw bounding boxes
        for bbox in annotation_data['bboxes']:
            xmin, ymin, xmax, ymax = bbox
            
            # Draw rectangle
            draw.rectangle(
                [xmin, ymin, xmax, ymax],
                outline="red",
                width=3
            )
            
            # Draw label
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Draw text background
            text = "ball"
            bbox_text = draw.textbbox((xmin, ymin - 25), text, font=font)
            draw.rectangle(bbox_text, fill="red")
            draw.text((xmin, ymin - 25), text, fill="white", font=font)
        
        # Save visualization
        output_path = output_dir / f"visualization_{idx+1:02d}_{annotation_data['filename']}"
        img.save(output_path)
        print(f"✅ Saved: {output_path} ({len(annotation_data['bboxes'])} ball(s))")
    
    print(f"\n✅ Created {len(sampled)} visualizations in {output_dir}")


def main():
    base_dir = Path("/workspace/soccer_cv_ball/data/raw/Open Soccer Ball Dataset")
    
    # Training split
    train_annotations_dir = base_dir / "training" / "training" / "annotations"
    train_images_dir = base_dir / "training" / "training" / "images"
    train_output_dir = base_dir / "visualizations" / "train"
    
    # Test split
    test_annotations_dir = base_dir / "test" / "ball" / "annotations"
    test_images_dir = base_dir / "test" / "ball" / "img"
    test_output_dir = base_dir / "visualizations" / "test"
    
    print("=" * 70)
    print("VISUALIZING OPEN SOCCER BALL DATASET ANNOTATIONS")
    print("=" * 70)
    
    # Visualize training annotations
    if train_annotations_dir.exists() and train_images_dir.exists():
        print("\n📦 Training split:")
        visualize_annotations(
            dataset_dir=base_dir / "training" / "training",
            annotations_dir=train_annotations_dir,
            images_dir=train_images_dir,
            output_dir=train_output_dir,
            num_samples=10
        )
    else:
        print(f"⚠️  Training directory not found")
    
    # Visualize test annotations (if available, show a few)
    if test_annotations_dir.exists() and test_images_dir.exists():
        print("\n📦 Test split:")
        visualize_annotations(
            dataset_dir=base_dir / "test" / "ball",
            annotations_dir=test_annotations_dir,
            images_dir=test_images_dir,
            output_dir=test_output_dir,
            num_samples=5  # Fewer from test set
        )
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"📁 Training visualizations: {train_output_dir}")
    if test_annotations_dir.exists():
        print(f"📁 Test visualizations: {test_output_dir}")


if __name__ == "__main__":
    main()
