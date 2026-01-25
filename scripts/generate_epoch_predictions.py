"""
Generate predictions on 10 fixed validation frames after each epoch.
Creates an HTML file showing predictions for comparison across epochs.
"""
import json
import random
from pathlib import Path
from PIL import Image
import torch
import base64
from io import BytesIO
from typing import List, Dict, Tuple
import numpy as np


def get_fixed_validation_frames(val_dataset_dir: Path, num_frames: int = 10, seed: int = 42) -> List[Dict]:
    """
    Get a fixed set of validation frames for consistent evaluation across epochs.
    
    Args:
        val_dataset_dir: Path to validation dataset directory
        num_frames: Number of frames to select
        seed: Random seed for reproducibility
    
    Returns:
        List of dicts with image info: {'image_id', 'file_name', 'width', 'height', 'annotations'}
    """
    annotation_file = val_dataset_dir / "_annotations.coco.json"
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    images = {img['id']: img for img in coco_data['images']}
    annotations = {img_id: [] for img_id in images.keys()}
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in annotations:
            annotations[img_id].append(ann)
    
    # Get images with annotations (balls)
    images_with_balls = [img_id for img_id, anns in annotations.items() if len(anns) > 0]
    
    if len(images_with_balls) < num_frames:
        print(f"⚠️  Warning: Only {len(images_with_balls)} images with annotations, using all of them")
        selected_ids = images_with_balls
    else:
        # Use fixed seed for reproducibility
        random.seed(seed)
        selected_ids = random.sample(images_with_balls, num_frames)
        random.seed()  # Reset seed
    
    selected_frames = []
    for img_id in selected_ids:
        img_info = images[img_id]
        selected_frames.append({
            'image_id': img_id,
            'file_name': img_info['file_name'],
            'width': img_info['width'],
            'height': img_info['height'],
            'annotations': annotations[img_id]
        })
    
    return selected_frames


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for HTML embedding."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def draw_bbox_on_image(image: Image.Image, bbox: List[float], label: str, 
                       color: str = "red", confidence: float = None) -> Image.Image:
    """
    Draw bounding box on image.
    
    Args:
        image: PIL Image
        bbox: [x, y, width, height] in COCO format
        label: Label text
        color: Box color
        confidence: Optional confidence score
    
    Returns:
        PIL Image with drawn bbox
    """
    from PIL import ImageDraw, ImageFont
    
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Convert COCO format [x, y, w, h] to [x1, y1, x2, y2]
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    
    # Draw label
    label_text = label
    if confidence is not None:
        label_text += f" {confidence:.2f}"
    
    # Get text size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    bbox_text = draw.textbbox((0, 0), label_text, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]
    
    # Draw text background
    draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
    draw.text((x1 + 2, y1 - text_height - 2), label_text, fill="white", font=font)
    
    return img


def generate_predictions_html(
    model,
    val_dataset_dir: Path,
    output_html_path: Path,
    epoch: int,
    num_frames: int = 10,
    seed: int = 42,
    confidence_threshold: float = 0.3
) -> None:
    """
    Generate HTML file with predictions on fixed validation frames.
    
    Args:
        model: RF-DETR model instance
        val_dataset_dir: Path to validation dataset
        output_html_path: Path to save HTML file
        epoch: Current epoch number
        num_frames: Number of frames to visualize
        seed: Random seed for frame selection
        confidence_threshold: Confidence threshold for predictions
    """
    print(f"\n📊 Generating predictions HTML for epoch {epoch}...")
    
    # Get fixed validation frames
    frames = get_fixed_validation_frames(val_dataset_dir, num_frames, seed)
    print(f"   Selected {len(frames)} frames for visualization")
    
    # Process each frame
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Epoch {epoch} Predictions - Ball Detection</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .epoch-info {{
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .frame-container {{
            background-color: white;
            margin: 20px 0;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .frame-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .image-container {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .image-box {{
            flex: 1;
            min-width: 300px;
        }}
        .image-box h3 {{
            margin-top: 0;
            color: #666;
        }}
        .prediction-image {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .legend {{
            margin-top: 10px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        .legend-box {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid;
            margin-right: 5px;
            vertical-align: middle;
        }}
        .gt-box {{ border-color: green; }}
        .pred-box {{ border-color: red; }}
    </style>
</head>
<body>
    <h1>Epoch {epoch} Predictions - Ball Detection</h1>
    <div class="epoch-info">
        <strong>Epoch:</strong> {epoch}/20<br>
        <strong>Frames:</strong> {len(frames)} fixed validation frames<br>
        <strong>Confidence Threshold:</strong> {confidence_threshold}
    </div>
"""
    
    for idx, frame_info in enumerate(frames):
        image_path = val_dataset_dir / frame_info['file_name']
        
        if not image_path.exists():
            print(f"⚠️  Warning: Image not found: {image_path}")
            continue
        
        # Load image
        try:
            pil_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Warning: Could not load image {image_path}: {e}")
            continue
        
        # Get ground truth annotations
        gt_boxes = []
        for ann in frame_info['annotations']:
            bbox = ann['bbox']  # [x, y, width, height]
            gt_boxes.append(bbox)
        
        # Run prediction
        try:
            detections = model.predict(pil_image, threshold=confidence_threshold)
            
            # Extract predictions
            pred_boxes = []
            pred_scores = []
            
            # RF-DETR returns detections with attributes
            if hasattr(detections, 'class_id'):
                num_detections = len(detections.class_id)
                for i in range(num_detections):
                    # Check if it's a ball (class_id 0 for ball-only model)
                    class_id = detections.class_id[i]
                    if class_id == 0:  # Ball class
                        confidence = detections.confidence[i]
                        # RF-DETR returns xyxy format [x_min, y_min, x_max, y_max]
                        xyxy = detections.xyxy[i]
                        # Convert to COCO format [x, y, width, height]
                        x, y, w, h = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                        pred_boxes.append([x, y, w, h])
                        pred_scores.append(float(confidence))
            elif isinstance(detections, (list, tuple)):
                for det in detections:
                    if hasattr(det, 'class_id') and det.class_id == 0:
                        confidence = det.confidence if hasattr(det, 'confidence') else 0.5
                        xyxy = det.xyxy if hasattr(det, 'xyxy') else [0, 0, 0, 0]
                        x, y, w, h = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                        pred_boxes.append([x, y, w, h])
                        pred_scores.append(float(confidence))
        except Exception as e:
            print(f"⚠️  Warning: Prediction failed for {frame_info['file_name']}: {e}")
            pred_boxes = []
            pred_scores = []
        
        # Draw ground truth
        img_gt = pil_image.copy()
        for bbox in gt_boxes:
            img_gt = draw_bbox_on_image(img_gt, bbox, "GT", "green")
        
        # Draw predictions
        img_pred = pil_image.copy()
        for bbox, score in zip(pred_boxes, pred_scores):
            img_pred = draw_bbox_on_image(img_pred, bbox, "Ball", "red", score)
        
        # Convert to base64
        img_gt_b64 = image_to_base64(img_gt)
        img_pred_b64 = image_to_base64(img_pred)
        
        # Add to HTML
        html_content += f"""
    <div class="frame-container">
        <div class="frame-title">Frame {idx + 1}: {frame_info['file_name']}</div>
        <div class="image-container">
            <div class="image-box">
                <h3>Ground Truth ({len(gt_boxes)} balls)</h3>
                <img src="{img_gt_b64}" alt="Ground Truth" class="prediction-image">
            </div>
            <div class="image-box">
                <h3>Predictions ({len(pred_boxes)} detected)</h3>
                <img src="{img_pred_b64}" alt="Predictions" class="prediction-image">
            </div>
        </div>
        <div class="legend">
            <div class="legend-item">
                <span class="legend-box gt-box"></span> Ground Truth
            </div>
            <div class="legend-item">
                <span class="legend-box pred-box"></span> Predictions
            </div>
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Save HTML file
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Saved predictions HTML to: {output_html_path}")


def main():
    """Test function"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--val-dataset', type=str, required=True, help='Path to validation dataset')
    parser.add_argument('--output', type=str, required=True, help='Output HTML path')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number')
    parser.add_argument('--num-frames', type=int, default=10, help='Number of frames')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Load model
    from rfdetr import RFDETRBase
    # Initialize with ball class for ball-only detection
    model = RFDETRBase(class_names=['ball'])
    
    # Load checkpoint
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        
        # Try different checkpoint formats
        loaded = False
        if 'model' in checkpoint:
            try:
                model_state = checkpoint['model']
                if isinstance(model_state, dict):
                    model.load_state_dict(model_state, strict=False)
                    loaded = True
            except Exception as e:
                print(f"⚠️  Error loading from 'model' key: {e}")
        
        if not loaded and 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                loaded = True
            except Exception as e:
                print(f"⚠️  Error loading from 'model_state_dict': {e}")
        
        if not loaded and 'state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                loaded = True
            except Exception as e:
                print(f"⚠️  Error loading from 'state_dict': {e}")
        
        if loaded:
            print(f"✅ Loaded checkpoint: {args.checkpoint}")
        else:
            print(f"⚠️  Could not load checkpoint weights, using pretrained model")
    
    # Generate HTML
    generate_predictions_html(
        model=model,
        val_dataset_dir=Path(args.val_dataset),
        output_html_path=Path(args.output),
        epoch=args.epoch,
        num_frames=args.num_frames,
        confidence_threshold=args.confidence
    )


if __name__ == "__main__":
    main()
