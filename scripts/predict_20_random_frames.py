#!/usr/bin/env python3
"""
Run predictions on 20 random validation frames using the trained model.
Generates an HTML file with ground truth vs predictions comparison.
"""
import json
import random
import argparse
from pathlib import Path
from PIL import Image
import torch
import base64
from io import BytesIO
from typing import List, Dict
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from rfdetr import RFDETRBase
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("Error: RF-DETR not installed. Install with: pip install rfdetr")
    sys.exit(1)


def get_random_validation_frames(val_dataset_dir: Path, num_frames: int = 20) -> List[Dict]:
    """
    Get random validation frames with ball annotations.
    
    Args:
        val_dataset_dir: Path to validation dataset directory
        num_frames: Number of frames to select
    
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
        selected_ids = random.sample(images_with_balls, num_frames)
    
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


def load_model_from_checkpoint(checkpoint_path: Path) -> RFDETRBase:
    """Load RF-DETR model from checkpoint."""
    print(f"📦 Loading model from checkpoint: {checkpoint_path}")
    
    # Initialize model
    model = RFDETRBase(class_names=['ball'])
    
    # Load checkpoint
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            
            # RF-DETR checkpoint structure: {'model': state_dict, 'optimizer': ..., 'epoch': ...}
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model_state = checkpoint['model']
                if isinstance(model_state, dict):
                    # Load into the underlying PyTorch model: model.model.model
                    if hasattr(model, 'model') and hasattr(model.model, 'model'):
                        # Filter out class embedding layers that might have size mismatches
                        current_model_state = model.model.model.state_dict()
                        filtered_state = {}
                        skipped_keys = []
                        
                        for key, value in model_state.items():
                            if key in current_model_state:
                                if current_model_state[key].shape == value.shape:
                                    filtered_state[key] = value
                                else:
                                    skipped_keys.append(key)
                            else:
                                skipped_keys.append(key)
                        
                        # Load filtered state dict
                        missing_keys, unexpected_keys = model.model.model.load_state_dict(filtered_state, strict=False)
                        if skipped_keys:
                            print(f"⚠️  Skipped {len(skipped_keys)} keys due to size mismatch")
                        if missing_keys:
                            print(f"⚠️  {len(missing_keys)} missing keys")
                        
                        epoch = checkpoint.get('epoch', 'N/A')
                        print(f"✅ Loaded checkpoint weights (epoch {epoch})")
                    else:
                        print(f"⚠️  Could not find model.model.model to load weights")
                        print(f"   Using pretrained weights")
                else:
                    print(f"⚠️  Checkpoint['model'] is not a dict, using pretrained weights")
            else:
                print(f"⚠️  Checkpoint format not recognized, using pretrained weights")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
            print(f"   Using pretrained weights")
    else:
        print(f"⚠️  Checkpoint not found, using pretrained weights")
    
    return model


def generate_predictions_html(
    model: RFDETRBase,
    val_dataset_dir: Path,
    output_html_path: Path,
    num_frames: int = 20,
    confidence_threshold: float = 0.3
) -> None:
    """
    Generate HTML file with predictions on random validation frames.
    
    Args:
        model: RF-DETR model instance
        val_dataset_dir: Path to validation dataset
        output_html_path: Path to save HTML file
        num_frames: Number of frames to visualize
        confidence_threshold: Confidence threshold for predictions
    """
    print(f"\n📊 Generating predictions HTML on {num_frames} random frames...")
    
    # Get random validation frames
    frames = get_random_validation_frames(val_dataset_dir, num_frames)
    print(f"   Selected {len(frames)} frames for visualization")
    
    # Process each frame
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ball Detection Predictions - {num_frames} Random Frames</title>
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
        .info {{
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
    <h1>Ball Detection Predictions - {num_frames} Random Frames</h1>
    <div class="info">
        <strong>Frames:</strong> {len(frames)} random validation frames<br>
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
    parser = argparse.ArgumentParser(
        description="Run predictions on 20 random validation frames using trained model"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/ball_detection_open_soccer_ball/checkpoint.pth',
        help='Path to model checkpoint (default: models/ball_detection_open_soccer_ball/checkpoint.pth)'
    )
    parser.add_argument(
        '--val-dataset',
        type=str,
        default='data/raw/Open Soccer Ball Dataset/test/ball_coco',
        help='Path to validation dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/ball_detection_open_soccer_ball/predictions_20_random_frames.html',
        help='Output HTML file path'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=20,
        help='Number of random frames to predict on (default: 20)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.3,
        help='Confidence threshold for predictions (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_model_from_checkpoint(Path(args.checkpoint))
    
    # Generate predictions HTML
    generate_predictions_html(
        model=model,
        val_dataset_dir=Path(args.val_dataset),
        output_html_path=Path(args.output),
        num_frames=args.num_frames,
        confidence_threshold=args.confidence
    )
    
    print(f"\n✅ Done! Open {args.output} in your browser to view the predictions.")


if __name__ == "__main__":
    main()
