#!/usr/bin/env python3
"""
Extract random frames from a video and run predictions on them.
Generates an HTML file with predictions visualization.
"""
import json
import random
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
import base64
from io import BytesIO
from typing import List, Dict
import sys
import cv2
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from rfdetr import RFDETRBase
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("Error: RF-DETR not installed. Install with: pip install rfdetr")
    sys.exit(1)


def extract_random_frames(video_path: Path, num_frames: int = 20, output_dir: Path = None) -> List[Dict]:
    """
    Extract random frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        output_dir: Optional directory to save extracted frames
    
    Returns:
        List of dicts with frame info: {'frame_number', 'image', 'timestamp'}
    """
    print(f"📹 Extracting {num_frames} random frames from video: {video_path}")
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Duration: {total_frames/fps:.2f} seconds")
    
    if total_frames < num_frames:
        print(f"⚠️  Warning: Video has only {total_frames} frames, using all of them")
        frame_numbers = list(range(total_frames))
    else:
        # Select random frame numbers
        frame_numbers = sorted(random.sample(range(total_frames), num_frames))
    
    extracted_frames = []
    
    for frame_num in frame_numbers:
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠️  Warning: Could not read frame {frame_num}")
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Calculate timestamp
        timestamp = frame_num / fps if fps > 0 else 0
        
        extracted_frames.append({
            'frame_number': frame_num,
            'image': pil_image,
            'timestamp': timestamp
        })
        
        # Optionally save frame
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            frame_path = output_dir / f"frame_{frame_num:06d}.jpg"
            pil_image.save(frame_path, quality=90)
    
    cap.release()
    print(f"✅ Extracted {len(extracted_frames)} frames")
    
    return extracted_frames


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
    frames: List[Dict],
    output_html_path: Path,
    video_name: str,
    confidence_threshold: float = 0.3,
    checkpoint_dir: Path = None,
    current_epoch: int = None
) -> None:
    """
    Generate HTML file with predictions on video frames with interactive confidence threshold.
    
    Args:
        model: RF-DETR model instance
        frames: List of frame dicts with 'frame_number', 'image', 'timestamp'
        output_html_path: Path to save HTML file
        video_name: Name of the video file
        confidence_threshold: Initial confidence threshold for predictions
    """
    print(f"\n📊 Generating predictions HTML on {len(frames)} frames...")
    print(f"   Running predictions with low threshold (0.1) to capture all detections...")
    
    # Process each frame - get ALL predictions with low threshold
    all_predictions = []
    for idx, frame_info in enumerate(frames):
        pil_image = frame_info['image']
        frame_num = frame_info['frame_number']
        timestamp = frame_info['timestamp']
        
        # Run prediction with very low threshold to get all detections
        try:
            detections = model.predict(pil_image, threshold=0.1)
            
            # Extract predictions
            pred_boxes = []
            pred_scores = []
            
            # RF-DETR returns detections with attributes
            if hasattr(detections, 'class_id'):
                num_detections = len(detections.class_id)
                for i in range(num_detections):
                    class_id = detections.class_id[i]
                    if class_id == 0:  # Ball class
                        confidence = detections.confidence[i]
                        xyxy = detections.xyxy[i]
                        x, y, w, h = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                        pred_boxes.append([x, y, w, h])
                        pred_scores.append(float(confidence))
            elif isinstance(detections, (list, tuple)):
                for det in detections:
                    if hasattr(det, 'class_id') and det.class_id == 0:
                        confidence = det.confidence if hasattr(det, 'confidence') else 0.5
                        xyxy = det.xyxy if hasattr(det, 'xyxy') else [0, 0, 0, 0]
                        x, y, w, h = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                        
                        # Filter out detections that are too large to be balls
                        area = w * h
                        if w <= MAX_BALL_WIDTH and h <= MAX_BALL_HEIGHT and area <= MAX_BALL_AREA:
                            pred_boxes.append([x, y, w, h])
                            pred_scores.append(float(confidence))
        except Exception as e:
            print(f"⚠️  Warning: Prediction failed for frame {frame_num}: {e}")
            pred_boxes = []
            pred_scores = []
        
        all_predictions.append({
            'frame_num': frame_num,
            'timestamp': timestamp,
            'boxes': pred_boxes,
            'scores': pred_scores
        })
    
    # Convert predictions to JavaScript format and get base64 images
    predictions_data = []
    original_images_b64 = []
    
    for idx, frame_info in enumerate(frames):
        pil_image = frame_info['image']
        frame_preds = all_predictions[idx]
        pred_boxes = frame_preds['boxes']
        pred_scores = frame_preds['scores']
        
        # Convert to format for JavaScript: [{'bbox': [x,y,w,h], 'score': score}, ...]
        js_predictions = [
            {'bbox': [float(x) for x in bbox], 'score': float(score)}
            for bbox, score in zip(pred_boxes, pred_scores)
        ]
        predictions_data.append(js_predictions)
        
        # Convert original image to base64
        img_orig_b64 = image_to_base64(pil_image)
        original_images_b64.append(img_orig_b64)
    
    # Process each frame
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ball Detection Predictions - {video_name}</title>
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
        .controls {{
            background-color: #e8f5e9;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .control-group {{
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .control-group label {{
            font-weight: bold;
            white-space: nowrap;
        }}
        select {{
            padding: 5px 8px;
            font-size: 14px;
            border: 1px solid #ddd;
            border-radius: 4px;
            min-width: 100px;
        }}
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        input[type="range"] {{
            flex: 1;
            max-width: 400px;
        }}
        .slider-value {{
            font-weight: bold;
            color: #4CAF50;
            min-width: 60px;
        }}
        .info {{
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 14px;
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
        .canvas-container {{
            position: relative;
            display: inline-block;
        }}
        .prediction-canvas {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: block;
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
        .pred-box {{ border-color: red; }}
    </style>
</head>
<body>
    <h1>Ball Detection Predictions - {video_name}</h1>
    <div class="controls">
        <div class="control-group">
            <label>Epoch:</label>
            <select id="epochSelect">
                <option value="{current_epoch if current_epoch is not None else 0}" selected>Epoch {current_epoch if current_epoch is not None else 0}</option>
            </select>
        </div>
        <div class="control-group">
            <label>Confidence Threshold:</label>
            <div class="slider-container">
                <input type="range" id="confidenceSlider" min="0.1" max="0.9" step="0.05" value="{confidence_threshold}">
                <span class="slider-value" id="confidenceValue">{confidence_threshold}</span>
            </div>
        </div>
        <div class="info">
            <strong>Video:</strong> {video_name} | 
            <strong>Frames:</strong> {len(frames)} random frames | 
            <strong>Total Detections:</strong> <span id="totalDetections">0</span>
        </div>
    </div>
"""
    
    for idx, frame_info in enumerate(frames):
        frame_num = frame_info['frame_number']
        timestamp = frame_info['timestamp']
        
        img_orig_b64 = original_images_b64[idx]
        
        # Format timestamp
        minutes = int(timestamp // 60)
        seconds = timestamp % 60
        time_str = f"{minutes}:{seconds:05.2f}"
        
        # Add to HTML with canvas
        html_content += f"""
    <div class="frame-container" data-frame-idx="{idx}">
        <div class="frame-title">Frame {idx + 1}: Frame #{frame_num} (Time: {time_str})</div>
        <div class="image-container">
            <div class="image-box">
                <h3>Original</h3>
                <img src="{img_orig_b64}" alt="Original" class="prediction-image">
            </div>
            <div class="image-box">
                <h3>Predictions (<span id="detectionCount_{idx}">0</span> detected)</h3>
                <div class="canvas-container">
                    <canvas id="canvas_{idx}" class="prediction-canvas"></canvas>
                </div>
            </div>
        </div>
        <div class="legend">
            <div class="legend-item">
                <span class="legend-box pred-box"></span> Predictions
            </div>
        </div>
    </div>
"""
    
    # Add JavaScript for interactive confidence threshold
    html_content += f"""
    <script>
        // Store predictions data by epoch
        const epochsPredictions = {{
            {current_epoch if current_epoch is not None else 0}: {json.dumps(predictions_data)}
        }};
        const defaultEpoch = {current_epoch if current_epoch is not None else 0};
        
        // Load original images
        const originalImages = [];
        const canvases = [];
        const ctxs = [];
        
        // Initialize canvases
        function initializeCanvases() {{
            let imagesLoaded = 0;
            const totalImages = {len(frames)};
            
            for (let i = 0; i < totalImages; i++) {{
                const canvas = document.getElementById('canvas_' + i);
                const img = new Image();
                const frameContainer = document.querySelector('[data-frame-idx="' + i + '"]');
                img.src = frameContainer.querySelector('.prediction-image').src;
                
                img.onload = function() {{
                    canvas.width = img.width;
                    canvas.height = img.height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    originalImages[i] = img;
                    canvases[i] = canvas;
                    ctxs[i] = ctx;
                    
                    imagesLoaded++;
                    if (imagesLoaded === totalImages) {{
                        // Small delay to ensure all canvases are ready, then update predictions
                        setTimeout(function() {{
                            updatePredictions();
                        }}, 200);
                    }}
                }};
                
                img.onerror = function() {{
                    console.error('Failed to load image for frame ' + i);
                    imagesLoaded++;
                    if (imagesLoaded === totalImages) {{
                        setTimeout(function() {{
                            updatePredictions();
                        }}, 200);
                    }}
                }};
            }}
        }}
        
        // Confidence slider
        const slider = document.getElementById('confidenceSlider');
        const confidenceValue = document.getElementById('confidenceValue');
        const totalDetectionsSpan = document.getElementById('totalDetections');
        
        function updatePredictions() {{
            const selectedEpoch = parseInt(document.getElementById('epochSelect').value) || 0;
            const threshold = parseFloat(slider.value);
            confidenceValue.textContent = threshold.toFixed(2);
            
            // Get predictions for selected epoch (epochsPredictions is an object mapping epoch -> frame_idx -> predictions)
            const epochPredictions = epochsPredictions[selectedEpoch] || epochsPredictions[defaultEpoch] || {{}};
            
            let totalDetections = 0;
            
            // Update each frame
            for (let i = 0; i < {len(frames)}; i++) {{
                const framePredictions = epochPredictions[i] || [];
                const canvas = canvases[i];
                const ctx = ctxs[i];
                const img = originalImages[i];
                
                if (!canvas || !ctx || !img) continue;
                
                // Clear and redraw original image
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                
                // Filter predictions by threshold
                const filtered = framePredictions.filter(p => p.score >= threshold);
                totalDetections += filtered.length;
                
                // Draw bounding boxes (slightly transparent)
                if (filtered.length > 0) {{
                    ctx.strokeStyle = 'rgba(255, 0, 0, 0.7)';
                    ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
                    ctx.lineWidth = 2;
                    ctx.font = 'bold 16px Arial';
                    ctx.textBaseline = 'top';
                    
                    filtered.forEach(pred => {{
                        const [x, y, w, h] = pred.bbox;
                        
                        // Validate coordinates
                        if (isNaN(x) || isNaN(y) || isNaN(w) || isNaN(h) || w <= 0 || h <= 0) {{
                            return;
                        }}
                        
                        // Draw filled rectangle (transparent)
                        ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
                        ctx.fillRect(x, y, w, h);
                        
                        // Draw border
                        ctx.strokeStyle = 'rgba(255, 0, 0, 0.7)';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, y, w, h);
                        
                        // Draw label with confidence
                        const label = 'Ball ' + pred.score.toFixed(2);
                        const textWidth = ctx.measureText(label).width;
                        const textHeight = 20;
                        
                        // Draw background for label
                        ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
                        ctx.fillRect(x, y - textHeight - 4, textWidth + 8, textHeight + 4);
                        
                        // Draw text
                        ctx.fillStyle = 'white';
                        ctx.fillText(label, x + 4, y - textHeight - 2);
                    }});
                }}
                
                // Update detection count
                const countSpan = document.getElementById('detectionCount_' + i);
                if (countSpan) {{
                    countSpan.textContent = filtered.length;
                }}
            }}
            
            // Update total detections
            totalDetectionsSpan.textContent = totalDetections;
        }}
        
        // Update on slider change
        slider.addEventListener('input', updatePredictions);
        
        // Update on epoch change
        document.getElementById('epochSelect').addEventListener('change', updatePredictions);
        
        // Initialize - ensure updatePredictions is called after images load
        if (document.readyState === 'complete') {{
            initializeCanvases();
        }} else {{
            window.addEventListener('load', function() {{
                initializeCanvases();
            }});
        }}
    </script>
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
        description="Extract random frames from video and run predictions"
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to video file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/ball_detection_open_soccer_ball/checkpoint.pth',
        help='Path to model checkpoint (default: models/ball_detection_open_soccer_ball/checkpoint.pth)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output HTML file path (default: auto-generated from video name)'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=20,
        help='Number of random frames to extract (default: 20)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.3,
        help='Confidence threshold for predictions (default: 0.3)'
    )
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save extracted frames to disk'
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    # Generate output path if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        video_name = video_path.stem
        output_path = Path(f"models/ball_detection_open_soccer_ball/predictions_{video_name}_20_frames.html")
    
    # Extract frames
    frames_dir = None
    if args.save_frames:
        frames_dir = output_path.parent / f"{video_path.stem}_frames"
    
    frames = extract_random_frames(video_path, args.num_frames, frames_dir)
    
    if not frames:
        print("❌ Error: No frames extracted from video")
        return
    
    # Load model and get epoch
    checkpoint_path = Path(args.checkpoint)
    model = load_model_from_checkpoint(checkpoint_path)
    
    # Get epoch from checkpoint
    current_epoch = None
    if checkpoint_path.exists():
        try:
            import torch
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            current_epoch = checkpoint.get('epoch', None)
        except:
            pass
    
    # Generate predictions HTML
    generate_predictions_html(
        model=model,
        frames=frames,
        output_html_path=output_path,
        video_name=video_path.name,
        confidence_threshold=args.confidence,
        current_epoch=current_epoch
    )
    
    print(f"\n✅ Done! Open {output_path} in your browser to view the predictions.")


if __name__ == "__main__":
    main()
