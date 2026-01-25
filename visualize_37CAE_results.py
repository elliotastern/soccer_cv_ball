#!/usr/bin/env python3
"""
Generate HTML visualization of model predictions on 37CAE video frames
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import base64
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent))

from src.perception.tracker import Tracker
from src.types import Detection
from scripts.evaluate_ball_model import load_model, detect_balls


def detect_with_trained_model_sahi(frame: np.ndarray, model=None) -> list:
    """Use trained ball detection model WITH SAHI"""
    if model is None:
        checkpoint_path = "models/ball_detection/checkpoint.pth"
        model = load_model(checkpoint_path)
    
    ball_detections = detect_balls(
        model, 
        frame, 
        confidence_threshold=0.3,
        use_sahi=True,
        sahi_slice_size=1288,
        sahi_overlap_ratio=0.2
    )
    
    detections = []
    for ball_det in ball_detections:
        detections.append(Detection(
            class_id=0,
            confidence=ball_det.confidence,
            bbox=ball_det.bbox,
            class_name='ball'
        ))
    
    return detections


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def draw_bbox_on_image(image: Image.Image, bbox: tuple, label: str, color: str, score: float = None) -> Image.Image:
    """Draw bounding box on PIL image"""
    from PIL import ImageDraw, ImageFont
    
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Color mapping
    colors = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0)
    }
    rgb_color = colors.get(color, (255, 0, 0))
    
    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=3)
    
    # Draw label
    label_text = f"{label}"
    if score is not None:
        label_text += f" {score:.2f}"
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Get text size
    bbox_text = draw.textbbox((0, 0), label_text, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]
    
    # Draw label background
    draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=rgb_color)
    draw.text((x1 + 2, y1 - text_height - 2), label_text, fill=(255, 255, 255), font=font)
    
    return img


def generate_html_visualization(video_path: str, num_frames: int = 20, output_path: str = None):
    """Generate HTML visualization of predictions"""
    print(f"📊 Generating HTML visualization for {num_frames} frames...")
    
    # Load model
    print("Loading model...")
    checkpoint_path = "models/ball_detection/checkpoint.pth"
    model = load_model(checkpoint_path)
    print("✅ Model loaded")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Select evenly spaced frames
    frame_indices = np.linspace(0, min(num_frames - 1, total_frames - 1), num_frames, dtype=int)
    
    frames_data = []
    
    print(f"Processing {len(frame_indices)} frames...")
    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Detect balls
        detections = detect_with_trained_model_sahi(frame, model=model)
        
        # Draw predictions
        img_with_preds = pil_image.copy()
        for det in detections:
            img_with_preds = draw_bbox_on_image(
                img_with_preds, 
                det.bbox, 
                "Ball", 
                "red", 
                det.confidence
            )
        
        # Convert to base64
        orig_b64 = image_to_base64(pil_image)
        pred_b64 = image_to_base64(img_with_preds)
        
        timestamp = frame_num / fps if fps > 0 else 0
        minutes = int(timestamp // 60)
        seconds = timestamp % 60
        
        frames_data.append({
            'frame_num': frame_num,
            'timestamp': f"{minutes}:{seconds:05.2f}",
            'original': orig_b64,
            'predictions': pred_b64,
            'num_detections': len(detections),
            'detections': [(d.bbox, d.confidence) for d in detections]
        })
        
        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(frame_indices)} frames...")
    
    cap.release()
    
    # Generate HTML
    if output_path is None:
        output_path = "models/ball_detection_open_soccer_ball/predictions_37CAE_visualization.html"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ball Detection Results - 37CAE053 Video</title>
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
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .frame-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
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
            margin-top: 15px;
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
            margin-right: 5px;
            vertical-align: middle;
            border: 2px solid;
        }}
        .pred-box {{
            background-color: rgba(255, 0, 0, 0.2);
            border-color: red;
        }}
    </style>
</head>
<body>
    <h1>⚽ Ball Detection Results - 37CAE053 Video</h1>
    <div class="info">
        <strong>Model:</strong> RF-DETR Base (20 epochs, SoccerSynth_sub) with SAHI<br>
        <strong>Video:</strong> 37CAE053-841F-4851-956E-CBF17A51C506.mp4<br>
        <strong>Frames Analyzed:</strong> {len(frames_data)} frames<br>
        <strong>Total Detections:</strong> {sum(f['num_detections'] for f in frames_data)} balls detected
    </div>
"""
    
    for idx, frame_data in enumerate(frames_data):
        html_content += f"""
    <div class="frame-container">
        <div class="frame-title">Frame {idx + 1}: Frame #{frame_data['frame_num']} (Time: {frame_data['timestamp']})</div>
        <div class="image-container">
            <div class="image-box">
                <h3>Original</h3>
                <img src="{frame_data['original']}" alt="Original Frame {idx + 1}" class="prediction-image">
            </div>
            <div class="image-box">
                <h3>Predictions ({frame_data['num_detections']} detected)</h3>
                <img src="{frame_data['predictions']}" alt="Predictions Frame {idx + 1}" class="prediction-image">
            </div>
        </div>
        <div class="legend">
            <div class="legend-item">
                <span class="legend-box pred-box"></span> Ball Detections
            </div>
            <div style="margin-top: 10px;">
                <strong>Detections:</strong> {frame_data['num_detections']} ball(s)
                {', '.join([f'Conf: {conf:.2f}' for _, conf in frame_data['detections']]) if frame_data['detections'] else 'None'}
            </div>
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Save HTML
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML visualization saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    video_path = "data/raw/real_data/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
    html_path = generate_html_visualization(video_path, num_frames=20)
    print(f"\n🌐 Open this file in your browser: {html_path}")
