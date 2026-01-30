#!/usr/bin/env python3
"""
Predict video frames with temporal smoothing.
Compares results with and without temporal smoothing.
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

from src.perception.temporal_smoothing import TemporalSmoother
from src.types import Detection


def defish_frame(frame, k=-0.32, alpha=0.0):
    """
    Undistorts fisheye image based on distortion coefficient 'k'.
    
    Args:
        frame: Input frame (BGR format from OpenCV)
        k: Distortion coefficient (negative for fisheye)
        alpha: 0=no black (cropped), 1=full frame with black edges
    
    Returns:
        Undistorted frame
    """
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
    D = np.array([k, 0, 0, 0, 0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


def extract_sequential_frames(video_path: Path, num_frames: int = 20, start_frame: int = None, frame_numbers: List[int] = None, fisheye_k: float = None, fisheye_alpha: float = 0.0) -> List[Dict]:
    """Extract sequential or specified frames from video."""
    print(f"📹 Extracting frames from video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    
    if frame_numbers is not None:
        # Use specified frame numbers
        frame_nums = sorted(frame_numbers)
    elif start_frame is not None:
        # Sequential from start_frame
        frame_nums = list(range(start_frame, min(start_frame + num_frames, total_frames)))
    else:
        # Random sequential
        max_start = max(0, total_frames - num_frames)
        start_frame = random.randint(0, max_start)
        frame_nums = list(range(start_frame, min(start_frame + num_frames, total_frames)))
    
    extracted_frames = []
    
    for frame_num in frame_nums:
        if frame_num >= total_frames:
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Apply fisheye correction if enabled
        if fisheye_k is not None:
            frame = defish_frame(frame, k=fisheye_k, alpha=fisheye_alpha)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        timestamp = frame_num / fps if fps > 0 else 0
        
        extracted_frames.append({
            'frame_number': frame_num,
            'image': pil_image,
            'timestamp': timestamp
        })
    
    cap.release()
    print(f"✅ Extracted {len(extracted_frames)} frames")
    return extracted_frames


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def load_model_from_checkpoint(checkpoint_path: Path) -> RFDETRBase:
    """Load RF-DETR model from checkpoint."""
    print(f"📦 Loading model from checkpoint: {checkpoint_path}")
    
    model = RFDETRBase(class_names=['ball'])
    
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model_state = checkpoint['model']
                if isinstance(model_state, dict):
                    if hasattr(model, 'model') and hasattr(model.model, 'model'):
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
                        
                        missing_keys, unexpected_keys = model.model.model.load_state_dict(filtered_state, strict=False)
                        if skipped_keys:
                            print(f"⚠️  Skipped {len(skipped_keys)} keys due to size mismatch")
                        if missing_keys:
                            print(f"⚠️  {len(missing_keys)} missing keys")
                        
                        epoch = checkpoint.get('epoch', 'N/A')
                        print(f"✅ Loaded checkpoint weights (epoch {epoch})")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
            print(f"   Using pretrained weights")
    else:
        print(f"⚠️  Checkpoint not found, using pretrained weights")
    
    return model


def detections_to_list(detections, confidence_threshold: float = 0.1) -> List[Detection]:
    """Convert RF-DETR detections to Detection objects."""
    result = []
    
    if hasattr(detections, 'class_id'):
        num_detections = len(detections.class_id)
        for i in range(num_detections):
            class_id = detections.class_id[i]
            if class_id == 0:  # Ball class
                confidence = float(detections.confidence[i])
                if confidence >= confidence_threshold:
                    xyxy = detections.xyxy[i]
                    x, y, w, h = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                    
                    result.append(Detection(
                        class_id=0,
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        class_name="ball"
                    ))
    elif isinstance(detections, (list, tuple)):
        for det in detections:
            if hasattr(det, 'class_id') and det.class_id == 0:
                confidence = det.confidence if hasattr(det, 'confidence') else 0.5
                if confidence >= confidence_threshold:
                    xyxy = det.xyxy if hasattr(det, 'xyxy') else [0, 0, 0, 0]
                    x, y, w, h = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                    
                    result.append(Detection(
                        class_id=0,
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        class_name="ball"
                    ))
    
    return result


def generate_comparison_html(
    frames: List[Dict],
    predictions_without_smoothing: List[Dict],
    predictions_with_smoothing: List[Dict],
    output_html_path: Path,
    video_name: str,
    confidence_threshold: float = 0.1
) -> None:
    """Generate HTML comparing predictions with and without temporal smoothing."""
    print(f"\n📊 Generating comparison HTML...")
    
    # Prepare data
    predictions_data = []
    images_b64 = []
    
    for idx, frame_info in enumerate(frames):
        pil_image = frame_info['image']
        frame_num = frame_info['frame_number']
        timestamp = frame_info['timestamp']
        
        # Get predictions
        pred_without = predictions_without_smoothing[idx]
        pred_with = predictions_with_smoothing[idx]
        
        # Convert to format for HTML
        boxes_without = []
        scores_without = []
        for det in pred_without.get('detections', []):
            x, y, w, h = det.bbox
            boxes_without.append([x, y, w, h])
            scores_without.append(det.confidence)
        
        boxes_with = []
        scores_with = []
        for det in pred_with.get('detections', []):
            x, y, w, h = det.bbox
            boxes_with.append([x, y, w, h])
            scores_with.append(det.confidence)
        
        predictions_data.append({
            'frame_num': frame_num,
            'timestamp': timestamp,
            'boxes_without': boxes_without,
            'scores_without': scores_without,
            'boxes_with': boxes_with,
            'scores_with': scores_with
        })
        
        images_b64.append(image_to_base64(pil_image))
    
    # Count detections
    total_without = sum(len(p['detections']) for p in predictions_without_smoothing)
    total_with = sum(len(p['detections']) for p in predictions_with_smoothing)
    frames_with_detections_without = sum(1 for p in predictions_without_smoothing if len(p['detections']) > 0)
    frames_with_detections_with = sum(1 for p in predictions_with_smoothing if len(p['detections']) > 0)
    
    print(f"   Without smoothing: {total_without} detections in {frames_with_detections_without}/{len(frames)} frames")
    print(f"   With smoothing: {total_with} detections in {frames_with_detections_with}/{len(frames)} frames")
    print(f"   Improvement: +{total_with - total_without} detections (+{frames_with_detections_with - frames_with_detections_without} frames)")
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Temporal Smoothing Comparison - {video_name}</title>
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
        .stats {{
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .stats h2 {{
            margin-top: 0;
            color: #4CAF50;
        }}
        .comparison {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .side {{
            flex: 1;
        }}
        .side h3 {{
            margin-top: 0;
            padding: 10px;
            border-radius: 5px;
        }}
        .without {{
            background-color: #ffebee;
        }}
        .without h3 {{
            background-color: #ef5350;
            color: white;
        }}
        .with {{
            background-color: #e8f5e9;
        }}
        .with h3 {{
            background-color: #4CAF50;
            color: white;
        }}
        .frame-container {{
            margin-bottom: 20px;
        }}
        .frame-info {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #666;
        }}
        img {{
            max-width: 100%;
            border: 2px solid #ddd;
            border-radius: 5px;
        }}
        .detection {{
            position: absolute;
            border: 2px solid;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
            padding: 2px 4px;
            background: rgba(255, 255, 255, 0.7);
        }}
        .detection.original {{
            border-color: #ef5350;
            color: #c62828;
        }}
        .detection.interpolated {{
            border-color: #4CAF50;
            color: #2e7d32;
        }}
        .image-container {{
            position: relative;
            display: inline-block;
            width: 100%;
        }}
        .image-container img {{
            display: block;
            width: 100%;
            height: auto;
        }}
        .detection-box {{
            position: absolute;
            border: 3px solid;
            border-radius: 3px;
            pointer-events: none;
            box-sizing: border-box;
        }}
        .detection-box.original {{
            border-color: #ef5350;
            background: rgba(239, 83, 80, 0.2);
        }}
        .detection-box.interpolated {{
            border-color: #4CAF50;
            background: rgba(76, 175, 80, 0.2);
        }}
        .detection-label {{
            position: absolute;
            top: -20px;
            left: 0;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 2px 6px;
            font-size: 11px;
            font-weight: bold;
            border-radius: 3px;
            white-space: nowrap;
        }}
        .detection-box.original .detection-label {{
            background: rgba(239, 83, 80, 0.9);
        }}
        .detection-box.interpolated .detection-label {{
            background: rgba(76, 175, 80, 0.9);
        }}
    </style>
</head>
<body>
    <h1>Temporal Smoothing Comparison</h1>
    <div class="stats">
        <h2>Results Summary</h2>
        <p><strong>Video:</strong> {video_name}</p>
        <p><strong>Frames analyzed:</strong> {len(frames)}</p>
        <p><strong>Confidence threshold:</strong> {confidence_threshold}</p>
        <hr>
        <h3>Without Temporal Smoothing:</h3>
        <ul>
            <li>Total detections: {total_without}</li>
            <li>Frames with detections: {frames_with_detections_without}/{len(frames)} ({frames_with_detections_without/len(frames)*100:.1f}%)</li>
        </ul>
        <h3>With Temporal Smoothing:</h3>
        <ul>
            <li>Total detections: {total_with}</li>
            <li>Frames with detections: {frames_with_detections_with}/{len(frames)} ({frames_with_detections_with/len(frames)*100:.1f}%)</li>
        </ul>
        <h3>Improvement:</h3>
        <ul>
            <li>+{total_with - total_without} detections ({((total_with - total_without) / max(total_without, 1)) * 100:.1f}% increase)</li>
            <li>+{frames_with_detections_with - frames_with_detections_without} frames ({((frames_with_detections_with - frames_with_detections_without) / max(frames_with_detections_without, 1)) * 100:.1f}% increase)</li>
        </ul>
    </div>
    
    <h2>Frame-by-Frame Comparison</h2>
"""
    
    for idx, (frame_info, pred_data) in enumerate(zip(frames, predictions_data)):
        frame_num = pred_data['frame_num']
        timestamp = pred_data['timestamp']
        img_b64 = images_b64[idx]
        
        # Create detection overlays
        boxes_without = pred_data['boxes_without']
        scores_without = pred_data['scores_without']
        boxes_with = pred_data['boxes_with']
        scores_with = pred_data['scores_with']
        
        # Get image dimensions for proper scaling
        pil_image = frame_info['image']
        img_width, img_height = pil_image.size
        
        # Generate bounding box overlays for without smoothing
        svg_overlay_without = ""
        for box, score in zip(boxes_without, scores_without):
            x, y, w, h = box
            # Ensure coordinates are valid and within image bounds
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = max(1, min(w, img_width - x))
            h = max(1, min(h, img_height - y))
            # Use percentage for responsive scaling
            x_pct = (x / img_width) * 100
            y_pct = (y / img_height) * 100
            w_pct = (w / img_width) * 100
            h_pct = (h / img_height) * 100
            svg_overlay_without += f'''
                    <div class="detection-box original" style="left: {x_pct}%; top: {y_pct}%; width: {w_pct}%; height: {h_pct}%;">
                        <span class="detection-label">{score:.2f}</span>
                    </div>'''
        
        # Generate bounding box overlays for with smoothing
        svg_overlay_with = ""
        for box, score in zip(boxes_with, scores_with):
            x, y, w, h = box
            # Ensure coordinates are valid and within image bounds
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = max(1, min(w, img_width - x))
            h = max(1, min(h, img_height - y))
            # Use percentage for responsive scaling
            x_pct = (x / img_width) * 100
            y_pct = (y / img_height) * 100
            w_pct = (w / img_width) * 100
            h_pct = (h / img_height) * 100
            svg_overlay_with += f'''
                    <div class="detection-box interpolated" style="left: {x_pct}%; top: {y_pct}%; width: {w_pct}%; height: {h_pct}%;">
                        <span class="detection-label">{score:.2f}</span>
                    </div>'''
        
        html_content += f"""
    <div class="frame-container">
        <div class="frame-info">Frame {frame_num} (t={timestamp:.2f}s)</div>
        <div class="comparison">
            <div class="side without">
                <h3>Without Smoothing ({len(boxes_without)} detections)</h3>
                <div class="image-container">
                    <img src="{img_b64}" alt="Frame {frame_num}">
                    {svg_overlay_without}
                </div>
            </div>
            <div class="side with">
                <h3>With Smoothing ({len(boxes_with)} detections)</h3>
                <div class="image-container">
                    <img src="{img_b64}" alt="Frame {frame_num}">
                    {svg_overlay_with}
                </div>
            </div>
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Saved comparison HTML to: {output_html_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict video frames with temporal smoothing")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-frames", type=int, default=20, help="Number of sequential frames to analyze")
    parser.add_argument("--confidence", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--output", type=str, help="Output HTML path")
    parser.add_argument("--start-frame", type=int, help="Starting frame number")
    parser.add_argument("--fisheye-k", type=float, default=None, help="Fisheye correction k value (default: disabled - model trained on fisheye images)")
    parser.add_argument("--fisheye-alpha", type=float, default=0.0, help="Fisheye correction alpha (0=cropped, 1=full, default 0)")
    parser.add_argument("--no-fisheye", action="store_true", help="Disable fisheye correction (default behavior)")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    checkpoint_path = Path(args.checkpoint)
    
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path)
    
    # Extract frames
    # NOTE: Model was trained on original fisheye images, so fisheye correction is disabled by default
    # Only enable if explicitly requested with --fisheye-k
    if args.no_fisheye:
        fisheye_k = None
    elif args.fisheye_k is not None:
        fisheye_k = args.fisheye_k
    else:
        # Default: no fisheye correction (model expects original fisheye images)
        fisheye_k = None
        print("ℹ️  Note: Fisheye correction disabled by default (model trained on original fisheye images)")
        print("   Use --fisheye-k <value> to enable fisheye correction")
    
    frames = extract_sequential_frames(video_path, args.num_frames, args.start_frame, fisheye_k=fisheye_k, fisheye_alpha=args.fisheye_alpha)
    
    # Initialize temporal smoother
    smoother = TemporalSmoother(
        min_track_length=2,
        max_gap_fill=5,
        isolation_threshold=1,
        velocity_threshold=100.0
    )
    
    # Process frames
    print(f"\n🔍 Processing frames with and without temporal smoothing...")
    
    predictions_without = []
    predictions_with = []
    
    for idx, frame_info in enumerate(frames):
        pil_image = frame_info['image']
        frame_num = frame_info['frame_number']
        
        # Get raw detections
        detections_raw = model.predict(pil_image, threshold=args.confidence)
        detections_list = detections_to_list(detections_raw, args.confidence)
        
        # Without smoothing
        predictions_without.append({
            'frame_number': frame_num,
            'detections': detections_list
        })
        
        # With smoothing
        smoothed_detections, track_info = smoother.update(detections_list, frame_num)
        predictions_with.append({
            'frame_number': frame_num,
            'detections': smoothed_detections,
            'track_info': track_info
        })
        
        if (idx + 1) % 5 == 0:
            print(f"   Processed {idx + 1}/{len(frames)} frames...")
    
    # Generate output path
    if args.output:
        output_path = Path(args.output)
    else:
        video_name = video_path.stem
        output_path = checkpoint_path.parent / f"predictions_{video_name}_temporal_smoothing.html"
    
    # Generate comparison HTML
    if fisheye_k is not None:
        fisheye_note = f" (fisheye k={fisheye_k:.3f}, alpha={args.fisheye_alpha:.2f})"
    else:
        fisheye_note = " (no fisheye correction)"
    generate_comparison_html(
        frames,
        predictions_without,
        predictions_with,
        output_path,
        video_path.name + fisheye_note,
        args.confidence
    )
    
    print(f"\n✅ Done! Open {output_path} in your browser to view the comparison.")


if __name__ == "__main__":
    main()
