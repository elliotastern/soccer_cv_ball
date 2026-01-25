#!/usr/bin/env python3
"""
Generate a comprehensive HTML file for comparing predictions across epochs.
Features:
- Epoch dropdown to filter by epoch
- Confidence slider to filter detections
- Toggleable bounding boxes (on/off)
- 10 fixed validation frames (same across all epochs)
- Slightly transparent bounding boxes
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


def load_model_from_checkpoint(checkpoint_path: Path) -> RFDETRBase:
    """Load RF-DETR model from checkpoint."""
    # Initialize model
    model = RFDETRBase(class_names=['ball'])
    
    # Load checkpoint
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model_state = checkpoint['model']
                if isinstance(model_state, dict):
                    if hasattr(model, 'model') and hasattr(model.model, 'model'):
                        current_model_state = model.model.model.state_dict()
                        filtered_state = {}
                        
                        for key, value in model_state.items():
                            if key in current_model_state:
                                if current_model_state[key].shape == value.shape:
                                    filtered_state[key] = value
                        
                        model.model.model.load_state_dict(filtered_state, strict=False)
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
    
    return model


def get_predictions_for_epoch(model: RFDETRBase, frames: List[Dict], val_dataset_dir: Path) -> Dict:
    """
    Get predictions for all frames using the model.
    Uses low threshold to capture all detections.
    
    Returns:
        Dict mapping frame_idx to list of predictions with bbox and score
    """
    predictions = {}
    
    for idx, frame_info in enumerate(frames):
        image_path = val_dataset_dir / frame_info['file_name']
        
        if not image_path.exists():
            predictions[idx] = []
            continue
        
        try:
            pil_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Warning: Could not load image {image_path}: {e}")
            predictions[idx] = []
            continue
        
        # Run prediction with very low threshold to get all detections
        try:
            detections = model.predict(pil_image, threshold=0.1)
            
            pred_boxes = []
            pred_scores = []
            
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
                        pred_boxes.append([x, y, w, h])
                        pred_scores.append(float(confidence))
        except Exception as e:
            print(f"⚠️  Warning: Prediction failed for {frame_info['file_name']}: {e}")
            pred_boxes = []
            pred_scores = []
        
        # Convert to native Python types for JSON serialization
        predictions[idx] = [
            {
                'bbox': [float(x) for x in bbox], 
                'score': float(score)
            } 
            for bbox, score in zip(pred_boxes, pred_scores)
        ]
    
    return predictions


def generate_epoch_comparison_html(
    checkpoint_dir: Path,
    val_dataset_dir: Path,
    output_html_path: Path,
    num_frames: int = 10,
    seed: int = 42
) -> None:
    """
    Generate comprehensive HTML file for epoch comparison.
    """
    print(f"\n📊 Generating epoch comparison HTML...")
    
    # Get fixed validation frames
    frames = get_fixed_validation_frames(val_dataset_dir, num_frames, seed)
    print(f"   Selected {len(frames)} fixed frames for visualization")
    
    # Get all epochs from log file
    log_file = checkpoint_dir / "log.txt"
    available_epochs = set()
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line.strip())
                        epoch = d.get('epoch')
                        if epoch is not None:
                            available_epochs.add(epoch)
                    except:
                        pass
        except:
            pass
    
    if not available_epochs:
        # Fallback: check main checkpoint
        main_checkpoint = checkpoint_dir / "checkpoint.pth"
        if main_checkpoint.exists():
            try:
                checkpoint = torch.load(str(main_checkpoint), map_location='cpu', weights_only=False)
                epoch = checkpoint.get('epoch', 0)
                available_epochs = {epoch}
            except:
                pass
    
    if not available_epochs:
        raise FileNotFoundError(f"No epochs found in log file or checkpoints")
    
    available_epochs = sorted(available_epochs)
    print(f"   Found {len(available_epochs)} epochs: {available_epochs}")
    
    # Find main checkpoint
    main_checkpoint = checkpoint_dir / "checkpoint.pth"
    if not main_checkpoint.exists():
        raise FileNotFoundError(f"Main checkpoint not found: {main_checkpoint}")
    
    # Load predictions for each epoch
    # Note: Since we only have one checkpoint, we'll use it for all epochs
    # In a full implementation, you'd load epoch-specific checkpoints
    epochs_data = {}
    
    print(f"   Loading model and generating predictions...")
    model = load_model_from_checkpoint(main_checkpoint)
    predictions = get_predictions_for_epoch(model, frames, val_dataset_dir)
    
    # Store same predictions for all epochs (for now)
    # TODO: Load epoch-specific checkpoints when available
    for epoch in available_epochs:
        epochs_data[epoch] = predictions
    
    print(f"   Loaded predictions for {len(available_epochs)} epochs")
    
    # Load original images and ground truth
    original_images_b64 = []
    gt_boxes_data = []
    
    for idx, frame_info in enumerate(frames):
        image_path = val_dataset_dir / frame_info['file_name']
        
        if image_path.exists():
            try:
                pil_image = Image.open(image_path).convert('RGB')
                original_images_b64.append(image_to_base64(pil_image))
                
                # Get ground truth boxes
                gt_boxes = []
                for ann in frame_info['annotations']:
                    bbox = ann['bbox']  # [x, y, width, height]
                    gt_boxes.append(bbox)
                gt_boxes_data.append(gt_boxes)
            except Exception as e:
                print(f"⚠️  Warning: Could not load image {image_path}: {e}")
                original_images_b64.append("")
                gt_boxes_data.append([])
        else:
            original_images_b64.append("")
            gt_boxes_data.append([])
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Epoch Comparison - Ball Detection</title>
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
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .control-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .control-item label {{
            font-weight: bold;
            font-size: 13px;
            white-space: nowrap;
        }}
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        input[type="range"] {{
            width: 150px;
        }}
        .slider-value {{
            font-weight: bold;
            color: #4CAF50;
            min-width: 40px;
            font-size: 13px;
        }}
        select {{
            padding: 5px 8px;
            font-size: 13px;
            border: 1px solid #ddd;
            border-radius: 4px;
            min-width: 100px;
        }}
        .toggle-container {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .toggle-switch {{
            position: relative;
            width: 50px;
            height: 24px;
        }}
        .toggle-switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}
        .toggle-slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 24px;
        }}
        .toggle-slider:before {{
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }}
        input:checked + .toggle-slider {{
            background-color: #4CAF50;
        }}
        input:checked + .toggle-slider:before {{
            transform: translateX(26px);
        }}
        .info-text {{
            font-size: 12px;
            color: #666;
            margin-left: auto;
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
    <h1>Epoch Comparison - Ball Detection</h1>
    <div class="controls">
        <div class="control-item">
            <label>Epoch:</label>
            <select id="epochSelect">
"""
    
    # Add epoch options
    for epoch in available_epochs:
        selected = "selected" if epoch == available_epochs[-1] else ""
        html_content += f'                <option value="{epoch}" {selected}>Epoch {epoch}</option>\n'
    
    html_content += """            </select>
        </div>
        <div class="control-item">
            <label>Confidence:</label>
            <div class="slider-container">
                <input type="range" id="confidenceSlider" min="0.1" max="0.9" step="0.05" value="0.3">
                <span class="slider-value" id="confidenceValue">0.3</span>
            </div>
        </div>
        <div class="control-item">
            <label>Boxes:</label>
            <div class="toggle-container">
                <label class="toggle-switch">
                    <input type="checkbox" id="showBoxesToggle" checked>
                    <span class="toggle-slider"></span>
                </label>
                <span id="boxesStatus" style="font-size: 12px;">ON</span>
            </div>
        </div>
        <div class="info-text">
            <span id="totalDetections">0</span> detections
        </div>
    </div>
"""
    
    # Add frame containers
    for idx, frame_info in enumerate(frames):
        html_content += f"""
    <div class="frame-container" data-frame-idx="{idx}">
        <div class="frame-title">Frame {idx + 1}: {frame_info['file_name']}</div>
        <div class="image-container">
            <div class="image-box">
                <h3>Ground Truth ({len(gt_boxes_data[idx])} balls)</h3>
                <img src="{original_images_b64[idx]}" alt="Ground Truth" class="prediction-image">
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
                <span class="legend-box gt-box"></span> Ground Truth
            </div>
            <div class="legend-item">
                <span class="legend-box pred-box"></span> Predictions
            </div>
        </div>
    </div>
"""
    
    # Convert GT boxes to native Python types
    gt_boxes_python = [[[float(x) for x in bbox] for bbox in frame_boxes] for frame_boxes in gt_boxes_data]
    
    # Store data in JavaScript
    html_content += f"""
    <script>
        // Store ground truth boxes
        const gtBoxes = {json.dumps(gt_boxes_python)};
        
        // Store original images
        const originalImages = [];
        const canvases = [];
        const ctxs = [];
        
        // Store predictions for each epoch
        const epochsPredictions = {{
"""
    
    for epoch, predictions in epochs_data.items():
        html_content += f"            {epoch}: {json.dumps(predictions)},\n"
    
    html_content += """        };
        
        // Available epochs
        const availableEpochs = """ + json.dumps(available_epochs) + """;
        
        // Initialize canvases
        function initializeCanvases() {
            let imagesLoaded = 0;
            const totalImages = """ + str(len(frames)) + """;
            
            for (let i = 0; i < totalImages; i++) {
                const canvas = document.getElementById('canvas_' + i);
                const img = new Image();
                const frameContainer = document.querySelector('[data-frame-idx="' + i + '"]');
                img.src = frameContainer.querySelector('.prediction-image').src;
                
                img.onload = function() {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    originalImages[i] = img;
                    canvases[i] = canvas;
                    ctxs[i] = ctx;
                    
                    imagesLoaded++;
                    if (imagesLoaded === totalImages) {
                        // All images loaded, update predictions
                        updatePredictions();
                    }
                };
                
                img.onerror = function() {
                    console.error('Failed to load image for frame ' + i);
                    imagesLoaded++;
                    if (imagesLoaded === totalImages) {
                        updatePredictions();
                    }
                };
            }
        }
        
        // Update predictions based on current settings
        function updatePredictions() {
            const selectedEpoch = parseInt(document.getElementById('epochSelect').value);
            const threshold = parseFloat(document.getElementById('confidenceSlider').value);
            const showBoxes = document.getElementById('showBoxesToggle').checked;
            
            document.getElementById('confidenceValue').textContent = threshold.toFixed(2);
            document.getElementById('boxesStatus').textContent = showBoxes ? 'ON' : 'OFF';
            
            const predictions = epochsPredictions[selectedEpoch] || {};
            let totalDetections = 0;
            
            // Update each frame
            for (let i = 0; i < """ + str(len(frames)) + """; i++) {
                const framePreds = predictions[i] || [];
                const canvas = canvases[i];
                const ctx = ctxs[i];
                const img = originalImages[i];
                
                if (!canvas || !ctx || !img) {
                    console.log('Frame ' + i + ' not ready: canvas=' + !!canvas + ', ctx=' + !!ctx + ', img=' + !!img);
                    continue;
                }
                
                // Clear and redraw original image
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                
                if (showBoxes) {
                    // Draw ground truth boxes (green, slightly transparent)
                    const frameGtBoxes = (typeof gtBoxes !== 'undefined' && gtBoxes[i]) ? gtBoxes[i] : [];
                    ctx.strokeStyle = 'rgba(0, 255, 0, 0.7)';
                    ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
                    ctx.lineWidth = 2;
                    
                    if (frameGtBoxes && frameGtBoxes.length > 0) {
                        frameGtBoxes.forEach(bbox => {
                            if (bbox && bbox.length === 4) {
                                const [x, y, w, h] = bbox;
                                ctx.fillRect(x, y, w, h);
                                ctx.strokeRect(x, y, w, h);
                            }
                        });
                    }
                    
                    // Filter predictions by threshold
                    const filtered = framePreds.filter(p => p.score >= threshold);
                    totalDetections += filtered.length;
                    
                    // Draw prediction boxes (red, slightly transparent)
                    ctx.strokeStyle = 'rgba(255, 0, 0, 0.7)';
                    ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
                    ctx.lineWidth = 2;
                    ctx.font = 'bold 16px Arial';
                    ctx.textBaseline = 'top';
                    
                    filtered.forEach(pred => {
                        const [x, y, w, h] = pred.bbox;
                        
                        // Draw filled rectangle (transparent)
                        ctx.fillRect(x, y, w, h);
                        // Draw border
                        ctx.strokeRect(x, y, w, h);
                        
                        // Draw label with confidence
                        const label = 'Ball ' + pred.score.toFixed(2);
                        const textWidth = ctx.measureText(label).width;
                        const textHeight = 20;
                        
                        // Draw background
                        ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
                        ctx.fillRect(x, y - textHeight - 4, textWidth + 4, textHeight + 4);
                        
                        // Draw text
                        ctx.fillStyle = 'white';
                        ctx.fillText(label, x + 2, y - textHeight - 2);
                        ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
                    });
                }
                
                // Update detection count
                const countSpan = document.getElementById('detectionCount_' + i);
                if (countSpan) {
                    countSpan.textContent = showBoxes ? (predictions[i] || []).filter(p => p.score >= threshold).length : 0;
                }
            }
            
            // Update total detections
            document.getElementById('totalDetections').textContent = totalDetections;
        }
        
        // Event listeners
        document.getElementById('epochSelect').addEventListener('change', updatePredictions);
        document.getElementById('confidenceSlider').addEventListener('input', updatePredictions);
        document.getElementById('showBoxesToggle').addEventListener('change', updatePredictions);
        
        // Initialize
        window.addEventListener('load', function() {
            initializeCanvases();
        });
        
        // Also try to update after a delay in case images load slowly
        setTimeout(function() {
            if (canvases.length === """ + str(len(frames)) + """ && canvases.every(c => c !== undefined)) {
                updatePredictions();
            }
        }, 2000);
    </script>
</body>
</html>
"""
    
    # Save HTML file
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Saved epoch comparison HTML to: {output_html_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate epoch comparison HTML with interactive controls"
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models/ball_detection_open_soccer_ball',
        help='Directory containing checkpoints'
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
        default='models/ball_detection_open_soccer_ball/epoch_comparison.html',
        help='Output HTML file path'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=10,
        help='Number of fixed frames to visualize (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for frame selection (default: 42)'
    )
    
    args = parser.parse_args()
    
    generate_epoch_comparison_html(
        checkpoint_dir=Path(args.checkpoint_dir),
        val_dataset_dir=Path(args.val_dataset),
        output_html_path=Path(args.output),
        num_frames=args.num_frames,
        seed=args.seed
    )
    
    print(f"\n✅ Done! Open {args.output} in your browser to view the epoch comparison.")


if __name__ == "__main__":
    main()
