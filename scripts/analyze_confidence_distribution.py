#!/usr/bin/env python3
"""
Analyze confidence score distribution for ball detections.
Helps understand why detections are being missed.
"""
import json
import random
import argparse
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import sys
import cv2
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from rfdetr import RFDETRBase
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("Error: RF-DETR not installed. Install with: pip install rfdetr")
    sys.exit(1)


def extract_random_frames(video_path: Path, num_frames: int = 20) -> list:
    """Extract random frames from video."""
    print(f"📹 Extracting {num_frames} random frames from video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames < num_frames:
        frame_numbers = list(range(total_frames))
    else:
        frame_numbers = sorted(random.sample(range(total_frames), num_frames))
    
    extracted_frames = []
    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        
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


def load_model(checkpoint_path: Path, device='cuda'):
    """Load RF-DETR model from checkpoint (same as predict_video_frames.py)."""
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
                if isinstance(model_state, dict) and hasattr(model, 'model') and hasattr(model.model, 'model'):
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
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
    
    return model, device


def predict_with_model(model, pil_image, device, threshold=0.0):
    """Run prediction on image and return all detections above threshold."""
    # Use RFDETRBase predict method (same as predict_video_frames.py)
    detections = model.predict(pil_image, threshold=threshold)
    
    # Extract ball detections
    ball_scores = []
    if hasattr(detections, 'class_id'):
        num_detections = len(detections.class_id)
        for i in range(num_detections):
            if detections.class_id[i] == 0:  # Ball class (0-indexed in RFDETRBase)
                confidence = float(detections.confidence[i])
                if confidence >= threshold:
                    ball_scores.append(confidence)
    elif isinstance(detections, (list, tuple)):
        for det in detections:
            if hasattr(det, 'class_id') and det.class_id == 0:
                confidence = det.confidence if hasattr(det, 'confidence') else 0.5
                if confidence >= threshold:
                    ball_scores.append(float(confidence))
    
    return ball_scores


def analyze_confidence_distribution(
    model,
    device,
    frames: list,
    thresholds: list = [0.1, 0.2, 0.3, 0.4, 0.5]
) -> dict:
    """Analyze confidence distribution across frames."""
    print(f"\n📊 Analyzing confidence distribution on {len(frames)} frames...")
    
    all_confidences = []
    frame_analyses = []
    
    for idx, frame_info in enumerate(frames):
        pil_image = frame_info['image']
        frame_num = frame_info['frame_number']
        
        # Get all detections with very low threshold
        try:
            frame_confidences = predict_with_model(model, pil_image, device, threshold=0.05)
            if frame_confidences:
                all_confidences.extend(frame_confidences)
        except Exception as e:
            print(f"⚠️  Warning: Prediction failed for frame {frame_num}: {e}")
            frame_confidences = []
        
        # Count detections at each threshold
        threshold_counts = {}
        for thresh in thresholds:
            count = sum(1 for c in frame_confidences if c >= thresh)
            threshold_counts[thresh] = count
        
        frame_analyses.append({
            'frame_number': frame_num,
            'timestamp': frame_info['timestamp'],
            'num_detections': len(frame_confidences),
            'confidences': frame_confidences,
            'threshold_counts': threshold_counts,
            'max_confidence': max(frame_confidences) if frame_confidences else 0.0,
            'min_confidence': min(frame_confidences) if frame_confidences else 0.0,
            'avg_confidence': np.mean(frame_confidences) if frame_confidences else 0.0
        })
    
    # Overall statistics
    if all_confidences:
        overall_stats = {
            'total_detections': len(all_confidences),
            'mean_confidence': float(np.mean(all_confidences)),
            'median_confidence': float(np.median(all_confidences)),
            'std_confidence': float(np.std(all_confidences)),
            'min_confidence': float(np.min(all_confidences)),
            'max_confidence': float(np.max(all_confidences)),
            'percentiles': {
                'p10': float(np.percentile(all_confidences, 10)),
                'p25': float(np.percentile(all_confidences, 25)),
                'p50': float(np.percentile(all_confidences, 50)),
                'p75': float(np.percentile(all_confidences, 75)),
                'p90': float(np.percentile(all_confidences, 90)),
                'p95': float(np.percentile(all_confidences, 95)),
                'p99': float(np.percentile(all_confidences, 99))
            }
        }
    else:
        overall_stats = {'total_detections': 0}
    
    # Threshold analysis
    threshold_analysis = {}
    for thresh in thresholds:
        detections_at_thresh = sum(1 for c in all_confidences if c >= thresh)
        frames_with_detections = sum(1 for fa in frame_analyses if fa['threshold_counts'][thresh] > 0)
        threshold_analysis[thresh] = {
            'total_detections': detections_at_thresh,
            'frames_with_detections': frames_with_detections,
            'avg_detections_per_frame': detections_at_thresh / len(frames) if frames else 0
        }
    
    return {
        'overall_stats': overall_stats,
        'threshold_analysis': threshold_analysis,
        'frame_analyses': frame_analyses
    }


def print_analysis(analysis: dict):
    """Print analysis results in human-readable format."""
    print("\n" + "="*60)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Overall statistics
    stats = analysis['overall_stats']
    if stats['total_detections'] > 0:
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total detections: {stats['total_detections']}")
        print(f"  Mean confidence:  {stats['mean_confidence']:.4f}")
        print(f"  Median confidence: {stats['median_confidence']:.4f}")
        print(f"  Std deviation:     {stats['std_confidence']:.4f}")
        print(f"  Min confidence:    {stats['min_confidence']:.4f}")
        print(f"  Max confidence:    {stats['max_confidence']:.4f}")
        
        print(f"\n📈 PERCENTILES:")
        p = stats['percentiles']
        print(f"  P10: {p['p10']:.4f}  P25: {p['p25']:.4f}  P50: {p['p50']:.4f}")
        print(f"  P75: {p['p75']:.4f}  P90: {p['p90']:.4f}  P95: {p['p95']:.4f}  P99: {p['p99']:.4f}")
    else:
        print("\n⚠️  No detections found!")
    
    # Threshold analysis
    print(f"\n🎯 THRESHOLD ANALYSIS:")
    print(f"  {'Threshold':<12} {'Detections':<12} {'Frames w/ Det':<15} {'Avg/Frame':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*15} {'-'*12}")
    for thresh in sorted(analysis['threshold_analysis'].keys()):
        ta = analysis['threshold_analysis'][thresh]
        print(f"  {thresh:<12.2f} {ta['total_detections']:<12} {ta['frames_with_detections']:<15} {ta['avg_detections_per_frame']:<12.2f}")
    
    # Frame-by-frame summary
    print(f"\n📋 FRAME-BY-FRAME SUMMARY:")
    print(f"  {'Frame':<8} {'Detections':<12} {'Max Conf':<12} {'Avg Conf':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for fa in analysis['frame_analyses'][:10]:  # Show first 10
        print(f"  {fa['frame_number']:<8} {fa['num_detections']:<12} {fa['max_confidence']:<12.4f} {fa['avg_confidence']:<12.4f}")
    if len(analysis['frame_analyses']) > 10:
        print(f"  ... ({len(analysis['frame_analyses']) - 10} more frames)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze confidence score distribution for ball detections"
    )
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoint.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--num-frames', type=int, default=20,
                       help='Number of random frames to analyze')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    parser.add_argument('--thresholds', type=float, nargs='+',
                       default=[0.1, 0.2, 0.3, 0.4, 0.5],
                       help='Thresholds to analyze')
    
    args = parser.parse_args()
    
    # Extract frames
    video_path = Path(args.video)
    frames = extract_random_frames(video_path, args.num_frames)
    
    if not frames:
        print("❌ Error: No frames extracted from video")
        return
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, device = load_model(checkpoint_path, device)
    
    # Analyze
    analysis = analyze_confidence_distribution(model, device, frames, args.thresholds)
    
    # Print results
    print_analysis(analysis)
    
    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n✅ Saved analysis to: {output_path}")


if __name__ == "__main__":
    main()
