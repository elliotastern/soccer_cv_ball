#!/usr/bin/env python3
"""
Sample every N frames from a video, run the best trained model to pre-label balls,
and save frames + COCO annotations to data/raw/prelabelled_balls for manual correction.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

# Project root (soccer_cv_ball)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from rfdetr import RFDETRBase
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False

DEFAULT_VIDEO = PROJECT_ROOT / "data/37CAE053-841F-4851-956E-CBF17A51C506.mp4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/raw/prelabelled_balls"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models/checkpoint_best_ema.pth"
SAMPLE_EVERY_N = 30
CONFIDENCE_THRESHOLD = 0.10
# Reject boxes that are too large to be a ball (pixels)
MAX_BALL_WIDTH = 200
MAX_BALL_HEIGHT = 200
MAX_BALL_AREA = 40000


def load_model_from_checkpoint(checkpoint_path: Path) -> RFDETRBase:
    """Load RF-DETR model from checkpoint (same logic as predict_video_frames.py)."""
    print(f"Loading model from: {checkpoint_path}")
    model = RFDETRBase(class_names=['ball'])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_state = checkpoint['model']
        if isinstance(model_state, dict):
            if hasattr(model, 'model') and hasattr(model.model, 'model'):
                current = model.model.model.state_dict()
                filtered = {k: v for k, v in model_state.items()
                            if k in current and current[k].shape == v.shape}
                model.model.model.load_state_dict(filtered, strict=False)
                skipped = len(model_state) - len(filtered)
                if skipped:
                    print(f"  Skipped {skipped} keys (shape mismatch)")
                print(f"  Loaded epoch {checkpoint.get('epoch', '?')}")
    return model


def extract_every_n_frames(video_path: Path, every_n: int) -> list:
    """Extract frames at 0, every_n, 2*every_n, ..."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_indices = list(range(0, total, every_n))
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        frames.append({
            'frame_number': i,
            'image': pil,
            'timestamp': i / fps if fps > 0 else 0,
            'width': pil.width,
            'height': pil.height,
        })
    cap.release()
    return frames


def predict_balls(model: RFDETRBase, pil_image: Image.Image, threshold: float) -> list:
    """Return list of (bbox [x,y,w,h], score) for ball class only."""
    try:
        detections = model.predict(pil_image, threshold=threshold)
    except Exception as e:
        return []
    boxes_scores = []
    if hasattr(detections, 'class_id'):
        n = len(detections.class_id)
        for i in range(n):
            if detections.class_id[i] != 0:
                continue
            xyxy = detections.xyxy[i]
            x, y = float(xyxy[0]), float(xyxy[1])
            w = float(xyxy[2] - xyxy[0])
            h = float(xyxy[3] - xyxy[1])
            if w > MAX_BALL_WIDTH or h > MAX_BALL_HEIGHT or w * h > MAX_BALL_AREA:
                continue
            score = float(detections.confidence[i])
            boxes_scores.append(([x, y, w, h], score))
    return boxes_scores


def main():
    parser = argparse.ArgumentParser(
        description="Sample every N frames, pre-label balls with best model, save to data/raw/prelabelled_balls"
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help="Input video path",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=SAMPLE_EVERY_N,
        help=f"Sample every N frames (default: {SAMPLE_EVERY_N})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for frames and COCO JSON",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Model checkpoint (default: best EMA)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Minimum confidence for ball detections",
    )
    args = parser.parse_args()

    if not RFDETR_AVAILABLE:
        print("Error: pip install rfdetr")
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    print("Extracting frames (every {} frames)...".format(args.every))
    frames = extract_every_n_frames(video_path, args.every)
    print(f"  Got {len(frames)} frames")

    print("Loading model...")
    model = load_model_from_checkpoint(checkpoint_path)

    # Build COCO structures
    coco_images = []
    coco_annotations = []
    ann_id = 1
    for idx, finfo in enumerate(frames):
        frame_num = finfo['frame_number']
        pil = finfo['image']
        w, h = finfo['width'], finfo['height']
        file_name = f"frame_{frame_num:06d}.jpg"
        image_id = idx + 1
        coco_images.append({
            "id": image_id,
            "width": w,
            "height": h,
            "file_name": file_name,
        })
        # Save image
        out_path = output_dir / file_name
        pil.save(out_path, quality=92)

        # Predict balls
        boxes_scores = predict_balls(model, pil, args.confidence)
        for bbox, score in boxes_scores:
            x, y, wb, hb = bbox
            coco_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [round(x, 2), round(y, 2), round(wb, 2), round(hb, 2)],
                "area": round(wb * hb, 2),
                "iscrowd": 0,
                "score": round(score, 4),
            })
            ann_id += 1

    categories = [{"id": 0, "name": "ball", "supercategory": "object"}]
    coco = {
        "info": {"description": "Pre-labelled balls (model predictions)", "version": "1.0"},
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    annotations_path = output_dir / "_annotations.coco.json"
    with open(annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    total_balls = len(coco_annotations)
    print(f"Done. Saved {len(frames)} images and {total_balls} ball annotations to {output_dir}")
    print(f"  COCO file: {annotations_path}")
    print("  Edit/correct labels in your labeling tool, then add to training data.")


if __name__ == "__main__":
    main()
