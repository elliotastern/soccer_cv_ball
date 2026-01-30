# Next Steps - Soccer Ball Detection Model

## Current Status
- **Phase 4 Training**: Complete (Epoch 89/90)
- **Validation mAP**: 0.684 (68.4%)
- **Small Objects mAP**: 0.660 (66.0%)
- **20-Frame Test Results**: 4/20 correct (20% recall), 0 false positives (100% precision)
- **Issue**: Model is too conservative - high precision but low recall

## Priority Actions

### 1. **Analyze Confidence Score Distribution** (HIGH PRIORITY)
**Goal**: Understand why detections are being missed

**Action**: Create script to analyze confidence scores across all 20 test frames
```bash
python3 scripts/analyze_confidence_distribution.py \
    --video data/37CAE053-841F-4851-956E-CBF17A51C506.mp4 \
    --checkpoint models/checkpoint.pth \
    --num-frames 20 \
    --output analysis/confidence_analysis_epoch89.json
```

**What to check**:
- Average confidence for detected balls
- Average confidence for missed balls (if any detections below threshold)
- Confidence distribution histogram
- Correlation between ball size and confidence
- Frame characteristics (motion blur, lighting, occlusion)

### 2. **Test Multiple Confidence Thresholds** (HIGH PRIORITY)
**Goal**: Find optimal threshold balancing precision and recall

**Action**: Run prediction test with multiple thresholds
```bash
# Test thresholds: 0.1, 0.2, 0.3, 0.4, 0.5
for threshold in 0.1 0.2 0.3 0.4 0.5; do
    python3 scripts/predict_video_frames.py \
        --video data/37CAE053-841F-4851-956E-CBF17A51C506.mp4 \
        --checkpoint models/checkpoint.pth \
        --num-frames 20 \
        --confidence $threshold \
        --output models/ball_detection_open_soccer_ball/predictions_epoch89_thresh${threshold}.html
done
```

**Metrics to track**:
- Precision (true positives / all detections)
- Recall (true positives / actual balls)
- F1 score
- False positive rate

### 3. **Analyze Missed Frames** (MEDIUM PRIORITY)
**Goal**: Identify patterns in frames where balls are missed

**Action**: Create script to analyze missed detections
```bash
python3 scripts/analyze_missed_detections.py \
    --video data/37CAE053-841F-4851-956E-CBF17A51C506.mp4 \
    --checkpoint models/checkpoint.pth \
    --ground-truth data/ground_truth_20_frames.json \
    --output analysis/missed_detections_epoch89.html
```

**What to analyze**:
- Ball size distribution (missed vs detected)
- Ball position (center vs edge, field vs crowd)
- Motion blur presence
- Lighting conditions
- Occlusion patterns
- Distance from camera

### 4. **Compare with Previous Epochs** (MEDIUM PRIORITY)
**Goal**: Understand if Phase 4 training improved or hurt real-world performance

**Action**: Compare epoch 70, 77, and 89 on same 20 frames
```bash
for epoch in 70 77 89; do
    checkpoint="models/checkpoint00${epoch}.pth"
    python3 scripts/predict_video_frames.py \
        --video data/37CAE053-841F-4851-956E-CBF17A51C506.mp4 \
        --checkpoint $checkpoint \
        --num-frames 20 \
        --confidence 0.3 \
        --output models/ball_detection_open_soccer_ball/predictions_epoch${epoch}_comparison.html
done
```

### 5. **Full Validation Set Evaluation** (MEDIUM PRIORITY)
**Goal**: Verify model performance on validation set matches expectations

**Action**: Run comprehensive evaluation
```bash
python3 scripts/evaluate_ball_model.py \
    --checkpoint models/checkpoint.pth \
    --dataset models/ball_detection_combined_optimized/dataset \
    --output evaluation/epoch89_full_evaluation.json
```

**Metrics to extract**:
- mAP@0.50:0.95 (should be ~0.684)
- mAP@0.50 (should be ~0.974)
- Small/Medium/Large object mAP
- Precision-Recall curve
- Confusion matrix

### 6. **Fine-Tune on Video-Specific Data** (LOW PRIORITY - if needed)
**Goal**: Improve performance on specific video if domain gap exists

**Action**: If analysis shows video-specific issues, fine-tune on similar frames
```bash
# Extract frames with balls from video
python3 scripts/extract_ball_frames.py \
    --video data/37CAE053-841F-4851-956E-CBF17A51C506.mp4 \
    --annotations data/video_annotations.json \
    --output data/video_fine_tune_dataset

# Fine-tune model
python3 scripts/train_ball.py \
    --config configs/fine_tune_video.yaml \
    --resume-from models/checkpoint.pth \
    --output-dir models
```

### 7. **Optimize Post-Processing** (LOW PRIORITY)
**Goal**: Improve detection through better NMS and filtering

**Action**: Experiment with post-processing parameters
- Adjust NMS IoU threshold
- Implement size-based filtering
- Add temporal consistency (if processing video)
- Consider ensemble methods

## Immediate Next Steps (Do First)

1. **Create confidence analysis script** - Understand the confidence distribution
2. **Test lower thresholds** - See if 0.2-0.3 improves recall
3. **Compare epoch 70 vs 89** - Check if Phase 4 actually helped

## Expected Outcomes

### Best Case Scenario
- Lower threshold (0.2-0.3) improves recall to 60-80% while maintaining >90% precision
- Model is working well, just needs threshold tuning

### Worst Case Scenario
- Lower threshold introduces many false positives
- Model has domain gap with test video
- May need video-specific fine-tuning

### Most Likely Scenario
- Optimal threshold around 0.2-0.3
- Recall improves to 50-70%
- Some false positives but manageable
- Model is production-ready with proper threshold tuning

## Files to Create

1. `scripts/analyze_confidence_distribution.py` - Confidence analysis
2. `scripts/analyze_missed_detections.py` - Missed detection analysis
3. `scripts/evaluate_ball_model.py` - Full evaluation script
4. `configs/fine_tune_video.yaml` - Video-specific fine-tuning config (if needed)

## Notes

- Current threshold in `local_detector.py` is 0.05 (very low)
- Prediction script uses 0.5 (very high)
- Need to find optimal balance
- Previous test (earlier epoch) had 7/20 (35%) with some false positives
- Current test has 4/20 (20%) with 0 false positives
- Trade-off: Precision vs Recall
