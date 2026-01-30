# Strategies to Improve Ball Detection Recall

## Current Performance (Epoch 89)
- **Recall**: 45% (9/20 frames)
- **Precision**: 100% (9/9 correct, 0 false positives)
- **Threshold**: 0.10 (optimal)

## Goal: Increase Recall from 45% to 70%+

---

## 1. **Data Augmentation & Training Improvements**

### A. More Diverse Training Data
- **Add more real-world videos** with varying:
  - Lighting conditions (sunny, cloudy, night games)
  - Camera angles (high, low, side views)
  - Ball sizes (close-up vs far away)
  - Field conditions (wet, dry, different grass types)
- **Target**: 2-3x more real-world training data

### B. Hard Negative Mining
- **Focus on missed detections**: Collect frames where ball is visible but not detected
- **Add to training set**: Manually annotate these frames
- **Expected improvement**: +10-15% recall

### C. Synthetic Data Refinement
- **Improve synthetic-to-real gap**: 
  - Better texture matching
  - More realistic lighting/shadows
  - Varied ball trajectories
- **Domain adaptation techniques**: Use adversarial training to bridge synthetic-real gap

### D. Multi-Scale Training (Already doing, but can enhance)
- **Current**: Training at 1280x720
- **Enhancement**: 
  - Add more extreme scales (very small balls, very large balls)
  - Test-ball-specific augmentation (rotation, blur, motion blur)

---

## 2. **Model Architecture Improvements**

### A. Test Different Backbones
- **Current**: ResNet-50
- **Try**:
  - ResNet-101 (deeper, better features)
  - EfficientNet (better efficiency/accuracy tradeoff)
  - Swin Transformer (vision transformer, better for small objects)

### B. Feature Pyramid Network (FPN) Enhancement
- **Current**: Basic FPN
- **Enhancement**: 
  - Add more pyramid levels for tiny objects
  - Use PANet (Path Aggregation Network) for better feature fusion

### C. Attention Mechanisms
- **Add spatial attention**: Focus model on ball-like regions
- **Add channel attention**: Emphasize important feature channels
- **Expected improvement**: +5-10% recall for small objects

---

## 3. **Training Strategy Improvements**

### A. Focal Loss Tuning
- **Current**: Using focal loss
- **Enhancement**:
  - Tune alpha and gamma parameters specifically for ball class
  - Use class-balanced focal loss (ball is minority class)
- **Expected improvement**: +5-8% recall

### B. Learning Rate Schedule
- **Current**: Cosine annealing
- **Try**:
  - Warm restarts (SGDR)
  - One-cycle policy
  - Adaptive learning rate based on validation mAP

### C. Longer Training
- **Current**: 90 epochs
- **Try**: 120-150 epochs with early stopping
- **Monitor**: Validation mAP should continue improving

### D. Ensemble Methods
- **Train multiple models** with different:
  - Initializations
  - Data augmentations
  - Architectures
- **Combine predictions**: Average or vote on detections
- **Expected improvement**: +5-10% recall

---

## 4. **Inference Improvements**

### A. Test Lower Thresholds (Carefully)
- **Current**: 0.10 (optimal for precision/recall balance)
- **Test**: 0.05, 0.08 (may increase false positives)
- **Monitor**: Precision should stay >90%

### B. Temporal Smoothing
- **Use video context**: Ball position in previous frames
- **Filter detections**: Remove isolated detections (likely false positives)
- **Interpolate**: Fill gaps between detections
- **Expected improvement**: +10-15% recall (by filling gaps)

### C. Multi-Scale Inference (SAHI)
- **Current**: Single-scale inference
- **Enhancement**: 
  - Use SAHI (Slicing Aided Hyper Inference)
  - Test-ball-specific slice sizes (smaller slices for tiny balls)
  - Overlap ratio tuning
- **Expected improvement**: +15-20% recall for small/distant balls

### D. Test-Time Augmentation (TTA)
- **Augment at inference**:
  - Multiple scales
  - Slight rotations
  - Brightness variations
- **Combine predictions**: Average or max across augmentations
- **Expected improvement**: +5-8% recall

---

## 5. **Post-Processing Improvements**

### A. Size Filtering
- **Current**: Basic size filtering
- **Enhancement**:
  - More sophisticated size priors (ball size varies with distance)
  - Use camera calibration to estimate expected ball size
  - Remove detections that are too large/small for context

### B. Motion-Based Filtering
- **Use optical flow**: Balls move differently than players/background
- **Track consistency**: Detections should follow smooth trajectories
- **Remove static detections**: Balls rarely stay in one place

### C. Context-Aware Filtering
- **Field constraints**: Ball should be on or near field
- **Player proximity**: Balls often near players
- **Game state**: Use game context (throw-ins, corners, etc.)

---

## 6. **Active Learning & Data Collection**

### A. Identify Failure Cases
- **Analyze missed detections**: What patterns?
  - Small balls? (distant)
  - Occluded balls? (behind players)
  - Blurry balls? (motion blur)
  - Unusual lighting? (shadows, reflections)
- **Collect more data** for these specific cases

### B. Iterative Training
- **Phase 5 Training**:
  1. Run inference on validation set
  2. Identify all missed detections
  3. Manually annotate these frames
  4. Add to training set
  5. Retrain model
- **Expected improvement**: +10-20% recall per iteration

### C. Semi-Supervised Learning
- **Use unlabeled data**: 
  - Run model on unlabeled videos
  - High-confidence detections → pseudo-labels
  - Add to training set
- **Expected improvement**: +5-10% recall

---

## 7. **Quick Wins (Easy to Implement)**

### Priority 1: Temporal Smoothing ⭐⭐⭐
- **Effort**: Low
- **Impact**: High (+10-15% recall)
- **Time**: 1-2 days
- **Implementation**: Track detections across frames, interpolate gaps

### Priority 2: SAHI Multi-Scale Inference ⭐⭐⭐
- **Effort**: Medium
- **Impact**: High (+15-20% recall for small balls)
- **Time**: 2-3 days
- **Implementation**: Integrate SAHI library, tune slice sizes

### Priority 3: Hard Negative Mining ⭐⭐
- **Effort**: Medium
- **Impact**: Medium (+10-15% recall)
- **Time**: 3-5 days
- **Implementation**: Identify missed detections, annotate, retrain

### Priority 4: Lower Threshold Testing ⭐
- **Effort**: Low
- **Impact**: Medium (+5-10% recall, may reduce precision)
- **Time**: 1 day
- **Implementation**: Test thresholds 0.05, 0.08, monitor precision

---

## Recommended Action Plan

### Phase 5A: Quick Wins (1-2 weeks)
1. ✅ **Update default threshold to 0.10** (DONE)
2. **Implement temporal smoothing** (tracking + interpolation)
3. **Test SAHI multi-scale inference**
4. **Test lower thresholds** (0.05, 0.08) with precision monitoring

### Phase 5B: Data Collection (2-3 weeks)
1. **Hard negative mining**: Collect and annotate missed detections
2. **Diverse data collection**: More videos with varying conditions
3. **Synthetic data refinement**: Better domain adaptation

### Phase 5C: Model Improvements (3-4 weeks)
1. **Architecture experiments**: Test different backbones
2. **Training improvements**: Focal loss tuning, longer training
3. **Ensemble methods**: Train multiple models

### Expected Results
- **After Phase 5A**: 60-65% recall (from 45%)
- **After Phase 5B**: 70-75% recall
- **After Phase 5C**: 75-80% recall

---

## Metrics to Track

1. **Recall**: Target 70%+ (currently 45%)
2. **Precision**: Maintain >95% (currently 100%)
3. **False Positive Rate**: Keep <5%
4. **Small ball detection**: Track separately (balls <20 pixels)
5. **Occluded ball detection**: Track separately

---

## Notes

- **Current strength**: 100% precision - model is very conservative
- **Main weakness**: Low recall - missing many valid balls
- **Key insight**: Model produces low confidence scores (mean 0.126), suggesting it's uncertain
- **Strategy**: Increase model confidence through better training, then use temporal/contextual post-processing
