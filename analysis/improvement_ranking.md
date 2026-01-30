# Ranked Model Improvement Strategies

## Current Performance Baseline
- **Recall**: 45% (9/20 frames)
- **Precision**: 100% (9/9 correct, 0 false positives)
- **Model**: RF-DETR BASE (31.85M parameters, Epoch 89)
- **Goal**: 70%+ recall while maintaining >95% precision

## Ranking Criteria
- **Impact**: Expected recall improvement
- **Effort**: Implementation time and complexity
- **ROI**: Impact/Effort ratio
- **Risk**: Chance of negative side effects
- **Feasibility**: Ease of implementation

---

## 🥇 TIER 1: Highest ROI (Do First)

### #1: SAHI Multi-Scale Inference ⭐⭐⭐⭐⭐
**Impact**: +15-20% recall (45% → 60-65%)  
**Effort**: Medium (2-3 days)  
**ROI**: Very High  
**Risk**: Low (post-processing, doesn't affect training)  
**Status**: Not implemented

**Why #1:**
- Highest impact for effort invested
- Specifically targets small/distant balls (main weakness)
- No training required - pure inference improvement
- Proven technique for small object detection
- Can be combined with existing temporal smoothing

**Implementation:**
- Integrate SAHI library
- Tune slice sizes for ball detection (smaller slices for tiny balls)
- Optimize overlap ratio
- Test on validation set

**Expected Result**: 60-65% recall

---

### #2: Hard Negative Mining ⭐⭐⭐⭐
**Impact**: +10-15% recall (45% → 55-60%)  
**Effort**: Medium (3-5 days)  
**ROI**: High  
**Risk**: Low (adds data, doesn't change architecture)  
**Status**: Not implemented

**Why #2:**
- Directly addresses missed detections
- High impact with moderate effort
- Iterative improvement (can repeat)
- No architecture changes needed
- Builds on existing training pipeline

**Implementation:**
1. Run inference on validation/test videos
2. Identify frames with visible balls but no detections
3. Manually annotate these frames
4. Add to training set (10-20% more data)
5. Retrain for 10-20 epochs

**Expected Result**: 55-60% recall after first iteration

---

### #3: Test Lower Confidence Thresholds ⭐⭐⭐⭐
**Impact**: +5-10% recall (45% → 50-55%)  
**Effort**: Low (1 day)  
**ROI**: Very High  
**Risk**: Medium (may reduce precision)  
**Status**: Partially tested (0.10 optimal, need to test 0.05, 0.08)

**Why #3:**
- Very low effort, decent impact
- Quick to test and validate
- Can be reverted if precision drops
- Model already produces low confidence scores (mean 0.126)

**Implementation:**
- Test thresholds: 0.05, 0.08
- Monitor precision (target >90%)
- Compare recall vs precision tradeoff
- Select optimal threshold

**Expected Result**: 50-55% recall (if precision stays >90%)

---

## 🥈 TIER 2: High Impact, Higher Effort

### #4: Temporal Smoothing (Already Implemented) ⭐⭐⭐
**Impact**: +10-15% recall (when gaps exist)  
**Effort**: Done ✅  
**ROI**: N/A (already done)  
**Status**: ✅ Implemented

**Note**: Already implemented and working. Shows +45pp improvement when gaps exist (35% → 80% on frames 1300+).

---

### #5: More Diverse Training Data ⭐⭐⭐
**Impact**: +10-15% recall  
**Effort**: High (2-3 weeks data collection)  
**ROI**: Medium  
**Risk**: Low  
**Status**: Not implemented

**Why #5:**
- Addresses domain generalization
- Improves robustness across conditions
- Requires significant data collection effort
- Long-term investment

**Implementation:**
- Collect videos with varying:
  - Lighting (sunny, cloudy, night)
  - Camera angles (high, low, side)
  - Ball sizes (close-up, distant)
  - Field conditions (wet, dry, different grass)
- Target: 2-3x more real-world data
- Annotate and add to training set

**Expected Result**: 55-60% recall after retraining

---

### #6: Focal Loss Tuning ⭐⭐⭐
**Impact**: +5-8% recall  
**Effort**: Medium (1-2 days)  
**ROI**: Medium  
**Risk**: Low (hyperparameter tuning)  
**Status**: Using default (alpha=0.25, gamma=2.0)

**Why #6:**
- Moderate impact, moderate effort
- Can be done in parallel with other improvements
- Low risk (just hyperparameter tuning)

**Implementation:**
- Test different alpha/gamma combinations
- Focus on ball class (minority class)
- Use validation mAP to select best values
- Retrain with optimal parameters

**Expected Result**: +5-8% recall improvement

---

## 🥉 TIER 3: Medium Impact, Various Effort

### #7: Test-Time Augmentation (TTA) ⭐⭐⭐
**Impact**: +5-8% recall  
**Effort**: Medium (2-3 days)  
**ROI**: Medium  
**Risk**: Low (inference-only)  
**Status**: Not implemented

**Why #7:**
- Moderate impact
- No training required
- Can be combined with other techniques
- Slight inference overhead

**Implementation:**
- Augment images at inference (scales, rotations, brightness)
- Run model on each augmentation
- Combine predictions (average or max)
- Test on validation set

**Expected Result**: +5-8% recall

---

### #8: Longer Training ⭐⭐
**Impact**: +3-5% recall  
**Effort**: High (weeks of training)  
**ROI**: Low  
**Risk**: Low (just more epochs)  
**Status**: Currently at 90 epochs

**Why #8:**
- Diminishing returns after 90 epochs
- High effort (time and compute)
- May not provide significant improvement
- Better to focus on data/architecture first

**Implementation:**
- Extend training to 120-150 epochs
- Use early stopping based on validation mAP
- Monitor for overfitting

**Expected Result**: +3-5% recall (marginal improvement)

---

### #9: Architecture Improvements (Different Backbone) ⭐⭐
**Impact**: +5-10% recall  
**Effort**: Very High (weeks of retraining)  
**ROI**: Low  
**Risk**: Medium (may not improve)  
**Status**: Using ResNet-50

**Why #9:**
- High effort, uncertain results
- Requires full retraining
- May not be necessary (current architecture is good)
- Better to optimize data/training first

**Options:**
- ResNet-101 (deeper, better features)
- EfficientNet (better efficiency)
- Swin Transformer (better for small objects)

**Expected Result**: +5-10% recall (if better backbone helps)

---

### #10: Ensemble Methods ⭐⭐
**Impact**: +5-10% recall  
**Effort**: Very High (train multiple models)  
**ROI**: Low  
**Risk**: Low  
**Status**: Not implemented

**Why #10:**
- Requires training multiple models (high compute cost)
- Moderate impact
- Better ROI from single model improvements first

**Implementation:**
- Train 3-5 models with different:
  - Initializations
  - Data augmentations
  - Architectures
- Combine predictions (average or vote)

**Expected Result**: +5-10% recall

---

## 📊 Summary Ranking

| Rank | Strategy | Impact | Effort | ROI | Priority |
|------|----------|--------|--------|-----|----------|
| 1 | **SAHI Multi-Scale Inference** | +15-20% | Medium | ⭐⭐⭐⭐⭐ | **DO FIRST** |
| 2 | **Hard Negative Mining** | +10-15% | Medium | ⭐⭐⭐⭐ | **DO SECOND** |
| 3 | **Lower Threshold Testing** | +5-10% | Low | ⭐⭐⭐⭐ | **DO THIRD** |
| 4 | Temporal Smoothing | +10-15% | Done ✅ | N/A | ✅ **DONE** |
| 5 | More Diverse Training Data | +10-15% | High | ⭐⭐⭐ | Do after quick wins |
| 6 | Focal Loss Tuning | +5-8% | Medium | ⭐⭐⭐ | Can do in parallel |
| 7 | Test-Time Augmentation | +5-8% | Medium | ⭐⭐⭐ | Optional |
| 8 | Longer Training | +3-5% | High | ⭐⭐ | Low priority |
| 9 | Architecture Changes | +5-10% | Very High | ⭐⭐ | Last resort |
| 10 | Ensemble Methods | +5-10% | Very High | ⭐⭐ | Last resort |

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1 week)
1. ✅ Temporal Smoothing (DONE)
2. **SAHI Multi-Scale Inference** (2-3 days)
3. **Lower Threshold Testing** (1 day)

**Expected**: 45% → 60-65% recall

### Phase 2: Data Improvements (2-3 weeks)
4. **Hard Negative Mining** (3-5 days + retraining)
5. **More Diverse Training Data** (2-3 weeks collection)

**Expected**: 60-65% → 70-75% recall

### Phase 3: Fine-Tuning (1-2 weeks)
6. **Focal Loss Tuning** (1-2 days + retraining)
7. **Test-Time Augmentation** (2-3 days, optional)

**Expected**: 70-75% → 75-80% recall

### Phase 4: Advanced (Only if needed)
8. Longer Training
9. Architecture Changes
10. Ensemble Methods

---

## Key Insights

1. **Inference improvements** (SAHI, TTA) have highest ROI - no retraining needed
2. **Data quality** (hard negative mining) is more important than quantity
3. **Current model is good** - focus on inference/post-processing first
4. **Precision is perfect** (100%) - can afford to be slightly more aggressive
5. **Model produces low confidence** (mean 0.126) - suggests need for better training or calibration

---

## Expected Final Performance

**After Phase 1-3:**
- **Recall**: 75-80% (from 45%)
- **Precision**: >95% (maintained)
- **Goal Achievement**: ✅ Exceeds 70% target

**Timeline**: 4-6 weeks total
