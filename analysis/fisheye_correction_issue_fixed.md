# Fisheye Correction Issue - Root Cause and Fix

## Problem Summary

After adding fisheye correction to the inference pipeline, detections started appearing in wrong frames/locations, even though detection count increased.

## Root Cause Analysis

### Investigation Results

1. **Model Training Data**: 
   - Model was trained on **original fisheye images** (no fisheye correction in training configs)
   - Training augmentations: resize, flip, color jitter, motion blur, mosaic
   - No fisheye correction/defish mentioned in any training configuration

2. **Test Results**:
   - **Without fisheye correction**: 0 detections on first 20 frames
   - **With fisheye k=-0.40**: 21 detections in 17/20 frames (but wrong locations)
   - **Original test (random frames, no fisheye)**: 9/20 frames worked correctly

3. **Domain Mismatch**:
   - Model expects fisheye-distorted images (as trained)
   - Applying fisheye correction before inference changes image geometry
   - This causes coordinate system mismatch and wrong detection locations

## Solution Implemented

### Changes Made

1. **Disabled fisheye correction by default** in `predict_with_temporal_smoothing.py`
   - Model trained on fisheye images → inference should use fisheye images
   - Fisheye correction now optional (use `--fisheye-k <value>` to enable)

2. **Updated default behavior**:
   - `--fisheye-k` defaults to `None` (disabled)
   - Added informative message when fisheye is disabled
   - Temporal smoothing still works (independent of fisheye correction)

### Code Changes

```python
# Before: Fisheye enabled by default (k=-0.32)
parser.add_argument("--fisheye-k", type=float, default=-0.32, ...)

# After: Fisheye disabled by default (model expects fisheye images)
parser.add_argument("--fisheye-k", type=float, default=None, 
                   help="Fisheye correction k value (default: disabled - model trained on fisheye images)")
```

## Recommendations

### For Current Model (Epoch 89)

1. **Do NOT use fisheye correction** for inference
   - Model was trained on original fisheye images
   - Using fisheye correction causes domain mismatch
   - Results in wrong detection locations

2. **Use temporal smoothing** (works correctly)
   - Temporal smoothing is independent of fisheye correction
   - Improves recall by filling gaps between detections

3. **Test on frames with visible balls**
   - First 20 frames might not have balls visible
   - Use random frame selection or known good frame ranges

### For Future Models

If fisheye correction is desired:

1. **Option A: Retrain model with fisheye correction**
   - Apply fisheye correction during training data preprocessing
   - Train model on defished images
   - Use defished images for inference

2. **Option B: Post-process coordinates**
   - If fisheye correction improves detection quality
   - Transform bounding box coordinates back to original image space
   - Requires inverse fisheye transformation

## Current Status

- ✅ Fisheye correction disabled by default
- ✅ Temporal smoothing still enabled
- ✅ Model should work correctly on original fisheye images
- ⚠️  First 20 frames show 0 detections (may not have visible balls)

## Next Steps

1. Test on known good frame ranges (e.g., frames 1300+ where we had detections before)
2. Verify detections appear in correct locations without fisheye correction
3. If fisheye correction is needed, consider retraining model with defished images

## Files Modified

- `scripts/predict_with_temporal_smoothing.py` - Disabled fisheye correction by default
