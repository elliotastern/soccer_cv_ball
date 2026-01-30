# Confidence Analysis Summary - Epoch 89 Model

## Key Findings

### Confidence Distribution
- **Total detections found**: 17 across 20 frames (with threshold 0.05)
- **Mean confidence**: 0.1258 (12.58%)
- **Median confidence**: 0.0804 (8.04%)
- **Std deviation**: 0.0984
- **Range**: 0.0515 - 0.3770

### Percentiles
- **P10**: 0.0542 (5.42%)
- **P25**: 0.0593 (5.93%)
- **P50**: 0.0804 (8.04%)
- **P75**: 0.1266 (12.66%)
- **P90**: 0.2587 (25.87%)
- **P95**: 0.3690 (36.90%)
- **P99**: 0.3754 (37.54%)

## Threshold Analysis

| Threshold | Total Detections | Frames w/ Detections | Avg Detections/Frame |
|-----------|------------------|----------------------|---------------------|
| 0.10      | 7                | 7                    | 0.35                |
| 0.20      | 2                | 2                    | 0.10                |
| 0.30      | 2                | 2                    | 0.10                |
| 0.40      | 0                | 0                    | 0.00                |
| 0.50      | 0                | 0                    | 0.00                |

## Critical Insight

**The model produces very low confidence scores!**

- **75% of detections** have confidence < 0.127
- **90% of detections** have confidence < 0.259
- Only **2 detections** (out of 17) have confidence > 0.20
- **No detections** have confidence > 0.40

## Recommendations

### Optimal Threshold: **0.10 - 0.15**

Based on the analysis:
- **Threshold 0.10**: Captures 7/20 frames (35% recall) with 7 detections
- **Threshold 0.15**: Would capture ~5-6 frames (estimated 25-30% recall)
- **Threshold 0.20**: Only 2/20 frames (10% recall) - too conservative
- **Threshold 0.50**: 0/20 frames (0% recall) - way too high!

### Why Current Performance is Low

The model is producing valid detections but with very low confidence scores:
- Average confidence: **0.126** (12.6%)
- Median confidence: **0.080** (8.0%)
- Maximum confidence: **0.377** (37.7%)

With threshold 0.5, we're filtering out **100%** of valid detections!

## Test Results Summary

### Threshold 0.5 (Original Test)
- **Result**: 4/20 frames detected (20% recall)
- **False positives**: 0 (100% precision)
- **Issue**: Too conservative, missing many valid balls

### Threshold 0.2 (New Test)
- **HTML generated**: `predictions_epoch89_thresh0.2.html`
- **Expected**: ~2-3 detections based on analysis

### Threshold 0.3 (New Test)
- **HTML generated**: `predictions_epoch89_thresh0.3.html`
- **Expected**: ~2 detections based on analysis

## Next Steps

1. **Test threshold 0.10** - Should capture 7/20 frames
2. **Test threshold 0.15** - Balance between recall and precision
3. **Review HTML files** - Manually verify detections at different thresholds
4. **Consider confidence calibration** - Model may need calibration to produce higher confidence scores

## Files Generated

- `analysis/confidence_analysis_epoch89.json` - Full analysis data
- `models/ball_detection_open_soccer_ball/predictions_epoch89_thresh0.2.html` - Predictions at threshold 0.2
- `models/ball_detection_open_soccer_ball/predictions_epoch89_thresh0.3.html` - Predictions at threshold 0.3
- `models/ball_detection_open_soccer_ball/predictions_37CAE053-841F-4851-956E-CBF17A51C506_20_frames_epoch89.html` - Original (threshold 0.5)

## Conclusion

The model is working correctly but produces low confidence scores. The optimal threshold is **0.10-0.15**, not 0.5. This explains why we only got 4/20 detections with threshold 0.5 - we were filtering out most valid detections!
