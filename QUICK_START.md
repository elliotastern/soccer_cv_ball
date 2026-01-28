# Quick Start Guide - Resume Training

## Current Status
- **Phase:** 1.5 (High-Resolution Training)
- **Epoch:** 51-60 in progress
- **Log:** `training_phase1.5_highres.log`
- **Checkpoint:** `models/checkpoint.pth` (epoch 49)

## Quick Commands

### Check Status
```bash
cd /workspace/soccer_cv_ball
tail -50 training_phase1.5_highres.log
ps aux | grep train_ball
```

### Resume Training (if stopped)
```bash
cd /workspace/soccer_cv_ball
python3 scripts/train_ball.py \
    --config configs/resume_with_highres_gradaccum.yaml \
    --output-dir models 2>&1 | tee training_phase1.5_highres.log &
```

### Monitor Progress
```bash
tail -f training_phase1.5_highres.log
python3 scripts/evaluate_training_progress.py
```

## Latest Results
- **Small Objects mAP:** 0.632 (cutting-edge!) ✅
- **Overall mAP:** 0.667 (excellent)
- **Improvement:** +5.7% from baseline

## Next Phase
After epoch 60: Evaluate and consider Phase 3 (multi-scale training)

See `RESUME_WORK_SUMMARY.md` for complete details.
