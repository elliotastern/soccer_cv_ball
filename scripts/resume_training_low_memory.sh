#!/bin/bash
# Resume training with 20% memory reduction
# Optimized settings to reduce system memory requirements

cd "$(dirname "$0")/.."

CHECKPOINT_PATH="/workspace/soccer_cv_ball/models/soccer ball/checkpoint_20_soccer_ball.pth"
CONFIG_PATH="configs/resume_20_epochs_low_memory.yaml"

echo "============================================================"
echo "RESUMING TRAINING WITH 20% MEMORY REDUCTION"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Config: $CONFIG_PATH"
echo ""
echo "Memory Optimizations Applied:"
echo "  - Resolution: 1288 -> 1152 (20% reduction)"
echo "  - Multi-scale: Disabled"
echo "  - num_workers: 2 -> 1"
echo "  - pin_memory: Disabled"
echo "  - prefetch_factor: Reduced"
echo "============================================================"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Error: Config not found at $CONFIG_PATH"
    exit 1
fi

# Run training
python scripts/train_ball.py \
    --config "$CONFIG_PATH" \
    --output-dir models

echo ""
echo "✅ Training resumed with low memory settings."
