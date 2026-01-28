#!/bin/bash
# Resume training from 20-epoch checkpoint
# This script provides a simple way to resume training

cd "$(dirname "$0")/.."

CHECKPOINT_PATH="/workspace/soccer_cv_ball/models/soccer ball/checkpoint_20_soccer_ball.pth"
CONFIG_PATH="configs/resume_20_epochs.yaml"

echo "============================================================"
echo "RESUMING TRAINING FROM 20-EPOCH CHECKPOINT"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Config: $CONFIG_PATH"
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
echo "✅ Training resumed. Check logs for progress."
