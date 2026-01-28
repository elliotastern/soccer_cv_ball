#!/bin/bash
# Fast checkpoint compression script

CHECKPOINT_PATH="/workspace/soccer_cv_ball/models/checkpoint.pth"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Compressing checkpoint (475MB)..."
gzip -c "$CHECKPOINT_PATH" > "${CHECKPOINT_PATH}.${TIMESTAMP}.gz"

echo "Original size:"
du -h "$CHECKPOINT_PATH"
echo "Compressed size:"
du -h "${CHECKPOINT_PATH}.${TIMESTAMP}.gz"

echo "Done! Saved to: ${CHECKPOINT_PATH}.${TIMESTAMP}.gz"
