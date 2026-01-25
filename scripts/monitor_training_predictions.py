#!/usr/bin/env python3
"""
Monitor training checkpoints and generate prediction HTML after each epoch.
Runs alongside training to automatically generate visualizations.
"""
import argparse
import time
import torch
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.generate_epoch_predictions import generate_predictions_html


def get_latest_checkpoint(checkpoint_dir: Path) -> tuple[Path, int] | tuple[None, None]:
    """
    Get the latest checkpoint file and its epoch number.
    
    Returns:
        (checkpoint_path, epoch_number) or (None, None) if no checkpoint found
    """
    checkpoint_files = list(checkpoint_dir.glob("checkpoint.pth"))
    if not checkpoint_files:
        return None, None
    
    checkpoint_path = checkpoint_files[0]
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        epoch = checkpoint.get('epoch', 0)
        return checkpoint_path, epoch
    except Exception as e:
        print(f"⚠️  Error loading checkpoint: {e}")
        return None, None


def monitor_and_generate_predictions(
    checkpoint_dir: Path,
    val_dataset_dir: Path,
    output_html_dir: Path,
    num_frames: int = 10,
    confidence_threshold: float = 0.3,
    check_interval: int = 60  # Check every 60 seconds
):
    """
    Monitor checkpoint directory and generate predictions HTML after each epoch.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        val_dataset_dir: Validation dataset directory
        output_html_dir: Directory to save HTML files
        num_frames: Number of frames to visualize
        confidence_threshold: Confidence threshold for predictions
        check_interval: How often to check for new checkpoints (seconds)
    """
    print("=" * 60)
    print("PREDICTION MONITORING STARTED")
    print("=" * 60)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Validation dataset: {val_dataset_dir}")
    print(f"Output HTML directory: {output_html_dir}")
    print(f"Check interval: {check_interval} seconds")
    print("=" * 60)
    
    # Load RF-DETR model (will be reloaded with checkpoint weights)
    try:
        from rfdetr import RFDETRBase
        # Initialize with ball class for ball-only detection
        model = RFDETRBase(class_names=['ball'])
        print("✅ RF-DETR model initialized")
    except Exception as e:
        print(f"❌ Error loading RF-DETR model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Track processed epochs
    processed_epochs = set()
    
    # Create output directory
    output_html_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 Monitoring for new checkpoints...")
    print(f"   (Will check every {check_interval} seconds)\n")
    
    try:
        while True:
            # Check for latest checkpoint
            checkpoint_path, epoch = get_latest_checkpoint(checkpoint_dir)
            
            if checkpoint_path and epoch is not None:
                # Check if we've already processed this epoch
                if epoch not in processed_epochs:
                    print(f"\n📊 New checkpoint detected: Epoch {epoch}")
                    print(f"   Checkpoint: {checkpoint_path.name}")
                    
                    # Load checkpoint into model
                    try:
                        checkpoint_data = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
                        
                        # Debug: Check checkpoint type
                        if not isinstance(checkpoint_data, dict):
                            print(f"⚠️  Checkpoint is not a dict (type: {type(checkpoint_data)}), skipping weight loading")
                            checkpoint_data = None
                        
                        # RF-DETR checkpoint structure: {'model': state_dict, 'optimizer': ..., 'epoch': ...}
                        # RF-DETR model structure: model.model.model is the actual PyTorch model
                        loaded = False
                        if checkpoint_data and isinstance(checkpoint_data, dict) and 'model' in checkpoint_data:
                            # RF-DETR format
                            try:
                                model_state = checkpoint_data['model']
                                if isinstance(model_state, dict):
                                    # Load into the underlying PyTorch model: model.model.model
                                    if hasattr(model, 'model') and hasattr(model.model, 'model'):
                                        # Filter out class embedding layers that might have size mismatches
                                        current_model_state = model.model.model.state_dict()
                                        filtered_state = {}
                                        skipped_keys = []
                                        
                                        for key, value in model_state.items():
                                            if key in current_model_state:
                                                if current_model_state[key].shape == value.shape:
                                                    filtered_state[key] = value
                                                else:
                                                    skipped_keys.append(key)
                                            else:
                                                skipped_keys.append(key)
                                        
                                        # Load filtered state dict
                                        missing_keys, unexpected_keys = model.model.model.load_state_dict(filtered_state, strict=False)
                                        if skipped_keys:
                                            print(f"⚠️  Skipped {len(skipped_keys)} keys due to size mismatch")
                                        if missing_keys:
                                            print(f"⚠️  {len(missing_keys)} missing keys")
                                        loaded = True
                                        print(f"✅ Loaded checkpoint weights (epoch {checkpoint_data.get('epoch', 'N/A')})")
                                    else:
                                        print(f"⚠️  Could not find model.model.model to load weights")
                            except Exception as e:
                                print(f"⚠️  Error loading from checkpoint: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        if not loaded:
                            print(f"⚠️  Could not load checkpoint weights, using current model")
                        
                        # Generate predictions HTML
                        output_html_path = output_html_dir / f"epoch_{epoch:02d}_predictions.html"
                        
                        generate_predictions_html(
                            model=model,
                            val_dataset_dir=val_dataset_dir,
                            output_html_path=output_html_path,
                            epoch=epoch,
                            num_frames=num_frames,
                            seed=42,  # Fixed seed for same frames
                            confidence_threshold=confidence_threshold
                        )
                        
                        # Mark as processed
                        processed_epochs.add(epoch)
                        print(f"✅ Epoch {epoch} predictions generated: {output_html_path}")
                        
                    except Exception as e:
                        print(f"❌ Error processing checkpoint: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Already processed, just waiting
                    pass
            else:
                # No checkpoint yet
                pass
            
            # Wait before next check
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error in monitoring loop: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Monitor training checkpoints and generate prediction HTML after each epoch"
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help='Directory containing training checkpoints'
    )
    parser.add_argument(
        '--val-dataset',
        type=str,
        required=True,
        help='Path to validation dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='training_predictions',
        help='Directory to save HTML files (default: training_predictions)'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=10,
        help='Number of frames to visualize (default: 10)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.3,
        help='Confidence threshold for predictions (default: 0.3)'
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=60,
        help='How often to check for new checkpoints in seconds (default: 60)'
    )
    
    args = parser.parse_args()
    
    monitor_and_generate_predictions(
        checkpoint_dir=Path(args.checkpoint_dir),
        val_dataset_dir=Path(args.val_dataset),
        output_html_dir=Path(args.output_dir),
        num_frames=args.num_frames,
        confidence_threshold=args.confidence,
        check_interval=args.check_interval
    )


if __name__ == "__main__":
    main()
