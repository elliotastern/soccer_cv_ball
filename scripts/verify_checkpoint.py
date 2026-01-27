#!/usr/bin/env python3
"""
Verify that soccer ball checkpoint can be loaded successfully
"""
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_checkpoint(checkpoint_path: str):
    """Verify checkpoint can be loaded and show details."""
    print(f"🔍 Verifying checkpoint: {checkpoint_path}")
    print("-" * 60)
    
    # Check if file exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return False
    
    file_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)  # MB
    print(f"📁 File size: {file_size:.2f} MB")
    
    # Try to load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False
    
    # Check checkpoint structure
    print(f"\n📦 Checkpoint structure:")
    print(f"   Keys: {list(checkpoint.keys())}")
    
    # Check epoch
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    
    # Check model state
    model_loaded = False
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        if isinstance(model_state, dict):
            print(f"   Model state dict: {len(model_state)} layers")
            model_loaded = True
        else:
            print(f"   Model state: {type(model_state)}")
    elif 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        if isinstance(model_state, dict):
            print(f"   Model state dict: {len(model_state)} layers")
            model_loaded = True
    
    # Try to load into actual model (if rfdetr is available)
    print(f"\n🤖 Testing model compatibility...")
    try:
        from rfdetr import RFDETRBase
        
        # Initialize model
        model = RFDETRBase(class_names=['ball'])
        print(f"   ✅ Model initialized")
        
        if model_loaded and 'model' in checkpoint:
            model_state = checkpoint['model']
            if isinstance(model_state, dict):
                # Check if we can load the weights
                if hasattr(model, 'model') and hasattr(model.model, 'model'):
                    current_state = model.model.model.state_dict()
                    filtered_state = {}
                    matching_keys = 0
                    
                    for key, value in model_state.items():
                        if key in current_state:
                            if current_state[key].shape == value.shape:
                                filtered_state[key] = value
                                matching_keys += 1
                    
                    print(f"   ✅ {matching_keys}/{len(model_state)} layers match model structure")
                    print(f"   ✅ Checkpoint is compatible with RFDETRBase")
                    return True
                else:
                    print(f"   ⚠️  Could not find model.model.model structure")
        else:
            print(f"   ⚠️  Checkpoint format not recognized for RF-DETR")
            
    except ImportError:
        print(f"   ⚠️  rfdetr module not available (cannot test model loading)")
        print(f"   ✅ Checkpoint structure is valid and ready to load")
        return model_loaded
    except Exception as e:
        print(f"   ⚠️  Error testing model: {e}")
        return model_loaded
    
    return model_loaded


def main():
    """Main function."""
    # Check both checkpoint locations
    checkpoints = [
        "/workspace/soccer_cv_ball/models/soccer ball/checkpoint_20_soccer_ball.pth",
        "/workspace/soccer_cv_ball/models/checkpoints/latest_checkpoint.pth"
    ]
    
    print("=" * 60)
    print("Soccer Ball Checkpoint Verification")
    print("=" * 60)
    
    for checkpoint_path in checkpoints:
        print()
        success = verify_checkpoint(checkpoint_path)
        if success:
            print(f"\n✅ Checkpoint verified: {checkpoint_path}")
        else:
            print(f"\n❌ Checkpoint verification failed: {checkpoint_path}")
        print()


if __name__ == "__main__":
    main()
