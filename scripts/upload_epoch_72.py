#!/usr/bin/env python3
"""Upload epoch 72 checkpoint to Hugging Face with specific name."""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

# Repository ID
REPO_ID = "eeeeeeeeeeeeee3/soccer-ball-detection"

# Get token from environment
TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

# Initialize API (will use cached token if available)
api = HfApi(token=TOKEN)

def upload_epoch_72():
    """Upload epoch 72 checkpoint to Hugging Face."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print(f"📦 Uploading epoch 72 checkpoint to: {REPO_ID}")
    print(f"   Project root: {project_root}\n")
    
    # Verify login
    try:
        user_info = api.whoami()
        print(f"🔐 Logged in as: {user_info.get('name', 'unknown')}\n")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("\n📝 To authenticate, set HUGGINGFACE_TOKEN environment variable")
        sys.exit(1)
    
    # Create repository if it doesn't exist
    print(f"📁 Ensuring repository exists: {REPO_ID}")
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True, token=TOKEN)
        print("✅ Repository ready\n")
    except HfHubHTTPError as e:
        if "already exists" in str(e).lower():
            print("✅ Repository already exists\n")
        else:
            print(f"⚠️  Repository creation warning: {e}\n")
    
    # Find epoch 72 checkpoint
    checkpoint_path = "models/checkpoint.pth"
    checkpoint_file = Path(checkpoint_path)
    
    if not checkpoint_file.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Verify it's epoch 72
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        epoch = ckpt.get('epoch', None)
        if epoch != 72:
            print(f"⚠️  Warning: checkpoint.pth is epoch {epoch}, not 72")
            print(f"   Proceeding anyway...\n")
    except Exception as e:
        print(f"⚠️  Could not verify epoch: {e}")
        print(f"   Proceeding anyway...\n")
    
    file_size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
    print(f"📤 Uploading {checkpoint_path} ({file_size_mb:.1f} MB) as epoch_72_checkpoint.pth...")
    
    # Upload as epoch_72_checkpoint.pth
    try:
        api.upload_file(
            path_or_fileobj=str(checkpoint_file),
            path_in_repo="checkpoints/epoch_72_checkpoint.pth",
            repo_id=REPO_ID,
            repo_type="model",
            token=TOKEN
        )
        print(f"   ✅ epoch_72_checkpoint.pth uploaded successfully\n")
        
        # Also upload as regular checkpoint.pth (latest)
        print(f"📤 Also uploading as latest checkpoint.pth...")
        api.upload_file(
            path_or_fileobj=str(checkpoint_file),
            path_in_repo="checkpoints/checkpoint.pth",
            repo_id=REPO_ID,
            repo_type="model",
            token=TOKEN
        )
        print(f"   ✅ checkpoint.pth uploaded successfully\n")
        
        print(f"\n✅ Upload complete!")
        print(f"   - epoch_72_checkpoint.pth (separate copy)")
        print(f"   - checkpoint.pth (latest)")
        print(f"   View at: https://huggingface.co/{REPO_ID}/tree/main/checkpoints")
        
    except Exception as e:
        print(f"   ❌ Failed to upload: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    upload_epoch_72()
