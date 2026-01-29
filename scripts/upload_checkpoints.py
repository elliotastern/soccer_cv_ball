#!/usr/bin/env python3
"""Upload model checkpoints to Hugging Face Hub (checkpoints first)."""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

# Repository ID
REPO_ID = "eeeeeeeeeeeeee3/soccer-ball-detection"  # From existing scripts

# Get token from environment
TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

# Initialize API (will use cached token if available)
api = HfApi(token=TOKEN)

# Checkpoints to upload (priority order)
CHECKPOINTS = [
    "models/checkpoint_best_ema.pth",      # Best EMA model (Epoch 66, 352MB)
    "models/checkpoint_best_total.pth",     # Best total model (122MB)
    "models/checkpoint.pth",                # Latest checkpoint (Epoch 69, Phase 3 final, 475MB)
    "models/checkpoint0069.pth",            # Epoch 69 checkpoint (Phase 3 final, 475MB)
    "models/checkpoint0059.pth",            # Epoch 59 checkpoint (Phase 1.5 final, 475MB)
]

def upload_checkpoints():
    """Upload checkpoints to Hugging Face."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print(f"📦 Uploading checkpoints to: {REPO_ID}")
    print(f"   Project root: {project_root}\n")
    
    # Verify login
    try:
        user_info = api.whoami()
        print(f"🔐 Logged in as: {user_info.get('name', 'unknown')}\n")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("\n📝 To authenticate, either:")
        print("   1. Set HUGGINGFACE_TOKEN environment variable:")
        print("      export HUGGINGFACE_TOKEN=your_token_here")
        print("   2. Or login via CLI:")
        print("      huggingface-cli login")
        print("   3. Or login programmatically:")
        print("      python3 -c \"from huggingface_hub import login; login()\"")
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
    
    # Upload checkpoints
    print("📤 Uploading checkpoints...\n")
    uploaded = 0
    failed = 0
    
    for checkpoint_path in CHECKPOINTS:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            print(f"⚠️  Skipping {checkpoint_path} (not found)")
            continue
        
        file_size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
        print(f"📤 Uploading {checkpoint_path} ({file_size_mb:.1f} MB)...")
        
        try:
            api.upload_file(
                path_or_fileobj=str(checkpoint_file),
                path_in_repo=f"checkpoints/{checkpoint_file.name}",
                repo_id=REPO_ID,
                repo_type="model",
                token=TOKEN
            )
            print(f"   ✅ {checkpoint_file.name} uploaded successfully\n")
            uploaded += 1
        except Exception as e:
            print(f"   ❌ Failed to upload {checkpoint_file.name}: {e}\n")
            failed += 1
    
    print(f"\n✅ Checkpoint upload complete!")
    print(f"   Uploaded: {uploaded} checkpoints")
    if failed > 0:
        print(f"   Failed: {failed} checkpoints")
    print(f"   View at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    upload_checkpoints()
