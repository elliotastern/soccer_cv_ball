#!/usr/bin/env python3
"""Upload folders to Hugging Face - uses folder upload to reduce commits."""

from huggingface_hub import HfApi
from pathlib import Path
import time

import os

REPO_ID = "eeeeeeeeeeeeee3/soccer-ball-detection"
TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")  # Set via environment variable
api = HfApi(token=TOKEN)

print("📤 Uploading folders to reduce commits...")

# Upload configs folder
print("\n📁 Uploading configs/")
try:
    api.upload_folder(
        folder_path="configs",
        repo_id=REPO_ID,
        repo_type="model",
        token=TOKEN
    )
    print("   ✅ configs/ uploaded")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Upload src folder
print("\n📁 Uploading src/")
try:
    api.upload_folder(
        folder_path="src",
        repo_id=REPO_ID,
        repo_type="model",
        token=TOKEN
    )
    print("   ✅ src/ uploaded")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Upload scripts folder (but exclude __pycache__)
print("\n📁 Uploading scripts/")
try:
    # Upload scripts folder, excluding cache
    api.upload_folder(
        folder_path="scripts",
        repo_id=REPO_ID,
        repo_type="model",
        token=TOKEN,
        ignore_patterns=["__pycache__", "*.pyc"]
    )
    print("   ✅ scripts/ uploaded")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Upload root markdown files
print("\n📁 Uploading documentation files...")
md_files = list(Path('.').glob('*.md'))
for md_file in md_files:
    if md_file.name == 'README.md':
        continue
    try:
        api.upload_file(
            path_or_fileobj=str(md_file),
            path_in_repo=md_file.name,
            repo_id=REPO_ID,
            repo_type="model",
            token=TOKEN
        )
        print(f"   ✅ {md_file.name}")
    except Exception as e:
        print(f"   ⚠️  {md_file.name}: {e}")

print(f"\n✅ Upload complete!")
print(f"View at: https://huggingface.co/{REPO_ID}")
