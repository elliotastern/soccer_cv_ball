#!/usr/bin/env python3
"""Simple upload - code files only."""

from huggingface_hub import HfApi
from pathlib import Path

import os

REPO_ID = "eeeeeeeeeeeeee3/soccer-ball-detection"
TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")  # Set via environment variable
api = HfApi(token=TOKEN)

# Upload directories in order
dirs_to_upload = [
    ("configs", ["*.yaml", "*.yml"]),
    ("src", ["*.py"]),
    ("scripts", ["*.py"]),
]

print("📤 Uploading code files...")
total = 0

for dir_name, patterns in dirs_to_upload:
    dir_path = Path(dir_name)
    if not dir_path.exists():
        continue
    
    print(f"\n📁 {dir_name}/")
    for pattern in patterns:
        for file_path in dir_path.rglob(pattern):
            if '__pycache__' in str(file_path):
                continue
            rel_path = file_path.relative_to(Path('.'))
            path_in_repo = str(rel_path).replace('\\', '/')
            
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=path_in_repo,
                    repo_id=REPO_ID,
                    repo_type="model"
                )
                total += 1
                if total % 5 == 0:
                    print(f"   ✅ {total} files...")
            except Exception as e:
                print(f"   ⚠️  {file_path.name}: {e}")

# Upload root Python files
print("\n📁 Root Python files")
for py_file in Path('.').glob('*.py'):
    if py_file.name == 'quick_push_hf.py':
        continue
    try:
        api.upload_file(
            path_or_fileobj=str(py_file),
            path_in_repo=py_file.name,
            repo_id=REPO_ID,
            repo_type="model"
        )
        total += 1
    except Exception as e:
        print(f"   ⚠️  {py_file.name}: {e}")

# Upload markdown files
print("\n📁 Documentation")
for md_file in Path('.').glob('*.md'):
    if md_file.name == 'README.md':
        continue
    try:
        api.upload_file(
            path_or_fileobj=str(md_file),
            path_in_repo=md_file.name,
            repo_id=REPO_ID,
            repo_type="model"
        )
        total += 1
    except Exception as e:
        print(f"   ⚠️  {md_file.name}: {e}")

print(f"\n✅ Uploaded {total} files!")
print(f"View at: https://huggingface.co/{REPO_ID}")
