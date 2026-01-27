#!/usr/bin/env python3
"""Upload all project files to Hugging Face in batches."""

from huggingface_hub import HfApi
from pathlib import Path
import os

import os

REPO_ID = "eeeeeeeeeeeeee3/soccer-ball-detection"
TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")  # Set via environment variable

api = HfApi(token=TOKEN)

def get_all_files(root_dir):
    """Get all files to upload, focusing on code and config files only."""
    # Only include these file types (code, configs, docs)
    include_extensions = {'.py', '.yaml', '.yml', '.md', '.txt', '.json', '.html', '.sh'}
    
    # Exclude these directories completely
    exclude_dirs = {
        '__pycache__', '.git', 'venv', 'env', '.venv', 
        'mlruns', 'logs', 'data', 'datasets', '.ipynb_checkpoints',
        'models/checkpoints'  # Exclude large model files
    }
    
    # Exclude these file extensions
    exclude_extensions = {
        '.log', '.cache', '.pyc', '.pyo', '.pyd', 
        '.jpg', '.jpeg', '.png', '.gif', '.pth', '.pt',
        '.tar', '.gz', '.zip'
    }
    
    files = []
    root = Path(root_dir)
    
    for file_path in root.rglob('*'):
        if file_path.is_file():
            path_str = str(file_path)
            
            # Skip excluded directories
            if any(excluded in path_str for excluded in exclude_dirs):
                continue
            
            # Only include specific file types
            if file_path.suffix not in include_extensions:
                continue
            
            # Skip excluded extensions
            if file_path.suffix in exclude_extensions:
                continue
            
            # Skip files in data subdirectories (even if they're JSON)
            if '/data/' in path_str and file_path.parent.name != 'data':
                # Only allow top-level data JSON files (like annotations)
                if 'annotations.json' not in path_str and 'data/' in path_str:
                    continue
            
            files.append(file_path)
    
    return sorted(files)

def upload_files(files, batch_size=20):
    """Upload files in batches."""
    total = len(files)
    uploaded = 0
    failed = []
    
    print(f"📤 Uploading {total} files in batches of {batch_size}...")
    
    for i in range(0, total, batch_size):
        batch = files[i:i+batch_size]
        print(f"\n📦 Batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} files)")
        
        for file_path in batch:
            try:
                rel_path = file_path.relative_to(Path('.'))
                path_in_repo = str(rel_path).replace('\\', '/')
                
                # Skip if already uploaded (README, main files)
                if path_in_repo in ['README.md', 'requirements.txt', 'main.py']:
                    continue
                
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=path_in_repo,
                    repo_id=REPO_ID,
                    repo_type="model"
                )
                uploaded += 1
                if uploaded % 10 == 0:
                    print(f"   ✅ {uploaded}/{total} files uploaded...")
            except Exception as e:
                failed.append((str(file_path), str(e)))
                print(f"   ⚠️  Failed: {file_path.name} - {e}")
    
    print(f"\n✅ Upload complete!")
    print(f"   Uploaded: {uploaded}/{total} files")
    if failed:
        print(f"   Failed: {len(failed)} files")
        for file_path, error in failed[:5]:
            print(f"      - {file_path}: {error}")
    
    return uploaded, failed

if __name__ == "__main__":
    print(f"🔍 Finding files to upload...")
    files = get_all_files('.')
    print(f"   Found {len(files)} files")
    
    uploaded, failed = upload_files(files)
    print(f"\n🎉 Done! View at: https://huggingface.co/{REPO_ID}")
