#!/usr/bin/env python3
"""
Push soccer ball detection project to Hugging Face Hub.
Includes code, configs, and trained model checkpoints.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

try:
    from huggingface_hub import HfApi, create_repo, login
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)


def get_files_to_upload(project_root: Path, exclude_patterns: List[str] = None) -> List[Path]:
    """Get list of files to upload, excluding patterns."""
    if exclude_patterns is None:
        exclude_patterns = [
            '__pycache__',
            '.git',
            '.gitignore',
            'venv',
            'env',
            '.venv',
            '.env',
            '.env.local',
            'mlruns',
            'logs',
            '*.log',
            '.DS_Store',
            'Thumbs.db',
            'data/raw',
            'data/output',
            'datasets',
            '*.tar.gz',
            '*.tar',
            '*.zip',
        ]
    
    files_to_upload = []
    
    # Files/directories to include
    include_patterns = [
        '*.py',
        '*.yaml',
        '*.yml',
        '*.md',
        '*.txt',
        '*.json',
        '*.html',
        '*.sh',
        '*.dockerfile',
        'Dockerfile',
        'docker-compose*.yml',
        'configs/',
        'src/',
        'scripts/',
        'models/checkpoints/*.pth',  # Include model checkpoints
    ]
    
    def should_include(file_path: Path) -> bool:
        """Check if file should be included."""
        rel_path = file_path.relative_to(project_root)
        rel_str = str(rel_path)
        
        # Check exclude patterns
        for pattern in exclude_patterns:
            if pattern in rel_str or file_path.name.startswith('.'):
                # Special handling for .gitkeep files
                if file_path.name == '.gitkeep':
                    return True
                return False
        
        # Check include patterns
        for pattern in include_patterns:
            if pattern.endswith('/'):
                # Directory pattern
                if rel_str.startswith(pattern[:-1]):
                    return True
            elif '*' in pattern:
                # Wildcard pattern
                import fnmatch
                if fnmatch.fnmatch(file_path.name, pattern):
                    return True
            else:
                # Exact match
                if file_path.name == pattern or rel_str == pattern:
                    return True
        
        return False
    
    # Walk through project directory
    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)
        
        # Filter directories to skip
        dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
        
        # Check files
        for file in files:
            file_path = root_path / file
            if should_include(file_path):
                files_to_upload.append(file_path)
    
    return sorted(files_to_upload)


def create_hf_readme(project_root: Path, repo_id: str) -> str:
    """Create README content for Hugging Face."""
    readme_content = f"""---
license: mit
tags:
- computer-vision
- object-detection
- soccer
- ball-detection
- detr
- rf-detr
- pytorch
datasets:
- custom
metrics:
- mAP
- precision
- recall
---

# Soccer Ball Detection with RF-DETR

Automated soccer ball detection pipeline using RF-DETR (Roboflow DETR) optimized for tiny object detection (<15 pixels).

## Model Details

### Architecture
- **Model**: RF-DETR Base
- **Backbone**: ResNet-50
- **Classes**: Ball (single class detection)
- **Input Resolution**: 1120x1120 (optimized for memory)
- **Precision**: Mixed Precision (FP16/FP32) training, FP16 inference

### Performance
Based on training evaluation report (Epoch 39):
- **mAP@0.5:0.95**: 0.682 (68.2%)
- **mAP@0.5**: 0.990 (99.0%)
- **Small Objects mAP**: 0.598 (59.8%)
- **Training Loss**: 3.073
- **Validation Loss**: 3.658

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```python
from src.perception.local_detector import LocalDetector
from pathlib import Path

# Initialize detector
detector = LocalDetector(
    model_path="models/checkpoints/latest_checkpoint.pth",
    config_path="configs/default.yaml"
)

# Detect ball in image
results = detector.detect(image_path)
```

### Process Video

```bash
python main.py --video path/to/video.mp4 --config configs/default.yaml --output data/output
```

## Training

### Train from Scratch

```bash
python scripts/train_ball.py \\
    --config configs/training.yaml \\
    --dataset-dir datasets/combined \\
    --output-dir models \\
    --epochs 50
```

### Resume Training

```bash
python scripts/train_ball.py \\
    --config configs/resume_20_epochs.yaml \\
    --dataset-dir datasets/combined \\
    --output-dir models \\
    --resume models/checkpoints/latest_checkpoint.pth \\
    --epochs 50
```

## Project Structure

```
soccer_cv_ball/
├── main.py                 # Main orchestrator
├── configs/               # Configuration files
├── src/
│   ├── perception/        # Detection, tracking
│   ├── analysis/         # Event detection
│   ├── visualization/    # Dashboard
│   └── training/         # Training utilities
├── scripts/               # Training and utility scripts
├── models/               # Model checkpoints
└── data/                 # Dataset (not included)
```

## Configuration

Key configuration files:
- `configs/training.yaml` - Main training configuration
- `configs/default.yaml` - Inference configuration
- `configs/resume_*.yaml` - Resume training configurations

## Datasets

This model was trained on:
- SoccerSynth-Detection (synthetic data)
- Open Soccer Ball Dataset
- Custom validation sets

## Precision Strategy

- **Training**: Mixed Precision (FP16/FP32) - RF-DETR default
- **Inference**: FP16 (half precision) for ~3x speedup
- **Future**: INT8 via QAT (Quantization-Aware Training) for edge devices

## Citation

If you use this model, please cite:

```bibtex
@software{{soccer_ball_detection,
  title={{Soccer Ball Detection with RF-DETR}},
  author={{Your Name}},
  year={{2026}},
  url={{https://huggingface.co/{repo_id}}}
}}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- RF-DETR by Roboflow
- SoccerSynth-Detection dataset
- Open Soccer Ball Dataset
"""
    return readme_content


def push_to_huggingface(
    repo_id: str,
    project_root: Path,
    token: Optional[str] = None,
    private: bool = False,
    include_models: bool = True
):
    """Push project to Hugging Face Hub using HTTP API."""
    
    project_root = Path(project_root).resolve()
    
    print(f"📦 Preparing to push to Hugging Face: {repo_id}")
    print(f"   Project root: {project_root}")
    
    # Initialize API
    api = HfApi(token=token)
    
    # Verify login
    try:
        user_info = api.whoami()
        print(f"🔐 Logged in as: {user_info.get('name', 'unknown')}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("   Please provide a valid token with --token argument")
        sys.exit(1)
    
    # Create repository
    print(f"📁 Creating repository: {repo_id}")
    try:
        api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
        print("✅ Repository created/exists")
    except HfHubHTTPError as e:
        if "already exists" in str(e).lower():
            print("✅ Repository already exists")
        else:
            print(f"⚠️  Repository creation warning: {e}")
    
    # Get files to upload
    print("🔍 Finding files to upload...")
    files_to_upload = get_files_to_upload(project_root)
    print(f"   Found {len(files_to_upload)} files")
    
    # Filter out model files if not including models
    if not include_models:
        files_to_upload = [f for f in files_to_upload if not str(f).endswith('.pth')]
        print(f"   Excluding model files: {len(files_to_upload)} files remaining")
    
    # Create README
    print("📝 Creating README.md...")
    readme_path = project_root / "README.md"
    readme_content = create_hf_readme(project_root, repo_id)
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print("✅ README.md created")
    
    # Upload files
    print("📤 Uploading files to Hugging Face Hub...")
    uploaded = 0
    failed = 0
    
    for file_path in files_to_upload:
        try:
            rel_path = file_path.relative_to(project_root)
            # Convert to forward slashes for HF
            path_in_repo = str(rel_path).replace('\\', '/')
            
            # Skip if it's the original README (we'll upload our generated one)
            if path_in_repo == "README.md" and file_path != readme_path:
                continue
            
            # Upload file
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            uploaded += 1
            if uploaded % 10 == 0:
                print(f"   Uploaded {uploaded}/{len(files_to_upload)} files...")
        except Exception as e:
            failed += 1
            print(f"   ⚠️  Failed to upload {file_path.name}: {e}")
    
    # Upload the generated README
    try:
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print("✅ README.md uploaded")
    except Exception as e:
        print(f"⚠️  Failed to upload README.md: {e}")
    
    print(f"\n✅ Upload complete!")
    print(f"   Uploaded: {uploaded} files")
    if failed > 0:
        print(f"   Failed: {failed} files")
    print(f"   View at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Push soccer ball detection project to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/soccer-ball-detection')"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or use huggingface-cli login)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository"
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Exclude model checkpoint files (.pth)"
    )
    
    args = parser.parse_args()
    
    push_to_huggingface(
        repo_id=args.repo_id,
        project_root=Path(args.project_root),
        token=args.token,
        private=args.private,
        include_models=not args.no_models
    )


if __name__ == "__main__":
    main()
