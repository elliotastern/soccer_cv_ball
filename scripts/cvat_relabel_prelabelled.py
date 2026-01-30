#!/usr/bin/env python3
"""
Upload prelabelled_balls (images + COCO JSON) to CVAT so you can relabel in the UI.
Creates a CVAT task, uploads images, imports pre-labels as COCO 1.0, prints task URL.
Requires: cvat-sdk, CVAT_URL / CVAT_USER / CVAT_PASS (or configs/auto_ingest.yaml).
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_cvat_config():
    """Load CVAT URL, user, password from config or env."""
    config = {
        "url": os.environ.get("CVAT_URL", "http://localhost:8080"),
        "username": os.environ.get("CVAT_USER", "admin"),
        "password": os.environ.get("CVAT_PASS", "admin"),
    }
    cfg_path = PROJECT_ROOT / "configs" / "auto_ingest.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}
        cvat = data.get("cvat", {})
        config["url"] = cvat.get("url", config["url"])
        config["username"] = cvat.get("username", config["username"])
        config["password"] = cvat.get("password", config["password"])
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Upload prelabelled_balls to CVAT for relabeling (creates task, uploads images, imports COCO)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "prelabelled_balls"),
        help="Directory with images and _annotations.coco.json",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="prelabelled_balls_relabel",
        help="CVAT task name",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    coco_path = input_dir / "_annotations.coco.json"
    if not coco_path.exists():
        print(f"Error: {coco_path} not found. Run prelabel_more_balls.py first.")
        sys.exit(1)

    try:
        from cvat_sdk import Client
        from cvat_sdk.api_client import models
    except ImportError:
        print("Error: cvat-sdk not installed. Run: pip install cvat-sdk")
        sys.exit(1)

    config = load_cvat_config()
    print(f"Connecting to CVAT at {config['url']}...")
    client = Client(url=config["url"])
    client.login((config["username"], config["password"]))

    # Image order must match COCO images[] so image_id lines up
    import json
    with open(coco_path) as f:
        coco = json.load(f)
    images = coco.get("images", [])
    if not images:
        print("Error: No images in _annotations.coco.json")
        sys.exit(1)
    file_names = [img["file_name"] for img in images]
    image_paths = [str(input_dir / fn) for fn in file_names]
    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        print(f"Error: Missing image files: {missing[:3]}{'...' if len(missing) > 3 else ''}")
        sys.exit(1)

    print(f"Creating task '{args.task_name}' with label 'ball'...")
    task_spec = models.TaskWriteRequest(
        name=args.task_name,
        labels=[models.LabelRequest(name="ball")],
        segment_size=len(image_paths),
    )
    task = client.tasks.create(spec=task_spec)
    print(f"Uploading {len(image_paths)} images...")
    client.tasks.create_data(task.id, image_paths)
    print("Waiting for CVAT to process images (10s)...")
    import time
    time.sleep(10)

    print("Importing COCO annotations...")
    client.tasks.import_annotations(
        id=task.id,
        format_name="COCO 1.0",
        filename=str(coco_path),
    )

    task_url = f"{config['url'].rstrip('/')}/tasks/{task.id}"
    print(f"Done. Open in CVAT to relabel: {task_url}")


if __name__ == "__main__":
    main()
