#!/usr/bin/env python3
"""
Create a comprehensive visualization for the open soccer ball dataset.
Generates a master HTML page that shows all splits with navigation.
"""
import json
import base64
from pathlib import Path
from typing import List, Dict
from PIL import Image
import io


def load_coco_annotations(annotation_path: str) -> Dict:
    """Load COCO format annotation file."""
    with open(annotation_path, 'r') as f:
        return json.load(f)


def get_images_with_balls(coco_data: Dict) -> List[Dict]:
    """Get images that have ball annotations."""
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Find ball category ID
    ball_category_id = None
    for cat_id, cat_name in categories.items():
        if cat_name.lower() == 'ball':
            ball_category_id = cat_id
            break
    
    if ball_category_id is None:
        raise ValueError("Ball category not found in annotations")
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] == ball_category_id:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann['bbox'])
    
    # Get images with balls
    images = {img['id']: img for img in coco_data['images']}
    images_with_balls = []
    
    for img_id in sorted(image_annotations.keys()):
        if img_id in images:
            images_with_balls.append({
                'image': images[img_id],
                'bboxes': image_annotations[img_id]
            })
    
    return images_with_balls


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 string."""
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
            img = Image.open(io.BytesIO(img_data))
            # Resize if too large (max 1920px width)
            max_width = 1920
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return ""


def generate_master_html(all_splits_data: Dict[str, List[Dict]], output_path: Path):
    """Generate master HTML with all splits."""
    
    total_images = sum(len(data['samples']) for data in all_splits_data.values())
    total_available = sum(data['total'] for data in all_splits_data.values())
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Open Soccer Ball Dataset - Complete Visualization</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        .header h1 {{
            color: white;
            margin-bottom: 15px;
            font-size: 2.5em;
        }}
        
        .header p {{
            color: rgba(255, 255, 255, 0.9);
            font-size: 16px;
            margin: 5px 0;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }}
        
        .stat-card h3 {{
            color: #FFC107;
            font-size: 2em;
            margin-bottom: 5px;
        }}
        
        .stat-card p {{
            color: #b0b0b0;
            font-size: 14px;
        }}
        
        .splits-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .split-section {{
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }}
        
        .split-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #3a3a3a;
        }}
        
        .split-header h2 {{
            color: #FFC107;
            font-size: 1.5em;
        }}
        
        .split-info {{
            color: #b0b0b0;
            font-size: 14px;
        }}
        
        .controls {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .toggle-btn {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}
        
        .toggle-btn:hover {{
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }}
        
        .toggle-btn.off {{
            background: #ff9800;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }}
        
        .frame-container {{
            background: #1a1a1a;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s;
        }}
        
        .frame-container:hover {{
            transform: translateY(-4px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }}
        
        .image-wrapper {{
            position: relative;
            width: 100%;
            margin-bottom: 8px;
            border-radius: 6px;
            overflow: hidden;
            background: #1a1a1a;
        }}
        
        .image-wrapper img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        .bbox-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}
        
        .bbox-overlay.hidden {{
            display: none;
        }}
        
        .bbox {{
            position: absolute;
            border: 3px solid #FFC107;
            background: rgba(255, 193, 7, 0.3);
            box-sizing: border-box;
        }}
        
        .bbox-label {{
            position: absolute;
            top: -20px;
            left: 0;
            background: rgba(0, 0, 0, 0.8);
            color: #FFC107;
            padding: 2px 6px;
            font-size: 11px;
            font-weight: bold;
            border-radius: 3px;
            white-space: nowrap;
        }}
        
        .frame-info {{
            color: #b0b0b0;
            font-size: 12px;
        }}
        
        .frame-info strong {{
            color: #FFC107;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚽ Open Soccer Ball Dataset - Complete Visualization</h1>
        <p><strong>Dataset:</strong> Football and Player Detection (Roboflow)</p>
        <p><strong>License:</strong> CC BY 4.0</p>
        <p><strong>Total Samples:</strong> {total_images} frames | <strong>Total Available:</strong> {total_available} frames with balls</p>
    </div>
    
    <div class="stats-grid">
"""
    
    for split_name, data in all_splits_data.items():
        html_content += f"""
        <div class="stat-card">
            <h3>{len(data['samples'])}</h3>
            <p>{split_name.upper()} Split<br/>({data['total']} total available)</p>
        </div>
"""
    
    html_content += """
    </div>
    
    <div class="controls">
        <button class="toggle-btn" id="toggleBtn" onclick="toggleAllBoxes()">Hide All Boxes</button>
        <div class="stats" style="color: #b0b0b0; font-size: 14px;">
            Use the toggle button or press 'H' to show/hide bounding boxes
        </div>
    </div>
    
    <div class="splits-container">
"""
    
    for split_name, data in all_splits_data.items():
        samples = data['samples']
        annotation_file = data['annotation_file']
        
        html_content += f"""
        <div class="split-section">
            <div class="split-header">
                <h2>{split_name.upper()} Split</h2>
                <div class="split-info">{len(samples)} samples</div>
            </div>
            <div class="grid" id="grid-{split_name}">
"""
        
        for idx, img_data in enumerate(samples):
            image_info = img_data['image']
            bboxes = img_data['bboxes']
            
            # Get image path
            annotation_dir = Path(annotation_file).parent
            image_path = annotation_dir / image_info['file_name']
            
            # Try alternative paths if image not found
            if not image_path.exists():
                images_dir = annotation_dir / 'images'
                if images_dir.exists():
                    image_path = images_dir / image_info['file_name']
            
            if not image_path.exists():
                continue
            
            # Convert image to base64
            img_base64 = image_to_base64(image_path)
            if not img_base64:
                continue
            
            # Calculate bbox positions
            img_width = image_info['width']
            img_height = image_info['height']
            
            bbox_html = ""
            for bbox in bboxes:
                x, y, w, h = bbox
                x_percent = (x / img_width) * 100
                y_percent = (y / img_height) * 100
                w_percent = (w / img_width) * 100
                h_percent = (h / img_height) * 100
                
                bbox_html += f"""
                <div class="bbox" style="left: {x_percent}%; top: {y_percent}%; width: {w_percent}%; height: {h_percent}%;">
                    <div class="bbox-label">ball</div>
                </div>"""
            
            html_content += f"""
            <div class="frame-container">
                <div class="image-wrapper">
                    <img src="{img_base64}" alt="{split_name} Frame {idx + 1}">
                    <div class="bbox-overlay" id="overlay-{split_name}-{idx}">
                        {bbox_html}
                    </div>
                </div>
                <div class="frame-info">
                    <strong>{split_name}</strong> Frame {idx + 1}: {image_info['file_name']}<br/>
                    <strong>{len(bboxes)}</strong> ball(s) | {img_width}x{img_height}
                </div>
            </div>
"""
        
        html_content += """
            </div>
        </div>
"""
    
    html_content += """
    </div>
    
    <script>
        let boxesVisible = true;
        
        function toggleAllBoxes() {
            boxesVisible = !boxesVisible;
            const overlays = document.querySelectorAll('.bbox-overlay');
            const btn = document.getElementById('toggleBtn');
            
            overlays.forEach(overlay => {
                if (boxesVisible) {
                    overlay.classList.remove('hidden');
                } else {
                    overlay.classList.add('hidden');
                }
            });
            
            if (boxesVisible) {
                btn.textContent = 'Hide All Boxes';
                btn.classList.remove('off');
            } else {
                btn.textContent = 'Show All Boxes';
                btn.classList.add('off');
            }
        }
        
        // Keyboard shortcut: 'H' to toggle
        document.addEventListener('keydown', function(event) {
            if (event.key === 'h' || event.key === 'H') {
                toggleAllBoxes();
            }
        });
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Generated master HTML: {output_path}")


def main():
    """Main function to create comprehensive visualization."""
    base_dir = Path("/workspace/soccer_coach_cv/data/raw/real_data/Validation images OFFICIAL")
    
    splits = {
        'train': base_dir / "train" / "_annotations.coco.json",
        'valid': base_dir / "valid" / "_annotations.coco.json",
        'test': base_dir / "test" / "_annotations.coco.json",
    }
    
    all_splits_data = {}
    
    for split_name, annotation_file in splits.items():
        if not annotation_file.exists():
            print(f"⚠️  Skipping {split_name}: {annotation_file} not found")
            continue
        
        print(f"📋 Loading {split_name} split...")
        coco_data = load_coco_annotations(str(annotation_file))
        
        print(f"🔍 Finding images with ball annotations in {split_name}...")
        images_with_balls = get_images_with_balls(coco_data)
        
        print(f"📊 Found {len(images_with_balls)} images with ball annotations in {split_name}")
        
        # Select up to 100 samples per split
        samples = images_with_balls[:100]
        print(f"✅ Selected {len(samples)} samples from {split_name} split")
        
        all_splits_data[split_name] = {
            'samples': samples,
            'total': len(images_with_balls),
            'annotation_file': str(annotation_file)
        }
    
    # Generate master HTML
    output_path = base_dir / "open_soccer_ball_dataset_visualization.html"
    print(f"\n🎨 Generating master HTML visualization...")
    generate_master_html(all_splits_data, output_path)
    
    print(f"\n✅ Done! Open {output_path} in your browser to view the complete visualization.")


if __name__ == "__main__":
    main()
