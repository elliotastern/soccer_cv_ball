#!/usr/bin/env python3
"""
Evaluate training progress and determine if we should continue training.
Checks metrics, loss trends, and overfitting signs.
"""

import torch
import yaml
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

def extract_metrics_from_log(log_path: str) -> Dict:
    """Extract metrics from training log."""
    with open(log_path, 'r') as f:
        log = f.read()
    
    metrics = {}
    
    # Extract mAP metrics
    map_pattern = r'Average Precision.*@\[ IoU=0\.50:0\.95.*area=\s+all.*maxDets=500 \] = ([\d.]+)'
    map50_pattern = r'Average Precision.*@\[ IoU=0\.50\s+\|.*area=\s+all.*maxDets=500 \] = ([\d.]+)'
    small_pattern = r'Average Precision.*@\[ IoU=0\.50:0\.95.*area= small.*maxDets=500 \] = ([\d.]+)'
    
    maps = [float(x) for x in re.findall(map_pattern, log)]
    map50s = [float(x) for x in re.findall(map50_pattern, log)]
    smalls = [float(x) for x in re.findall(small_pattern, log)]
    
    if maps:
        metrics['mAP_50_95'] = {
            'latest': maps[-1],
            'previous': maps[-2] if len(maps) > 1 else None,
            'improvement': maps[-1] - maps[0] if len(maps) > 0 else 0
        }
    
    if map50s:
        metrics['mAP_50'] = map50s[-1]
    
    if smalls:
        metrics['small_objects'] = {
            'latest': smalls[-1],
            'previous': smalls[-2] if len(smalls) > 1 else None,
            'improvement': smalls[-1] - smalls[0] if len(smalls) > 0 else 0
        }
    
    return metrics

def check_checkpoint(checkpoint_path: str) -> Dict:
    """Check checkpoint information."""
    if not Path(checkpoint_path).exists():
        return {'exists': False}
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return {
        'exists': True,
        'epoch': ckpt.get('epoch', 'N/A'),
        'size_mb': Path(checkpoint_path).stat().st_size / (1024 * 1024)
    }

def evaluate_progress(config_path: str, log_path: str = 'training_resume.log') -> Dict:
    """Evaluate training progress and make recommendation."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    current_epoch = config['checkpoint']['start_epoch']
    target_epochs = config['training']['epochs']
    
    # Extract metrics
    metrics = extract_metrics_from_log(log_path) if Path(log_path).exists() else {}
    
    # Check checkpoint
    checkpoint_info = check_checkpoint(config['checkpoint']['resume_from'])
    
    # Analyze progress
    analysis = {
        'current_epoch': current_epoch,
        'target_epochs': target_epochs,
        'progress_pct': (current_epoch / target_epochs * 100) if target_epochs > 0 else 0,
        'metrics': metrics,
        'checkpoint': checkpoint_info,
        'recommendation': None,
        'reasoning': []
    }
    
    # Make recommendation
    if current_epoch < target_epochs:
        analysis['recommendation'] = 'continue_to_target'
        analysis['reasoning'].append(f"Still training to target ({current_epoch}/{target_epochs})")
    elif metrics:
        # Check if still improving
        if 'mAP_50_95' in metrics and metrics['mAP_50_95']['previous']:
            improvement = metrics['mAP_50_95']['latest'] - metrics['mAP_50_95']['previous']
            if improvement > 0.001:  # Still improving
                analysis['recommendation'] = 'continue_training'
                analysis['reasoning'].append(f"mAP still improving (+{improvement:.4f})")
            elif improvement < -0.005:  # Declining (possible overfitting)
                analysis['recommendation'] = 'stop_training'
                analysis['reasoning'].append(f"mAP declining ({improvement:.4f}) - possible overfitting")
            else:
                analysis['recommendation'] = 'consider_extending'
                analysis['reasoning'].append("Metrics plateauing, but could improve with more epochs")
        
        # Check small objects (ball detection)
        if 'small_objects' in metrics:
            small_map = metrics['small_objects']['latest']
            if small_map < 0.70:  # Room for improvement
                analysis['recommendation'] = 'continue_training'
                analysis['reasoning'].append(f"Small objects mAP ({small_map:.3f}) has room for improvement")
    else:
        analysis['recommendation'] = 'evaluate_manually'
        analysis['reasoning'].append("Insufficient metrics data - evaluate manually")
    
    return analysis

def print_analysis(analysis: Dict):
    """Print formatted analysis."""
    print("=" * 70)
    print("TRAINING PROGRESS EVALUATION")
    print("=" * 70)
    print(f"\n📊 Current Status:")
    print(f"  Epoch: {analysis['current_epoch']}/{analysis['target_epochs']} ({analysis['progress_pct']:.1f}%)")
    
    if analysis['checkpoint']['exists']:
        print(f"  Checkpoint: Epoch {analysis['checkpoint']['epoch']}, {analysis['checkpoint']['size_mb']:.1f} MB")
    
    if analysis['metrics']:
        print(f"\n📈 Metrics:")
        if 'mAP_50_95' in analysis['metrics']:
            m = analysis['metrics']['mAP_50_95']
            print(f"  mAP@0.5:0.95: {m['latest']:.4f}")
            if m['previous']:
                change = m['latest'] - m['previous']
                print(f"    Change: {change:+.4f}")
        
        if 'mAP_50' in analysis['metrics']:
            print(f"  mAP@0.5: {analysis['metrics']['mAP_50']:.4f} ({analysis['metrics']['mAP_50']*100:.1f}%)")
        
        if 'small_objects' in analysis['metrics']:
            s = analysis['metrics']['small_objects']
            print(f"  Small Objects (Ball): {s['latest']:.4f}")
            if s['previous']:
                change = s['latest'] - s['previous']
                print(f"    Change: {change:+.4f}")
    
    print(f"\n💡 Recommendation: {analysis['recommendation'].upper().replace('_', ' ')}")
    print(f"\n📝 Reasoning:")
    for reason in analysis['reasoning']:
        print(f"  • {reason}")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/resume_20_epochs_low_memory.yaml'
    log_path = sys.argv[2] if len(sys.argv) > 2 else 'training_resume.log'
    
    analysis = evaluate_progress(config_path, log_path)
    print_analysis(analysis)
