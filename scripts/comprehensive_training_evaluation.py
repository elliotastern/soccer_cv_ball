#!/usr/bin/env python3
"""
Comprehensive Training Evaluation Script
Analyzes training progress, detects plateaus, overfitting, and provides recommendations.
"""

import re
import json
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

def extract_metrics_from_logs(log_paths: List[str]) -> Dict[int, Dict]:
    """Extract comprehensive metrics from training logs."""
    all_metrics = {}
    
    for log_path in log_paths:
        if not Path(log_path).exists():
            continue
        
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract epoch evaluation metrics
        epoch_pattern = r'Epoch: \[(\d+)\].*?Average Precision.*?@\[ IoU=0\.50:0\.95.*?area=\s+all.*?maxDets=500 \] = ([\d.]+).*?Average Precision.*?@\[ IoU=0\.50\s+\|.*?area=\s+all.*?maxDets=500 \] = ([\d.]+).*?Average Precision.*?@\[ IoU=0\.50:0\.95.*?area= small.*?maxDets=500 \] = ([\d.]+)'
        
        matches = re.finditer(epoch_pattern, content, re.DOTALL)
        for match in matches:
            epoch = int(match.group(1))
            map_50_95 = float(match.group(2))
            map_50 = float(match.group(3))
            map_small = float(match.group(4))
            
            if epoch not in all_metrics:
                all_metrics[epoch] = {}
            all_metrics[epoch].update({
                'map_50_95': map_50_95,
                'map_50': map_50,
                'map_small': map_small
            })
        
        # Extract training loss per epoch
        for epoch in all_metrics.keys():
            train_pattern = rf'Epoch: \[{epoch}\].*?Total time:.*?Averaged stats:.*?loss: ([\d.]+) \('
            train_match = re.search(train_pattern, content, re.DOTALL)
            if train_match:
                all_metrics[epoch]['train_loss'] = float(train_match.group(1))
            
            # Extract validation loss
            val_pattern = rf'Epoch: \[{epoch}\].*?Test: Total time:.*?Averaged stats:.*?loss: ([\d.]+) \('
            val_match = re.search(val_pattern, content, re.DOTALL)
            if val_match:
                all_metrics[epoch]['val_loss'] = float(val_match.group(1))
    
    return all_metrics

def analyze_trends(metrics: Dict[int, Dict]) -> Dict:
    """Analyze training trends and detect issues."""
    if len(metrics) < 3:
        return {'error': 'Insufficient data for analysis'}
    
    epochs = sorted(metrics.keys())
    analysis = {
        'total_epochs': len(epochs),
        'epoch_range': (epochs[0], epochs[-1]),
        'overall_improvement': {},
        'recent_trend': {},
        'improvement_rate': {},
        'plateau_detected': False,
        'plateau_epochs': [],
        'overfitting_detected': False,
        'overfitting_evidence': [],
        'recommendations': []
    }
    
    # Overall improvement
    first_epoch = epochs[0]
    last_epoch = epochs[-1]
    analysis['overall_improvement'] = {
        'map_50_95': metrics[last_epoch]['map_50_95'] - metrics[first_epoch]['map_50_95'],
        'map_small': metrics[last_epoch]['map_small'] - metrics[first_epoch]['map_small']
    }
    
    # Recent trend (last 5 epochs)
    if len(epochs) >= 5:
        recent = epochs[-5:]
        analysis['recent_trend'] = {
            'map_50_95_change': metrics[recent[-1]]['map_50_95'] - metrics[recent[0]]['map_50_95'],
            'map_small_change': metrics[recent[-1]]['map_small'] - metrics[recent[0]]['map_small'],
            'epochs': recent
        }
    
    # Improvement rate (per epoch)
    if len(epochs) > 1:
        analysis['improvement_rate'] = {
            'map_50_95_per_epoch': analysis['overall_improvement']['map_50_95'] / (len(epochs) - 1),
            'map_small_per_epoch': analysis['overall_improvement']['map_small'] / (len(epochs) - 1)
        }
    
    # Plateau detection (3+ consecutive epochs with <0.001 improvement)
    plateau_count = 0
    plateau_start = None
    for i in range(len(epochs) - 1):
        change = metrics[epochs[i+1]]['map_50_95'] - metrics[epochs[i]]['map_50_95']
        if abs(change) < 0.001:
            if plateau_start is None:
                plateau_start = epochs[i]
            plateau_count += 1
        else:
            if plateau_count >= 3:
                analysis['plateau_detected'] = True
                analysis['plateau_epochs'] = list(range(plateau_start, epochs[i] + 1))
            plateau_count = 0
            plateau_start = None
    
    if plateau_count >= 3:
        analysis['plateau_detected'] = True
        analysis['plateau_epochs'] = list(range(plateau_start, epochs[-1] + 1))
    
    # Overfitting detection
    train_losses = [(e, metrics[e]['train_loss']) for e in epochs if 'train_loss' in metrics[e]]
    val_losses = [(e, metrics[e]['val_loss']) for e in epochs if 'val_loss' in metrics[e]]
    
    if len(train_losses) >= 3 and len(val_losses) >= 3:
        # Check last 3 epochs
        recent_train = train_losses[-3:]
        recent_val = val_losses[-3:]
        
        train_trend = recent_train[-1][1] - recent_train[0][1]  # Negative = decreasing
        val_trend = recent_val[-1][1] - recent_val[0][1]  # Positive = increasing
        
        if train_trend < -0.01 and val_trend > 0.01:  # Train decreasing, val increasing
            analysis['overfitting_detected'] = True
            analysis['overfitting_evidence'] = [
                f"Train loss decreased by {abs(train_trend):.4f}",
                f"Val loss increased by {val_trend:.4f}"
            ]
    
    # Generate recommendations
    if analysis['overfitting_detected']:
        analysis['recommendations'].append("OVERFITTING: Reduce learning rate or add regularization")
        analysis['recommendations'].append("Consider early stopping or learning rate decay")
    
    if analysis['plateau_detected']:
        analysis['recommendations'].append("PLATEAU: Consider learning rate decay or augmentation")
        analysis['recommendations'].append("Small objects may benefit from higher resolution or multi-scale training")
    
    if metrics[last_epoch]['map_small'] < 0.65:
        analysis['recommendations'].append("Small objects mAP < 0.65: Focus on small object augmentation")
        analysis['recommendations'].append("Consider increasing resolution or enabling multi-scale training")
    
    if analysis['improvement_rate'].get('map_50_95_per_epoch', 0) > 0.001:
        analysis['recommendations'].append("Metrics still improving: Continue training to 50-60 epochs")
    
    if analysis['recent_trend'].get('map_50_95_change', 0) < -0.005:
        analysis['recommendations'].append("Recent decline detected: Consider reducing learning rate")
    
    return analysis

def check_config(config_path: str) -> Dict:
    """Check training configuration."""
    if not Path(config_path).exists():
        return {'error': f'Config file not found: {config_path}'}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config.get('training', {})
    dataset_config = config.get('dataset', {})
    
    config_analysis = {
        'learning_rate': training_config.get('learning_rate', 'unknown'),
        'lr_schedule': 'Not specified (RF-DETR may have defaults)',
        'batch_size': training_config.get('batch_size', 'unknown'),
        'resolution': training_config.get('resolution', 'unknown'),
        'multi_scale': training_config.get('multi_scale', False),
        'augmentation': 'Not specified in config (RF-DETR defaults unknown)',
        'grad_accum_steps': training_config.get('grad_accum_steps', 'unknown'),
        'issues': []
    }
    
    # Check for potential issues
    if training_config.get('learning_rate', 0) > 0.0003:
        config_analysis['issues'].append('Learning rate may be too high (>0.0003)')
    
    if training_config.get('batch_size', 0) < 4:
        config_analysis['issues'].append('Very small batch size may limit gradient quality')
    
    if not training_config.get('multi_scale', False):
        config_analysis['issues'].append('Multi-scale training disabled (may help small objects)')
    
    if training_config.get('resolution', 1288) < 1200:
        config_analysis['issues'].append('Resolution reduced (may hurt small object detection)')
    
    return config_analysis

def generate_recommendations(metrics: Dict[int, Dict], analysis: Dict, config_analysis: Dict) -> List[str]:
    """Generate specific recommendations based on analysis."""
    recommendations = []
    
    epochs = sorted(metrics.keys())
    last_epoch = epochs[-1]
    current_map = metrics[last_epoch]['map_50_95']
    current_small = metrics[last_epoch]['map_small']
    
    # Scenario 1: Still improving
    if (analysis['improvement_rate'].get('map_50_95_per_epoch', 0) > 0.001 and 
        current_small < 0.70):
        recommendations.append("✅ CONTINUE TRAINING: Metrics still improving and small objects < 0.70")
        recommendations.append("   → Continue to 50-60 epochs with current settings")
    
    # Scenario 2: Plateauing
    if analysis['plateau_detected']:
        recommendations.append("⚠️ PLATEAU DETECTED: Metrics not improving")
        recommendations.append("   → Consider learning rate decay (reduce LR by 0.5x)")
        recommendations.append("   → Enable multi-scale training if memory allows")
        recommendations.append("   → Increase resolution if memory allows")
        recommendations.append("   → Add/strengthen augmentation")
    
    # Scenario 3: Overfitting
    if analysis['overfitting_detected']:
        recommendations.append("❌ OVERFITTING DETECTED: Validation loss increasing")
        recommendations.append("   → Reduce learning rate by 0.5x or 0.1x")
        recommendations.append("   → Consider early stopping")
        recommendations.append("   → Add regularization (weight decay, dropout)")
    
    # Scenario 4: Small objects need improvement
    if current_small < 0.65 and current_map > 0.68:
        recommendations.append("🎯 SMALL OBJECTS NEED IMPROVEMENT")
        recommendations.append("   → Focus on small object augmentation (copy-paste, mosaic)")
        recommendations.append("   → Increase resolution if possible")
        recommendations.append("   → Enable multi-scale training")
        recommendations.append("   → Consider adjusting loss weights for small objects")
    
    # Configuration-specific recommendations
    if config_analysis.get('learning_rate', 0) == 0.0002 and analysis['plateau_detected']:
        recommendations.append("⚙️ CONFIG CHANGE: Add learning rate schedule")
        recommendations.append("   → Use step decay or cosine annealing")
        recommendations.append("   → Reduce LR by 0.5x at epoch 40, 0.1x at epoch 50")
    
    if not config_analysis.get('multi_scale', False) and current_small < 0.65:
        recommendations.append("⚙️ CONFIG CHANGE: Enable multi-scale training")
        recommendations.append("   → May help with small object detection")
        recommendations.append("   → Monitor memory usage")
    
    return recommendations

def print_evaluation_report(metrics: Dict[int, Dict], analysis: Dict, config_analysis: Dict, recommendations: List[str]):
    """Print comprehensive evaluation report."""
    print("=" * 70)
    print("COMPREHENSIVE TRAINING EVALUATION REPORT")
    print("=" * 70)
    
    epochs = sorted(metrics.keys())
    last_epoch = epochs[-1]
    
    print(f"\n📊 CURRENT STATUS")
    print(f"  Epochs analyzed: {len(epochs)} (Epoch {epochs[0]} to {epochs[-1]})")
    print(f"  Latest Epoch: {last_epoch}")
    print(f"  Latest mAP@0.5:0.95: {metrics[last_epoch]['map_50_95']:.4f}")
    print(f"  Latest mAP@0.5: {metrics[last_epoch]['map_50']:.4f} ({metrics[last_epoch]['map_50']*100:.1f}%)")
    print(f"  Latest Small Objects mAP: {metrics[last_epoch]['map_small']:.4f}")
    
    if 'train_loss' in metrics[last_epoch]:
        print(f"  Latest Training Loss: {metrics[last_epoch]['train_loss']:.4f}")
    if 'val_loss' in metrics[last_epoch]:
        print(f"  Latest Validation Loss: {metrics[last_epoch]['val_loss']:.4f}")
    
    print(f"\n📈 OVERALL PROGRESSION")
    print(f"  mAP@0.5:0.95 improvement: {analysis['overall_improvement']['map_50_95']:+.4f}")
    print(f"  Small objects improvement: {analysis['overall_improvement']['map_small']:+.4f}")
    print(f"  Improvement rate (per epoch):")
    print(f"    mAP@0.5:0.95: {analysis['improvement_rate'].get('map_50_95_per_epoch', 0):+.4f}")
    print(f"    Small objects: {analysis['improvement_rate'].get('map_small_per_epoch', 0):+.4f}")
    
    if 'recent_trend' in analysis and analysis['recent_trend']:
        print(f"\n📉 RECENT TREND (Last 5 epochs)")
        print(f"  mAP@0.5:0.95 change: {analysis['recent_trend']['map_50_95_change']:+.4f}")
        print(f"  Small objects change: {analysis['recent_trend']['map_small_change']:+.4f}")
    
    print(f"\n⚠️ ISSUES DETECTED")
    print(f"  Plateau: {analysis['plateau_detected']}")
    if analysis['plateau_epochs']:
        print(f"    Plateau epochs: {analysis['plateau_epochs']}")
    print(f"  Overfitting: {analysis['overfitting_detected']}")
    if analysis['overfitting_evidence']:
        for evidence in analysis['overfitting_evidence']:
            print(f"    {evidence}")
    
    print(f"\n⚙️ CONFIGURATION")
    print(f"  Learning Rate: {config_analysis.get('learning_rate', 'unknown')}")
    print(f"  LR Schedule: {config_analysis.get('lr_schedule', 'unknown')}")
    print(f"  Batch Size: {config_analysis.get('batch_size', 'unknown')}")
    print(f"  Resolution: {config_analysis.get('resolution', 'unknown')}")
    print(f"  Multi-scale: {config_analysis.get('multi_scale', False)}")
    if config_analysis.get('issues'):
        print(f"  Potential Issues:")
        for issue in config_analysis['issues']:
            print(f"    - {issue}")
    
    print(f"\n💡 RECOMMENDATIONS")
    if recommendations:
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("  No specific recommendations - training appears healthy")
    
    print("\n" + "=" * 70)

def main():
    import sys
    
    # Default paths
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/resume_20_epochs_low_memory.yaml'
    log_paths = [
        'training_resume.log',
        'training_resume_epoch39_final.log'
    ]
    
    # Extract metrics
    print("Extracting metrics from training logs...")
    metrics = extract_metrics_from_logs(log_paths)
    
    if not metrics:
        print("ERROR: No metrics found in logs")
        return
    
    # Analyze trends
    print("Analyzing training trends...")
    analysis = analyze_trends(metrics)
    
    # Check configuration
    print("Checking training configuration...")
    config_analysis = check_config(config_path)
    
    # Generate recommendations
    recommendations = generate_recommendations(metrics, analysis, config_analysis)
    
    # Print report
    print_evaluation_report(metrics, analysis, config_analysis, recommendations)
    
    # Save detailed results
    output = {
        'metrics': {str(k): v for k, v in sorted(metrics.items())},
        'analysis': analysis,
        'config_analysis': config_analysis,
        'recommendations': recommendations
    }
    
    output_path = 'training_evaluation_detailed.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Detailed results saved to: {output_path}")

if __name__ == '__main__':
    main()
