"""
Comprehensive Foundation Model Ablation Study
==============================================
Compares multiple architectures and backbones:

1. ResNet50 (ImageNet pretrained) - BASELINE
2. ResNet50 (trained from scratch)
3. CNN + DINOv2 Hybrid
4. CNN + DINOv3 Hybrid - STATE-OF-THE-ART

For each configuration:
- RGB (3 channels)
- Luminance (3 channels)  
- Chrominance (7 channels)
- All features (10 channels)

Usage:
    # Run full ablation (takes ~4-5 days)
    python run_foundation_model_ablation.py --all

    # Test specific backbone
    python run_foundation_model_ablation.py --backbone dinov3 --feature_config all
    
    # Quick test
    python run_foundation_model_ablation.py --backbone dinov3 --feature_config rgb --epochs 10
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import argparse
import time
import sys

sys.path.append('src/training')
from train_segmentation_models import train_model as train_baseline
from train_hybrid_dinov3 import train_hybrid_model


def run_baseline_training(
    model_type='deeplabv3plus',
    feature_config='all',
    use_pretrained=True,
    epochs=100,
    batch_size=4,
    output_dir='experiments/results/foundation_ablation'
):
    """Run baseline ResNet50 training"""
    
    print("\n" + "="*70)
    print(f"BASELINE: {model_type.upper()} - {feature_config.upper()}")
    print(f"Pretrained: {use_pretrained}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        model, history, metrics = train_baseline(
            model_type=model_type,
            feature_config=feature_config,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=1e-4 if use_pretrained else 1e-3,
            output_dir=f"{output_dir}/resnet50_{'pretrained' if use_pretrained else 'scratch'}"
        )
        
        elapsed_time = time.time() - start_time
        metrics['time_hours'] = elapsed_time / 3600
        metrics['backbone'] = f"resnet50_{'pretrained' if use_pretrained else 'scratch'}"
        metrics['status'] = 'success'
        
        return metrics
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        elapsed_time = time.time() - start_time
        return {
            'backbone': f"resnet50_{'pretrained' if use_pretrained else 'scratch'}",
            'feature_config': feature_config,
            'status': 'failed',
            'time_hours': elapsed_time / 3600,
            'error': str(e)
        }


def run_hybrid_training(
    backbone='dinov3',
    dinov3_size='base',
    feature_config='all',
    epochs=100,
    batch_size=4,
    output_dir='experiments/results/foundation_ablation'
):
    """Run hybrid CNN-DINOv3 training"""
    
    print("\n" + "="*70)
    print(f"HYBRID: CNN-{backbone.upper()} - {feature_config.upper()}")
    print(f"DINOv3 size: {dinov3_size}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        model, history, metrics = train_hybrid_model(
            feature_config=feature_config,
            dinov3_size=dinov3_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=1e-4,
            freeze_dinov3=True,
            output_dir=f"{output_dir}/hybrid_{backbone}_{dinov3_size}"
        )
        
        elapsed_time = time.time() - start_time
        metrics['time_hours'] = elapsed_time / 3600
        metrics['backbone'] = f"hybrid_{backbone}_{dinov3_size}"
        metrics['status'] = 'success'
        
        return metrics
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        elapsed_time = time.time() - start_time
        return {
            'backbone': f"hybrid_{backbone}_{dinov3_size}",
            'feature_config': feature_config,
            'status': 'failed',
            'time_hours': elapsed_time / 3600,
            'error': str(e)
        }


def run_comprehensive_ablation(
    backbones=['resnet50_pretrained', 'hybrid_dinov3_base'],
    feature_configs=['rgb', 'all'],
    epochs=100,
    batch_size=4,
    output_dir='experiments/results/foundation_ablation',
    skip_existing=True
):
    """
    Run comprehensive ablation study across all configurations.
    
    Args:
        backbones: List of backbones to test
        feature_configs: List of feature configs to test
        epochs: Number of epochs per config
        batch_size: Batch size
        output_dir: Output directory
        skip_existing: Skip configs that already have results
    """
    
    print("="*70)
    print("COMPREHENSIVE FOUNDATION MODEL ABLATION STUDY")
    print("="*70)
    print(f"\nBackbones: {backbones}")
    print(f"Feature configs: {feature_configs}")
    print(f"Epochs per config: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    
    results = []
    total_configs = len(backbones) * len(feature_configs)
    current_config = 0
    
    for backbone in backbones:
        for feature_config in feature_configs:
            current_config += 1
            
            print(f"\n{'='*70}")
            print(f"Configuration {current_config}/{total_configs}")
            print(f"Backbone: {backbone} | Features: {feature_config}")
            print(f"{'='*70}")
            
            # Check if results exist
            if backbone.startswith('resnet50'):
                results_path = Path(output_dir) / backbone / feature_config / 'final_metrics.json'
            else:
                results_path = Path(output_dir) / backbone / feature_config / 'final_metrics.json'
            
            if skip_existing and results_path.exists():
                print(f"✓ Results already exist, loading from {results_path}")
                with open(results_path, 'r') as f:
                    metrics = json.load(f)
                results.append(metrics)
                continue
            
            # Run training based on backbone type
            if backbone == 'resnet50_pretrained':
                metrics = run_baseline_training(
                    model_type='deeplabv3plus',
                    feature_config=feature_config,
                    use_pretrained=True,
                    epochs=epochs,
                    batch_size=batch_size,
                    output_dir=output_dir
                )
            elif backbone == 'resnet50_scratch':
                metrics = run_baseline_training(
                    model_type='deeplabv3plus',
                    feature_config=feature_config,
                    use_pretrained=False,
                    epochs=epochs,
                    batch_size=batch_size,
                    output_dir=output_dir
                )
            elif backbone.startswith('hybrid_dinov3'):
                # Extract size from backbone name (e.g., 'hybrid_dinov3_base' -> 'base')
                size = backbone.split('_')[-1]
                metrics = run_hybrid_training(
                    backbone='dinov3',
                    dinov3_size=size,
                    feature_config=feature_config,
                    epochs=epochs,
                    batch_size=batch_size,
                    output_dir=output_dir
                )
            elif backbone.startswith('hybrid_dinov2'):
                size = backbone.split('_')[-1]
                metrics = run_hybrid_training(
                    backbone='dinov2',
                    dinov3_size=size,  # Note: reusing same parameter
                    feature_config=feature_config,
                    epochs=epochs,
                    batch_size=batch_size,
                    output_dir=output_dir
                )
            else:
                print(f"✗ Unknown backbone: {backbone}")
                continue
            
            results.append(metrics)
    
    # Create comprehensive comparison
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        'backbone', 'feature_config', 'n_channels',
        'best_val_dice', 'best_val_iou',
        'total_params', 'time_hours', 'status'
    ]
    
    df = df[[col for col in column_order if col in df.columns]]
    
    # Sort by best_val_dice descending
    if 'best_val_dice' in df.columns:
        df = df.sort_values('best_val_dice', ascending=False, na_position='last')
    
    print("\n" + df.to_string(index=False))
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / 'foundation_model_ablation_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")
    
    # Save as JSON
    json_path = output_path / 'foundation_model_ablation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {json_path}")
    
    # Generate summary
    summary = generate_summary(df)
    
    summary_path = output_path / 'foundation_model_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"✓ Summary saved to {summary_path}")
    
    print("\n" + summary)
    
    return df


def generate_summary(df):
    """Generate comprehensive summary for paper"""
    
    summary = f"""
{'='*70}
FOUNDATION MODEL ABLATION STUDY SUMMARY
{'='*70}

Table: Comprehensive Backbone Comparison

| Backbone | Config | Channels | Dice ↑ | IoU ↑ | Params | Time (h) |
|----------|--------|----------|--------|-------|---------|----------|
"""
    
    for _, row in df.iterrows():
        backbone = row['backbone']
        config = row['feature_config']
        channels = row.get('n_channels', 'N/A')
        dice = row.get('best_val_dice', 0.0)
        iou = row.get('best_val_iou', 0.0)
        params = row.get('total_params', 0) / 1e6
        time_h = row.get('time_hours', 0.0)
        
        summary += f"| {backbone:20s} | {config:6s} | {channels:8} | {dice:.4f} | {iou:.4f} | {params:6.1f}M | {time_h:8.2f} |\n"
    
    summary += "\n" + "="*70 + "\n"
    summary += "KEY FINDINGS:\n\n"
    
    # Find best overall
    if 'best_val_dice' in df.columns and df['best_val_dice'].notna().any():
        best_idx = df['best_val_dice'].idxmax()
        best = df.loc[best_idx]
        
        summary += f"1. BEST OVERALL: {best['backbone']} - {best['feature_config'].upper()}\n"
        summary += f"   Dice: {best['best_val_dice']:.4f}\n"
        summary += f"   IoU: {best['best_val_iou']:.4f}\n\n"
        
        # Compare with baseline
        baseline = df[df['backbone'].str.contains('resnet50_pretrained')].copy()
        if not baseline.empty:
            rgb_baseline = baseline[baseline['feature_config'] == 'rgb']
            if not rgb_baseline.empty:
                baseline_dice = rgb_baseline['best_val_dice'].iloc[0]
                improvement = ((best['best_val_dice'] - baseline_dice) / baseline_dice) * 100
                
                summary += f"2. IMPROVEMENT OVER BASELINE:\n"
                summary += f"   Baseline (ResNet50 + RGB): {baseline_dice:.4f}\n"
                summary += f"   Best model: {best['best_val_dice']:.4f}\n"
                summary += f"   Improvement: +{improvement:.2f}%\n\n"
        
        # Compare RGB vs All for best backbone
        best_backbone = best['backbone']
        backbone_results = df[df['backbone'] == best_backbone]
        
        if not backbone_results.empty:
            rgb_row = backbone_results[backbone_results['feature_config'] == 'rgb']
            all_row = backbone_results[backbone_results['feature_config'] == 'all']
            
            if not rgb_row.empty and not all_row.empty:
                rgb_dice = rgb_row['best_val_dice'].iloc[0]
                all_dice = all_row['best_val_dice'].iloc[0]
                
                summary += f"3. RGB vs ALL FEATURES ({best_backbone}):\n"
                summary += f"   RGB (3ch): {rgb_dice:.4f}\n"
                summary += f"   All (10ch): {all_dice:.4f}\n"
                
                if all_dice > rgb_dice:
                    improvement = ((all_dice - rgb_dice) / rgb_dice) * 100
                    summary += f"   ✓ All features WIN by +{improvement:.2f}%\n"
                    summary += f"   → Multi-channel approach SUCCESSFUL!\n\n"
                else:
                    decline = ((rgb_dice - all_dice) / all_dice) * 100
                    summary += f"   ✗ RGB still wins by +{decline:.2f}%\n"
                    summary += f"   → Further investigation needed\n\n"
    
    # Architecture comparison
    summary += "4. ARCHITECTURE RANKING:\n"
    
    # Group by backbone and average performance
    if 'best_val_dice' in df.columns:
        backbone_avg = df.groupby('backbone')['best_val_dice'].mean().sort_values(ascending=False)
        for i, (backbone, avg_dice) in enumerate(backbone_avg.items(), 1):
            summary += f"   {i}. {backbone}: {avg_dice:.4f} (avg across all features)\n"
    
    summary += "\n" + "="*70 + "\n"
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive foundation model ablation study"
    )
    
    parser.add_argument(
        '--backbone',
        type=str,
        default=None,
        choices=[
            'resnet50_pretrained',
            'resnet50_scratch',
            'hybrid_dinov3_small',
            'hybrid_dinov3_base',
            'hybrid_dinov3_large',
            'hybrid_dinov2_base'
        ],
        help='Specific backbone to test (if None, tests all)'
    )
    
    parser.add_argument(
        '--feature_config',
        type=str,
        default=None,
        choices=['rgb', 'luminance', 'chrominance', 'all'],
        help='Specific feature config (if None, tests all)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run complete ablation study (all backbones x all features)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs per configuration'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/results/foundation_ablation',
        help='Output directory'
    )
    
    parser.add_argument(
        '--no_skip_existing',
        action='store_true',
        help='Re-run even if results exist'
    )
    
    args = parser.parse_args()
    
    skip_existing = not args.no_skip_existing
    
    # Determine which backbones and configs to run
    if args.all:
        # Full ablation study
        backbones = [
            'resnet50_pretrained',
            'resnet50_scratch',
            'hybrid_dinov3_base',
            # 'hybrid_dinov3_large',  # Uncomment if resources allow
        ]
        feature_configs = ['rgb', 'luminance', 'chrominance', 'all']
    elif args.backbone and args.feature_config:
        # Single configuration
        backbones = [args.backbone]
        feature_configs = [args.feature_config]
    elif args.backbone:
        # Single backbone, all features
        backbones = [args.backbone]
        feature_configs = ['rgb', 'luminance', 'chrominance', 'all']
    elif args.feature_config:
        # All backbones, single feature
        backbones = ['resnet50_pretrained', 'hybrid_dinov3_base']
        feature_configs = [args.feature_config]
    else:
        # Default: quick comparison
        print("No specific configuration specified. Running quick comparison...")
        backbones = ['resnet50_pretrained', 'hybrid_dinov3_base']
        feature_configs = ['rgb', 'all']
    
    # Run ablation study
    df = run_comprehensive_ablation(
        backbones=backbones,
        feature_configs=feature_configs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        skip_existing=skip_existing
    )
    
    print("\n✓ Foundation model ablation study completed successfully!")


if __name__ == "__main__":
    main()
