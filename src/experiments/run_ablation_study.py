"""
Ablation Study Runner - Experiment 2
=====================================
Runs all configurations for ablation study:
1. RGB baseline (3 channels)
2. Luminance-only (3 channels)
3. Chrominance-only (7 channels)
4. All features (10 channels)
5. Top-5 features (data-driven selection)

From research plan Section 3.4: Ablation Study

Usage:
    python run_ablation_study.py --model unet
    python run_ablation_study.py --model deeplabv3plus --epochs 50
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import argparse
import time
import sys
sys.path.append('experiments')
sys.path.append('src/training')
from train_segmentation_models import execute_model


def run_training(model_type, feature_config, epochs=100, batch_size=4, output_dir='experiments/results/ablation'):
    """
    Run training for a single configuration.
    
    Args:
        model_type: 'unet' or 'deeplabv3plus'
        feature_config: Feature configuration to use
        epochs: Number of training epochs
        batch_size: Batch size
        output_dir: Output directory
        
    Returns:
        Dictionary with training results
    """
    print("\n" + "="*70)
    print(f"RUNNING: {model_type.upper()} with {feature_config.upper()} features")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # If the main function returns something (e.g., metrics), capture it
        execute_model(model_type, feature_config, epochs, batch_size, 
                     output_dir, True)
        print("Training finished successfully.")
    except Exception as e:
        # Handle exceptions directly instead of relying on subprocess error codes
        print(f"✗ Training failed: {e}")
        elapsed_time = time.time() - start_time
        return {
            'model': model_type,
            'feature_config': feature_config,
            'status': 'failed',
            'time_hours': elapsed_time / 3600,
            'error': str(e)
        }
    
    elapsed_time = time.time() - start_time
    
    # Load results
    results_path = Path(output_dir) / model_type / feature_config / 'final_metrics.json'
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            metrics = json.load(f)
        
        metrics['time_hours'] = elapsed_time / 3600
        metrics['status'] = 'success'
        
        print(f"\n✓ Completed {model_type} - {feature_config}")
        print(f"  Best Val Dice: {metrics.get('best_val_dice', 'N/A'):.4f}")
        print(f"  Best Val IoU:  {metrics.get('best_val_iou', 'N/A'):.4f}")
        print(f"  Time: {metrics['time_hours']:.2f} hours")
        
        return metrics
    else:
        print(f"✗ Results file not found for {model_type} - {feature_config}")
        return {
            'model': model_type,
            'feature_config': feature_config,
            'status': 'failed',
            'time_hours': elapsed_time / 3600
        }


def run_ablation_study(
    model_type='unet',
    epochs=100,
    batch_size=4,
    output_dir='experiments/results/ablation',
    skip_existing=True
):
    """
    Run complete ablation study.
    
    Args:
        model_type: Model architecture ('unet' or 'deeplabv3plus')
        epochs: Number of epochs per configuration
        batch_size: Batch size
        output_dir: Output directory
        skip_existing: Skip configurations that already have results
    """
    
    print("="*70)
    print(f"ABLATION STUDY - {model_type.upper()}")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {model_type}")
    print(f"  Epochs per config: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output directory: {output_dir}")
    print(f"  Skip existing: {skip_existing}")
    
    # Define configurations (from research plan)
    configurations = [
        {
            'name': 'rgb',
            'description': 'RGB baseline (3 channels)',
            'channels': 3
        },
        {
            'name': 'luminance',
            'description': 'Luminance-only (3 channels)',
            'channels': 3
        },
        {
            'name': 'chrominance',
            'description': 'Chrominance-only (7 channels)',
            'channels': 7
        },
        {
            'name': 'all',
            'description': 'All features (10 channels)',
            'channels': 10
        }
    ]
    
    print(f"\nConfigurations to run:")
    for i, config in enumerate(configurations, 1):
        print(f"  {i}. {config['description']}")
    
    # Run each configuration
    results = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n{'='*70}")
        print(f"Configuration {i}/{len(configurations)}: {config['description']}")
        print(f"{'='*70}")
        
        # Check if results already exist
        results_path = Path(output_dir) / model_type / config['name'] / 'final_metrics.json'
        
        if skip_existing and results_path.exists():
            print(f"✓ Results already exist, loading from {results_path}")
            with open(results_path, 'r') as f:
                metrics = json.load(f)
            results.append(metrics)
            continue
        
        # Run training
        metrics = run_training(
            model_type=model_type,
            feature_config=config['name'],
            epochs=epochs,
            batch_size=batch_size,
            output_dir=output_dir
        )
        
        results.append(metrics)
    
    # Create comparison table
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        'feature_config', 'n_channels', 
        'best_val_dice', 'best_val_iou',
        'final_val_dice', 'final_val_iou',
        'total_params', 'epochs_trained', 'time_hours', 'status'
    ]
    
    df = df[[col for col in column_order if col in df.columns]]
    
    # Sort by best_val_dice descending (handle missing values)
    if 'best_val_dice' in df.columns:
        df = df.sort_values('best_val_dice', ascending=False, na_position='last')
    
    print("\n" + df.to_string(index=False))
    
    # Save results
    output_path = Path(output_dir) / model_type
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / 'ablation_study_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")
    
    # Save as JSON
    json_path = output_path / 'ablation_study_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {json_path}")
    
    # Create summary for paper
    summary = create_paper_summary(df, model_type)
    
    summary_path = output_path / 'ablation_study_summary.txt'
    with open(summary_path, 'wb') as f:
        f.write(summary.encode('utf-8'))
    print(f"✓ Summary saved to {summary_path}")
    
    print("\n" + summary)
    
    return df


def create_paper_summary(df, model_type):
    """
    Create formatted summary for paper.
    
    FIXED: Proper handling of scalar values from DataFrame.
    """
    
    summary = f"""
{'='*70}
ABLATION STUDY SUMMARY - {model_type.upper()}
{'='*70}

Table: Ablation Study Results

| Configuration | Channels | IoU ↑ | F1/Dice ↑ | Params | Time (h) |
|--------------|----------|-------|-----------|---------|----------|
"""
    
    for _, row in df.iterrows():
        config_name = row['feature_config'].capitalize()
        channels = row.get('n_channels', 'N/A')
        iou = row.get('best_val_iou', 0.0)
        dice = row.get('best_val_dice', 0.0)
        params = row.get('total_params', 0) / 1e6 if row.get('total_params') else 0  # Convert to millions
        time_h = row.get('time_hours', 0.0)
        
        summary += f"| {config_name:12s} | {channels:8} | {iou:.4f} | {dice:.4f} | {params:6.2f}M | {time_h:8.2f} |\n"
    
    summary += "\n"
    
    # Find best configuration (with error handling)
    try:
        if 'best_val_dice' in df.columns and df['best_val_dice'].notna().any():
            best_idx = df['best_val_dice'].idxmax()
            best_config = df.loc[best_idx]
            
            # Calculate improvements
            rgb_row = df[df['feature_config'] == 'rgb']
            if not rgb_row.empty and 'best_val_dice' in rgb_row.columns:
                rgb_dice = rgb_row['best_val_dice'].iloc[0]
                best_dice = best_config['best_val_dice']
                
                if pd.notna(rgb_dice) and pd.notna(best_dice) and rgb_dice > 0:
                    improvement = ((best_dice - rgb_dice) / rgb_dice) * 100
                    
                    summary += f"""
Key Findings:

1. Best Configuration: {best_config['feature_config'].upper()}
   - Dice coefficient: {best_dice:.4f}
   - IoU: {best_config.get('best_val_iou', 0):.4f}
   - Improvement over RGB baseline: +{improvement:.2f}%

"""
    except Exception as e:
        print(f"Warning: Could not generate best configuration summary: {e}")
        summary += "\nKey Findings: Unable to determine best configuration\n\n"
    
    # Luminance vs Chrominance comparison
    summary += "2. Luminance vs Chrominance:\n"
    
    try:
        # FIXED: Proper extraction of scalar values
        lum_row = df[df['feature_config'] == 'luminance']
        chr_row = df[df['feature_config'] == 'chrominance']
        
        # Check if rows exist and have the required column
        if not lum_row.empty and 'best_val_dice' in lum_row.columns:
            lum_dice = lum_row['best_val_dice'].iloc[0]  # Get scalar value
        else:
            lum_dice = None
            
        if not chr_row.empty and 'best_val_dice' in chr_row.columns:
            chr_dice = chr_row['best_val_dice'].iloc[0]  # Get scalar value
        else:
            chr_dice = None
        
        # Now check if values are valid (not None and not NaN)
        if lum_dice is not None and pd.notna(lum_dice):
            summary += f"   - Luminance-only: {lum_dice:.4f}\n"
        else:
            summary += f"   - Luminance-only: Not available\n"
            
        if chr_dice is not None and pd.notna(chr_dice):
            summary += f"   - Chrominance-only: {chr_dice:.4f}\n"
        else:
            summary += f"   - Chrominance-only: Not available\n"
        
        # Compare if both are available
        if (lum_dice is not None and pd.notna(lum_dice) and 
            chr_dice is not None and pd.notna(chr_dice)):
            
            if lum_dice > chr_dice and chr_dice > 0:
                diff = ((lum_dice - chr_dice) / chr_dice) * 100
                summary += f"   - Luminance outperforms chrominance by {diff:.2f}%\n"
            elif chr_dice > lum_dice and lum_dice > 0:
                diff = ((chr_dice - lum_dice) / lum_dice) * 100
                summary += f"   - Chrominance outperforms luminance by {diff:.2f}%\n"
            else:
                summary += f"   - Performance is similar\n"
        
    except Exception as e:
        print(f"Warning: Could not generate luminance vs chrominance comparison: {e}")
        summary += "   - Comparison unavailable\n"
    
    # All features performance
    summary += f"\n3. All Features Performance:\n"
    
    try:
        all_row = df[df['feature_config'] == 'all']
        
        if not all_row.empty and 'best_val_dice' in all_row.columns:
            all_dice = all_row['best_val_dice'].iloc[0]
            
            if pd.notna(all_dice):
                summary += f"   - Dice coefficient: {all_dice:.4f}\n"
                
                # Compare with luminance if available
                if lum_dice is not None and pd.notna(lum_dice) and lum_dice > 0:
                    improvement_lum = ((all_dice - lum_dice) / lum_dice) * 100
                    summary += f"   - Improvement over luminance-only: {improvement_lum:+.2f}%\n"
                
                # Compare with chrominance if available
                if chr_dice is not None and pd.notna(chr_dice) and chr_dice > 0:
                    improvement_chr = ((all_dice - chr_dice) / chr_dice) * 100
                    summary += f"   - Improvement over chrominance-only: {improvement_chr:+.2f}%\n"
            else:
                summary += f"   - Results not available\n"
        else:
            summary += f"   - Results not available\n"
            
    except Exception as e:
        print(f"Warning: Could not generate all features summary: {e}")
        summary += "   - Results not available\n"
    
    summary += f"\n{'='*70}\n"
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study for river segmentation"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='both',
        choices=['unet', 'deeplabv3plus', 'both'],
        help='Model architecture to use'
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
        default=2,
        help='Batch size'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/results/ablation',
        help='Output directory'
    )
    
    parser.add_argument(
        '--no_skip_existing',
        action='store_true',
        help='Re-run even if results exist'
    )
    
    args = parser.parse_args()
    
    skip_existing = not args.no_skip_existing

    print(f'skip_existing : {skip_existing}')
    
    # Run ablation study
    if args.model == 'both':
        # Run for both U-Net and DeepLabv3+
        print("Running ablation study for BOTH U-Net and DeepLabv3+")
        
        print("\n" + "="*70)
        print("PART 1: U-NET")
        print("="*70)
        df_unet = run_ablation_study(
            model_type='unet',
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            skip_existing=skip_existing
        )
        
        print("\n" + "="*70)
        print("PART 2: DEEPLABV3+")
        print("="*70)
        df_deeplab = run_ablation_study(
            model_type='deeplabv3plus',
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            skip_existing=skip_existing
        )
        
        # Compare models
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        try:
            comparison = pd.DataFrame({
                'U-Net Dice': df_unet.set_index('feature_config')['best_val_dice'],
                'U-Net IoU': df_unet.set_index('feature_config')['best_val_iou'],
                'DeepLabv3+ Dice': df_deeplab.set_index('feature_config')['best_val_dice'],
                'DeepLabv3+ IoU': df_deeplab.set_index('feature_config')['best_val_iou']
            })
            
            print("\n" + comparison.to_string())
            
            comparison_path = Path(args.output_dir) / 'model_comparison.csv'
            comparison.to_csv(comparison_path)
            print(f"\n✓ Comparison saved to {comparison_path}")
        except Exception as e:
            print(f"Warning: Could not create model comparison: {e}")
        
    else:
        # Run for single model
        df = run_ablation_study(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            skip_existing=skip_existing
        )
    
    print("\n✓ Ablation study completed successfully!")


if __name__ == "__main__":
    main()

