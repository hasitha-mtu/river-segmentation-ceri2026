"""
Ablation Study Without Pretrained Weights
==========================================
Critical experiment to determine if pretrained weight adaptation is the bottleneck.

This runs the SAME configurations as the pretrained study, but trains from scratch.

Research Questions:
1. Does RGB still dominate without pretrained weights?
2. Do all features (10ch) perform better when trained from scratch?
3. How much benefit do pretrained weights provide for each configuration?

Usage:
    python run_ablation_no_pretrain.py --model deeplabv3plus
    python run_ablation_no_pretrain.py --model both --epochs 150
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import argparse
import time
import sys
import tensorflow as tf
from tensorflow import keras
import yaml

sys.path.append('src/utils')
from loss_util import dice_coefficient, iou_metric, dice_loss, combined_loss

sys.path.append('src/data')
from feature_extraction import FeatureExtractor
from dataset import RiverSegmentationDataset, create_augmentation_pipeline

sys.path.append('src/models')
from unet import create_unet_model
from deeplabv3plus_adapted import create_deeplabv3plus


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_no_pretrain(
    model_type: str,
    input_shape: tuple,
    config: dict
):
    """
    Create segmentation model WITHOUT pretrained weights.
    
    Args:
        model_type: 'unet' or 'deeplabv3plus'
        input_shape: (H, W, C) input shape
        config: Configuration dictionary
        
    Returns:
        Keras model
    """
    if model_type == 'unet':
        print(f"Creating U-Net model (NO PRETRAINED) with input shape {input_shape}")
        model = create_unet_model(input_shape=input_shape, num_classes=1)
        
    elif model_type == 'deeplabv3plus':
        print(f"Creating DeepLabv3+ model (NO PRETRAINED) with input shape {input_shape}")
        
        backbone = config['models']['deeplabv3plus'].get('backbone', 'resnet50')
        
        # KEY CHANGE: weights=None for all configurations
        model = create_deeplabv3plus(
            input_shape=input_shape,
            num_classes=1,
            backbone=backbone,
            weights=None,  # ← NO PRETRAINED WEIGHTS
            activation='sigmoid',
            freeze_backbone=False  # ← Must train entire model from scratch
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_model_no_pretrain(
    model_type: str = 'unet',
    feature_config: str = 'all',
    config_path: str = 'config/config.yaml',
    data_dir: str = 'data/processed',
    output_dir: str = 'experiments/results/no_pretrain',
    batch_size: int = None,
    epochs: int = None,
    learning_rate: float = None
):
    """
    Train model from scratch (no pretrained weights).
    
    Key differences from pretrained training:
    - Higher learning rate (1e-3 instead of 1e-4)
    - More epochs (150 instead of 100)
    - All layers trainable
    - Stronger data augmentation
    """
    
    print("="*70)
    print(f"TRAINING FROM SCRATCH: {model_type.upper()} - {feature_config.upper()}")
    print("="*70)
    print("⚠️  NO PRETRAINED WEIGHTS - Training from random initialization")
    
    # Load configuration
    config = load_config(config_path)
    
    # Override with training-from-scratch defaults
    if batch_size is None:
        batch_size = config['training']['batch_size']
    if epochs is None:
        epochs = 150  # More epochs for training from scratch
    if learning_rate is None:
        learning_rate = 1e-3  # Higher LR for scratch training
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_type}")
    print(f"  Features: {feature_config}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate} (higher for scratch training)")
    print(f"  Pretrained weights: None")
    print(f"  Backbone trainable: True")
    
    # Create output directories
    output_path = Path(output_dir) / model_type / feature_config
    checkpoint_dir = output_path / 'checkpoints'
    log_dir = output_path / 'logs'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize feature extractor
    print("\n1. Initializing feature extractor...")
    extractor = FeatureExtractor(config=config.get('features', {}))
    
    # Create augmentation pipeline
    print("2. Creating augmentation pipeline...")
    train_augmentation = create_augmentation_pipeline(config.get('augmentation', {}))
    
    # Create datasets
    print("3. Loading datasets...")
    
    train_dataset = RiverSegmentationDataset(
        image_dir=str(Path(data_dir) / 'train' / 'images'),
        mask_dir=str(Path(data_dir) / 'train' / 'masks'),
        feature_extractor=extractor,
        feature_config=feature_config,
        augmentation_pipeline=train_augmentation,
        image_size=tuple(config['dataset']['working_resolution'])
    )
    
    val_dataset = RiverSegmentationDataset(
        image_dir=str(Path(data_dir) / 'val' / 'images'),
        mask_dir=str(Path(data_dir) / 'val' / 'masks'),
        feature_extractor=extractor,
        feature_config=feature_config,
        augmentation_pipeline=None,
        image_size=tuple(config['dataset']['working_resolution'])
    )
    
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val:   {len(val_dataset)} images")
    
    # Create TensorFlow datasets
    tf_train = train_dataset.create_dataset(
        batch_size=batch_size,
        shuffle=True,
        repeat=True,
        cache=False
    )
    
    tf_val = val_dataset.create_dataset(
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        cache=True
    )
    
    # Create model WITHOUT pretrained weights
    print("4. Creating model WITHOUT pretrained weights...")
    n_channels = train_dataset.get_n_channels()
    img_size = config['dataset']['working_resolution']
    input_shape = (img_size[0], img_size[1], n_channels)
    
    model = create_model_no_pretrain(model_type, input_shape, config)
    
    # Print model info
    print(f"   Input shape: {input_shape}")
    print(f"   Total parameters: {model.count_params():,}")
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   All layers trainable: True")
    
    # Compile model
    print("5. Compiling model...")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[
            dice_coefficient,
            iou_metric,
            'binary_accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    # Calculate steps
    steps_per_epoch = len(train_dataset) // batch_size
    validation_steps = len(val_dataset) // batch_size
    
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")
    
    # Callbacks
    print("6. Setting up callbacks...")
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'best_model.h5'),
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            mode='max',
            patience=20,  # More patience for scratch training
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coefficient',
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=0,
            write_graph=True,
            update_freq='epoch'
        ),
        keras.callbacks.CSVLogger(
            filename=str(output_path / 'training_log.csv'),
            separator=',',
            append=False
        )
    ]
    
    # Train
    print("\n7. Starting training...")
    print("="*70)
    
    history = model.fit(
        tf_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=tf_val,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Training completed!")
    
    # Save final model
    final_model_path = output_path / 'final_model.h5'
    model.save(str(final_model_path))
    print(f"✓ Final model saved to {final_model_path}")
    
    # Save training history
    history_path = output_path / 'training_history.json'
    with open(history_path, 'w') as f:
        history_dict = {
            key: [float(val) for val in values]
            for key, values in history.history.items()
        }
        json.dump(history_dict, f, indent=2)
    print(f"✓ Training history saved to {history_path}")
    
    # Save final metrics summary
    final_metrics = {
        'model_type': model_type,
        'feature_config': feature_config,
        'n_channels': n_channels,
        'total_params': int(model.count_params()),
        'pretrained_weights': False,  # ← KEY FLAG
        'epochs_trained': len(history.history['loss']),
        'best_val_dice': float(max(history.history['val_dice_coefficient'])),
        'best_val_iou': float(max(history.history['val_iou_metric'])),
        'final_val_dice': float(history.history['val_dice_coefficient'][-1]),
        'final_val_iou': float(history.history['val_iou_metric'][-1]),
        'config': {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'trainable_layers': 'all'
        }
    }
    
    metrics_path = output_path / 'final_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"✓ Final metrics saved to {metrics_path}")
    
    # Print final metrics
    print("\n" + "="*70)
    print("FINAL METRICS (NO PRETRAIN)")
    print("="*70)
    print(f"Best Val Dice:  {final_metrics['best_val_dice']:.4f}")
    print(f"Best Val IoU:   {final_metrics['best_val_iou']:.4f}")
    print(f"Final Val Dice: {final_metrics['final_val_dice']:.4f}")
    print(f"Final Val IoU:  {final_metrics['final_val_iou']:.4f}")
    
    return model, history, final_metrics


def run_ablation_no_pretrain(
    model_type='deeplabv3plus',
    epochs=150,
    batch_size=2,
    output_dir='experiments/results/no_pretrain',
    skip_existing=True
):
    """
    Run complete ablation study WITHOUT pretrained weights.
    
    Args:
        model_type: Model architecture ('unet' or 'deeplabv3plus')
        epochs: Number of epochs (default 150 for scratch training)
        batch_size: Batch size
        output_dir: Output directory
        skip_existing: Skip configurations that already have results
    """
    
    print("="*70)
    print(f"ABLATION STUDY (NO PRETRAIN) - {model_type.upper()}")
    print("="*70)
    print("⚠️  All models trained from RANDOM INITIALIZATION")
    print(f"\nConfiguration:")
    print(f"  Model: {model_type}")
    print(f"  Epochs per config: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output directory: {output_dir}")
    print(f"  Skip existing: {skip_existing}")
    
    # Define configurations (same as pretrained study)
    configurations = [
        {
            'name': 'rgb',
            'description': 'RGB baseline (3 channels) - NO PRETRAIN',
            'channels': 3
        },
        {
            'name': 'luminance',
            'description': 'Luminance-only (3 channels) - NO PRETRAIN',
            'channels': 3
        },
        {
            'name': 'chrominance',
            'description': 'Chrominance-only (7 channels) - NO PRETRAIN',
            'channels': 7
        },
        {
            'name': 'all',
            'description': 'All features (10 channels) - NO PRETRAIN',
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
        start_time = time.time()
        
        try:
            _, _, metrics = train_model_no_pretrain(
                model_type=model_type,
                feature_config=config['name'],
                epochs=epochs,
                batch_size=batch_size,
                output_dir=output_dir
            )
            
            elapsed_time = time.time() - start_time
            metrics['time_hours'] = elapsed_time / 3600
            metrics['status'] = 'success'
            
            print(f"\n✓ Completed {model_type} - {config['name']}")
            print(f"  Best Val Dice: {metrics['best_val_dice']:.4f}")
            print(f"  Best Val IoU:  {metrics['best_val_iou']:.4f}")
            print(f"  Time: {metrics['time_hours']:.2f} hours")
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            elapsed_time = time.time() - start_time
            metrics = {
                'model_type': model_type,
                'feature_config': config['name'],
                'status': 'failed',
                'time_hours': elapsed_time / 3600,
                'error': str(e)
            }
        
        results.append(metrics)
    
    # Create comparison table
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS (NO PRETRAIN)")
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
    
    # Sort by best_val_dice descending
    if 'best_val_dice' in df.columns:
        df = df.sort_values('best_val_dice', ascending=False, na_position='last')
    
    print("\n" + df.to_string(index=False))
    
    # Save results
    output_path = Path(output_dir) / model_type
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / 'ablation_no_pretrain_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")
    
    # Save as JSON
    json_path = output_path / 'ablation_no_pretrain_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {json_path}")
    
    return df


def compare_pretrained_vs_scratch(
    model_type='deeplabv3plus',
    pretrain_dir='experiments/results/ablation',
    no_pretrain_dir='experiments/results/no_pretrain'
):
    """
    Compare results between pretrained and no-pretrained experiments.
    
    This is the KEY ANALYSIS to understand if weight adaptation is the bottleneck!
    """
    
    print("\n" + "="*70)
    print(f"PRETRAINED vs NO-PRETRAINED COMPARISON - {model_type.upper()}")
    print("="*70)
    
    # Load pretrained results
    pretrain_path = Path(pretrain_dir) / model_type / 'ablation_study_results.csv'
    no_pretrain_path = Path(no_pretrain_dir) / model_type / 'ablation_no_pretrain_results.csv'
    
    if not pretrain_path.exists():
        print(f"✗ Pretrained results not found at {pretrain_path}")
        return None
    
    if not no_pretrain_path.exists():
        print(f"✗ No-pretrain results not found at {no_pretrain_path}")
        return None
    
    df_pretrain = pd.read_csv(pretrain_path)
    df_no_pretrain = pd.read_csv(no_pretrain_path)
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Config': df_pretrain['feature_config'],
        'Pretrained Dice': df_pretrain['best_val_dice'],
        'Scratch Dice': df_no_pretrain['best_val_dice'],
        'Pretrained IoU': df_pretrain['best_val_iou'],
        'Scratch IoU': df_no_pretrain['best_val_iou'],
    })
    
    # Calculate improvements from pretraining
    comparison['Pretrain Benefit (Dice)'] = (
        comparison['Pretrained Dice'] - comparison['Scratch Dice']
    )
    comparison['Pretrain Benefit (%)'] = (
        (comparison['Pretrained Dice'] - comparison['Scratch Dice']) / 
        comparison['Scratch Dice'] * 100
    )
    
    print("\n" + comparison.to_string(index=False))
    
    # Save comparison
    output_path = Path(pretrain_dir)
    comparison_path = output_path / f'{model_type}_pretrain_vs_scratch.csv'
    comparison.to_csv(comparison_path, index=False)
    print(f"\n✓ Comparison saved to {comparison_path}")
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Which config benefits most from pretraining?
    max_benefit_idx = comparison['Pretrain Benefit (Dice)'].idxmax()
    max_benefit_config = comparison.loc[max_benefit_idx]
    
    print(f"\n1. Configuration that benefits MOST from pretrained weights:")
    print(f"   {max_benefit_config['Config'].upper()}")
    print(f"   Improvement: +{max_benefit_config['Pretrain Benefit (Dice)']:.4f} Dice")
    print(f"   Percentage: +{max_benefit_config['Pretrain Benefit (%)']:.2f}%")
    
    # Which config benefits least?
    min_benefit_idx = comparison['Pretrain Benefit (Dice)'].idxmin()
    min_benefit_config = comparison.loc[min_benefit_idx]
    
    print(f"\n2. Configuration that benefits LEAST from pretrained weights:")
    print(f"   {min_benefit_config['Config'].upper()}")
    print(f"   Improvement: +{min_benefit_config['Pretrain Benefit (Dice)']:.4f} Dice")
    print(f"   Percentage: +{min_benefit_config['Pretrain Benefit (%)']:.2f}%")
    
    # Critical question: Does "all" beat RGB without pretrain?
    rgb_scratch = comparison[comparison['Config'] == 'rgb']['Scratch Dice'].values[0]
    all_scratch = comparison[comparison['Config'] == 'all']['Scratch Dice'].values[0]
    
    print(f"\n3. CRITICAL FINDING: All Features vs RGB (WITHOUT Pretrain)")
    print(f"   RGB (scratch):  {rgb_scratch:.4f}")
    print(f"   All (scratch):  {all_scratch:.4f}")
    
    if all_scratch > rgb_scratch:
        improvement = ((all_scratch - rgb_scratch) / rgb_scratch) * 100
        print(f"   ✓ All features WIN by +{improvement:.2f}%")
        print(f"   → Conclusion: Projection layer IS the bottleneck!")
        print(f"   → Additional features ARE useful, but adaptation hurts them")
    else:
        decline = ((rgb_scratch - all_scratch) / rgb_scratch) * 100
        print(f"   ✗ RGB still wins by +{decline:.2f}%")
        print(f"   → Conclusion: Additional features don't help this task")
        print(f"   → RGB contains sufficient information for river segmentation")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study WITHOUT pretrained weights"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='deeplabv3plus',
        choices=['unet', 'deeplabv3plus', 'both'],
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
        help='Number of epochs (more for scratch training)'
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
        default='experiments/results/no_pretrain',
        help='Output directory for no-pretrain results'
    )
    
    parser.add_argument(
        '--pretrain_dir',
        type=str,
        default='experiments/results/ablation',
        help='Directory with pretrained results for comparison'
    )
    
    parser.add_argument(
        '--no_skip_existing',
        action='store_true',
        help='Re-run even if results exist'
    )
    
    parser.add_argument(
        '--compare_only',
        action='store_true',
        help='Only run comparison (skip training)'
    )
    
    args = parser.parse_args()
    
    skip_existing = not args.no_skip_existing
    
    if args.compare_only:
        # Only compare existing results
        print("Running comparison only (no training)")
        if args.model == 'both':
            compare_pretrained_vs_scratch('unet', args.pretrain_dir, args.output_dir)
            compare_pretrained_vs_scratch('deeplabv3plus', args.pretrain_dir, args.output_dir)
        else:
            compare_pretrained_vs_scratch(args.model, args.pretrain_dir, args.output_dir)
        return
    
    # Run ablation study without pretrained weights
    if args.model == 'both':
        print("Running ablation study for BOTH U-Net and DeepLabv3+ (NO PRETRAIN)")
        
        print("\n" + "="*70)
        print("PART 1: U-NET (NO PRETRAIN)")
        print("="*70)
        df_unet = run_ablation_no_pretrain(
            model_type='unet',
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            skip_existing=skip_existing
        )
        
        print("\n" + "="*70)
        print("PART 2: DEEPLABV3+ (NO PRETRAIN)")
        print("="*70)
        df_deeplab = run_ablation_no_pretrain(
            model_type='deeplabv3plus',
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            skip_existing=skip_existing
        )
        
        # Compare with pretrained results
        print("\n" + "="*70)
        print("COMPARING WITH PRETRAINED RESULTS")
        print("="*70)
        
        compare_pretrained_vs_scratch('unet', args.pretrain_dir, args.output_dir)
        compare_pretrained_vs_scratch('deeplabv3plus', args.pretrain_dir, args.output_dir)
        
    else:
        # Run for single model
        df = run_ablation_no_pretrain(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            skip_existing=skip_existing
        )
        
        # Compare with pretrained results
        compare_pretrained_vs_scratch(args.model, args.pretrain_dir, args.output_dir)
    
    print("\n✓ Ablation study (no pretrain) completed successfully!")


if __name__ == "__main__":
    main()
