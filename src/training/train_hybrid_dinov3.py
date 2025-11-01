"""
Training Script for Hybrid CNN-DINOv3
======================================
Integrates the hybrid model with your existing training pipeline.

This script:
1. Uses your existing dataset pipeline
2. Adapts data for dual-input model (all channels + RGB)
3. Trains hybrid CNN-DINOv3 model
4. Compares results with baseline

Usage:
    python train_hybrid_dinov3.py --feature_config all --dinov3_size base --epochs 100
"""

import tensorflow as tf
from tensorflow import keras
import argparse
import sys
import yaml
from pathlib import Path
import json
import os

# Import your existing utilities
sys.path.append('src/utils')
from loss_util import dice_coefficient, iou_metric, dice_loss, combined_loss

sys.path.append('src/data')
from feature_extraction import FeatureExtractor
from dataset import RiverSegmentationDataset, create_augmentation_pipeline

# Import hybrid model
sys.path.append('src/models')
from hybrid_cnn_dinov3 import create_hybrid_model, load_dinov3_weights


class DualInputDatasetWrapper:
    """
    Wrapper to convert single-input dataset to dual-input format
    required by Hybrid CNN-DINOv3.
    
    Transforms:
        (image, mask) → ((image_all, image_rgb), mask)
    """
    
    def __init__(self, dataset):
        """
        Args:
            dataset: Original dataset that yields (image, mask)
        """
        self.dataset = dataset
    
    def __call__(self, *args, **kwargs):
        """Transform dataset to dual-input format"""
        
        def transform_fn(image, mask):
            """
            Transform single image to dual inputs.
            
            Args:
                image: (H, W, C) where C is 3, 7, or 10
                mask: (H, W, 1) segmentation mask
            
            Returns:
                ((image_all, image_rgb), mask)
            """
            # Extract RGB channels (first 3 channels)
            image_rgb = image[..., :3]
            
            # All channels as-is
            image_all = image
            
            return (image_all, image_rgb), mask
        
        return self.dataset.map(
            transform_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_hybrid_model(
    feature_config: str = 'all',
    dinov3_size: str = 'base',
    config_path: str = 'config/config.yaml',
    data_dir: str = 'data/processed',
    output_dir: str = 'experiments/results/hybrid_dinov3',
    batch_size: int = None,
    epochs: int = None,
    learning_rate: float = None,
    freeze_dinov3: bool = True,
    resume_from: str = None
):
    """
    Train Hybrid CNN-DINOv3 model.
    
    Args:
        feature_config: 'all', 'luminance', 'chrominance', or 'rgb'
        dinov3_size: 'small', 'base', 'large', or 'giant'
        config_path: Path to configuration file
        data_dir: Directory with processed data
        output_dir: Output directory for results
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        freeze_dinov3: If True, freeze DINOv3 branch
        resume_from: Path to checkpoint to resume from
    """
    
    print("="*70)
    print(f"TRAINING HYBRID CNN-DINOV3")
    print("="*70)
    print(f"Feature config: {feature_config}")
    print(f"DINOv3 size: {dinov3_size}")
    print(f"DINOv3 frozen: {freeze_dinov3}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Override config with command-line arguments
    if batch_size is None:
        batch_size = config['training']['batch_size']
    if epochs is None:
        epochs = config['training']['epochs']
    if learning_rate is None:
        learning_rate = config['training']['initial_lr']
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    # Create output directories
    output_path = Path(output_dir) / dinov3_size / feature_config
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
    
    # Wrap datasets for dual-input format
    print("4. Converting to dual-input format...")
    wrapper = DualInputDatasetWrapper(tf_train)
    tf_train_dual = wrapper()
    
    wrapper_val = DualInputDatasetWrapper(tf_val)
    tf_val_dual = wrapper_val()
    
    # Create hybrid model
    print("5. Creating Hybrid CNN-DINOv3 model...")
    n_channels = train_dataset.get_n_channels()
    img_size = config['dataset']['working_resolution']
    input_shape = (img_size[0], img_size[1], n_channels)
    
    model = create_hybrid_model(
        input_shape=input_shape,
        num_classes=1,
        dinov3_size=dinov3_size,
        freeze_dinov3=freeze_dinov3,
        activation='sigmoid'
    )
    
    # Load pretrained weights if available
    if resume_from:
        print(f"\n   Loading checkpoint from {resume_from}")
        model.load_weights(resume_from)
    
    # Print model info
    print(f"\n   Input shape: {input_shape}")
    print(f"   DINOv3 size: {dinov3_size}")
    print(f"   Total parameters: {model.count_params():,}")
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   DINOv3 frozen: {freeze_dinov3}")
    
    # Compile model
    print("\n6. Compiling model...")
    
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
    print("\n7. Setting up callbacks...")
    
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
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coefficient',
            mode='max',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
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
    print("\n8. Starting training...")
    print("="*70)
    
    history = model.fit(
        tf_train_dual,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=tf_val_dual,
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
        'model_type': f'hybrid_cnn_dinov3_{dinov3_size}',
        'feature_config': feature_config,
        'n_channels': n_channels,
        'dinov3_size': dinov3_size,
        'dinov3_frozen': freeze_dinov3,
        'total_params': int(model.count_params()),
        'trainable_params': int(trainable),
        'epochs_trained': len(history.history['loss']),
        'best_val_dice': float(max(history.history['val_dice_coefficient'])),
        'best_val_iou': float(max(history.history['val_iou_metric'])),
        'final_val_dice': float(history.history['val_dice_coefficient'][-1]),
        'final_val_iou': float(history.history['val_iou_metric'][-1]),
        'config': {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate
        }
    }
    
    metrics_path = output_path / 'final_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"✓ Final metrics saved to {metrics_path}")
    
    # Print final metrics
    print("\n" + "="*70)
    print("FINAL METRICS - HYBRID CNN-DINOV3")
    print("="*70)
    print(f"Model: Hybrid CNN-DINOv3 ({dinov3_size})")
    print(f"Features: {feature_config} ({n_channels} channels)")
    print(f"Best Val Dice:  {final_metrics['best_val_dice']:.4f}")
    print(f"Best Val IoU:   {final_metrics['best_val_iou']:.4f}")
    print(f"Final Val Dice: {final_metrics['final_val_dice']:.4f}")
    print(f"Final Val IoU:  {final_metrics['final_val_iou']:.4f}")
    
    return model, history, final_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid CNN-DINOv3 model for river segmentation"
    )
    
    parser.add_argument(
        '--feature_config',
        type=str,
        default='all',
        choices=['all', 'luminance', 'chrominance', 'rgb'],
        help='Which features to use'
    )
    
    parser.add_argument(
        '--dinov3_size',
        type=str,
        default='base',
        choices=['small', 'base', 'large', 'giant'],
        help='Size of DINOv3 backbone'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Directory with processed data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/results/hybrid_dinov3',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--no_freeze_dinov3',
        action='store_true',
        help='Do not freeze DINOv3 branch (finetune it)'
    )
    
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device ID to use'
    )
    
    args = parser.parse_args()
    
    # Set GPU
    if args.gpu is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
                print(f"Using GPU {args.gpu}: {gpus[args.gpu].name}")
            except RuntimeError as e:
                print(f"Error setting GPU: {e}")
    
    # Check GPU availability
    print("\nGPU Check:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ {len(gpus)} GPU(s) available")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("⚠ No GPU found, using CPU (will be slow!)")
    
    # Train model
    model, history, metrics = train_hybrid_model(
        feature_config=args.feature_config,
        dinov3_size=args.dinov3_size,
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        freeze_dinov3=not args.no_freeze_dinov3,
        resume_from=args.resume_from
    )
    
    print("\n✓ Training completed successfully!")
    print(f"\nView training progress with TensorBoard:")
    print(f"  tensorboard --logdir {args.output_dir}/{args.dinov3_size}/{args.feature_config}/logs")


if __name__ == "__main__":
    main()
