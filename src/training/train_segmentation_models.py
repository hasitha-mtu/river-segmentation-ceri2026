"""
Training Script for River Segmentation Models
==============================================
Train U-Net or DeepLabv3+ with different feature configurations.

Usage:
    python train_segmentation_models.py --model unet --feature_config all
    python train_segmentation_models.py --model deeplabv3plus --feature_config luminance
"""

import tensorflow as tf
from tensorflow import keras
import argparse
import sys
import yaml
from pathlib import Path
import json
import os

sys.path.append('src/utils')
from loss_util import dice_coefficient, iou_metric, dice_loss, combined_loss

sys.path.append('src/data')
from feature_extraction import FeatureExtractor
from dataset import RiverSegmentationDataset, create_augmentation_pipeline

sys.path.append('src/models')
from unet import create_unet_model
from unet_pretrained import create_unet_pretrained
from deeplabv3plus_adapted import create_deeplabv3plus

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(
    model_type: str,
    input_shape: tuple,
    config: dict
):
    """
    Create segmentation model.
    
    Args:
        model_type: 'unet' or 'deeplabv3plus'
        input_shape: (H, W, C) input shape
        config: Configuration dictionary
        
    Returns:
        Keras model
    """
    if model_type == 'unet':
        print(f"Creating U-Net model with input shape {input_shape}")
        model = create_unet_model(input_shape=input_shape, num_classes=1)

    if model_type == 'unet_pretrained':
        print(f"Creating U-Net model with input shape {input_shape}")
        model = create_unet_pretrained(input_shape=input_shape, num_classes=1)
        
    elif model_type == 'deeplabv3plus':
        print(f"Creating DeepLabv3+ model with input shape {input_shape}")
        
        # Use ImageNet weights only for 3-channel inputs
        weights = 'imagenet' if input_shape[2] == 3 else None
        backbone = config['models']['deeplabv3plus'].get('backbone', 'resnet50')
        
        model = create_deeplabv3plus(
            input_shape=input_shape,
            num_classes=1,
            backbone=backbone,
            weights=weights,
            activation='sigmoid'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_model(
    model_type: str = 'unet',
    feature_config: str = 'all',
    config_path: str = 'config/config.yaml',
    data_dir: str = 'data/processed',
    output_dir: str = 'experiments/results',
    batch_size: int = None,
    epochs: int = None,
    learning_rate: float = None,
    resume_from: str = None
):
    """
    Main training function.
    
    Args:
        model_type: 'unet' or 'deeplabv3plus'
        feature_config: 'all', 'luminance', 'chrominance', 'rgb'
        config_path: Path to configuration file
        data_dir: Directory with processed data
        output_dir: Output directory for results
        batch_size: Batch size (overrides config)
        epochs: Number of epochs (overrides config)
        learning_rate: Learning rate (overrides config)
        resume_from: Path to checkpoint to resume from
    """
    
    print("="*70)
    print(f"TRAINING {model_type.upper()} - {feature_config.upper()} FEATURES")
    print("="*70)
    
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
    print(f"  Model: {model_type}")
    print(f"  Features: {feature_config}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    
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
    
    # Create model
    print("4. Creating model...")
    n_channels = train_dataset.get_n_channels()
    img_size = config['dataset']['working_resolution']
    input_shape = (img_size[0], img_size[1], n_channels)
    
    model = create_model(model_type, input_shape, config)
    
    # Load checkpoint if resuming
    if resume_from:
        print(f"   Loading checkpoint from {resume_from}")
        model.load_weights(resume_from)
    
    # Print model info
    print(f"   Input shape: {input_shape}")
    print(f"   Total parameters: {model.count_params():,}")
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   Trainable parameters: {trainable:,}")
    
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

    class ExponentialDecayCallback(keras.callbacks.Callback):
        def __init__(self, initial_lr=1e-3, decay_steps=1000, decay_rate=0.96, staircase=True):
            super().__init__()
            self.initial_lr = initial_lr
            self.decay_steps = decay_steps
            self.decay_rate = decay_rate
            self.staircase = staircase
            self.step_count = 0
        
        def on_train_batch_begin(self, batch, logs=None):
            if self.staircase:
                step = self.step_count // self.decay_steps
            else:
                step = self.step_count / self.decay_steps
            new_lr = self.initial_lr * (self.decay_rate ** step)
            keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            self.step_count += 1
        
        def on_epoch_end(self, epoch, logs=None):
            """Log current learning rate"""
            current_lr = keras.backend.get_value(self.model.optimizer.learning_rate)
            print(f"\nEpoch {epoch + 1}: Learning Rate = {current_lr:.6f}")
    
    # Callbacks
    print("6. Setting up callbacks...")
    
    callbacks = [

        # ExponentialDecay 
        ExponentialDecayCallback(
            initial_lr=learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        ),

        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'best_model.h5'),
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Save checkpoints every epoch
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'epoch_{epoch:03d}_val_dice_{val_dice_coefficient:.4f}.h5'),
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=False,
            save_freq='epoch',
            verbose=0
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training'].get('lr_factor', 0.5),
            patience=config['training'].get('lr_patience', 7),
            min_lr=1e-7,
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            mode='max',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # CSV logger
        keras.callbacks.CSVLogger(
            filename=str(output_path / 'training_log.csv'),
            separator=',',
            append=False
        ),
        
        # Custom callback to save metrics
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: save_epoch_metrics(
                epoch, logs, output_path
            )
        )
    ]
    
    # Train model
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
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    # Save final model
    final_model_path = output_path / 'final_model.h5'
    model.save(str(final_model_path))
    print(f"\n✓ Final model saved to {final_model_path}")
    
    # Save training history
    history_path = output_path / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert numpy values to Python types
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
    print("FINAL METRICS")
    print("="*70)
    print(f"Best Val Dice:  {final_metrics['best_val_dice']:.4f}")
    print(f"Best Val IoU:   {final_metrics['best_val_iou']:.4f}")
    print(f"Final Val Dice: {final_metrics['final_val_dice']:.4f}")
    print(f"Final Val IoU:  {final_metrics['final_val_iou']:.4f}")
    
    return model, history, final_metrics


def save_epoch_metrics(epoch, logs, output_path):
    """Save metrics after each epoch"""
    metrics_file = output_path / 'epoch_metrics.jsonl'
    
    epoch_data = {
        'epoch': epoch + 1,
        **{k: float(v) for k, v in logs.items()}
    }
    
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(epoch_data) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Train segmentation models for river detection"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='unet',
        choices=['unet', 'deeplabv3plus'],
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--feature_config',
        type=str,
        default='all',
        choices=['all', 'luminance', 'chrominance', 'rgb'],
        help='Which features to use'
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
        default='experiments/results/training',
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
        '--resume_from',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--mixed_precision',
        action='store_true',
        help='Enable mixed precision training'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device ID to use'
    )
    
    args = parser.parse_args()

    print(f'User arguments received: {args}')
    
    # Set GPU
    if args.gpu is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[args.gpu], True) # Allocate only as much GPU memory as is currently needed for the application and dynamically grows the memory usage as required.
                print(f"Using GPU {args.gpu}: {gpus[args.gpu].name}")
            except RuntimeError as e:
                print(f"Error setting GPU: {e}")
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        print("Enabling mixed precision training")
        tf.keras.mixed_precision.set_global_policy('mixed_float16') # Significantly speeds up model training and reduces GPU memory usage by using a combination of 16-bit floating-point (float16) and 32-bit floating-point (float32) data types.
    
    # Check GPU availability
    print("\nGPU Check:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ {len(gpus)} GPU(s) available")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("⚠ No GPU found, using CPU")
    
    # Train model
    model, history, metrics = train_model(
        model_type=args.model,
        feature_config=args.feature_config,
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        resume_from=args.resume_from
    )
    
    print("\n✓ Training completed successfully!")
    print(f"\nView training progress with TensorBoard:")
    print(f"  tensorboard --logdir {args.output_dir}/{args.model}/{args.feature_config}/logs")


def execute_model(model_type, feature_config, epochs, batch_size, output_dir, mixed_precision, 
                  config='config/config.yaml',
                  data_dir='data/processed',
                  lr=None,
                  resume_from=None,
                  gpu=None):
    print(f'Input params model_type:{model_type},feature_config:{feature_config}, epochs:{epochs}, \
          batch_size:{batch_size},output_dir:{output_dir},mixed_precision:{mixed_precision}, \
            config:{config},data_dir:{data_dir},lr:{lr},resume_from:{resume_from},gpu:{gpu}')
    
    # Set GPU
    if gpu is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[gpu], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[gpu], True) 
                print(f"Using GPU {gpu}: {gpus[gpu].name}")
            except RuntimeError as e:
                print(f"Error setting GPU: {e}")
    
    # Enable mixed precision if requested
    if mixed_precision:
        print("Enabling mixed precision training")
        tf.keras.mixed_precision.set_global_policy('mixed_float16') 
    
    # Check GPU availability
    print("\nGPU Check:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ {len(gpus)} GPU(s) available")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("⚠ No GPU found, using CPU")
    
    # Train model
    _model, _history, _metrics = train_model(
        model_type=model_type,
        feature_config=feature_config,
        config_path=config,
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=lr,
        resume_from=resume_from
    )
    
    print("\n✓ Training completed successfully!")
    print(f"\nView training progress with TensorBoard:")
    print(f"  tensorboard --logdir {output_dir}/{model_type}/{feature_config}/logs")

if __name__ == "__main__":
    print(os.getcwd())
    main()