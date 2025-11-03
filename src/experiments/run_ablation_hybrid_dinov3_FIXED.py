"""
Ablation Study for Hybrid PyTorch-TensorFlow DINOv3 Model
==========================================================
Comprehensive ablation study to evaluate the effectiveness of the hybrid architecture
that combines CNN (all 10 channels) with pretrained DINOv3 (RGB only).

Research Questions:
1. Does the hybrid model outperform CNN-only baseline?
2. How much does pretrained DINOv3 contribute to performance?
3. What's the contribution of different components (CNN, DINOv3, fusion)?
4. Does cross-attention fusion perform better than simple concatenation?

Experimental Configurations:
A. CNN-Only Baseline (10 channels, no DINOv3)
B. DINOv3-Only Baseline (RGB only, no CNN branch)
C. Hybrid with Concatenation Fusion (baseline fusion)
D. Hybrid with Cross-Attention Fusion (full model)
E. Hybrid with Different DINOv3 Sizes (small, base, large)
F. Hybrid with Frozen vs Fine-tuned DINOv3

Usage:
    # Run full ablation study
    python run_ablation_hybrid_dinov3.py --experiment all
    
    # Run specific experiment
    python run_ablation_hybrid_dinov3.py --experiment cnn_only
    python run_ablation_hybrid_dinov3.py --experiment hybrid_full
    
    # Compare with existing baselines
    python run_ablation_hybrid_dinov3.py --compare_only --baseline_dir experiments/results/ablation
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
import numpy as np

# Add project paths
sys.path.append('src/utils')
from loss_util import dice_coefficient, iou_metric, dice_loss, combined_loss

sys.path.append('src/data')
from feature_extraction import FeatureExtractor
from dataset import RiverSegmentationDataset, create_augmentation_pipeline

# Import hybrid model components
from hybrid_pytorch_tf_dinov3 import (
    create_hybrid_pytorch_tf_model,
    TFPyTorchDINOv3Layer,
    MultiScaleCNNBranch,
    CrossAttentionFusion,
    SegmentationDecoder
)


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ============================================================================
# MODEL CREATION FUNCTIONS FOR DIFFERENT ABLATION EXPERIMENTS
# ============================================================================

def create_cnn_only_model(input_shape, num_classes=1):
    """
    Ablation: CNN-only baseline (no DINOv3)
    Processes all 10 channels through CNN branch only
    """
    print("\n" + "="*70)
    print("CREATING CNN-ONLY MODEL (Ablation Baseline)")
    print("="*70)
    print(f"Input shape: {input_shape}")
    print("Components: CNN branch only (no DINOv3)")
    print("="*70)
    
    input_layer = keras.layers.Input(shape=input_shape, name='input')
    
    # CNN branch
    cnn_branch = MultiScaleCNNBranch(name='cnn_branch')
    features = cnn_branch(input_layer)
    
    # Decoder
    decoder = SegmentationDecoder(num_classes=num_classes, activation='sigmoid', name='decoder')
    output = decoder(features)
    
    model = keras.Model(inputs=input_layer, outputs=output, name='cnn_only')
    
    print(f"\n✓ Model created: {model.count_params():,} parameters")
    return model


def create_dinov3_only_model(input_shape, num_classes=1, dinov3_model='facebook/dinov2-base'):
    """
    Ablation: DINOv2-only baseline (no CNN branch)
    Processes RGB through DINOv2 only
    
    Note: Using publicly available DINOv2 models (no authentication required)
    """
    print("\n" + "="*70)
    print("CREATING DINOV2-ONLY MODEL (Ablation Baseline)")
    print("="*70)
    print(f"Input shape: {input_shape}")
    print(f"DINOv2 model: {dinov3_model}")
    print("Components: DINOv2 only (no CNN branch)")
    print("="*70)
    
    # Input is RGB only (first 3 channels)
    input_layer = keras.layers.Input(shape=(input_shape[0], input_shape[1], 3), name='input_rgb')
    
    # DINOv3 branch
    dinov3_layer = TFPyTorchDINOv3Layer(
        model_name=dinov3_model,
        output_size=(input_shape[0] // 4, input_shape[1] // 4),
        name='dinov3_branch'
    )
    features = dinov3_layer(input_layer)
    
    # Decoder
    decoder = SegmentationDecoder(num_classes=num_classes, activation='sigmoid', name='decoder')
    output = decoder(features)
    
    model = keras.Model(inputs=input_layer, outputs=output, name='dinov3_only')
    
    print(f"\n✓ Model created (TF params only): {model.count_params():,} parameters")
    print("  Note: PyTorch DINOv3 parameters not included in count")
    return model


def create_hybrid_concat_model(input_shape, num_classes=1, dinov3_model='facebook/dinov2-base'):
    """
    Ablation: Hybrid with simple concatenation fusion (no cross-attention)
    Tests if cross-attention is necessary or if concatenation is sufficient
    
    Note: Using publicly available DINOv2 models (no authentication required)
    """
    print("\n" + "="*70)
    print("CREATING HYBRID MODEL WITH CONCATENATION FUSION")
    print("="*70)
    print(f"Input shape: {input_shape}")
    print("Components: CNN + DINOv3 + Concatenation fusion")
    print("="*70)
    
    input_all = keras.layers.Input(shape=input_shape, name='input_all_channels')
    input_rgb = keras.layers.Input(shape=(input_shape[0], input_shape[1], 3), name='input_rgb')
    
    # CNN branch
    cnn_branch = MultiScaleCNNBranch(name='cnn_branch')
    cnn_features = cnn_branch(input_all)
    
    # DINOv3 branch
    dinov3_layer = TFPyTorchDINOv3Layer(
        model_name=dinov3_model,
        output_size=(input_shape[0] // 4, input_shape[1] // 4),
        name='dinov3_branch'
    )
    dinov3_features = dinov3_layer(input_rgb)
    
    # Simple concatenation fusion (ablation: no cross-attention)
    h = tf.shape(cnn_features)[1]
    w = tf.shape(cnn_features)[2]
    dinov3_resized = tf.image.resize(dinov3_features, [h, w])
    fused = keras.layers.Concatenate(name='concat_fusion')([cnn_features, dinov3_resized])
    
    # Projection to reduce channels
    fused = keras.layers.Conv2D(256, 1, activation='relu', name='fusion_projection')(fused)
    
    # Decoder
    decoder = SegmentationDecoder(num_classes=num_classes, activation='sigmoid', name='decoder')
    output = decoder(fused)
    
    model = keras.Model(
        inputs=[input_all, input_rgb],
        outputs=output,
        name='hybrid_concat'
    )
    
    print(f"\n✓ Model created: {model.count_params():,} parameters")
    return model


def create_hybrid_full_model(
    input_shape,
    num_classes=1,
    dinov3_model='facebook/dinov2-base',
    dinov3_size='base'
):
    """
    Full hybrid model with cross-attention fusion (Model 2)
    This is the complete architecture being tested
    
    Note: Using publicly available DINOv2 models (no authentication required)
    """
    print("\n" + "="*70)
    print("CREATING FULL HYBRID MODEL WITH CROSS-ATTENTION")
    print("="*70)
    print(f"Input shape: {input_shape}")
    print(f"DINOv3: {dinov3_size}")
    print("Components: CNN + DINOv3 + Cross-Attention fusion")
    print("="*70)
    
    # Use the full model from hybrid_pytorch_tf_dinov3.py
    model = create_hybrid_pytorch_tf_model(
        input_shape=input_shape,
        num_classes=num_classes,
        dinov3_model=dinov3_model,
        activation='sigmoid'
    )
    
    return model


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_ablation_model(
    model,
    experiment_name: str,
    config_path: str = 'config/config.yaml',
    data_dir: str = 'data/processed',
    output_dir: str = 'experiments/results/ablation_hybrid',
    batch_size: int = None,
    epochs: int = None,
    learning_rate: float = None,
    feature_config: str = 'all'
):
    """
    Train a model for ablation study.
    
    Args:
        model: Compiled Keras model
        experiment_name: Name of the experiment (e.g., 'cnn_only', 'hybrid_full')
        config_path: Path to config file
        data_dir: Data directory
        output_dir: Output directory for results
        batch_size: Batch size (None to use config default)
        epochs: Number of epochs (None to use config default)
        learning_rate: Learning rate (None to use config default)
        feature_config: Feature configuration ('all', 'rgb', etc.)
    
    Returns:
        Training history and best metrics
    """
    
    print("\n" + "="*70)
    print(f"TRAINING: {experiment_name.upper()}")
    print("="*70)
    
    # Load configuration
    config = load_config(config_path)
    
    # Use config defaults if not specified
    if batch_size is None:
        batch_size = config['training'].get('batch_size', 2)
    if epochs is None:
        epochs = config['training'].get('epochs', 100)
    if learning_rate is None:
        learning_rate = config['training'].get('learning_rate', 1e-4)
    
    print(f"\nConfiguration:")
    print(f"  Experiment: {experiment_name}")
    print(f"  Features: {feature_config}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    # Create output directories
    output_path = Path(output_dir) / experiment_name
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
    
    # Compile model
    print("4. Compiling model...")
    
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
    
    # Print model info
    print(f"   Total parameters: {model.count_params():,}")
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   Trainable parameters: {trainable:,}")
    
    # Calculate steps
    steps_per_epoch = len(train_dataset) // batch_size
    validation_steps = len(val_dataset) // batch_size
    
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")
    
    # Callbacks
    print("5. Setting up callbacks...")
    
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
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coefficient',
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=0,
            write_graph=False
        ),
        keras.callbacks.CSVLogger(
            str(output_path / 'training_log.csv'),
            append=False
        )
    ]
    
    # Train model
    print(f"\n6. Training model for {epochs} epochs...")
    print("-" * 70)
    
    start_time = time.time()
    
    history = model.fit(
        tf_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=tf_val,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print("-" * 70)
    print(f"✓ Training completed in {training_time/60:.2f} minutes")
    
    # Get best metrics
    best_epoch = np.argmax(history.history['val_dice_coefficient'])
    best_metrics = {
        'experiment': experiment_name,
        'feature_config': feature_config,
        'best_epoch': int(best_epoch + 1),
        'best_val_dice': float(history.history['val_dice_coefficient'][best_epoch]),
        'best_val_iou': float(history.history['val_iou_metric'][best_epoch]),
        'best_val_accuracy': float(history.history['val_binary_accuracy'][best_epoch]),
        'best_val_precision': float(history.history['val_precision'][best_epoch]),
        'best_val_recall': float(history.history['val_recall'][best_epoch]),
        'final_train_dice': float(history.history['dice_coefficient'][-1]),
        'final_val_dice': float(history.history['val_dice_coefficient'][-1]),
        'training_time_minutes': float(training_time / 60),
        'total_epochs': len(history.history['loss']),
        'total_params': int(model.count_params()),
        'trainable_params': int(trainable)
    }
    
    # Print summary
    print("\n" + "="*70)
    print(f"EXPERIMENT SUMMARY: {experiment_name.upper()}")
    print("="*70)
    print(f"Best Epoch:        {best_metrics['best_epoch']}/{epochs}")
    print(f"Best Val Dice:     {best_metrics['best_val_dice']:.4f}")
    print(f"Best Val IoU:      {best_metrics['best_val_iou']:.4f}")
    print(f"Val Accuracy:      {best_metrics['best_val_accuracy']:.4f}")
    print(f"Val Precision:     {best_metrics['best_val_precision']:.4f}")
    print(f"Val Recall:        {best_metrics['best_val_recall']:.4f}")
    print(f"Training Time:     {best_metrics['training_time_minutes']:.2f} min")
    print("="*70)
    
    # Save results
    results_file = output_path / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(best_metrics, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")
    
    # Save history
    history_df = pd.DataFrame(history.history)
    history_file = output_path / 'training_history.csv'
    history_df.to_csv(history_file, index=False)
    print(f"✓ Training history saved to {history_file}")
    
    return history, best_metrics


# ============================================================================
# ABLATION STUDY EXECUTION
# ============================================================================

def run_full_ablation_study(
    config_path: str = 'config/config.yaml',
    data_dir: str = 'data/processed',
    output_dir: str = 'experiments/results/ablation_hybrid',
    batch_size: int = 2,
    epochs: int = 100,
    skip_existing: bool = True
):
    """
    Run complete ablation study with all experiments.
    
    Experiments:
    1. CNN-Only Baseline
    2. DINOv3-Only Baseline
    3. Hybrid with Concatenation Fusion
    4. Hybrid with Cross-Attention Fusion (Full Model)
    5. Hybrid with Different DINOv3 Sizes
    """
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ABLATION STUDY: HYBRID PYTORCH-TENSORFLOW DINOV3")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Epochs per experiment: {epochs}")
    print(f"Batch size: {batch_size}")
    print("="*80)
    
    config = load_config(config_path)
    img_size = config['dataset']['working_resolution']
    
    all_results = []
    
    # ========================================================================
    # EXPERIMENT 1: CNN-Only Baseline (10 channels)
    # ========================================================================
    exp_name = 'cnn_only'
    exp_path = Path(output_dir) / exp_name / 'results.json'
    
    if skip_existing and exp_path.exists():
        print(f"\n⏭️  Skipping {exp_name} (results exist)")
        with open(exp_path, 'r') as f:
            results = json.load(f)
        all_results.append(results)
    else:
        print("\n" + "="*70)
        print("EXPERIMENT 1: CNN-ONLY BASELINE")
        print("="*70)
        
        model = create_cnn_only_model(
            input_shape=(img_size[0], img_size[1], 10),
            num_classes=1
        )
        
        _, results = train_ablation_model(
            model=model,
            experiment_name=exp_name,
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            feature_config='all'
        )
        
        all_results.append(results)
    
    # ========================================================================
    # EXPERIMENT 2: DINOv3-Only Baseline (RGB only)
    # ========================================================================
    exp_name = 'dinov3_only'
    exp_path = Path(output_dir) / exp_name / 'results.json'
    
    if skip_existing and exp_path.exists():
        print(f"\n⏭️  Skipping {exp_name} (results exist)")
        with open(exp_path, 'r') as f:
            results = json.load(f)
        all_results.append(results)
    else:
        print("\n" + "="*70)
        print("EXPERIMENT 2: DINOV3-ONLY BASELINE")
        print("="*70)
        
        model = create_dinov3_only_model(
            input_shape=(img_size[0], img_size[1], 10),
            num_classes=1
        )
        
        _, results = train_ablation_model(
            model=model,
            experiment_name=exp_name,
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            feature_config='rgb'
        )
        
        all_results.append(results)
    
    # ========================================================================
    # EXPERIMENT 3: Hybrid with Concatenation Fusion
    # ========================================================================
    exp_name = 'hybrid_concat'
    exp_path = Path(output_dir) / exp_name / 'results.json'
    
    if skip_existing and exp_path.exists():
        print(f"\n⏭️  Skipping {exp_name} (results exist)")
        with open(exp_path, 'r') as f:
            results = json.load(f)
        all_results.append(results)
    else:
        print("\n" + "="*70)
        print("EXPERIMENT 3: HYBRID WITH CONCATENATION FUSION")
        print("="*70)
        
        model = create_hybrid_concat_model(
            input_shape=(img_size[0], img_size[1], 10),
            num_classes=1
        )
        
        _, results = train_ablation_model(
            model=model,
            experiment_name=exp_name,
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            feature_config='all'
        )
        
        all_results.append(results)
    
    # ========================================================================
    # EXPERIMENT 4: Full Hybrid Model (Cross-Attention)
    # ========================================================================
    exp_name = 'hybrid_full'
    exp_path = Path(output_dir) / exp_name / 'results.json'
    
    if skip_existing and exp_path.exists():
        print(f"\n⏭️  Skipping {exp_name} (results exist)")
        with open(exp_path, 'r') as f:
            results = json.load(f)
        all_results.append(results)
    else:
        print("\n" + "="*70)
        print("EXPERIMENT 4: FULL HYBRID MODEL (CROSS-ATTENTION)")
        print("="*70)
        
        model = create_hybrid_full_model(
            input_shape=(img_size[0], img_size[1], 10),
            num_classes=1,
            dinov3_size='base'
        )
        
        _, results = train_ablation_model(
            model=model,
            experiment_name=exp_name,
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            feature_config='all'
        )
        
        all_results.append(results)
    
    # ========================================================================
    # EXPERIMENT 5: Hybrid with Small DINOv2
    # ========================================================================
    exp_name = 'hybrid_dinov3_small'
    exp_path = Path(output_dir) / exp_name / 'results.json'
    
    if skip_existing and exp_path.exists():
        print(f"\n⏭️  Skipping {exp_name} (results exist)")
        with open(exp_path, 'r') as f:
            results = json.load(f)
        all_results.append(results)
    else:
        print("\n" + "="*70)
        print("EXPERIMENT 5: HYBRID WITH SMALL DINOV2")
        print("="*70)
        
        model = create_hybrid_full_model(
            input_shape=(img_size[0], img_size[1], 10),
            num_classes=1,
            dinov3_model='facebook/dinov2-small',  # Use small DINOv2
            dinov3_size='small'
        )
        
        _, results = train_ablation_model(
            model=model,
            experiment_name=exp_name,
            config_path=config_path,
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            feature_config='all'
        )
        
        all_results.append(results)
    
    # ========================================================================
    # Save consolidated results
    # ========================================================================
    print("\n" + "="*70)
    print("CONSOLIDATING RESULTS")
    print("="*70)
    
    df = pd.DataFrame(all_results)
    
    # Reorder columns
    column_order = [
        'experiment',
        'feature_config',
        'best_val_dice',
        'best_val_iou',
        'best_val_accuracy',
        'best_val_precision',
        'best_val_recall',
        'best_epoch',
        'training_time_minutes',
        'total_params',
        'trainable_params'
    ]
    df = df[column_order]
    
    # Sort by performance
    df = df.sort_values('best_val_dice', ascending=False)
    
    # Save results
    output_path = Path(output_dir)
    
    csv_path = output_path / 'ablation_study_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")
    
    json_path = output_path / 'ablation_study_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Results saved to {json_path}")
    
    # Print results table
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    return df


def analyze_ablation_results(results_dir='experiments/results/ablation_hybrid'):
    """
    Analyze ablation study results and generate insights.
    """
    
    print("\n" + "="*70)
    print("ABLATION STUDY ANALYSIS")
    print("="*70)
    
    # Load results
    results_path = Path(results_dir) / 'ablation_study_results.csv'
    
    if not results_path.exists():
        print(f"✗ Results not found at {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    
    # Get baseline and best performance
    cnn_only = df[df['experiment'] == 'cnn_only'].iloc[0]
    dinov3_only = df[df['experiment'] == 'dinov3_only'].iloc[0]
    hybrid_concat = df[df['experiment'] == 'hybrid_concat'].iloc[0]
    hybrid_full = df[df['experiment'] == 'hybrid_full'].iloc[0]
    
    print("\n1. COMPONENT CONTRIBUTIONS")
    print("-" * 70)
    
    print(f"\nCNN-Only Baseline:     {cnn_only['best_val_dice']:.4f} Dice")
    print(f"DINOv3-Only Baseline:  {dinov3_only['best_val_dice']:.4f} Dice")
    print(f"Hybrid (Concat):       {hybrid_concat['best_val_dice']:.4f} Dice")
    print(f"Hybrid (Full):         {hybrid_full['best_val_dice']:.4f} Dice")
    
    # Calculate improvements
    print("\n2. IMPROVEMENTS OVER BASELINES")
    print("-" * 70)
    
    # Hybrid vs CNN-only
    improvement_cnn = ((hybrid_full['best_val_dice'] - cnn_only['best_val_dice']) / 
                       cnn_only['best_val_dice'] * 100)
    print(f"\nHybrid vs CNN-Only:    +{improvement_cnn:.2f}%")
    print(f"  Absolute gain: +{hybrid_full['best_val_dice'] - cnn_only['best_val_dice']:.4f} Dice")
    
    # Hybrid vs DINOv3-only
    improvement_dinov3 = ((hybrid_full['best_val_dice'] - dinov3_only['best_val_dice']) / 
                          dinov3_only['best_val_dice'] * 100)
    print(f"\nHybrid vs DINOv3-Only: +{improvement_dinov3:.2f}%")
    print(f"  Absolute gain: +{hybrid_full['best_val_dice'] - dinov3_only['best_val_dice']:.4f} Dice")
    
    # Cross-attention vs concatenation
    improvement_fusion = ((hybrid_full['best_val_dice'] - hybrid_concat['best_val_dice']) / 
                          hybrid_concat['best_val_dice'] * 100)
    print(f"\n3. FUSION STRATEGY IMPACT")
    print("-" * 70)
    print(f"Cross-Attention vs Concat: +{improvement_fusion:.2f}%")
    print(f"  Absolute gain: +{hybrid_full['best_val_dice'] - hybrid_concat['best_val_dice']:.4f} Dice")
    
    if improvement_fusion > 0:
        print("  ✓ Cross-attention fusion provides benefit")
    else:
        print("  ⚠️  Concatenation performs similarly (cross-attention may not be necessary)")
    
    # DINOv3 size comparison
    if 'hybrid_dinov3_small' in df['experiment'].values:
        small_dinov3 = df[df['experiment'] == 'hybrid_dinov3_small'].iloc[0]
        print(f"\n4. DINOV3 MODEL SIZE IMPACT")
        print("-" * 70)
        print(f"Small DINOv3:  {small_dinov3['best_val_dice']:.4f} Dice")
        print(f"Base DINOv3:   {hybrid_full['best_val_dice']:.4f} Dice")
        
        size_diff = ((hybrid_full['best_val_dice'] - small_dinov3['best_val_dice']) / 
                     small_dinov3['best_val_dice'] * 100)
        print(f"Base vs Small: +{size_diff:.2f}%")
    
    # Key findings
    print("\n5. KEY FINDINGS")
    print("-" * 70)
    
    best_model = df.iloc[0]
    print(f"\n✓ Best Configuration: {best_model['experiment'].upper()}")
    print(f"  Dice Score: {best_model['best_val_dice']:.4f}")
    print(f"  IoU Score:  {best_model['best_val_iou']:.4f}")
    
    if improvement_cnn > 0:
        print(f"\n✓ Hybrid architecture provides {improvement_cnn:.2f}% improvement over CNN-only")
        print("  → Pretrained DINOv3 features are beneficial")
    
    if improvement_dinov3 > 0:
        print(f"\n✓ Adding CNN branch improves {improvement_dinov3:.2f}% over DINOv3-only")
        print("  → Additional channels (beyond RGB) contain useful information")
    
    if improvement_fusion > 1:  # > 1% improvement
        print(f"\n✓ Cross-attention fusion adds {improvement_fusion:.2f}% over concatenation")
        print("  → Learned feature interaction is valuable")
    
    print("\n" + "="*70)
    
    return df


def compare_with_baselines(
    hybrid_results_dir='experiments/results/ablation_hybrid',
    baseline_results_dir='experiments/results/ablation'
):
    """
    Compare hybrid model results with baseline models (U-Net, DeepLabv3+).
    """
    
    print("\n" + "="*70)
    print("HYBRID vs BASELINE MODELS COMPARISON")
    print("="*70)
    
    # Load hybrid results
    hybrid_path = Path(hybrid_results_dir) / 'ablation_study_results.csv'
    if not hybrid_path.exists():
        print(f"✗ Hybrid results not found at {hybrid_path}")
        return None
    
    df_hybrid = pd.read_csv(hybrid_path)
    best_hybrid = df_hybrid.iloc[0]
    
    print(f"\nBest Hybrid Model: {best_hybrid['experiment']}")
    print(f"  Dice: {best_hybrid['best_val_dice']:.4f}")
    print(f"  IoU:  {best_hybrid['best_val_iou']:.4f}")
    
    # Try to load baseline results
    baseline_models = ['unet', 'deeplabv3plus']
    baseline_results = {}
    
    for model in baseline_models:
        baseline_path = Path(baseline_results_dir) / model / 'ablation_study_results.csv'
        if baseline_path.exists():
            df_baseline = pd.read_csv(baseline_path)
            best_config = df_baseline.sort_values('best_val_dice', ascending=False).iloc[0]
            baseline_results[model] = best_config
            
            print(f"\nBest {model.upper()}:")
            print(f"  Config: {best_config['feature_config']}")
            print(f"  Dice: {best_config['best_val_dice']:.4f}")
            print(f"  IoU:  {best_config['best_val_iou']:.4f}")
    
    # Compare performance
    if baseline_results:
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)
        
        for model_name, baseline in baseline_results.items():
            improvement = ((best_hybrid['best_val_dice'] - baseline['best_val_dice']) / 
                          baseline['best_val_dice'] * 100)
            
            print(f"\nHybrid vs {model_name.upper()}:")
            print(f"  Improvement: {improvement:+.2f}%")
            print(f"  Absolute: {best_hybrid['best_val_dice'] - baseline['best_val_dice']:+.4f} Dice")
            
            if improvement > 0:
                print(f"  ✓ Hybrid model outperforms {model_name}")
            else:
                print(f"  ⚠️  {model_name} still performs better")
    
    else:
        print("\n⚠️  No baseline results found for comparison")
        print(f"   Please run ablation studies for U-Net and DeepLabv3+ first")
    
    print("\n" + "="*70)
    
    return df_hybrid


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study for Hybrid PyTorch-TensorFlow DINOv3 model"
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default='all',
        choices=['all', 'cnn_only', 'dinov3_only', 'hybrid_concat', 'hybrid_full', 'hybrid_small'],
        help='Which experiment to run'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
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
        default='experiments/results/ablation_hybrid',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--baseline_dir',
        type=str,
        default='experiments/results/ablation',
        help='Directory with baseline model results'
    )
    
    parser.add_argument(
        '--no_skip_existing',
        action='store_true',
        help='Re-run even if results exist'
    )
    
    parser.add_argument(
        '--analyze_only',
        action='store_true',
        help='Only analyze existing results (skip training)'
    )
    
    parser.add_argument(
        '--compare_only',
        action='store_true',
        help='Only compare with baselines (skip training)'
    )
    
    args = parser.parse_args()
    
    skip_existing = not args.no_skip_existing
    
    # Analyze existing results
    if args.analyze_only:
        print("Analyzing existing results only (no training)")
        analyze_ablation_results(args.output_dir)
        compare_with_baselines(args.output_dir, args.baseline_dir)
        return
    
    # Compare with baselines only
    if args.compare_only:
        print("Comparing with baselines only (no training)")
        compare_with_baselines(args.output_dir, args.baseline_dir)
        return
    
    # Run ablation study
    if args.experiment == 'all':
        print("Running full ablation study for Hybrid DINOv3 model")
        df = run_full_ablation_study(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            skip_existing=skip_existing
        )
        
        # Analyze results
        analyze_ablation_results(args.output_dir)
        
        # Compare with baselines
        compare_with_baselines(args.output_dir, args.baseline_dir)
        
    else:
        # Run single experiment
        print(f"Running single experiment: {args.experiment}")
        
        config = load_config()
        img_size = config['dataset']['working_resolution']
        
        if args.experiment == 'cnn_only':
            model = create_cnn_only_model(
                input_shape=(img_size[0], img_size[1], 10)
            )
        elif args.experiment == 'dinov3_only':
            model = create_dinov3_only_model(
                input_shape=(img_size[0], img_size[1], 10)
            )
        elif args.experiment == 'hybrid_concat':
            model = create_hybrid_concat_model(
                input_shape=(img_size[0], img_size[1], 10)
            )
        elif args.experiment == 'hybrid_full':
            model = create_hybrid_full_model(
                input_shape=(img_size[0], img_size[1], 10),
                dinov3_size='base'
            )
        elif args.experiment == 'hybrid_small':
            model = create_hybrid_full_model(
                input_shape=(img_size[0], img_size[1], 10),
                dinov3_model='facebook/dinov2-small',  # Use small DINOv2
                dinov3_size='small'
            )
        
        train_ablation_model(
            model=model,
            experiment_name=args.experiment,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            feature_config='all' if 'cnn' in args.experiment or 'hybrid' in args.experiment else 'rgb'
        )
    
    print("\n✓ Ablation study completed successfully!")


if __name__ == "__main__":
    main()
