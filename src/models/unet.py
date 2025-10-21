"""
U-Net Model Implementation in TensorFlow/Keras
===============================================
For river segmentation with multi-channel features.

Based on: Ronneberger et al. (2015) - "U-Net: Convolutional Networks 
for Biomedical Image Segmentation"

Supports variable input channels (3, 8, 10, 18) for ablation study.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys
from pathlib import Path
from typing import Tuple

sys.path.append('src')


def create_unet_model(
    input_shape: Tuple[int, int, int] = (512, 512, 18),
    num_classes: int = 1,
    filters: int = 64,
    dropout: float = 0.5
) -> keras.Model:
    """
    Create U-Net model for semantic segmentation.
    
    Args:
        input_shape: (H, W, C) where C is number of feature channels
        num_classes: Number of output classes (1 for binary segmentation)
        filters: Number of filters in first layer (doubles each level)
        dropout: Dropout rate
        
    Returns:
        keras.Model
        
    Example:
        # For all 18 features
        model = create_unet_model(input_shape=(512, 512, 18))
        
        # For RGB baseline
        model = create_unet_model(input_shape=(512, 512, 3))
        
        # For luminance only
        model = create_unet_model(input_shape=(512, 512, 8))
    """
    
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Encoder (downsampling path)
    # Block 1
    c1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name='enc1_conv1')(inputs)
    c1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name='enc1_conv2')(c1)
    p1 = layers.MaxPooling2D((2, 2), name='enc1_pool')(c1)
    
    # Block 2
    c2 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same', name='enc2_conv1')(p1)
    c2 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same', name='enc2_conv2')(c2)
    p2 = layers.MaxPooling2D((2, 2), name='enc2_pool')(c2)
    
    # Block 3
    c3 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same', name='enc3_conv1')(p2)
    c3 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same', name='enc3_conv2')(c3)
    p3 = layers.MaxPooling2D((2, 2), name='enc3_pool')(c3)
    
    # Block 4
    c4 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same', name='enc4_conv1')(p3)
    c4 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same', name='enc4_conv2')(c4)
    p4 = layers.MaxPooling2D((2, 2), name='enc4_pool')(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same', name='bottleneck_conv1')(p4)
    c5 = layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same', name='bottleneck_conv2')(c5)
    c5 = layers.Dropout(dropout, name='bottleneck_dropout')(c5)
    
    # Decoder (upsampling path)
    # Block 6
    u6 = layers.Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same', name='dec4_upsample')(c5)
    u6 = layers.concatenate([u6, c4], name='dec4_concat')
    c6 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same', name='dec4_conv1')(u6)
    c6 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same', name='dec4_conv2')(c6)
    
    # Block 7
    u7 = layers.Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same', name='dec3_upsample')(c6)
    u7 = layers.concatenate([u7, c3], name='dec3_concat')
    c7 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same', name='dec3_conv1')(u7)
    c7 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same', name='dec3_conv2')(c7)
    
    # Block 8
    u8 = layers.Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same', name='dec2_upsample')(c7)
    u8 = layers.concatenate([u8, c2], name='dec2_concat')
    c8 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same', name='dec2_conv1')(u8)
    c8 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same', name='dec2_conv2')(c8)
    
    # Block 9
    u9 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name='dec1_upsample')(c8)
    u9 = layers.concatenate([u9, c1], name='dec1_concat')
    c9 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name='dec1_conv1')(u9)
    c9 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name='dec1_conv2')(c9)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', name='output')(c9)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs], name='U-Net')
    
    return model


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient metric for segmentation.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    return dice


def dice_loss(y_true, y_pred):
    """Dice loss for training"""
    return 1 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """
    Combined loss: BCE + Dice + Focal
    From research plan Section 3.7
    """
    # Binary crossentropy (per pixel)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)  # Average over all pixels
    
    # Dice loss (already a scalar)
    dice = dice_loss(y_true, y_pred)
    
    # Focal loss
    alpha = 0.25
    gamma = 2.0
    
    # Calculate focal loss per pixel
    bce_per_pixel = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_exp = tf.exp(-bce_per_pixel)
    focal = alpha * tf.pow(1 - bce_exp, gamma) * bce_per_pixel
    focal = tf.reduce_mean(focal)  # Average over all pixels
    
    # Combine (equal weights as per config)
    total_loss = bce + dice + focal
    
    return total_loss


def iou_metric(y_true, y_pred, threshold=0.5):
    """
    Intersection over Union (IoU) metric.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        threshold: Threshold for binary prediction
        
    Returns:
        IoU score
    """
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred_binary)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("U-Net Model - TensorFlow Implementation")
    print("="*70)
    
    # Test 1: Standard 3-channel RGB input
    print("\n1. Testing with RGB input (3 channels)")
    print("-" * 70)
    model_rgb = create_unet_model(
        input_shape=(512, 512, 3),
        num_classes=1
    )
    print(f"✓ Model created: {model_rgb.name}")
    print(f"✓ Input shape: {model_rgb.input_shape}")
    print(f"✓ Output shape: {model_rgb.output_shape}")
    print(f"✓ Total parameters: {model_rgb.count_params():,}")
    
    # Test 2: All 18 features
    print("\n2. Testing with all features (18 channels)")
    print("-" * 70)
    model_all = create_unet_model(
        input_shape=(512, 512, 18),
        num_classes=1
    )
    print(f"✓ Model created: {model_all.name}")
    print(f"✓ Input shape: {model_all.input_shape}")
    print(f"✓ Output shape: {model_all.output_shape}")
    print(f"✓ Total parameters: {model_all.count_params():,}")
    
    # Test 3: Luminance only (8 channels)
    print("\n3. Testing with luminance only (8 channels)")
    print("-" * 70)
    model_lum = create_unet_model(
        input_shape=(512, 512, 8),
        num_classes=1
    )
    print(f"✓ Model created: {model_lum.name}")
    print(f"✓ Input shape: {model_lum.input_shape}")
    print(f"✓ Output shape: {model_lum.output_shape}")
    
    # Test 4: Chrominance only (10 channels)
    print("\n4. Testing with chrominance only (10 channels)")
    print("-" * 70)
    model_chr = create_unet_model(
        input_shape=(512, 512, 10),
        num_classes=1
    )
    print(f"✓ Model created: {model_chr.name}")
    print(f"✓ Input shape: {model_chr.input_shape}")
    print(f"✓ Output shape: {model_chr.output_shape}")
    
    # Test 5: Forward pass
    print("\n5. Testing forward pass")
    print("-" * 70)
    
    # Create dummy input
    dummy_input = np.random.rand(2, 512, 512, 18).astype(np.float32)
    
    # Forward pass
    output = model_all.predict(dummy_input, verbose=0)
    
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test 6: Test loss functions
    print("\n6. Testing loss functions")
    print("-" * 70)
    
    # Create dummy predictions and ground truth with proper shape
    y_true = tf.constant([[[[1.0]], [[0.0]], [[1.0]], [[0.0]]]], dtype=tf.float32)
    y_pred = tf.constant([[[[0.9]], [[0.1]], [[0.8]], [[0.2]]]], dtype=tf.float32)
    
    print(f"Ground truth shape: {y_true.shape}")
    print(f"Prediction shape: {y_pred.shape}")
    
    # Calculate metrics
    dice_val = dice_coefficient(y_true, y_pred)
    iou_val = iou_metric(y_true, y_pred)
    combined_loss_val = combined_loss(y_true, y_pred)
    
    print(f"✓ Dice coefficient: {float(dice_val):.4f}")
    print(f"✓ IoU: {float(iou_val):.4f}")
    print(f"✓ Combined loss: {float(combined_loss_val):.4f}")
    
    # Test 7: Model compilation
    print("\n7. Testing model compilation")
    print("-" * 70)
    
    model_all.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[
            dice_coefficient,
            iou_metric,
            'binary_accuracy'
        ]
    )
    print(f"✓ Model compiled successfully")
    
    # Test 8: Test with batch prediction
    print("\n8. Testing batch prediction")
    print("-" * 70)
    
    batch_size = 4
    dummy_batch = np.random.rand(batch_size, 512, 512, 18).astype(np.float32)
    batch_output = model_all.predict(dummy_batch, verbose=0)
    
    print(f"✓ Batch input shape: {dummy_batch.shape}")
    print(f"✓ Batch output shape: {batch_output.shape}")
    print(f"✓ All outputs in [0, 1]: {(batch_output >= 0).all() and (batch_output <= 1).all()}")
    
    # Print model summary
    print("\n9. Model Architecture Summary")
    print("-" * 70)
    print("\nU-Net with 18-channel input:")
    model_all.summary()
    
    # Test 9: Compare parameter counts across configurations
    print("\n10. Parameter comparison across configurations")
    print("-" * 70)
    
    configs = [
        ("RGB (3 ch)", (512, 512, 3)),
        ("Luminance (8 ch)", (512, 512, 8)),
        ("Chrominance (10 ch)", (512, 512, 10)),
        ("All features (18 ch)", (512, 512, 18))
    ]
    
    print(f"\n{'Configuration':<25} {'Parameters':>15} {'Size (MB)':>12}")
    print("-" * 55)
    
    for name, shape in configs:
        model_temp = create_unet_model(input_shape=shape)
        params = model_temp.count_params()
        size_mb = params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)
        print(f"{name:<25} {params:>15,} {size_mb:>12.2f}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
    print("\nU-Net Features:")
    print("  • Encoder-decoder architecture with skip connections")
    print("  • 5 encoder blocks with max pooling")
    print("  • 4 decoder blocks with transposed convolution")
    print("  • Support for variable input channels (3, 8, 10, 18)")
    print("  • Combined loss function (BCE + Dice + Focal)")
    print("  • Custom metrics (Dice coefficient, IoU)")
    print("  • Dropout in bottleneck for regularization")
    
    print("\nUsage Example:")
    print("""
    from tf_model_example import create_unet_model, combined_loss, dice_coefficient, iou_metric
    
    # Create model
    model = create_unet_model(input_shape=(512, 512, 18))
    
    # Compile
    model.compile(
        optimizer='adam',
        loss=combined_loss,
        metrics=[dice_coefficient, iou_metric]
    )
    
    # Train
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=val_dataset
    )
    """)