"""
DeepLabv3+ Implementation in TensorFlow/Keras
==============================================
For river segmentation with multi-channel features.

Based on: Chen et al. (2018) - "Encoder-Decoder with Atrous Separable 
Convolution for Semantic Image Segmentation"

Supports variable input channels (3, 8, 10, 18) for ablation study.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional


class DeepLabV3Plus:
    """
    DeepLabv3+ model for semantic segmentation.
    
    Architecture:
    - Encoder: ResNet50 or ResNet101 backbone with atrous convolution
    - ASPP: Atrous Spatial Pyramid Pooling
    - Decoder: Upsampling with skip connections
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (512, 512, 18),
        num_classes: int = 1,
        backbone: str = 'resnet50',
        output_stride: int = 16,
        weights: Optional[str] = 'imagenet',
        activation: str = 'sigmoid'
    ):
        """
        Args:
            input_shape: (H, W, C) input shape
            num_classes: Number of output classes (1 for binary)
            backbone: Backbone architecture ('resnet50', 'resnet101')
            output_stride: Output stride (8 or 16)
            weights: Pretrained weights ('imagenet' or None)
            activation: Final activation ('sigmoid' for binary, 'softmax' for multi-class)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.output_stride = output_stride
        self.weights = weights
        self.activation = activation
        
        # For non-standard input channels, we can't use ImageNet weights directly
        if input_shape[2] != 3 and weights == 'imagenet':
            print(f"Warning: Input has {input_shape[2]} channels, ImageNet weights "
                  f"will be adapted from 3 channels")
    
    def build_model(self) -> Model:
        """Build complete DeepLabv3+ model"""
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Build encoder (backbone + ASPP)
        encoder_output, low_level_features = self._build_encoder(inputs)
        
        # Build decoder
        x = self._build_decoder(encoder_output, low_level_features)
        
        # Final upsampling to original size using Lambda with tf.image.resize
        target_h, target_w = self.input_shape[0], self.input_shape[1]
        x = layers.Lambda(
            lambda x: tf.image.resize(x, (target_h, target_w), method='bilinear'),
            name='final_upsample'
        )(x)
        
        outputs = layers.Conv2D(
            self.num_classes,
            kernel_size=1,
            padding='same',
            activation=self.activation,
            name='output'
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='DeepLabV3Plus')
        
        return model
    
    def _build_encoder(self, inputs):
        """
        Build encoder with backbone and ASPP.
        
        Returns:
            encoder_output: Output from ASPP
            low_level_features: Low-level features for decoder skip connection
        """
        # Get backbone
        if self.backbone_name == 'resnet50':
            backbone = self._get_resnet50_backbone(inputs)
        elif self.backbone_name == 'resnet101':
            backbone = self._get_resnet101_backbone(inputs)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        # Extract features at different scales
        # low_level_features: early layer with high resolution
        # high_level_features: deep layer with semantic information
        low_level_features = backbone.get_layer('conv2_block3_out').output  # 1/4 resolution
        high_level_features = backbone.get_layer('conv4_block6_out').output  # 1/16 resolution
        
        # Apply ASPP module
        aspp_output = self._atrous_spatial_pyramid_pooling(high_level_features)
        
        return aspp_output, low_level_features
    
    def _get_resnet50_backbone(self, inputs):
        """Get ResNet50 backbone, adapting for different input channels"""
        
        if self.input_shape[2] == 3:
            # Standard 3-channel input, can use pretrained weights
            backbone = keras.applications.ResNet50(
                include_top=False,
                weights=self.weights,
                input_tensor=inputs
            )
        else:
            # Non-standard input channels
            # Strategy: Use a conv layer to project to 3 channels, then use pretrained ResNet50
            print(f"Adapting ResNet50 for {self.input_shape[2]} input channels")
            
            # Project input channels to 3 channels to use pretrained weights
            x = layers.Conv2D(
                3,
                kernel_size=1,
                padding='same',
                name='input_projection'
            )(inputs)
            
            # Now use standard ResNet50 with pretrained weights
            backbone = keras.applications.ResNet50(
                include_top=False,
                weights=self.weights if self.weights == 'imagenet' else None,
                input_tensor=x
            )
        
        # Freeze backbone if using pretrained weights
        if self.weights == 'imagenet':
            for layer in backbone.layers:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
        
        return backbone
    
    def _get_resnet101_backbone(self, inputs):
        """Get ResNet101 backbone"""
        
        if self.input_shape[2] == 3:
            backbone = keras.applications.ResNet101(
                include_top=False,
                weights=self.weights,
                input_tensor=inputs
            )
        else:
            # Similar adaptation as ResNet50 for non-3-channel inputs
            x = layers.Conv2D(
                64,
                kernel_size=7,
                strides=2,
                padding='same',
                name='conv1_custom'
            )(inputs)
            x = layers.BatchNormalization(name='bn_conv1')(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
            
            backbone_base = keras.applications.ResNet101(
                include_top=False,
                weights=self.weights if self.weights == 'imagenet' else None,
                input_shape=(None, None, 3)
            )
            
            for layer in backbone_base.layers[5:]:
                x = layer(x)
            
            backbone = Model(inputs=inputs, outputs=x)
        
        if self.weights == 'imagenet':
            for layer in backbone.layers:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
        
        return backbone
    
    def _atrous_spatial_pyramid_pooling(self, features):
        """
        Atrous Spatial Pyramid Pooling (ASPP) module.
        
        Applies parallel atrous convolutions with different rates to capture
        multi-scale context.
        
        Args:
            features: Input features from backbone
            
        Returns:
            ASPP output features
        """
        # Get static shape for upsampling
        input_shape = features.shape
        h, w = input_shape[1], input_shape[2]
        
        # Image pooling branch
        pool = layers.GlobalAveragePooling2D(keepdims=True)(features)
        pool = layers.Conv2D(256, 1, padding='same', use_bias=False)(pool)
        pool = layers.BatchNormalization()(pool)
        pool = layers.Activation('relu')(pool)
        
        # Upsample back to original spatial dimensions
        # Use Lambda layer with tf.image.resize for dynamic shapes
        pool = layers.Lambda(
            lambda x: tf.image.resize(x, (h, w), method='bilinear'),
            name='aspp_image_pooling'
        )(pool)
        
        # 1x1 convolution branch
        conv1 = layers.Conv2D(256, 1, padding='same', use_bias=False)(features)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Activation('relu')(conv1)
        
        # 3x3 atrous convolution branches with different rates
        # Rate 6
        conv2 = layers.Conv2D(
            256, 3, padding='same', dilation_rate=6, use_bias=False
        )(features)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Activation('relu')(conv2)
        
        # Rate 12
        conv3 = layers.Conv2D(
            256, 3, padding='same', dilation_rate=12, use_bias=False
        )(features)
        conv3 = layers.BatchNormalization()(conv3)
        conv3 = layers.Activation('relu')(conv3)
        
        # Rate 18
        conv4 = layers.Conv2D(
            256, 3, padding='same', dilation_rate=18, use_bias=False
        )(features)
        conv4 = layers.BatchNormalization()(conv4)
        conv4 = layers.Activation('relu')(conv4)
        
        # Concatenate all branches
        x = layers.Concatenate()([pool, conv1, conv2, conv3, conv4])
        
        # Final 1x1 convolution
        x = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        
        return x
    
    def _build_decoder(self, encoder_output, low_level_features):
        """
        Build decoder with skip connections.
        
        Args:
            encoder_output: Output from ASPP
            low_level_features: Low-level features from encoder
            
        Returns:
            Decoder output
        """
        # Get target spatial dimensions from low-level features
        target_h, target_w = low_level_features.shape[1], low_level_features.shape[2]
        
        # Process low-level features
        low_level = layers.Conv2D(
            48, 1, padding='same', use_bias=False
        )(low_level_features)
        low_level = layers.BatchNormalization()(low_level)
        low_level = layers.Activation('relu')(low_level)
        
        # Upsample encoder output using Lambda with tf.image.resize
        x = layers.Lambda(
            lambda x: tf.image.resize(x, (target_h, target_w), method='bilinear'),
            name='decoder_upsample'
        )(encoder_output)
        
        # Concatenate with low-level features
        x = layers.Concatenate()([x, low_level])
        
        # Decoder convolutions
        x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        
        return x


def create_deeplabv3plus(
    input_shape: Tuple[int, int, int] = (512, 512, 18),
    num_classes: int = 1,
    backbone: str = 'resnet50',
    weights: Optional[str] = 'imagenet',
    activation: str = 'sigmoid'
) -> Model:
    """
    Factory function to create DeepLabv3+ model.
    
    Args:
        input_shape: (H, W, C) input shape
        num_classes: Number of output classes
        backbone: Backbone architecture ('resnet50' or 'resnet101')
        weights: Pretrained weights ('imagenet' or None)
        activation: Final activation function
        
    Returns:
        Compiled Keras model
        
    Example:
        # For all 18 features
        model = create_deeplabv3plus(input_shape=(512, 512, 18))
        
        # For RGB baseline
        model = create_deeplabv3plus(input_shape=(512, 512, 3))
        
        # For luminance only
        model = create_deeplabv3plus(input_shape=(512, 512, 8))
    """
    deeplabv3plus = DeepLabV3Plus(
        input_shape=input_shape,
        num_classes=num_classes,
        backbone=backbone,
        weights=weights,
        activation=activation
    )
    
    model = deeplabv3plus.build_model()
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("DeepLabv3+ Model - TensorFlow Implementation")
    print("="*70)
    
    # Test 1: Standard 3-channel RGB input
    print("\n1. Testing with RGB input (3 channels)")
    print("-" * 70)
    model_rgb = create_deeplabv3plus(
        input_shape=(512, 512, 3),
        num_classes=1,
        backbone='resnet50',
        weights='imagenet'
    )
    print(f"✓ Model created: {model_rgb.name}")
    print(f"✓ Input shape: {model_rgb.input_shape}")
    print(f"✓ Output shape: {model_rgb.output_shape}")
    print(f"✓ Total parameters: {model_rgb.count_params():,}")
    
    # Test 2: All 18 features
    print("\n2. Testing with all features (18 channels)")
    print("-" * 70)
    model_all = create_deeplabv3plus(
        input_shape=(512, 512, 18),
        num_classes=1,
        backbone='resnet50',
        weights=None  # No pretrained weights for 18 channels
    )
    print(f"✓ Model created: {model_all.name}")
    print(f"✓ Input shape: {model_all.input_shape}")
    print(f"✓ Output shape: {model_all.output_shape}")
    print(f"✓ Total parameters: {model_all.count_params():,}")
    
    # Test 3: Luminance only (8 channels)
    print("\n3. Testing with luminance only (8 channels)")
    print("-" * 70)
    model_lum = create_deeplabv3plus(
        input_shape=(512, 512, 8),
        num_classes=1,
        backbone='resnet50',
        weights=None
    )
    print(f"✓ Model created: {model_lum.name}")
    print(f"✓ Input shape: {model_lum.input_shape}")
    print(f"✓ Output shape: {model_lum.output_shape}")
    
    # Test 4: Chrominance only (10 channels)
    print("\n4. Testing with chrominance only (10 channels)")
    print("-" * 70)
    model_chr = create_deeplabv3plus(
        input_shape=(512, 512, 10),
        num_classes=1,
        backbone='resnet50',
        weights=None
    )
    print(f"✓ Model created: {model_chr.name}")
    print(f"✓ Input shape: {model_chr.input_shape}")
    print(f"✓ Output shape: {model_chr.output_shape}")
    
    # Test forward pass
    print("\n5. Testing forward pass")
    print("-" * 70)
    import numpy as np
    
    # Create dummy input
    dummy_input = np.random.rand(2, 512, 512, 18).astype(np.float32)
    
    # Forward pass
    output = model_all.predict(dummy_input, verbose=0)
    
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Print model summary
    print("\n6. Model Architecture Summary")
    print("-" * 70)
    print("\nDeepLabv3+ with 18-channel input:")
    model_all.summary()
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
    print("\nDeepLabv3+ Features:")
    print("  • Atrous Spatial Pyramid Pooling (ASPP)")
    print("  • Multi-scale context aggregation")
    print("  • Skip connections from encoder to decoder")
    print("  • Support for variable input channels (3, 8, 10, 18)")
    print("  • Optional ImageNet pretrained backbone")
    print("  • ResNet50 or ResNet101 backbone options")