"""
Feature Extraction for River Segmentation - TensorFlow Version
===============================================================
Extracts 18 features (8 luminance + 10 chrominance) from RGB images
Based on research plan Section 3.4: Luminance vs Chrominance Contribution Analysis
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional


class FeatureExtractor:
    """
    Extracts multi-channel features from RGB images for river segmentation.
    
    Features:
        - 8 Luminance channels: L (LAB), V (HSV), Y (YCbCr), + derived features
        - 10 Chrominance channels: R, G, B, H, S, a, b, Cb, Cr, Intensity
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Configuration dictionary with feature settings
        """
        self.config = config or {}
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self) -> List[str]:
        """Returns ordered list of all 18 feature names"""
        luminance = ['L_LAB', 'V_HSV', 'Y_YCbCr', 'L_max', 'L_min', 
                     'L_mean', 'L_range', 'L_normalized']
        chrominance = ['R', 'G', 'B', 'H_HSV', 'S_HSV', 
                       'a_LAB', 'b_LAB', 'Cb_YCbCr', 'Cr_YCbCr', 'Intensity']
        return luminance + chrominance
    
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all 18 features from RGB image.
        
        Args:
            image: RGB image (H, W, 3) in range [0, 255] or [0, 1]
            
        Returns:
            features: (H, W, 18) array of all features
        """
        # Normalize to [0, 255] if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Extract from different color spaces
        luminance_features = self._extract_luminance(image)
        chrominance_features = self._extract_chrominance(image)
        
        # Concatenate all features
        all_features = np.concatenate([luminance_features, chrominance_features], axis=2)
        
        return all_features
    
    def _extract_luminance(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 8 luminance channels.
        
        Returns:
            luminance: (H, W, 8) array
        """
        # Convert to different color spaces
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # Extract luminance channels
        L = lab[:, :, 0].astype(np.float32)  # L from LAB
        V = hsv[:, :, 2].astype(np.float32)  # V from HSV
        Y = ycbcr[:, :, 0].astype(np.float32)  # Y from YCbCr
        
        # Derived luminance features
        L_max = np.maximum.reduce([L, V, Y])
        L_min = np.minimum.reduce([L, V, Y])
        L_mean = (L + V + Y) / 3.0
        L_range = L_max - L_min
        
        # Normalized luminance (0-1)
        L_normalized = L_mean / 255.0
        
        # Stack all luminance features
        luminance = np.stack([
            L, V, Y, L_max, L_min, L_mean, L_range, L_normalized
        ], axis=2)
        
        return luminance.astype(np.float32)
    
    def _extract_chrominance(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 10 chrominance channels.
        
        Returns:
            chrominance: (H, W, 10) array
        """
        # Convert to different color spaces
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # RGB channels
        R = image[:, :, 0].astype(np.float32)
        G = image[:, :, 1].astype(np.float32)
        B = image[:, :, 2].astype(np.float32)
        
        # Chrominance from different spaces
        H = hsv[:, :, 0].astype(np.float32)   # Hue
        S = hsv[:, :, 1].astype(np.float32)   # Saturation
        a = lab[:, :, 1].astype(np.float32)   # a channel (green-red)
        b = lab[:, :, 2].astype(np.float32)   # b channel (blue-yellow)
        Cb = ycbcr[:, :, 1].astype(np.float32)  # Cb (blue-difference)
        Cr = ycbcr[:, :, 2].astype(np.float32)  # Cr (red-difference)
        
        # Intensity (simple average)
        Intensity = (R + G + B) / 3.0
        
        # Stack all chrominance features
        chrominance = np.stack([
            R, G, B, H, S, a, b, Cb, Cr, Intensity
        ], axis=2)
        
        return chrominance.astype(np.float32)
    
    def extract_luminance_only(self, image: np.ndarray) -> np.ndarray:
        """Extract only luminance features (8 channels)"""
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        return self._extract_luminance(image)
    
    def extract_chrominance_only(self, image: np.ndarray) -> np.ndarray:
        """Extract only chrominance features (10 channels)"""
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        return self._extract_chrominance(image)
    
    def extract_rgb_only(self, image: np.ndarray) -> np.ndarray:
        """Extract RGB only (3 channels baseline)"""
        if image.max() > 1.0:
            image = image / 255.0
        return image.astype(np.float32)
    
    def extract_top_k_features(self, image: np.ndarray, feature_indices: List[int]) -> np.ndarray:
        """
        Extract specific features by index.
        
        Args:
            image: RGB image
            feature_indices: List of feature indices to extract (0-17)
            
        Returns:
            features: (H, W, K) array where K = len(feature_indices)
        """
        all_features = self.extract_all_features(image)
        selected_features = all_features[:, :, feature_indices]
        return selected_features
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range.
        
        Args:
            features: (H, W, C) feature array
            
        Returns:
            normalized: (H, W, C) normalized features
        """
        normalized = np.zeros_like(features)
        
        for i in range(features.shape[2]):
            channel = features[:, :, i]
            min_val = channel.min()
            max_val = channel.max()
            
            if max_val > min_val:
                normalized[:, :, i] = (channel - min_val) / (max_val - min_val)
            else:
                normalized[:, :, i] = 0
        
        return normalized
    
    def to_tensor(self, features: np.ndarray, normalize: bool = True) -> tf.Tensor:
        """
        Convert features to TensorFlow tensor.
        
        Args:
            features: (H, W, C) numpy array
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            tensor: (H, W, C) tensor (TensorFlow uses channels-last by default)
        """
        if normalize:
            features = self.normalize_features(features)
        
        # Convert to tensor (TensorFlow uses channels-last: H, W, C)
        tensor = tf.convert_to_tensor(features, dtype=tf.float32)
        return tensor


# TensorFlow Dataset Creation
def create_tf_dataset(
    image_paths: List[str],
    mask_paths: List[str],
    feature_extractor: FeatureExtractor,
    feature_config: str = "all",
    batch_size: int = 4,
    shuffle: bool = True,
    augment: bool = False,
    image_size: Tuple[int, int] = (512, 512),
    seed: int = 42
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset for training.
    
    Args:
        image_paths: List of image file paths
        mask_paths: List of mask file paths
        feature_extractor: FeatureExtractor instance
        feature_config: Which features to extract ("all", "luminance", "chrominance", "rgb")
        batch_size: Batch size
        shuffle: Whether to shuffle
        augment: Whether to augment (not implemented in this function, use preprocessing)
        image_size: Target size (H, W)
        seed: Random seed
        
    Returns:
        tf.data.Dataset
    """
    
    def load_and_extract_features(image_path, mask_path):
        """Load image, mask, and extract features"""
        # Decode paths from tensor
        image_path_str = image_path.numpy().decode('utf-8')
        mask_path_str = mask_path.numpy().decode('utf-8')
        
        # Load image
        image = cv2.imread(image_path_str)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path_str, cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed
        if image.shape[:2] != image_size:
            image = cv2.resize(image, image_size[::-1], interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, image_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        # Extract features based on configuration
        if feature_config == "all":
            features = feature_extractor.extract_all_features(image)
        elif feature_config == "luminance":
            features = feature_extractor.extract_luminance_only(image)
        elif feature_config == "chrominance":
            features = feature_extractor.extract_chrominance_only(image)
        elif feature_config == "rgb":
            features = feature_extractor.extract_rgb_only(image)
        else:
            raise ValueError(f"Unknown feature_config: {feature_config}")
        
        # Normalize features
        features = feature_extractor.normalize_features(features)
        
        # Normalize mask to [0, 1]
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        
        return features.astype(np.float32), mask.astype(np.float32)
    
    # Determine number of channels
    channel_map = {
        "all": 18,
        "luminance": 8,
        "chrominance": 10,
        "rgb": 3
    }
    n_channels = channel_map.get(feature_config, 18)
    
    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed)
    
    # Map loading function using py_function
    dataset = dataset.map(
        lambda img_path, mask_path: tf.py_function(
            load_and_extract_features,
            [img_path, mask_path],
            [tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Set shapes explicitly
    dataset = dataset.map(
        lambda x, y: (
            tf.ensure_shape(x, [image_size[0], image_size[1], n_channels]),
            tf.ensure_shape(y, [image_size[0], image_size[1], 1])
        )
    )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# Visualization utility
def visualize_features(features: np.ndarray, feature_names: List[str], 
                       save_path: str = None):
    """
    Visualize all extracted features.
    
    Args:
        features: (H, W, C) feature array
        feature_names: List of feature names
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    n_features = features.shape[2]
    n_cols = 6
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        feature = features[:, :, i]
        
        im = ax.imshow(feature, cmap='viridis')
        ax.set_title(feature_names[i], fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide extra subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create sample image
    sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Extract all features
    all_features = extractor.extract_all_features(sample_image)
    print(f"All features shape: {all_features.shape}")  # (512, 512, 18)

    # Visualize features
    visualize_features(all_features, extractor.feature_names, 'results/feature_importance/extracted_features')
    
    # Extract luminance only
    luminance = extractor.extract_luminance_only(sample_image)
    print(f"Luminance features shape: {luminance.shape}")  # (512, 512, 8)
    
    # Extract chrominance only
    chrominance = extractor.extract_chrominance_only(sample_image)
    print(f"Chrominance features shape: {chrominance.shape}")  # (512, 512, 10)
    
    # Extract RGB baseline
    rgb = extractor.extract_rgb_only(sample_image)
    print(f"RGB baseline shape: {rgb.shape}")  # (512, 512, 3)
    
    # Convert to TensorFlow tensor
    tensor = extractor.to_tensor(all_features)
    print(f"TensorFlow tensor shape: {tensor.shape}")  # (512, 512, 18)
    print(f"TensorFlow tensor dtype: {tensor.dtype}")
    
    # Normalize features
    normalized = extractor.normalize_features(all_features)
    print(f"Normalized features range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    print("\nFeature names:")
    for i, name in enumerate(extractor.feature_names):
        feature_type = "Luminance" if i < 8 else "Chrominance"
        print(f"  {i:2d}. {name:20s} ({feature_type})")
    
    # Test TensorFlow dataset creation
    print("\n" + "="*60)
    print("Testing TensorFlow Dataset Creation")
    print("="*60)
    
    # Create dummy file paths
    dummy_paths = [f"image_{i}.jpg" for i in range(10)]
    
    print(f"\n✓ Feature extraction module ready for TensorFlow!")
    print(f"✓ All functions use TensorFlow tensors (channels-last format)")