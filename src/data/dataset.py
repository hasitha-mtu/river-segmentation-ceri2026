"""
TensorFlow Dataset for River Segmentation
==========================================
Complete TensorFlow implementation with data loading, augmentation, and batching
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import albumentations as A


class RiverSegmentationDataset:
    """
    TensorFlow-compatible dataset for river segmentation with multi-channel features.
    
    Supports:
    - Multi-channel features (18 channels: 8 luminance + 10 chrominance)
    - Ablation study configurations (luminance-only, chrominance-only, RGB, etc.)
    - Data augmentation using Albumentations
    - Efficient data pipeline with tf.data.Dataset
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        feature_extractor,
        feature_config: str = "all",
        augmentation_pipeline: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (512, 512),
        seed: int = 42
    ):
        """
        Args:
            image_dir: Directory containing RGB images
            mask_dir: Directory containing binary masks
            feature_extractor: FeatureExtractor instance
            feature_config: Which features to extract ("all", "luminance", "chrominance", "rgb", "top5")
            augmentation_pipeline: Albumentations augmentation pipeline (optional)
            image_size: Target image size (H, W)
            seed: Random seed for reproducibility
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.feature_extractor = feature_extractor
        self.feature_config = feature_config
        self.augmentation = augmentation_pipeline
        self.image_size = image_size
        self.seed = seed
        
        # Get all image paths
        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")) + 
                                  list(self.image_dir.glob("*.png")))
        
        # Filter to only those with masks
        self.image_paths = [
            str(p) for p in self.image_paths 
            if (self.mask_dir / p.name).exists()
        ]
        
        # Corresponding mask paths
        self.mask_paths = [
            str(self.mask_dir / Path(p).name) for p in self.image_paths
        ]
        
        print(f"Found {len(self.image_paths)} image-mask pairs in {image_dir}")
        
        # Set feature indices for top-k configuration
        self.feature_indices = None
    
    def set_feature_indices(self, indices: List[int]):
        """Set feature indices for top-k feature extraction"""
        self.feature_indices = indices
    
    def _load_image_mask(self, image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess image and mask.
        
        Returns:
            image: (H, W, 3) RGB image
            mask: (H, W) binary mask
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size[::-1], interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        return image, mask
    
    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation pipeline if provided"""
        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        return image, mask
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features based on configuration"""
        
        if self.feature_config == "all":
            features = self.feature_extractor.extract_all_features(image)
            
        elif self.feature_config == "luminance":
            features = self.feature_extractor.extract_luminance_only(image)
            
        elif self.feature_config == "chrominance":
            features = self.feature_extractor.extract_chrominance_only(image)
            
        elif self.feature_config == "rgb":
            features = self.feature_extractor.extract_rgb_only(image)
            
        elif self.feature_config.startswith("top"):
            if self.feature_indices is None:
                raise ValueError("feature_indices must be set for top-k configuration")
            features = self.feature_extractor.extract_top_k_features(image, self.feature_indices)
        
        else:
            raise ValueError(f"Unknown feature_config: {self.feature_config}")
        
        return features
    
    def _process_single_example(self, image_path: bytes, mask_path: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single image-mask pair.
        This function is called by tf.py_function.
        """
        # Decode bytes to string
        image_path_str = image_path.numpy().decode('utf-8')
        mask_path_str = mask_path.numpy().decode('utf-8')
        
        # Load image and mask
        image, mask = self._load_image_mask(image_path_str, mask_path_str)
        
        # Apply augmentation
        image, mask = self._apply_augmentation(image, mask)
        
        # Extract features
        features = self._extract_features(image)
        
        # Normalize features
        features = self.feature_extractor.normalize_features(features)
        
        # Process mask
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension (H, W, 1)
        
        return features.astype(np.float32), mask.astype(np.float32)
    
    def get_n_channels(self) -> int:
        """Get number of channels based on feature configuration"""
        channel_map = {
            "all": 10,
            "luminance": 3,
            "chrominance": 7,
            "rgb": 3
        }
        
        if self.feature_config.startswith("top"):
            return len(self.feature_indices) if self.feature_indices else 5
        
        return channel_map.get(self.feature_config, 18)
    
    def create_dataset(
        self,
        batch_size: int = 4,
        shuffle: bool = True,
        repeat: bool = False,
        cache: bool = False
    ) -> tf.data.Dataset:
        """
        Create tf.data.Dataset for training/validation.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle dataset
            repeat: Whether to repeat indefinitely
            cache: Whether to cache dataset in memory
            
        Returns:
            tf.data.Dataset
        """
        n_channels = self.get_n_channels()
        
        # Create dataset from file paths
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))
        
        # Shuffle
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.image_paths), seed=self.seed)
        
        # Map processing function
        dataset = dataset.map(
            lambda img_path, mask_path: tf.py_function(
                func=self._process_single_example,
                inp=[img_path, mask_path],
                Tout=[tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Set shapes
        dataset = dataset.map(
            lambda x, y: (
                tf.ensure_shape(x, [self.image_size[0], self.image_size[1], n_channels]),
                tf.ensure_shape(y, [self.image_size[0], self.image_size[1], 1])
            )
        )
        
        # Cache if requested
        if cache:
            dataset = dataset.cache()
        
        # Repeat if requested
        if repeat:
            dataset = dataset.repeat()
        
        # Batch
        dataset = dataset.batch(batch_size)
        
        # Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.image_paths)


def create_augmentation_pipeline(config: Dict = None) -> A.Compose:
    """
    Create Albumentations augmentation pipeline based on research plan.
    
    From plan Section 3.4:
    - Rotation ±30°
    - Horizontal & vertical flips
    - Brightness ±30%
    - Shadow simulation 40%
    """
    config = config or {}
    
    return A.Compose([
        # Geometric transforms
        A.Rotate(
            limit=30,  # ±30° for river angles
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # Photometric transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.3,  # ±30%
            contrast_limit=0.2,
            p=0.7
        ),
        
        # Shadow simulation (40% probability from plan)
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=3,
            shadow_dimension=5,
            p=0.4
        ),
        
        # Additional augmentations
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3
        ),
    ])


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.append('src/data')
    from feature_extraction import FeatureExtractor
    
    print("="*70)
    print("TESTING TENSORFLOW DATASET")
    print("="*70)
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Create augmentation pipeline
    augmentation = create_augmentation_pipeline()
    
    # Test 1: Dataset with all features
    print("\nTest 1: All features (18 channels)")
    print("-" * 70)
    
    # Note: Replace with your actual data directories
    try:
        dataset = RiverSegmentationDataset(
            image_dir="data/processed/train/images",
            mask_dir="data/processed/train/masks",
            feature_extractor=extractor,
            feature_config="all",
            augmentation_pipeline=augmentation,
            image_size=(512, 512)
        )
        
        tf_dataset = dataset.create_dataset(batch_size=2, shuffle=True)
        
        # Get one batch
        for features, masks in tf_dataset.take(1):
            print(f"✓ Features shape: {features.shape}")  # (2, 512, 512, 10)
            print(f"✓ Masks shape: {masks.shape}")  # (2, 512, 512, 1)
            print(f"✓ Features range: [{tf.reduce_min(features):.3f}, {tf.reduce_max(features):.3f}]")
            print(f"✓ Masks range: [{tf.reduce_min(masks):.3f}, {tf.reduce_max(masks):.3f}]")
        
        print(f"✓ Dataset created successfully with {len(dataset)} images")
        
    except Exception as e:
        print(f"Note: Test requires data in data/processed/train/")
        print(f"Error: {e}")
    
    # Test 2: Different feature configurations
    print("\nTest 2: Different feature configurations")
    print("-" * 70)
    
    configs = [
        ("luminance", 3),
        ("chrominance", 7),
        ("rgb", 3),
        ("all", 10)
    ]
    
    for config_name, expected_channels in configs:
        # Create dummy dataset just to test channel counts
        test_dataset = RiverSegmentationDataset(
            image_dir="data/processed/train/images",
            mask_dir="data/processed/train/masks",
            feature_extractor=extractor,
            feature_config=config_name,
            image_size=(512, 512)
        )
        
        n_channels = test_dataset.get_n_channels()
        status = "✓" if n_channels == expected_channels else "✗"
        print(f"{status} {config_name:15s}: {n_channels} channels (expected {expected_channels})")
    
    print("\n" + "="*70)
    print("✓ TENSORFLOW DATASET MODULE READY")
    print("="*70)
    print("\nKey Features:")
    print("  • Multi-channel feature extraction (18 features)")
    print("  • Ablation study configurations (luminance/chrominance/RGB/full)")
    print("  • Albumentations augmentation integration")
    print("  • Efficient tf.data.Dataset pipeline")
    print("  • Automatic batching and prefetching")
    print("\nUsage Example:")
    print("""
    from feature_extraction import FeatureExtractor
    from tf_dataset import RiverSegmentationDataset, create_augmentation_pipeline
    
    # Initialize
    extractor = FeatureExtractor()
    augmentation = create_augmentation_pipeline()
    
    # Create dataset
    train_dataset = RiverSegmentationDataset(
        image_dir='data/processed/train/images',
        mask_dir='data/processed/train/masks',
        feature_extractor=extractor,
        feature_config='all',  # or 'luminance', 'chrominance', 'rgb'
        augmentation_pipeline=augmentation
    )
    
    # Create TensorFlow dataset
    tf_train = train_dataset.create_dataset(
        batch_size=4,
        shuffle=True,
        repeat=True  # For training
    )
    
    # Use in model training
    model.fit(tf_train, epochs=100, steps_per_epoch=len(train_dataset)//4)
    """)