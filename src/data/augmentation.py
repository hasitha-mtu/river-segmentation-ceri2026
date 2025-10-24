"""
Data Augmentation for River Segmentation - TensorFlow
======================================================
Based on research plan Section 3.4: Dataset Enhancement
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Dict
import albumentations as A
import cv2

class RiverAugmentation:
    """
    Data augmentation pipeline for river segmentation.
    
    Based on research plan:
    - Rotation: ±30° (rivers at different angles)
    - Flips: horizontal, vertical
    - Brightness: ±30% (simulate lighting variation)
    - Shadow simulation: 40% probability
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.augmentation_pipeline = self._create_pipeline()
    
    def _create_pipeline(self) -> A.Compose:
        """
        Create Albumentations pipeline based on research plan.
        
        From plan Section 3.4:
        - Rotation ±30°
        - Horizontal & vertical flips
        - Brightness ±30%
        - Shadow simulation 40%
        - Augmentation factor: 8x (415 → 3,320 images)
        """
        
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
                shadow_roi=(0, 0.5, 1, 1),  # Lower half more likely
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=0.4
            ),
            
            # Additional augmentations for robustness
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            
            # Slight hue/saturation shifts (vegetation variation)
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),
        ])
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to image and mask.
        
        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W) or (H, W, 1)
            
        Returns:
            augmented_image, augmented_mask
        """
        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Apply augmentation
        augmented = self.augmentation_pipeline(image=image, mask=mask)
        
        return augmented['image'], augmented['mask']


# TensorFlow-native augmentation (alternative approach)
@tf.function
def tf_augment(image: tf.Tensor, mask: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow-native augmentation for faster pipeline.
    
    Args:
        image: (H, W, C) image tensor
        mask: (H, W, 1) mask tensor
        training: Whether in training mode
        
    Returns:
        augmented_image, augmented_mask
    """
    if not training:
        return image, mask
    
    # Concatenate for synchronized transforms
    combined = tf.concat([image, mask], axis=-1)
    
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        combined = tf.image.flip_left_right(combined)
    
    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        combined = tf.image.flip_up_down(combined)
    
    # Split back
    image = combined[:, :, :-1]
    mask = combined[:, :, -1:]
    
    # Random brightness (±30%)
    if tf.random.uniform(()) > 0.3:
        image = tf.image.random_brightness(image, max_delta=0.3)
    
    # Random contrast
    if tf.random.uniform(()) > 0.3:
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, mask


def create_multiscale_crops(
    image: np.ndarray,
    mask: np.ndarray,
    crop_sizes: list = [(2048, 2048), (1024, 1024)],
    target_size: Tuple[int, int] = (512, 512),
    n_crops_per_size: int = 2
) -> Tuple[list, list]:
    """
    Create multi-scale crops from original high-resolution images.
    
    From research plan Section 3.4:
    - Center crop: 2048×2048 → resize to 512×512
    - Multiple 1024×1024 crops → resize to 512×512
    - Creates 3-4× more training samples
    
    Args:
        image: Original high-res image (5280, 3956, 3)
        mask: Original high-res mask (5280, 3956)
        crop_sizes: List of crop sizes to extract
        target_size: Final size to resize crops to
        n_crops_per_size: Number of crops per size
        
    Returns:
        cropped_images: List of (H, W, 3) crops
        cropped_masks: List of (H, W) mask crops
    """
    import cv2
    
    h, w = image.shape[:2]
    cropped_images = []
    cropped_masks = []
    
    for crop_h, crop_w in crop_sizes:
        # Skip if crop size larger than image
        if crop_h > h or crop_w > w:
            continue
        
        for _ in range(n_crops_per_size):
            # Random crop location
            if crop_h == h and crop_w == w:
                # Full image
                y, x = 0, 0
            else:
                y = np.random.randint(0, h - crop_h + 1)
                x = np.random.randint(0, w - crop_w + 1)
            
            # Extract crop
            img_crop = image[y:y+crop_h, x:x+crop_w]
            mask_crop = mask[y:y+crop_h, x:x+crop_w]
            
            # Resize to target size
            img_resized = cv2.resize(img_crop, target_size[::-1], interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_crop, target_size[::-1], interpolation=cv2.INTER_NEAREST)
            
            cropped_images.append(img_resized)
            cropped_masks.append(mask_resized)
    
    return cropped_images, cropped_masks


class SeasonalVariationAugmentation:
    """
    Simulate seasonal canopy variation.
    
    From research plan Section 3.4:
    - March images → Add foliage overlay (simulate June)
    - June images → Remove foliage (simulate March)
    - Creates additional 200-300 synthetic samples
    """
    
    def __init__(self):
        self.foliage_transform = A.Compose([
            # Simulate adding foliage (March → June)
            A.ColorJitter(brightness=(-0.1, -0.05), saturation=(0.1, 0.2), p=1.0),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=-10, p=1.0),
        ])
        
        self.defoliation_transform = A.Compose([
            # Simulate removing foliage (June → March)
            A.ColorJitter(brightness=(0.05, 0.15), saturation=(-0.2, -0.1), p=1.0),
            A.HueSaturationValue(hue_shift_limit=-5, sat_shift_limit=-20, val_shift_limit=10, p=1.0),
        ])
    
    def add_foliage(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """March → June simulation"""
        augmented = self.foliage_transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    
    def remove_foliage(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """June → March simulation"""
        augmented = self.defoliation_transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']


# Example usage and testing
if __name__ == "__main__":
    
    # Test augmentation pipeline
    print("Testing River Augmentation Pipeline")
    print("=" * 50)
    
    # Create sample data
    sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    sample_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
    
    # Initialize augmentation
    augmenter = RiverAugmentation()
    
    # Apply augmentation
    aug_image, aug_mask = augmenter(sample_image, sample_mask)
    
    print(f"Original image shape: {sample_image.shape}")
    print(f"Augmented image shape: {aug_image.shape}")
    print(f"Original mask shape: {sample_mask.shape}")
    print(f"Augmented mask shape: {aug_mask.shape}")
    
    # Test multi-scale crops
    print("\nTesting Multi-scale Crops")
    print("=" * 50)
    
    # Simulate high-res image
    high_res_image = np.random.randint(0, 255, (3956, 5280, 3), dtype=np.uint8)
    high_res_mask = np.random.randint(0, 2, (3956, 5280), dtype=np.uint8) * 255
    
    crop_images, crop_masks = create_multiscale_crops(
        high_res_image,
        high_res_mask,
        crop_sizes=[(2048, 2048), (1024, 1024)],
        target_size=(512, 512),
        n_crops_per_size=2
    )
    
    print(f"Number of crops generated: {len(crop_images)}")
    print(f"Expected: 4 (2 sizes × 2 crops each)")
    print(f"Crop shape: {crop_images[0].shape}")
    
    # Test seasonal variation
    print("\nTesting Seasonal Variation Augmentation")
    print("=" * 50)
    
    seasonal_aug = SeasonalVariationAugmentation()
    
    # March → June
    june_sim, mask = seasonal_aug.add_foliage(sample_image, sample_mask)
    print(f"March → June simulation created")
    
    # June → March
    march_sim, mask = seasonal_aug.remove_foliage(sample_image, sample_mask)
    print(f"June → March simulation created")
    
    print("\n✓ All augmentation tests completed successfully!")

