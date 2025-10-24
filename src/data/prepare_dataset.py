"""
Dataset Preparation Script
===========================
Prepare River Bride dataset for experiments:
1. Resize images from 5280×3956 to 512×512
2. Split into train/val/test (60/20/20)
3. Stratify by canopy density
4. Create metadata file

Usage:
    python prepare_dataset.py --raw_dir data/raw --output_dir data/processed
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml


def load_quality_report(report_path: str) -> pd.DataFrame:
    """
    Load quality assessment report to get canopy densities.
    
    Args:
        report_path: Path to quality report CSV or metadata file
        
    Returns:
        DataFrame with image metadata
    """
    if Path(report_path).exists():
        if report_path.endswith('.csv'):
            return pd.read_csv(report_path)
        elif report_path.endswith('.json'):
            with open(report_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame.from_dict(data, orient='index')
    return None


def estimate_canopy_density(image: np.ndarray) -> float:
    """
    Estimate canopy density from image.
    
    Uses simple heuristic: darker pixels indicate more canopy coverage.
    
    Args:
        image: RGB image
        
    Returns:
        canopy_density: Estimated density (0-1)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    
    # Define thresholds for vegetation/shadow
    dark_threshold = 80  # Dark pixels (heavy shadow/dense canopy)
    moderate_threshold = 150  # Moderate pixels
    
    # Calculate percentages
    dark_pixels = np.sum(v_channel < dark_threshold)
    moderate_pixels = np.sum((v_channel >= dark_threshold) & (v_channel < moderate_threshold))
    total_pixels = v_channel.size
    
    # Weighted estimate
    canopy_density = (dark_pixels * 1.0 + moderate_pixels * 0.5) / total_pixels
    
    return canopy_density


def categorize_canopy_density(density: float) -> str:
    """
    Categorize canopy density into sparse/moderate/dense/very_dense.
    
    From research plan:
    - Sparse: <30%
    - Moderate: 30-60%
    - Dense: 60-80%
    - Very Dense: >80%
    """
    if density < 0.30:
        return 'sparse'
    elif density < 0.60:
        return 'moderate'
    elif density < 0.80:
        return 'dense'
    else:
        return 'very_dense'


def merge_small_categories(df, min_samples=10):
    """
    Merge small canopy categories to ensure enough samples for splitting.
    
    Args:
        df: DataFrame with 'canopy_category' column
        min_samples: Minimum samples required per category
        
    Returns:
        DataFrame with merged categories
    """
    category_counts = df['canopy_category'].value_counts()
    
    if category_counts.min() >= min_samples:
        return df  # No merging needed
    
    print(f"\n⚠ Some categories have fewer than {min_samples} samples.")
    print("  Merging small categories for better splitting...")
    
    df = df.copy()
    
    # Merge strategy: Combine adjacent categories
    # sparse + moderate → low_canopy
    # dense + very_dense → high_canopy
    
    def merge_categories(cat):
        if cat in ['sparse', 'moderate']:
            return 'low_canopy'
        else:  # dense, very_dense
            return 'high_canopy'
    
    df['canopy_category_merged'] = df['canopy_category'].apply(merge_categories)
    
    print("\n  Original categories:")
    print(df['canopy_category'].value_counts())
    print("\n  Merged categories:")
    print(df['canopy_category_merged'].value_counts())
    
    # Use merged categories for splitting
    df['canopy_category'] = df['canopy_category_merged']
    df = df.drop('canopy_category_merged', axis=1)
    
    return df


def resize_image(image: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target (height, width)
        
    Returns:
        Resized image
    """
    return cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """
    Resize mask to target size using nearest neighbor interpolation.
    
    Args:
        mask: Input mask
        target_size: Target (height, width)
        
    Returns:
        Resized mask
    """
    return cv2.resize(mask, target_size[::-1], interpolation=cv2.INTER_NEAREST)


def prepare_dataset(
    raw_dir: str,
    output_dir: str,
    target_size: tuple = (512, 512),
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    metadata_path: str = None,
    random_seed: int = 42
):
    """
    Prepare dataset from raw images.
    
    Args:
        raw_dir: Directory with raw images and masks
        output_dir: Output directory for processed dataset
        target_size: Target image size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        metadata_path: Optional path to existing metadata
        random_seed: Random seed for reproducibility
    """
    
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("DATASET PREPARATION")
    print("="*70)
    print(f"Raw directory: {raw_path}")
    print(f"Output directory: {output_path}")
    print(f"Target size: {target_size}")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print()
    
    # Get all image paths
    image_dir = raw_path / 'images'
    mask_dir = raw_path / 'masks'
    
    image_paths = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    
    print(f"Found {len(image_paths)} images")
    
    # Load or create metadata
    if metadata_path and Path(metadata_path).exists():
        print(f"Loading metadata from {metadata_path}")
        metadata_df = load_quality_report(metadata_path)
    else:
        print("No metadata found, will estimate canopy densities")
        metadata_df = None
    
    # Process all images and collect metadata
    print("\nProcessing images...")
    dataset_metadata = []
    
    for img_path in tqdm(image_paths):
        mask_path = mask_dir / img_path.name
        
        if not mask_path.exists():
            print(f"Warning: No mask for {img_path.name}, skipping")
            continue
        
        # Load image and mask
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Get or estimate canopy density
        if metadata_df is not None and img_path.name in metadata_df.index:
            canopy_density = metadata_df.loc[img_path.name, 'canopy_density']
        else:
            canopy_density = estimate_canopy_density(img)
        
        # Categorize
        canopy_category = categorize_canopy_density(canopy_density)
        
        # Calculate water coverage
        water_coverage = np.sum(mask > 0) / mask.size
        
        # Store metadata
        dataset_metadata.append({
            'image_name': img_path.name,
            'original_size': f"{img.shape[0]}x{img.shape[1]}",
            'canopy_density': canopy_density,
            'canopy_category': canopy_category,
            'water_coverage': water_coverage,
        })
    
    # Convert to DataFrame
    metadata_df = pd.DataFrame(dataset_metadata)
    
    print(f"\nProcessed {len(metadata_df)} images")
    print("\nCanopy density distribution:")
    print(metadata_df['canopy_category'].value_counts())
    
    # Check if we need to merge categories
    category_counts = metadata_df['canopy_category'].value_counts()
    min_samples = category_counts.min()
    
    # Need at least 2 samples per category for stratification
    if min_samples < 2:
        print(f"\n⚠ Merging categories (some have only {min_samples} sample(s))")
        metadata_df = merge_small_categories(metadata_df, min_samples=2)
        use_stratification = True
    elif min_samples < 10:
        print(f"\n⚠ Warning: Some categories have few samples (min: {min_samples})")
        print("  Consider merging categories for better splits")
        # Ask user or merge automatically
        metadata_df = merge_small_categories(metadata_df, min_samples=10)
        use_stratification = True
    else:
        use_stratification = True
    
    # Stratified split
    print("\nPerforming split...")
    
    if use_stratification:
        print("  Using stratified split (preserving canopy distribution)")
        # Split into train and temp (val+test)
        train_df, temp_df = train_test_split(
            metadata_df,
            train_size=train_ratio,
            random_state=random_seed,
            stratify=metadata_df['canopy_category']
        )
        
        # Check if we can stratify the second split
        temp_category_counts = temp_df['canopy_category'].value_counts()
        if temp_category_counts.min() >= 2:
            # Split temp into val and test
            val_size = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_size,
                random_state=random_seed,
                stratify=temp_df['canopy_category']
            )
        else:
            print("  ⚠ Cannot stratify val/test split, using random split for this step")
            val_size = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_size,
                random_state=random_seed
            )
    else:
        print("  Using random split (no stratification)")
        # Simple random split without stratification
        train_df, temp_df = train_test_split(
            metadata_df,
            train_size=train_ratio,
            random_state=random_seed
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            random_state=random_seed
        )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} images ({len(train_df)/len(metadata_df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} images ({len(val_df)/len(metadata_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} images ({len(test_df)/len(metadata_df)*100:.1f}%)")
    
    # Verify stratification
    print("\nCanopy distribution per split:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{split_name}:")
        print(split_df['canopy_category'].value_counts())
    
    # Process and save images
    print("\nResizing and saving images...")
    
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            img_name = row['image_name']
            
            # Load original
            img = cv2.imread(str(image_dir / img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_dir / img_name), cv2.IMREAD_GRAYSCALE)
            
            # Resize
            img_resized = resize_image(img, target_size)
            mask_resized = resize_mask(mask, target_size)
            
            # Save
            cv2.imwrite(
                str(output_path / split_name / 'images' / img_name),
                cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                str(output_path / split_name / 'masks' / img_name),
                mask_resized
            )
    
    # Save metadata
    print("\nSaving metadata...")
    
    for split_name, split_df in splits.items():
        split_df.to_csv(output_path / f'{split_name}_metadata.csv', index=False)
    
    metadata_df.to_csv(output_path / 'full_metadata.csv', index=False)
    
    # Save split information
    split_info = {
        'train_images': train_df['image_name'].tolist(),
        'val_images': val_df['image_name'].tolist(),
        'test_images': test_df['image_name'].tolist(),
        'target_size': target_size,
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'random_seed': random_seed
    }
    
    with open(output_path / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Create summary
    summary = {
        'total_images': len(metadata_df),
        'train_images': len(train_df),
        'val_images': len(val_df),
        'test_images': len(test_df),
        'target_size': target_size,
        'canopy_distribution': metadata_df['canopy_category'].value_counts().to_dict(),
        'mean_canopy_density': float(metadata_df['canopy_density'].mean()),
        'std_canopy_density': float(metadata_df['canopy_density'].std()),
        'mean_water_coverage': float(metadata_df['water_coverage'].mean())
    }
    
    with open(output_path / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETED")
    print("="*70)
    print(f"\n✓ Processed {len(metadata_df)} images")
    print(f"✓ Resized to {target_size[0]}×{target_size[1]}")
    print(f"✓ Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    print(f"✓ Metadata saved to {output_path}")
    
    print("\n📊 Dataset Summary:")
    print(f"  • Total images: {summary['total_images']}")
    print(f"  • Mean canopy density: {summary['mean_canopy_density']:.3f} ± {summary['std_canopy_density']:.3f}")
    print(f"  • Mean water coverage: {summary['mean_water_coverage']:.3f}")
    print(f"\n  • Canopy categories:")
    for cat, count in summary['canopy_distribution'].items():
        print(f"      - {cat.capitalize()}: {count} images")


def main():
    parser = argparse.ArgumentParser(description="Prepare river segmentation dataset")
    
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/raw',
        help='Directory containing raw images and masks'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Output directory for processed dataset'
    )
    
    parser.add_argument(
        '--target_size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='Target image size (height width)'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.60,
        help='Training set ratio'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.20,
        help='Validation set ratio'
    )
    
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.20,
        help='Test set ratio'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        default=None,
        help='Path to existing metadata file (CSV or JSON)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    prepare_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        metadata_path=args.metadata,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()