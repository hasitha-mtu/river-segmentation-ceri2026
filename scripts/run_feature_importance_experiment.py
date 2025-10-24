"""
Main Script: Run Experiment 1 - Feature Importance Analysis
============================================================
Complete workflow for feature importance quantification
Based on Research Plan Section 3.4

Usage:
    python run_experiment1.py --data_dir data/processed --output_dir experiments/results
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
import pickle
import json

# Import custom modules
import sys
sys.path.append('src/data')
sys.path.append('src/experiment')
sys.path.append('src/utils')

from feature_extraction import FeatureExtractor
from feature_importance import FeatureImportanceAnalyzer
from visualization import FeatureImportanceVisualizer


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(data_dir: str, split: str = "all"):
    """
    Load images and masks from directory.
    
    Args:
        data_dir: Directory containing images/ and masks/ subdirectories
        split: Which split to load ("train", "val", "test", "all")
        
    Returns:
        images: List of RGB images
        masks: List of binary masks
        image_names: List of image filenames
    """
    data_path = Path(data_dir)
    
    if split == "all":
        image_dir = data_path / "images"
        mask_dir = data_path / "masks"
    else:
        image_dir = data_path / split / "images"
        mask_dir = data_path / split / "masks"
    
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    
    images = []
    masks = []
    image_names = []
    
    print(f"Loading {len(image_paths)} images from {image_dir}...")
    
    for img_path in tqdm(image_paths):
        mask_path = mask_dir / img_path.name
        
        if not mask_path.exists():
            print(f"Warning: No mask found for {img_path.name}, skipping")
            continue
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        images.append(img)
        masks.append(mask)
        image_names.append(img_path.name)
    
    print(f"Loaded {len(images)} image-mask pairs")
    
    return images, masks, image_names


def calculate_canopy_density(mask: np.ndarray, image: np.ndarray = None) -> float:
    """
    Calculate canopy density for an image.
    
    For now, uses a simple heuristic based on shadow/darkness.
    In practice, you should have pre-computed canopy densities from your quality assessment.
    
    Args:
        mask: Binary mask
        image: RGB image (optional, for better estimation)
        
    Returns:
        canopy_density: Value between 0 and 1
    """
    if image is not None:
        # Convert to HSV and use V channel as proxy for canopy
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        
        # Estimate canopy as dark regions (V < threshold)
        dark_pixels = np.sum(v_channel < 100)
        total_pixels = v_channel.size
        canopy_density = dark_pixels / total_pixels
    else:
        # Fallback: random assignment (replace with your actual data)
        canopy_density = np.random.uniform(0.1, 0.9)
    
    return canopy_density


def load_canopy_densities(metadata_path: str, image_names: list) -> list:
    """
    Load pre-computed canopy densities from metadata.
    
    Args:
        metadata_path: Path to metadata file (JSON or CSV)
        image_names: List of image names to get densities for
        
    Returns:
        List of canopy density values
    """
    metadata_file = Path(metadata_path)
    
    if not metadata_file.exists():
        print(f"Warning: Metadata file {metadata_path} not found")
        print("Using estimated canopy densities")
        return None
    
    # Load metadata
    if metadata_file.suffix == '.json':
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    elif metadata_file.suffix == '.csv':
        metadata = pd.read_csv(metadata_file)
        metadata = metadata.set_index('image_name').to_dict('index')
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_file.suffix}")
    
    # Extract canopy densities
    densities = []
    for name in image_names:
        if name in metadata:
            densities.append(metadata[name].get('canopy_density', 0.5))
        else:
            print(f"Warning: No metadata for {name}, using default 0.5")
            densities.append(0.5)
    
    return densities


def main(args):
    """Main experiment workflow"""
    
    print("="*70)
    print("EXPERIMENT 1: FEATURE IMPORTANCE QUANTIFICATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Config file: {args.config}")
    print()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load dataset
    print("\n" + "="*70)
    print("STEP 1: LOADING DATASET")
    print("="*70)
    
    images, masks, image_names = load_dataset(args.data_dir, split="all")
    
    # Load or calculate canopy densities
    if args.metadata:
        canopy_densities = load_canopy_densities(args.metadata, image_names)
    else:
        print("\nCalculating canopy densities...")
        canopy_densities = [calculate_canopy_density(m, img) for m, img in zip(masks, images)]
    
    print(f"\nCanopy density statistics:")
    print(f"  Mean: {np.mean(canopy_densities):.3f}")
    print(f"  Std: {np.std(canopy_densities):.3f}")
    print(f"  Min: {np.min(canopy_densities):.3f}")
    print(f"  Max: {np.max(canopy_densities):.3f}")
    
    # Step 2: Initialize feature extractor
    print("\n" + "="*70)
    print("STEP 2: INITIALIZING FEATURE EXTRACTOR")
    print("="*70)
    
    extractor = FeatureExtractor(config=config.get('features', {}))
    print(f"Extracting {len(extractor.feature_names)} features:")
    for i, name in enumerate(extractor.feature_names):
        feature_type = "Luminance" if i < 8 else "Chrominance"
        print(f"  {i+1:2d}. {name:20s} ({feature_type})")
    
    # Step 3: Run feature importance analysis
    print("\n" + "="*70)
    print("STEP 3: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    analyzer = FeatureImportanceAnalyzer(extractor, config=config)
    
    # Overall analysis
    print("\n--- Overall Analysis (All Images) ---")
    X, y = analyzer.prepare_data_for_rf(
        images, 
        masks, 
        max_samples_per_image=config['experiment_feature_importance'].get('max_samples', 10000)
    )
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    n_estimators = config['experiment_feature_importance'].get('n_estimators', 100)
    rf = analyzer.train_random_forest(X_train, y_train, X_test, y_test, n_estimators=n_estimators)
    
    # Calculate importances
    print("\n--- Calculating Feature Importances ---")
    mdi = analyzer.calculate_mdi_importance(rf)
    perm = analyzer.calculate_permutation_importance(rf, X_test, y_test, n_repeats=10)
    shap_vals, shap_df = analyzer.calculate_shap_values(rf, X_test, max_samples=1000)
    
    # Calculate luminance contribution
    print("\n--- Calculating Luminance Contribution ---")
    contributions = analyzer.calculate_luminance_contribution()
    
    # Step 4: Stratified analysis by canopy density
    if args.stratify:
        print("\n" + "="*70)
        print("STEP 4: STRATIFIED ANALYSIS BY CANOPY DENSITY")
        print("="*70)
        
        stratified_results = analyzer.analyze_by_canopy_density(
            images, masks, canopy_densities
        )
    
    # Step 5: Save results
    print("\n" + "="*70)
    print("STEP 5: SAVING RESULTS")
    print("="*70)
    
    analyzer.save_results(output_dir / "feature_importance")
    
    # Save summary report
    summary = {
        'n_images': len(images),
        'n_features': 18,
        'rf_performance': analyzer.results.get('rf_metrics', {}),
        'luminance_contribution': contributions,
        'top_10_features_mdi': list(mdi.head(10)['feature']),
        'top_10_features_perm': list(perm.head(10)['feature']),
        'top_10_features_shap': list(shap_df.head(10)['feature'])
    }
    
    with open(output_dir / "experiment1_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    # Step 6: Create visualizations
    if args.visualize:
        print("\n" + "="*70)
        print("STEP 6: CREATING VISUALIZATIONS")
        print("="*70)
        
        visualizer = FeatureImportanceVisualizer(
            analyzer.results,
            output_dir=output_dir / "figures"
        )
        
        print("\nGenerating figures...")
        visualizer.plot_feature_importance_comparison(save=True)
        visualizer.plot_luminance_contribution_by_method(save=True)
        
        if args.stratify:
            visualizer.plot_stratified_by_canopy(save=True)
        
        visualizer.plot_correlation_matrix(X_test, extractor.feature_names, save=True)
        visualizer.plot_top_features_across_methods(top_k=5, save=True)
        
        # Create comprehensive paper figure
        visualizer.create_paper_figure(save=True)
        
        print(f"\nFigures saved to: {output_dir / 'figures'}")
    
    # Step 7: Print final summary
    print("\n" + "="*70)
    print("EXPERIMENT 1 COMPLETED SUCCESSFULLY")
    print("="*70)
    
    print("\n📊 KEY FINDINGS:")
    print(f"  • Random Forest Accuracy: {analyzer.results['rf_metrics']['accuracy']:.4f}")
    print(f"  • Random Forest F1-Score: {analyzer.results['rf_metrics']['f1_score']:.4f}")
    
    for method, contrib in contributions.items():
        print(f"\n  • {method.replace('_', ' ').title()}:")
        print(f"      - Luminance:    {contrib['luminance_pct']:.2f}%")
        print(f"      - Chrominance:  {contrib['chrominance_pct']:.2f}%")
        print(f"      - L:C Ratio:    {contrib['ratio']:.2f}:1")
    
    print(f"\n  • Top 5 Features (MDI):")
    for i, row in mdi.head(5).iterrows():
        print(f"      {int(row['rank'])}. {row['feature']}")
    
    print("\n✓ All results and figures saved successfully!")
    print(f"✓ Check {output_dir} for complete results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 1: Feature Importance Analysis")
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Directory containing processed images and masks'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/results/experiment1',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        default=None,
        help='Path to metadata file with canopy densities (JSON or CSV)'
    )
    
    parser.add_argument(
        '--stratify',
        action='store_true',
        help='Perform stratified analysis by canopy density'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization plots'
    )
    
    args = parser.parse_args()
    
    main(args)