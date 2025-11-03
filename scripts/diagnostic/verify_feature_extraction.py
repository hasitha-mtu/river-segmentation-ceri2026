"""
Feature Extraction Verification Script
Diagnoses issues with luminance/chrominance feature extraction
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats

import sys
sys.path.append('data')
sys.path.append('experiments')

# def extract_all_features(image_path):
#     """Extract all 13 features exactly as in training"""
#     img = cv2.imread(str(image_path))
#     if img is None:
#         raise ValueError(f"Could not load image: {image_path}")
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     # Convert to float32 for precision
#     img = img.astype(np.float32) / 255.0
    
#     # Initialize feature dictionary
#     features = {}
    
#     # RGB channels (3)
#     features['R'] = img[:,:,2]  # OpenCV is BGR
#     features['G'] = img[:,:,1]
#     features['B'] = img[:,:,0]
    
#     # Convert back to uint8 for color conversions
#     img_uint8 = (img * 255).astype(np.uint8)
    
#     # LAB color space
#     lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
#     features['L_LAB'] = lab[:,:,0].astype(np.float32) / 255.0  # Normalize to [0,1]
#     # features['a_LAB'] = lab[:,:,1].astype(np.float32) / 255.0
#     # features['b_LAB'] = lab[:,:,2].astype(np.float32) / 255.0
    
#     # # HSV color space
#     # hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)
#     # features['H_HSV'] = hsv[:,:,0].astype(np.float32) / 179.0  # Hue is [0,179]
#     # features['S_HSV'] = hsv[:,:,1].astype(np.float32) / 255.0
#     # features['V_HSV'] = hsv[:,:,2].astype(np.float32) / 255.0
    
#     # # YCbCr color space
#     # ycbcr = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2YCrCb)
#     # features['Y_YCbCr'] = ycbcr[:,:,0].astype(np.float32) / 255.0
#     # features['Cb_YCbCr'] = ycbcr[:,:,1].astype(np.float32) / 255.0
#     # features['Cr_YCbCr'] = ycbcr[:,:,2].astype(np.float32) / 255.0
    
#     # Derived luminance features
#     L_max = np.maximum.reduce([features['R'], features['G'], features['B']])
#     L_min = np.minimum.reduce([features['R'], features['G'], features['B']])
#     # L_mean = (features['R'] + features['G'] + features['B']) / 3.0
#     features['L_range'] = L_max - L_min


    
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5) 
#     features['L_texture'] = cv2.magnitude(sobelx, sobely) / 255.0 
    
#     # Normalized luminance (avoid division by zero)
#     # L_sum = features['R'] + features['G'] + features['B']
#     # features['L_normalized'] = np.where(L_sum > 1e-6, 
#     #                                     L_mean / (L_sum + 1e-6),
#     #                                     0.0)
    
#     return features

def extract_all_features(image_path):
    """Extract all 10 features exactly as in training"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to float32 for precision
    img = img.astype(np.float32) / 255.0
    
    # Initialize feature dictionary
    features = {}
    
    # RGB channels (3)
    features['R'] = img[:,:,2]  # OpenCV is BGR
    features['G'] = img[:,:,1]
    features['B'] = img[:,:,0]
    
    # Convert back to uint8 for color conversions
    img_uint8 = (img * 255).astype(np.uint8)
    
    # LAB color space
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
    features['L_LAB'] = lab[:,:,0].astype(np.float32) / 255.0  # Normalize to [0,1]
    
    # Derived luminance features
    L_max = np.maximum.reduce([features['R'], features['G'], features['B']])
    L_min = np.minimum.reduce([features['R'], features['G'], features['B']])
    features['L_range'] = L_max - L_min
    
    # L_TEXTURE - FIXED VERSION
    # Convert to grayscale properly (from BGR)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # Apply Sobel operators
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to [0, 1] using max normalization
    if gradient_magnitude.max() > 0:
        features['L_texture'] = gradient_magnitude / gradient_magnitude.max()
    else:
        features['L_texture'] = gradient_magnitude  # All zeros
    
    return features

def verify_feature_ranges(features):
    """Check if all features are in expected ranges"""
    print("\n" + "="*80)
    print("FEATURE RANGE VERIFICATION")
    print("="*80)
    
    issues = []
    
    for name, feature in features.items():
        min_val = feature.min()
        max_val = feature.max()
        mean_val = feature.mean()
        std_val = feature.std()
        
        # Check for issues
        has_nan = np.isnan(feature).any()
        has_inf = np.isinf(feature).any()
        out_of_range = (min_val < -0.1) or (max_val > 1.1)
        
        status = "✓"
        if has_nan or has_inf or out_of_range:
            status = "❌"
            if has_nan:
                issues.append(f"{name}: Contains NaN values")
            if has_inf:
                issues.append(f"{name}: Contains infinite values")
            if out_of_range:
                issues.append(f"{name}: Out of range [{min_val:.3f}, {max_val:.3f}]")
        
        print(f"{status} {name:15s} | Range: [{min_val:.3f}, {max_val:.3f}] | "
              f"Mean: {mean_val:.3f} ± {std_val:.3f}")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("\n✓ All features in valid range [0, 1]")
        return True

def compute_feature_correlations(features):
    """Compute correlation matrix between all features"""
    print("\n" + "="*80)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*80)
    
    # Flatten all features and create dataframe
    feature_vectors = {}
    for name, feature in features.items():
        feature_vectors[name] = feature.flatten()
    
    df = pd.DataFrame(feature_vectors)
    
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Separate luminance and chrominance features
    luminance_features = ['L_LAB', 'L_range', 'L_texture']
    chrominance_features = ['H_HSV', 'S_HSV', 'a_LAB', 
                           'b_LAB', 'Cb_YCbCr', 'Cr_YCbCr']
    
    # Find highly correlated luminance features
    print("\nLuminance Feature Correlations:")
    print("-" * 60)
    
    high_corr_pairs = []
    for i, feat1 in enumerate(luminance_features):
        for feat2 in luminance_features[i+1:]:
            if feat1 in corr_matrix.index and feat2 in corr_matrix.columns:
                corr = corr_matrix.loc[feat1, feat2]
                print(f"  {feat1:15s} ↔ {feat2:15s}: {corr:6.3f}", end="")
                if abs(corr) > 0.95:
                    print(" ⚠️  HIGHLY CORRELATED!")
                    high_corr_pairs.append((feat1, feat2, corr))
                else:
                    print()
    
    if high_corr_pairs:
        print("\n⚠️  REDUNDANT FEATURES DETECTED:")
        print("   These features may cause multicollinearity issues:")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"   - {feat1} ↔ {feat2}: {corr:.3f}")
        print("   Recommendation: Remove redundant features or use PCA")
    
    return corr_matrix, luminance_features, chrominance_features

def analyze_luminance_contrast(features, mask_path=None):
    """Analyze luminance contrast between water and non-water regions"""
    print("\n" + "="*80)
    print("LUMINANCE CONTRAST ANALYSIS")
    print("="*80)
    
    luminance_features = ['L_LAB', 'L_range', 'L_texture']
    
    if mask_path and Path(mask_path).exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = mask > 127  # Binary mask
            
            print("\nLuminance Statistics by Class:")
            print("-" * 60)
            print(f"{'Feature':<15s} | {'Water Mean':>10s} | {'Non-Water':>10s} | {'Contrast':>10s} | Status")
            print("-" * 60)
            
            for feat_name in luminance_features:
                if feat_name in features:
                    feat = features[feat_name]
                    
                    water_vals = feat[mask]
                    non_water_vals = feat[~mask]
                    
                    water_mean = water_vals.mean()
                    non_water_mean = non_water_vals.mean()
                    
                    # Compute contrast (normalized difference)
                    contrast = abs(water_mean - non_water_mean)
                    
                    # Check if contrast is sufficient
                    status = "✓" if contrast > 0.1 else "⚠️ LOW"
                    
                    print(f"{feat_name:<15s} | {water_mean:10.3f} | {non_water_mean:10.3f} | "
                          f"{contrast:10.3f} | {status}")
            
            # Statistical significance test
            print("\nStatistical Significance (t-test):")
            print("-" * 60)
            for feat_name in luminance_features:
                if feat_name in features:
                    feat = features[feat_name]
                    water_vals = feat[mask].flatten()
                    non_water_vals = feat[~mask].flatten()
                    
                    # Sample to avoid memory issues
                    if len(water_vals) > 10000:
                        water_vals = np.random.choice(water_vals, 10000, replace=False)
                    if len(non_water_vals) > 10000:
                        non_water_vals = np.random.choice(non_water_vals, 10000, replace=False)
                    
                    t_stat, p_value = stats.ttest_ind(water_vals, non_water_vals)
                    
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"  {feat_name:<15s}: t={t_stat:7.2f}, p={p_value:.2e} {sig}")
        else:
            print("⚠️  Could not load mask file")
    else:
        print("⚠️  No mask file provided - skipping contrast analysis")
        print("   Provide mask path to analyze water vs non-water luminance difference")

def visualize_features(features, output_dir):
    """Create comprehensive feature visualizations"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Feature distributions
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, (name, feature) in enumerate(features.items()):
        if idx < len(axes):
            ax = axes[idx]
            ax.hist(feature.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_title(name, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'feature_distributions.png'}")
    plt.close()
    
    # 2. Feature images (luminance only)
    luminance_names = ['L_LAB', 'L_range', 'L_texture']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, name in enumerate(luminance_names):
        if name in features:
            ax = axes[idx]
            im = ax.imshow(features[name], cmap='gray', vmin=0, vmax=1)
            ax.set_title(name, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'luminance_features_visual.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'luminance_features_visual.png'}")
    plt.close()
    
    # 3. RGB vs Luminance comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 8))
    
    # Original RGB
    rgb_img = np.stack([features['R'], features['G'], features['B']], axis=-1)
    axes[0].imshow(rgb_img)
    axes[0].set_title('Original RGB', fontweight='bold')
    axes[0].axis('off')
    
    # Three main luminance channels 
    for idx, (name, title) in enumerate([('L_LAB', 'L (LAB)'), 
                                          ('L_range', 'L_range'),
                                          ('L_texture', 'L_texture')]):
        if name in features:
            axes[idx+1].imshow(features[name], cmap='gray', vmin=0, vmax=1)
            axes[idx+1].set_title(title, fontweight='bold')
            axes[idx+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rgb_vs_luminance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'rgb_vs_luminance.png'}")
    plt.close()

def visualize_correlation_matrix(corr_matrix, luminance_features, chrominance_features, output_dir):
    """Create correlation matrix heatmap"""
    output_dir = Path(output_dir)
    
    # Reorder features: luminance first, then chrominance
    ordered_features = luminance_features + chrominance_features
    ordered_features = [f for f in ordered_features if f in corr_matrix.index]
    
    corr_ordered = corr_matrix.loc[ordered_features, ordered_features]
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_ordered, dtype=bool), k=1)
    
    sns.heatmap(corr_ordered, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title('Feature Correlation Matrix\n(Luminance features: top-left, Chrominance features: bottom-right)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add dividing lines between luminance and chrominance
    n_lum = len([f for f in luminance_features if f in ordered_features])
    plt.axhline(y=n_lum, color='red', linewidth=2, linestyle='--', alpha=0.7)
    plt.axvline(x=n_lum, color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix_detailed.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'correlation_matrix_detailed.png'}")
    plt.close()

def main(image_path, mask_path, output_dir):
    """Main verification pipeline"""
    print("\n" + "="*80)
    print("FEATURE EXTRACTION VERIFICATION PIPELINE")
    print("="*80)
    
    # Configuration
    print("\nConfiguration:")
    print("  Please provide the following paths:")
    
    # Get paths from user or use defaults
    # import sys
    # if len(sys.argv) > 1:
    #     image_path = sys.argv[1]
    #     mask_path = sys.argv[2] if len(sys.argv) > 2 else None
    #     output_dir = sys.argv[3] if len(sys.argv) > 3 else './verification_output'
    # else:
    #     # Use defaults or prompt
    #     image_path = input("  Sample image path (or press Enter for '/mnt/user-data/uploads/sample.jpg'): ").strip()
    #     if not image_path:
    #         image_path = '/mnt/user-data/uploads/sample.jpg'
        
    #     mask_path = input("  Ground truth mask path (optional, press Enter to skip): ").strip()
    #     if not mask_path:
    #         mask_path = None
        
    #     output_dir = input("  Output directory (or press Enter for './verification_output'): ").strip()
    #     if not output_dir:
    #         output_dir = './verification_output'
    
    print(f"\n  Image: {image_path}")
    print(f"  Mask: {mask_path if mask_path else 'Not provided'}")
    print(f"  Output: {output_dir}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"\n❌ ERROR: Image not found: {image_path}")
        print("\nUsage:")
        print("  python verify_feature_extraction.py <image_path> [mask_path] [output_dir]")
        return
    
    # Extract features
    print("\nExtracting features...")
    features = extract_all_features(image_path)
    print(f"✓ Extracted {len(features)} features")
    
    # Run verifications
    ranges_ok = verify_feature_ranges(features)
    corr_matrix, lum_features, chrom_features = compute_feature_correlations(features)
    analyze_luminance_contrast(features, mask_path)
    
    # Generate visualizations
    visualize_features(features, output_dir)
    visualize_correlation_matrix(corr_matrix, lum_features, chrom_features, output_dir)
    
    # Summary report
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    if ranges_ok:
        print("✓ Feature ranges: All valid")
    else:
        print("❌ Feature ranges: Issues detected (see above)")
    
    # Check for redundancy
    high_corr_count = 0
    for i in range(len(lum_features)):
        for j in range(i+1, len(lum_features)):
            if lum_features[i] in corr_matrix.index and lum_features[j] in corr_matrix.columns:
                if abs(corr_matrix.loc[lum_features[i], lum_features[j]]) > 0.95:
                    high_corr_count += 1
    
    if high_corr_count > 0:
        print(f"⚠️  Feature redundancy: {high_corr_count} highly correlated pairs (>0.95)")
    else:
        print("✓ Feature redundancy: No issues")
    
    print(f"\n✓ Verification complete! Results saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review visualizations in output directory")
    print("  2. Check correlation matrix for redundant features")
    print("  3. If issues found, fix feature extraction code")
    print("  4. Run ablation study again with corrected features")

if __name__ == '__main__':
    image_path =  'data/raw/images/DJI_20250728094628_0280_V_July.png'
    mask_path = 'data/raw/masks/DJI_20250728094628_0280_V_July.png'
    output_dir =  'experiments/diagnostic/feature_extraction'
    main(image_path, mask_path, output_dir)
