"""
Dataset Canopy Density Analysis
Analyzes the distribution of canopy coverage in your training dataset
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

import sys
sys.path.append('data')
sys.path.append('experiments')

def estimate_canopy_density(image_path):
    """
    Estimate canopy density from image darkness
    Returns: density (0=no canopy, 1=complete canopy)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]
    
    # Canopy density based on darkness
    # Lower V values = darker = more canopy
    mean_brightness = v_channel.mean() / 255.0
    canopy_density = 1 - mean_brightness
    
    # Also compute standard deviation (uniform darkness suggests heavy canopy)
    brightness_std = v_channel.std() / 255.0
    
    # Additional metrics
    dark_pixel_ratio = (v_channel < 100).sum() / v_channel.size
    very_dark_ratio = (v_channel < 50).sum() / v_channel.size
    
    return {
        'canopy_density': canopy_density,
        'mean_brightness': mean_brightness,
        'brightness_std': brightness_std,
        'dark_pixel_ratio': dark_pixel_ratio,
        'very_dark_ratio': very_dark_ratio
    }

def analyze_color_information(image_path):
    """
    Analyze how much color information is preserved
    Low saturation + narrow hue range = canopy filtering color
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    
    # Color metrics
    mean_saturation = s_channel.mean() / 255.0
    saturation_std = s_channel.std() / 255.0
    
    # Hue diversity (circular standard deviation)
    hue_variance = np.var(h_channel)
    
    # Check if colors are mostly green (forest canopy effect)
    green_hue_mask = (h_channel > 30) & (h_channel < 90)  # Green range in HSV
    green_ratio = green_hue_mask.sum() / h_channel.size
    
    return {
        'mean_saturation': mean_saturation,
        'saturation_std': saturation_std,
        'hue_variance': hue_variance,
        'green_ratio': green_ratio
    }

def analyze_luminance_chrominance_separation(image_path, mask_path=None):
    """
    Check if luminance and chrominance are separable in the image
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    img_float = img.astype(np.float32) / 255.0
    
    # Extract luminance (V from HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    luminance = hsv[:,:,2].astype(np.float32) / 255.0
    
    # Extract chrominance (S from HSV)
    chrominance = hsv[:,:,1].astype(np.float32) / 255.0
    
    # If mask available, compute separability
    if mask_path and Path(mask_path).exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask_binary = mask > 127
            
            # Luminance separability
            lum_water = luminance[mask_binary].mean()
            lum_non_water = luminance[~mask_binary].mean()
            lum_contrast = abs(lum_water - lum_non_water)
            
            # Chrominance separability
            chrom_water = chrominance[mask_binary].mean()
            chrom_non_water = chrominance[~mask_binary].mean()
            chrom_contrast = abs(chrom_water - chrom_non_water)
            
            # Which is more separable?
            separability_ratio = lum_contrast / (chrom_contrast + 1e-8)
            
            return {
                'luminance_contrast': lum_contrast,
                'chrominance_contrast': chrom_contrast,
                'separability_ratio': separability_ratio,
                'luminance_dominant': separability_ratio > 1.5
            }
    
    return None

def categorize_canopy_level(density):
    """Categorize canopy density into levels"""
    if density < 0.3:
        return 'sparse'
    elif density < 0.5:
        return 'moderate'
    elif density < 0.7:
        return 'dense'
    else:
        return 'very_dense'

def analyze_dataset(image_dir, mask_dir=None, sample_size=None):
    """
    Analyze entire dataset for canopy characteristics
    """
    print("\n" + "="*80)
    print("DATASET CANOPY DENSITY ANALYSIS")
    print("="*80)
    
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir) if mask_dir else None
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(image_dir.glob(f'**/*{ext}')))
    
    print(f"\nFound {len(image_paths)} images in {image_dir}")
    
    if sample_size and sample_size < len(image_paths):
        print(f"Sampling {sample_size} images for analysis...")
        image_paths = np.random.choice(image_paths, sample_size, replace=False).tolist()
    
    # Analyze each image
    results = []
    
    print("\nAnalyzing images...")
    for img_path in tqdm(image_paths):
        # Basic canopy metrics
        canopy_metrics = estimate_canopy_density(img_path)
        if canopy_metrics is None:
            continue
        
        # Color information
        color_metrics = analyze_color_information(img_path)
        
        # Find corresponding mask if available
        mask_path = None
        if mask_dir:
            mask_name = img_path.stem + '.png'
            potential_mask = mask_dir / mask_name
            if potential_mask.exists():
                mask_path = potential_mask
        
        # Separability analysis
        sep_metrics = analyze_luminance_chrominance_separation(img_path, mask_path)
        
        # Combine results
        result = {
            'image_name': img_path.name,
            'canopy_level': categorize_canopy_level(canopy_metrics['canopy_density']),
            **canopy_metrics,
            **color_metrics
        }
        
        if sep_metrics:
            result.update(sep_metrics)
        
        results.append(result)
    
    # Convert to dataframe
    df = pd.DataFrame(results)
    
    return df

def visualize_results(df, output_dir):
    """Create comprehensive visualizations of dataset characteristics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Canopy density distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    ax1 = axes[0, 0]
    ax1.hist(df['canopy_density'], bins=30, color='darkgreen', edgecolor='black', alpha=0.7)
    ax1.axvline(df['canopy_density'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {df['canopy_density'].mean():.3f}")
    ax1.axvline(df['canopy_density'].median(), color='blue', linestyle='--', 
                linewidth=2, label=f"Median: {df['canopy_density'].median():.3f}")
    ax1.set_xlabel('Canopy Density', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Canopy Density', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Canopy level counts
    ax2 = axes[0, 1]
    canopy_counts = df['canopy_level'].value_counts()
    colors_map = {'sparse': '#90EE90', 'moderate': '#FFD700', 
                  'dense': '#FF8C00', 'very_dense': '#8B0000'}
    colors = [colors_map.get(level, 'gray') for level in canopy_counts.index]
    
    bars = ax2.bar(canopy_counts.index, canopy_counts.values, color=colors, 
                   edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Canopy Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax2.set_title('Dataset Composition by Canopy Level', fontsize=14, fontweight='bold')
    
    # Add percentage labels
    total = len(df)
    for bar, count in zip(bars, canopy_counts.values):
        height = bar.get_height()
        percentage = (count / total) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # Brightness vs Saturation scatter
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['mean_brightness'], df['mean_saturation'], 
                         c=df['canopy_density'], cmap='RdYlGn_r', 
                         s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax3.set_xlabel('Mean Brightness', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Saturation', fontsize=12, fontweight='bold')
    ax3.set_title('Brightness vs Saturation (colored by canopy density)', 
                  fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Canopy Density', fontsize=10, fontweight='bold')
    
    # Dark pixel ratios
    ax4 = axes[1, 1]
    ax4.hist(df['dark_pixel_ratio'], bins=30, color='navy', 
             edgecolor='black', alpha=0.7, label='Dark pixels (<100)')
    ax4.hist(df['very_dark_ratio'], bins=30, color='darkred', 
             edgecolor='black', alpha=0.7, label='Very dark pixels (<50)')
    ax4.set_xlabel('Ratio of Dark Pixels', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Dark Pixels', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'canopy_density_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'canopy_density_analysis.png'}")
    plt.close()
    
    # 2. Separability analysis (if available)
    if 'luminance_contrast' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Luminance vs Chrominance contrast
        ax1 = axes[0]
        x = np.arange(len(df))
        width = 0.35
        
        ax1.bar(x - width/2, df['luminance_contrast'], width, 
                label='Luminance Contrast', color='gold', alpha=0.8)
        ax1.bar(x + width/2, df['chrominance_contrast'], width, 
                label='Chrominance Contrast', color='steelblue', alpha=0.8)
        ax1.set_xlabel('Image Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Contrast (|water - non-water|)', fontsize=12, fontweight='bold')
        ax1.set_title('Luminance vs Chrominance Separability', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        
        # Separability ratio histogram
        ax2 = axes[1]
        ax2.hist(df['separability_ratio'], bins=30, color='purple', 
                edgecolor='black', alpha=0.7)
        ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, 
                   label='Equal separability')
        ax2.axvline(1.5, color='orange', linestyle='--', linewidth=2, 
                   label='Luminance dominant threshold')
        ax2.set_xlabel('Separability Ratio (Lum/Chrom)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution of Separability Ratios', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'separability_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'separability_analysis.png'}")
        plt.close()
    
    # 3. Color information analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Saturation by canopy level
    ax1 = axes[0]
    df.boxplot(column='mean_saturation', by='canopy_level', ax=ax1)
    ax1.set_xlabel('Canopy Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Saturation', fontsize=12, fontweight='bold')
    ax1.set_title('Color Saturation by Canopy Level', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove default title
    
    # Green dominance
    ax2 = axes[1]
    ax2.scatter(df['canopy_density'], df['green_ratio'], 
               c=df['mean_saturation'], cmap='viridis', 
               s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax2.set_xlabel('Canopy Density', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Green Color Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Green Dominance vs Canopy Density', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'color_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'color_analysis.png'}")
    plt.close()

def print_summary_statistics(df):
    """Print comprehensive summary statistics"""
    print("\n" + "="*80)
    print("DATASET SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal images analyzed: {len(df)}")
    
    print("\n1. CANOPY DENSITY DISTRIBUTION:")
    print("-" * 60)
    print(f"  Mean:   {df['canopy_density'].mean():.3f}")
    print(f"  Median: {df['canopy_density'].median():.3f}")
    print(f"  Std:    {df['canopy_density'].std():.3f}")
    print(f"  Min:    {df['canopy_density'].min():.3f}")
    print(f"  Max:    {df['canopy_density'].max():.3f}")
    
    print("\n2. CANOPY LEVEL BREAKDOWN:")
    print("-" * 60)
    canopy_counts = df['canopy_level'].value_counts()
    for level in ['sparse', 'moderate', 'dense', 'very_dense']:
        count = canopy_counts.get(level, 0)
        percentage = (count / len(df)) * 100
        print(f"  {level.capitalize():12s}: {count:4d} images ({percentage:5.1f}%)")
    
    print("\n3. BRIGHTNESS CHARACTERISTICS:")
    print("-" * 60)
    print(f"  Mean brightness:      {df['mean_brightness'].mean():.3f}")
    print(f"  Dark pixel ratio:     {df['dark_pixel_ratio'].mean():.3f}")
    print(f"  Very dark ratio:      {df['very_dark_ratio'].mean():.3f}")
    
    print("\n4. COLOR INFORMATION:")
    print("-" * 60)
    print(f"  Mean saturation:      {df['mean_saturation'].mean():.3f}")
    print(f"  Green dominance:      {df['green_ratio'].mean():.3f}")
    
    if 'luminance_contrast' in df.columns:
        print("\n5. SEPARABILITY ANALYSIS:")
        print("-" * 60)
        print(f"  Mean luminance contrast:    {df['luminance_contrast'].mean():.3f}")
        print(f"  Mean chrominance contrast:  {df['chrominance_contrast'].mean():.3f}")
        print(f"  Mean separability ratio:    {df['separability_ratio'].mean():.3f}")
        
        lum_dominant = (df['luminance_dominant'] == True).sum()
        lum_percentage = (lum_dominant / len(df)) * 100
        print(f"  Luminance-dominant images:  {lum_dominant} ({lum_percentage:.1f}%)")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    mean_canopy = df['canopy_density'].mean()
    mean_sat = df['mean_saturation'].mean()
    
    if mean_canopy < 0.3:
        print("✓ Dataset is MOSTLY WELL-LIT (sparse canopy)")
        print("  → RGB should work well")
        print("  → Luminance-only may be insufficient")
        print("  → Color information is reliable")
    elif mean_canopy < 0.5:
        print("✓ Dataset has MODERATE CANOPY coverage")
        print("  → Mixed lighting conditions")
        print("  → Both RGB and luminance features useful")
        print("  → Feature combination may help")
    elif mean_canopy < 0.7:
        print("✓ Dataset has DENSE CANOPY coverage")
        print("  → Limited lighting in most images")
        print("  → Luminance features should be important")
        print("  → Color information may be degraded")
    else:
        print("✓ Dataset has VERY DENSE CANOPY coverage")
        print("  → Heavily shadowed scenes")
        print("  → Luminance features should dominate")
        print("  → Color information severely limited")
    
    if mean_sat < 0.3:
        print("\n⚠️  LOW SATURATION detected")
        print("  → Color information is weak across dataset")
        print("  → This supports luminance-prioritized approach")
    elif mean_sat > 0.5:
        print("\n✓ HIGH SATURATION detected")
        print("  → Color information is strong")
        print("  → This may explain why RGB performs well")
    
    if 'separability_ratio' in df.columns:
        mean_sep = df['separability_ratio'].mean()
        if mean_sep > 1.5:
            print("\n✓ LUMINANCE IS MORE SEPARABLE")
            print("  → Water/non-water distinction clearer in luminance")
            print("  → Supports luminance-prioritized hypothesis")
        elif mean_sep < 0.67:
            print("\n⚠️  CHROMINANCE IS MORE SEPARABLE")
            print("  → Water/non-water distinction clearer in color")
            print("  → This may explain unexpected ablation results!")
        else:
            print("\n~ BALANCED SEPARABILITY")
            print("  → Both luminance and chrominance useful")
            print("  → Feature combination should help")

def main(image_dir, mask_dir, output_dir, sample_size):
    """Main analysis pipeline"""
    print("\n" + "="*80)
    print("DATASET CANOPY DENSITY ANALYSIS PIPELINE")
    print("="*80)
    
    # Configuration
    # import sys
    # if len(sys.argv) > 1:
    #     image_dir = sys.argv[1]
    #     mask_dir = sys.argv[2] if len(sys.argv) > 2 else None
    #     output_dir = sys.argv[3] if len(sys.argv) > 3 else './dataset_analysis_output'
    #     sample_size = int(sys.argv[4]) if len(sys.argv) > 4 else None
    # else:
    #     image_dir = input("  Image directory path: ").strip()
    #     mask_dir = input("  Mask directory path (optional, press Enter to skip): ").strip()
    #     if not mask_dir:
    #         mask_dir = None
    #     output_dir = input("  Output directory (or press Enter for './dataset_analysis_output'): ").strip()
    #     if not output_dir:
    #         output_dir = './dataset_analysis_output'
        
    #     sample_input = input("  Sample size (or press Enter to analyze all): ").strip()
    #     sample_size = int(sample_input) if sample_input else None
    
    # Analyze dataset
    df = analyze_dataset(image_dir, mask_dir, sample_size)
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    df.to_csv(output_dir / 'dataset_analysis.csv', index=False)
    print(f"\n✓ Saved results to: {output_dir / 'dataset_analysis.csv'}")
    
    # Generate visualizations
    visualize_results(df, output_dir)
    
    # Print summary
    print_summary_statistics(df)
    
    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review visualizations to understand dataset characteristics")
    print("  2. Check if canopy density aligns with expectations")
    print("  3. Verify if luminance separability supports hypothesis")
    print("  4. Use findings to interpret ablation study results")

if __name__ == '__main__':
    image_dir =  'data/raw/images'
    mask_dir = 'data/raw/masks'
    output_dir =  'experiments/diagnostic/dataset_canopy'
    sample_size = None
    main(image_dir, mask_dir, output_dir, sample_size)
