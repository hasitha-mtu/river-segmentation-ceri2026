"""
1. Canopy Density Analysis
2. Spatial Distribution
3. Annotation Quality
4. Image Quality Metrics
"""

import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import ndimage
from skimage import measure, filters, exposure
from sklearn.cluster import DBSCAN
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURATION
# =====================================================

class Config:
    """Configuration for dataset analysis"""
    # Paths - UPDATE THESE TO YOUR PATHS
    IMAGE_DIR = "data/images"         
    MASK_DIR = "data/masks"            
    OUTPUT_DIR = "analysis_results"     
    
    # Image specifications
    ORIGINAL_SIZE = (5280, 3956)      
    WORKING_SIZE = (512, 512)          
    
    # Analysis parameters
    CANOPY_THRESHOLD_NDVI = 0.2        # NDVI > 0.2 = vegetation,  generally, values above (0.2) indicate vegetation, with higher values signifying denser, healthier vegetation
    BLUR_THRESHOLD = 100               # Laplacian variance threshold
    MIN_RIVER_AREA = 1000              # Minimum river pixels
    
    # Months for temporal analysis
    MARCH_IDENTIFIER = "March"            # How March is marked in filenames
    JUNE_IDENTIFIER = "June"             # How June is marked in filenames
    JULY_IDENTIFIER = "July"             # How July is marked in filenames


# =====================================================
# 1. CANOPY DENSITY ANALYSIS
# =====================================================

class CanopyDensityAnalyzer:
    """Analyzes vegetation coverage (canopy density) in drone images"""
    
    def __init__(self, config):
        self.config = config
        
    def calculate_pseudo_ndvi(self, image):
        """
        Calculate pseudo-NDVI from RGB image
        NDVI = (NIR - Red) / (NIR + Red)
        For RGB: approximate using (Green - Red) / (Green + Red)
        """
        green = image[:, :, 1].astype(float)
        red = image[:, :, 0].astype(float)
        
        ndvi = (green - red) / (green + red + 1e-6)
        return ndvi
    
    def calculate_vegetation_index(self, image):
        """
        Multiple vegetation indicators for robust canopy detection
        """
        # Method 1: Pseudo-NDVI
        ndvi = self.calculate_pseudo_ndvi(image)
        
        # Method 2: ExG (Excess Green Index), a vegetation index that uses RGB (red, green, blue) color data from images to identify and quantify greenery which highlights green areas by giving more weight to the green channel than to the red and blue channels. 
        # This helps distinguish vegetation from other backgrounds like soil and shadows, making it useful for agricultural monitoring, canopy structure analysis, and mapping. 
        r = image[:, :, 0].astype(float) / 255.0
        g = image[:, :, 1].astype(float) / 255.0
        b = image[:, :, 2].astype(float) / 255.0
        exg = 2 * g - r - b
        
        # Method 3: HSV-based (high saturation + green hue)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        
        # Green hue range: 35-85 in OpenCV (0-179 scale)
        green_hue = ((h > 35) & (h < 85) & (s > 50)).astype(float)
        
        return ndvi, exg, green_hue
    
    def estimate_canopy_density(self, image):
        """
        Estimate canopy density percentage for an image
        Returns: percentage (0-100), vegetation mask, method scores
        """
        ndvi, exg, green_hue = self.calculate_vegetation_index(image)
        
        # Combine methods with weights
        # NDVI is most reliable
        veg_ndvi = (ndvi > self.config.CANOPY_THRESHOLD_NDVI).astype(float)
        veg_exg = (exg > 0.1).astype(float)
        
        # Combined vegetation mask (majority vote)
        veg_combined = ((veg_ndvi + veg_exg + green_hue) >= 2).astype(np.uint8)
        
        # Calculate percentage
        total_pixels = image.shape[0] * image.shape[1]
        veg_pixels = np.sum(veg_combined)
        canopy_percentage = (veg_pixels / total_pixels) * 100
        
        metrics = {
            'canopy_percentage': canopy_percentage,
            'ndvi_mean': np.mean(ndvi),
            'ndvi_std': np.std(ndvi),
            'exg_mean': np.mean(exg),
            'vegetation_pixels': veg_pixels,
            'total_pixels': total_pixels
        }
        
        return canopy_percentage, veg_combined, metrics
    
    def classify_canopy_density(self, percentage):
        """Classify canopy density into categories"""
        if percentage < 30:
            return "Sparse"
        elif percentage < 60:
            return "Moderate"
        elif percentage < 80:
            return "Dense"
        else:
            return "Very Dense"


# =====================================================
# 2. SPATIAL DISTRIBUTION ANALYSIS
# =====================================================

class SpatialDistributionAnalyzer:
    """Analyzes spatial coverage and distribution of dataset"""
    
    def __init__(self, config):
        self.config = config
    
    def extract_gps_from_filename(self, filename):
        """
        Extract GPS coordinates from filename if available
        Modify this based on your filename convention
        
        Example formats:
        - IMG_20250315_lat51.9123_lon-8.6234.jpg
        - DJI_0001_51.9123_-8.6234.jpg
        """
        # This is a placeholder - adjust based on your filenames
        # If no GPS in filenames, return None
        try:
            # Example parsing (modify as needed)
            parts = filename.split('_')
            for i, part in enumerate(parts):
                if 'lat' in part.lower():
                    lat = float(part.replace('lat', '').replace('Lat', ''))
                    lon = float(parts[i+1].replace('lon', '').replace('Lon', ''))
                    return lat, lon
        except:
            pass
        return None, None
    
    def estimate_coverage_from_masks(self, mask_paths):
        """
        Estimate spatial coverage by analyzing mask overlap
        """
        print("Analyzing spatial coverage from masks...")
        
        # Find unique river sections by clustering mask centroids
        centroids = []
        areas = []
        
        for mask_path in tqdm(mask_paths[:50], desc="Sampling masks"):  # Sample for speed
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            # Find river region centroid
            labeled = measure.label(mask > 127)
            regions = measure.regionprops(labeled)
            
            if regions:
                largest_region = max(regions, key=lambda r: r.area)
                centroids.append(largest_region.centroid)
                areas.append(largest_region.area)
        
        centroids = np.array(centroids)
        
        # Cluster to find unique locations
        if len(centroids) > 0:
            clustering = DBSCAN(eps=50, min_samples=2).fit(centroids)
            n_locations = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        else:
            n_locations = 0
        
        return {
            'estimated_unique_locations': n_locations,
            'total_images': len(mask_paths),
            'coverage_ratio': n_locations / len(mask_paths) if len(mask_paths) > 0 else 0,
            'mean_river_area': np.mean(areas) if areas else 0
        }
    
    def analyze_temporal_distribution(self, image_paths):
        """Analyze distribution across March,June and July"""
        march_count = 0
        june_count = 0
        july_count = 0
        other_count = 0
        
        for path in image_paths:
            filename = os.path.basename(path)
            if self.config.MARCH_IDENTIFIER in filename:
                march_count += 1
            elif self.config.JUNE_IDENTIFIER in filename:
                june_count += 1
            elif self.config.JULY_IDENTIFIER in filename:
                july_count += 1
            else:
                other_count += 1
        
        print(f'march_count: {march_count}')
        print(f'june_count: {june_count}')
        print(f'july_count: {july_count}')
        
        return {
            'march_images': march_count,
            'june_images': june_count,
            'july_images': july_count,
            'other_images': other_count,
            'total_images': len(image_paths)
        }


# =====================================================
# 3. ANNOTATION QUALITY CHECK
# =====================================================

class AnnotationQualityChecker:
    """Checks quality and consistency of annotations"""
    
    def __init__(self, config):
        self.config = config
    
    def check_mask_properties(self, mask):
        """Check basic mask properties"""
        # Check if binary
        unique_values = np.unique(mask)
        is_binary = len(unique_values) <= 2
        
        # Check for artifacts (very small isolated regions)
        labeled = measure.label(mask > 127)
        regions = measure.regionprops(labeled)
        
        small_artifacts = sum(1 for r in regions if r.area < 50)
        large_holes = sum(1 for r in regions if r.area < 500 and r.area > 50)
        
        # Check river continuity (should be mostly one connected component)
        main_component_area = max([r.area for r in regions]) if regions else 0
        total_area = np.sum(mask > 127)
        continuity_ratio = main_component_area / total_area if total_area > 0 else 0
        
        return {
            'is_binary': is_binary,
            'unique_values': unique_values.tolist(),
            'num_components': len(regions),
            'small_artifacts': small_artifacts,
            'large_holes': large_holes,
            'continuity_ratio': continuity_ratio,
            'river_area_pixels': total_area
        }
    
    def check_boundary_quality(self, mask):
        """Check annotation boundary smoothness and precision"""
        # Edge detection on mask
        edges = cv2.Canny(mask, 50, 150)
        edge_pixels = np.sum(edges > 0)
        
        # Calculate boundary smoothness (low gradient variance = smooth)
        gradients = np.gradient(mask.astype(float))
        gradient_variance = np.var(gradients)
        
        # Check for jagged edges (high frequency in boundary)
        boundary = cv2.dilate(mask, np.ones((3,3)), iterations=1) - mask
        boundary_roughness = np.sum(boundary > 0) / edge_pixels if edge_pixels > 0 else 0
        
        return {
            'edge_pixels': edge_pixels,
            'gradient_variance': gradient_variance,
            'boundary_roughness': boundary_roughness
        }
    
    def check_image_mask_alignment(self, image, mask):
        """Check if mask aligns well with image features"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find image edges
        image_edges = cv2.Canny(gray, 50, 150)
        
        # Find mask edges
        mask_edges = cv2.Canny(mask, 50, 150)
        
        # Calculate overlap
        edge_overlap = np.sum((image_edges > 0) & (mask_edges > 0))
        total_mask_edges = np.sum(mask_edges > 0)
        
        alignment_score = edge_overlap / total_mask_edges if total_mask_edges > 0 else 0
        
        return {
            'alignment_score': alignment_score,
            'mask_edges': total_mask_edges,
            'overlapping_edges': edge_overlap
        }


# =====================================================
# 4. IMAGE QUALITY METRICS
# =====================================================

class ImageQualityAnalyzer:
    """Analyzes technical quality of images"""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Classify
        if laplacian_var < 50:
            quality = "Blurry"
        elif laplacian_var < self.config.BLUR_THRESHOLD:
            quality = "Acceptable"
        else:
            quality = "Sharp"
        
        return laplacian_var, quality
    
    def calculate_brightness_distribution(self, image):
        """Analyze brightness and contrast"""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L_channel = lab[:, :, 0]
        
        metrics = {
            'mean_brightness': np.mean(L_channel),
            'std_brightness': np.std(L_channel),
            'min_brightness': np.min(L_channel),
            'max_brightness': np.max(L_channel),
            'dynamic_range': np.max(L_channel) - np.min(L_channel)
        }
        
        # Classify exposure
        if metrics['mean_brightness'] < 80:
            exposure = "Underexposed"
        elif metrics['mean_brightness'] > 170:
            exposure = "Overexposed"
        else:
            exposure = "Well Exposed"
        
        metrics['exposure_quality'] = exposure
        return metrics
    
    def calculate_shadow_coverage(self, image):
        """Estimate percentage of image in shadow"""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L_channel = lab[:, :, 0]
        
        # Shadow threshold (dark regions)
        shadow_mask = L_channel < 70
        shadow_percentage = (np.sum(shadow_mask) / shadow_mask.size) * 100
        
        return shadow_percentage
    
    def calculate_color_distribution(self, image):
        """Analyze color properties"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        metrics = {
            'mean_hue': np.mean(hsv[:, :, 0]),
            'mean_saturation': np.mean(hsv[:, :, 1]),
            'std_saturation': np.std(hsv[:, :, 1]),
            'color_diversity': np.std(hsv[:, :, 0])  # Hue variance
        }
        
        return metrics
    
    def detect_noise(self, image):
        """Estimate image noise level"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Estimate noise using high-frequency content
        noise_sigma = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))
        
        if noise_sigma < 5:
            noise_level = "Low"
        elif noise_sigma < 15:
            noise_level = "Moderate"
        else:
            noise_level = "High"
        
        return noise_sigma, noise_level


# =====================================================
# 5. COMPREHENSIVE DATASET ANALYZER
# =====================================================

class DatasetAnalyzer:
    """Main class that orchestrates all analyses"""
    
    def __init__(self, config):
        self.config = config
        self.canopy_analyzer = CanopyDensityAnalyzer(config)
        self.spatial_analyzer = SpatialDistributionAnalyzer(config)
        self.annotation_checker = AnnotationQualityChecker(config)
        self.quality_analyzer = ImageQualityAnalyzer(config)
        
        # Create output directory
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    def get_image_mask_pairs(self):
        """Get all image-mask pairs"""
        image_paths = sorted(list(Path(self.config.IMAGE_DIR).glob('*.jpg')) + 
                           list(Path(self.config.IMAGE_DIR).glob('*.png')))
        mask_paths = sorted(list(Path(self.config.MASK_DIR).glob('*.jpg')) + 
                          list(Path(self.config.MASK_DIR).glob('*.png')))
        
        # Match images to masks
        pairs = []
        for img_path in image_paths:
            # Find corresponding mask (adjust matching logic as needed)
            img_name = img_path.stem
            mask_path = None
            
            for m_path in mask_paths:
                if img_name in m_path.stem or m_path.stem in img_name:
                    mask_path = m_path
                    break
            
            if mask_path:
                pairs.append((str(img_path), str(mask_path)))
        
        print(f"Found {len(pairs)} image-mask pairs out of {len(image_paths)} images")
        return pairs
    
    def analyze_single_image(self, image_path, mask_path):
        """Analyze a single image-mask pair"""
        # Load image and mask
        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        
        # Resize to working size for analysis
        image_resized = cv2.resize(image, self.config.WORKING_SIZE)
        mask_resized = cv2.resize(mask, self.config.WORKING_SIZE)
        
        results = {
            'filename': os.path.basename(image_path),
            'image_path': image_path,
            'mask_path': mask_path
        }
        
        # 1. Canopy density
        canopy_pct, veg_mask, canopy_metrics = self.canopy_analyzer.estimate_canopy_density(image_resized)
        results['canopy_percentage'] = canopy_pct
        results['canopy_category'] = self.canopy_analyzer.classify_canopy_density(canopy_pct)
        results.update({f'canopy_{k}': v for k, v in canopy_metrics.items()})
        
        # 2. Annotation quality
        mask_props = self.annotation_checker.check_mask_properties(mask_resized)
        results.update({f'mask_{k}': v for k, v in mask_props.items()})
        
        boundary_quality = self.annotation_checker.check_boundary_quality(mask_resized)
        results.update({f'boundary_{k}': v for k, v in boundary_quality.items()})
        
        alignment = self.annotation_checker.check_image_mask_alignment(image_resized, mask_resized)
        results.update({f'alignment_{k}': v for k, v in alignment.items()})
        
        # 3. Image quality
        sharpness, sharp_quality = self.quality_analyzer.calculate_sharpness(image_resized)
        results['sharpness'] = sharpness
        results['sharpness_quality'] = sharp_quality
        
        brightness = self.quality_analyzer.calculate_brightness_distribution(image_resized)
        results.update({f'brightness_{k}': v for k, v in brightness.items()})
        
        results['shadow_percentage'] = self.quality_analyzer.calculate_shadow_coverage(image_resized)
        
        color_metrics = self.quality_analyzer.calculate_color_distribution(image_resized)
        results.update({f'color_{k}': v for k, v in color_metrics.items()})
        
        noise_sigma, noise_level = self.quality_analyzer.detect_noise(image_resized)
        results['noise_sigma'] = noise_sigma
        results['noise_level'] = noise_level
        
        return results
    
    def analyze_all_images(self, max_images=None):
        """
        Analyze all images in dataset
        
        Args:
            max_images: If set, only analyze first N images (for testing)
        """
        pairs = self.get_image_mask_pairs()
        
        if max_images:
            pairs = pairs[:max_images]
            print(f"Analyzing first {max_images} images for testing...")
        
        all_results = []
        
        for img_path, mask_path in tqdm(pairs, desc="Analyzing dataset"):
            result = self.analyze_single_image(img_path, mask_path)
            if result:
                all_results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Add temporal classification
        df['month'] = df['filename'].apply(lambda x: 
            'March' if self.config.MARCH_IDENTIFIER in x 
            else 'June' if self.config.JUNE_IDENTIFIER in x 
            else 'July' if self.config.JULY_IDENTIFIER in x 
            else 'Other')
        
        return df
    
    def generate_summary_statistics(self, df):
        """Generate summary statistics"""
        summary = {
            'total_images': len(df),
            'date_generated': datetime.now().isoformat(),
            
            # Temporal distribution
            'march_images': len(df[df['month'] == 'March']),
            'june_images': len(df[df['month'] == 'June']),
            'july_images': len(df[df['month'] == 'July']),
            
            # Canopy density
            'canopy_mean': df['canopy_percentage'].mean(),
            'canopy_std': df['canopy_percentage'].std(),
            'canopy_min': df['canopy_percentage'].min(),
            'canopy_max': df['canopy_percentage'].max(),
            
            # By category
            'sparse_count': len(df[df['canopy_category'] == 'Sparse']),
            'moderate_count': len(df[df['canopy_category'] == 'Moderate']),
            'dense_count': len(df[df['canopy_category'] == 'Dense']),
            'very_dense_count': len(df[df['canopy_category'] == 'Very Dense']),
            
            # Image quality
            'mean_sharpness': df['sharpness'].mean(),
            'sharp_images': len(df[df['sharpness_quality'] == 'Sharp']),
            'blurry_images': len(df[df['sharpness_quality'] == 'Blurry']),
            
            'mean_brightness': df['brightness_mean_brightness'].mean(),
            'underexposed': len(df[df['brightness_exposure_quality'] == 'Underexposed']),
            'overexposed': len(df[df['brightness_exposure_quality'] == 'Overexposed']),
            
            'mean_shadow_coverage': df['shadow_percentage'].mean(),
            
            # Annotation quality
            'mean_continuity': df['mask_continuity_ratio'].mean(),
            'mean_alignment': df['alignment_alignment_score'].mean(),
            'images_with_artifacts': len(df[df['mask_small_artifacts'] > 5]),
        }
        
        return summary
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Canopy density distribution
        ax1 = plt.subplot(3, 4, 1)
        df['canopy_percentage'].hist(bins=30, ax=ax1, color='forestgreen', edgecolor='black')
        ax1.axvline(df['canopy_percentage'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["canopy_percentage"].mean():.1f}%')
        ax1.set_xlabel('Canopy Coverage (%)', fontweight='bold')
        ax1.set_ylabel('Number of Images', fontweight='bold')
        ax1.set_title('Canopy Density Distribution', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Canopy by category
        ax2 = plt.subplot(3, 4, 2)
        category_counts = df['canopy_category'].value_counts()
        colors_cat = ['#90EE90', '#FFD700', '#FF8C00', '#8B4513']
        category_counts.plot(kind='bar', ax=ax2, color=colors_cat)
        ax2.set_xlabel('Canopy Category', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Images by Canopy Category', fontweight='bold', fontsize=14)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        for i, v in enumerate(category_counts):
            ax2.text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        # 3. March vs June vs July canopy
        ax3 = plt.subplot(3, 4, 3)
        df_march_june_july = df[df['month'].isin(['March', 'June', 'July'])]
        df_march_june_july.boxplot(column='canopy_percentage', by='month', ax=ax3)
        ax3.set_xlabel('Month', fontweight='bold')
        ax3.set_ylabel('Canopy Coverage (%)', fontweight='bold')
        ax3.set_title('Canopy Density: March vs June vs July', fontweight='bold', fontsize=14)
        plt.sca(ax3)
        plt.xticks([1, 2, 3], ['March', 'June', 'July'])
        
        # 4. Sharpness distribution
        ax4 = plt.subplot(3, 4, 4)
        df['sharpness'].hist(bins=30, ax=ax4, color='skyblue', edgecolor='black')
        ax4.axvline(self.config.BLUR_THRESHOLD, color='red', linestyle='--', 
                   label=f'Threshold: {self.config.BLUR_THRESHOLD}')
        ax4.set_xlabel('Sharpness (Laplacian Variance)', fontweight='bold')
        ax4.set_ylabel('Count', fontweight='bold')
        ax4.set_title('Image Sharpness Distribution', fontweight='bold', fontsize=14)
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Brightness distribution
        ax5 = plt.subplot(3, 4, 5)
        df['brightness_mean_brightness'].hist(bins=30, ax=ax5, color='gold', edgecolor='black')
        ax5.set_xlabel('Mean Brightness', fontweight='bold')
        ax5.set_ylabel('Count', fontweight='bold')
        ax5.set_title('Brightness Distribution', fontweight='bold', fontsize=14)
        ax5.grid(alpha=0.3)
        
        # 6. Shadow coverage
        ax6 = plt.subplot(3, 4, 6)
        df['shadow_percentage'].hist(bins=30, ax=ax6, color='gray', edgecolor='black')
        ax6.set_xlabel('Shadow Coverage (%)', fontweight='bold')
        ax6.set_ylabel('Count', fontweight='bold')
        ax6.set_title('Shadow Coverage Distribution', fontweight='bold', fontsize=14)
        ax6.grid(alpha=0.3)
        
        # 7. Annotation continuity
        ax7 = plt.subplot(3, 4, 7)
        df['mask_continuity_ratio'].hist(bins=30, ax=ax7, color='purple', edgecolor='black')
        ax7.set_xlabel('Continuity Ratio', fontweight='bold')
        ax7.set_ylabel('Count', fontweight='bold')
        ax7.set_title('River Mask Continuity', fontweight='bold', fontsize=14)
        ax7.grid(alpha=0.3)
        
        # 8. Canopy vs Shadow correlation
        ax8 = plt.subplot(3, 4, 8)
        ax8.scatter(df['canopy_percentage'], df['shadow_percentage'], 
                   alpha=0.6, c=df['canopy_percentage'], cmap='Greens')
        ax8.set_xlabel('Canopy Coverage (%)', fontweight='bold')
        ax8.set_ylabel('Shadow Coverage (%)', fontweight='bold')
        ax8.set_title('Canopy vs Shadow Correlation', fontweight='bold', fontsize=14)
        ax8.grid(alpha=0.3)
        
        # 9. River area distribution
        ax9 = plt.subplot(3, 4, 9)
        df['mask_river_area_pixels'].hist(bins=30, ax=ax9, color='blue', edgecolor='black')
        ax9.set_xlabel('River Area (pixels)', fontweight='bold')
        ax9.set_ylabel('Count', fontweight='bold')
        ax9.set_title('River Area Distribution', fontweight='bold', fontsize=14)
        ax9.grid(alpha=0.3)
        
        # 10. Quality heatmap
        ax10 = plt.subplot(3, 4, 10)
        quality_matrix = df[['sharpness', 'brightness_mean_brightness', 
                            'mask_continuity_ratio', 'alignment_alignment_score']].corr()
        sns.heatmap(quality_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax10, 
                   cbar_kws={'label': 'Correlation'})
        ax10.set_title('Quality Metrics Correlation', fontweight='bold', fontsize=14)
        
        # 11. Monthly distribution
        ax11 = plt.subplot(3, 4, 11)
        month_counts = df['month'].value_counts()
        month_counts.plot(kind='pie', ax=ax11, autopct='%1.1f%%', startangle=90,
                         colors=['#87CEEB', '#FFB6C1', '#D3D3D3'])
        ax11.set_ylabel('')
        ax11.set_title('Temporal Distribution', fontweight='bold', fontsize=14)
        
        # 12. Overall quality score
        ax12 = plt.subplot(3, 4, 12)
        # Composite quality score (normalized)
        df['quality_score'] = (
            (df['sharpness'] / df['sharpness'].max()) * 0.3 +
            (df['mask_continuity_ratio']) * 0.3 +
            (df['alignment_alignment_score']) * 0.2 +
            (1 - df['shadow_percentage'] / 100) * 0.2
        )
        df['quality_score'].hist(bins=30, ax=ax12, color='orange', edgecolor='black')
        ax12.set_xlabel('Composite Quality Score', fontweight='bold')
        ax12.set_ylabel('Count', fontweight='bold')
        ax12.set_title('Overall Image Quality Distribution', fontweight='bold', fontsize=14)
        ax12.axvline(df['quality_score'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["quality_score"].mean():.2f}')
        ax12.legend()
        ax12.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, 'dataset_analysis_overview.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved visualization to {self.config.OUTPUT_DIR}/dataset_analysis_overview.png")
    
    def create_stratification_report(self, df):
        """Create detailed stratification for train/val/test split"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # By canopy category
        ax1 = axes[0, 0]
        canopy_dist = df.groupby('canopy_category').size()
        canopy_dist.plot(kind='bar', ax=ax1, color='forestgreen')
        ax1.set_title('Distribution by Canopy Category', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Count', fontweight='bold')
        ax1.set_xlabel('Category', fontweight='bold')
        for i, v in enumerate(canopy_dist):
            ax1.text(i, v + 2, str(v), ha='center', fontweight='bold')
        
        # By month
        ax2 = axes[0, 1]
        month_dist = df.groupby('month').size()
        month_dist.plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_title('Distribution by Month', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_xlabel('Month', fontweight='bold')
        for i, v in enumerate(month_dist):
            ax2.text(i, v + 2, str(v), ha='center', fontweight='bold')
        
        # By sharpness quality
        ax3 = axes[0, 2]
        sharp_dist = df.groupby('sharpness_quality').size()
        sharp_dist.plot(kind='bar', ax=ax3, color='purple')
        ax3.set_title('Distribution by Sharpness', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Count', fontweight='bold')
        ax3.set_xlabel('Quality', fontweight='bold')
        for i, v in enumerate(sharp_dist):
            ax3.text(i, v + 2, str(v), ha='center', fontweight='bold')
        
        # By exposure quality
        ax4 = axes[1, 0]
        exp_dist = df.groupby('brightness_exposure_quality').size()
        exp_dist.plot(kind='bar', ax=ax4, color='gold')
        ax4.set_title('Distribution by Exposure', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Count', fontweight='bold')
        ax4.set_xlabel('Exposure', fontweight='bold')
        for i, v in enumerate(exp_dist):
            ax4.text(i, v + 2, str(v), ha='center', fontweight='bold')
        
        # Canopy by month (stacked)
        ax5 = axes[1, 1]
        pd.crosstab(df['month'], df['canopy_category']).plot(kind='bar', stacked=True, ax=ax5)
        ax5.set_title('Canopy Category by Month', fontweight='bold', fontsize=12)
        ax5.set_ylabel('Count', fontweight='bold')
        ax5.set_xlabel('Month', fontweight='bold')
        ax5.legend(title='Canopy', bbox_to_anchor=(1.05, 1))
        
        # Quality vs Canopy
        ax6 = axes[1, 2]
        quality_by_canopy = df.groupby('canopy_category')['quality_score'].mean().sort_values()
        quality_by_canopy.plot(kind='barh', ax=ax6, color='orange')
        ax6.set_title('Mean Quality Score by Canopy', fontweight='bold', fontsize=12)
        ax6.set_xlabel('Mean Quality Score', fontweight='bold')
        ax6.set_ylabel('Canopy Category', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, 'stratification_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved stratification to {self.config.OUTPUT_DIR}/stratification_analysis.png")
    
    def suggest_train_val_test_split(self, df):
        """Suggest stratified split for training"""
        print("\n" + "="*60)
        print("SUGGESTED TRAIN/VAL/TEST SPLIT")
        print("="*60)
        
        # Stratify by canopy category and month
        train_indices = []
        val_indices = []
        test_indices = []
        
        for category in df['canopy_category'].unique():
            for month in df['month'].unique():
                subset = df[(df['canopy_category'] == category) & (df['month'] == month)]
                n = len(subset)
                
                if n == 0:
                    continue
                
                # 60-20-20 split
                n_train = int(0.6 * n)
                n_val = int(0.2 * n)
                
                indices = subset.index.tolist()
                np.random.shuffle(indices)
                
                train_indices.extend(indices[:n_train])
                val_indices.extend(indices[n_train:n_train+n_val])
                test_indices.extend(indices[n_train+n_val:])
        
        split_info = {
            'train': {
                'indices': train_indices,
                'count': len(train_indices),
                'percentage': len(train_indices) / len(df) * 100
            },
            'val': {
                'indices': val_indices,
                'count': len(val_indices),
                'percentage': len(val_indices) / len(df) * 100
            },
            'test': {
                'indices': test_indices,
                'count': len(test_indices),
                'percentage': len(test_indices) / len(df) * 100
            }
        }
        
        print(f"Train: {split_info['train']['count']} images ({split_info['train']['percentage']:.1f}%)")
        print(f"Val:   {split_info['val']['count']} images ({split_info['val']['percentage']:.1f}%)")
        print(f"Test:  {split_info['test']['count']} images ({split_info['test']['percentage']:.1f}%)")
        
        # Check stratification
        print("\nStratification Check:")
        for split_name, split_data in [('Train', train_indices), ('Val', val_indices), ('Test', test_indices)]:
            split_df = df.loc[split_data]
            print(f"\n{split_name}:")
            print(f"  Canopy categories: {split_df['canopy_category'].value_counts().to_dict()}")
            print(f"  Months: {split_df['month'].value_counts().to_dict()}")
            print(f"  Mean canopy: {split_df['canopy_percentage'].mean():.1f}%")
        
        # Save split to JSON
        split_save = {
            'train': [df.loc[i, 'filename'] for i in train_indices],
            'val': [df.loc[i, 'filename'] for i in val_indices],
            'test': [df.loc[i, 'filename'] for i in test_indices],
            'metadata': {
                'total_images': len(df),
                'split_date': datetime.now().isoformat(),
                'split_ratios': '60-20-20',
                'stratification': 'canopy_category + month'
            }
        }
        
        with open(os.path.join(self.config.OUTPUT_DIR, 'train_val_test_split.json'), 'w') as f:
            json.dump(split_save, f, indent=4)
        
        print(f"\n✓ Saved split to {self.config.OUTPUT_DIR}/train_val_test_split.json")
        
        return split_info
    
    def generate_report(self, df, summary):
        """Generate comprehensive text report"""
        report_path = os.path.join(self.config.OUTPUT_DIR, 'dataset_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATASET QUALITY ASSESSMENT REPORT\n")
            f.write("River Bride, Crookstown - Drone Image Dataset\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {summary['date_generated']}\n")
            f.write(f"Total Images Analyzed: {summary['total_images']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("1. TEMPORAL DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"March 2025:  {summary['march_images']} images ({summary['march_images']/summary['total_images']*100:.1f}%)\n")
            f.write(f"June 2025:   {summary['june_images']} images ({summary['june_images']/summary['total_images']*100:.1f}%)\n")
            f.write(f"June 2025:   {summary['july_images']} images ({summary['july_images']/summary['total_images']*100:.1f}%)\n")
            f.write(f"Other:       {summary['total_images'] - summary['march_images'] - summary['june_images'] - summary['july_images']} images\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("2. CANOPY DENSITY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Canopy Coverage: {summary['canopy_mean']:.1f}% (±{summary['canopy_std']:.1f}%)\n")
            f.write(f"Range: {summary['canopy_min']:.1f}% - {summary['canopy_max']:.1f}%\n\n")
            f.write("Distribution by Category:\n")
            f.write(f"  Sparse (<30%):      {summary['sparse_count']} images ({summary['sparse_count']/summary['total_images']*100:.1f}%)\n")
            f.write(f"  Moderate (30-60%):  {summary['moderate_count']} images ({summary['moderate_count']/summary['total_images']*100:.1f}%)\n")
            f.write(f"  Dense (60-80%):     {summary['dense_count']} images ({summary['dense_count']/summary['total_images']*100:.1f}%)\n")
            f.write(f"  Very Dense (>80%):  {summary['very_dense_count']} images ({summary['very_dense_count']/summary['total_images']*100:.1f}%)\n\n")
            
            # Seasonal comparison
            march_canopy = df[df['month'] == 'March']['canopy_percentage'].mean()
            june_canopy = df[df['month'] == 'June']['canopy_percentage'].mean()
            july_canopy = df[df['month'] == 'July']['canopy_percentage'].mean()
            f.write(f"Seasonal Comparison:\n")
            f.write(f"  March Mean: {march_canopy:.1f}%\n")
            f.write(f"  June Mean:  {june_canopy:.1f}%\n")
            f.write(f"  July Mean:  {july_canopy:.1f}%\n")
            f.write(f"  Difference: {june_canopy - march_canopy:.1f}% (June higher)\n\n")
            f.write(f"  Difference: {july_canopy - march_canopy:.1f}% (July higher)\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("3. IMAGE QUALITY METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Sharpness:\n")
            f.write(f"  Mean: {summary['mean_sharpness']:.1f}\n")
            f.write(f"  Sharp images: {summary['sharp_images']} ({summary['sharp_images']/summary['total_images']*100:.1f}%)\n")
            f.write(f"  Blurry images: {summary['blurry_images']} ({summary['blurry_images']/summary['total_images']*100:.1f}%)\n\n")
            
            f.write(f"Exposure:\n")
            f.write(f"  Mean brightness: {summary['mean_brightness']:.1f}\n")
            f.write(f"  Underexposed: {summary['underexposed']} images\n")
            f.write(f"  Overexposed:  {summary['overexposed']} images\n")
            f.write(f"  Well exposed: {summary['total_images'] - summary['underexposed'] - summary['overexposed']} images\n\n")
            
            f.write(f"Shadow Coverage:\n")
            f.write(f"  Mean: {summary['mean_shadow_coverage']:.1f}%\n")
            f.write(f"  Note: High shadow coverage expected under canopy\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("4. ANNOTATION QUALITY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Continuity Ratio: {summary['mean_continuity']:.3f}\n")
            f.write(f"  (1.0 = perfect single component, lower = fragmented)\n\n")
            f.write(f"Mean Alignment Score: {summary['mean_alignment']:.3f}\n")
            f.write(f"  (measures mask-image edge agreement)\n\n")
            f.write(f"Images with artifacts (>5 small regions): {summary['images_with_artifacts']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("5. RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            
            # Quality issues
            issues = []
            if summary['blurry_images'] > summary['total_images'] * 0.1:
                issues.append(f"⚠ {summary['blurry_images']} blurry images detected - consider excluding from training")
            if summary['images_with_artifacts'] > summary['total_images'] * 0.1:
                issues.append(f"⚠ {summary['images_with_artifacts']} images have annotation artifacts - review and clean")
            if summary['mean_continuity'] < 0.8:
                issues.append(f"⚠ Low mean continuity ({summary['mean_continuity']:.2f}) - river masks may be fragmented")
            
            if issues:
                f.write("Issues Found:\n")
                for issue in issues:
                    f.write(f"  {issue.encode('utf-8')}\n")
            else:
                f.write("No major quality issues detected\n")
            
            f.write("\nDataset Strengths:\n")
            f.write(f"Good canopy diversity ({summary['canopy_min']:.0f}% - {summary['canopy_max']:.0f}%)\n")
            f.write(f"Temporal variation (March vs June vs July)\n")
            if summary['mean_alignment'] > 0.5:
                f.write(f"Good annotation-image alignment\n")
            
            f.write("\nRecommended Next Steps:\n")
            f.write("  1. Review and clean images with low quality scores\n")
            f.write("  2. Verify annotations for images with low continuity\n")
            f.write("  3. Use stratified split (see train_val_test_split.json)\n")
            f.write("  4. Consider data augmentation for underrepresented categories\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ Saved report to {report_path}")
    
    def run_complete_analysis(self, max_images=None):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("STARTING COMPLETE DATASET ANALYSIS")
        print("="*80 + "\n")
        
        # 1. Analyze all images
        print("Step 1: Analyzing all images...")
        df = self.analyze_all_images(max_images=max_images)
        
        # Save raw data
        csv_path = os.path.join(self.config.OUTPUT_DIR, 'dataset_analysis_full.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved detailed results to {csv_path}")
        
        # 2. Generate summary
        print("\nStep 2: Generating summary statistics...")
        summary = self.generate_summary_statistics(df)
        
        # Save summary as JSON
        summary_path = os.path.join(self.config.OUTPUT_DIR, 'summary_statistics.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"✓ Saved summary to {summary_path}")
        
        # 3. Create visualizations
        print("\nStep 3: Creating visualizations...")
        self.create_visualizations(df)
        self.create_stratification_report(df)
        
        # 4. Suggest split
        print("\nStep 4: Suggesting train/val/test split...")
        split_info = self.suggest_train_val_test_split(df)
        
        # 5. Generate report
        print("\nStep 5: Generating comprehensive report...")
        self.generate_report(df, summary)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {self.config.OUTPUT_DIR}/")
        print("\nGenerated files:")
        print(f"  1. dataset_analysis_full.csv - Detailed per-image metrics")
        print(f"  2. summary_statistics.json - Overall dataset statistics")
        print(f"  3. dataset_analysis_overview.png - Main visualizations")
        print(f"  4. stratification_analysis.png - Distribution analysis")
        print(f"  5. train_val_test_split.json - Suggested data split")
        print(f"  6. dataset_analysis_report.txt - Comprehensive text report")
        
        return df, summary, split_info


# =====================================================
# 6. EXAMPLE USAGE & MAIN EXECUTION
# =====================================================

def main():
    """Main execution function"""
    
    # Update these paths to match your setup!
    config = Config()
    config.IMAGE_DIR = "data/raw/images"      
    config.MASK_DIR = "data/raw/masks"        
    config.OUTPUT_DIR = "analysis_results"
    
    config.MARCH_IDENTIFIER = "March"
    config.JUNE_IDENTIFIER = "June"
    config.JULY_IDENTIFIER = "July"
    
    # Create analyzer
    analyzer = DatasetAnalyzer(config)
    
    # Run complete analysis
    # For testing on small subset, use: max_images=50
    # For full dataset (415 images), use: max_images=None
    df, summary, split_info = analyzer.run_complete_analysis(max_images=None)
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"Total Images: {summary['total_images']}")
    print(f"Canopy Range: {summary['canopy_min']:.1f}% - {summary['canopy_max']:.1f}%")
    print(f"Mean Canopy: {summary['canopy_mean']:.1f}% (±{summary['canopy_std']:.1f}%)")
    print(f"\nCanopy Categories:")
    print(f"  Sparse:      {summary['sparse_count']} images")
    print(f"  Moderate:    {summary['moderate_count']} images")
    print(f"  Dense:       {summary['dense_count']} images")
    print(f"  Very Dense:  {summary['very_dense_count']} images")
    print(f"\nTemporal Split:")
    print(f"  March: {summary['march_images']} images")
    print(f"  June:  {summary['june_images']} images")
    print(f"  July:  {summary['july_images']} images")
    print("="*80)
    
    return df, summary, split_info


if __name__ == "__main__":
    # Run the analysis
    df, summary, split_info = main()
    
    print("\n✓ Analysis complete! Check the 'analysis_results' folder for outputs.")
    print("\nTo use the results in your paper:")
    print("  1. Quote canopy density range in Methods section")
    print("  2. Show distribution figure in Results")
    print("  3. Use suggested split for training")
    print("  4. Reference quality metrics in limitations")
