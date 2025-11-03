#!/usr/bin/env python3
"""
Advanced River Shape Completion Methods - Comprehensive Comparison
This script applies 10+ advanced methods to complete missing patches in river masks
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Import libraries (with graceful degradation)
try:
    from scipy.ndimage import distance_transform_edt, binary_fill_holes
    from scipy.interpolate import Rbf
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, some methods will be skipped")

try:
    from skimage.segmentation import random_walker, active_contour, morphological_geodesic_active_contour
    from skimage.segmentation import inverse_gaussian_gradient
    from skimage.filters import gaussian
    from skimage.morphology import skeletonize, medial_axis
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available, some methods will be skipped")

try:
    from sklearn.decomposition import DictionaryLearning
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available, dictionary learning will be skipped")


class RiverCompletion:
    """Collection of advanced river completion methods"""
    
    def __init__(self, img_path, output_path):
        self.img_path = img_path
        self.output_path = output_path
        self.img = cv2.imread(img_path, 0)
        if self.img is None:
            raise ValueError(f"Could not load image from {img_path}")
        self.binary = (self.img > 127).astype(np.uint8)
        self.results = {}
        
    def method_1_morphological_closing(self):
        """Basic morphological closing"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(self.binary, cv2.MORPH_CLOSE, kernel)
        return closed * 255
    
    def method_2_convex_hull(self):
        """Convex hull filling"""
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(self.binary)
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                hull = cv2.convexHull(contour)
                cv2.drawContours(result, [hull], -1, 1, thickness=cv2.FILLED)
        return result * 255
    
    def method_3_binary_fill_holes(self):
        """Fill holes using scipy"""
        if not SCIPY_AVAILABLE:
            return None
        filled = binary_fill_holes(self.binary).astype(np.uint8)
        return filled * 255
    
    def method_4_opencv_inpainting_telea(self):
        """Deep learning inpainting - Telea algorithm"""
        # Create mask of small gaps
        mask = cv2.threshold(self.binary, 0, 255, cv2.THRESH_BINARY_INV)[1]
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        gap_mask = np.zeros_like(self.img)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:  # Small gaps only
                cv2.drawContours(gap_mask, [cnt], -1, 255, -1)
        
        if gap_mask.max() == 0:
            return self.img
        
        result = cv2.inpaint(self.img, gap_mask, 3, cv2.INPAINT_TELEA)
        return result
    
    def method_5_opencv_inpainting_ns(self):
        """Deep learning inpainting - Navier-Stokes algorithm"""
        # Create mask of small gaps
        mask = cv2.threshold(self.binary, 0, 255, cv2.THRESH_BINARY_INV)[1]
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        gap_mask = np.zeros_like(self.img)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:  # Small gaps only
                cv2.drawContours(gap_mask, [cnt], -1, 255, -1)
        
        if gap_mask.max() == 0:
            return self.img
            
        result = cv2.inpaint(self.img, gap_mask, 3, cv2.INPAINT_NS)
        return result
    
    def method_6_random_walker(self):
        """Graph-based segmentation using random walker"""
        if not SKIMAGE_AVAILABLE:
            return None
        
        markers = np.zeros_like(self.img, dtype=np.int32)
        
        # Create certain regions
        kernel = np.ones((5, 5), np.uint8)
        certain_river = cv2.erode(self.binary, kernel, iterations=2)
        markers[certain_river == 1] = 1
        
        certain_bg = cv2.dilate(1 - self.binary, kernel, iterations=3)
        markers[certain_bg == 1] = 2
        
        try:
            labels = random_walker(self.img, markers, beta=130, mode='cg_j')
            result = (labels == 1).astype(np.uint8) * 255
            return result
        except:
            return None
    
    def method_7_tps_interpolation(self):
        """Thin plate spline boundary smoothing"""
        if not SCIPY_AVAILABLE:
            return None
        
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return self.binary * 255
        
        largest_contour = max(contours, key=cv2.contourArea)
        points = largest_contour.squeeze()
        
        if len(points.shape) == 1 or len(points) < 10:
            return self.binary * 255
        
        # Sample points
        sample_rate = max(1, len(points) // 300)
        sampled_points = points[::sample_rate]
        
        x, y = sampled_points[:, 0], sampled_points[:, 1]
        
        try:
            # Fit RBF
            rbf_x = Rbf(np.arange(len(x)), x, function='thin_plate', smooth=3)
            rbf_y = Rbf(np.arange(len(y)), y, function='thin_plate', smooth=3)
            
            # Generate smooth boundary
            t_new = np.linspace(0, len(x)-1, len(x)*2)
            x_smooth = rbf_x(t_new)
            y_smooth = rbf_y(t_new)
            
            # Create filled result
            result = np.zeros_like(self.img)
            smooth_contour = np.column_stack([x_smooth, y_smooth]).astype(np.int32)
            cv2.fillPoly(result, [smooth_contour], 255)
            
            return result
        except:
            return self.binary * 255
    
    def method_8_active_contours(self):
        """Active contours (snakes)"""
        if not SKIMAGE_AVAILABLE:
            return None
        
        img_smooth = gaussian(self.img, 2)
        
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self.binary * 255
        
        largest_contour = max(contours, key=cv2.contourArea)
        init = largest_contour.squeeze()
        
        if len(init.shape) == 1 or len(init) < 10:
            return self.binary * 255
        
        try:
            snake = active_contour(
                img_smooth,
                init[:, ::-1],
                alpha=0.015,
                beta=10,
                gamma=0.001,
                max_iterations=1000
            )
            
            result = np.zeros_like(self.img)
            snake_int = np.round(snake[:, ::-1]).astype(np.int32)
            cv2.fillPoly(result, [snake_int], 255)
            
            return result
        except:
            return self.binary * 255
    
    def method_9_grabcut(self):
        """GrabCut algorithm"""
        img_color = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        
        mask = np.zeros(self.img.shape, np.uint8)
        mask[self.binary == 1] = cv2.GC_PR_FGD
        mask[self.binary == 0] = cv2.GC_PR_BGD
        
        kernel = np.ones((10, 10), np.uint8)
        certain_fg = cv2.erode(self.binary, kernel, iterations=2)
        mask[certain_fg == 1] = cv2.GC_FGD
        
        certain_bg = cv2.dilate(1 - self.binary, kernel, iterations=5)
        mask[certain_bg == 1] = cv2.GC_BGD
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(img_color, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            result = mask2 * 255
            return result
        except:
            return self.binary * 255
    
    def method_10_geodesic_active_contour(self):
        """Geodesic active contours"""
        if not SKIMAGE_AVAILABLE:
            return None
        
        init_ls = self.binary.astype(np.int8)
        gimage = inverse_gaussian_gradient(self.img)
        
        try:
            result = morphological_geodesic_active_contour(
                gimage,
                iterations=50,
                init_level_set=init_ls,
                smoothing=1,
                balloon=-1,
                threshold=0.69
            )
            return (result * 255).astype(np.uint8)
        except:
            return None
    
    def method_11_poisson_blending(self):
        """Poisson image editing"""
        kernel = np.ones((20, 20), np.uint8)
        dilated = cv2.dilate(self.binary, kernel, iterations=1)
        blend_mask = cv2.subtract(dilated, self.binary)
        
        if blend_mask.max() == 0:
            return self.img
        
        img_3ch = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        
        moments = cv2.moments(blend_mask)
        if moments['m00'] == 0:
            return self.img
        
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        center = (cx, cy)
        
        try:
            result = cv2.seamlessClone(
                img_3ch,
                img_3ch,
                blend_mask * 255,
                center,
                cv2.NORMAL_CLONE
            )
            return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        except:
            return self.img
    
    def method_12_dictionary_learning(self):
        """Sparse coding / dictionary learning"""
        if not SKLEARN_AVAILABLE:
            return None
        
        patch_size = 8
        patches = []
        
        for i in range(0, self.img.shape[0] - patch_size, patch_size // 2):
            for j in range(0, self.img.shape[1] - patch_size, patch_size // 2):
                patch = self.binary[i:i+patch_size, j:j+patch_size]
                if patch.mean() > 0.8:
                    patches.append(patch.flatten())
        
        if len(patches) < 20:
            return self.binary * 255
        
        patches = np.array(patches)
        
        try:
            dict_learner = DictionaryLearning(
                n_components=30,
                alpha=1,
                max_iter=10,
                random_state=42
            )
            dict_learner.fit(patches)
            
            result = self.binary.copy().astype(np.float32)
            
            for i in range(0, self.img.shape[0] - patch_size, patch_size):
                for j in range(0, self.img.shape[1] - patch_size, patch_size):
                    patch = self.binary[i:i+patch_size, j:j+patch_size]
                    if patch.mean() > 0.3 and patch.mean() < 0.7:
                        patch_vec = patch.flatten().reshape(1, -1)
                        reconstructed = dict_learner.transform(patch_vec)
                        reconstructed_patch = np.dot(reconstructed, dict_learner.components_)
                        result[i:i+patch_size, j:j+patch_size] = reconstructed_patch.reshape(patch_size, patch_size)
            
            return ((result > 0.5) * 255).astype(np.uint8)
        except:
            return self.binary * 255
    
    def method_13_combined_best(self):
        """Combined approach using multiple methods"""
        # Step 1: Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(self.binary, cv2.MORPH_CLOSE, kernel)
        
        # Step 2: Fill holes
        if SCIPY_AVAILABLE:
            filled = binary_fill_holes(closed).astype(np.uint8)
        else:
            filled = closed
        
        # Step 3: Smooth with opening
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_smooth)
        
        return smoothed * 255
    
    def run_all_methods(self):
        """Run all methods and store results"""
        print("Running advanced river completion methods...")
        print("=" * 60)
        
        methods = [
            ("Original", lambda: self.img),
            ("1. Morphological Closing", self.method_1_morphological_closing),
            ("2. Convex Hull", self.method_2_convex_hull),
            ("3. Binary Fill Holes", self.method_3_binary_fill_holes),
            ("4. Inpainting (Telea)", self.method_4_opencv_inpainting_telea),
            ("5. Inpainting (Navier-Stokes)", self.method_5_opencv_inpainting_ns),
            ("6. Random Walker", self.method_6_random_walker),
            ("7. TPS Interpolation", self.method_7_tps_interpolation),
            ("8. Active Contours", self.method_8_active_contours),
            ("9. GrabCut", self.method_9_grabcut),
            ("10. Geodesic Active Contour", self.method_10_geodesic_active_contour),
            ("11. Poisson Blending", self.method_11_poisson_blending),
            ("12. Dictionary Learning", self.method_12_dictionary_learning),
            ("13. Combined Best", self.method_13_combined_best),
        ]
        
        for name, method in methods:
            try:
                print(f"Processing: {name}...", end=" ")
                result = method()
                if result is not None:
                    self.results[name] = result
                    print("✓ Success")
                else:
                    print("✗ Skipped (dependencies missing)")
            except Exception as e:
                print(f"✗ Failed: {str(e)}")
        
        print("=" * 60)
        print(f"Completed {len(self.results)} methods successfully\n")
        
        return self.results
    
    def calculate_metrics(self, result):
        """Calculate quality metrics for a result"""
        if result is None:
            return {}
        
        result_binary = (result > 127).astype(np.uint8)
        
        # Calculate metrics
        total_pixels = result.shape[0] * result.shape[1]
        river_pixels = np.sum(result_binary)
        river_percentage = (river_pixels / total_pixels) * 100
        
        # Find contours for shape analysis
        contours, _ = cv2.findContours(result_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Compactness (lower is more compact/smooth)
            if area > 0:
                compactness = (perimeter ** 2) / (4 * np.pi * area)
            else:
                compactness = 0
            
            # Number of gaps (holes)
            hierarchy = cv2.findContours(result_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]
            num_holes = 0
            if hierarchy is not None:
                num_holes = np.sum(hierarchy[0][:, 3] >= 0) - 1
        else:
            area = 0
            perimeter = 0
            compactness = 0
            num_holes = 0
        
        return {
            'river_percentage': river_percentage,
            'area': area,
            'perimeter': perimeter,
            'compactness': compactness,
            'num_holes': max(0, num_holes)
        }
    
    def visualize_results(self):
        """Create comprehensive visualization of all results"""
        if not self.results:
            print("No results to visualize. Run run_all_methods() first.")
            return
        
        n_results = len(self.results)
        n_cols = 4
        n_rows = (n_results + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(20, 5 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)
        
        for idx, (name, result) in enumerate(self.results.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            if result is not None:
                ax.imshow(result, cmap='gray')
                
                # Calculate and display metrics
                metrics = self.calculate_metrics(result)
                
                if name != "Original":
                    info_text = f"Coverage: {metrics['river_percentage']:.1f}%\n"
                    info_text += f"Holes: {metrics['num_holes']}\n"
                    info_text += f"Compactness: {metrics['compactness']:.2f}"
                    
                    ax.text(0.02, 0.98, info_text,
                           transform=ax.transAxes,
                           fontsize=8,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(name, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle('Advanced River Completion Methods - Comprehensive Comparison',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(f'{self.output_path}/river_completion_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
        plt.close()
        
        return output_path
    
    def create_detailed_comparison(self):
        """Create detailed comparison with metrics table"""
        if not self.results:
            return
        
        # Calculate metrics for all results
        all_metrics = {}
        for name, result in self.results.items():
            if name != "Original":
                all_metrics[name] = self.calculate_metrics(result)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 2, 1], hspace=0.3)
        
        # Top: Best 6 methods
        gs_top = gs[0].subgridspec(2, 3, hspace=0.2, wspace=0.15)
        
        # Sort by compactness (lower is better)
        sorted_methods = sorted(all_metrics.items(), 
                              key=lambda x: x[1].get('compactness', float('inf')))[:6]
        
        for idx, (name, metrics) in enumerate(sorted_methods):
            ax = fig.add_subplot(gs_top[idx // 3, idx % 3])
            result = self.results[name]
            ax.imshow(result, cmap='gray')
            
            info_text = f"Coverage: {metrics['river_percentage']:.1f}%\n"
            info_text += f"Holes: {metrics['num_holes']}\n"
            info_text += f"Compactness: {metrics['compactness']:.2f}"
            
            ax.text(0.02, 0.98, info_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax.set_title(f"{name} (Rank #{idx+1})", fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # Middle: Metrics comparison chart
        ax_middle = fig.add_subplot(gs[1])
        
        methods_list = list(all_metrics.keys())
        coverage = [all_metrics[m]['river_percentage'] for m in methods_list]
        compactness = [all_metrics[m]['compactness'] for m in methods_list]
        holes = [all_metrics[m]['num_holes'] for m in methods_list]
        
        x = np.arange(len(methods_list))
        width = 0.25
        
        ax_middle.bar(x - width, coverage, width, label='Coverage %', alpha=0.8)
        ax_middle.bar(x, [c*10 for c in compactness], width, label='Compactness ×10', alpha=0.8)
        ax_middle.bar(x + width, holes, width, label='Number of Holes', alpha=0.8)
        
        ax_middle.set_xlabel('Methods', fontsize=11, fontweight='bold')
        ax_middle.set_ylabel('Metric Values', fontsize=11, fontweight='bold')
        ax_middle.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
        ax_middle.set_xticks(x)
        ax_middle.set_xticklabels([m.split('.')[1].strip() if '.' in m else m 
                                   for m in methods_list], rotation=45, ha='right')
        ax_middle.legend()
        ax_middle.grid(axis='y', alpha=0.3)
        
        # Bottom: Recommendations
        ax_bottom = fig.add_subplot(gs[2])
        ax_bottom.axis('off')
        
        recommendations = """
        📊 METHOD RECOMMENDATIONS:
        
        • Best Overall: Combined Best (Method 13) - Uses multiple techniques for robust completion
        • Smoothest Boundaries: TPS Interpolation (Method 7) - Creates smooth, natural-looking curves
        • Gap Filling: Morphological Closing (Method 1) - Simple and effective for small gaps
        • Large Missing Regions: Random Walker (Method 6) or GrabCut (Method 9) - Intelligent region growing
        • Preserve Original Shape: Binary Fill Holes (Method 3) - Only fills internal gaps
        • Natural Looking: Active Contours (Method 8) or Geodesic Active Contour (Method 10)
        
        💡 TIPS:
        - Lower compactness = smoother, more natural boundaries
        - Zero holes = complete filling
        - Choose method based on your specific needs (speed vs. quality vs. preservation)
        """
        
        ax_bottom.text(0.05, 0.95, recommendations,
                      transform=ax_bottom.transAxes,
                      fontsize=10,
                      verticalalignment='top',
                      fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle('Detailed River Completion Analysis & Recommendations',
                    fontsize=16, fontweight='bold')
        
        plt.savefig(f'{self.output_path}/river_completion_detailed_comparison.png', dpi=150, bbox_inches='tight')

        print(f"✓ Detailed comparison saved to: {output_path}")
        plt.close()
        
        return output_path
    
    def save_individual_results(self, output_dir='/mnt/user-data/outputs'):
        """Save each result as individual file"""
        import os
        
        individual_dir = os.path.join(output_dir, 'individual_results')
        os.makedirs(individual_dir, exist_ok=True)
        
        for name, result in self.results.items():
            if name != "Original" and result is not None:
                filename = name.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
                filepath = os.path.join(individual_dir, f'{filename}.png')
                cv2.imwrite(filepath, result)
        
        print(f"\n✓ Individual results saved to: {individual_dir}/")
        return individual_dir


def main(img_path, output_path):
    """Main execution function"""
    print("\n" + "="*60)
    print("ADVANCED RIVER SHAPE COMPLETION - COMPREHENSIVE ANALYSIS")
    print("="*60 + "\n")
    
    print(f"Loading image: {img_path}")
    print(f"Output image path: {output_path}")
    river = RiverCompletion(img_path, output_path)
    
    # Run all methods
    results = river.run_all_methods()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    river.visualize_results()
    river.create_detailed_comparison()
    
    # Save individual results
    river.save_individual_results()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total methods tested: {len(results)}")
    print(f"Successful completions: {len([r for r in results.values() if r is not None])}")
    
    # Find best method
    all_metrics = {}
    for name, result in results.items():
        if name != "Original" and result is not None:
            metrics = river.calculate_metrics(result)
            all_metrics[name] = metrics
    
    if all_metrics:
        # Best by different criteria
        best_coverage = max(all_metrics.items(), key=lambda x: x[1]['river_percentage'])
        best_smooth = min(all_metrics.items(), key=lambda x: x[1]['compactness'])
        best_no_holes = min(all_metrics.items(), key=lambda x: x[1]['num_holes'])
        
        print(f"\n🏆 Best Coverage: {best_coverage[0]} ({best_coverage[1]['river_percentage']:.1f}%)")
        print(f"🏆 Smoothest: {best_smooth[0]} (Compactness: {best_smooth[1]['compactness']:.2f})")
        print(f"🏆 Most Complete: {best_no_holes[0]} ({best_no_holes[1]['num_holes']} holes)")
    
    print("\n" + "="*60)
    print("✓ Analysis complete! Check the output files for detailed results.")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    sys.path.append('src/data')
    sys.path.append('experiments')
    output_path = 'experiments\\results'
    # mask_path = 'data\processed\\train\masks\DJI_20250324092955_0011_V_March.png'
    mask_path = 'data\processed\\train\masks\DJI_20250324094552_0023_V_March.png'
    image_path = 'data\processed\\train\images\DJI_20250324092955_0011_V_March.png'
    main(mask_path, output_path)


