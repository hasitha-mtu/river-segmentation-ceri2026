#!/usr/bin/env python3
"""
Advanced Smooth Polygon Methods
================================

Instead of convex hull (connects outermost points), we implement methods that:
1. Connect nearest points for smoother, more natural boundaries
2. Create concave hulls (alpha shapes)
3. Use various smoothing techniques
4. Balance between detail preservation and smoothness
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import Delaunay, distance_matrix
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev, UnivariateSpline
from glob import glob
import os
import warnings
warnings.filterwarnings('ignore')


class SmoothPolygonBuilder:
    """Build smooth polygons by connecting nearest points"""
    
    def __init__(self, input_file, output_path, index):
        self.img = cv2.imread(input_file, 0)
        self.binary = (self.img > 127).astype(np.uint8)
        self.output_path = output_path
        self.index = index
        self.results = {}
        
    def get_boundary_points(self, subsample_factor=1):
        """Extract boundary points from the mask"""
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        points = largest.squeeze()
        
        if len(points.shape) == 1:
            return None
        
        # Subsample if needed
        if subsample_factor > 1:
            points = points[::subsample_factor]
        
        return points
    
    def method_1_alpha_shape(self, alpha=50):
        """
        Alpha Shape (Concave Hull)
        Creates a polygon that can have concave regions
        Alpha controls how "tight" the hull is
        """
        points = self.get_boundary_points(subsample_factor=3)
        if points is None:
            return None
        
        # Compute Delaunay triangulation
        tri = Delaunay(points)
        
        # Find edges
        edges = set()
        edge_lengths = []
        
        for simplex in tri.simplices:
            # For each triangle, add its edges
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                edges.add(edge)
                
                # Calculate edge length
                p1, p2 = points[edge[0]], points[edge[1]]
                length = np.linalg.norm(p1 - p2)
                edge_lengths.append((edge, length))
        
        # Keep only edges shorter than alpha
        alpha_edges = [e for e, l in edge_lengths if l < alpha]
        
        # Build boundary from edges
        result = np.zeros_like(self.img)
        
        # Draw edges
        for edge in alpha_edges:
            p1 = tuple(points[edge[0]])
            p2 = tuple(points[edge[1]])
            cv2.line(result, p1, p2, 255, 1)
        
        # Fill the polygon
        result = self._fill_polygon_from_edges(result)
        
        return result
    
    def method_2_rolling_ball(self, radius=20):
        """
        Rolling Ball Algorithm
        Simulates rolling a ball around the boundary
        Creates smooth concave hull
        """
        points = self.get_boundary_points(subsample_factor=2)
        if points is None:
            return None
        
        # Use morphological opening (erosion + dilation)
        # This is like rolling a ball
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        
        # Opening removes protrusions
        opened = cv2.morphologyEx(self.binary * 255, cv2.MORPH_OPEN, kernel)
        
        # Closing fills gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def method_3_k_nearest_smoothing(self, k=10, iterations=3):
        """
        K-Nearest Neighbor Smoothing
        Each point moves toward average of k nearest neighbors
        """
        points = self.get_boundary_points(subsample_factor=2)
        if points is None:
            return None
        
        points_float = points.astype(np.float32)
        
        for iteration in range(iterations):
            # Calculate distance matrix
            dist_mat = cdist(points_float, points_float)
            
            # For each point, find k nearest neighbors
            smoothed = np.zeros_like(points_float)
            
            for i in range(len(points_float)):
                # Get k nearest (excluding self)
                nearest_idx = np.argsort(dist_mat[i])[1:k+1]
                nearest_points = points_float[nearest_idx]
                
                # Average position
                smoothed[i] = nearest_points.mean(axis=0)
            
            points_float = smoothed
        
        # Create result
        result = np.zeros_like(self.img)
        smoothed_int = np.round(smoothed).astype(np.int32)
        cv2.fillPoly(result, [smoothed_int], 255)
        
        return result
    
    def method_4_spline_interpolation(self, smoothing_factor=5):
        """
        B-Spline Interpolation
        Fits smooth spline through boundary points
        """
        points = self.get_boundary_points(subsample_factor=5)
        if points is None:
            return None
        
        # Close the curve
        points = np.vstack([points, points[0]])
        
        try:
            # Fit spline
            tck, u = splprep([points[:, 0], points[:, 1]], s=smoothing_factor*len(points), per=True)
            
            # Evaluate spline at many points for smooth curve
            u_new = np.linspace(0, 1, len(points) * 3)
            x_new, y_new = splev(u_new, tck)
            
            # Create result
            result = np.zeros_like(self.img)
            smooth_points = np.column_stack([x_new, y_new]).astype(np.int32)
            cv2.fillPoly(result, [smooth_points], 255)
            
            return result
        except:
            return None
    
    def method_5_gaussian_blur_threshold(self, sigma=5):
        """
        Gaussian Blur + Threshold
        Blur the mask then threshold back to binary
        Creates smooth boundaries
        """
        # Blur
        blurred = cv2.GaussianBlur(self.binary.astype(np.float32), (0, 0), sigma)
        
        # Threshold
        result = (blurred > 0.5).astype(np.uint8) * 255
        
        return result
    
    def method_6_chaikin_smoothing(self, iterations=3):
        """
        Chaikin's Corner Cutting Algorithm
        Iteratively cuts corners to create smooth curves
        """
        points = self.get_boundary_points(subsample_factor=3)
        if points is None:
            return None
        
        points_float = points.astype(np.float32)
        
        for _ in range(iterations):
            new_points = []
            n = len(points_float)
            
            for i in range(n):
                p0 = points_float[i]
                p1 = points_float[(i + 1) % n]
                
                # Cut corner: create two new points
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                
                new_points.append(q)
                new_points.append(r)
            
            points_float = np.array(new_points)
        
        # Create result
        result = np.zeros_like(self.img)
        smooth_int = np.round(points_float).astype(np.int32)
        cv2.fillPoly(result, [smooth_int], 255)
        
        return result
    
    def method_7_savitzky_golay(self, window_length=51, polyorder=3):
        """
        Savitzky-Golay Filter
        Fits polynomial to sliding window
        Smooths while preserving features
        """
        from scipy.signal import savgol_filter
        
        points = self.get_boundary_points(subsample_factor=2)
        if points is None:
            return None
        
        # Ensure window_length is odd and less than number of points
        window_length = min(window_length, len(points) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(polyorder + 2, window_length)
        
        try:
            # Smooth x and y coordinates separately
            x_smooth = savgol_filter(points[:, 0], window_length, polyorder, mode='wrap')
            y_smooth = savgol_filter(points[:, 1], window_length, polyorder, mode='wrap')
            
            smooth_points = np.column_stack([x_smooth, y_smooth])
            
            # Create result
            result = np.zeros_like(self.img)
            smooth_int = np.round(smooth_points).astype(np.int32)
            cv2.fillPoly(result, [smooth_int], 255)
            
            return result
        except:
            return None
    
    def method_8_nearest_neighbor_hull(self, max_distance=30):
        """
        Nearest Neighbor Hull
        Connect points to nearest neighbor within distance threshold
        Creates concave hull following shape closely
        """
        points = self.get_boundary_points(subsample_factor=3)
        if points is None:
            return None
        
        # Build graph of nearest neighbors
        n = len(points)
        dist_mat = cdist(points, points)
        
        # For each point, find nearest neighbor within max_distance
        connections = []
        visited = set()
        
        # Start from first point
        current = 0
        path = [current]
        visited.add(current)
        
        while len(visited) < n:
            # Find nearest unvisited neighbor
            distances = dist_mat[current]
            distances[list(visited)] = np.inf
            
            nearest = np.argmin(distances)
            
            if distances[nearest] > max_distance:
                # No close neighbor, find closest unvisited
                distances = dist_mat[current]
                distances[list(visited)] = np.inf
                if np.all(np.isinf(distances)):
                    break
                nearest = np.argmin(distances)
            
            path.append(nearest)
            visited.add(nearest)
            current = nearest
        
        # Close the path
        path.append(path[0])
        
        # Create result
        result = np.zeros_like(self.img)
        hull_points = points[path]
        cv2.fillPoly(result, [hull_points], 255)
        
        return result
    
    def method_9_distance_transform_threshold(self, percentile=95):
        """
        Distance Transform with Threshold
        Use distance transform to find "core" of river
        Then expand smoothly
        """
        from scipy.ndimage import distance_transform_edt
        
        # Distance transform
        dist = distance_transform_edt(self.binary)
        
        # Threshold at percentile
        threshold = np.percentile(dist[dist > 0], percentile - 20)
        core = (dist >= threshold).astype(np.uint8)
        
        # Smooth expansion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        result = cv2.morphologyEx(core * 255, cv2.MORPH_CLOSE, kernel)
        
        return result
    
    def method_10_weighted_voronoi(self):
        """
        Weighted Voronoi Diagram
        Creates smooth regions based on distance to boundary points
        """
        points = self.get_boundary_points(subsample_factor=5)
        if points is None:
            return None
        
        # Create grid
        h, w = self.img.shape
        y, x = np.mgrid[0:h, 0:w]
        grid_points = np.column_stack([x.ravel(), y.ravel()])
        
        # Find nearest boundary point for each grid point
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        distances, indices = tree.query(grid_points)
        
        # Threshold by distance
        max_dist = np.percentile(distances[self.binary.ravel() > 0], 90)
        result = (distances < max_dist).reshape(h, w).astype(np.uint8) * 255
        
        return result
    
    def _fill_polygon_from_edges(self, edge_image):
        """Helper to fill polygon from edge image"""
        # Find contours from edges
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return edge_image
        
        # Fill largest contour
        result = np.zeros_like(edge_image)
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(result, [largest], -1, 255, -1)
        
        return result
    
    def run_all_methods(self):
        """Run all smooth polygon methods"""
        print("\n" + "="*70)
        print("SMOOTH POLYGON BUILDING METHODS")
        print("="*70)
        
        methods = [
            ("Original", lambda: self.img),
            ("1. Alpha Shape (Concave Hull)", lambda: self.method_1_alpha_shape(alpha=50)),
            ("2. Rolling Ball", lambda: self.method_2_rolling_ball(radius=20)),
            ("3. K-Nearest Smoothing", lambda: self.method_3_k_nearest_smoothing(k=10, iterations=3)),
            ("4. Spline Interpolation", lambda: self.method_4_spline_interpolation(smoothing_factor=5)),
            ("5. Gaussian Blur + Threshold", lambda: self.method_5_gaussian_blur_threshold(sigma=5)),
            ("6. Chaikin Corner Cutting", lambda: self.method_6_chaikin_smoothing(iterations=3)),
            ("7. Savitzky-Golay Filter", lambda: self.method_7_savitzky_golay(window_length=51)),
            ("8. Nearest Neighbor Hull", lambda: self.method_8_nearest_neighbor_hull(max_distance=30)),
            ("9. Distance Transform", lambda: self.method_9_distance_transform_threshold(percentile=95)),
            ("10. Weighted Voronoi", lambda: self.method_10_weighted_voronoi()),
        ]
        
        for name, method in methods:
            try:
                print(f"Processing: {name}...", end=" ")
                result = method()
                if result is not None:
                    self.results[name] = result
                    print("✓ Success")
                else:
                    print("✗ Skipped")
            except Exception as e:
                print(f"✗ Failed: {str(e)}")
        
        print("="*70)
        print(f"Completed {len(self.results)} methods successfully\n")
        
        return self.results
    
    def visualize_results(self):
        """Create comprehensive visualization"""
        n_results = len(self.results)
        n_cols = 4
        n_rows = (n_results + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(20, 5 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)
        
        for idx, (name, result) in enumerate(self.results.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            ax.imshow(result, cmap='gray')
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.axis('off')
            
            # Add boundary visualization
            if name != "Original":
                # Find contours
                binary = (result > 127).astype(np.uint8)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    perimeter = cv2.arcLength(largest, True)
                    area = cv2.contourArea(largest)
                    
                    if area > 0:
                        compactness = (perimeter ** 2) / (4 * np.pi * area)
                    else:
                        compactness = 0
                    
                    info = f"Compactness: {compactness:.2f}\n"
                    info += f"Perimeter: {perimeter:.0f}"
                    
                    ax.text(0.02, 0.98, info, transform=ax.transAxes,
                           va='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                           fontfamily='monospace')
        
        plt.suptitle('Smooth Polygon Methods - Connecting Nearest Points',
                    fontsize=16, fontweight='bold')
        
        plt.savefig(f'{self.output_path}/smooth_polygon_comparison_{self.index}.png',
                   dpi=150, bbox_inches='tight')
        print("✓ Visualization saved")
        plt.close()


def create_comparison_with_convex_hull(input_file, output_path, index):
    """Compare smooth polygon methods with convex hull"""
    
    img = cv2.imread(input_file, 0)
    binary = (img > 127).astype(np.uint8)
    
    # Get contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    
    largest = max(contours, key=cv2.contourArea)
    
    # Convex hull (connects outermost points)
    convex_hull = cv2.convexHull(largest)
    
    # Alpha shape (concave hull - connects nearest points)
    builder = SmoothPolygonBuilder(input_file, output_path, index)
    alpha_shape = builder.method_1_alpha_shape(alpha=50)
    
    # Spline (smooth curve through points)
    spline = builder.method_4_spline_interpolation(smoothing_factor=5)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original\n(Jagged boundaries)', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Convex hull
    convex_result = np.zeros_like(img)
    cv2.drawContours(convex_result, [convex_hull], -1, 255, -1)
    axes[0, 1].imshow(convex_result, cmap='gray')
    axes[0, 1].set_title('Convex Hull\n(Outermost points)', fontweight='bold', color='red')
    axes[0, 1].axis('off')
    
    # Alpha shape
    axes[0, 2].imshow(alpha_shape if alpha_shape is not None else img, cmap='gray')
    axes[0, 2].set_title('Alpha Shape\n(Nearest points - concave)', fontweight='bold', color='green')
    axes[0, 2].axis('off')
    
    # Spline
    axes[0, 3].imshow(spline if spline is not None else img, cmap='gray')
    axes[0, 3].set_title('Spline Interpolation\n(Smooth curve)', fontweight='bold', color='blue')
    axes[0, 3].axis('off')
    
    # Overlays
    for i, (result, title, color) in enumerate([
        (convex_result, 'Convex Hull Overlay', (255, 0, 0)),
        (alpha_shape if alpha_shape is not None else img, 'Alpha Shape Overlay', (0, 255, 0)),
        (spline if spline is not None else img, 'Spline Overlay', (0, 0, 255))
    ]):
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        result_binary = (result > 127).astype(np.uint8)
        
        # Original in blue
        overlay[:, :, 2] = binary * 200
        
        # Method result in other color
        overlay[:, :, 0] += result_binary * color[0] // 2
        overlay[:, :, 1] += result_binary * color[1] // 2
        overlay[:, :, 2] += result_binary * color[2] // 2
        
        axes[1, i+1].imshow(overlay)
        axes[1, i+1].set_title(title, fontweight='bold')
        axes[1, i+1].axis('off')
    
    # Explanation
    axes[1, 0].axis('off')
    explanation = """
    COMPARISON:
    
    🔴 Convex Hull:
      • Connects outermost points
      • Always convex (no dents)
      • Oversimplifies shape
      • Loses detail
    
    🟢 Alpha Shape:
      • Connects nearest points
      • Can be concave
      • Follows shape closely
      • Preserves detail
    
    🔵 Spline:
      • Smooth curve through points
      • Very smooth boundaries
      • Natural appearance
      • May over-smooth
    
    YOUR IDEA = Alpha Shape!
    (Connecting nearest points)
    """
    
    axes[1, 0].text(0.1, 0.9, explanation, transform=axes[1, 0].transAxes,
                   va='top', fontsize=10, fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Your Idea: Connecting Nearest Points vs Outermost Points',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_path}/nearest_points_vs_convex_hull_{index}.png',
               dpi=150, bbox_inches='tight')
    print("✓ Comparison visualization created")
    plt.close()


def main(input_file, output_path, index):
    os.makedirs(output_path, exist_ok=True)
    print("\n" + "="*70)
    print("IMPLEMENTING YOUR IDEA: Connecting Nearest Points")
    print("="*70)
    
    # Create comparison
    print("\n1. Creating comparison with convex hull...")
    create_comparison_with_convex_hull(input_file, output_path, index)
    
    # Run all methods
    print("\n2. Running all smooth polygon methods...")
    builder = SmoothPolygonBuilder(input_file, output_path, index)
    results = builder.run_all_methods()
    
    # Visualize
    print("\n3. Creating comprehensive visualization...")
    builder.visualize_results()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("""
    For connecting nearest points (your idea):
    
    ✅ BEST: Alpha Shape (Method 1)
       • Exactly what you described!
       • Connects nearest points within distance threshold
       • Creates concave hull following shape
       • Preserves river details
    
    ✅ ALSO GOOD: K-Nearest Smoothing (Method 3)
       • Each point moves to average of k nearest
       • Very smooth results
       • Still follows shape closely
    
    ✅ FOR VERY SMOOTH: Spline Interpolation (Method 4)
       • Fits smooth curve through points
       • Most natural looking
       • Great for visualization
    
    ⚠️  AVOID FOR MEASUREMENTS: Rolling Ball, Gaussian Blur
       • Change boundaries significantly
       • Not suitable for width measurement
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    sys.path.append('experiments')
    
    model = 'unet'
    input_path = f'results/models/{model}/output'
    output_path = f'results/models/{model}/output/updated'
    files = glob(f'{input_path}/predictions_all_*.png')
    i = 1
    for file_path in files:
        main(file_path, output_path, i)
        i = i+1

    model = 'deeplabv3plus'
    input_path = f'results/models/{model}/output'
    output_path = f'results/models/{model}/output/updated'
    files = glob(f'{input_path}/predictions_all_*.png')
    i = 1
    for file_path in files:
        main(file_path, output_path, i)
        i = i+1


