#!/usr/bin/env python3
"""
Parameter Tuning Guide for Smooth Polygon Methods
==================================================

Shows how different parameters affect the smoothness and detail preservation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


def demonstrate_alpha_shape_parameters():
    """Show how alpha parameter affects the concave hull"""
    
    img_path = '/mnt/user-data/uploads/DJI_20250324094601_0036_V_March.png'
    img = cv2.imread(img_path, 0)
    binary = (img > 127).astype(np.uint8)
    
    # Get boundary points
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return
    
    largest = max(contours, key=cv2.contourArea)
    points = largest.squeeze()[::3]  # Subsample
    
    if len(points.shape) == 1:
        return
    
    # Test different alpha values
    alpha_values = [20, 40, 60, 100, 200, 500]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, alpha in enumerate(alpha_values):
        # Compute Delaunay triangulation
        tri = Delaunay(points)
        
        # Find edges shorter than alpha
        edges = []
        for simplex in tri.simplices:
            for i in range(3):
                p1_idx = simplex[i]
                p2_idx = simplex[(i+1)%3]
                
                p1 = points[p1_idx]
                p2 = points[p2_idx]
                length = np.linalg.norm(p1 - p2)
                
                if length < alpha:
                    edges.append((p1_idx, p2_idx))
        
        # Draw result
        result = np.zeros_like(img, dtype=np.uint8)
        
        # Draw edges
        for p1_idx, p2_idx in edges:
            p1 = tuple(points[p1_idx])
            p2 = tuple(points[p2_idx])
            cv2.line(result, p1, p2, 128, 1)
        
        # Try to fill
        filled = result.copy()
        contours_result, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_result:
            cv2.drawContours(filled, contours_result, -1, 255, -1)
        
        axes[idx].imshow(filled, cmap='gray')
        axes[idx].set_title(f'Alpha = {alpha}\n{"Too tight" if alpha < 50 else "Too loose" if alpha > 150 else "Good balance"}',
                           fontsize=11, fontweight='bold',
                           color='red' if alpha < 50 or alpha > 150 else 'green')
        axes[idx].axis('off')
        
        # Calculate metrics
        if filled.max() > 0:
            binary_result = (filled > 127).astype(np.uint8)
            contours_metric, _ = cv2.findContours(binary_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_metric:
                largest_c = max(contours_metric, key=cv2.contourArea)
                perimeter = cv2.arcLength(largest_c, True)
                area = cv2.contourArea(largest_c)
                compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
                
                info = f"Compactness: {compactness:.2f}\n"
                info += f"# Edges: {len(edges)}"
                
                axes[idx].text(0.02, 0.98, info, transform=axes[idx].transAxes,
                              va='top', fontsize=9,
                              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                              fontfamily='monospace')
    
    plt.suptitle('Alpha Shape Parameter Tuning\n' +
                'Alpha controls maximum edge length = how "tight" the hull is',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/alpha_shape_parameter_tuning.png',
               dpi=150, bbox_inches='tight')
    print("✓ Alpha shape parameter tuning visualization created")
    plt.close()


def demonstrate_k_nearest_parameters():
    """Show how k affects nearest neighbor smoothing"""
    
    img_path = '/mnt/user-data/uploads/DJI_20250324094601_0036_V_March.png'
    img = cv2.imread(img_path, 0)
    binary = (img > 127).astype(np.uint8)
    
    # Get boundary points
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return
    
    largest = max(contours, key=cv2.contourArea)
    points = largest.squeeze()[::2]
    
    if len(points.shape) == 1:
        return
    
    k_values = [3, 5, 10, 20, 30, 50]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, k in enumerate(k_values):
        points_float = points.astype(np.float32).copy()
        
        # 3 iterations of smoothing
        for iteration in range(3):
            dist_mat = cdist(points_float, points_float)
            smoothed = np.zeros_like(points_float)
            
            for i in range(len(points_float)):
                nearest_idx = np.argsort(dist_mat[i])[1:k+1]
                nearest_points = points_float[nearest_idx]
                smoothed[i] = nearest_points.mean(axis=0)
            
            points_float = smoothed
        
        # Create result
        result = np.zeros_like(img)
        smoothed_int = np.round(points_float).astype(np.int32)
        cv2.fillPoly(result, [smoothed_int], 255)
        
        axes[idx].imshow(result, cmap='gray')
        axes[idx].set_title(f'k = {k} neighbors\n{"Less smooth" if k < 10 else "Over-smoothed" if k > 25 else "Good"}',
                           fontsize=11, fontweight='bold',
                           color='orange' if k < 10 or k > 25 else 'green')
        axes[idx].axis('off')
        
        # Calculate compactness
        binary_result = (result > 127).astype(np.uint8)
        contours_result, _ = cv2.findContours(binary_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_result:
            largest_c = max(contours_result, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_c, True)
            area = cv2.contourArea(largest_c)
            compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
            
            info = f"Compactness: {compactness:.2f}"
            axes[idx].text(0.02, 0.98, info, transform=axes[idx].transAxes,
                          va='top', fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                          fontfamily='monospace')
    
    plt.suptitle('K-Nearest Neighbor Smoothing Parameter Tuning\n' +
                'k controls how many neighbors influence each point',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/k_nearest_parameter_tuning.png',
               dpi=150, bbox_inches='tight')
    print("✓ K-nearest parameter tuning visualization created")
    plt.close()


def create_method_comparison_guide():
    """Create detailed comparison guide"""
    
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    guide_text = """
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║           SMOOTH POLYGON METHODS - DETAILED COMPARISON GUIDE               ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    
    YOUR IDEA: "Connect nearest points instead of outermost points"
    ────────────────────────────────────────────────────────────────────────────
    
    🎯 ALPHA SHAPE (Concave Hull) - EXACTLY YOUR IDEA! ⭐
    ───────────────────────────────────────────────────
    How it works:
      1. Create triangulation of all boundary points
      2. Keep only edges shorter than threshold (alpha)
      3. This connects nearby points, not distant ones
      4. Creates concave hull that follows shape closely
    
    Parameters:
      • alpha: Maximum edge length (20-100 for rivers)
        - Low (20-40): Very tight, preserves detail, may fragment
        - Medium (50-80): Balanced, follows shape well ✅
        - High (100+): Loose, approaches convex hull
    
    Best for:
      ✅ Following natural river shape
      ✅ Preserving concave regions (bends)
      ✅ Detail preservation with smoothing
    
    Code:
      from scipy.spatial import Delaunay
      tri = Delaunay(boundary_points)
      # Keep edges where length < alpha
    
    ────────────────────────────────────────────────────────────────────────────
    
    🎯 K-NEAREST NEIGHBOR SMOOTHING - ITERATIVE AVERAGING
    ─────────────────────────────────────────────────────
    How it works:
      1. For each point, find k nearest neighbors
      2. Move point to average position of neighbors
      3. Repeat for smooth convergence
    
    Parameters:
      • k: Number of neighbors (5-20 for rivers)
        - Low (3-5): Minimal smoothing, preserves shape
        - Medium (10-15): Good balance ✅
        - High (20+): Heavy smoothing, may over-simplify
      • iterations: How many times to repeat (2-5)
    
    Best for:
      ✅ Gradual smoothing
      ✅ Removing jagged edges
      ✅ Controllable smoothness
    
    Code:
      for point in boundary:
          neighbors = find_k_nearest(point, k)
          point_new = mean(neighbors)
    
    ────────────────────────────────────────────────────────────────────────────
    
    🎯 SPLINE INTERPOLATION - MATHEMATICAL SMOOTH CURVE
    ───────────────────────────────────────────────────
    How it works:
      1. Fit B-spline through boundary points
      2. Evaluate spline at many points for smooth curve
      3. Creates mathematically smooth result
    
    Parameters:
      • smoothing_factor: How much to smooth (1-10)
        - Low (1-3): Follows points closely
        - Medium (5-7): Smooth but natural ✅
        - High (10+): Very smooth, may distort
    
    Best for:
      ✅ Most visually smooth results
      ✅ Natural-looking boundaries
      ✅ Visualization and presentation
    
    Avoid for:
      ❌ Width measurement (changes boundaries)
      ❌ When detail preservation critical
    
    Code:
      from scipy.interpolate import splprep, splev
      tck, u = splprep([x, y], s=smoothing_factor)
      x_smooth, y_smooth = splev(u_new, tck)
    
    ────────────────────────────────────────────────────────────────────────────
    
    🎯 CHAIKIN CORNER CUTTING - GEOMETRIC SMOOTHING
    ───────────────────────────────────────────────
    How it works:
      1. For each edge, cut corners by creating 2 new points
      2. Point at 1/4 and 3/4 along edge
      3. Repeat to progressively smooth
    
    Parameters:
      • iterations: How many times to cut (2-5)
        - 1-2: Slight smoothing
        - 3-4: Good smoothing ✅
        - 5+: Very smooth, doubles points each time
    
    Best for:
      ✅ Simple algorithm
      ✅ Predictable results
      ✅ Fast processing
    
    Code:
      for edge in polygon:
          q = 0.75 * p0 + 0.25 * p1
          r = 0.25 * p0 + 0.75 * p1
          new_points.append([q, r])
    
    ────────────────────────────────────────────────────────────────────────────
    
    📊 COMPARISON TABLE
    ───────────────────────────────────────────────────────────────────────────
    
    Method                  | Smoothness | Detail | Speed | Width OK? | Your Idea?
    ────────────────────────┼────────────┼────────┼───────┼───────────┼───────────
    Alpha Shape             | Medium     | High   | Fast  | YES ✅    | YES ⭐
    K-Nearest Smoothing     | High       | Medium | Medium| Mostly ⚠️ | YES ⭐
    Spline Interpolation    | Very High  | Low    | Fast  | NO ❌     | Partial
    Chaikin Corner Cutting  | High       | Medium | Fast  | Mostly ⚠️ | Partial
    Rolling Ball            | High       | Low    | Fast  | NO ❌     | No
    Gaussian Blur           | Very High  | Low    | Fast  | NO ❌     | No
    Savitzky-Golay          | Medium     | High   | Fast  | YES ✅    | Partial
    
    ────────────────────────────────────────────────────────────────────────────
    
    🎯 RECOMMENDATIONS BY USE CASE
    ───────────────────────────────────────────────────────────────────────────
    
    📏 WIDTH MEASUREMENT (Your Main Goal):
       1st Choice: Alpha Shape (alpha=50-70) ⭐⭐⭐⭐⭐
       2nd Choice: Savitzky-Golay Filter    ⭐⭐⭐⭐
       3rd Choice: K-Nearest (k=5-8)        ⭐⭐⭐
    
    🎨 VISUALIZATION/PRESENTATION:
       1st Choice: Spline Interpolation     ⭐⭐⭐⭐⭐
       2nd Choice: Chaikin Corner Cutting   ⭐⭐⭐⭐
       3rd Choice: Gaussian Blur            ⭐⭐⭐
    
    🚀 SPEED CRITICAL:
       1st Choice: Rolling Ball             ⭐⭐⭐⭐⭐
       2nd Choice: Gaussian Blur            ⭐⭐⭐⭐⭐
       3rd Choice: Alpha Shape              ⭐⭐⭐⭐
    
    🔍 DETAIL PRESERVATION:
       1st Choice: Alpha Shape              ⭐⭐⭐⭐⭐
       2nd Choice: Savitzky-Golay           ⭐⭐⭐⭐
       3rd Choice: K-Nearest (low k)        ⭐⭐⭐
    
    ────────────────────────────────────────────────────────────────────────────
    
    💡 PARAMETER TUNING TIPS
    ───────────────────────────────────────────────────────────────────────────
    
    Alpha Shape:
      • Start with alpha = 50
      • If too fragmented → increase alpha
      • If too simple → decrease alpha
      • Sweet spot usually 40-80 for rivers
    
    K-Nearest:
      • Start with k = 10, iterations = 3
      • If too jagged → increase k
      • If over-smoothed → decrease k
      • More iterations = smoother but slower
    
    Spline:
      • Start with smoothing_factor = 5
      • If too wiggly → increase smoothing
      • If too smooth → decrease smoothing
      • Factor = 0 passes through all points exactly
    
    ────────────────────────────────────────────────────────────────────────────
    
    ✨ YOUR SPECIFIC QUESTION ANSWERED
    ───────────────────────────────────────────────────────────────────────────
    
    Question: "Can we connect nearest points instead of outermost points?"
    
    Answer: YES! This is called ALPHA SHAPE or CONCAVE HULL
    
    Implementation:
      1. Triangulate all boundary points (Delaunay)
      2. Set distance threshold (alpha)
      3. Keep only edges shorter than alpha
      4. This naturally connects nearby points
      5. Result: Smooth concave hull following river shape
    
    This is MUCH BETTER than convex hull for rivers because:
      ✅ Preserves bends and curves
      ✅ Follows natural shape
      ✅ Doesn't over-simplify
      ✅ Suitable for width measurement
      ✅ Exactly what you envisioned!
    
    ────────────────────────────────────────────────────────────────────────────
    """
    
    ax.text(0.02, 0.98, guide_text, transform=ax.transAxes,
           va='top', ha='left', fontsize=9.5,
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.2))
    
    plt.title('Complete Guide: Connecting Nearest Points for Smooth Polygons',
             fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/smooth_polygon_complete_guide.png',
               dpi=150, bbox_inches='tight')
    print("✓ Complete guide created")
    plt.close()


def main():
    print("\n" + "="*70)
    print("CREATING PARAMETER TUNING GUIDES")
    print("="*70 + "\n")
    
    print("1. Alpha shape parameter demonstration...")
    demonstrate_alpha_shape_parameters()
    
    print("\n2. K-nearest parameter demonstration...")
    demonstrate_k_nearest_parameters()
    
    print("\n3. Creating complete guide...")
    create_method_comparison_guide()
    
    print("\n" + "="*70)
    print("✓ All guides created successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
