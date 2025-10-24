"""
Visualization Utilities for Feature Importance Analysis
========================================================
Create publication-quality figures for CERI conference paper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import shap
from pathlib import Path

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150
sns.set_palette("colorblind")


class FeatureImportanceVisualizer:
    """Create visualizations for feature importance analysis"""
    
    def __init__(self, results: Dict, output_dir: str = "experiments/results/figures"):
        """
        Args:
            results: Results dictionary from FeatureImportanceAnalyzer
            output_dir: Directory to save figures
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_feature_importance_comparison(self, save: bool = True):
        """
        Compare feature importances across three methods.
        Creates Figure for paper showing MDI, Permutation, and SHAP importance.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        methods = [
            ('mdi_importance', 'importance', 'Mean Decrease in Impurity'),
            ('permutation_importance', 'importance_mean', 'Permutation Importance'),
            ('shap_importance', 'shap_importance', 'SHAP Values')
        ]
        
        for ax, (method_key, col, title) in zip(axes, methods):
            if method_key not in self.results:
                continue
            
            df = self.results[method_key].head(18)  # All 18 features
            
            # Color-code: luminance (blue) vs chrominance (orange)
            colors = ['#1f77b4' if i < 8 else '#ff7f0e' for i in range(len(df))]
            
            # Horizontal bar plot
            ax.barh(range(len(df)), df[col], color=colors)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df['feature'], fontsize=9)
            ax.set_xlabel('Importance', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Luminance (8 features)'),
            Patch(facecolor='#ff7f0e', label='Chrominance (10 features)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'feature_importance_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / 'feature_importance_comparison.pdf', 
                       bbox_inches='tight')
        
        plt.show()
        
    def plot_luminance_contribution_by_method(self, save: bool = True):
        """
        Bar chart showing luminance vs chrominance contribution across methods.
        Key figure for demonstrating research hypothesis.
        """
        if 'luminance_contribution' not in self.results:
            print("Warning: No luminance contribution data found")
            return
        
        contrib = self.results['luminance_contribution']
        
        methods = list(contrib.keys())
        luminance_pcts = [contrib[m]['luminance_pct'] for m in methods]
        chrominance_pcts = [contrib[m]['chrominance_pct'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, luminance_pcts, width, 
                      label='Luminance', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, chrominance_pcts, width,
                      label='Chrominance', color='#ff7f0e', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)