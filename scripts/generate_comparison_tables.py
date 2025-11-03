"""
Generate Comprehensive Comparison Tables for Paper
===================================================
Creates publication-ready tables comparing:
1. Pretrained vs No-Pretrained for each model
2. Feature configurations within each training regime
3. Cross-analysis to identify bottlenecks

Usage:
    python generate_comparison_tables.py --model deeplabv3plus
    python generate_comparison_tables.py --model both --save_latex
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(pretrain_dir, no_pretrain_dir, model_type):
    """Load both pretrained and no-pretrained results"""
    
    # Load pretrained results
    pretrain_path = Path(pretrain_dir) / model_type / 'ablation_study_results.csv'
    
    if not pretrain_path.exists():
        print(f"Warning: Pretrained results not found at {pretrain_path}")
        df_pretrain = None
    else:
        df_pretrain = pd.read_csv(pretrain_path)
        df_pretrain['training_type'] = 'Pretrained'
    
    # Load no-pretrained results
    no_pretrain_path = Path(no_pretrain_dir) / model_type / 'ablation_no_pretrain_results.csv'
    
    if not no_pretrain_path.exists():
        print(f"Warning: No-pretrain results not found at {no_pretrain_path}")
        df_no_pretrain = None
    else:
        df_no_pretrain = pd.read_csv(no_pretrain_path)
        df_no_pretrain['training_type'] = 'From Scratch'
    
    return df_pretrain, df_no_pretrain


def create_paper_table_1(df_pretrain, df_no_pretrain, model_type):
    """
    Table 1: Feature Configuration Comparison (Pretrained vs Scratch)
    
    Format:
    | Config | Pretrain Dice | Scratch Dice | Δ Dice | Benefit (%) |
    """
    
    if df_pretrain is None or df_no_pretrain is None:
        return "Missing data for comparison"
    
    # Merge dataframes
    comparison = pd.merge(
        df_pretrain[['feature_config', 'best_val_dice', 'best_val_iou']],
        df_no_pretrain[['feature_config', 'best_val_dice', 'best_val_iou']],
        on='feature_config',
        suffixes=('_pretrain', '_scratch')
    )
    
    # Calculate benefits
    comparison['dice_benefit'] = (
        comparison['best_val_dice_pretrain'] - comparison['best_val_dice_scratch']
    )
    comparison['dice_benefit_pct'] = (
        comparison['dice_benefit'] / comparison['best_val_dice_scratch'] * 100
    )
    
    comparison['iou_benefit'] = (
        comparison['best_val_iou_pretrain'] - comparison['best_val_iou_scratch']
    )
    
    # Sort by pretrain performance
    comparison = comparison.sort_values('best_val_dice_pretrain', ascending=False)
    
    # Format for paper
    table = f"""
Table 1: {model_type.upper()} - Pretrained vs From-Scratch Training
{'='*80}

| Configuration | Pretrained | From Scratch | Δ Dice  | Benefit | IoU (P) | IoU (S) |
|--------------|------------|--------------|---------|---------|---------|---------|
"""
    
    for _, row in comparison.iterrows():
        config = row['feature_config'].capitalize()
        dice_p = row['best_val_dice_pretrain']
        dice_s = row['best_val_dice_scratch']
        delta = row['dice_benefit']
        benefit = row['dice_benefit_pct']
        iou_p = row['best_val_iou_pretrain']
        iou_s = row['best_val_iou_scratch']
        
        table += f"| {config:12s} | {dice_p:.4f}     | {dice_s:.4f}       | {delta:+.4f} | {benefit:+6.2f}% | {iou_p:.4f}  | {iou_s:.4f}  |\n"
    
    table += "\nP = Pretrained, S = From Scratch\n"
    table += "Δ Dice = Pretrained - Scratch (positive = pretrained better)\n"
    
    return table, comparison


def create_paper_table_2(df_pretrain, df_no_pretrain):
    """
    Table 2: Critical Analysis - Does Projection Hurt?
    
    Key question: Do multi-channel configs perform better without pretrained weights?
    """
    
    if df_pretrain is None or df_no_pretrain is None:
        return "Missing data for analysis"
    
    analysis = """
Table 2: Critical Analysis - Feature Adaptation Impact
{'='*80}

Question: Is the projection layer (10ch→3ch) hurting performance?
Method: Compare ranking changes between pretrained and scratch training

"""
    
    # Get rankings
    pretrain_ranked = df_pretrain.sort_values('best_val_dice', ascending=False).reset_index(drop=True)
    scratch_ranked = df_no_pretrain.sort_values('best_val_dice', ascending=False).reset_index(drop=True)
    
    analysis += "\nPRETRAINED RANKING:\n"
    for i, row in pretrain_ranked.iterrows():
        analysis += f"  {i+1}. {row['feature_config'].upper():12s} - Dice: {row['best_val_dice']:.4f}\n"
    
    analysis += "\nFROM SCRATCH RANKING:\n"
    for i, row in scratch_ranked.iterrows():
        analysis += f"  {i+1}. {row['feature_config'].upper():12s} - Dice: {row['best_val_dice']:.4f}\n"
    
    # Key findings
    analysis += "\n" + "="*80 + "\n"
    analysis += "KEY FINDINGS:\n\n"
    
    # Check if RGB wins in both cases
    pretrain_winner = pretrain_ranked.iloc[0]['feature_config']
    scratch_winner = scratch_ranked.iloc[0]['feature_config']
    
    analysis += f"1. Pretrained Winner: {pretrain_winner.upper()}\n"
    analysis += f"2. Scratch Winner: {scratch_winner.upper()}\n\n"
    
    # Compare All vs RGB in both regimes
    rgb_pretrain = df_pretrain[df_pretrain['feature_config'] == 'rgb']['best_val_dice'].values[0]
    all_pretrain = df_pretrain[df_pretrain['feature_config'] == 'all']['best_val_dice'].values[0]
    
    rgb_scratch = df_no_pretrain[df_no_pretrain['feature_config'] == 'rgb']['best_val_dice'].values[0]
    all_scratch = df_no_pretrain[df_no_pretrain['feature_config'] == 'all']['best_val_dice'].values[0]
    
    analysis += "3. RGB vs ALL FEATURES:\n"
    analysis += f"   Pretrained: RGB {rgb_pretrain:.4f} vs All {all_pretrain:.4f} "
    if rgb_pretrain > all_pretrain:
        gap_p = ((rgb_pretrain - all_pretrain) / all_pretrain * 100)
        analysis += f"(RGB wins by +{gap_p:.2f}%)\n"
    else:
        gap_p = ((all_pretrain - rgb_pretrain) / rgb_pretrain * 100)
        analysis += f"(All wins by +{gap_p:.2f}%)\n"
    
    analysis += f"   Scratch:    RGB {rgb_scratch:.4f} vs All {all_scratch:.4f} "
    if rgb_scratch > all_scratch:
        gap_s = ((rgb_scratch - all_scratch) / all_scratch * 100)
        analysis += f"(RGB wins by +{gap_s:.2f}%)\n"
    else:
        gap_s = ((all_scratch - rgb_scratch) / rgb_scratch * 100)
        analysis += f"(All wins by +{gap_s:.2f}%)\n"
    
    # Conclusion
    analysis += "\n4. CONCLUSION:\n"
    
    if rgb_pretrain > all_pretrain and rgb_scratch > all_scratch:
        analysis += "   → RGB dominates in BOTH training regimes\n"
        analysis += "   → Additional features don't help (inherent to task)\n"
        analysis += "   → Projection layer is NOT the main bottleneck\n"
    elif rgb_pretrain > all_pretrain and all_scratch > rgb_scratch:
        analysis += "   ✓ CRITICAL FINDING: All features WIN when trained from scratch!\n"
        analysis += "   → Projection layer (10ch→3ch) IS the bottleneck\n"
        analysis += "   → Additional features ARE useful but adaptation loses information\n"
        analysis += "   → Recommendation: Develop better adaptation strategy\n"
    elif all_pretrain > rgb_pretrain and rgb_scratch > all_scratch:
        analysis += "   → Pretrained weights help multi-channel adaptation\n"
        analysis += "   → But features struggle to learn from scratch\n"
    else:
        analysis += "   → All features win in BOTH regimes\n"
        analysis += "   → Current pretrained approach already works well\n"
    
    return analysis


def create_paper_table_3(df_pretrain, df_no_pretrain, model_type):
    """
    Table 3: Computational Comparison
    """
    
    if df_pretrain is None or df_no_pretrain is None:
        return "Missing data"
    
    table = f"""
Table 3: {model_type.upper()} - Computational Cost Comparison
{'='*80}

| Config       | Channels | Params  | Pretrain Time | Scratch Time | Time Ratio |
|--------------|----------|---------|---------------|--------------|------------|
"""
    
    for config in ['rgb', 'luminance', 'chrominance', 'all']:
        pretrain_row = df_pretrain[df_pretrain['feature_config'] == config]
        scratch_row = df_no_pretrain[df_no_pretrain['feature_config'] == config]
        
        if pretrain_row.empty or scratch_row.empty:
            continue
        
        channels = pretrain_row['n_channels'].values[0]
        params = pretrain_row['total_params'].values[0] / 1e6  # Millions
        
        time_p = pretrain_row.get('time_hours', pd.Series([0])).values[0]
        time_s = scratch_row.get('time_hours', pd.Series([0])).values[0]
        
        if time_s > 0:
            ratio = time_s / time_p
        else:
            ratio = 0
        
        table += f"| {config.capitalize():12s} | {channels:8d} | {params:6.2f}M | {time_p:12.2f}h | {time_s:11.2f}h | {ratio:9.2f}x |\n"
    
    return table


def create_visualization(df_pretrain, df_no_pretrain, model_type, output_dir):
    """Create comparison visualizations"""
    
    if df_pretrain is None or df_no_pretrain is None:
        print("Missing data for visualization")
        return
    
    # Prepare data
    configs = ['RGB', 'Luminance', 'Chrominance', 'All']
    
    pretrain_dice = []
    scratch_dice = []
    
    for config in ['rgb', 'luminance', 'chrominance', 'all']:
        p_val = df_pretrain[df_pretrain['feature_config'] == config]['best_val_dice'].values
        s_val = df_no_pretrain[df_no_pretrain['feature_config'] == config]['best_val_dice'].values
        
        pretrain_dice.append(p_val[0] if len(p_val) > 0 else 0)
        scratch_dice.append(s_val[0] if len(s_val) > 0 else 0)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pretrain_dice, width, label='Pretrained', color='#2E86AB')
    bars2 = ax.bar(x + width/2, scratch_dice, width, label='From Scratch', color='#A23B72')
    
    ax.set_xlabel('Feature Configuration', fontsize=12)
    ax.set_ylabel('Dice Coefficient', fontsize=12)
    ax.set_title(f'{model_type.upper()}: Pretrained vs From-Scratch Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig_path = output_path / f'{model_type}_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {fig_path}")
    
    plt.close()


def generate_latex_table(comparison, model_type):
    """Generate LaTeX table for paper"""
    
    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{model_type.upper()}: Pretrained vs From-Scratch Training}}
\\label{{tab:{model_type}_comparison}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
Configuration & \\multicolumn{{2}}{{c}}{{Dice Coefficient}} & $\\Delta$ Dice & Benefit & \\multicolumn{{2}}{{c}}{{IoU}} \\\\
             & Pretrained & Scratch & & (\\%) & Pretrained & Scratch \\\\
\\midrule
"""
    
    for _, row in comparison.iterrows():
        config = row['feature_config'].capitalize()
        dice_p = row['best_val_dice_pretrain']
        dice_s = row['best_val_dice_scratch']
        delta = row['dice_benefit']
        benefit = row['dice_benefit_pct']
        iou_p = row['best_val_iou_pretrain']
        iou_s = row['best_val_iou_scratch']
        
        latex += f"{config} & {dice_p:.4f} & {dice_s:.4f} & {delta:+.4f} & {benefit:+.2f} & {iou_p:.4f} & {iou_s:.4f} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    return latex


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive comparison tables"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='both',
        choices=['unet', 'deeplabv3plus', 'both'],
        help='Model to analyze'
    )
    
    parser.add_argument(
        '--pretrain_dir',
        type=str,
        default='experiments/results/ablation_study/pretrain',
        help='Directory with pretrained results'
    )
    
    parser.add_argument(
        '--no_pretrain_dir',
        type=str,
        default='experiments/results/ablation_study/no_pretrain',
        help='Directory with no-pretrain results'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/results/comparison',
        help='Output directory for tables and plots'
    )
    
    parser.add_argument(
        '--save_latex',
        action='store_true',
        help='Generate LaTeX tables'
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    models = ['unet', 'deeplabv3plus'] if args.model == 'both' else [args.model]
    
    for model_type in models:
        print("\n" + "="*80)
        print(f"GENERATING COMPARISON TABLES - {model_type.upper()}")
        print("="*80)
        
        # Load results
        df_pretrain, df_no_pretrain = load_results(
            args.pretrain_dir,
            args.no_pretrain_dir,
            model_type
        )
        
        if df_pretrain is None or df_no_pretrain is None:
            print(f"Skipping {model_type} - missing data")
            continue
        
        # Generate Table 1
        print("\nGenerating Table 1: Feature Configuration Comparison...")
        table1, comparison = create_paper_table_1(df_pretrain, df_no_pretrain, model_type)
        print(table1)
        
        table1_path = output_path / f'{model_type}_table1_comparison.txt'
        with open(table1_path, 'wb') as f:
            f.write(table1.encode('utf-8'))
        print(f"✓ Saved to {table1_path}")
        
        # Generate Table 2
        print("\nGenerating Table 2: Critical Analysis...")
        table2 = create_paper_table_2(df_pretrain, df_no_pretrain)
        print(table2)
        
        table2_path = output_path / f'{model_type}_table2_analysis.txt'
        with open(table2_path, 'wb') as f:
            f.write(table2.encode('utf-8'))
        print(f"✓ Saved to {table2_path}")
        
        # Generate Table 3
        print("\nGenerating Table 3: Computational Comparison...")
        table3 = create_paper_table_3(df_pretrain, df_no_pretrain, model_type)
        print(table3)
        
        table3_path = output_path / f'{model_type}_table3_computational.txt'
        with open(table3_path, 'wb') as f:
            f.write(table3.encode('utf-8'))
        print(f"✓ Saved to {table3_path}")
        
        # Generate visualization
        print("\nGenerating visualization...")
        create_visualization(df_pretrain, df_no_pretrain, model_type, args.output_dir)
        
        # Generate LaTeX if requested
        if args.save_latex:
            print("\nGenerating LaTeX table...")
            latex = generate_latex_table(comparison, model_type)
            
            latex_path = output_path / f'{model_type}_latex_table.tex'
            with open(latex_path, 'w') as f:
                f.write(latex)
            print(f"✓ LaTeX table saved to {latex_path}")
    
    print("\n" + "="*80)
    print("✓ All comparison tables generated successfully!")
    print(f"✓ Results saved to {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
