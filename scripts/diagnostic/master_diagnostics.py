#!/usr/bin/env python3
"""
Master Diagnostic Script
Runs all verification steps to diagnose unexpected ablation study results
"""

import subprocess
import sys
from pathlib import Path
import argparse

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def run_feature_extraction_verification(image_path, mask_path=None, output_dir="./verification_output"):
    """Run feature extraction verification"""
    print_header("STEP 1: FEATURE EXTRACTION VERIFICATION")
    
    cmd = ["python3", "verify_feature_extraction.py", image_path]
    if mask_path:
        cmd.append(mask_path)
    cmd.append(output_dir)
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    
    return result.returncode == 0

def run_dataset_analysis(image_dir, mask_dir=None, output_dir="./dataset_analysis_output", sample_size=None):
    """Run dataset canopy density analysis"""
    print_header("STEP 2: DATASET CANOPY DENSITY ANALYSIS")
    
    cmd = ["python3", "analyze_dataset_canopy.py", image_dir]
    if mask_dir:
        cmd.append(mask_dir)
    cmd.append(output_dir)
    if sample_size:
        cmd.append(str(sample_size))
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    
    return result.returncode == 0

def run_unet_debugging(model_path=None, input_channels=3):
    """Run U-Net debugging"""
    print_header("STEP 3: U-NET DEBUGGING")
    
    if model_path:
        cmd = ["python3", "debug_unet.py", model_path, str(input_channels)]
        print(f"Running: {' '.join(cmd)}\n")
    else:
        cmd = ["python3", "debug_unet.py"]
        print("Running interactive U-Net debugging...\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    return result.returncode == 0

def generate_final_report(output_dir="./diagnostic_report"):
    """Generate comprehensive diagnostic report"""
    print_header("GENERATING FINAL DIAGNOSTIC REPORT")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    report = """
# COMPREHENSIVE DIAGNOSTIC REPORT
# Luminance-Prioritised Deep Learning for River Segmentation
# Date: """ + str(Path.cwd()) + """

## DIAGNOSTIC SUMMARY

This report consolidates findings from all verification steps:

### 1. Feature Extraction Verification
Location: ./verification_output/

Key checks:
- ✓ Feature ranges [0, 1]
- ✓ No NaN or infinite values
- ✓ Correlation analysis between luminance features
- ✓ Visual inspection of extracted features

### 2. Dataset Canopy Density Analysis
Location: ./dataset_analysis_output/

Key findings:
- Dataset canopy composition (sparse/moderate/dense/very_dense)
- Mean canopy density and distribution
- Luminance vs chrominance separability
- Color information preservation

### 3. U-Net Debugging
Key checks:
- ✓ Architecture configuration (output channels, activation)
- ✓ Forward pass functionality
- ✓ Loss computation and gradient flow
- ✓ Prediction thresholding

## NEXT STEPS BASED ON FINDINGS

### If Feature Extraction Issues Found:
1. Fix normalization of luminance channels
2. Remove highly correlated features (correlation > 0.95)
3. Verify color space conversions
4. Rerun ablation study with corrected features

### If Dataset Is Well-Lit (Canopy Density < 0.3):
1. Accept that RGB works best for your dataset
2. Revise hypothesis: "Luminance prioritization helps under DENSE canopy"
3. Split dataset by canopy density and retest
4. Focus paper on conditions where luminance helps

### If U-Net Implementation Issues Found:
1. Fix output layer (ensure 1 channel, sigmoid activation)
2. Check loss function (binary_crossentropy)
3. Verify data pipeline (normalization, mask format)
4. Retrain U-Net with corrections

### If No Issues Found:
This suggests that:
1. Random Forest feature importance ≠ CNN feature importance
2. Transfer learning (pre-trained weights) > hand-crafted features
3. RGB is genuinely better for this specific dataset
4. Paper should focus on this negative result and lessons learned

## RECOMMENDATIONS FOR CERI PAPER

### Option A: Honest Negative Result (Recommended)
Title: "When Feature Engineering Fails: A Case Study in Deep Learning 
       for Environmental Monitoring"

Key message:
- Feature importance analysis suggested luminance dominance
- Ablation study showed RGB baseline outperforms engineered features
- Analysis of discrepancy provides insights into transfer learning
- Lessons learned for practitioners

### Option B: Focus on What Works
Title: "Efficient Deep Learning for River Segmentation Under Forest Canopy
       Using Standard RGB Inputs"

Key message:
- Achieved 0.841 Dice with simple RGB + DeepLabv3+
- Pre-trained models outperform complex feature engineering
- Practical solution for OPW flood monitoring
- Simpler = better for deployment

### Option C: Conditional Success
Title: "Luminance-Prioritised Deep Learning: Performance Depends on
       Canopy Density"

Key message:
- Stratify analysis by canopy density
- Show luminance helps under dense canopy (>60% coverage)
- RGB works best for sparse canopy (<30% coverage)
- Provide decision tree for practitioners

## CONCLUSION

The unexpected results are scientifically valuable! They highlight:

1. Importance of validating feature importance with end-to-end evaluation
2. Power of transfer learning (pre-trained weights matter)
3. Gap between interpretability methods (RF) and model behavior (CNN)
4. Need for dataset-specific validation

Your research contributes to honest scientific discourse and helps
prevent others from making similar assumptions.

Next action: Review diagnostic outputs and decide on paper narrative.
"""
    
    report_path = output_dir / "FINAL_DIAGNOSTIC_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Final report saved to: {report_path}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Master diagnostic script for ablation study analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full diagnostic with sample image and dataset
  python master_diagnostics.py \\
    --sample-image ./data/sample.jpg \\
    --sample-mask ./data/sample_mask.png \\
    --dataset-dir ./data/images \\
    --mask-dir ./data/masks \\
    --unet-model ./models/unet_rgb.h5

  # Quick check with just sample image
  python master_diagnostics.py --sample-image ./data/sample.jpg

  # Dataset analysis only
  python master_diagnostics.py --dataset-dir ./data/images --skip-feature-check
        """
    )
    
    # Feature extraction args
    parser.add_argument('--sample-image', type=str, 
                       help='Path to sample image for feature extraction verification')
    parser.add_argument('--sample-mask', type=str,
                       help='Path to sample mask (optional)')
    parser.add_argument('--feature-output', type=str, default='./verification_output',
                       help='Output directory for feature verification')
    
    # Dataset analysis args
    parser.add_argument('--dataset-dir', type=str,
                       help='Directory containing all training images')
    parser.add_argument('--mask-dir', type=str,
                       help='Directory containing all masks (optional)')
    parser.add_argument('--dataset-output', type=str, default='./dataset_analysis_output',
                       help='Output directory for dataset analysis')
    parser.add_argument('--sample-size', type=int,
                       help='Number of images to sample for analysis (default: all)')
    
    # U-Net debugging args
    parser.add_argument('--unet-model', type=str,
                       help='Path to U-Net model file (.h5 or .keras)')
    parser.add_argument('--input-channels', type=int, default=3,
                       help='Number of input channels (3/8/18)')
    
    # Control flow
    parser.add_argument('--skip-feature-check', action='store_true',
                       help='Skip feature extraction verification')
    parser.add_argument('--skip-dataset-analysis', action='store_true',
                       help='Skip dataset analysis')
    parser.add_argument('--skip-unet-debug', action='store_true',
                       help='Skip U-Net debugging')
    parser.add_argument('--final-report-dir', type=str, default='./diagnostic_report',
                       help='Directory for final diagnostic report')
    
    args = parser.parse_args()
    
    print_header("MASTER DIAGNOSTIC PIPELINE")
    print("This script will run comprehensive diagnostics to identify")
    print("the root causes of unexpected ablation study results.\n")
    
    success_flags = {
        'feature_extraction': None,
        'dataset_analysis': None,
        'unet_debugging': None
    }
    
    # Step 1: Feature Extraction Verification
    if not args.skip_feature_check:
        if args.sample_image:
            success = run_feature_extraction_verification(
                args.sample_image, 
                args.sample_mask,
                args.feature_output
            )
            success_flags['feature_extraction'] = success
        else:
            print("\n⚠️  Skipping feature extraction (no --sample-image provided)")
            print("   Provide --sample-image to verify feature extraction")
    
    # Step 2: Dataset Analysis
    if not args.skip_dataset_analysis:
        if args.dataset_dir:
            success = run_dataset_analysis(
                args.dataset_dir,
                args.mask_dir,
                args.dataset_output,
                args.sample_size
            )
            success_flags['dataset_analysis'] = success
        else:
            print("\n⚠️  Skipping dataset analysis (no --dataset-dir provided)")
            print("   Provide --dataset-dir to analyze canopy density distribution")
    
    # Step 3: U-Net Debugging
    if not args.skip_unet_debug:
        if args.unet_model or not any([args.sample_image, args.dataset_dir]):
            success = run_unet_debugging(args.unet_model, args.input_channels)
            success_flags['unet_debugging'] = success
        else:
            print("\n⚠️  Skipping U-Net debugging (no --unet-model provided)")
            print("   Provide --unet-model to debug U-Net implementation")
    
    # Generate final report
    generate_final_report(args.final_report_dir)
    
    # Print summary
    print_header("DIAGNOSTIC PIPELINE COMPLETE")
    
    print("Results:")
    for step, success in success_flags.items():
        if success is None:
            status = "⊝ SKIPPED"
        elif success:
            status = "✓ SUCCESS"
        else:
            status = "❌ FAILED"
        print(f"  {status:12s} {step.replace('_', ' ').title()}")
    
    print("\nOutput locations:")
    if not args.skip_feature_check and args.sample_image:
        print(f"  - Feature verification: {args.feature_output}")
    if not args.skip_dataset_analysis and args.dataset_dir:
        print(f"  - Dataset analysis:     {args.dataset_output}")
    print(f"  - Final report:         {args.final_report_dir}")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("\n1. Review all visualization outputs")
    print("2. Read FINAL_DIAGNOSTIC_REPORT.md")
    print("3. Based on findings, choose next action:")
    print("   a) Fix feature extraction and retrain")
    print("   b) Accept RGB works best and focus paper on this")
    print("   c) Stratify by canopy density and re-analyze")
    print("   d) Fix U-Net and retrain")
    print("\n4. Update CERI conference paper based on findings")
    print("\n")

if __name__ == '__main__':
    main()
