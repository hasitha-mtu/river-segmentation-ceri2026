"""
Test Setup Script
=================
Verify that all dependencies and modules are correctly installed and working.

Usage:
    python test_setup.py
"""

import sys
import importlib
from pathlib import Path


def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name:30s} - OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name:30s} - FAILED: {e}")
        return False


def test_tensorflow_gpu():
    """Test TensorFlow GPU availability"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ TensorFlow GPU                - {len(gpus)} GPU(s) available")
            for gpu in gpus:
                print(f"    - {gpu.name}")
            return True
        else:
            print("⚠ TensorFlow GPU                - No GPU found (CPU mode)")
            return True
    except Exception as e:
        print(f"✗ TensorFlow GPU                - Error: {e}")
        return False


def test_custom_modules():
    """Test custom modules"""
    sys.path.append('src/data')
    sys.path.append('src/experiments')
    sys.path.append('src/utils')
    
    modules = [
        'feature_extraction',
        'feature_importance',
        'visualization'
    ]
    
    all_ok = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module:30s} - OK")
        except ImportError as e:
            print(f"✗ {module:30s} - FAILED: {e}")
            all_ok = False
    
    return all_ok


def test_data_directories():
    """Check if data directories exist"""
    paths = [
        'data/raw/images',
        'data/raw/masks',
        'config'
    ]
    
    all_ok = True
    for path in paths:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                n_files = len(list(p.glob('*')))
                print(f"✓ {str(p):30s} - Exists ({n_files} files)")
            else:
                print(f"✓ {str(p):30s} - Exists")
        else:
            print(f"⚠ {str(p):30s} - Not found (create if needed)")
            all_ok = False
    
    return all_ok


def test_feature_extraction():
    """Test feature extraction functionality"""
    try:
        import numpy as np
        sys.path.append('src/data')
        from feature_extraction import FeatureExtractor
        
        # Create test image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Initialize extractor
        extractor = FeatureExtractor()
        
        # Test all feature extraction methods
        all_features = extractor.extract_all_features(test_image)
        assert all_features.shape == (512, 512, 18), f"Expected (512, 512, 18), got {all_features.shape}"
        
        luminance = extractor.extract_luminance_only(test_image)
        assert luminance.shape == (512, 512, 8), f"Expected (512, 512, 8), got {luminance.shape}"
        
        chrominance = extractor.extract_chrominance_only(test_image)
        assert chrominance.shape == (512, 512, 10), f"Expected (512, 512, 10), got {chrominance.shape}"
        
        rgb = extractor.extract_rgb_only(test_image)
        assert rgb.shape == (512, 512, 3), f"Expected (512, 512, 3), got {rgb.shape}"
        
        print(f"✓ Feature Extraction Test       - OK (18 features extracted)")
        return True
        
    except Exception as e:
        print(f"✗ Feature Extraction Test       - FAILED: {e}")
        return False


def test_augmentation():
    """Test data augmentation"""
    try:
        import numpy as np
        import cv2
        sys.path.append('src/data')
        from augmentation import RiverAugmentation, create_multiscale_crops
        
        # Create test data
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
        
        # Test augmentation
        augmenter = RiverAugmentation()
        aug_img, aug_mask = augmenter(test_image, test_mask)
        
        assert aug_img.shape == test_image.shape
        assert aug_mask.shape == test_mask.shape
        
        # Test multiscale crops
        high_res_img = np.random.randint(0, 255, (3956, 5280, 3), dtype=np.uint8)
        high_res_mask = np.random.randint(0, 2, (3956, 5280), dtype=np.uint8) * 255
        
        crop_imgs, crop_masks = create_multiscale_crops(
            high_res_img, high_res_mask,
            crop_sizes=[(2048, 2048), (1024, 1024)],
            target_size=(512, 512),
            n_crops_per_size=2
        )
        
        assert len(crop_imgs) == 4  # 2 sizes × 2 crops each
        assert all(img.shape == (512, 512, 3) for img in crop_imgs)
        
        print(f"✓ Data Augmentation Test        - OK")
        return True
        
    except Exception as e:
        print(f"✗ Data Augmentation Test        - FAILED: {e}")
        return False


def print_system_info():
    """Print system information"""
    import platform
    import numpy as np
    
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy version: {np.__version__}")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except:
        print("TensorFlow: Not installed")
    
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except:
        print("OpenCV: Not installed")
    
    print()


def main():
    """Run all tests"""
    print("="*70)
    print("TESTING RIVER SEGMENTATION SETUP")
    print("="*70)
    print()
    
    # Print system info
    print_system_info()
    
    print("="*70)
    print("DEPENDENCY CHECK")
    print("="*70)
    
    # Test core dependencies
    dependencies = [
        ('tensorflow', 'TensorFlow'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
        ('shap', 'SHAP'),
        ('albumentations', 'Albumentations'),
    ]
    
    dep_results = []
    for module, name in dependencies:
        dep_results.append(test_import(module, name))
    
    print()
    
    # Test TensorFlow GPU
    print("="*70)
    print("GPU CHECK")
    print("="*70)
    gpu_ok = test_tensorflow_gpu()
    print()
    
    # Test custom modules
    print("="*70)
    print("CUSTOM MODULES CHECK")
    print("="*70)
    custom_ok = test_custom_modules()
    print()
    
    # Test data directories
    print("="*70)
    print("DIRECTORY STRUCTURE CHECK")
    print("="*70)
    dir_ok = test_data_directories()
    print()
    
    # Test feature extraction
    print("="*70)
    print("FUNCTIONAL TESTS")
    print("="*70)
    feature_ok = test_feature_extraction()
    aug_ok = test_augmentation()
    print()
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_deps_ok = all(dep_results)
    
    results = {
        'Dependencies': all_deps_ok,
        'GPU Support': gpu_ok,
        'Custom Modules': custom_ok,
        'Directories': dir_ok,
        'Feature Extraction': feature_ok,
        'Augmentation': aug_ok
    }
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:25s} {status}")
    
    print()
    
    all_ok = all(results.values())
    
    if all_ok:
        print("="*70)
        print("✓ ALL TESTS PASSED - SETUP COMPLETE!")
        print("="*70)
        print("\nYou can now proceed with:")
        print("  1. python prepare_dataset.py")
        print("  2. python run_experiment1.py --stratify --visualize")
        print()
        return 0
    else:
        print("="*70)
        print("⚠ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease address the failures above before proceeding.")
        print("\nCommon fixes:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - Missing modules: Ensure all .py files are in src/ directory")
        print("  - Missing directories: Create data/raw/images and data/raw/masks")
        print()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
