"""
Enhanced River Segmentation Model Testing with Multi-Channel Support
=====================================================================
Tests models trained with different feature configurations:
- RGB baseline (3 channels)
- Luminance only (8 channels)
- All features (18 channels: 8 luminance + 10 chrominance)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json
from datetime import datetime
from pathlib import Path

# Import feature extraction
sys.path.append('src/data')
try:
    from feature_extraction import FeatureExtractor
except ImportError:
    print("Warning: feature_extraction.py not found. Place it in src/data/ or current directory.")
    FeatureExtractor = None

sys.path.append('src/models')
from unet import combined_loss, dice_coefficient, dice_loss, iou_metric

class MultiChannelModelTester:
    def __init__(self, model_path, test_data_dir, feature_config='all', results_dir='test_results'):
        """
        Initialize tester with multi-channel support
        
        Args:
            model_path: Path to saved model (.h5 or SavedModel format)
            test_data_dir: Directory containing test images and masks
            feature_config: Feature configuration ('rgb', 'luminance', 'chrominance', 'all')
            results_dir: Directory to save test results
        """
        self.model_path = model_path
        self.test_data_dir = test_data_dir
        self.feature_config = feature_config
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize feature extractor
        if FeatureExtractor is not None:
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor = None
            print("Warning: FeatureExtractor not available, will use RGB only")
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = keras.models.load_model(model_path,
                                             custom_objects={'combined_loss': combined_loss,
                                                             'dice_coefficient': dice_coefficient,
                                                             'dice_loss': dice_loss,
                                                             'iou_metric': iou_metric})
        print("✓ Model loaded successfully")
        
        # Get expected input channels
        self.expected_channels = self.model.input_shape[-1]
        print(f"✓ Model expects {self.expected_channels} input channels")
        
        # Validate feature config matches model
        self._validate_feature_config()
        
        self.test_images = []
        self.test_masks = []
        self.test_features = []
        self.predictions = []
        self.filenames = []
        
    def _validate_feature_config(self):
        """Validate that feature config matches model input channels"""
        channel_map = {
            'rgb': 3,
            'luminance': 8,
            'chrominance': 10,
            'all': 18
        }
        
        expected = channel_map.get(self.feature_config, 18)
        
        if expected != self.expected_channels:
            print(f"\n⚠ Warning: Feature config '{self.feature_config}' provides {expected} channels")
            print(f"  but model expects {self.expected_channels} channels")
            print(f"  Adjusting feature_config automatically...")
            
            # Find matching config
            for config, channels in channel_map.items():
                if channels == self.expected_channels:
                    self.feature_config = config
                    print(f"  → Using feature_config='{config}'")
                    break
    
    def _extract_features(self, image):
        """Extract features based on configuration"""
        if self.feature_extractor is None or self.feature_config == 'rgb':
            # RGB baseline (3 channels)
            features = image.astype(np.float32) / 255.0
        elif self.feature_config == 'luminance':
            # Luminance only (8 channels)
            features = self.feature_extractor.extract_luminance_only(image)
            features = self.feature_extractor.normalize_features(features)
        elif self.feature_config == 'chrominance':
            # Chrominance only (10 channels)
            features = self.feature_extractor.extract_chrominance_only(image)
            features = self.feature_extractor.normalize_features(features)
        elif self.feature_config == 'all':
            # All features (18 channels)
            features = self.feature_extractor.extract_all_features(image)
            features = self.feature_extractor.normalize_features(features)
        else:
            raise ValueError(f"Unknown feature_config: {self.feature_config}")
        
        return features
    
    def load_test_data(self, img_size=(512, 512)):
        """Load test images, extract features, and load masks"""
        print(f"\nLoading test data from {self.test_data_dir}")
        
        images_dir = os.path.join(self.test_data_dir, 'images')
        masks_dir = os.path.join(self.test_data_dir, 'masks')
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(image_files)} images")
        print(f"Extracting features using config: '{self.feature_config}' ({self.expected_channels} channels)")
        
        for img_file in image_files:
            # Load RGB image
            img_path = os.path.join(images_dir, img_file)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, img_size)
            img_np = img.numpy().astype(np.uint8)
            
            # Extract features
            features = self._extract_features(img_np)
            
            # Load corresponding mask
            mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                mask = tf.io.read_file(mask_path)
                mask = tf.image.decode_image(mask, channels=1)
                mask = tf.image.resize(mask, img_size, method='nearest')
                mask = tf.cast(mask, tf.float32) / 255.0
                
                self.test_images.append(img_np)
                self.test_features.append(features)
                self.test_masks.append(mask.numpy())
                self.filenames.append(img_file)
        
        self.test_images = np.array(self.test_images)
        self.test_features = np.array(self.test_features)
        self.test_masks = np.array(self.test_masks)
        
        print(f"✓ Loaded {len(self.test_images)} test images")
        print(f"✓ RGB images shape: {self.test_images.shape}")
        print(f"✓ Features shape: {self.test_features.shape}")
        print(f"✓ Masks shape: {self.test_masks.shape}")
        
        return self.test_features, self.test_masks
    
    def run_inference(self, batch_size=8):
        """Run inference on test dataset using extracted features"""
        print(f"\nRunning inference on test dataset...")
        print(f"Input shape: {self.test_features.shape}")
        
        self.predictions = self.model.predict(self.test_features, batch_size=batch_size)
        
        # Threshold predictions to binary masks
        self.predictions_binary = (self.predictions > 0.5).astype(np.float32)
        
        print("✓ Inference completed")
        print(f"✓ Predictions shape: {self.predictions.shape}")
        
        return self.predictions, self.predictions_binary
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        print("\nCalculating metrics...")
        
        metrics = {}
        
        # Flatten arrays for pixel-wise metrics
        y_true = self.test_masks.flatten()
        y_pred = self.predictions_binary.flatten()
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                             (metrics['precision'] + metrics['recall']) if \
                             (metrics['precision'] + metrics['recall']) > 0 else 0
        metrics['iou'] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        metrics['dice_coefficient'] = 2 * tp / (2 * tp + fp + fn) if \
                                     (2 * tp + fp + fn) > 0 else 0
        
        # Per-image metrics
        image_ious = []
        image_dice = []
        
        for i in range(len(self.test_masks)):
            true_mask = self.test_masks[i].flatten()
            pred_mask = self.predictions_binary[i].flatten()
            
            intersection = np.sum(true_mask * pred_mask)
            union = np.sum(true_mask) + np.sum(pred_mask) - intersection
            
            iou = intersection / union if union > 0 else 0
            dice = 2 * intersection / (np.sum(true_mask) + np.sum(pred_mask)) \
                   if (np.sum(true_mask) + np.sum(pred_mask)) > 0 else 0
            
            image_ious.append(iou)
            image_dice.append(dice)
        
        metrics['mean_iou_per_image'] = np.mean(image_ious)
        metrics['mean_dice_per_image'] = np.mean(image_dice)
        metrics['std_iou_per_image'] = np.std(image_ious)
        metrics['std_dice_per_image'] = np.std(image_dice)
        
        self.metrics = metrics
        
        print("\n" + "="*70)
        print(f"TEST RESULTS - Feature Config: {self.feature_config.upper()} ({self.expected_channels} channels)")
        print("="*70)
        for key, value in metrics.items():
            print(f"{key:25s}: {value:.4f}")
        print("="*70)
        
        return metrics
    
    def visualize_predictions(self, num_samples=10):
        """Visualize predictions vs ground truth"""
        print(f"\nGenerating visualizations for {num_samples} samples...")
        
        num_samples = min(num_samples, len(self.test_images))
        indices = np.random.choice(len(self.test_images), num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, i in enumerate(indices):
            # Original RGB image
            axes[idx, 0].imshow(self.test_images[i])
            axes[idx, 0].set_title(f'RGB Input\n{self.filenames[i]}')
            axes[idx, 0].axis('off')
            
            # Ground truth mask
            axes[idx, 1].imshow(self.test_masks[i].squeeze(), cmap='gray')
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            # Predicted mask
            axes[idx, 2].imshow(self.predictions_binary[i].squeeze(), cmap='gray')
            axes[idx, 2].set_title('Prediction')
            axes[idx, 2].axis('off')
            
            # Overlay
            overlay = self.test_images[i].copy() / 255.0
            mask_overlay = np.zeros_like(overlay)
            mask_overlay[:, :, 1] = self.predictions_binary[i].squeeze()  # Green for prediction
            mask_overlay[:, :, 0] = self.test_masks[i].squeeze()  # Red for ground truth
            
            axes[idx, 3].imshow(overlay * 0.7 + mask_overlay * 0.3)
            axes[idx, 3].set_title('Overlay (Red: GT, Green: Pred)')
            axes[idx, 3].axis('off')

            # Save Prediction
            save_path = os.path.join(self.results_dir, f'predictions_{self.feature_config}_{i}.png')
            plt.imsave(save_path, self.predictions_binary[i].squeeze())
        
        plt.suptitle(f'Predictions - {self.feature_config.upper()} ({self.expected_channels} channels)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, f'predictions_{self.feature_config}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
        plt.close()
    
    def visualize_feature_channels(self, sample_idx=0, save_prefix='features'):
        """Visualize extracted feature channels"""
        if self.feature_config == 'rgb':
            print("Skipping feature visualization for RGB baseline")
            return
        
        print(f"\nVisualizing feature channels for sample {sample_idx}...")
        
        features = self.test_features[sample_idx]
        n_channels = features.shape[-1]
        
        # Get feature names
        if self.feature_extractor:
            all_names = self.feature_extractor.feature_names
            if self.feature_config == 'luminance':
                feature_names = all_names[:8]
            elif self.feature_config == 'chrominance':
                feature_names = all_names[8:]
            else:
                feature_names = all_names
        else:
            feature_names = [f'Channel {i}' for i in range(n_channels)]
        
        # Create visualization
        n_cols = 6
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3))
        axes = axes.flatten()
        
        for i in range(n_channels):
            ax = axes[i]
            channel = features[:, :, i]
            
            im = ax.imshow(channel, cmap='viridis')
            ax.set_title(feature_names[i], fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide extra subplots
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Feature Channels - {self.feature_config.upper()} ({n_channels} channels)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, f'{save_prefix}_{self.feature_config}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved feature visualization to {save_path}")
        plt.close()
    
    def save_results(self):
        """Save test results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'model_path': self.model_path,
            'test_data_dir': self.test_data_dir,
            'feature_config': self.feature_config,
            'num_channels': self.expected_channels,
            'num_test_images': len(self.test_images),
            'metrics': self.metrics
        }
        
        results_path = os.path.join(self.results_dir, 
                                   f'test_results_{self.feature_config}_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(str(results), f, indent=4)
        
        print(f"\n✓ Results saved to {results_path}")
        return results_path


def main(model):
    """Main testing function"""
    import argparse
    sys.path.append('tests')
    sys.path.append('experiments')
    sys.path.append('data')
    sys.path.append('results')
    """Main testing function"""
    # Configuration
    MODEL_PATH = f'experiments/results/training/{model}/all/final_model.h5'  # Update with your model path
    TEST_DATA_DIR = 'data/processed/test'  # Update with your test data directory
    RESULTS_DIR = f'results/models/{model}'
    
    parser = argparse.ArgumentParser(description='Test river segmentation model with multi-channel support')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to trained model')
    parser.add_argument('--test_dir', type=str, default=TEST_DATA_DIR, help='Test data directory')
    parser.add_argument('--feature_config', type=str, default='all',
                       choices=['rgb', 'luminance', 'chrominance', 'all'],
                       help='Feature configuration (default: all)')
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR,
                       help='Results directory (default: test_results)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for inference (default: 8)')
    parser.add_argument('--visualize_features', action='store_true',
                       help='Visualize extracted features')
    
    args = parser.parse_args()
    
    # Initialize tester
    print("\n" + "="*70)
    print("MULTI-CHANNEL MODEL TESTING")
    print("="*70)
    
    tester = MultiChannelModelTester(
        model_path=args.model,
        test_data_dir=args.test_dir,
        feature_config=args.feature_config,
        results_dir=args.results_dir
    )
    
    # Load test data
    tester.load_test_data(img_size=(512, 512))
    
    # Visualize features if requested
    if args.visualize_features:
        tester.visualize_feature_channels(sample_idx=0)
    
    # Run inference
    tester.run_inference(batch_size=args.batch_size)
    
    # Calculate metrics
    tester.calculate_metrics()
    
    # Generate visualizations
    tester.visualize_predictions(num_samples=10)
    
    # Save results
    tester.save_results()
    
    print("\n✓ Testing completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main('unet')
    main('deeplabv3plus')
