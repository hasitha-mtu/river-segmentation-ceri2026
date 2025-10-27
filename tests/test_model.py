"""
River Segmentation Model Testing Script
Tests trained models on test dataset and generates evaluation metrics
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime
import sys
sys.path.append('src/models')
from unet import combined_loss, dice_coefficient, dice_loss, iou_metric

class RiverSegmentationTester:
    def __init__(self, model_path, test_data_dir, results_dir='test_results'):
        """
        Initialize the tester
        
        Args:
            model_path: Path to saved model (.h5 or SavedModel format)
            test_data_dir: Directory containing test images and masks
            results_dir: Directory to save test results
        """
        self.model_path = model_path
        self.test_data_dir = test_data_dir
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load model 
        print(f"Loading model from {model_path}")
        self.model = keras.models.load_model(model_path,
                                             custom_objects={'combined_loss': combined_loss,
                                                             'dice_coefficient': dice_coefficient,
                                                             'dice_loss': dice_loss,
                                                             'iou_metric': iou_metric})
        print("Model loaded successfully")
        
        self.test_images = []
        self.test_masks = []
        self.predictions = []
        
    def load_test_data(self, img_size=(512, 512)):
        """Load test images and masks"""
        print(f"Loading test data from {self.test_data_dir}")
        
        images_dir = os.path.join(self.test_data_dir, 'images')
        masks_dir = os.path.join(self.test_data_dir, 'masks')
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_file in image_files:
            # Load image
            img_path = os.path.join(images_dir, img_file)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, img_size)
            img = tf.cast(img, tf.float32) / 255.0
            
            # Load corresponding mask
            mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                mask = tf.io.read_file(mask_path)
                mask = tf.image.decode_image(mask, channels=1)
                mask = tf.image.resize(mask, img_size, method='nearest')
                mask = tf.cast(mask, tf.float32) / 255.0
                
                self.test_images.append(img.numpy())
                self.test_masks.append(mask.numpy())
        
        self.test_images = np.array(self.test_images)
        self.test_masks = np.array(self.test_masks)
        
        print(f"Loaded {len(self.test_images)} test images")
        return self.test_images, self.test_masks
    
    def run_inference(self, batch_size=8):
        """Run inference on test dataset"""
        print("Running inference on test dataset...")
        
        self.predictions = self.model.predict(self.test_images, batch_size=batch_size)
        
        # Threshold predictions to binary masks
        self.predictions_binary = (self.predictions > 0.5).astype(np.float32)
        
        print("Inference completed")
        return self.predictions, self.predictions_binary
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        print("Calculating metrics...")
        
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
        
        print("\n=== Test Results ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        return metrics
    
    def visualize_predictions(self, num_samples=10):
        """Visualize predictions vs ground truth"""
        print(f"Generating visualizations for {num_samples} samples...")
        
        num_samples = min(num_samples, len(self.test_images))
        indices = np.random.choice(len(self.test_images), num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, i in enumerate(indices):
            # Original image
            axes[idx, 0].imshow(self.test_images[i])
            axes[idx, 0].set_title('Input Image')
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
            overlay = self.test_images[i].copy()
            mask_overlay = np.zeros_like(overlay)
            mask_overlay[:, :, 1] = self.predictions_binary[i].squeeze()  # Green for prediction
            mask_overlay[:, :, 0] = self.test_masks[i].squeeze()  # Red for ground truth
            
            axes[idx, 3].imshow(overlay * 0.7 + mask_overlay * 0.3)
            axes[idx, 3].set_title('Overlay (Red: GT, Green: Pred)')
            axes[idx, 3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'predictions_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close()
    
    def save_results(self):
        """Save test results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'model_path': self.model_path,
            'test_data_dir': self.test_data_dir,
            'num_test_images': len(self.test_images),
            'metrics': self.metrics
        }
        
        results_path = os.path.join(self.results_dir, f'test_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {results_path}")
        return results_path
    
    def generate_error_analysis(self, threshold=0.5):
        """Generate error analysis for difficult cases"""
        print("Generating error analysis...")
        
        image_ious = []
        for i in range(len(self.test_masks)):
            true_mask = self.test_masks[i].flatten()
            pred_mask = self.predictions_binary[i].flatten()
            
            intersection = np.sum(true_mask * pred_mask)
            union = np.sum(true_mask) + np.sum(pred_mask) - intersection
            iou = intersection / union if union > 0 else 0
            image_ious.append((i, iou))
        
        # Sort by IoU (worst first)
        image_ious.sort(key=lambda x: x[1])
        
        # Visualize worst cases
        num_worst = min(5, len(image_ious))
        fig, axes = plt.subplots(num_worst, 3, figsize=(12, 4 * num_worst))
        
        if num_worst == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(num_worst):
            i, iou = image_ious[idx]
            
            axes[idx, 0].imshow(self.test_images[i])
            axes[idx, 0].set_title(f'Image (IoU: {iou:.3f})')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(self.test_masks[i].squeeze(), cmap='gray')
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(self.predictions_binary[i].squeeze(), cmap='gray')
            axes[idx, 2].set_title('Prediction')
            axes[idx, 2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'worst_predictions.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error analysis to {save_path}")
        plt.close()


def main(model_type):
    sys.path.append('tests')
    sys.path.append('experiments')
    sys.path.append('data')
    """Main testing function"""
    # Configuration
    MODEL_PATH = f'experiments/results/training/{model_type}/all/final_model.h5'  # Update with your model path
    TEST_DATA_DIR = 'data/processed/test'  # Update with your test data directory
    RESULTS_DIR = 'tests/results'
    
    # Initialize tester
    tester = RiverSegmentationTester(
        model_path=MODEL_PATH,
        test_data_dir=TEST_DATA_DIR,
        results_dir=RESULTS_DIR
    )
    
    # Load test data
    tester.load_test_data(img_size=(256, 256))
    
    # Run inference
    tester.run_inference(batch_size=8)
    
    # Calculate metrics
    tester.calculate_metrics()
    
    # Generate visualizations
    tester.visualize_predictions(num_samples=10)
    
    # Error analysis
    tester.generate_error_analysis()
    
    # Save results
    tester.save_results()
    
    print("\nTesting completed successfully!")


if __name__ == '__main__':
    main('unet')
    main('deeplabv3plus')

