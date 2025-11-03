"""
U-Net Implementation Debugger
Identifies issues causing complete failure (~0.076 Dice)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pathlib import Path

def inspect_model_architecture(model):
    """Comprehensive model architecture inspection"""
    print("\n" + "="*80)
    print("U-NET ARCHITECTURE INSPECTION")
    print("="*80)
    
    print("\n1. MODEL SUMMARY:")
    print("-" * 60)
    model.summary()
    
    print("\n2. INPUT/OUTPUT SHAPES:")
    print("-" * 60)
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    # Check output layer
    output_layer = model.layers[-1]
    print(f"\n3. OUTPUT LAYER DETAILS:")
    print("-" * 60)
    print(f"  Layer type:   {type(output_layer).__name__}")
    print(f"  Layer config: {output_layer.get_config()}")
    
    # Check activation
    if hasattr(output_layer, 'activation'):
        print(f"  Activation:   {output_layer.activation.__name__}")
    
    # Expected for binary segmentation
    print("\n4. EXPECTED CONFIGURATION (Binary Segmentation):")
    print("-" * 60)
    print("  ✓ Output shape:  (None, H, W, 1)  ← Single channel")
    print("  ✓ Activation:    sigmoid          ← Binary probabilities [0,1]")
    print("  ✓ Loss:          binary_crossentropy or dice_loss")
    
    # Check if configuration matches
    issues = []
    
    output_channels = model.output_shape[-1]
    if output_channels != 1:
        issues.append(f"Output channels: {output_channels} (should be 1)")
    
    if hasattr(output_layer, 'activation'):
        if output_layer.activation.__name__ not in ['sigmoid', 'linear']:
            issues.append(f"Activation: {output_layer.activation.__name__} (should be sigmoid)")
    
    if issues:
        print("\n⚠️  POTENTIAL ISSUES DETECTED:")
        for issue in issues:
            print(f"   ❌ {issue}")
    else:
        print("\n✓ Architecture looks correct")
    
    return issues

def test_forward_pass(model, input_channels=3):
    """Test model forward pass with dummy data"""
    print("\n" + "="*80)
    print("FORWARD PASS TEST")
    print("="*80)
    
    # Create dummy input
    batch_size = 2
    height, width = 512, 512
    
    print(f"\n1. Creating dummy input: ({batch_size}, {height}, {width}, {input_channels})")
    dummy_input = np.random.rand(batch_size, height, width, input_channels).astype(np.float32)
    
    print(f"  Input range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
    
    # Forward pass
    print("\n2. Running forward pass...")
    try:
        predictions = model.predict(dummy_input, verbose=0)
        print("  ✓ Forward pass successful")
        
        print(f"\n3. PREDICTION ANALYSIS:")
        print("-" * 60)
        print(f"  Shape:  {predictions.shape}")
        print(f"  Range:  [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"  Mean:   {predictions.mean():.6f}")
        print(f"  Std:    {predictions.std():.6f}")
        
        # Check for common issues
        issues = []
        
        if predictions.shape[-1] != 1:
            issues.append(f"Output has {predictions.shape[-1]} channels (should be 1)")
        
        if predictions.min() < -0.1 or predictions.max() > 1.1:
            issues.append(f"Output range [{predictions.min():.3f}, {predictions.max():.3f}] outside [0,1]")
        
        if np.abs(predictions.mean() - 0.5) > 0.4:
            issues.append(f"Mean {predictions.mean():.3f} far from 0.5 (untrained model should be ~0.5)")
        
        if np.isnan(predictions).any():
            issues.append("Contains NaN values!")
        
        if np.isinf(predictions).any():
            issues.append("Contains infinite values!")
        
        if issues:
            print("\n⚠️  ISSUES DETECTED:")
            for issue in issues:
                print(f"   ❌ {issue}")
            return False
        else:
            print("\n✓ Predictions look reasonable")
            return True
            
    except Exception as e:
        print(f"  ❌ Forward pass failed: {str(e)}")
        return False

def test_loss_computation(model, input_channels=3):
    """Test loss computation with dummy data"""
    print("\n" + "="*80)
    print("LOSS COMPUTATION TEST")
    print("="*80)
    
    # Create dummy data
    batch_size = 4
    height, width = 512, 512
    
    X_dummy = np.random.rand(batch_size, height, width, input_channels).astype(np.float32)
    y_dummy = np.random.randint(0, 2, (batch_size, height, width, 1)).astype(np.float32)
    
    print(f"\n1. Dummy data created:")
    print(f"  X shape: {X_dummy.shape}, range: [{X_dummy.min():.3f}, {X_dummy.max():.3f}]")
    print(f"  y shape: {y_dummy.shape}, range: [{y_dummy.min():.0f}, {y_dummy.max():.0f}]")
    print(f"  Positive ratio: {y_dummy.mean():.3f}")
    
    # Get model's loss function
    loss_fn = model.loss
    print(f"\n2. Loss function: {loss_fn}")
    
    # Forward pass
    print("\n3. Computing loss...")
    try:
        predictions = model.predict(X_dummy, verbose=0)
        
        # Compute loss
        if isinstance(loss_fn, str):
            loss_fn = keras.losses.get(loss_fn)
        
        loss_value = loss_fn(y_dummy, predictions).numpy()
        
        print(f"  Loss value: {loss_value:.6f}")
        
        # Interpret loss value
        print("\n4. LOSS INTERPRETATION:")
        print("-" * 60)
        
        if 'binary_crossentropy' in str(loss_fn).lower():
            random_loss = -np.log(0.5)  # ≈ 0.693
            print(f"  Binary cross-entropy detected")
            print(f"  Random prediction loss: ~{random_loss:.3f}")
            
            if abs(loss_value - random_loss) < 0.1:
                print(f"  ⚠️  Loss ≈ {random_loss:.3f} suggests model is predicting randomly!")
            elif loss_value > 1.0:
                print(f"  ⚠️  Loss > 1.0 suggests poor predictions")
            else:
                print(f"  ✓ Loss seems reasonable for untrained model")
        
        # Check gradient flow
        print("\n5. GRADIENT FLOW TEST:")
        print("-" * 60)
        
        with tf.GradientTape() as tape:
            preds = model(X_dummy, training=True)
            loss = loss_fn(y_dummy, preds)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Check if gradients exist and are reasonable
        grad_norms = [tf.norm(g).numpy() if g is not None else 0 for g in gradients]
        
        print(f"  Number of trainable variables: {len(model.trainable_variables)}")
        print(f"  Number with gradients: {sum(1 for g in gradients if g is not None)}")
        print(f"  Mean gradient norm: {np.mean(grad_norms):.6f}")
        print(f"  Max gradient norm: {np.max(grad_norms):.6f}")
        
        if np.mean(grad_norms) < 1e-8:
            print("  ⚠️  Vanishing gradients detected!")
        elif np.max(grad_norms) > 100:
            print("  ⚠️  Exploding gradients detected!")
        else:
            print("  ✓ Gradients look reasonable")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Loss computation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_thresholding(model, input_channels=3):
    """Test if thresholding is done correctly"""
    print("\n" + "="*80)
    print("PREDICTION THRESHOLDING TEST")
    print("="*80)
    
    # Create dummy input
    X_dummy = np.random.rand(1, 512, 512, input_channels).astype(np.float32)
    
    # Get raw predictions
    pred_raw = model.predict(X_dummy, verbose=0)
    
    print(f"\n1. Raw predictions:")
    print(f"  Shape: {pred_raw.shape}")
    print(f"  Range: [{pred_raw.min():.6f}, {pred_raw.max():.6f}]")
    print(f"  Mean: {pred_raw.mean():.6f}")
    
    # Apply thresholding
    pred_binary = (pred_raw > 0.5).astype(np.float32)
    
    print(f"\n2. After thresholding (>0.5):")
    print(f"  Unique values: {np.unique(pred_binary)}")
    print(f"  Positive ratio: {pred_binary.mean():.3f}")
    
    # Common mistake: double thresholding
    pred_double_threshold = (pred_binary > 0.5).astype(np.float32)
    
    print(f"\n3. After DOUBLE thresholding (wrong!):")
    print(f"  Unique values: {np.unique(pred_double_threshold)}")
    print(f"  Positive ratio: {pred_double_threshold.mean():.3f}")
    
    print("\n4. GUIDANCE:")
    print("-" * 60)
    print("  ✓ Correct: pred_binary = (pred > 0.5).astype(float)")
    print("  ✗ Wrong:   pred_binary = ((pred > 0.5) > 0.5).astype(float)")
    print("  ✗ Wrong:   If pred is already binary, don't threshold again")

def diagnose_dice_score(y_true, y_pred):
    """Diagnose why Dice score might be ~0.076"""
    print("\n" + "="*80)
    print("DICE SCORE DIAGNOSTIC")
    print("="*80)
    
    # Ensure binary
    y_true_binary = (y_true > 0.5).astype(np.float32)
    y_pred_binary = (y_pred > 0.5).astype(np.float32)
    
    # Compute Dice components
    intersection = np.sum(y_true_binary * y_pred_binary)
    sum_true = np.sum(y_true_binary)
    sum_pred = np.sum(y_pred_binary)
    
    dice = (2 * intersection + 1e-7) / (sum_true + sum_pred + 1e-7)
    
    print(f"\n1. Dice computation:")
    print(f"  Intersection: {intersection:.0f}")
    print(f"  Sum(y_true):  {sum_true:.0f}")
    print(f"  Sum(y_pred):  {sum_pred:.0f}")
    print(f"  Dice score:   {dice:.6f}")
    
    print(f"\n2. Analysis:")
    print(f"  Ground truth positive ratio: {y_true_binary.mean():.3f}")
    print(f"  Prediction positive ratio:   {y_pred_binary.mean():.3f}")
    
    # Check for common issues
    if dice < 0.1:
        print("\n⚠️  EXTREMELY LOW DICE SCORE (<0.1)")
        
        if sum_pred == 0:
            print("  ❌ Model predicts ALL NEGATIVE (no water detected)")
            print("     → Check class imbalance in training")
            print("     → Try weighted loss function")
            
        elif sum_pred == y_pred_binary.size:
            print("  ❌ Model predicts ALL POSITIVE (everything is water)")
            print("     → Model is stuck predicting one class")
            print("     → Check loss function and learning rate")
            
        elif sum_true == 0:
            print("  ❌ Ground truth has NO POSITIVE samples")
            print("     → Data loading issue!")
            
        else:
            print("  ❌ Very poor overlap between prediction and ground truth")
            print("     → Model is predicting but in wrong locations")
            print("     → Check data preprocessing and augmentation")

def create_test_unet(input_channels=3):
    """Create a simple test U-Net for verification"""
    print("\n" + "="*80)
    print("CREATING TEST U-NET")
    print("="*80)
    
    inputs = keras.Input(shape=(512, 512, input_channels))
    
    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    # Bottleneck
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    
    # Decoder
    u1 = layers.UpSampling2D(2)(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)
    
    u2 = layers.UpSampling2D(2)(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(c5)
    
    # Output - CRITICAL: 1 channel with sigmoid
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    print("  ✓ Test U-Net created")
    print(f"  Input:  {model.input_shape}")
    print(f"  Output: {model.output_shape}")
    
    return model

def compare_with_working_config():
    """Compare current config with known working configuration"""
    print("\n" + "="*80)
    print("REFERENCE: KNOWN WORKING CONFIGURATION")
    print("="*80)
    
    print("\n1. ARCHITECTURE:")
    print("-" * 60)
    print("  Input:  (None, 512, 512, 3)")
    print("  Output: (None, 512, 512, 1)  ← MUST be 1 channel")
    print("  Final activation: sigmoid    ← MUST be sigmoid for binary")
    
    print("\n2. COMPILATION:")
    print("-" * 60)
    print("  Loss: 'binary_crossentropy' or dice_loss")
    print("  Optimizer: Adam(lr=1e-4)")
    print("  Metrics: ['accuracy', dice_coefficient, iou_score]")
    
    print("\n3. TRAINING:")
    print("-" * 60)
    print("  Batch size: 4-8")
    print("  Epochs: 100")
    print("  Input range: [0, 1]  ← Normalize images")
    print("  Mask range: {0, 1}   ← Binary masks")
    
    print("\n4. EVALUATION:")
    print("-" * 60)
    print("  pred = model.predict(X_test)")
    print("  pred_binary = (pred > 0.5).astype(float)")
    print("  dice = compute_dice(y_test, pred_binary)")
    
    print("\n5. COMMON MISTAKES:")
    print("-" * 60)
    print("  ❌ Output channels = 2 (using softmax)")
    print("  ❌ Forgetting to normalize input images")
    print("  ❌ Using categorical_crossentropy instead of binary")
    print("  ❌ Double thresholding predictions")
    print("  ❌ Channel mismatch in skip connections")

def main():
    """Main diagnostic pipeline"""
    print("\n" + "="*80)
    print("U-NET DEBUGGING PIPELINE")
    print("="*80)
    
    # Configuration
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        input_channels = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    else:
        print("\nOptions:")
        print("  1. Load existing model for diagnosis")
        print("  2. Create test model and verify")
        
        choice = input("\nChoice (1/2): ").strip()
        
        if choice == '1':
            model_path = input("  Model path (.h5 or .keras): ").strip()
            input_channels = int(input("  Input channels (3/8/18): ").strip() or "3")
        else:
            model_path = None
            input_channels = int(input("  Input channels (3/8/18): ").strip() or "3")
    
    if model_path:
        print(f"\nLoading model from: {model_path}")
        try:
            model = keras.models.load_model(model_path, compile=False)
            print("  ✓ Model loaded")
        except Exception as e:
            print(f"  ❌ Failed to load model: {str(e)}")
            return
    else:
        print(f"\nCreating test U-Net with {input_channels} input channels...")
        model = create_test_unet(input_channels)
    
    # Run diagnostics
    print("\n" + "="*80)
    print("RUNNING DIAGNOSTICS")
    print("="*80)
    
    arch_issues = inspect_model_architecture(model)
    forward_ok = test_forward_pass(model, input_channels)
    loss_ok = test_loss_computation(model, input_channels)
    test_prediction_thresholding(model, input_channels)
    
    # Test with dummy data
    print("\n" + "="*80)
    print("DUMMY DATA TEST")
    print("="*80)
    
    X_test = np.random.rand(2, 512, 512, input_channels).astype(np.float32)
    y_test = np.random.randint(0, 2, (2, 512, 512, 1)).astype(np.float32)
    
    pred = model.predict(X_test, verbose=0)
    diagnose_dice_score(y_test, pred)
    
    # Show reference configuration
    compare_with_working_config()
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    if arch_issues:
        print("\n❌ ARCHITECTURE ISSUES FOUND:")
        for issue in arch_issues:
            print(f"   - {issue}")
    else:
        print("\n✓ Architecture looks correct")
    
    if not forward_ok:
        print("\n❌ FORWARD PASS ISSUES")
    else:
        print("\n✓ Forward pass works")
    
    if not loss_ok:
        print("\n❌ LOSS COMPUTATION ISSUES")
    else:
        print("\n✓ Loss computation works")
    
    print("\n" + "="*80)
    print("RECOMMENDED ACTIONS")
    print("="*80)
    
    if arch_issues:
        print("\n1. FIX ARCHITECTURE:")
        print("   - Ensure output layer has 1 channel")
        print("   - Use sigmoid activation for binary segmentation")
        print("   - Check skip connection dimensions")
    
    print("\n2. VERIFY DATA PIPELINE:")
    print("   - Check if images are normalized [0, 1]")
    print("   - Verify masks are binary {0, 1}")
    print("   - Ensure no data leakage or wrong splits")
    
    print("\n3. CHECK TRAINING CONFIGURATION:")
    print("   - Use binary_crossentropy or dice_loss")
    print("   - Try lower learning rate (1e-4)")
    print("   - Increase batch size if memory allows")
    
    print("\n4. MONITOR TRAINING:")
    print("   - Plot loss curve (should decrease)")
    print("   - Plot Dice curve (should increase)")
    print("   - Visualize predictions every few epochs")
    
    print("\n5. DEBUG SPECIFIC TO 0.076 DICE:")
    print("   If Dice ≈ 0.076 specifically:")
    print("   - Check if model predicts all zeros or all ones")
    print("   - Verify class balance in dataset")
    print("   - Try class weights or focal loss")
    print("   - Check if masks are inverted (0=water, 1=background)")

if __name__ == '__main__':
    main()
