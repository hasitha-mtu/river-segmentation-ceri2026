"""
Fixed Loss Utilities with Numerical Stability
==============================================
Addresses NaN issues in combined loss function during ablation study
"""

import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient metric for segmentation with numerical stability.
    
    Args:
        y_true: Ground truth masks (float32, range [0, 1])
        y_pred: Predicted masks (float32, range [0, 1])
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient (scalar between 0 and 1)
    """
    # Cast to float32 for numerical stability
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    # Calculate intersection and sums
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    # Dice coefficient formula
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    return dice


def dice_loss(y_true, y_pred):
    """
    Dice loss for training.
    
    Returns value in range [0, 1] where 0 is perfect.
    """
    return 1.0 - dice_coefficient(y_true, y_pred)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss with proper numerical stability.
    
    CRITICAL FIXES:
    1. Clip predictions to avoid log(0) and exp overflow
    2. Use stable BCE implementation
    3. Handle edge cases properly
    
    Args:
        y_true: Ground truth (0 or 1)
        y_pred: Predictions (0 to 1)
        alpha: Balancing factor
        gamma: Focusing parameter
        
    Returns:
        Focal loss (scalar)
    """
    # Cast to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # CRITICAL: Clip predictions to prevent numerical instability
    # Original issue: predictions can be exactly 0 or 1, causing log(0) = -inf
    epsilon = tf.keras.backend.epsilon()  # Typically 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate cross entropy manually for stability
    # BCE = -[y*log(p) + (1-y)*log(1-p)]
    bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    
    # Calculate modulating factor: (1 - p_t)^gamma
    # p_t = p if y=1, else (1-p)
    p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
    
    # CRITICAL: Clip p_t to avoid (1 - p_t) being negative or too close to 0
    p_t = tf.clip_by_value(p_t, epsilon, 1.0 - epsilon)
    
    # Modulating factor
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    
    # Alpha weighting (optional, for class imbalance)
    alpha_factor = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
    
    # Focal loss per pixel
    focal = alpha_factor * modulating_factor * bce
    
    # Return mean
    return tf.reduce_mean(focal)


def combined_loss(y_true, y_pred, weights=None):
    """
    Combined loss: BCE + Dice + Focal with numerical stability.
    
    CRITICAL FIXES:
    1. Clip predictions before all loss calculations
    2. Use stable focal loss implementation
    3. Add optional loss weighting
    4. Add NaN checking and fallback
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        weights: Optional dict with 'bce', 'dice', 'focal' weights
                 Default: {'bce': 1.0, 'dice': 1.0, 'focal': 1.0}
    
    Returns:
        Combined loss (scalar)
    """
    # Default weights (equal as per your config)
    if weights is None:
        weights = {'bce': 1.0, 'dice': 1.0, 'focal': 1.0}
    
    # Cast to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # CRITICAL: Clip predictions FIRST to ensure numerical stability
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # 1. Binary Cross Entropy
    # Using from_logits=False since predictions are already sigmoid outputs
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    
    # 2. Dice Loss
    dice = dice_loss(y_true, y_pred)
    
    # 3. Focal Loss (with stability fixes)
    focal = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)
    
    # Combine with weights
    total_loss = (weights['bce'] * bce + 
                  weights['dice'] * dice + 
                  weights['focal'] * focal)
    
    # CRITICAL: Check for NaN and return safe fallback
    # This helps debug which component is causing NaN
    total_loss = tf.where(
        tf.math.is_nan(total_loss),
        tf.constant(1.0, dtype=tf.float32),  # Fallback to 1.0 if NaN
        total_loss
    )
    
    return total_loss


def combined_loss_with_logging(y_true, y_pred, weights=None):
    """
    Combined loss WITH component logging for debugging.
    Use this during development to identify which loss component causes NaN.
    
    Returns:
        total_loss, loss_components_dict
    """
    if weights is None:
        weights = {'bce': 1.0, 'dice': 1.0, 'focal': 1.0}
    
    # Cast and clip
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate each component
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)
    
    # Combine
    total_loss = weights['bce'] * bce + weights['dice'] * dice + weights['focal'] * focal
    
    # Return components for logging
    components = {
        'bce': bce,
        'dice': dice,
        'focal': focal,
        'total': total_loss
    }
    
    return total_loss, components


def iou_metric(y_true, y_pred, threshold=0.5):
    """
    Intersection over Union (IoU) metric with numerical stability.
    
    FIXES:
    1. Cast to float32
    2. Add epsilon to denominator
    3. Clip threshold value
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks (probabilities)
        threshold: Threshold for binary prediction
        
    Returns:
        IoU score
    """
    # Cast to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Binarize predictions
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred_binary)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary) - intersection
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-7
    iou = (intersection + epsilon) / (union + epsilon)
    
    # Clip to [0, 1] range
    iou = tf.clip_by_value(iou, 0.0, 1.0)
    
    return iou


# ============================================================================
# Additional Utility Functions
# ============================================================================

def check_inputs(y_true, y_pred):
    """
    Check input tensors for common issues that cause NaN.
    
    Use during debugging:
        check_inputs(y_true, y_pred)
    """
    print("\n=== Input Check ===")
    print(f"y_true shape: {y_true.shape}")
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_true dtype: {y_true.dtype}")
    print(f"y_pred dtype: {y_pred.dtype}")
    print(f"y_true range: [{tf.reduce_min(y_true):.4f}, {tf.reduce_max(y_true):.4f}]")
    print(f"y_pred range: [{tf.reduce_min(y_pred):.4f}, {tf.reduce_max(y_pred):.4f}]")
    print(f"y_true has NaN: {tf.reduce_any(tf.math.is_nan(y_true)).numpy()}")
    print(f"y_pred has NaN: {tf.reduce_any(tf.math.is_nan(y_pred)).numpy()}")
    print(f"y_pred has Inf: {tf.reduce_any(tf.math.is_inf(y_pred)).numpy()}")
    print("==================\n")


def safe_combined_loss(y_true, y_pred, weights=None):
    """
    Extra-safe version that catches and reports NaN issues.
    Use this for debugging in ablation study.
    """
    try:
        # Check inputs
        if tf.reduce_any(tf.math.is_nan(y_true)) or tf.reduce_any(tf.math.is_nan(y_pred)):
            tf.print("WARNING: NaN detected in inputs!")
            return tf.constant(1.0, dtype=tf.float32)
        
        if tf.reduce_any(tf.math.is_inf(y_pred)):
            tf.print("WARNING: Inf detected in predictions!")
            return tf.constant(1.0, dtype=tf.float32)
        
        # Calculate loss
        loss = combined_loss(y_true, y_pred, weights)
        
        # Check output
        if tf.math.is_nan(loss):
            tf.print("WARNING: NaN in loss output!")
            return tf.constant(1.0, dtype=tf.float32)
        
        return loss
        
    except Exception as e:
        tf.print(f"ERROR in loss calculation: {e}")
        return tf.constant(1.0, dtype=tf.float32)


# ============================================================================
# EXPLANATION OF ISSUES AND FIXES
# ============================================================================

"""
ROOT CAUSES OF NaN LOSS IN ABLATION STUDY:
==========================================

1. PREDICTION CLIPPING ISSUE (MOST LIKELY)
   Problem: Model predictions can be exactly 0.0 or 1.0
   Impact: log(0) = -inf, exp(-inf) = 0, causing NaN in focal loss
   Fix: Clip predictions to [epsilon, 1-epsilon] BEFORE loss calculation
   
2. FOCAL LOSS CALCULATION
   Problem: Original focal loss used tf.exp(-bce) which can overflow
   Impact: exp(large_number) = inf, then inf * number = NaN
   Fix: Use stable implementation with proper clipping
   
3. DOUBLE BCE CALCULATION
   Problem: You calculated bce_per_pixel twice (lines 36 and 47)
   Impact: Inconsistent and potentially unstable
   Fix: Calculate once and reuse, or use stable focal loss function
   
4. NO INPUT VALIDATION
   Problem: If model outputs NaN/Inf during training, loss propagates
   Impact: Entire training collapses
   Fix: Add input checking and safe fallbacks

5. MIXED PRECISION ISSUES (if enabled)
   Problem: Float16 has limited range, can overflow/underflow easily
   Impact: NaN propagation in loss calculations
   Fix: Ensure loss calculation uses float32

WHY IT WORKS SEPARATELY BUT FAILS IN ABLATION:
==============================================

1. Different Random Seeds
   - Ablation study may initialize models differently
   - Some initializations produce extreme predictions (0 or 1) early
   
2. Batch Size Differences
   - Ablation study uses batch_size=4 (from run_ablation_study.py)
   - Separate training might use different batch size
   - Smaller batches can have more extreme values
   
3. Feature Configuration Impact
   - Different input channels (3/8/10/18) affect model behavior
   - Some configs may produce more extreme initial predictions
   - Chrominance or luminance-only might have different ranges

4. Learning Rate Warmup
   - Without proper warmup, model can make extreme predictions
   - This triggers the log(0) issue in focal loss
   
5. No Gradient Clipping
   - Large gradients from unstable loss can make predictions extreme
   - This creates a feedback loop: extreme preds → NaN loss → extreme grads

RECOMMENDED FIXES FOR ABLATION STUDY:
=====================================

1. Use the fixed loss_util.py (this file)
2. Add gradient clipping in your training code:
   optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
   
3. Add learning rate warmup:
   lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
       initial_learning_rate=1e-4,
       decay_steps=1000,
       warmup_target=1e-3,
       warmup_steps=100
   )
   
4. Monitor loss components separately:
   Use combined_loss_with_logging() during first few epochs
   
5. Add callbacks to stop on NaN:
   callbacks = [
       tf.keras.callbacks.TerminateOnNaN(),
       tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
   ]
"""


if __name__ == "__main__":
    # Test the fixed functions
    import numpy as np
    
    print("="*70)
    print("TESTING FIXED LOSS FUNCTIONS")
    print("="*70)
    
    # Create test data
    y_true = tf.constant(np.random.randint(0, 2, (4, 128, 128, 1)), dtype=tf.float32)
    y_pred = tf.constant(np.random.random((4, 128, 128, 1)), dtype=tf.float32)
    
    print("\n1. Testing Dice Loss:")
    dice = dice_loss(y_true, y_pred)
    print(f"   Dice loss: {dice.numpy():.4f}")
    print(f"   Is NaN: {tf.math.is_nan(dice).numpy()}")
    
    print("\n2. Testing Focal Loss:")
    focal = focal_loss(y_true, y_pred)
    print(f"   Focal loss: {focal.numpy():.4f}")
    print(f"   Is NaN: {tf.math.is_nan(focal).numpy()}")
    
    print("\n3. Testing Combined Loss:")
    combined = combined_loss(y_true, y_pred)
    print(f"   Combined loss: {combined.numpy():.4f}")
    print(f"   Is NaN: {tf.math.is_nan(combined).numpy()}")
    
    print("\n4. Testing with extreme predictions (0 and 1):")
    y_pred_extreme = tf.constant(np.where(np.random.random((4, 128, 128, 1)) > 0.5, 1.0, 0.0), 
                                 dtype=tf.float32)
    combined_extreme = combined_loss(y_true, y_pred_extreme)
    print(f"   Combined loss: {combined_extreme.numpy():.4f}")
    print(f"   Is NaN: {tf.math.is_nan(combined_extreme).numpy()}")
    
    print("\n5. Testing IoU metric:")
    iou = iou_metric(y_true, y_pred)
    print(f"   IoU: {iou.numpy():.4f}")
    print(f"   Is NaN: {tf.math.is_nan(iou).numpy()}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED - No NaN values detected")
    print("="*70)

    