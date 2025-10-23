import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from skimage.color import rgb2yuv # Using skimage for standard color space conversion
import cv2


def analyze_low_light_noise(image_path, img_size=512, window_size=7):
    """
    Simulates a low-light water image, converts it to YUV, calculates 
    Luminance and Chrominance noise maps, and plots the results.

    Args:
        img_size (int): The dimension for the square image (e.g., 512 for 512x512).
        window_size (int): The size of the local window for standard deviation calculation.
    """
    
    # --- 1. Create a Simulated Low-Light RGB Image ---
    np.random.seed(42)
    # rgb_img = np.zeros((img_size, img_size, 3), dtype=np.float32)

    rgb_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    rgb_img = rgb_img.astype(np.float32)/255.0

    # Create a base color gradient to simulate water absorption/depth variation
    x, y = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))
    # Simulate a base dark blue color
    # R_base = 0.05 + 0.05 * x  # Faint red gradient
    # G_base = 0.10 + 0.10 * y  # Faint green gradient
    B_base = 0.20 - 0.10 * x  # Darker blue, decreasing towards the right (absorption)

    # rgb_img[:, :, 0] = R_base
    # rgb_img[:, :, 1] = G_base
    rgb_img[:, :, 2] = B_base

    # --- 2. Add Non-uniform Noise (Simulating Sensor Noise Amplification) ---
    low_noise_std = 0.005
    high_noise_std = 0.05

    # General low noise across all channels
    rgb_img += np.random.normal(0, low_noise_std, rgb_img.shape)

    # Amplified noise: Blue channel noise is higher in the darker left half
    # (where the base B value is high, but overall light is low)
    # The left side (low x) is where B_base is higher, simulating a deep blue area 
    # that is poorly lit overall, thus suffering from high noise.
    noise_mask = (B_base < 0.15) # Example: Darker areas (right side) have less noise for contrast
    noise_amp = np.where(noise_mask, high_noise_std * 0.2, high_noise_std)
    blue_noise = np.random.normal(0, noise_amp, (img_size, img_size))

    rgb_img[:, :, 2] += blue_noise # Add amplified noise to Blue channel

    # Clip the values to a valid [0, 1] range
    rgb_img = np.clip(rgb_img, 0, 1)

    # --- 3. Convert RGB to YUV ---
    # Y: Luminance, U: Chrominance (Blue - Y), V: Chrominance (Red - Y)
    yuv_img = rgb2yuv(rgb_img)
    Y_channel = yuv_img[:, :, 0] 
    U_channel = yuv_img[:, :, 1]
    V_channel = yuv_img[:, :, 2]

    # --- 4. Define Noise Estimator ---
    def std_filter(arr):
        """Calculates the standard deviation of a local array."""
        return np.std(arr)

    # --- 5. Calculate Noise Maps ---
    chrominance_noise_map = generic_filter(U_channel, std_filter, size=window_size)
    luminance_noise_map = generic_filter(Y_channel, std_filter, size=window_size)

    # --- 6. Plot the Results ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Analysis of Noise in Simulated {img_size}x{img_size} Low-Light Image (Window Size: {window_size}\\times{window_size})', fontsize=18)

    # Row 1: The Input and Separated Channels
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('Simulated Low-Light RGB Water Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(Y_channel, cmap='gray')
    axes[0, 1].set_title('Luminance Channel (Y)')
    axes[0, 1].axis('off')

    # Chrominance (U) channel uses a divergent colormap (e.g., 'bwr')
    im_u = axes[0, 2].imshow(U_channel, cmap='bwr', vmin=U_channel.min(), vmax=U_channel.max())
    axes[0, 2].set_title('Chrominance Channel (U)')
    axes[0, 2].axis('off')
    
    # Row 2: The Noise Maps
    axes[1, 0].remove() # Remove this empty subplot to give space to the others

    # Luminance Noise Map
    im_y_noise = axes[1, 1].imshow(luminance_noise_map, cmap='hot', interpolation='nearest')
    axes[1, 1].set_title('Luminance (Y) Noise Map (Lower $\\sigma$)')
    axes[1, 1].axis('off')
    cbar_y = fig.colorbar(im_y_noise, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar_y.set_label('Local $\\sigma$')

    # Chrominance Noise Map (The critical result)
    im_u_noise = axes[1, 2].imshow(chrominance_noise_map, cmap='hot', interpolation='nearest')
    axes[1, 2].set_title('Chrominance (U) Noise Map (Higher $\\sigma$)')
    axes[1, 2].axis('off')
    cbar_u = fig.colorbar(im_u_noise, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar_u.set_label('Local $\\sigma$')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() # Use plt.show() when running locally

def plot_image(image_path):
    rgb_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    rgb_img = rgb_img.astype(np.float32)/255.0

    # # Create a faint base color (e.g., a dark blue-green area)
    # rgb_img[:, :, 0] = 0.1  # Red (R)
    # rgb_img[:, :, 1] = 0.15 # Green (G)
    # rgb_img[:, :, 2] = 0.2  # Blue (B)

    # Add non-uniform Gaussian noise, simulating low-light sensor noise.
    high_noise_region = slice(20, 220)
    low_noise_std = 0.01
    high_noise_std = 0.08 

    # Add base noise to all channels
    rgb_img += np.random.normal(0, low_noise_std, rgb_img.shape)

    # Add amplified noise specifically to the Blue channel in the center region
    rgb_img[high_noise_region, high_noise_region, 2] += np.random.normal(0, high_noise_std, (200, 200))

    # Clip the values to a valid [0, 1] range
    rgb_img = np.clip(rgb_img, 0, 1)

    # 2. Convert RGB to YUV (Luminance and Chrominance separation)
    yuv_img = rgb2yuv(rgb_img)
    Y_channel = yuv_img[:, :, 0] # Luminance
    U_channel = yuv_img[:, :, 1] # Chrominance (Blue - Y)
    V_channel = yuv_img[:, :, 2] # Chrominance (Red - Y)

    # 3. Define the Local Standard Deviation Filter (Noise estimator)
    def std_filter(arr):
        return np.std(arr)

    # 4. Calculate the Noise Map for a Chrominance channel (U-channel selected here)
    window_size = 7
    chrominance_noise_map = generic_filter(U_channel, std_filter, size=window_size)

    # 5. Plot the results 
    # ----------------------------------------------------------------------
    # START OF PLOTTING CODE
    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Analysis of Noise in a Simulated Low-Light RGB Image (via YUV Conversion)', fontsize=16)

    # Plot 1: Luminance Channel (Y)
    axes[0].imshow(Y_channel, cmap='gray')
    axes[0].set_title('Luminance Channel (Y)')
    axes[0].axis('off')

    # Plot 2: Chrominance Channel (U)
    # U/V channels have a mean near 0, so using 'bwr' is suitable
    im_u = axes[1].imshow(U_channel, cmap='bwr', interpolation='nearest')
    axes[1].set_title('Chrominance Channel (U)')
    axes[1].axis('off')

    # Plot 3: Chrominance Noise Map (for U channel)
    # Using 'hot' cmap emphasizes high noise regions (red/yellow)
    im_noise = axes[2].imshow(chrominance_noise_map, cmap='hot', interpolation='nearest')
    axes[2].set_title(f'U-Channel Noise Map (Local $\\sigma$, {window_size}\\times{window_size})')
    axes[2].axis('off')

    # Add a color bar for the noise map scale
    cbar = fig.colorbar(im_noise, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Local Noise Magnitude ($\\sigma$)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig('results/feature_importance/feature_rgb_noise_analysis.png')
    plt.show()
    # ----------------------------------------------------------------------
    # END OF PLOTTING CODE
    # ----------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    sys.path.append('src/data')
    sys.path.append('experiments')
    # 1. Simulate a low-light RGB image (100x100 pixels)
    np.random.seed(42)
    img_size = 512
    image_path = 'data\\raw\images\DJI_20250324094544_0012_V_March.png'    
    # rgb_img = np.zeros((img_size, img_size, 3))
    analyze_low_light_noise(image_path)
    plot_image(image_path)

