import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_ndwi_with_edge_detection(image_path):
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Extract Green and Blue channels
    G = rgb_image[:, :, 1].astype(np.float32)  # Green channel
    B = rgb_image[:, :, 2].astype(np.float32)  # Blue channel

    # Compute NDWI
    ndwi = (G - B) / (G + B + 1e-5)

    # Normalize NDWI
    ndwi_normalized = (ndwi - np.min(ndwi)) / (np.max(ndwi) - np.min(ndwi))

    # Apply thresholding to get a water mask
    threshold = 0.05  # Adjust based on your image
    water_mask = (ndwi > threshold).astype(np.uint8) * 255

    # Apply Edge Detection (Sobel)
    sobelx = cv2.Sobel(water_mask, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
    sobely = cv2.Sobel(water_mask, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges
    sobel_edges = cv2.magnitude(sobelx, sobely)  # Compute gradient magnitude
    sobel_edges = (sobel_edges / np.max(sobel_edges) * 255).astype(np.uint8)  # Normalize

    # Apply Edge Detection (Canny)
    canny_edges = cv2.Canny(water_mask, 50, 150)  # Adjust thresholds as needed

    # Display Results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(ndwi_normalized, cmap="jet")
    axes[1].set_title("NDWI Map")
    axes[1].axis("off")

    axes[2].imshow(sobel_edges, cmap="gray")
    axes[2].set_title("Sobel Edge Detection")
    axes[2].axis("off")

    axes[3].imshow(canny_edges, cmap="gray")
    axes[3].set_title("Canny Edge Detection")
    axes[3].axis("off")

    plt.show()


if __name__ == "__main__":
    import sys
    sys.path.append('src/data')
    pass

