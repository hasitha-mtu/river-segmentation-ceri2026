import cv2
import numpy as np
from PIL import Image
from skimage import io
from scipy import ndimage


def morphological_closing(mask_path, output_path):
    # Load the image
    img = cv2.imread(mask_path, 0)

    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # Morphological closing (dilation followed by erosion)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Save result
    cv2.imwrite(f'{output_path}/morphological_closing.png', closed)

def flood_fill(mask_path, output_path):
    img = cv2.imread(mask_path, 0)
    img_filled = img.copy()

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill all contours
    cv2.drawContours(img_filled, contours, -1, 255, thickness=cv2.FILLED)

    cv2.imwrite(f'{output_path}/flood_fill.png', img_filled)

def convex_hull(mask_path, output_path):
    img = cv2.imread(mask_path, 0)
    output = np.zeros_like(img)

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get convex hull
        hull = cv2.convexHull(contour)
        cv2.drawContours(output, [hull], -1, 255, thickness=cv2.FILLED)

    cv2.imwrite(f'{output_path}/convex_hull.png', output)

def binary_dilation_and_erosion(mask_path, output_path):
    img = cv2.imread(mask_path, 0)

    # Dilate to close gaps
    kernel = np.ones((20, 20), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=2)

    # Erode to restore approximate original size
    eroded = cv2.erode(dilated, kernel, iterations=2)

    cv2.imwrite(f'{output_path}/binary_dilation_and_erosion.png', eroded)

def skimage_binary_fill_holes(mask_path, output_path):
    img = io.imread(mask_path, as_gray=True) > 0.5

    # Fill holes in binary image
    filled = ndimage.binary_fill_holes(img)

    io.imsave(f'{output_path}/skimage_binary_fill_holes.png', (filled * 255).astype('uint8'))

def combining_method(mask_path, output_path):
    img = cv2.imread(mask_path, 0)

    # Step 1: Morphological closing for small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Step 2: Fill holes
    filled = ndimage.binary_fill_holes(closed > 127).astype(np.uint8) * 255

    # Step 3: Optional - smooth edges
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_smooth)

    cv2.imwrite(f'{output_path}/combining_method.png', smoothed)


if __name__ == "__main__":
    import sys
    sys.path.append('src/data')
    sys.path.append('experiments')
    mask_path = 'data\processed\\train\masks\DJI_20250324092955_0011_V_March.png'
    output_path = 'experiments\\results'
    morphological_closing(mask_path, output_path)
    flood_fill(mask_path, output_path)
    convex_hull(mask_path, output_path)
    binary_dilation_and_erosion(mask_path, output_path)
    skimage_binary_fill_holes(mask_path, output_path)
    combining_method(mask_path, output_path)

