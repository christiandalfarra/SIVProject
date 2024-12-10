# Creating the Python script for the enhanced image processing pipeline with segmentation.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_processing_pipeline(image_path):
    # Load the input image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Preprocessing (Gaussian Blur)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    
    # Step 3: Enhancement (Histogram Equalization)
    equalized = cv2.equalizeHist(blurred)
    
    # Step 4: Feature Extraction (Edge Detection using Canny)
    edges = cv2.Canny(equalized, threshold1=50, threshold2=150)
    
    # Step 5: Segmentation
    # Method 1: Thresholding
    _, binary = cv2.threshold(equalized, 127, 255, cv2.THRESH_BINARY)
    
    # Method 2: Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Method 3: K-Means Clustering
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented = centers[labels.flatten()].reshape(image.shape).astype("uint8")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original")
    plt.subplot(2, 3, 2), plt.imshow(grayscale, cmap="gray"), plt.title("Grayscale")
    plt.subplot(2, 3, 3), plt.imshow(blurred, cmap="gray"), plt.title("Blurred")
    plt.subplot(2, 3, 4), plt.imshow(edges, cmap="gray"), plt.title("Edges")
    plt.subplot(2, 3, 5), plt.imshow(binary, cmap="gray"), plt.title("Thresholding")
    plt.subplot(2, 3, 6), plt.imshow(segmented), plt.title("K-Means Segmentation")
    plt.tight_layout()
    plt.show()

# Example usage
# Replace 'image.jpg' with the path to your image file.

image_processing_pipeline('P3.jpg')