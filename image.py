import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim
import warnings

# Helper function to divide the image into blocks
def block_shaped(arr, block_size):
    h, w = arr.shape
    bh, bw = block_size
    assert h % bh == 0, "Height must be divisible by block height"
    assert w % bw == 0, "Width must be divisible by block width"
    return (arr.reshape(h // bh, bh, -1, bw)
               .swapaxes(1, 2)
               .reshape(-1, bh, bw))

# Function to compute SSIM for adjacent blocks
def compute_block_ssim(blocks):
    block_similarities = []
    for i in range(len(blocks) - 1):
        ssim_value, _ = ssim(blocks[i], blocks[i + 1], data_range=blocks[i].max() - blocks[i].min(), full=True)
        if np.isnan(ssim_value) or np.isinf(ssim_value):
            ssim_value = 1  # Treat as perfectly similar if SSIM fails
        block_similarities.append(ssim_value)
    return np.mean(block_similarities)

# Load and validate the image
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}. Please check the path.")
    return img

# Main processing function
def process_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram analysis (top-left and bottom-right quadrants)
    hist1, _ = np.histogram(gray[:100, :100].ravel(), bins=256, range=[0, 256])
    hist2, _ = np.histogram(gray[100:, 100:].ravel(), bins=256, range=[0, 256])
    histogram_diff = np.mean(np.abs(hist1 - hist2))

    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges)

    # Block-based analysis with SSIM
    blocks = block_shaped(gray, (16, 16))
    block_similarity_avg = compute_block_ssim(blocks)

    # Feature extraction (Local Binary Patterns)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))

    # Combine features into a single feature vector
    features = np.hstack([histogram_diff, block_similarity_avg, edge_density, lbp_hist.mean()])

    return features, gray, edges, hist1, hist2

# Machine learning: Anomaly detection using Isolation Forest
def detect_anomalies(features):
    # Check and handle NaN, infinite values
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
    features = features.reshape(1, -1)

    # Apply Isolation Forest for anomaly detection
    model = IsolationForest(contamination=0.01)
    model.fit(features)
    prediction = model.predict(features)
    return prediction[0]

# Visualization function
def visualize_results(gray, edges, hist1, hist2):
    plt.figure(figsize=(12, 4))

    # Display grayscale image
    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Image")

    # Histogram comparison
    plt.subplot(1, 3, 2)
    plt.plot(hist1, label="Histogram 1 (Top-Left)")
    plt.plot(hist2, label="Histogram 2 (Bottom-Right)")
    plt.legend()
    plt.title("Histogram Comparison")

    # Display edges
    plt.subplot(1, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")
    
    plt.show()

# Main code execution
if __name__ == "__main__":
    # Load the image
    img_path = r"I:\python\py code\CSDF\image.jpeg"
    img = load_image(img_path)

    # Process the image and extract features
    features, gray, edges, hist1, hist2 = process_image(img)

    # Perform tampering detection
    prediction = detect_anomalies(features)

    # Evaluate and print results
    if prediction == -1:
        print("Image is likely tampered.")
    else:
        print("Image appears to be intact.")

    # Visualize the analysis results
    visualize_results(gray, edges, hist1, hist2)
