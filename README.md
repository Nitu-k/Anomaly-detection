# Anomaly-detection
#Image Tampering Detection Using Python
Features
Image Processing: Converts images to grayscale, performs histogram analysis, and edge detection.
Block-Based SSIM Analysis: Computes Structural Similarity Index between image blocks.
Feature Extraction: Uses Local Binary Patterns and combines features for anomaly detection.
Anomaly Detection: Employs Isolation Forest for identifying tampered images.

Usage
Load Image: The script processes a specified image path.
Run Detection: The model predicts whether the image is intact or tampered.
Visualization: Displays grayscale image, histograms, and edge detection results.

Requirements
Python 3.x, OpenCV, NumPy, Matplotlib, scikit-image, scikit-learn

Output
The program outputs whether an image is likely tampered and displays visual analyses such as grayscale conversion, histogram comparison, and edge detection.
