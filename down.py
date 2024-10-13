import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_clahe, (5, 5), 0)
    
    # Apply global thresholding for better bone segmentation
    _, bone_mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # Apply morphology to enhance bone mask
    kernel = np.ones((5, 5), np.uint8)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel)

    # Apply the bone mask
    segmented = cv2.bitwise_and(blurred, blurred, mask=bone_mask)
    
    return segmented

def classify_fracture_severity(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Could not load the image.")
        return
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Apply Canny edge detection with adjusted parameters
    edges = cv2.Canny(thresh, 50, 150)
    
    # Morphological operations to close gaps and remove small noise
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours by increasing the area threshold and adding aspect ratio checks
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000 and (cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3]) < 5]
    
    if not filtered_contours:
        print("No significant contours detected. The image is likely not fractured.")
        return "not fractured"
    
    # Select the largest contour as the most likely fracture region
    fracture_contour = max(filtered_contours, key=cv2.contourArea)
    
    # Get the bounding rectangle for the fracture
    x, y, w, h = cv2.boundingRect(fracture_contour)
    num_fragments = len(filtered_contours)
    
    # Example thresholds for severity classification
    length_threshold_mild = 50
    length_threshold_moderate = 100
    fragmentation_threshold_mild = 1
    fragmentation_threshold_moderate = 3

    # Determine severity based on thresholds
    if h <= length_threshold_mild and num_fragments <= fragmentation_threshold_mild:
        severity = 'mild'
    elif h <= length_threshold_moderate and num_fragments <= fragmentation_threshold_moderate:
        severity = 'moderate'
    else:
        severity = 'severe'
    
    # Draw the bounding box on the original image
    image_with_box = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_box)
    plt.title(f'Fracture Severity: {severity}')
    plt.show()
    
    return severity

# Test the function on an image
severity = classify_fracture_severity(r'C:\Users\pavib\OneDrive\Desktop\bone fracture\BoneFractureDataset\training\not_fractured\9-rotated2-rotated1-rotated3-rotated1.jpg')
