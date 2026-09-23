import cv2
import numpy as np
from scipy.ndimage import uniform_filter

def lee_filter(img, size=7):
    """
    Calculate local variance using scipy.ndimage.uniform_filter.
    Compute weights: weight = img_variance / (img_variance + overall_variance).
    Apply the formula and return the filtered image.
    """
    img = img.astype(np.float32)
    img_mean = uniform_filter(img, size)
    img_sqr_mean = uniform_filter(img**2, size)
    img_variance = img_sqr_mean - img_mean**2
    
    overall_variance = np.var(img)
    
    # Avoid division by zero
    weight = img_variance / (img_variance + overall_variance + 1e-8)
    
    filtered_img = img_mean + weight * (img - img_mean)
    return np.clip(filtered_img, 0, 255).astype(np.uint8)

def apply_clahe(img):
    """
    Use cv2.createCLAHE with clipLimit=3.0 and tileGridSize=(8, 8).
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess_sonar_pipeline(image_bytes):
    """
    Accepts raw image bytes from an API upload, decodes it into a grayscale NumPy array,
    applies the Lee Filter, applies CLAHE, and returns the tensor-ready array.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image from bytes")
        
    filtered_img = lee_filter(img, size=7)
    enhanced_img = apply_clahe(filtered_img)
    
    return enhanced_img
