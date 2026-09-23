import numpy as np
import cv2

def validate_shadow(img, obb_xywhr, towfish_altitude):
    """
    Validates acoustic shadow presence strictly behind the detected object.
    obb_xywhr: [x_center, y_center, width, height, angle_in_radians]
    """
    h, w = img.shape
    xc, yc, box_w, box_h, angle = obb_xywhr
    
    # Sonar physics: Assume towfish nadir (sound source) is at the top center of the waterfall feed (x=w/2, y=0)
    nadir_x, nadir_y = w / 2, 0
    
    # Calculate directional vector from the nadir to the object center
    dir_x = xc - nadir_x
    dir_y = yc - nadir_y
    magnitude = np.sqrt(dir_x**2 + dir_y**2) + 1e-6
    
    # Normalize the direction vector
    dir_x, dir_y = dir_x / magnitude, dir_y / magnitude
    
    # Define a Region of Interest (ROI) immediately behind the object
    # We step outside the bounding box radius along the vector path
    shadow_step = max(box_w, box_h) * 0.6 
    shadow_center_x = int(xc + (dir_x * shadow_step))
    shadow_center_y = int(yc + (dir_y * shadow_step))
    
    # Extract a 50x50 pixel sample patch in the calculated shadow zone
    patch_size = 25
    x1 = max(0, shadow_center_x - patch_size)
    y1 = max(0, shadow_center_y - patch_size)
    x2 = min(w, shadow_center_x + patch_size)
    y2 = min(h, shadow_center_y + patch_size)
    
    shadow_roi = img[y1:y2, x1:x2]
    if shadow_roi.size == 0:
        return False # Edge of image, cannot validate mathematically
        
    # Acoustic shadow pixels are near-black (intensity < 40 out of 255)
    dark_pixels = np.sum(shadow_roi < 40)
    total_pixels = shadow_roi.size
    shadow_ratio = dark_pixels / total_pixels
    
    # If more than 30% of the ROI behind the object is dark, the physics check passes
    return shadow_ratio > 0.30