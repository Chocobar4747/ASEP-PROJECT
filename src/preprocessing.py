"""
Image Preprocessing Pipeline
==============================
Handles ROI cropping, resizing, normalization, and color feature extraction
for dyed sample images.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_image(image_path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def crop_roi(
    image: np.ndarray,
    method: str = "smart",
    crop_fraction: float = 0.5,
) -> np.ndarray:
    """
    Crop the Region of Interest from the image.
    
    Args:
        image: Input image (H, W, 3)
        method: 'center' for center crop, 'smart' for saturation-based contour detection
        crop_fraction: Fraction of image to keep (for center crop fallback)
    
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    
    if method == "center":
        # Center crop
        margin_h = int(h * (1 - crop_fraction) / 2)
        margin_w = int(w * (1 - crop_fraction) / 2)
        cropped = image[margin_h:h - margin_h, margin_w:w - margin_w]
        return cropped
    
    elif method in ["contour", "smart"]:
        # Real-world fix: Use the Saturation channel to find the highly colored liquid, 
        # naturally ignoring gray/white backgrounds (walls, tables, skin)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        
        # Blur heavily to reduce noise and reflections
        blurred = cv2.GaussianBlur(s_channel, (15, 15), 0)
        
        # Adaptive threshold on saturation
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up small specks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (assumed to be the liquid sample)
            largest = max(contours, key=cv2.contourArea)
            
            # Only accept if it's a reasonably large object (>5% of image area)
            if cv2.contourArea(largest) > (h * w * 0.05):
                x, y, cw, ch = cv2.boundingRect(largest)
                
                # Real-world fix: Crop slightly INSIDE the detected bounding box 
                # to avoid specular reflections off the glass edges of the flask/cuvette
                inset_x = int(cw * 0.15)
                inset_y = int(ch * 0.15)
                
                final_x = min(x + inset_x, w - 10)
                final_y = min(y + inset_y, h - 10)
                final_w = max(10, cw - 2 * inset_x)
                final_h = max(10, ch - 2 * inset_y)
                
                return image[final_y:final_y+final_h, final_x:final_x+final_w]
        
        # Fallback to a tight center crop if no contours found
        return crop_roi(image, method="center", crop_fraction=crop_fraction)
    
    else:
        raise ValueError(f"Unknown crop method: {method}")


def resize_image(image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image to target size using bilinear interpolation."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def normalize_pixels(
    image: np.ndarray,
    method: str = "standard",
) -> np.ndarray:
    """
    Normalize pixel values.
    
    Args:
        image: Input image (H, W, 3) with uint8 values [0, 255]
        method: 'standard' for [0,1], 'imagenet' for ImageNet mean/std
    
    Returns:
        Normalized image as float32
    """
    img = image.astype(np.float32) / 255.0
    
    if method == "imagenet":
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    
    return img


def extract_color_features(image: np.ndarray) -> Dict[str, float]:
    """
    Extract comprehensive color features from an image.
    
    Returns dict with RGB and HSV statistics plus dominant color info.
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # ── RGB Features ──
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    features = {
        "r_mean": float(r.mean()),
        "r_std": float(r.std()),
        "g_mean": float(g.mean()),
        "g_std": float(g.std()),
        "b_mean": float(b.mean()),
        "b_std": float(b.std()),
    }
    
    # ── HSV Features ──
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    features.update({
        "h_mean": float(h.mean()),
        "h_std": float(h.std()),
        "s_mean": float(s.mean()),
        "s_std": float(s.std()),
        "v_mean": float(v.mean()),
        "v_std": float(v.std()),
    })
    
    # ── Derived Features ──
    # Color intensity (grayscale equivalent)
    features["intensity"] = float(0.299 * r.mean() + 0.587 * g.mean() + 0.114 * b.mean())
    
    # Color ratios (useful for dye identification)
    total = r.mean() + g.mean() + b.mean() + 1e-8
    features["r_ratio"] = float(r.mean() / total)
    features["g_ratio"] = float(g.mean() / total)
    features["b_ratio"] = float(b.mean() / total)
    
    # Dominant color via k-means (k=3)
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    
    # Find the most frequent cluster
    unique, counts = np.unique(labels, return_counts=True)
    dominant_idx = unique[np.argmax(counts)]
    dominant_color = centers[dominant_idx]
    
    features["dominant_r"] = float(dominant_color[0])
    features["dominant_g"] = float(dominant_color[1])
    features["dominant_b"] = float(dominant_color[2])
    
    return features


def preprocess_pipeline(
    image_path: str,
    crop_method: str = "center",
    target_size: Tuple[int, int] = (224, 224),
    normalize_method: str = "standard",
    extract_features: bool = True,
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """
    Complete preprocessing pipeline: load → crop → resize → normalize.
    
    Args:
        image_path: Path to the input image
        crop_method: ROI cropping method
        target_size: Output image dimensions
        normalize_method: Pixel normalization method
        extract_features: Whether to extract color features
    
    Returns:
        Tuple of (preprocessed_image, features_dict or None)
    """
    # Load
    image = load_image(image_path)
    
    # Extract features before normalization (needs uint8)
    features = None
    if extract_features:
        features = extract_color_features(image)
    
    # Crop ROI
    image = crop_roi(image, method=crop_method)
    
    # Resize
    image = resize_image(image, size=target_size)
    
    # Normalize
    image = normalize_pixels(image, method=normalize_method)
    
    return image, features
