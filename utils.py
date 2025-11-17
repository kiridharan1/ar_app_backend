"""
Utility functions for YOLOv11n Live Detection Backend
"""

import cv2
import numpy as np
import base64
from typing import List, Dict, Any, Tuple


def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to OpenCV image (BGR format)
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        np.ndarray: Decoded image in BGR format
    """
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def resize_frame(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Resize image to target size while maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        
    Returns:
        np.ndarray: Resized image
    """
    return cv2.resize(image, target_size)


def format_detections(results) -> List[Dict[str, Any]]:
    """
    Format YOLO detection results for JSON serialization
    
    Args:
        results: YOLO inference results
        
    Returns:
        List[Dict]: Formatted detections with bbox, confidence, and class
    """
    detections = []
    
    if results and len(results) > 0:
        result = results[0] 
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy() 
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                detection = {
                    'bbox': boxes[i].tolist(), 
                    'confidence': float(confidences[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': result.names[class_ids[i]] if hasattr(result, 'names') else f'class_{class_ids[i]}'
                }
                detections.append(detection)
    
    return detections


def filter_detections_by_confidence(detections: List[Dict[str, Any]], 
                                  min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """
    Filter detections by minimum confidence threshold
    
    Args:
        detections: List of detection dictionaries
        min_confidence: Minimum confidence threshold
        
    Returns:
        List[Dict]: Filtered detections
    """
    return [det for det in detections if det['confidence'] >= min_confidence]


def filter_detections_by_class(detections: List[Dict[str, Any]], 
                             allowed_classes: List[int]) -> List[Dict[str, Any]]:
    """
    Filter detections by allowed class IDs
    
    Args:
        detections: List of detection dictionaries
        allowed_classes: List of allowed class IDs
        
    Returns:
        List[Dict]: Filtered detections
    """
    return [det for det in detections if det['class_id'] in allowed_classes]


def compress_image(image: np.ndarray, quality: int = 80) -> str:
    """
    Compress image to JPEG and return as base64 string
    
    Args:
        image: Input image
        quality: JPEG quality (0-100)
        
    Returns:
        str: Base64 encoded compressed image
    """
    # Encode image to JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', image, encode_params)
    
    # Convert to base64
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return image_base64


def get_image_info(image: np.ndarray) -> Dict[str, Any]:
    """
    Get basic information about an image
    
    Args:
        image: Input image
        
    Returns:
        Dict: Image information
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'dtype': str(image.dtype)
    }
