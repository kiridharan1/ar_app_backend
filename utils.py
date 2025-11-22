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
        
        if image is None:
            raise ValueError("Failed to decode image data - invalid image format")
        
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


def format_detections(results, model=None) -> List[Dict[str, Any]]:
    """
    Format YOLO detection results for JSON serialization
    
    Args:
        results: YOLO inference results
        model: Optional YOLO model object for accessing class names
        
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
            
            # Get class names from model or result
            names = None
            if model is not None and hasattr(model, 'names'):
                names = model.names
            elif hasattr(result, 'names'):
                names = result.names
            
            for i in range(len(boxes)):
                class_id = int(class_ids[i])
                if names is not None and class_id in names:
                    class_name = names[class_id]
                else:
                    class_name = f'class_{class_id}'
                
                detection = {
                    'bbox': boxes[i].tolist(), 
                    'confidence': float(confidences[i]),
                    'class_id': class_id,
                    'class_name': class_name
                }
                detections.append(detection)
    
    return detections


def get_yolo_scale_info(result, original_width: int, original_height: int, image_size: int = 640) -> Dict[str, Any]:
    """
    Get scaling information from YOLO results to transform coordinates back to original image
    
    Args:
        result: YOLO result object
        original_width: Original image width
        original_height: Original image height
        image_size: YOLO processing image size (default 640)
        
    Returns:
        Dict with scale factors and padding information
    """
    if hasattr(result, 'orig_shape') and result.orig_shape is not None:
        orig_h, orig_w = result.orig_shape
    else:
        orig_h, orig_w = original_height, original_width
    
    if hasattr(result, 'shape') and result.shape is not None:
        processed_h, processed_w = result.shape[-2:]
    else:
        processed_h = processed_w = image_size
    
    scale = min(processed_w / orig_w, processed_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    pad_x = (processed_w - new_w) / 2
    pad_y = (processed_h - new_h) / 2
    
    return {
        'scale': scale,
        'pad_x': pad_x,
        'pad_y': pad_y,
        'new_w': new_w,
        'new_h': new_h,
        'processed_w': processed_w,
        'processed_h': processed_h
    }


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


def scale_bboxes_to_original(detections: List[Dict[str, Any]], 
                             original_width: int, 
                             original_height: int,
                             scale_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Scale bounding box coordinates from YOLO's letterboxed coordinate system back to original image dimensions
    
    Args:
        detections: List of detection dictionaries with bbox coordinates
        original_width: Original image width
        original_height: Original image height
        scale_info: Dictionary with scale, pad_x, pad_y from get_yolo_scale_info()
        
    Returns:
        List[Dict]: Detections with scaled bounding boxes
    """
    scale = scale_info['scale']
    pad_x = scale_info['pad_x']
    pad_y = scale_info['pad_y']
    
    if scale == 1.0 and pad_x == 0 and pad_y == 0:
        return detections
    
    scaled_detections = []
    for det in detections:
        bbox = det['bbox']
        
        x1, y1, x2, y2 = bbox
        
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))
        
        scaled_bbox = [x1, y1, x2, y2]
        scaled_det = det.copy()
        scaled_det['bbox'] = scaled_bbox
        scaled_detections.append(scaled_det)
    
    return scaled_detections