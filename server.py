import asyncio
import logging
import time
import traceback
from typing import Dict, Any, Optional
import socketio
import uvicorn
from ultralytics import YOLO
import numpy as np

import config
from utils import (
    decode_base64_to_image,
    resize_frame,
    format_detections,
    filter_detections_by_confidence,
    filter_detections_by_class,
    get_image_info
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class YOLODetectionServer:
    """Main YOLOv11n detection server with Socket.IO support"""
    
    def __init__(self):
        self.sio = socketio.AsyncServer(
            cors_allowed_origins=config.CORS_ORIGINS,
            async_mode='asgi'
        )
        self.app = socketio.ASGIApp(self.sio)
        self.model: Optional[YOLO] = None
        self.connected_clients = set()
        self.stats = {
            'total_frames_processed': 0,
            'total_detections': 0,
            'average_inference_time': 0.0,
            'server_start_time': time.time()
        }
        
        self._apply_preset_config()
        
        self._validate_config()
        
        self._register_handlers()
        
    def _apply_preset_config(self):
        """Apply preset configuration if specified in config"""
        if hasattr(config, 'ACTIVE_PRESET') and config.ACTIVE_PRESET:
            preset_name = config.ACTIVE_PRESET
            logger.info(f"Applying preset configuration: {preset_name}")
            
            try:
                if preset_name == "HIGH_ACCURACY":
                    config.CONFIDENCE_THRESHOLD = config.PRESET_HIGH_ACCURACY['conf']
                    config.IOU_THRESHOLD = config.PRESET_HIGH_ACCURACY['iou']
                    config.MAX_DETECTIONS = config.PRESET_HIGH_ACCURACY['max_det']
                elif preset_name == "BALANCED":
                    config.CONFIDENCE_THRESHOLD = config.PRESET_BALANCED['conf']
                    config.IOU_THRESHOLD = config.PRESET_BALANCED['iou']
                    config.MAX_DETECTIONS = config.PRESET_BALANCED['max_det']
                elif preset_name == "HIGH_RECALL":
                    config.CONFIDENCE_THRESHOLD = config.PRESET_HIGH_RECALL['conf']
                    config.IOU_THRESHOLD = config.PRESET_HIGH_RECALL['iou']
                    config.MAX_DETECTIONS = config.PRESET_HIGH_RECALL['max_det']
                elif preset_name == "VERY_HIGH_ACCURACY":
                    config.CONFIDENCE_THRESHOLD = config.PRESET_VERY_HIGH_ACCURACY['conf']
                    config.IOU_THRESHOLD = config.PRESET_VERY_HIGH_ACCURACY['iou']
                    config.MAX_DETECTIONS = config.PRESET_VERY_HIGH_ACCURACY['max_det']
                else:
                    logger.warning(f"Unknown preset: {preset_name}. Using default configuration.")
                    return
                
                logger.info(f"Preset applied: conf={config.CONFIDENCE_THRESHOLD}, iou={config.IOU_THRESHOLD}, max_det={config.MAX_DETECTIONS}")
            except Exception as e:
                logger.error(f"Failed to apply preset {preset_name}: {str(e)}")
                logger.error("Using default configuration instead.")
        
    def _validate_config(self):
        """Validate configuration parameters"""
        errors = []
        
        if not (0.0 <= config.CONFIDENCE_THRESHOLD <= 1.0):
            errors.append(f"CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got {config.CONFIDENCE_THRESHOLD}")
        
        if not (0.0 <= config.IOU_THRESHOLD <= 1.0):
            errors.append(f"IOU_THRESHOLD must be between 0.0 and 1.0, got {config.IOU_THRESHOLD}")
        
        if config.IMAGE_SIZE <= 0:
            errors.append(f"IMAGE_SIZE must be positive, got {config.IMAGE_SIZE}")
        
        if config.MAX_DETECTIONS <= 0:
            errors.append(f"MAX_DETECTIONS must be positive, got {config.MAX_DETECTIONS}")
        
        if not (1 <= config.SERVER_PORT <= 65535):
            errors.append(f"SERVER_PORT must be between 1 and 65535, got {config.SERVER_PORT}")
        
        if config.FRAME_MAX_DIM <= 0:
            errors.append(f"FRAME_MAX_DIM must be positive, got {config.FRAME_MAX_DIM}")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            raise ValueError("Invalid configuration parameters")
        
        logger.info("Configuration validation passed")
        
    def _register_handlers(self):
        """Register Socket.IO event handlers"""
        
        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection"""
            self.connected_clients.add(sid)
            logger.info(f"Client connected: {sid} (Total: {len(self.connected_clients)})")
            
            await self.sio.emit('server_info', {
                'message': 'Connected to YOLOv11n Detection Server',
                'model_loaded': self.model is not None,
                'device': config.DEVICE,
                'confidence_threshold': config.CONFIDENCE_THRESHOLD,
                'server_uptime': time.time() - self.stats['server_start_time']
            }, room=sid)
            
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            if sid in self.connected_clients:
                self.connected_clients.remove(sid)
            logger.info(f"Client disconnected: {sid} (Total: {len(self.connected_clients)})")
            
        @self.sio.event
        async def ping(sid, data=None):
            """Handle ping for connection testing"""
            await self.sio.emit('pong', {
                'timestamp': time.time(),
                'server_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }, room=sid)
            
        @self.sio.event
        async def frame(sid, data):
            """Handle incoming frame for detection"""
            try:
                start_time = time.time()
                
                if not isinstance(data, dict) or 'image' not in data:
                    await self.sio.emit('error', {
                        'message': 'Invalid frame data. Expected {"image": "base64_string"}'
                    }, room=sid)
                    return
                
                image_data = data['image']
                if not isinstance(image_data, str):
                    await self.sio.emit('error', {
                        'message': 'Image data must be a base64 string'
                    }, room=sid)
                    return
                
                detections = await self._process_frame(image_data, data.get('options', {}))
                
                processing_time = time.time() - start_time
                
                self.stats['total_frames_processed'] += 1
                self.stats['total_detections'] += len(detections)
                
                current_avg = self.stats['average_inference_time']
                total_frames = self.stats['total_frames_processed']
                self.stats['average_inference_time'] = (
                    (current_avg * (total_frames - 1) + processing_time) / total_frames
                )
                
                await self.sio.emit('detections', {
                    'detections': detections,
                    'processing_time': processing_time,
                    'frame_info': {
                        'frame_number': self.stats['total_frames_processed'],
                        'detection_count': len(detections)
                    },
                    'server_stats': {
                        'total_frames': self.stats['total_frames_processed'],
                        'total_detections': self.stats['total_detections'],
                        'average_time': self.stats['average_inference_time']
                    },
                    'display_config': {
                        'show_confidence': config.SHOW_CONFIDENCE,
                        'show_class_id': config.SHOW_CLASS_ID,
                        'bbox_color': config.BBOX_COLOR,
                        'bbox_thickness': config.BBOX_THICKNESS,
                        'font_scale': config.FONT_SCALE
                    }
                }, room=sid)
                
                if self.stats['total_frames_processed'] % config.PERFORMANCE_LOG_INTERVAL == 0:
                    logger.info(
                        f"Processed {self.stats['total_frames_processed']} frames, "
                        f"avg time: {self.stats['average_inference_time']:.3f}s, "
                        f"total detections: {self.stats['total_detections']}"
                    )
                    
            except Exception as e:
                logger.error(f"Error processing frame from {sid}: {str(e)}")
                logger.error(traceback.format_exc())
                await self.sio.emit('error', {
                    'message': f'Frame processing error: {str(e)}'
                }, room=sid)
                
        @self.sio.event
        async def get_stats(sid, data=None):
            """Handle stats request"""
            await self.sio.emit('stats', {
                'server_stats': self.stats,
                'connected_clients': len(self.connected_clients),
                'model_info': {
                    'loaded': self.model is not None,
                    'device': config.DEVICE,
                    'model_path': config.MODEL_PATH
                },
                'config': {
                    'confidence_threshold': config.CONFIDENCE_THRESHOLD,
                    'iou_threshold': config.IOU_THRESHOLD,
                    'max_detections': config.MAX_DETECTIONS,
                    'image_size': config.IMAGE_SIZE
                }
            }, room=sid)
            
    async def _process_frame(self, image_data: str, options: Dict[str, Any]) -> list:
        """Process a single frame for object detection"""
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        image = decode_base64_to_image(image_data)
        
        image_info = get_image_info(image)
        logger.debug(f"Processing image: {image_info['width']}x{image_info['height']}")
        
        if config.FRAME_MAX_DIM > 0:
            height, width = image.shape[:2]
            if max(height, width) > config.FRAME_MAX_DIM:
                if width > height:
                    new_width = config.FRAME_MAX_DIM
                    new_height = int(height * (config.FRAME_MAX_DIM / width))
                else:
                    new_height = config.FRAME_MAX_DIM
                    new_width = int(width * (config.FRAME_MAX_DIM / height))
                
                image = resize_frame(image, (new_width, new_height))
                logger.debug(f"Resized image to: {new_width}x{new_height}")
        
        yolo_params = {
            'conf': config.CONFIDENCE_THRESHOLD,
            'iou': config.IOU_THRESHOLD,
            'verbose': config.VERBOSE_INFERENCE,
            'device': config.DEVICE,
            'half': config.HALF_PRECISION,
            'max_det': config.MAX_DETECTIONS,
            'agnostic_nms': config.AGNOSTIC_NMS,
        }
        results = self.model(image, **yolo_params)
        
        detections = format_detections(results)
        
        if 'min_confidence' in options:
            min_conf = float(options['min_confidence'])
            detections = filter_detections_by_confidence(detections, min_conf)
            
        if 'allowed_classes' in options:
            allowed_classes = [int(c) for c in options['allowed_classes']]
            detections = filter_detections_by_class(detections, allowed_classes)
        
        return detections
        
    async def load_model(self):
        """Load the YOLOv11n model"""
        try:
            logger.info(f"Loading YOLOv11n model from: {config.MODEL_PATH}")
            logger.info(f"Device: {config.DEVICE}")
            logger.info(f"Configuration: conf={config.CONFIDENCE_THRESHOLD}, iou={config.IOU_THRESHOLD}, max_det={config.MAX_DETECTIONS}")
            
            self.model = YOLO(config.MODEL_PATH)
            
            test_image = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
            test_yolo_params = {
                'conf': config.CONFIDENCE_THRESHOLD,
                'iou': config.IOU_THRESHOLD,
                'verbose': config.VERBOSE_INFERENCE,
                'device': config.DEVICE,
                'half': config.HALF_PRECISION,
                'max_det': config.MAX_DETECTIONS,
                'agnostic_nms': config.AGNOSTIC_NMS,
            }
            test_results = self.model(test_image, **test_yolo_params)
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model classes: {len(self.model.names)} classes")
            logger.info(f"Model device: {next(self.model.model.parameters()).device}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    async def start_server(self):
        """Start the Socket.IO server"""
        try:
            if not await self.load_model():
                logger.error("Failed to load model. Server cannot start.")
                return False
                
            logger.info(f"Starting YOLOv11n Detection Server...")
            logger.info(f"Server: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
            logger.info(f"Device: {config.DEVICE}")
            logger.info(f"Confidence: {config.CONFIDENCE_THRESHOLD:.1%}")
            logger.info(f"Image Size: {config.IMAGE_SIZE}")
            logger.info(f"CORS Origins: {config.CORS_ORIGINS}")
            logger.info("=" * config.LOG_SEPARATOR_LENGTH)
            
            config_uvicorn = uvicorn.Config(
                app=self.app,
                host=config.SERVER_HOST,
                port=config.SERVER_PORT,
                log_level="info"
            )
            server = uvicorn.Server(config_uvicorn)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Server failed to start: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            'server_stats': self.stats,
            'connected_clients': len(self.connected_clients),
            'model_loaded': self.model is not None,
            'config': {
                'host': config.SERVER_HOST,
                'port': config.SERVER_PORT,
                'device': config.DEVICE,
                'model_path': config.MODEL_PATH,
                'confidence_threshold': config.CONFIDENCE_THRESHOLD,
                'iou_threshold': config.IOU_THRESHOLD,
                'max_detections': config.MAX_DETECTIONS,
                'image_size': config.IMAGE_SIZE,
                'cors_origins': config.CORS_ORIGINS
            }
        }


async def main():
    """Main function to start the server"""
    server = YOLODetectionServer()
    await server.start_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server crashed: {str(e)}")
        logger.error(traceback.format_exc())
