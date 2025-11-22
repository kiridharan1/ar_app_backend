"""
YOLOv11n Detection Test Client - Static Image Testing
Tests the server with static images
"""

import asyncio
import base64
import cv2
import socketio
import time
import json
import os
import glob
from typing import Optional, List
import numpy as np

import config

class YOLOImageTestClient:
    """Test client for YOLOv11n detection server with static images"""
    
    def __init__(self, server_url: str = "http://localhost:3000"):
        self.server_url = server_url
        self.sio = socketio.AsyncClient()
        self.running = False
        self.stats = {
            'images_processed': 0,
            'total_detections': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        self._register_handlers()
        
    def _register_handlers(self):
        """Register Socket.IO event handlers"""
        
        @self.sio.event
        async def connect():
            print("Connected to YOLOv11n Detection Server")
            
        @self.sio.event
        async def disconnect():
            print("Disconnected from server")
            
        @self.sio.event
        async def server_info(data):
            print(f"Server Info: {data['message']}")
            print(f"Model loaded: {data['model_loaded']}")
            print(f"Device: {data['device']}")
            print(f"Confidence: {data['confidence_threshold']:.1%}")
            
        @self.sio.event
        async def detections(data):
            """Handle detection results"""
            self.stats['images_processed'] += 1
            self.stats['total_processing_time'] += data['processing_time']
            self.stats['total_detections'] += len(data['detections'])
            
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['images_processed']
            )
            
            detections = data['detections']
            frame_info = data['frame_info']
            
            print(f"Image {frame_info['frame_number']}: "
                  f"{len(detections)} detections, "
                  f"{data['processing_time']:.3f}s processing time")
            
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                class_id = detection['class_id']
                
                print(f"   Detection {i+1}: {class_name} ({confidence:.1%}) "
                      f"at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
                
        @self.sio.event
        async def error(data):
            print(f"Server error: {data['message']}")
            
        @self.sio.event
        async def stats(data):
            """Handle server statistics"""
            server_stats = data['server_stats']
            print(f"Server Stats:")
            print(f"   Total frames processed: {server_stats['total_frames_processed']}")
            print(f"   Total detections: {server_stats['total_detections']}")
            print(f"   Average processing time: {server_stats['average_inference_time']:.3f}s")
            
    async def connect_to_server(self):
        """Connect to the detection server"""
        try:
            await self.sio.connect(self.server_url)
            print(f"Connecting to {self.server_url}...")
            await asyncio.sleep(1) 
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
            
    def find_test_images(self, directory: str = "test_images") -> List[str]:
        """Find test images in directory"""
        if not os.path.exists(directory):
            print(f"Creating test images directory: {directory}")
            os.makedirs(directory)

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(directory, ext)))
            image_files.extend(glob.glob(os.path.join(directory, ext.upper())))
            
        return sorted(image_files)
        
    def create_sample_image(self, filename: str = "test_images/sample.jpg"):
        """Create a sample test image if none exist"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green rectangle
        cv2.rectangle(img, (300, 150), (400, 250), (255, 0, 0), -1)  # Blue rectangle
        cv2.circle(img, (500, 200), 50, (0, 0, 255), -1)  # Red circle
        
        cv2.putText(img, "YOLOv11n Test Image", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, img)
        print(f"Created sample test image: {filename}")
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"Could not read image: {image_path}")
                
            height, width = image.shape[:2]
            if max(height, width) > config.FRAME_MAX_DIM:
                if width > height:
                    new_width = config.FRAME_MAX_DIM
                    new_height = int(height * (config.FRAME_MAX_DIM / width))
                else:
                    new_height = config.FRAME_MAX_DIM
                    new_width = int(width * (config.FRAME_MAX_DIM / height))
                
                image = cv2.resize(image, (new_width, new_height))
                print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
            
        except Exception as e:
            raise Exception(f"Failed to encode image {image_path}: {e}")
            
    async def send_image(self, image_path: str, options: dict = None):
        """Send image to server for detection"""
        try:
            image_base64 = self.encode_image_to_base64(image_path)

            image_data = {
                'image': image_base64,
                'options': options or {}
            }
            
            print(f"Sending image: {os.path.basename(image_path)}")
            await self.sio.emit('frame', image_data)
            
        except Exception as e:
            print(f"Error sending image {image_path}: {e}")
            
    def draw_detections_on_image(self, image_path: str, detections: list, output_path: str = None):
        """Draw detection results on image and save"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                return
                
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                class_id = detection['class_id']
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), config.BBOX_COLOR, config.BBOX_THICKNESS)
                
                if config.SHOW_CONFIDENCE:
                    label = f"{class_name}: {confidence:.1%}"
                else:
                    label = class_name
                    
                if config.SHOW_CLASS_ID:
                    label += f" (ID: {class_id})"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), config.BBOX_COLOR, -1)
                
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, (255, 255, 255), 2)
            
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = f"test_images/results/{base_name}_detected.jpg"
                
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            print(f"Saved detection result: {output_path}")
            
        except Exception as e:
            print(f"Error drawing detections: {e}")
            
    def print_stats(self):
        """Print client statistics"""
        print(f"Client Stats:")
        print(f"   Images processed: {self.stats['images_processed']}")
        print(f"   Total detections: {self.stats['total_detections']}")
        print(f"   Average processing time: {self.stats['average_processing_time']:.3f}s")
        print(f"   Total processing time: {self.stats['total_processing_time']:.3f}s")
        
    async def run_image_test(self, image_paths: List[str] = None):
        """Run image test with provided images or find test images"""
        print("Starting YOLOv11n Image Test Client")
        print("=" * 50)
        
        if not await self.connect_to_server():
            return
            
        if image_paths is None:
            image_paths = self.find_test_images()
            
        if not image_paths:
            print("No test images found. Creating sample image...")
            self.create_sample_image()
            image_paths = self.find_test_images()
            
        if not image_paths:
            print("No test images available")
            return
            
        print(f"Found {len(image_paths)} test images")
        print("=" * 50)
        
        self.running = True
        
        try:
            for i, image_path in enumerate(image_paths):
                if not self.running:
                    break
                    
                print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                options = {
                    'min_confidence': config.CONFIDENCE_THRESHOLD,
                    # 'allowed_classes': [0, 1, 2]  # Uncomment to filter specific classes
                }
                
                await self.send_image(image_path, options)
                
                await asyncio.sleep(2)
                
                if i < len(image_paths) - 1:
                    try:
                        response = input("Press Enter to continue, 'q' to quit, 's' for stats: ").strip().lower()
                        if response == 'q':
                            break
                        elif response == 's':
                            self.print_stats()
                            await self.sio.emit('get_stats')
                    except KeyboardInterrupt:
                        break
                        
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        except Exception as e:
            print(f"Test error: {e}")
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.sio.connected:
            await self.sio.disconnect()
        print("Cleanup completed")


async def main():
    """Main function"""
    client = YOLOImageTestClient()    
    await client.run_image_test()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        print(f"Client error: {e}")
