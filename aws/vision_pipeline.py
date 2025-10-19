import cv2
import numpy as np
from PIL import Image
from typing import List
from DinoModule import DinoModule, DetectionResult
from DepthModule import DepthModule


class VisionPipeline:
    """
    Vision pipeline class for processing images through computer vision algorithms.
    """
    
    def __init__(self):
        # Initialize DINO module
        self.dino_module = DinoModule()
        self.device = self.dino_module.device

    
    def detect(self, image: Image.Image, labels: List[str], threshold: float = 0.3) -> List[DetectionResult]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """
        return self.dino_module.detect(image, labels, threshold)
        
    
    def process_image(self, file_object):
        """
        Process image through vision pipeline from file object in memory.
        """
        try:
            # Convert file object to numpy array for OpenCV
            file_bytes = file_object.read()
            file_object.seek(0)  # Reset file pointer
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(file_bytes, np.uint8)
            
            # Decode image from memory
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image from memory")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Use grounded segmentation for complete pipeline
            detection_labels = ["person", "door", "chair", "table", "block", "wall"]
            detections = self.dino_module.detect(pil_image, detection_labels, 0.3)
            
            # Convert DetectionResult objects to dictionaries for JSON serialization
            detections_dict = []
            for detection in detections:
                detections_dict.append(detection.to_dict())

            # Prepare processing results
            processing_results = {
                'detections': detections_dict
            }

            try:
                depth_module = self.makeDepthMap(detections_dict, image_rgb.shape)
                if depth_module is not None:
                    self.depth_module = depth_module
            except Exception as e:
                print(f"Error in makeDepthMap: {e}")
                return {
                    'error': f'Depth map creation failed: {str(e)}',
                    'processing_status': 'failed'
                }
            
            return processing_results
            
        except Exception as e:
            return {
                'error': f'Vision processing failed: {str(e)}',
                'processing_status': 'failed'
            }
    
    def makeDepthMap(self, detections_dict, image_shape):
        # Extract height and width from image shape
        height, width = image_shape[:2]
        
        # Create a binary mask initialized with zeros
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill mask areas where detections are found
        for detection in detections_dict:
            if 'box' in detection:
                box = detection['box']
                xmin = max(0, min(int(box['xmin']), width-1))
                ymin = max(0, min(int(box['ymin']), height-1))
                xmax = max(0, min(int(box['xmax']), width-1))
                ymax = max(0, min(int(box['ymax']), height-1))
                
                # Fill the bounding box area 
                mask[ymin:ymax, xmin:xmax] = 1
        
        # Create a simple depth map with radial gradient
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        # Create coordinate grids for vectorized operations
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        center_x, center_y = width // 2, height // 2
        
        # Vectorized distance calculation from center
        distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Create depth gradient using vectorized operations
        depth_map = 1.0 - (distance_from_center / max_distance)
        
        try:
            depth_module = DepthModule(depth_map, mask)
            return depth_module.get_buzz_values()
        except Exception as e:
            print(f"Error in DepthModule: {e}")
            return None
        
    def get_depth_module(self):
        """Get the current depth module if available."""
        return getattr(self, 'depth_module', None)