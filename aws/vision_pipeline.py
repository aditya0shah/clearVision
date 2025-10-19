import cv2
import numpy as np
import io
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
        
    
    def process_image(self, file_object, depth_file=None):
        """
        Process image through vision pipeline from file object in memory.
        Optionally accepts a depth file for depth processing.
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
                # Use provided depth file if available, otherwise generate synthetic depth
                if depth_file is not None:
                    depth_module = self.process_depth_file(depth_file, detections_dict, image_rgb.shape)
                else:
                    depth_module = self.makeDepthMap(detections_dict, image_rgb.shape)
                
                if depth_module is not None:
                    self.depth_module = depth_module
                    # Convert numpy arrays to Python native types for JSON serialization
                    buzz_values = depth_module if isinstance(depth_module, list) else depth_module.get_buzz_values()
                    # Ensure all values are Python native types
                    buzz_values = [float(val) for val in buzz_values]
                    
                    processing_results['depth_processing'] = {
                        'buzz_values': buzz_values,
                        'depth_source': 'provided_file' if depth_file is not None else 'synthetic'
                    }
            except Exception as e:
                print(f"Error in depth processing: {e}")
                return {
                    'error': f'Depth processing failed: {str(e)}',
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
        
        # Create depth gradient using vectorized operations (0-10 meters)
        depth_map = (1.0 - (distance_from_center / max_distance)) * 1.0 # Scale to 0-10 meters
        
        try:
            depth_module = DepthModule(depth_map, mask)
            buzz_values = depth_module.get_buzz_values()
            # Convert numpy types to Python native types for JSON serialization
            return [float(val) for val in buzz_values]
        except Exception as e:
            print(f"Error in DepthModule: {e}")
            return None
        
    def process_depth_file(self, depth_file, detections_dict, image_shape):
        """
        Process a depth file and create a DepthModule instance.
        Handles JPG-serialized depth data from RealSense cameras.
        """
        try:
            # Read depth file bytes
            depth_bytes = depth_file.read()
            depth_file.seek(0)  # Reset file pointer
            
            print(f"📊 Processing depth data: {len(depth_bytes)} bytes")
            print(f"📊 Expected image shape: {image_shape}")
            
            # Try different loading methods for JPG-serialized depth data
            depth_map = None
            
            # Method 1: Try loading as JPG image first (most likely for your case)
            try:
                # Decode as JPG image
                nparr = np.frombuffer(depth_bytes, np.uint8)
                depth_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if depth_image is not None:
                    # Convert from uint8 to float32 and scale to 0-10 meters
                    # JPG values 0-255 need to be scaled to 0-10 meters
                    depth_map = 6.0 * (depth_image.astype(np.float32) / 255.0)  # Scale to 0-10 meters
                    print(f"✅ Loaded depth data as JPG image: {depth_map.shape}")
                    print(f"📊 Depth range: {depth_map.min():.3f}m to {depth_map.max():.3f}m")
            except Exception as e1:
                print(f"⚠️  Failed to load as JPG: {e1}")
                
                # Method 2: Try loading as .npy file
                try:
                    depth_map = np.load(io.BytesIO(depth_bytes), allow_pickle=False)
                    print(f"✅ Loaded depth data as .npy file: {depth_map.shape}")
                except Exception as e2:
                    print(f"⚠️  Failed to load as .npy without pickle: {e2}")
                    
                    # Method 3: Try loading as .npy file with pickle
                    try:
                        depth_map = np.load(io.BytesIO(depth_bytes), allow_pickle=True)
                        print(f"✅ Loaded depth data as .npy with pickle: {depth_map.shape}")
                    except Exception as e3:
                        print(f"⚠️  Failed to load as .npy with pickle: {e3}")
                        
                        # Method 4: Try loading as raw float32 array (fallback)
                        try:
                            expected_size = image_shape[0] * image_shape[1] * 4
                            if len(depth_bytes) == expected_size:
                                depth_map = np.frombuffer(depth_bytes, dtype=np.float32)
                                depth_map = depth_map.reshape(image_shape[0], image_shape[1])
                                print(f"✅ Loaded depth data as raw float32: {depth_map.shape}")
                            else:
                                raise ValueError(f"File size {len(depth_bytes)} doesn't match expected {expected_size}")
                        except Exception as e4:
                            print(f"❌ Failed to load depth data with all methods")
                            print(f"   JPG: {e1}")
                            print(f"   .npy without pickle: {e2}")
                            print(f"   .npy with pickle: {e3}")
                            print(f"   Raw float32: {e4}")
                            raise ValueError(f"Cannot load depth data. Tried JPG, .npy, and raw float32 formats.")
            
            if depth_map is None:
                raise ValueError("Failed to load depth data")
            
            # Validate depth map
            if depth_map.ndim != 2:
                raise ValueError(f"Depth map must be 2D, got {depth_map.ndim}D")
            
            if depth_map.shape != (image_shape[0], image_shape[1]):
                print(f"⚠️  Depth map shape {depth_map.shape} doesn't match image shape {image_shape[:2]}")
                # Try to resize if possible
                if depth_map.size == image_shape[0] * image_shape[1]:
                    depth_map = depth_map.reshape(image_shape[0], image_shape[1])
                    print("✅ Reshaped depth map to match image dimensions")
                else:
                    raise ValueError(f"Depth map size {depth_map.size} doesn't match image size {image_shape[0] * image_shape[1]}")
            
            # Ensure float32 data type
            if depth_map.dtype != np.float32:
                depth_map = depth_map.astype(np.float32)
                print("✅ Converted depth map to float32")
            
            print(f"✅ Depth map validated: shape={depth_map.shape}, dtype={depth_map.dtype}")
            print(f"📊 Depth range: {depth_map.min():.3f}m to {depth_map.max():.3f}m")
            
            # Log some sample depth values for debugging
            if depth_map.size > 0:
                non_zero_depths = depth_map[depth_map > 0]
                if len(non_zero_depths) > 0:
                    print(f"📊 Sample depths: min={non_zero_depths.min():.3f}m, max={non_zero_depths.max():.3f}m, mean={non_zero_depths.mean():.3f}m")
            
            # Create mask from detections
            height, width = image_shape[:2]
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
            
            # Create DepthModule with provided depth map
            depth_module = DepthModule(depth_map, mask)
            buzz_values = depth_module.get_buzz_values()
            # Convert numpy types to Python native types for JSON serialization
            return [float(val) for val in buzz_values]
            
        except Exception as e:
            print(f"Error processing depth file: {e}")
            # Fallback to synthetic depth if depth file processing fails
            return self.makeDepthMap(detections_dict, image_shape)
    
    def get_depth_module(self):
        """Get the current depth module if available."""
        return getattr(self, 'depth_module', None)