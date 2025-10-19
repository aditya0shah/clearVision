import cv2
import numpy as np
from datetime import datetime
import threading
import time
from PIL import Image
import torch
from transformers import AutoModelForMaskGeneration, AutoProcessor
from typing import List, Dict, Any, Optional, Tuple, Union
from DinoModule import DinoModule, DetectionResult
from DepthModule import DepthModule


class VisionPipeline:
    """
    Vision pipeline class for processing images through computer vision algorithms.
    """
    
    def __init__(self, version="1.0.0", display_window=True):
        self.version = version
        self.display_window = False
        self.window_name = "Vision Pipeline Stream"
        self.window_initialized = False
        self.processing_stats = {
            'total_processed': 0,
            'successful_processing': 0,
            'failed_processing': 0
        }
        
        # Initialize display window if enabled
        if self.display_window:
            self._initialize_display_window()
            
        # Initialize DINO module
        self.dino_module = DinoModule()
        self.device = self.dino_module.device

        # Initialize SAM model for segmentation
        self.segmentator = AutoModelForMaskGeneration.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

    
    def _initialize_display_window(self):
        """Initialize the display window if not already done."""
        if not self.window_initialized:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.window_initialized = True
            print(f"Vision Pipeline display window '{self.window_name}' initialized")

            
    def detect(self, image: Image.Image, labels: List[str], threshold: float = 0.3, detector_id: Optional[str] = None) -> List[DetectionResult]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """
        return self.dino_module.detect(image, labels, threshold, detector_id)
        
    def mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        """Convert a binary mask to a polygon."""
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []
            
        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the vertices of the contour
        polygon = largest_contour.reshape(-1, 2).tolist()

        return polygon

    def polygon_to_mask(self, polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert a polygon to a segmentation mask.

        Args:
        - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
        - image_shape (tuple): Shape of the image (height, width) for the mask.

        Returns:
        - np.ndarray: Segmentation mask with the polygon filled.
        """
        # Create an empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        if not polygon:
            return mask

        # Convert polygon to an array of points
        pts = np.array(polygon, dtype=np.int32)

        # Fill the polygon with white color (255)
        cv2.fillPoly(mask, [pts], color=(255,))

        return mask
        
    def refine_masks(self, masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
        """Refine segmentation masks."""
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        if polygon_refinement:
            for idx, mask in enumerate(masks):
                shape = mask.shape
                polygon = self.mask_to_polygon(mask)
                if polygon:
                    mask = self.polygon_to_mask(polygon, shape)
                    masks[idx] = mask

        return masks


    def get_boxes(self, results: List[DetectionResult]) -> List[List[List[float]]]:
        """Extract bounding boxes from detection results."""
        return self.dino_module.get_boxes(results)

    def _get_boxes(self, detection_results: List[Dict[str, Any]]) -> List[List[List[float]]]:
        """Extract bounding boxes from detection results in SAM format."""
        boxes = []
        for detection in detection_results:
            box = detection['box']
            # Convert to format expected by SAM: [[[x1, y1, x2, y2]]]
            # SAM expects: List[List[List[float]]] - list of objects, each object has list of boxes, each box is [x1, y1, x2, y2]
            boxes.append([[box['xmin'], box['ymin'], box['xmax'], box['ymax']]])
        return boxes

    def segment(self, image: Image.Image, detection_results: List[DetectionResult], polygon_refinement: bool = False) -> List[DetectionResult]:
        """
        Perform segmentation using SAM on detected objects.
        
        Args:
            image: PIL Image
            detection_results: List of DetectionResult objects from DINO
            polygon_refinement: Whether to apply polygon refinement
            
        Returns:
            List of DetectionResult objects with added mask information
        """
        try:
            if not detection_results:
                return detection_results
                
            print("Starting SAM segmentation...")
            boxes = self.get_boxes(detection_results)
            print(f"SAM input boxes format: {boxes}")
            inputs = self.processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.device)

            outputs = self.segmentator(**inputs)
            masks = self.processor.post_process_masks(
                masks=outputs.pred_masks,
                original_sizes=inputs.original_sizes,
                reshaped_input_sizes=inputs.reshaped_input_sizes
            )[0]

            masks = self.refine_masks(masks, polygon_refinement)

            # Add masks to detection results
            for detection_result, mask in zip(detection_results, masks):
                detection_result.mask = mask

            print(f"SAM segmentation completed for {len(detection_results)} objects")
            return detection_results
            
        except Exception as e:
            print(f"Error in SAM segmentation: {e}")
            return detection_results

    def grounded_segmentation(
        self,
        image: Union[Image.Image, str],
        labels: List[str],
        threshold: float = 0.3,
        polygon_refinement: bool = False,
        detector_id: Optional[str] = None,
        segmenter_id: Optional[str] = None
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        """
        Complete grounded segmentation pipeline using DINO + SAM.
        
        Args:
            image: PIL Image or path/URL to image
            labels: List of labels to detect
            threshold: Detection threshold
            polygon_refinement: Whether to apply polygon refinement
            detector_id: Optional detector model ID
            segmenter_id: Optional segmenter model ID
            
        Returns:
            Tuple of (image_array, detection_results_with_masks)
        """
        if isinstance(image, str):
            image = self.dino_module.load_image(image)

        detections = self.detect(image, labels, threshold, detector_id)
        # detections = self.segment(image, detections, polygon_refinement)

        return np.array(image), detections
    
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
            detection_labels = ["person", "door"]
            detections = self.dino_module.detect(pil_image, detection_labels, 0.3)
            
            # Convert DetectionResult objects to dictionaries for JSON serialization
            detections_dict = []
            for detection in detections:
                det_dict = detection.to_dict()
                # # Convert mask to list if present for JSON serialization
                # if 'mask' in det_dict and det_dict['mask'] is not None:
                #     det_dict['mask_shape'] = det_dict['mask'].shape
                #     det_dict['mask'] = 'mask_present'  # Don't serialize the actual mask data
                detections_dict.append(det_dict)

            # print( "Detections:")

            print(detections_dict)

            # Prepare processing results
            processing_results = {
                'detections': detections_dict
            }

            print( "Processing results:")
            print(processing_results)

            try:
                depth_module = self.makeDepthMap(detections_dict, image_rgb.shape)
                if depth_module is not None:
                    # Store the depth module for potential future use
                    self.depth_module = depth_module
                    print("Depth map processing completed successfully")
                else:
                    print("Depth map processing failed - no depth module created")
            except Exception as e:
                print(f"Error in makeDepthMap: {e}")
                return {
                    'error': f'Depth map creation failed: {str(e)}',
                    'processing_status': 'failed',
                    'vision_pipeline_version': self.version,
                    'processing_timestamp': datetime.utcnow().isoformat()
                }
            
            return processing_results
            
        except Exception as e:
            return {
                'error': f'Vision processing failed: {str(e)}',
                'processing_status': 'failed',
                'vision_pipeline_version': self.version,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
    
    def makeDepthMap(self, detections_dict, image_shape):
        print(f"Image shape: {image_shape}")
        print(f"Type of detections_dict: {type(detections_dict)}")
        
        # Extract height and width from image shape
        height, width = image_shape[:2]
        
        # Create a binary mask initialized with zeros
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill mask areas where detections are found
        for detection in detections_dict:
            if 'box' in detection:
                box = detection['box']
                xmin = int(box['xmin'])
                ymin = int(box['ymin'])
                xmax = int(box['xmax'])
                ymax = int(box['ymax'])
                
                # Ensure coordinates are within image bounds
                xmin = max(0, min(xmin, width-1))
                ymin = max(0, min(ymin, height-1))
                xmax = max(0, min(xmax, width-1))
                ymax = max(0, min(ymax, height-1))
                
                # Fill the bounding box area 
                mask[ymin:ymax, xmin:xmax] = 1
                
                print(f"Added detection mask for {detection.get('label', 'unknown')} at ({xmin},{ymin})-({xmax},{ymax})")
        
        # Create a placeholder depth map (you can replace this with actual depth estimation)
        # For now, create a simple depth map with gradient
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        # Create a simple depth gradient (closer objects have higher values)
        for i in range(height):
            for j in range(width):

                center_x, center_y = width // 2, height // 2
                distance_from_center = np.sqrt((j - center_x)**2 + (i - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                depth_map[i, j] = 1.0 - (distance_from_center / max_distance)
        
        print(f"Created mask with shape: {mask.shape}")
        print(f"Created depth map with shape: {depth_map.shape}")
        print(f"Mask has {np.sum(mask > 0)} non-zero pixels")
        

        try:
            depth_module = DepthModule(depth_map, mask)
            print("DepthModule initialized successfully")
        
            
        except Exception as e:
            print(f"Error initializing DepthModule: {e}")
            return None
        
        try:
            buzz_values = depth_module.get_buzz_values()
            print(f"Depth buzz values: {buzz_values}")
            return buzz_values
        except Exception as e:
            print(f"Error getting buzz values: {e}")
            return None
        
    def get_depth_module(self):
        """Get the current depth module if available."""
        return getattr(self, 'depth_module', None)
    
    def get_depth_buzz_values(self):
        """Get buzz values from the depth module if available."""
        depth_module = self.get_depth_module()
        if depth_module is not None:
            return depth_module.get_buzz_valus()
        return None