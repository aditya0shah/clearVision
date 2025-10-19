from transformers import pipeline
import torch
from PIL import Image
from typing import List, Dict, Any, Optional
import requests


class DetectionResult:
    """Class to represent detection results with box and mask information."""
    
    def __init__(self, label: str, score: float, box: Dict[str, float], mask: Optional[Any] = None):
        self.label = label
        self.score = score
        self.box = box
        self.mask = mask
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionResult':
        """Create DetectionResult from dictionary."""
        return cls(
            label=data['label'],
            score=data['score'],
            box=data['box']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'label': self.label,
            'score': self.score,
            'box': self.box
        }
        if self.mask is not None:
            result['mask'] = self.mask
        return result


class DinoModule:
    """
    Grounding DINO module for zero-shot object detection.
    """
    
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny"):
        """
        Initialize the DINO module.
        
        Args:
            model_id: Hugging Face model ID for Grounding DINO
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.object_detector = pipeline(
            model=model_id, 
            task="zero-shot-object-detection", 
            device=self.device
        )
        print(f"DINO module initialized with model: {model_id}")
        print(f"Device: {self.device}")

    def load_image(self, image_str: str) -> Image.Image:
        """Load image from URL or file path."""
        if image_str.startswith("http"):
            image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_str).convert("RGB")
        return image

    def detect(
        self, 
        image: Image.Image, 
        labels: List[str], 
        threshold: float = 0.3, 
        detector_id: Optional[str] = None
    ) -> List[DetectionResult]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        
        Args:
            image: PIL Image to process
            labels: List of labels to detect
            threshold: Detection confidence threshold
            detector_id: Optional model ID (uses default if None)
            
        Returns:
            List of DetectionResult objects
        """
        try:
            # Format labels (add period if not present)
            formatted_labels = [label if label.endswith(".") else label+"." for label in labels]

            print("Entering object detection pipeline")
            results = self.object_detector(image, candidate_labels=formatted_labels, threshold=threshold)
            print("Done with pipeline")
            
            # Convert results to DetectionResult objects
            detection_results = []
            for result in results:
                detection_results.append(DetectionResult.from_dict(result))
            
            return detection_results
            
        except Exception as e:
            print(f"Error in Grounding DINO detection: {e}")
            return []

    def get_boxes(self, results: List[DetectionResult]) -> List[List[List[float]]]:
        """Extract bounding boxes from detection results for SAM."""
        boxes = []
        for result in results:
            xyxy = result.box
            # Convert to SAM format: each detection should be a list of boxes
            # SAM expects: List[List[List[float]]] where each inner list is [x1, y1, x2, y2]
            # But we need to group all boxes for a single image
            boxes.append([xyxy['xmin'], xyxy['ymin'], xyxy['xmax'], xyxy['ymax']])
        return [boxes]  # Wrap in another list for single image

    def detect_and_format_for_sam(
        self, 
        image: Image.Image, 
        labels: List[str], 
        threshold: float = 0.3
    ) -> tuple[List[DetectionResult], List[List[List[float]]]]:
        """
        Detect objects and format results for SAM segmentation.
        
        Returns:
            Tuple of (detection_results, boxes_for_sam)
        """
        detections = self.detect(image, labels, threshold)
        boxes = self.get_boxes(detections)
        return detections, boxes