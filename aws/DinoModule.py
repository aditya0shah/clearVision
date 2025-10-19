from transformers import pipeline
import torch
from PIL import Image
from typing import List, Dict, Any


class DetectionResult:
    """Class to represent detection results with box and mask information."""
    
    def __init__(self, label: str, score: float, box: Dict[str, float]):
        self.label = label
        self.score = score
        self.box = box
    
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
        return {
            'label': self.label,
            'score': self.score,
            'box': self.box
        }


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


    def detect(
        self, 
        image: Image.Image, 
        labels: List[str], 
        threshold: float = 0.3
    ) -> List[DetectionResult]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        
        Args:
            image: PIL Image to process
            labels: List of labels to detect
            threshold: Detection confidence threshold
            
        Returns:
            List of DetectionResult objects
        """
        try:
            # Format labels (add period if not present)
            formatted_labels = [label if label.endswith(".") else label+"." for label in labels]
            results = self.object_detector(image, candidate_labels=formatted_labels, threshold=threshold)
            
            # Convert results to DetectionResult objects
            return [DetectionResult.from_dict(result) for result in results]
            
        except Exception as e:
            print(f"Error in Grounding DINO detection: {e}")
            return []
