"""
Computer Vision Service - CV Model Abstraction

Supports multiple CV models:
- YOLO (current weights)
- YOLO v8
- YOLO v9
- Custom CV models

Separated from LLM service for clean architecture.
"""

import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
from threading import local
import cv2


# Thread-local storage for models
_thread_local = local()


class CVService(ABC):
    """Abstract base class for computer vision models"""
    
    @abstractmethod
    def detect_objects(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.6
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Detect objects in frame.
        
        Returns:
            List of (class_name, confidence, (x1, y1, x2, y2))
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return model name for logging/tracking"""
        pass


class YOLOService(CVService):
    """YOLO Computer Vision Service"""
    
    def __init__(self, model_path: str = "weights.pt", model_version: str = "current"):
        self.model_path = model_path
        self.model_version = model_version
        self._model = None
    
    def _get_model(self):
        """Thread-safe model loading"""
        if not hasattr(_thread_local, "yolo_model") or _thread_local.yolo_model is None:
            from ultralytics import YOLO
            _thread_local.yolo_model = YOLO(self.model_path)
        return _thread_local.yolo_model
    
    def detect_objects(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.6
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Detect objects using YOLO"""
        model = self._get_model()
        results = model(frame, conf=confidence_threshold)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append((class_name, confidence, (x1, y1, x2, y2)))
        
        return detections
    
    def get_results_object(self, frame: np.ndarray, confidence_threshold: float = 0.6):
        """
        Get raw YOLO results object (for visualization).
        Returns ultralytics Results object.
        """
        model = self._get_model()
        results = model(frame, conf=confidence_threshold)
        return results[0] if results else None
    
    def get_model_name(self) -> str:
        return f"yolo_{self.model_version}"
    
    def get_class_names(self) -> dict:
        """Get all class names from model"""
        model = self._get_model()
        return model.names


def get_cv_service(batch_params) -> CVService:
    """
    Factory function to get CV service based on BatchParameters
    
    Args:
        batch_params: BatchParameters instance
    
    Returns:
        CVService instance for the configured model
    """
    from ..batch_parameters import CVModel
    
    if batch_params.cv_model in [CVModel.YOLO_CURRENT, CVModel.YOLO_V8, CVModel.YOLO_V9]:
        # Extract version from enum
        version = batch_params.cv_model.value.replace("yolo_", "")
        return YOLOService(
            model_path=batch_params.cv_model_path,
            model_version=version
        )
    elif batch_params.cv_model == CVModel.CUSTOM_CV:
        raise NotImplementedError("Custom CV models not yet implemented")
    else:
        raise ValueError(f"Unsupported CV model: {batch_params.cv_model}")
