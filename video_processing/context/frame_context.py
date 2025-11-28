from dataclasses import dataclass, field
from typing import List, Tuple, Optional, FrozenSet, Any

@dataclass
class Detection:
    """Single object detection"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    
@dataclass
class FrameContext:
    """Everything known about a single frame"""
    frame_number: int
    timestamp: float  # seconds from video start
    
    # Raw data
    detections: List[Detection] = field(default_factory=list)
    motion_score: Optional[float] = None
    raw_results: Any = None  # Cached YOLO results object (optional)
    
    # Relationships (computed from detections)
    relationships: List[FrozenSet[str]] = field(default_factory=list)
    
    # Classification result (filled after LLM call)
    action: Optional[str] = None
    tool: Optional[str] = None
    tool_guess: Optional[str] = None      # NEW: Freeform guess when tool="unknown"
    api_calls_used: int = 0               # NEW: Track API cost per frame
    
    @property
    def object_names(self) -> List[str]:
        return [d.class_name for d in self.detections]
    
    @property
    def tools_detected(self) -> List[str]:
        non_tools = {"person", "hand", "hardhat", "safety vest", "glove"}
        return [d.class_name for d in self.detections if d.class_name.lower() not in non_tools]
