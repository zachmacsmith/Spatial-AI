"""
Visualization - Optional rendering of overlays, bounding boxes, and relationship lines

Separated from core processing so videos can be generated independently.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import colorsys


def get_color_for_class(class_name: str, all_class_names: set) -> Tuple[int, int, int]:
    """
    Get consistent color for object class using golden ratio hashing.
    
    Args:
        class_name: Name of the object class
        all_class_names: Set of all class names (for consistent hashing)
    
    Returns:
        BGR color tuple
    """
    if not hasattr(get_color_for_class, 'color_map'):
        get_color_for_class.color_map = {}
    
    if class_name not in get_color_for_class.color_map:
        # Generate color using HSV with golden ratio
        num_classes = len(all_class_names)
        class_idx = list(all_class_names).index(class_name) if class_name in all_class_names else len(get_color_for_class.color_map)
        hue = (class_idx * 0.618033988749895) % 1.0  # Golden ratio
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        # Convert to BGR for OpenCV
        get_color_for_class.color_map[class_name] = tuple(int(c * 255) for c in rgb[::-1])
    
    return get_color_for_class.color_map[class_name]


def _get_font_scale(target_height_px: int, thickness: int = 2) -> float:
    """
    Calculate OpenCV font scale to match target pixel height.
    
    Approximate for FONT_HERSHEY_SIMPLEX.
    Base height is ~30px at scale 1.0.
    """
    base_height = 30
    return target_height_px / base_height


def overlay_action_label(
    frame: np.ndarray,
    label: str,
    font_size: int = 200,
    position: Tuple[int, int] = (30, 100),
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Overlay action label text on frame.
    
    Args:
        frame: Input frame (BGR)
        label: Text label to overlay
        font_size: Font size in pixels
        position: (x, y) position for text
        color: RGB color tuple (note: will be converted to BGR)
    
    Returns:
        Frame with text overlay
    """
    result = frame.copy()
    
    # Calculate font scale
    font_scale = _get_font_scale(font_size)
    thickness = max(2, int(font_scale * 2))
    
    # Draw text with outline for better visibility
    cv2.putText(
        result, 
        label, 
        position, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        (0, 0, 0),  # Black outline
        thickness + 4, 
        cv2.LINE_AA
    )
    
    cv2.putText(
        result, 
        label, 
        position, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        color[::-1],  # Convert RGB to BGR
        thickness, 
        cv2.LINE_AA
    )
    
    return result


def draw_bounding_boxes(
    frame: np.ndarray,
    detections: List[Tuple[str, float, Tuple[int, int, int, int]]],
    all_class_names: Optional[set] = None,
    font_size: int = 40
) -> np.ndarray:
    """
    Draw bounding boxes for detected objects.
    
    Args:
        frame: Input frame (BGR)
        detections: List of (class_name, confidence, (x1, y1, x2, y2))
        all_class_names: Optional set of all class names for consistent colors
        font_size: Font size for labels
    
    Returns:
        Frame with bounding boxes
    """
    if all_class_names is None:
        all_class_names = set(det[0] for det in detections)
    
    result = frame.copy()
    
    # Calculate font scale
    font_scale = _get_font_scale(font_size)
    thickness = max(1, int(font_scale * 1.5))
    
    for class_name, confidence, (x1, y1, x2, y2) in detections:
        color = get_color_for_class(class_name, all_class_names)
        label = f"{class_name} {confidence:.2f}"
        
        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            result, 
            (x1, y1 - text_h - 10), 
            (x1 + text_w, y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            result,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255), # White text
            thickness,
            cv2.LINE_AA
        )
        
    return result


def draw_bounding_boxes_from_yolo_results(
    frame: np.ndarray,
    yolo_results,
    all_class_names: Optional[set] = None,
    font_size: int = 40
) -> np.ndarray:
    """
    Draw bounding boxes from YOLO results object.
    
    Args:
        frame: Input frame (BGR)
        yolo_results: YOLO results object from ultralytics
        all_class_names: Optional set of all class names for consistent colors
        font_size: Font size for labels
    
    Returns:
        Frame with bounding boxes
    """
    if yolo_results is None:
        return frame
    
    if all_class_names is None:
        all_class_names = set(yolo_results.names.values())
    
    result = frame.copy()
    
    # Calculate font scale
    font_scale = _get_font_scale(font_size)
    thickness = max(1, int(font_scale * 1.5))
    
    for box, cls_id, conf in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls, yolo_results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        class_name = yolo_results.names[int(cls_id)]
        color = get_color_for_class(class_name, all_class_names)
        label = f"{class_name} {conf:.2f}"
        
        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            result, 
            (x1, y1 - text_h - 10), 
            (x1 + text_w, y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            result,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255), # White text
            thickness,
            cv2.LINE_AA
        )
    
    return result


def draw_relationship_lines(
    frame: np.ndarray,
    line_info: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 5
) -> np.ndarray:
    """
    Draw lines showing object relationships.
    
    Args:
        frame: Input frame (BGR)
        line_info: List of (center1, center2) tuples
        color: BGR color for lines (default: yellow)
        thickness: Line thickness
    
    Returns:
        Frame with relationship lines
    """
    frame_copy = frame.copy()
    
    for center1, center2 in line_info:
        pt1 = (int(center1[0]), int(center1[1]))
        pt2 = (int(center2[0]), int(center2[1]))
        
        # Draw line
        cv2.line(frame_copy, pt1, pt2, color, thickness)
        
        # Draw circles at centers
        cv2.circle(frame_copy, pt1, 10, color, -1)
        cv2.circle(frame_copy, pt2, 10, color, -1)
    
    return frame_copy


def create_visualization(
    frame: np.ndarray,
    action_label: Optional[str] = None,
    detections: Optional[List[Tuple[str, float, Tuple[int, int, int, int]]]] = None,
    yolo_results = None,
    relationship_lines: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    batch_params = None
) -> np.ndarray:
    """
    Create complete visualization with all enabled overlays.
    
    Args:
        frame: Input frame (BGR)
        action_label: Optional action label to overlay
        detections: Optional list of detections for bounding boxes
        yolo_results: Optional YOLO results object (alternative to detections)
        relationship_lines: Optional relationship line info
        batch_params: BatchParameters to check what to draw
    
    Returns:
        Frame with all enabled visualizations
    """
    result = frame.copy()
    
    # Determine font size
    font_size = 200
    if batch_params:
        # Use a slightly smaller font for boxes than the main action label
        # e.g. 40% of the main label size, but at least 20
        font_size = max(20, int(batch_params.overlay_font_size * 0.4))
    
    # Draw bounding boxes if enabled
    if batch_params and batch_params.draw_bounding_boxes:
        if yolo_results is not None:
            result = draw_bounding_boxes_from_yolo_results(result, yolo_results, font_size=font_size)
        elif detections is not None:
            result = draw_bounding_boxes(result, detections, font_size=font_size)
    
    # Draw relationship lines if enabled
    if batch_params and batch_params.draw_relationship_lines and relationship_lines:
        result = draw_relationship_lines(result, relationship_lines)
    
    # Draw action label if enabled
    if batch_params and batch_params.draw_action_labels and action_label:
        result = overlay_action_label(result, action_label, font_size=batch_params.overlay_font_size)
    
    return result
