"""
Tool Detector - Tool detection with registry pattern

Supports multiple detection methods:
- LLM direct (ask LLM which tool)
- LLM with context (provide detected objects)
- CV inference (infer from CV detections)
- Hybrid (combine LLM + CV)
- Custom methods (extensible via registry)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from difflib import get_close_matches


# Registry for tool detection methods
TOOL_DETECTION_REGISTRY = {}


def register_tool_detector(name: str):
    """Decorator to register tool detection methods"""
    def decorator(func):
        TOOL_DETECTION_REGISTRY[name] = func
        return func
    return decorator


@register_tool_detector("llm_direct")
def detect_tool_llm_direct(
    frames: List[np.ndarray],
    batch_params,
    llm_service,
    prompt_builder,
    detected_objects: Optional[List[Tuple[str, float]]] = None
) -> Optional[str]:
    """
    Ask LLM directly which tool is being used.
    Original TestingClass.py approach.
    """
    # Build prompt
    prompt = prompt_builder.build_tool_detection_prompt(detected_objects=None)
    
    # Send to LLM
    response = llm_service.send_multiframe_prompt(
        frames=frames,
        prompt_text=prompt,
        max_tokens=50,
        temperature=batch_params.llm_temperature
    )
    
    # Parse response
    text = response.lower()
    
    # Try exact match
    if text in batch_params.allowed_tools:
        return text
    
    # Try fuzzy match
    match = get_close_matches(text, batch_params.allowed_tools, n=1)
    if match:
        return match[0]
    
    # Try substring match
    for tool in batch_params.allowed_tools:
        if tool in text:
            return tool
    
    return "unknown"


@register_tool_detector("llm_with_context")
def detect_tool_llm_with_context(
    frames: List[np.ndarray],
    batch_params,
    llm_service,
    prompt_builder,
    detected_objects: Optional[List[Tuple[str, float]]] = None
) -> Optional[str]:
    """
    Ask LLM which tool, but provide detected objects as context.
    """
    # Build prompt with object context
    prompt = prompt_builder.build_tool_detection_prompt(detected_objects=detected_objects)
    
    # Send to LLM
    response = llm_service.send_multiframe_prompt(
        frames=frames,
        prompt_text=prompt,
        max_tokens=50,
        temperature=batch_params.llm_temperature
    )
    
    # Parse response (same as direct)
    text = response.lower()
    
    if text in batch_params.allowed_tools:
        return text
    
    match = get_close_matches(text, batch_params.allowed_tools, n=1)
    if match:
        return match[0]
    
    for tool in batch_params.allowed_tools:
        if tool in text:
            return tool
    
    return "unknown"


@register_tool_detector("cv_inference")
def detect_tool_cv_inference(
    frames: List[np.ndarray],
    batch_params,
    llm_service,
    prompt_builder,
    detected_objects: Optional[List[Tuple[str, float]]] = None
) -> Optional[str]:
    """
    Infer tool from CV detections.
    Current TestingClass2/3/Integrated/FINAL approach.
    
    Looks for objects that aren't person/hand/safety equipment.
    """
    if not detected_objects:
        return None
    
    # Exclude common non-tool objects
    excluded_objects = {"person", "hand", "hardhat", "safety vest", "glove"}
    
    # Find first object that's not excluded
    for obj_name, confidence in detected_objects:
        if obj_name.lower() not in excluded_objects:
            # Check if it's in allowed tools
            if obj_name.lower() in [t.lower() for t in batch_params.allowed_tools]:
                return obj_name.lower()
            # Otherwise return it anyway (might be a tool we didn't list)
            return obj_name.lower()
    
    return None


@register_tool_detector("hybrid")
def detect_tool_hybrid(
    frames: List[np.ndarray],
    batch_params,
    llm_service,
    prompt_builder,
    detected_objects: Optional[List[Tuple[str, float]]] = None
) -> Optional[str]:
    """
    Hybrid approach: Try CV first, fall back to LLM if uncertain.
    """
    # Try CV inference first
    cv_tool = detect_tool_cv_inference(
        frames, batch_params, llm_service, prompt_builder, detected_objects
    )
    
    # If CV found a tool with high confidence, use it
    if cv_tool and detected_objects:
        # Check if top detection is a tool
        top_obj = detected_objects[0]
        if top_obj[0].lower() == cv_tool and top_obj[1] > 0.8:
            return cv_tool
    
    # Otherwise ask LLM
    return detect_tool_llm_with_context(
        frames, batch_params, llm_service, prompt_builder, detected_objects
    )


def detect_tool(
    frames: List[np.ndarray],
    batch_params,
    llm_service,
    prompt_builder,
    detected_objects: Optional[List[Tuple[str, float]]] = None
) -> Optional[str]:
    """
    Main dispatcher for tool detection.
    
    Args:
        frames: List of frames to analyze
        batch_params: BatchParameters instance
        llm_service: LLM service instance
        prompt_builder: PromptBuilder instance
        detected_objects: Optional list of (object_name, confidence)
    
    Returns:
        Detected tool name or None
    """
    if not batch_params.enable_tool_detection:
        return None
    
    method = batch_params.tool_detection_method.value
    
    if method not in TOOL_DETECTION_REGISTRY:
        raise ValueError(f"Unknown tool detection method: {method}")
    
    detector_func = TOOL_DETECTION_REGISTRY[method]
    
    return detector_func(
        frames=frames,
        batch_params=batch_params,
        llm_service=llm_service,
        prompt_builder=prompt_builder,
        detected_objects=detected_objects
    )
