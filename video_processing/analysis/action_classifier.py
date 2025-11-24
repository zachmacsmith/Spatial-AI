"""
Action Classifier - Action classification with registry pattern

Supports multiple classification methods:
- LLM multiframe (current approach)
- LLM singleframe
- CV-based
- Hybrid
- Custom methods (extensible via registry)
"""

from typing import List, Dict, Optional
import numpy as np
from difflib import get_close_matches


# Registry for action classification methods
ACTION_CLASSIFICATION_REGISTRY = {}


def register_action_classifier(name: str):
    """Decorator to register action classification methods"""
    def decorator(func):
        ACTION_CLASSIFICATION_REGISTRY[name] = func
        return func
    return decorator


@register_action_classifier("llm_multiframe")
def classify_action_llm_multiframe(
    frames: List[np.ndarray],
    batch_params,
    llm_service,
    prompt_builder,
    motion_score: Optional[float] = None,
    detected_objects: Optional[List] = None
) -> str:
    """
    Classify action using LLM with multiple frames.
    This is the current/standard approach.
    """
    # Build prompt
    prompt = prompt_builder.build_action_classification_prompt(
        motion_score=motion_score,
        detected_objects=detected_objects
    )
    
    # Send to LLM
    response = llm_service.send_multiframe_prompt(
        frames=frames,
        prompt_text=prompt,
        max_tokens=batch_params.llm_max_tokens,
        temperature=batch_params.llm_temperature
    )
    
    # Parse response
    text = response.lower()
    
    # Try exact match first
    if text in batch_params.allowed_actions:
        return text
    
    # Try fuzzy match
    match = get_close_matches(text, batch_params.allowed_actions, n=1)
    if match:
        return match[0]
    
    # Try substring match
    for action in batch_params.allowed_actions:
        if action in text:
            return action
    
    # Fallback to first action
    return batch_params.allowed_actions[0]


@register_action_classifier("llm_singleframe")
def classify_action_llm_singleframe(
    frames: List[np.ndarray],
    batch_params,
    llm_service,
    prompt_builder,
    motion_score: Optional[float] = None,
    detected_objects: Optional[List] = None
) -> str:
    """
    Classify action using LLM with single frame (middle frame).
    Faster but potentially less accurate.
    """
    # Use middle frame only
    middle_frame = frames[len(frames) // 2]
    
    # Build prompt
    prompt = prompt_builder.build_action_classification_prompt(
        motion_score=motion_score,
        detected_objects=detected_objects
    )
    
    # Send to LLM
    response = llm_service.send_multiframe_prompt(
        frames=[middle_frame],
        prompt_text=prompt,
        max_tokens=batch_params.llm_max_tokens,
        temperature=batch_params.llm_temperature
    )
    
    # Parse response (same as multiframe)
    text = response.lower()
    
    if text in batch_params.allowed_actions:
        return text
    
    match = get_close_matches(text, batch_params.allowed_actions, n=1)
    if match:
        return match[0]
    
    for action in batch_params.allowed_actions:
        if action in text:
            return action
    
    return batch_params.allowed_actions[0]


@register_action_classifier("cv_based")
def classify_action_cv_based(
    frames: List[np.ndarray],
    batch_params,
    llm_service,
    prompt_builder,
    motion_score: Optional[float] = None,
    detected_objects: Optional[List] = None
) -> str:
    """
    Classify action based on CV features (motion + objects).
    No LLM required - faster but simpler logic.
    """
    # Simple heuristic based on motion and objects
    if motion_score is not None and motion_score < batch_params.motion_score_threshold_idle:
        return "idle"
    
    # Check if tool-like objects detected
    if detected_objects:
        tool_objects = [obj for obj in detected_objects if obj[0] not in ["person", "hand", "hardhat", "safety vest"]]
        if tool_objects:
            return "using tool"
    
    # Default to moving if motion detected
    if motion_score is not None and motion_score >= batch_params.motion_score_threshold_idle:
        return "moving"
    
    return "idle"


@register_action_classifier("hybrid")
def classify_action_hybrid(
    frames: List[np.ndarray],
    batch_params,
    llm_service,
    prompt_builder,
    motion_score: Optional[float] = None,
    detected_objects: Optional[List] = None
) -> str:
    """
    Hybrid approach: Use CV for simple cases, LLM for complex cases.
    """
    # If clearly idle (low motion, no objects), skip LLM
    if motion_score is not None and motion_score < 0.05 and not detected_objects:
        return "idle"
    
    # Otherwise use LLM
    return classify_action_llm_multiframe(
        frames, batch_params, llm_service, prompt_builder,
        motion_score, detected_objects
    )


def classify_action(
    frames: List[np.ndarray],
    batch_params,
    llm_service,
    prompt_builder,
    motion_score: Optional[float] = None,
    detected_objects: Optional[List] = None
) -> str:
    """
    Main dispatcher for action classification.
    
    Args:
        frames: List of frames to analyze
        batch_params: BatchParameters instance
        llm_service: LLM service instance
        prompt_builder: PromptBuilder instance
        motion_score: Optional motion score
        detected_objects: Optional list of detected objects
    
    Returns:
        Classified action (one of batch_params.allowed_actions)
    """
    method = batch_params.action_classification_method.value
    
    if method not in ACTION_CLASSIFICATION_REGISTRY:
        raise ValueError(f"Unknown action classification method: {method}")
    
    classifier_func = ACTION_CLASSIFICATION_REGISTRY[method]
    
    return classifier_func(
        frames=frames,
        batch_params=batch_params,
        llm_service=llm_service,
        prompt_builder=prompt_builder,
        motion_score=motion_score,
        detected_objects=detected_objects
    )


def apply_temporal_smoothing(
    frame_labels: List[str],
    allowed_actions: List[str],
    window_size: int = 9
) -> List[str]:
    """
    Apply temporal smoothing to reduce jitter in classifications.
    
    For each frame, looks at surrounding frames and picks most common action.
    
    Args:
        frame_labels: List of action labels (one per frame)
        allowed_actions: List of allowed action names
        window_size: Size of smoothing window
    
    Returns:
        Smoothed list of action labels
    """
    frame_count = len(frame_labels)
    smoothed_labels = frame_labels.copy()
    
    for i in range(frame_count):
        start = max(0, i - window_size)
        end = min(frame_count, i + window_size)
        
        # Count actions in window
        counts = {action: 0 for action in allowed_actions}
        for j in range(start, end):
            label = frame_labels[j]
            if label in counts:
                counts[label] += 1
        
        # Pick most common
        if sum(counts.values()) > 0:
            smoothed_labels[i] = max(counts, key=counts.get)
        else:
            smoothed_labels[i] = allowed_actions[0]  # Fallback
    
    return smoothed_labels
