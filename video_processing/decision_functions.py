"""
Decision Functions
Pluggable logic for making classification decisions.
"""
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass
import numpy as np

from .context import FrameContext, ContextStore, Detection, DecisionContext
from .batch_parameters import BatchParameters
from .analysis.action_mapper import ActionMapper
from .analysis.aggregation_helpers import ensure_cv_for_range, aggregate_detections

# Instantiate global mapper
ACTION_MAPPER = ActionMapper()



# ==========================================
# REGISTRIES
# ==========================================

STATE_CHECK_REGISTRY: Dict[str, Callable[[DecisionContext], str]] = {}
OBJECT_CHECK_REGISTRY: Dict[str, Callable[[DecisionContext], str]] = {}
UNKNOWN_OBJECT_CHECK_REGISTRY: Dict[str, Callable[[DecisionContext], str]] = {}

def register_state_check(name: str):
    def decorator(func):
        STATE_CHECK_REGISTRY[name] = func
        return func
    return decorator

def get_state_check(name: str) -> Callable[[DecisionContext], str]:
    if name not in STATE_CHECK_REGISTRY:
        raise ValueError(f"Unknown state check: {name}. Available: {list(STATE_CHECK_REGISTRY.keys())}")
    return STATE_CHECK_REGISTRY[name]

def register_object_check(name: str):
    def decorator(func):
        OBJECT_CHECK_REGISTRY[name] = func
        return func
    return decorator

def get_object_check(name: str) -> Callable[[DecisionContext], str]:
    if name not in OBJECT_CHECK_REGISTRY:
        raise ValueError(f"Unknown object check: {name}. Available: {list(OBJECT_CHECK_REGISTRY.keys())}")
    return OBJECT_CHECK_REGISTRY[name]

def register_unknown_object_check(name: str):
    def decorator(func):
        UNKNOWN_OBJECT_CHECK_REGISTRY[name] = func
        return func
    return decorator

def get_unknown_object_check(name: str) -> Callable[[DecisionContext], str]:
    if name not in UNKNOWN_OBJECT_CHECK_REGISTRY:
        raise ValueError(f"Unknown unknown object check: {name}. Available: {list(UNKNOWN_OBJECT_CHECK_REGISTRY.keys())}")
    return UNKNOWN_OBJECT_CHECK_REGISTRY[name]

# ==========================================
# STATE CHECK IMPLEMENTATIONS
# ==========================================

@register_state_check("motion_threshold")
def state_check_motion_threshold(ctx: DecisionContext) -> str:
    """
    Determine state based on motion score and tool detections.
    0 API calls.
    """
    motion = ctx.frame_context.motion_score
    threshold = ctx.batch_params.motion_score_threshold_idle
    
    if motion is None:
        return "uncertain"
        
    # Check for tools
    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
    has_tools = any(d.class_name not in ignored for d in ctx.frame_context.detections)
    
    if motion < threshold:
        return "idle"
    elif has_tools:
        return "using tool"
    else:
        return "moving"

@register_state_check("llm_direct")
def state_check_llm_direct(ctx: DecisionContext) -> str:
    """
    Ask LLM directly.
    1 API call.
    """
    actions_str = ", ".join([f"'{a}'" for a in ctx.batch_params.allowed_actions])
    prompt = (
        f"This is a POV of a construction worker.\n\n"
        f"{ctx.context_text}\n\n"
        f"CRITICAL INSTRUCTION: Identify the worker's action.\n"
        f"You must respond with EXACTLY ONE of the following words: {actions_str}.\n"
        f"Do NOT provide any reasoning, preamble, or extra text.\n"
        f"Response:"
    )
    response = ctx.call_llm(prompt, valid_options=ctx.batch_params.allowed_actions, log_label="State Check (Direct)").lower()
    
    if "using tool" in response or "using_tool" in response:
        return "using tool"
    elif "moving" in response:
        return "moving"
    elif "idle" in response:
        return "idle"
    else:
        return "uncertain"

@register_state_check("hybrid_motion_then_llm")
def state_check_hybrid(ctx: DecisionContext) -> str:
    """
    Use motion for obvious cases, LLM for edge cases.
    0-1 API calls.
    """
    motion = ctx.frame_context.motion_score
    threshold = ctx.batch_params.motion_score_threshold_idle
    
    # Check for tools
    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
    has_tools = any(d.class_name not in ignored for d in ctx.frame_context.detections)
    
    if motion is not None:
        # Very low motion -> idle
        if motion < (threshold * 0.5) and not has_tools:
            return "idle"
            
        # Very high motion -> moving (if no tools)
        if motion > (threshold * 2.0) and not has_tools:
            return "moving"
            
    # Fallback to LLM
    return state_check_llm_direct(ctx)

@register_state_check("cv_objects_only")
def state_check_cv_objects(ctx: DecisionContext) -> str:
    """
    Rely heavily on CV object detection.
    0 API calls.
    """
    # Check relationships for "hand" holding "tool"
    # (Simplified: just check if tools are present for now, as relationships might be sparse)
    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
    has_tools = any(d.class_name not in ignored for d in ctx.frame_context.detections)
    
    if has_tools:
        return "using tool"
        
    motion = ctx.frame_context.motion_score
    threshold = ctx.batch_params.motion_score_threshold_idle
    
    if motion is not None and motion < threshold:
        return "idle"
    else:
        return "moving"

@register_state_check("legacy_testing_class")
def state_check_legacy(ctx: DecisionContext) -> str:
    """
    Replicate TestingClass.py logic.
    1 API call.
    """
    motion = ctx.frame_context.motion_score
    motion_str = f"{motion:.2f}" if motion is not None else "unknown"
    
    actions_str = ", ".join(ctx.batch_params.allowed_actions)
    prompt = (
        f"This is a POV of a person. Motion score: {motion_str}, "
        f"a motion score of 0-0.16 suggests the person is idle, "
        f"10 or above suggests they are moving. "
        f"Check if hands are visible and classify behavior. "
        f"Classify the action: {actions_str}. "
        f"Respond with ONE word only."
    )
    
    response = ctx.call_llm(prompt, valid_options=ctx.batch_params.allowed_actions, log_label="State Check (Legacy)").lower()
    
    if "using tool" in response or "using_tool" in response:
        return "using tool"
    elif "moving" in response:
        return "moving"
    elif "idle" in response:
        return "idle"
    else:
        return "uncertain"

@register_state_check("legacy_softened")
def state_check_legacy_softened(ctx: DecisionContext) -> str:
    """
    Replicate TestingClass.py logic but with softened prompt about idle/moving.
    1 API call.
    """
    motion = ctx.frame_context.motion_score
    motion_str = f"{motion:.2f}" if motion is not None else "unknown"
    
    actions_str = ", ".join(ctx.batch_params.allowed_actions)
    prompt = (
        f"This is a POV of a person. Motion score: {motion_str}, "
        f"a motion score of 0-0.16 suggests the person is idle, "
        f"10 or above suggests they are moving. "
        f"Check if hands or tools are visible and classify behavior. "
        f"If they appear to not be using tools, they might be idle or moving. "
        f"Classify the action: {actions_str}. "
        f"Respond with ONE word only."
    )
    
    response = ctx.call_llm(prompt, valid_options=ctx.batch_params.allowed_actions, log_label="State Check (Legacy Softened)").lower()
    
    if "using tool" in response or "using_tool" in response:
        return "using tool"
    elif "moving" in response:
        return "moving"
    elif "idle" in response:
        return "idle"
    else:
        return "uncertain"

@register_state_check("action_mapping")
def state_check_with_action_mapping(ctx: DecisionContext) -> str:
    """
    Use ActionMapper to determine action from objects, falling back to LLM.
    0-1 API calls.
    """
    # 1. Check ActionMapper for definitive actions
    detected_objects = [d.class_name for d in ctx.frame_context.detections]
    definitive_action = ACTION_MAPPER.get_definitive_action(detected_objects)
    
    if definitive_action:
        print(f"ActionMapper found definitive action: {definitive_action}")
        return definitive_action
        
    # 2. Get likely actions for hints
    likely_actions = ACTION_MAPPER.get_likely_actions(detected_objects)
    
    # Combine likely actions with allowed actions
    valid_options = list(set(ctx.batch_params.allowed_actions + likely_actions))
    actions_str = ", ".join([f"'{a}'" for a in valid_options])
    
    hint_text = ""
    if likely_actions:
        hint_text = f"Based on objects ({', '.join(detected_objects)}), likely actions are: {', '.join(likely_actions)}.\n"
    
    prompt = (
        f"This is a POV of a construction worker.\n\n"
        f"{ctx.context_text}\n\n"
        f"{hint_text}"
        f"What is the worker doing? Respond with exactly one word from: {actions_str}."
    )
    
    response = ctx.call_llm(prompt, valid_options=valid_options, log_label="State Check (Mapped)").lower()
    
    if "using tool" in response or "using_tool" in response:
        return "using tool"
    elif "moving" in response:
        return "moving"
    elif "idle" in response:
        return "idle"
    # Return mapped actions directly
    elif response in likely_actions:
        return response
    else:
        return "uncertain"

@register_state_check("llm_multiframe")
def state_check_llm_multiframe(ctx: DecisionContext) -> str:
    """
    Ask LLM with multi-frame context (before/after frames).
    1 API call.
    """
    # Prepare multi-frame context
    current_frame_num = ctx.frame_context.frame_number
    gap = ctx.batch_params.multi_frame_gap
    count = ctx.batch_params.multi_frame_count
    
    frames_to_send = []
    
    # Previous frames
    for i in range(count, 0, -1):
        fn = current_frame_num - (i * gap)
        if ctx.frame_cache:
            frame = ctx.frame_cache.get(fn)
            if frame is not None:
                frames_to_send.append(frame)
            
    # Current frame(s)
    frames_to_send.extend(ctx.frames)
    
    # Next frames
    for i in range(1, count + 1):
        fn = current_frame_num + (i * gap)
        if ctx.frame_cache:
            frame = ctx.frame_cache.get(fn)
            if frame is not None:
                frames_to_send.append(frame)
                
    # Explain context in prompt
    actions_str = ", ".join([f"'{a}'" for a in ctx.batch_params.allowed_actions])
    prompt = (
        f"This is a POV of a construction worker. You are viewing a sequence of frames.\n"
        f"The middle frame is the current moment. Previous and subsequent frames are provided for context.\n\n"
        f"{ctx.context_text}\n\n"
        f"What is the worker doing in the CURRENT (middle) frame? Respond with exactly one word from: {actions_str}."
    )
    
    # Temporarily swap frames for the call
    original_frames = ctx.frames
    ctx.frames = frames_to_send
    try:
        response = ctx.call_llm(prompt, valid_options=ctx.batch_params.allowed_actions, log_label=f"State Check (Multi-Frame {len(frames_to_send)}f)").lower()
    finally:
        ctx.frames = original_frames
    
    if "using tool" in response or "using_tool" in response:
        return "using tool"
    elif "moving" in response:
        return "moving"
    elif "idle" in response:
        return "idle"
    else:
        return "uncertain"

# ==========================================
# OBJECT CHECK IMPLEMENTATIONS
# ==========================================

@register_object_check("cv_detection")
def object_check_cv(ctx: DecisionContext) -> str:
    """
    Return first detected tool from CV, or mapped action (e.g. measuring).
    0 API calls.
    """
    # 1. Check ActionMapper for definitive actions (e.g. pencil + tape -> measuring)
    detected_objects = [d.class_name for d in ctx.frame_context.detections]
    definitive_action = ACTION_MAPPER.get_definitive_action(detected_objects)
    
    if definitive_action:
        return definitive_action

    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
    tools = [d for d in ctx.frame_context.detections if d.class_name not in ignored]
    
    if tools:
        # Sort by confidence
        tools.sort(key=lambda x: x.confidence, reverse=True)
        result = tools[0].class_name
        print(f"\n=== Object Check (CV) ===\nDetected: {result} (from {len(tools)} tools)\n==================\n")
        return result
        
    print(f"\n=== Object Check (CV) ===\nDetected: unknown (no valid tools)\n==================\n")
    return "unknown"

@register_object_check("llm_direct")
def object_check_llm(ctx: DecisionContext) -> str:
    """
    Ask LLM to identify tool.
    1 API call.
    """
    options_list = ctx.batch_params.allowed_tools
    options_str = ", ".join([f"'{opt}'" for opt in options_list])

    prompt = (
        f"This is a POV of a construction worker using a tool.\n\n"
        f"{ctx.context_text}\n\n"
        f"CRITICAL INSTRUCTION: Identify the tool being used.\n"
        f"You must respond with EXACTLY ONE of the following words: {options_str}.\n"
        f"Do NOT provide any reasoning, preamble, or extra text.\n"
        f"If you are not 100% sure, or if the tool is not in the list, respond with 'unknown'.\n"
        f"Response:"
    )
    response = ctx.call_llm(prompt, valid_options=options_list, log_label="Object Check (Direct)").lower().strip()
    return response

@register_object_check("llm_with_cv_hint")
def object_check_llm_hint(ctx: DecisionContext) -> str:
    """
    Ask LLM but provide CV detections as hint.
    1 API call.
    """
    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
    tools = [d.class_name for d in ctx.frame_context.detections if d.class_name not in ignored]
    hint = ", ".join(tools) if tools else "none"
    
    prompt = (
        f"This is a POV of a construction worker using a tool.\n\n"
        f"{ctx.context_text}\n\n"
        f"CV Hint: {hint}\n"
        f"What tool are they using? Respond with the tool name only (e.g., 'drill', 'hammer'). "
        f"If you cannot identify it, respond with 'unknown'."
    )
    response = ctx.call_llm(prompt, valid_options=ctx.batch_params.allowed_tools, log_label="Object Check (Hint)").lower().strip()
    return response

@register_object_check("llm_multiframe")
def object_check_llm_multiframe(ctx: DecisionContext) -> str:
    """
    Ask LLM to identify tool with multi-frame context.
    1 API call.
    """
    # Prepare multi-frame context
    current_frame_num = ctx.frame_context.frame_number
    gap = ctx.batch_params.multi_frame_gap
    count = ctx.batch_params.multi_frame_count
    
    frames_to_send = []
    
    # Previous frames
    for i in range(count, 0, -1):
        fn = current_frame_num - (i * gap)
        if ctx.frame_cache:
            frame = ctx.frame_cache.get(fn)
            if frame is not None:
                frames_to_send.append(frame)
            
    # Current frame(s)
    frames_to_send.extend(ctx.frames)
    
    # Next frames
    for i in range(1, count + 1):
        fn = current_frame_num + (i * gap)
        if ctx.frame_cache:
            frame = ctx.frame_cache.get(fn)
            if frame is not None:
                frames_to_send.append(frame)
    
    prompt = (
        f"This is a POV of a construction worker using a tool. You are viewing a sequence of frames.\n"
        f"The middle frame is the current moment.\n\n"
        f"{ctx.context_text}\n\n"
        f"What tool are they using in the CURRENT (middle) frame? Respond with the tool name only (e.g., 'drill', 'hammer'). "
        f"If you cannot identify it, respond with 'unknown'."
    )
    
    # Temporarily swap frames for the call
    original_frames = ctx.frames
    ctx.frames = frames_to_send
    try:
        response = ctx.call_llm(prompt, valid_options=ctx.batch_params.allowed_tools, log_label=f"Object Check (Multi-Frame {len(frames_to_send)}f)").lower().strip()
    finally:
        ctx.frames = original_frames
        
    return response

@register_object_check("cv_then_llm")
def object_check_cv_then_llm(ctx: DecisionContext) -> str:
    """
    Try CV first, then LLM.
    0-1 API calls.
    """
    tool = object_check_cv(ctx)
    if tool != "unknown":
        return tool
        
    return object_check_llm(ctx)

@register_object_check("llm_with_cv_hint")
def object_check_llm_hint(ctx: DecisionContext) -> str:
    """
    Ask LLM but provide CV detections as hint.
    1 API call.
    """
    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
    tools = [d.class_name for d in ctx.frame_context.detections if d.class_name not in ignored]
    hint = ", ".join(tools) if tools else "none"
    
    prompt = (
        f"This is a POV of a construction worker using a tool.\n\n"
        f"Object detection suggests: {hint}\n\n"
        f"{ctx.context_text}\n\n"
        f"What tool are they using? Confirm the suggestion or provide a better name. "
        f"Respond with the tool name only."
    )
    return ctx.call_llm(prompt, valid_options=ctx.batch_params.allowed_tools, log_label="Object Check (Hint)").lower().strip()

@register_object_check("llm_with_relationships")
def object_check_llm_relationships(ctx: DecisionContext) -> str:
    """
    Ask LLM, providing CV detections AND relationship context.
    1 API call.
    """
    # 1. Get CV detections
    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
    tools = [d.class_name for d in ctx.frame_context.detections if d.class_name not in ignored]
    hint = ", ".join(tools) if tools else "none"
    
    # 2. Get Relationships
    relationships_text = "none"
    if ctx.frame_context.relationships:
        rel_strings = []
        for rel_set in ctx.frame_context.relationships:
            # rel_set is a frozenset of object names
            items = list(rel_set)
            if len(items) >= 2:
                rel_strings.append(f"{items[0]} is close to {items[1]}")
        
        if rel_strings:
            relationships_text = "; ".join(rel_strings)
    
    prompt = (
        f"This is a POV of a construction worker using a tool.\n\n"
        f"Object detection suggests: {hint}\n"
        f"Spatial relationships detected: {relationships_text}\n\n"
        f"{ctx.context_text}\n\n"
        f"What tool are they using? Consider the spatial relationships (e.g., hand close to tool).\n"
        f"Respond with the tool name only."
    )
    return ctx.call_llm(prompt, valid_options=ctx.batch_params.allowed_tools, log_label="Object Check (Relationships)").lower().strip()

@register_object_check("legacy_testing_class")
def object_check_legacy(ctx: DecisionContext) -> str:
    """
    Replicate TestingClass.py logic (ask_claude_which_tool).
    1 API call.
    """
    tools_str = ", ".join(ctx.batch_params.allowed_tools)
    prompt = (
        f"This is a POV of a person using a tool.\n"
        f"What tool is being used? (e.g. {tools_str})\n"
        f"Respond with the tool name only."
    )
    return ctx.call_llm(prompt, valid_options=ctx.batch_params.allowed_tools, log_label="Object Check (Legacy)").lower().strip()

# ==========================================
# UNKNOWN OBJECT CHECK IMPLEMENTATIONS
# ==========================================

@register_unknown_object_check("llm_guess")
def unknown_check_llm(ctx: DecisionContext) -> str:
    """
    Ask LLM to guess the object.
    1 API call.
    """
    prompt = (
        f"The worker is using an object that was not automatically identified.\n"
        f"Look closely at the image.\n"
        f"What tool or object are they using? Be specific (1-4 words)."
    )
    return ctx.call_llm(prompt, log_label="Unknown Check (Guess)").lower().strip()

@register_object_check("llm_strict")
def object_check_strict(ctx: DecisionContext) -> str:
    """
    Ask LLM with extremely strict constraints.
    1 API call.
    """
    # 1. Get CV detections for context (optional, but helpful)
    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
    tools = [d.class_name for d in ctx.frame_context.detections if d.class_name not in ignored]
    hint = ", ".join(tools) if tools else "none"
    
    options_list = ctx.batch_params.allowed_tools
    options_str = ", ".join([f"'{opt}'" for opt in options_list])
    
    prompt = (
        f"This is a POV of a construction worker using a tool.\n\n"
        f"Object detection suggests: {hint}\n\n"
        f"{ctx.context_text}\n\n"
        f"CRITICAL INSTRUCTION: Identify the tool being used.\n"
        f"You must respond with EXACTLY ONE of the following words: {options_str}.\n"
        f"Do NOT provide any reasoning, preamble, or extra text.\n"
        f"If you are not 100% sure, or if the tool is not in the list, respond with 'unknown'.\n"
        f"Response:"
    )
    return ctx.call_llm(prompt, valid_options=options_list, log_label="Object Check (Strict)").lower().strip()


@register_object_check("llm_strict_softened")
def object_check_strict_softened(ctx: DecisionContext) -> str:
    """
    Ask LLM with strict constraints but softened uncertainty phrasing.
    1 API call.
    """
    # 1. Get CV detections for context (optional, but helpful)
    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
    tools = [d.class_name for d in ctx.frame_context.detections if d.class_name not in ignored]
    hint = ", ".join(tools) if tools else "none"
    
    options_list = ctx.batch_params.allowed_tools
    options_str = ", ".join([f"'{opt}'" for opt in options_list])
    
    prompt = (
        f"This is a POV of a construction worker using a tool.\n\n"
        f"Object detection suggests: {hint}\n\n"
        f"{ctx.context_text}\n\n"
        f"CRITICAL INSTRUCTION: Identify the tool being used.\n"
        f"You must respond with EXACTLY ONE of the following words: {options_str}.\n"
        f"Do NOT provide any reasoning, preamble, or extra text.\n"
        f"If the answer provides unclear, may be more than one of the tools, or a different one altogether, say 'unknown'.\n"
        f"Response:"
    )
    return ctx.call_llm(prompt, valid_options=options_list, log_label="Object Check (Strict Softened)").lower().strip()


    return ctx.call_llm(prompt, valid_options=options_list, log_label="Object Check (Strict Softened)").lower().strip()


@register_object_check("llm_aggregation_softened")
def object_check_aggregation_softened(ctx: DecisionContext) -> str:
    """
    Aggregate detections since the last keyframe, map 'saw' -> 'power saw', 
    and use softened uncertainty phrasing.
    """
    current_frame = ctx.frame_context.frame_number
    
    # Find previous keyframe to define the interval
    start_frame = 1
    for f in range(current_frame - 1, 0, -1):
        c = ctx.context_store.get(f)
        if c and c.action:
            start_frame = f
            break
            
    # Ensure minimum aggregation window for dense classification strategies
    min_lookback = 30
    if current_frame - start_frame < min_lookback:
        start_frame = max(1, current_frame - min_lookback)
            
    end_frame = current_frame
    
    ensure_cv_for_range(ctx, start_frame, end_frame)
    
    # Refresh frame_context from store
    refreshed_context = ctx.context_store.get(current_frame)
    if refreshed_context:
        ctx.frame_context = refreshed_context
    
    # 1. Current Frame Detections
    current_objects = []
    if ctx.frame_context.detections:
        ignored = {"person", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
        for d in ctx.frame_context.detections:
            if d.class_name.lower() not in ignored:
                name = d.class_name
                # Map "saw" -> "power saw"
                if name.lower() == "saw":
                    name = "power saw"
                current_objects.append(f"{name} ({d.confidence:.2f})")
    
    current_hint = ", ".join(current_objects) if current_objects else "none"

    # 2. Aggregated Detections (Previous Interval)
    agg_end = max(start_frame, end_frame - 1)
    
    if agg_end < start_frame:
        aggregated_scores = {}
    else:
        aggregated_scores = aggregate_detections(ctx, start_frame, agg_end)
    
    if aggregated_scores:
        # Sort by frequency first, then confidence
        sorted_items = sorted(aggregated_scores.items(), key=lambda x: (x[1]['frequency'], x[1]['confidence']), reverse=True)
        
        agg_hints = []
        for obj, stats in sorted_items:
            name = obj
            # Map "saw" -> "power saw"
            if name.lower() == "saw":
                name = "power saw"
                
            conf = stats['confidence']
            freq = stats['frequency']
            agg_hints.append(f"{name} ({conf:.2f} conf, {freq:.0%} of frames)")
            
        agg_hint = ", ".join(agg_hints)
    else:
        agg_hint = "none"
    
    options_list = ctx.batch_params.allowed_tools
    options_str = ", ".join([f"'{opt}'" for opt in options_list])
    
    # Filter out redundant "Objects:" line and "No objects detected." from context_text
    filtered_context = "\n".join([
        line for line in ctx.context_text.split('\n') 
        if not line.strip().startswith("Objects:") and "No objects detected." not in line
    ])
    
    prompt = (
        f"This is a POV of a construction worker using a tool.\n\n"
       
        f"CRITICAL INSTRUCTION: Identify the tool being used.\n"
        f"You must respond with EXACTLY ONE of the following words: {options_str}.\n"
        f"Do NOT provide any reasoning, preamble, or extra text.\n"
        f"If the answer provides unclear, may be more than one of the tools, or a different one altogether, say 'unknown'.\n"
        f"Objects identified in CURRENT frame: {current_hint}\n"
        f"Objects identified in PREVIOUS interval: {agg_hint}\n"
        f"{filtered_context}\n\n"
        f"Response:"
    )
    return ctx.call_llm(prompt, valid_options=options_list, log_label="Object Check (Agg Softened)").lower().strip()


@register_object_check("llm_strict_confident")
def object_check_strict_confident(ctx: DecisionContext) -> str:
    """
    Ask LLM with strict constraints and explicit confidence instruction.
    1 API call.
    """
    # 1. Get CV detections for context (optional, but helpful)
    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
    tools = [d.class_name for d in ctx.frame_context.detections if d.class_name not in ignored]
    hint = ", ".join(tools) if tools else "none"
    
    options_list = ctx.batch_params.allowed_tools
    options_str = ", ".join([f"'{opt}'" for opt in options_list])
    
    prompt = (
        f"This is a POV of a construction worker using a tool.\n\n"
        f"Object detection suggests: {hint}\n\n"
        f"{ctx.context_text}\n\n"
        f"CRITICAL INSTRUCTION: Identify the tool being used.\n"
        f"You must respond with EXACTLY ONE of the following words: {options_str}.\n"
        f"Do NOT provide any reasoning, preamble, or extra text.\n"
        f"ONLY CHOOSE A TOOL IF YOU ARE VERY CONFIDENT, and if there is ANY UNCERTAINTY select 'unknown'.\n"
        f"Response:"
    )
    return ctx.call_llm(prompt, valid_options=options_list, log_label="Object Check (Strict Confident)").lower().strip()


@register_object_check("llm_two_step_recheck")
def object_check_two_step_recheck(ctx: DecisionContext) -> str:
    """
    Two-step process:
    1. Strict check (current frame only).
    2. If unknown, Recheck with Aggregation (1 sec lookback) AND multiframe visual context.
    """
    # Step 1: Strict Check with Confidence Instruction
    result = object_check_strict_confident(ctx)
    
    if result != "unknown":
        return result
        
    # Step 2: Recheck with Aggregation AND Multiframe Context
    print(">> Step 1 returned 'unknown'. Rechecking with aggregation and multiframes...")
    
    # Fetch frames for the last 2 seconds
    current_frame = ctx.frame_context.frame_number
    fps = ctx.context_store.fps
    start_frame = max(1, int(current_frame - (2 * fps)))
    
    # Sample ~10 frames from this interval
    frame_numbers = np.linspace(start_frame, current_frame, num=10, dtype=int).tolist()
    frame_numbers = sorted(list(set(frame_numbers))) # Deduplicate and sort
    
    recheck_frames = []
    if ctx.frame_cache:
        for fn in frame_numbers:
            frame = ctx.frame_cache.get(fn)
            if frame is not None:
                recheck_frames.append(frame)
    
    # Fallback to current frame if cache miss or empty
    if not recheck_frames:
        recheck_frames = ctx.frames
        
    # We need to call object_check_aggregation_softened but pass these frames.
    # Since object_check_aggregation_softened calls ctx.call_llm internally,
    # we can't easily inject frames unless we modify that function too OR
    # we manually construct the call here.
    
    # Manually constructing the call to ensure frames are passed:
    # Reuse logic from aggregation_softened but call llm with specific frames
    
    # ... Wait, better to refactor aggregation_softened to accept optional frames?
    # Or just copy the logic since it's short?
    # Let's copy and adapt to ensure we pass the frames.
    
    # 1. Aggregation Logic (Same as aggregation_softened)
    # Find previous keyframe to define the interval
    agg_start_frame = 1
    for f in range(current_frame - 1, 0, -1):
        c = ctx.context_store.get(f)
        if c and c.action:
            agg_start_frame = f
            break
            
    # Ensure at least 2 seconds of lookback for aggregation stats too
    min_lookback = int(2 * fps)
    if current_frame - agg_start_frame < min_lookback:
        agg_start_frame = max(1, current_frame - min_lookback)
            
    end_frame = current_frame
    ensure_cv_for_range(ctx, agg_start_frame, end_frame)
    
    # Refresh frame_context
    refreshed_context = ctx.context_store.get(current_frame)
    if refreshed_context:
        ctx.frame_context = refreshed_context
    
    # Current Detections
    current_objects = []
    if ctx.frame_context.detections:
        ignored = {"person", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
        for d in ctx.frame_context.detections:
            if d.class_name.lower() not in ignored:
                name = d.class_name
                if name.lower() == "saw": name = "power saw"
                current_objects.append(f"{name} ({d.confidence:.2f})")
    current_hint = ", ".join(current_objects) if current_objects else "none"

    # Aggregated Detections
    agg_end = max(agg_start_frame, end_frame - 1)
    if agg_end < agg_start_frame:
        aggregated_scores = {}
    else:
        aggregated_scores = aggregate_detections(ctx, agg_start_frame, agg_end)
    
    if aggregated_scores:
        sorted_items = sorted(aggregated_scores.items(), key=lambda x: (x[1]['frequency'], x[1]['confidence']), reverse=True)
        agg_hints = []
        for obj, stats in sorted_items:
            name = obj
            if name.lower() == "saw": name = "power saw"
            agg_hints.append(f"{name} ({stats['confidence']:.2f} conf, {stats['frequency']:.0%} of frames)")
        agg_hint = ", ".join(agg_hints)
    else:
        agg_hint = "none"
    
    options_list = ctx.batch_params.allowed_tools
    options_str = ", ".join([f"'{opt}'" for opt in options_list])
    
    filtered_context = "\n".join([
        line for line in ctx.context_text.split('\n') 
        if not line.strip().startswith("Objects:") and "No objects detected." not in line
    ])
    
    prompt = (
        f"This is a POV of a construction worker using a tool.\n\n"
        f"CRITICAL INSTRUCTION: Identify the tool being used.\n"
        f"You must respond with EXACTLY ONE of the following words: {options_str}.\n"
        f"Do NOT provide any reasoning, preamble, or extra text.\n"
        f"If the answer provides unclear, may be more than one of the tools, or a different one altogether, say 'unknown'.\n"
        f"Objects identified in CURRENT frame: {current_hint}\n"
        f"Objects identified in PREVIOUS interval: {agg_hint}\n"
        f"{filtered_context}\n\n"
        f"Response:"
    )
    
    # Pass the sampled frames here!
    return ctx.call_llm(prompt, valid_options=options_list, log_label="Object Check (Recheck Multiframe)", frames=recheck_frames).lower().strip()


@register_object_check("llm_with_interval_aggregation")
def object_check_interval_aggregation(ctx: DecisionContext) -> str:
    """
    Aggregate detections since the last keyframe (or start of video).
    """
    current_frame = ctx.frame_context.frame_number
    
    # Find previous keyframe to define the interval
    start_frame = 1
    for f in range(current_frame - 1, 0, -1):
        c = ctx.context_store.get(f)
        if c and c.action:
            start_frame = f
            break
            
    # CRITICAL FIX: Ensure minimum aggregation window for dense classification strategies
    # If the found start_frame is too close (e.g. just 1 frame ago), look back further.
    min_lookback = 30
    if current_frame - start_frame < min_lookback:
        start_frame = max(1, current_frame - min_lookback)
            
    end_frame = current_frame
    
    ensure_cv_for_range(ctx, start_frame, end_frame)
    
    # CRITICAL FIX: Refresh frame_context from store because ensure_cv_for_range 
    # might have updated it (or created a new one if it was missing/empty).
    # The ctx.frame_context object might be stale if it was a copy or if the store replaced it.
    refreshed_context = ctx.context_store.get(current_frame)
    if refreshed_context:
        ctx.frame_context = refreshed_context
    
    # 1. Current Frame Detections
    current_objects = []
    if ctx.frame_context.detections:
        # Filter ignored (allow hands in current frame to match context text)
        ignored = {"person", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
        for d in ctx.frame_context.detections:
            if d.class_name.lower() not in ignored:
                current_objects.append(f"{d.class_name} ({d.confidence:.2f})")
    
    current_hint = ", ".join(current_objects) if current_objects else "none"

    # 2. Aggregated Detections (Previous Interval)
    # We aggregate up to current_frame - 1 to separate past context
    agg_end = max(start_frame, end_frame - 1)
    
    # print(f"[DEBUG] Interval Aggregation: Current={current_frame}, Interval=[{start_frame}, {agg_end}]")
    
    if agg_end < start_frame:
        # Interval is empty (start_frame == end_frame)
        aggregated_scores = {}
    else:
        aggregated_scores = aggregate_detections(ctx, start_frame, agg_end)
    
    if aggregated_scores:
        # Sort by frequency first, then confidence
        sorted_items = sorted(aggregated_scores.items(), key=lambda x: (x[1]['frequency'], x[1]['confidence']), reverse=True)
        
        agg_hints = []
        for obj, stats in sorted_items:
            conf = stats['confidence']
            freq = stats['frequency']
            agg_hints.append(f"{obj} ({conf:.2f} conf, {freq:.0%} of frames)")
            
        agg_hint = ", ".join(agg_hints)
    else:
        agg_hint = "none"
    
    options_list = ctx.batch_params.allowed_tools
    options_str = ", ".join([f"'{opt}'" for opt in options_list])
    
    # Filter out redundant "Objects:" line and "No objects detected." from context_text
    filtered_context = "\n".join([
        line for line in ctx.context_text.split('\n') 
        if not line.strip().startswith("Objects:") and "No objects detected." not in line
    ])
    
    prompt = (
        f"This is a POV of a construction worker using a tool.\n\n"
        f"Objects in CURRENT frame: {current_hint}\n"
        f"Objects in PREVIOUS interval: {agg_hint}\n\n"
        f"{filtered_context}\n\n"
        f"CRITICAL INSTRUCTION: Identify the tool being used.\n"
        f"You must respond with EXACTLY ONE of the following words: {options_str}.\n"
        f"Do NOT provide any reasoning, preamble, or extra text.\n"
        f"If you are not 100% sure, or if the tool is not in the list, respond with 'unknown'.\n"
        f"Response:"
    )
    return ctx.call_llm(prompt, valid_options=options_list, log_label="Object Check (Interval Agg)").lower().strip()

@register_object_check("llm_with_1sec_aggregation")
def object_check_1sec_aggregation(ctx: DecisionContext) -> str:
    """
    Aggregate detections over the past 1 second.
    """
    current_frame = ctx.frame_context.frame_number
    fps = int(ctx.context_store.fps) if ctx.context_store.fps else 30
    
    start_frame = max(1, current_frame - fps)
    end_frame = current_frame
    
    ensure_cv_for_range(ctx, start_frame, end_frame)
    
    # 1. Current Frame Detections
    current_objects = []
    if ctx.frame_context.detections:
        ignored = {"person", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
        for d in ctx.frame_context.detections:
            if d.class_name.lower() not in ignored:
                current_objects.append(f"{d.class_name} ({d.confidence:.2f})")
    
    current_hint = ", ".join(current_objects) if current_objects else "none"

    # 2. Aggregated Detections (Previous Interval)
    agg_end = max(start_frame, end_frame - 1)
    if agg_end < start_frame:
        aggregated_scores = {}
    else:
        aggregated_scores = aggregate_detections(ctx, start_frame, agg_end)
    
    if aggregated_scores:
        sorted_items = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
        agg_hint = ", ".join([f"{obj} ({score:.2f})" for obj, score in sorted_items])
    else:
        agg_hint = "none"
        
    options_list = ctx.batch_params.allowed_tools
    options_str = ", ".join([f"'{opt}'" for opt in options_list])
    
    # Filter out redundant "Objects:" line and "No objects detected." from context_text
    filtered_context = "\n".join([
        line for line in ctx.context_text.split('\n') 
        if not line.strip().startswith("Objects:") and "No objects detected." not in line
    ])
    
    prompt = (
        f"This is a POV of a construction worker using a tool.\n\n"
        f"Objects in CURRENT frame: {current_hint}\n"
        f"Objects in PREVIOUS 1 second: {agg_hint}\n\n"
        f"{filtered_context}\n\n"
        f"CRITICAL INSTRUCTION: Identify the tool being used.\n"
        f"You must respond with EXACTLY ONE of the following words: {options_str}.\n"
        f"Do NOT provide any reasoning, preamble, or extra text.\n"
        f"If you are not 100% sure, or if the tool is not in the list, respond with 'unknown'.\n"
        f"Response:"
    )
    return ctx.call_llm(prompt, valid_options=options_list, log_label="Object Check (1s Agg)").lower().strip()

@register_unknown_object_check("llm_guess_with_options")
def unknown_check_options(ctx: DecisionContext) -> str:
    """
    Ask LLM to guess from categories.
    1 API call.
    """
    prompt = (
        f"The worker is using an unidentified object.\n"
        f"Is it:\n"
        f"1. A building material (lumber, pipe, drywall)\n"
        f"2. A fastener (screw, nail, bolt)\n"
        f"3. A container (bucket, box)\n"
        f"4. Another tool\n\n"
        f"Respond with your best guess of the specific object name."
    )
    return ctx.call_llm(prompt, log_label="Unknown Check (Options)").lower().strip()

@register_unknown_object_check("cv_class_name")
def unknown_check_cv(ctx: DecisionContext) -> str:
    """
    Return raw CV class name if available.
    0 API calls.
    """
    return object_check_cv(ctx)

@register_unknown_object_check("temporal_majority")
def unknown_check_temporal_majority(ctx: DecisionContext) -> str:
    """
    Guess based on majority of tools seen in recent history.
    0 API calls.
    """
    from collections import Counter
    
    # Get all frames from context store
    # (In a real scenario, we might want to limit to last N seconds, 
    # but context_store.frames is already a windowed deque)
    history = ctx.context_store.frames
    
    if not history:
        return "unknown"
        
    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face", "arm", "leg"}
    tool_counts = Counter()
    
    for frame_ctx in history:
        for detection in frame_ctx.detections:
            name = detection.class_name.lower()
            if name not in ignored:
                tool_counts[name] += 1
                
    if not tool_counts:
        return "unknown"
        
    # Get most common
    most_common = tool_counts.most_common(1)[0][0]
    print(f"\n=== Unknown Check (Temporal) ===\nGuess: {most_common} (from history)\n==================\n")
    return most_common

@register_unknown_object_check("skip")
def unknown_check_skip(ctx: DecisionContext) -> str:
    """
    Do not guess.
    0 API calls.
    """
    return ""
