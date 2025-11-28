"""
Decision Functions
Pluggable logic for making classification decisions.
"""
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass
import numpy as np

from .context import FrameContext, ContextStore
from .batch_parameters import BatchParameters

@dataclass
class DecisionContext:
    """
    All information available for making a decision.
    Decision functions receive this and use whatever subset they need.
    """
    frames: List[np.ndarray]
    frame_context: FrameContext
    context_store: ContextStore
    context_text: str
    llm_service: Any
    batch_params: BatchParameters
    api_calls_used: int = 0
    
    def call_llm(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Wrapper for LLM calls that tracks usage.
        Increments self.api_calls_used.
        Returns stripped response text.
        """
        self.api_calls_used += 1
        result = self.llm_service.send_multiframe_prompt(
            frames=self.frames,
            prompt_text=prompt,
            max_tokens=max_tokens
        ).strip()
        print(f"LLM Call: {result}")
        return result
        
    def call_llm_no_image(self, prompt: str, max_tokens: int = 50) -> str:
        """
        LLM call without sending images (cheaper).
        For decisions that only need text context.
        Increments self.api_calls_used.
        """
        # TODO: Implement text-only call if service supports it
        # For now, just use same method but maybe empty frames? 
        # Or just pass frames anyway since our interface requires them.
        return self.call_llm(prompt, max_tokens)

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
    prompt = (
        f"This is a POV of a construction worker.\n\n"
        f"{ctx.context_text}\n\n"
        f"What is the worker doing? Respond with exactly one word: 'idle', 'moving', or 'using tool'."
    )
    response = ctx.call_llm(prompt).lower()
    
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
    
    prompt = (
        f"This is a POV of a person. Motion score: {motion_str}, "
        f"a motion score of 0-0.16 suggests the person is idle, "
        f"10 or above suggests they are moving. "
        f"Check if hands are visible and classify behavior. "
        f"Classify the action: idle, moving, using tool. "
        f"Respond with ONE word only."
    )
    
    response = ctx.call_llm(prompt).lower()
    
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
    Return first detected tool from CV.
    0 API calls.
    """
    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
    tools = [d for d in ctx.frame_context.detections if d.class_name not in ignored]
    
    if tools:
        # Sort by confidence
        tools.sort(key=lambda x: x.confidence, reverse=True)
        return tools[0].class_name
        
    return "unknown"

@register_object_check("llm_direct")
def object_check_llm(ctx: DecisionContext) -> str:
    """
    Ask LLM to identify tool.
    1 API call.
    """
    prompt = (
        f"This is a POV of a construction worker using a tool.\n\n"
        f"{ctx.context_text}\n\n"
        f"What tool are they using? Respond with the tool name only (e.g., 'drill', 'hammer'). "
        f"If you cannot identify it, respond with 'unknown'."
    )
    response = ctx.call_llm(prompt).lower().strip()
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
    return ctx.call_llm(prompt).lower().strip()

@register_object_check("legacy_testing_class")
def object_check_legacy(ctx: DecisionContext) -> str:
    """
    Replicate TestingClass.py logic (ask_claude_which_tool).
    1 API call.
    """
    prompt = (
        f"This is a POV of a person using a tool.\n"
        f"What tool is being used?\n"
        f"Respond with the tool name only."
    )
    return ctx.call_llm(prompt).lower().strip()

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
    return ctx.call_llm(prompt).lower().strip()

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
    return ctx.call_llm(prompt).lower().strip()

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
    return most_common

@register_unknown_object_check("skip")
def unknown_check_skip(ctx: DecisionContext) -> str:
    """
    Do not guess.
    0 API calls.
    """
    return ""
