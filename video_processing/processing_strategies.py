import time
from typing import List, Dict, Callable, Optional, Any
import numpy as np

from .batch_parameters import BatchParameters
from .context import ContextStore, ComposableContextBuilder, FrameContext
from .api_request_batcher import APIRequestBatcher, APIBatchRequest
from .utils.video_utils import sample_interval_frames, calculate_motion_score
from .prompting_protocols import get_prompting_protocol, ClassificationResult

# Registry for processing strategies
PROCESSING_STRATEGY_REGISTRY: Dict[str, Callable] = {}

def register_processing_strategy(name: str):
    """Decorator to register a processing strategy"""
    def decorator(func: Callable):
        PROCESSING_STRATEGY_REGISTRY[name] = func
        return func
    return decorator

def get_processing_strategy(name: str) -> Callable:
    """Get a processing strategy by name"""
    if name not in PROCESSING_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown processing strategy: {name}. Available: {list(PROCESSING_STRATEGY_REGISTRY.keys())}")
    return PROCESSING_STRATEGY_REGISTRY[name]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _interpolate_from_keyframes(
    keyframe_numbers: List[int],
    context_store: ContextStore,
    frame_count: int
) -> List[str]:
    """
    Fill frame labels by interpolating from classified keyframes.
    Each frame gets the label of the nearest classified keyframe.
    """
    frame_labels = [""] * frame_count
    
    # Get all classified frames from store
    classified_frames = []
    for kf in keyframe_numbers:
        ctx = context_store.get(kf)
        if ctx and ctx.action:
            classified_frames.append((kf, ctx))
            
    if not classified_frames:
        return ["idle"] * frame_count
        
    # Sort just in case
    classified_frames.sort(key=lambda x: x[0])
    
    # Fill frames
    for i in range(frame_count):
        frame_num = i + 1
        
        # Find nearest classified frame
        nearest_dist = float('inf')
        nearest_action = "idle"
        
        for kf, ctx in classified_frames:
            dist = abs(frame_num - kf)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_action = ctx.action
                nearest_tool = ctx.tool
                nearest_guess = ctx.tool_guess
            elif dist == nearest_dist:
                # Tie-break: prefer earlier frame (arbitrary but consistent)
                pass
            else:
                # Optimization: if distance starts increasing, we passed the nearest
                # (since classified_frames is sorted)
                break
                
        frame_labels[i] = nearest_action
        
        # Update context store for this frame
        current_ctx = context_store.get(frame_num)
        if not current_ctx:
            # Create minimal context if missing
            # We need timestamp. context_store has fps.
            timestamp = frame_num / context_store.fps
            current_ctx = FrameContext(frame_num, timestamp)
            context_store.add(current_ctx)
            
        # Set interpolated values
        current_ctx.action = nearest_action
        current_ctx.tool = nearest_tool
        current_ctx.tool_guess = nearest_guess
        
    return frame_labels

def _needs_classification(
    frame_number: int,
    context_store: ContextStore,
    last_classified_frame: Optional[int],
    batch_params: BatchParameters
) -> bool:
    """
    Determine if a frame needs classification based on context changes.
    Used by SMART strategy.
    """
    # 1. Always classify if no previous classification
    if last_classified_frame is None:
        return True
        
    # 2. Check gap
    if (frame_number - last_classified_frame) > batch_params.smart_classification_max_gap:
        return True
        
    # Get contexts
    current_ctx = context_store.get(frame_number)
    prev_ctx = context_store.get(last_classified_frame)
    
    if not current_ctx or not prev_ctx:
        return True
        
    # 3. Check objects (excluding ignored ones)
    ignored_objects = {'person', 'hand', 'hardhat', 'safety vest', 'glove', 'helmet', 'vest'}
    
    curr_objs = set(current_ctx.object_names) - ignored_objects
    prev_objs = set(prev_ctx.object_names) - ignored_objects
    
    if curr_objs != prev_objs:
        return True
        
    # 4. Check motion (if available/computed)
    # Note: motion_score might be None for keyframes initially, 
    # but we might have it if we computed it during pre-computation or if it's an interval
    # For keyframes, we might not have motion score yet.
    # We'll skip motion check for now or rely on other factors.
    
    # 5. Check relationships
    if set(current_ctx.relationships) != set(prev_ctx.relationships):
        return True
        
    return False

def _classify_with_protocol(
    frames: List[np.ndarray],
    frame_context: FrameContext,
    context_store: ContextStore,
    context_builder: ComposableContextBuilder,
    llm_service: Any,
    batch_params: BatchParameters
) -> ClassificationResult:
    """Helper to classify frames using the configured protocol"""
    protocol = get_prompting_protocol(batch_params.prompting_protocol.value)
    
    # Build context text
    context_text = context_builder.build(context_store, frame_context)
    
    return protocol.classify(
        frames=frames,
        frame_context=frame_context,
        context_store=context_store,
        context_text=context_text,
        llm_service=llm_service,
        batch_params=batch_params
    )

# ==========================================
# STRATEGIES
# ==========================================

@register_processing_strategy("classify_all")
def strategy_classify_all(
    keyframe_numbers: List[int],
    frame_count: int,
    context_store: ContextStore,
    context_builder: ComposableContextBuilder,
    api_batcher: APIRequestBatcher,
    llm_service,
    frame_cache: Dict[int, np.ndarray],
    batch_params: BatchParameters
) -> List[str]:
    """
    Current behavior: Classify every keyframe, then classify every interval.
    """
    print(f"Strategy: CLASSIFY_ALL ({len(keyframe_numbers)} keyframes + intervals)")
    frame_labels = [""] * frame_count
    total_api_calls = 0
    
    # 1. Classify Keyframes
    for kf_number in keyframe_numbers:
        frame = frame_cache.get(kf_number)
        if frame is None: continue 
        
        ctx = context_store.get(kf_number)
        
        # Use protocol
        result = _classify_with_protocol(
            frames=[frame],
            frame_context=ctx,
            context_store=context_store,
            context_builder=context_builder,
            llm_service=llm_service,
            batch_params=batch_params
        )
        
        # Update context
        ctx.action = result.action
        ctx.tool = result.tool
        ctx.tool_guess = result.tool_guess
        ctx.api_calls_used = result.api_calls_used
        total_api_calls += result.api_calls_used
        
        frame_labels[kf_number - 1] = result.action
            
    # 2. Process Intervals
    for idx in range(len(keyframe_numbers) - 1):
        start = keyframe_numbers[idx]
        end = keyframe_numbers[idx + 1]
        
        frames_to_send = sample_interval_frames(
            frame_cache, start, end, batch_params.num_frames_per_interval
        )
        motion = calculate_motion_score(frames_to_send, batch_params.motion_ignore_threshold)
        
        start_ctx = context_store.get(start)
        
        # Create temporary context for interval
        from copy import copy
        interval_ctx = copy(start_ctx)
        interval_ctx.motion_score = motion
        
        # Use protocol
        result = _classify_with_protocol(
            frames=frames_to_send,
            frame_context=interval_ctx,
            context_store=context_store,
            context_builder=context_builder,
            llm_service=llm_service,
            batch_params=batch_params
        )
        
        total_api_calls += result.api_calls_used
        
        # Fill labels
        for f in range(start, end):
            frame_labels[f] = result.action
            
            # Update context store for interval frames
            ctx = context_store.get(f)
            if not ctx:
                # Create minimal context if missing
                timestamp = f / context_store.fps
                ctx = FrameContext(f, timestamp)
                context_store.add(ctx)
            
            ctx.action = result.action
            ctx.tool = result.tool
            ctx.tool_guess = result.tool_guess
            ctx.api_calls_used = 0  # Amortized or 0 for interval frames
            
    # Fill tail
    last_kf = keyframe_numbers[-1]
    for f in range(last_kf, frame_count):
        frame_labels[f] = frame_labels[last_kf - 1]
        
        # Update context store for tail frames
        ctx = context_store.get(f + 1) # frame_labels is 0-indexed, context is 1-indexed
        if not ctx:
            timestamp = (f + 1) / context_store.fps
            ctx = FrameContext(f + 1, timestamp)
            context_store.add(ctx)
            
        last_ctx = context_store.get(last_kf)
        if last_ctx:
            ctx.action = last_ctx.action
            ctx.tool = last_ctx.tool
            ctx.tool_guess = last_ctx.tool_guess
        
    print(f"  Total API calls: {total_api_calls}")
    return frame_labels

@register_processing_strategy("keyframes_only")
def strategy_keyframes_only(
    keyframe_numbers: List[int],
    frame_count: int,
    context_store: ContextStore,
    context_builder: ComposableContextBuilder,
    api_batcher: APIRequestBatcher,
    llm_service,
    frame_cache: Dict[int, np.ndarray],
    batch_params: BatchParameters
) -> List[str]:
    """
    Classify all keyframes. Interpolate intervals.
    """
    print(f"Strategy: KEYFRAMES_ONLY ({len(keyframe_numbers)} keyframes)")
    total_api_calls = 0
    
    # 1. Classify Keyframes
    for kf_number in keyframe_numbers:
        frame = frame_cache.get(kf_number)
        if frame is None: continue
        
        ctx = context_store.get(kf_number)
        
        # Use protocol
        result = _classify_with_protocol(
            frames=[frame],
            frame_context=ctx,
            context_store=context_store,
            context_builder=context_builder,
            llm_service=llm_service,
            batch_params=batch_params
        )
        
        # Update context
        ctx.action = result.action
        ctx.tool = result.tool
        ctx.tool_guess = result.tool_guess
        ctx.api_calls_used = result.api_calls_used
        total_api_calls += result.api_calls_used
            
    print(f"  Total API calls: {total_api_calls}")
    # 2. Interpolate
    return _interpolate_from_keyframes(keyframe_numbers, context_store, frame_count)

@register_processing_strategy("smart")
def strategy_smart(
    keyframe_numbers: List[int],
    frame_count: int,
    context_store: ContextStore,
    context_builder: ComposableContextBuilder,
    api_batcher: APIRequestBatcher,
    llm_service,
    frame_cache: Dict[int, np.ndarray],
    batch_params: BatchParameters
) -> List[str]:
    """
    Classify keyframes only when context changes significantly.
    """
    print(f"Strategy: SMART (Change detection)")
    
    classified_kf_numbers = []
    last_classified = None
    total_api_calls = 0
    
    # 1. Identify keyframes to classify
    for kf_number in keyframe_numbers:
        if _needs_classification(kf_number, context_store, last_classified, batch_params):
            classified_kf_numbers.append(kf_number)
            last_classified = kf_number
        # Don't try to inherit yet - nothing classified
                
    print(f"  Selected {len(classified_kf_numbers)}/{len(keyframe_numbers)} keyframes for classification")
    
    # 2. Classify selected keyframes
    for kf_number in classified_kf_numbers:
        frame = frame_cache.get(kf_number)
        if frame is None: continue
        
        ctx = context_store.get(kf_number)
        
        # Use protocol
        result = _classify_with_protocol(
            frames=[frame],
            frame_context=ctx,
            context_store=context_store,
            context_builder=context_builder,
            llm_service=llm_service,
            batch_params=batch_params
        )
        
        # Update context
        ctx.action = result.action
        ctx.tool = result.tool
        ctx.tool_guess = result.tool_guess
        ctx.api_calls_used = result.api_calls_used
        total_api_calls += result.api_calls_used
            
    # 3. Interpolate (using ALL keyframes, filling in the skipped ones)
    
    last_action = "idle"
    for kf_number in keyframe_numbers:
        ctx = context_store.get(kf_number)
        if kf_number in classified_kf_numbers:
            if ctx and ctx.action:
                last_action = ctx.action
        else:
            if ctx:
                ctx.action = last_action
                
    print(f"  Total API calls: {total_api_calls}")
    return _interpolate_from_keyframes(keyframe_numbers, context_store, frame_count)

@register_processing_strategy("keyframes_sampled")
def strategy_keyframes_sampled(
    keyframe_numbers: List[int],
    frame_count: int,
    context_store: ContextStore,
    context_builder: ComposableContextBuilder,
    api_batcher: APIRequestBatcher,
    llm_service,
    frame_cache: Dict[int, np.ndarray],
    batch_params: BatchParameters
) -> List[str]:
    """
    Classify every Nth keyframe.
    """
    rate = batch_params.keyframe_sample_rate
    print(f"Strategy: KEYFRAMES_SAMPLED (Every {rate}th keyframe)")
    total_api_calls = 0
    
    # Select keyframes
    selected_keyframes = keyframe_numbers[::rate]
    
    # Classify
    for kf_number in selected_keyframes:
        frame = frame_cache.get(kf_number)
        if frame is None: continue
        
        ctx = context_store.get(kf_number)
        
        # Use protocol
        result = _classify_with_protocol(
            frames=[frame],
            frame_context=ctx,
            context_store=context_store,
            context_builder=context_builder,
            llm_service=llm_service,
            batch_params=batch_params
        )
        
        # Update context
        ctx.action = result.action
        ctx.tool = result.tool
        ctx.tool_guess = result.tool_guess
        ctx.api_calls_used = result.api_calls_used
        total_api_calls += result.api_calls_used
            
    print(f"  Total API calls: {total_api_calls}")
    return _interpolate_from_keyframes(keyframe_numbers, context_store, frame_count)
