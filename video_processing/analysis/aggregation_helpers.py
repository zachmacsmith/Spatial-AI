from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np

from ..context import DecisionContext, FrameContext, Detection

def ensure_cv_for_range(ctx: DecisionContext, start_frame: int, end_frame: int):
    """
    Lazy-load CV results for a range of frames.
    Respects cv_detection_frequency.
    """
    if not ctx.cv_service or not ctx.frame_cache:
        return

    freq = ctx.batch_params.cv_detection_frequency
    if freq <= 0:
        freq = 1 # Default to every frame if 0? Or maybe 0 means NONE?
        return

    # Determine frames to check
    frames_to_check = list(range(start_frame, end_frame, freq))
    
    # Always check the end_frame (current keyframe) if not covered
    if end_frame not in frames_to_check:
        frames_to_check.append(end_frame)
        
    for frame_num in frames_to_check:
        # Check if context already exists and has detections
        c = ctx.context_store.get(frame_num)
        if c and c.detections is not None:
            continue

        # Load frame
        try:
            frame = ctx.frame_cache.get(frame_num)
            if frame is None:
                continue
        except Exception:
            continue

        # Run CV
        detections = []
        raw_results = ctx.cv_service.get_results_object(frame, ctx.batch_params.cv_confidence_threshold)
        
        if raw_results:
            for box, cls_id, conf in zip(
                raw_results.boxes.xyxy,
                raw_results.boxes.cls,
                raw_results.boxes.conf
            ):
                class_name = raw_results.names[int(cls_id)]
                box_tuple = tuple(map(int, box.cpu().numpy()))
                detections.append(Detection(class_name, float(conf), box_tuple))

        # Update/Create Context
        if not c:
            timestamp = frame_num / ctx.context_store.fps if ctx.context_store.fps else 0
            c = FrameContext(frame_num, timestamp)
            ctx.context_store.add(c)
        
        c.detections = detections
        c.raw_results = raw_results

def aggregate_detections(ctx: DecisionContext, start_frame: int, end_frame: int) -> Dict[str, float]:
    """
    Helper to aggregate detections over a frame range.
    Returns a dict of {object_name: max_confidence}.
    """
    aggregated_confidences = defaultdict(list)
    object_frames_seen = defaultdict(set)
    
    ignored = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face", "arm", "leg"}
    
    # Iterate through frames in range
    for f in range(start_frame, end_frame + 1):
        c = ctx.context_store.get(f)
        if not c or c.detections is None:
            continue
            
        for d in c.detections:
            name = d.class_name.lower()
            if name not in ignored:
                aggregated_confidences[name].append(d.confidence)
                object_frames_seen[name].add(f)
                
    # Calculate final scores
    final_scores = {}
    for name, confs in aggregated_confidences.items():
        # Score = max_confidence * (frames_seen / total_frames_with_detections) ?
        # Or just max confidence?
        # Let's use max confidence for now, maybe boosted by frequency?
        # Simple: Max confidence.
        final_scores[name] = max(confs)
        
    return final_scores
