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
    frames_to_check_for_cv = list(range(start_frame, end_frame, freq))
    
    # Always check the end_frame (current keyframe) if not covered
    if end_frame not in frames_to_check_for_cv:
        frames_to_check_for_cv.append(end_frame)
    
    # print(f"[DEBUG] ensure_cv_for_range: Checking {start_frame}-{end_frame}. Frames to CV: {frames_to_check_for_cv}")
    
    for frame_num in range(start_frame, end_frame + 1):
        # Only run CV for frames specified by frequency
        if frame_num not in frames_to_check_for_cv:
            continue

        # Skip if already processed (e.g., if frame_num is a keyframe)
        c = ctx.context_store.get(frame_num) # Get context here to use it below
        if c:
            # Check if it has detections
            if c.detections:
                # print(f"[DEBUG] Frame {frame_num} skipped (already has {len(c.detections)} detections)")
                continue
            elif c.raw_results:
                 # print(f"[DEBUG] Frame {frame_num} skipped (has raw_results but 0 detections)")
                 continue
            else:
                 # print(f"[DEBUG] Frame {frame_num} in store but NO detections/raw_results. Re-running CV.")
                 # Force re-run
                 pass
            
        # Load frame
        try:
            frame = ctx.frame_cache.get(frame_num)
            if frame is None:
                # print(f"[DEBUG] Frame {frame_num} could not be loaded (None)")
                continue
        except Exception as e:
            print(f"[ERROR] ensure_cv_for_range failed to load frame {frame_num}: {e}")
            continue

        # Run CV
        detections = []
        raw_results = None
        
        if ctx.cv_service:
            # Get full results object for caching
            raw_results = ctx.cv_service.get_results_object(frame, ctx.batch_params.cv_confidence_threshold)
            
            # Extract detections for context
            if raw_results:
                for box, cls_id, conf in zip(
                    raw_results.boxes.xyxy,
                    raw_results.boxes.cls,
                    raw_results.boxes.conf
                ):
                    class_name = raw_results.names[int(cls_id)]
                    box_tuple = tuple(map(int, box.cpu().numpy()))
                    detections.append(Detection(class_name, float(conf), box_tuple))
            
            if detections:
                # print(f"[DEBUG] CV found {len(detections)} objects in frame {frame_num}: {[d.class_name for d in detections]}")
                pass
            else:
                # print(f"[DEBUG] CV found 0 objects in frame {frame_num}")
                pass

        # Update/Create Context
        if not c:
            timestamp = frame_num / ctx.context_store.fps if ctx.context_store.fps else 0
            c = FrameContext(frame_num, timestamp)
            ctx.context_store.add(c)
        
        c.detections = detections
        c.raw_results = raw_results
        
        if detections:
            # print(f"[DEBUG] ensure_cv_for_range: Loaded {len(detections)} detections for frame {frame_num}")
            pass
        else:
            # print(f"[DEBUG] ensure_cv_for_range: No detections for frame {frame_num}")
            pass

def aggregate_detections(ctx: DecisionContext, start_frame: int, end_frame: int) -> Dict[str, Dict[str, float]]:
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
                
    # Calculate final scores and frequency
    final_scores = {}
    total_frames = end_frame - start_frame + 1
    
    # Avoid division by zero
    if total_frames <= 0:
        total_frames = 1
        
    for name, confs in aggregated_confidences.items():
        max_conf = max(confs)
        frames_present = len(object_frames_seen[name])
        frequency = frames_present / total_frames
        
        final_scores[name] = {
            "confidence": max_conf,
            "frequency": frequency,
            "count": frames_present
        }
    
    # print(f"[DEBUG] aggregate_detections: Processed {total_frames} frames. Found {len(final_scores)} unique objects.")
    # if final_scores:
    #     print(f"[DEBUG] Objects found: {list(final_scores.keys())}")
        
    return final_scores
