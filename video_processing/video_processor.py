"""
Video Processor - Main orchestrator for video processing pipeline

This is the main entry point that ties all modules together.
Conditionally executes features based on BatchParameters configuration.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np

# Import all services and utilities
from .ai.llm_service import get_llm_service
from .ai.cv_service import get_cv_service
from .ai.prompt_builder import PromptBuilder
from .analysis.action_classifier import classify_action, apply_temporal_smoothing
from .analysis.tool_detector import detect_tool
from .analysis.relationship_tracker import RelationshipTracker
from .utils.video_utils import (
    get_video_properties,
    load_all_frames,
    get_frame_from_cache,
    sample_interval_frames,
    calculate_motion_score,
    load_keyframe_numbers,
    create_video_writer,
    extract_keyframes,
    FrameLoader
)
from .utils.visualization import create_visualization
from .output.output_manager import (
    save_actions_csv,
    save_relationships_csv,
    save_batch_metadata
)
from .processing_strategies import get_processing_strategy


class RateLimiter:
    """
    Rate limiter for API calls to prevent hitting requests-per-minute limits.
    
    Thread-safe implementation that tracks API calls and enforces delays.
    """
    
    def __init__(self, requests_per_minute: int, buffer: float = 0.1):
        """
        Args:
            requests_per_minute: Maximum requests allowed per minute
            buffer: Safety buffer (0.1 = 10% slower than limit)
        """
        self.requests_per_minute = requests_per_minute
        self.buffer = buffer
        
        # Calculate minimum time between requests (in seconds)
        # Add buffer for safety
        self.min_interval = (60.0 / requests_per_minute) * (1 + buffer)
        
        self.last_request_time = 0
        self.lock = Lock()
        self.request_count = 0
        
        print(f"Rate Limiter: {requests_per_minute} req/min, "
              f"min interval: {self.min_interval:.2f}s")
    
    def wait_if_needed(self):
        """
        Wait if necessary to respect rate limit.
        Call this before making an API request.
        """
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                print(f"  [Rate Limit] Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
            
            self.last_request_time = time.time()
            self.request_count += 1
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        return {
            'total_requests': self.request_count,
            'requests_per_minute': self.requests_per_minute,
            'min_interval': self.min_interval
        }


def _precompute_keyframe_contexts(
    keyframe_numbers: List[int],
    frame_cache: Dict[int, np.ndarray],
    context_store: "ContextStore",
    cv_service,
    batch_params,
    video_props,
    relationship_tracker=None
) -> None:
    """
    Pre-compute and store context for all keyframes.
    Must run before classification so strategies can query history.
    """
    from .context import FrameContext, Detection
    
    print(f"\nPhase 1: Pre-computing context for {len(keyframe_numbers)} keyframes...")
    
    actual_frame_count = len(frame_cache)
    
    for kf_number in keyframe_numbers:
        frame = get_frame_from_cache(frame_cache, kf_number)
        
        # Run YOLO
        detections = []
        relationships = []
        raw_results = None
        
        if cv_service:
            # Get full results object for caching
            raw_results = cv_service.get_results_object(frame, batch_params.cv_confidence_threshold)
            
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
        
        # Compute relationships (lightweight snapshot for context)
        if detections:
            # Simple proximity check for context building
            if relationship_tracker:
                 det_tuples = [(d.bbox, d.class_name, d.confidence) for d in detections]
                 rel_sets, _ = relationship_tracker.find_relationships(
                    det_tuples, video_props.width, kf_number
                 )
                 relationships = rel_sets
        
        # Add to context store
        ctx = FrameContext(
            frame_number=kf_number,
            timestamp=kf_number / video_props.fps,
            detections=detections,
            motion_score=None,  # Will be computed during interval processing
            relationships=relationships,
            raw_results=raw_results # Cache for video generation
        )
        context_store.add(ctx)
        
        if kf_number % 50 == 0:
            print(f"  Pre-computed {kf_number}/{actual_frame_count} frames")
            
    print("✓ Context pre-computation complete")


def process_video(video_name: str, batch_params) -> Dict[str, str]:
    """
    Main video processing function.
    
    This orchestrates the entire pipeline:
    1. Load video and frames
    2. Pre-compute context (CV + Relationships) for ALL keyframes
    3. Classify keyframes using rich context
    4. Process intervals using context
    5. Apply temporal smoothing
    6. Generate labeled video (optional)
    7. Save outputs (CSVs, metadata)
    
    Args:
        video_name: Name of video (without .mp4 extension)
        batch_params: BatchParameters instance
    
    Returns:
        Dict mapping output_type -> file_path
    """
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"Batch ID: {batch_params.batch_id}")
    print(f"Config: {batch_params.config_name}")
    print(f"Context Strategy: {batch_params.context_strategy.value}")
    print(f"Processing Strategy: {batch_params.processing_strategy.value}")
    print(f"{'='*60}\n")
    
    # Save batch configuration
    if batch_params.save_config_with_outputs:
        batch_params.save_batch_config()
    
    # ==========================================
    # 1. SETUP
    # ==========================================
    
    video_path = os.path.join(batch_params.video_directory, f"{video_name}.mp4")
    keyframes_folder = os.path.join(batch_params.keyframes_directory, f"keyframes{video_name}")
    
    # Get video properties
    video_props = get_video_properties(video_path)
    print(f"Video: {video_props.width}x{video_props.height}, {video_props.fps:.2f} fps, "
          f"{video_props.frame_count} frames, {video_props.duration:.2f}s")
    
    # Load keyframe numbers
    try:
        keyframe_numbers = load_keyframe_numbers(keyframes_folder)
        if not keyframe_numbers:
            raise FileNotFoundError("No keyframes found")
    except FileNotFoundError:
        print(f"⚠ Keyframes not found in {keyframes_folder}")
        print("  Generating keyframes automatically...")
        extract_keyframes(video_path, keyframes_folder)
        keyframe_numbers = load_keyframe_numbers(keyframes_folder)
        
    print(f"Keyframes: {len(keyframe_numbers)}")
    
    # Pre-load all frames if enabled
    # Load frames
    if batch_params.preload_all_frames:
        # Legacy mode: load all frames into memory
        # Use for short videos or systems with high RAM
        print("Pre-loading all frames into memory...")
        frame_cache = load_all_frames(video_path)
        print(f"Loaded {len(frame_cache)} frames into memory")
    else:
        # Streaming mode: load on-demand with LRU cache
        # Use for longer videos or limited RAM
        print(f"Using streaming frame loader (cache size: {batch_params.frame_cache_size} frames)")
        frame_cache = FrameLoader(video_path, max_frames=batch_params.frame_cache_size)
        
        # Preload keyframes for faster access during classification
        if batch_params.preload_keyframes and keyframe_numbers:
            print(f"Preloading {len(keyframe_numbers)} keyframes into cache...")
            frame_cache.preload(keyframe_numbers)
            print(f"Cache populated: {len(frame_cache._cache)}/{batch_params.frame_cache_size} frames")
    
    # Use actual loaded frame count (not reported metadata count)
    actual_frame_count = len(frame_cache)
    
    # Initialize frame labels based on actual frames
    frame_labels = [""] * actual_frame_count
    
    # Initialize services
    llm_service = get_llm_service(batch_params)
    prompt_builder = PromptBuilder(batch_params)
    
    # Initialize Context components
    from .context import ContextStore, get_context_builder, FrameContext, Detection
    context_store = ContextStore(fps=video_props.fps, window_seconds=5.0)
    context_builder = get_context_builder(batch_params.context_strategy.value)
    
    # Initialize rate limiter if enabled
    rate_limiter = None
    if batch_params.enable_rate_limiting:
        rate_limiter = RateLimiter(
            requests_per_minute=batch_params.api_requests_per_minute,
            buffer=batch_params.rate_limit_buffer
        )
    
    cv_service = None
    if batch_params.enable_object_detection:
        cv_service = get_cv_service(batch_params)
        print(f"CV Model: {cv_service.get_model_name()}")
    
    print(f"LLM: {llm_service.get_provider_name()}")
    
    # Initialize relationship tracker if enabled
    relationship_tracker = None
    if batch_params.enable_relationship_tracking:
        relationship_tracker = RelationshipTracker(
            fps=video_props.fps,
            proximity_threshold_percent=batch_params.proximity_threshold_percent
        )
        print("Relationship tracking enabled")
    
    # ==========================================
    # 2. PRE-COMPUTE CONTEXT (The "See" Phase)
    # ==========================================
    
    _precompute_keyframe_contexts(
        keyframe_numbers,
        frame_cache,
        context_store,
        cv_service,
        batch_params,
        video_props,
        relationship_tracker
    )
    
    # ==========================================
    # 3. CLASSIFY & PROCESS (The "Think" & "Act" Phases)
    # ==========================================
    
    # Initialize API batcher
    from .api_request_batcher import create_api_request_batcher
    api_batcher = create_api_request_batcher(batch_params)
    
    # Dispatch to selected strategy
    strategy = get_processing_strategy(batch_params.processing_strategy.value)
    
    print(f"\nPhase 2 & 3: Executing strategy '{batch_params.processing_strategy.value}'...")
    
    frame_labels = strategy(
        keyframe_numbers=keyframe_numbers,
        frame_count=actual_frame_count,
        context_store=context_store,
        context_builder=context_builder,
        api_batcher=api_batcher,
        llm_service=llm_service,
        prompt_builder=prompt_builder,
        frame_cache=frame_cache,
        batch_params=batch_params
    )
    
    # ==========================================
    # 5. TEMPORAL SMOOTHING
    # ==========================================
    
    if batch_params.enable_temporal_smoothing:
        print("\nApplying temporal smoothing...")
        frame_labels = apply_temporal_smoothing(
            frame_labels,
            batch_params.allowed_actions,
            batch_params.temporal_smoothing_window
        )
    
    # ==========================================
    # 6. GENERATE LABELED VIDEO (OPTIONAL)
    # ==========================================
    
    output_files = {}
    
    if batch_params.generate_labeled_video:
        print("\nGenerating labeled video...")
        
        # Organize videos by batch ID
        batch_folder = Path(batch_params.video_output_directory) / batch_params.batch_id
        batch_folder.mkdir(parents=True, exist_ok=True)
        video_output_path = batch_folder / f"{video_name}.mp4"
        video_output_path = str(video_output_path)
        
        writer = create_video_writer(
            video_output_path,
            video_props.fps,
            video_props.width,
            video_props.height,
            batch_params.video_codec
        )
        
        current_yolo_results = None
        current_relationships = []
        
        # Reset relationship tracker for full fidelity tracking
        if relationship_tracker:
            relationship_tracker = RelationshipTracker(
                fps=video_props.fps,
                proximity_threshold_percent=batch_params.proximity_threshold_percent
            )
        
        # Choose iteration method based on frame_cache type
        if hasattr(frame_cache, 'iter_frames'):
            # Streaming mode: use efficient sequential iteration
            frame_iterator = frame_cache.iter_frames(1, actual_frame_count)
        else:
            # Legacy mode: iterate over dict
            frame_iterator = ((i + 1, get_frame_from_cache(frame_cache, i + 1)) for i in range(actual_frame_count))

        for frame_num, frame in frame_iterator:
            i = frame_num - 1  # Convert to 0-indexed for compatibility
            
            # Check if we have cached results for this frame (it was a keyframe)
            ctx = context_store.get(frame_num)
            if ctx and ctx.raw_results:
                current_yolo_results = ctx.raw_results
            elif cv_service and i % batch_params.cv_detection_frequency == 0:
                # Run YOLO periodically if not a keyframe
                current_yolo_results = cv_service.get_results_object(frame, batch_params.cv_confidence_threshold)
            
            # Track relationships (full fidelity for video)
            line_info = []
            if relationship_tracker and current_yolo_results:
                # Extract detections from current results
                detections = []
                for box, cls_id, conf in zip(
                    current_yolo_results.boxes.xyxy,
                    current_yolo_results.boxes.cls,
                    current_yolo_results.boxes.conf
                ):
                    class_name = current_yolo_results.names[int(cls_id)]
                    box_tuple = tuple(map(int, box.cpu().numpy()))
                    detections.append((box_tuple, class_name, float(conf)))
                
                relationships, line_info = relationship_tracker.find_relationships(
                    detections, video_props.width, frame_num
                )
                relationship_tracker.update(relationships, frame_num)
            
            # Refine "using tool" label with detected object
            display_label = frame_labels[i]
            if display_label == "using tool":
                tool_name = "unknown"
                if current_yolo_results:
                    # Filter out non-tool objects
                    ignored_objects = {"person", "hand", "hardhat", "safety vest", "glove", "helmet", "vest", "face"}
                    detected_tools = []
                    
                    if hasattr(current_yolo_results, 'boxes'):
                        for cls_id, conf in zip(current_yolo_results.boxes.cls, current_yolo_results.boxes.conf):
                            name = current_yolo_results.names[int(cls_id)]
                            if name not in ignored_objects:
                                detected_tools.append((name, float(conf)))
                    
                    if detected_tools:
                        # Sort by confidence
                        detected_tools.sort(key=lambda x: x[1], reverse=True)
                        tool_name = detected_tools[0][0]
                        display_label = f"using + {tool_name}"
                    else:
                        display_label = "using unknown"
                else:
                    display_label = "using unknown"

            # Create visualization
            frame_with_viz = create_visualization(
                frame=frame,
                action_label=display_label,
                yolo_results=current_yolo_results,
                relationship_lines=line_info,
                batch_params=batch_params
            )
            
            writer.write(frame_with_viz)
        
        writer.release()
        print(f"Saved labeled video: {video_output_path}")
        output_files['video'] = video_output_path
        
        # Finalize relationships
        if relationship_tracker:
            relationship_tracker.finalize(actual_frame_count)
    
    # ==========================================
    # 7. SAVE OUTPUTS
    # ==========================================
    
    print("\nSaving outputs...")
    
    # Save actions CSV
    if batch_params.save_actions_csv:
        actions_csv_path = save_actions_csv(
            video_name=video_name,
            fps=video_props.fps,
            frame_count=actual_frame_count,
            batch_params=batch_params,
            context_store=context_store,
            output_path=None
        )
        output_files['actions_csv'] = actions_csv_path
        print(f"Saved actions CSV: {actions_csv_path}")
    
    # Save relationships CSV
    if batch_params.save_relationships_csv and relationship_tracker:
        relationships_csv_path = save_relationships_csv(
            video_name,
            relationship_tracker.get_relationships_csv_data(),
            video_props.fps,
            batch_params
        )
        output_files['relationships_csv'] = relationships_csv_path
        print(f"Saved relationships CSV: {relationships_csv_path}")
    
    # Save batch metadata
    processing_time = time.time() - start_time
    metadata_path = save_batch_metadata(
        video_name,
        batch_params,
        processing_time,
        output_files
    )
    output_files['metadata'] = metadata_path
    
    # Print rate limiter stats if enabled
    if rate_limiter:
        stats = rate_limiter.get_stats()
        print(f"\nAPI Rate Limiting Stats:")
        print(f"  Total API requests: {stats['total_requests']}")
        print(f"  Configured limit: {stats['requests_per_minute']} req/min")
        print(f"  Min interval: {stats['min_interval']:.2f}s")
    
    print(f"\n{'='*60}")
    print(f"✓ Completed {video_name}")
    print(f"Processing time: {processing_time:.2f}s ({processing_time/60:.2f} min)")
    print(f"Speed ratio: {processing_time/video_props.duration:.2f}x realtime")
    print(f"Batch ID: {batch_params.batch_id}")
    print(f"{'='*60}\n")
    
    # Cleanup frame loader if streaming mode
    if hasattr(frame_cache, 'close'):
        stats = frame_cache.get_stats()
        print(f"\nFrame cache statistics:")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache misses: {stats['cache_misses']}")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")
        print(f"  Peak cache size: {stats['current_cache_size']}/{stats['max_cache_size']}")
        frame_cache.close()

    return output_files
