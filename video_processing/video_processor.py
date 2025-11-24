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
    create_video_writer
)
from .utils.visualization import create_visualization
from .output.output_manager import (
    save_actions_csv,
    save_relationships_csv,
    save_batch_metadata
)


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


def process_video(video_name: str, batch_params) -> Dict[str, str]:
    """
    Main video processing function.
    
    This orchestrates the entire pipeline:
    1. Load video and frames
    2. Process keyframes (action classification)
    3. Process intervals between keyframes
    4. Apply temporal smoothing
    5. Generate labeled video (optional)
    6. Save outputs (CSVs, metadata)
    
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
    keyframe_numbers = load_keyframe_numbers(keyframes_folder)
    print(f"Keyframes: {len(keyframe_numbers)}")
    
    # Pre-load all frames if enabled
    if batch_params.preload_all_frames:
        print("Pre-loading all frames...")
        frame_cache = load_all_frames(video_path)
        print(f"Loaded {len(frame_cache)} frames into cache")
    else:
        # TODO: Implement on-demand frame loading
        raise NotImplementedError("On-demand frame loading not yet implemented")
    
    # Use actual loaded frame count (not reported metadata count)
    actual_frame_count = len(frame_cache)
    
    # Initialize frame labels based on actual frames
    frame_labels = [""] * actual_frame_count
    
    # Initialize services
    llm_service = get_llm_service(batch_params)
    prompt_builder = PromptBuilder(batch_params)
    
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
    # 2. PROCESS KEYFRAMES
    # ==========================================
    
    print(f"\nProcessing {len(keyframe_numbers)} keyframes...")
    
    # Choose processing strategy: batch or sequential
    if batch_params.enable_batch_processing:
        # BATCH PROCESSING MODE (Efficient!)
        from .api_request_batcher import create_api_request_batcher, APIBatchRequest
        
        api_batcher = create_api_request_batcher(batch_params)
        
        # Collect all keyframe requests
        for kf_number in keyframe_numbers:
            frame = get_frame_from_cache(frame_cache, kf_number)
            
            # Detect objects if enabled
            detected_objects = None
            if cv_service:
                detections = cv_service.detect_objects(frame, batch_params.cv_confidence_threshold)
                detected_objects = [(name, conf) for name, conf, box in detections]
            
            # Build prompt
            prompt_text = prompt_builder.build_action_classification_prompt(
                motion_score=None,
                detected_objects=detected_objects
            )
            
            # Add to batch
            api_batcher.add_request(APIBatchRequest(
                request_id=f"keyframe_{kf_number}",
                frames=[frame],
                prompt_text=prompt_text,
                context={'detected_objects': detected_objects}
            ))
        
        # Process all keyframes in batches
        results = api_batcher.process_all(llm_service, prompt_builder)
        
        # Store results
        for kf_number in keyframe_numbers:
            label = results.get(f"keyframe_{kf_number}", "idle")
            frame_labels[kf_number - 1] = label
            print(f"Keyframe {kf_number}: {label}")
    
    else:
        # SEQUENTIAL PROCESSING MODE (Original)
        def process_keyframe(kf_number: int) -> tuple:
            """Process single keyframe"""
            # Apply rate limiting before API call
            if rate_limiter:
                rate_limiter.wait_if_needed()
            
            frame = get_frame_from_cache(frame_cache, kf_number)
            
            # Detect objects if enabled
            detected_objects = None
            if cv_service:
                detections = cv_service.detect_objects(frame, batch_params.cv_confidence_threshold)
                detected_objects = [(name, conf) for name, conf, box in detections]
            
            # Classify action
            label = classify_action(
                frames=[frame],
                batch_params=batch_params,
                llm_service=llm_service,
                prompt_builder=prompt_builder,
                motion_score=None,
                detected_objects=detected_objects
            )
            
            print(f"Keyframe {kf_number}: {label}")
            return kf_number, label
        
        # Process keyframes in parallel
        with ThreadPoolExecutor(max_workers=batch_params.max_workers_keyframes) as executor:
            keyframe_results = list(executor.map(process_keyframe, keyframe_numbers))
        
        # Store keyframe results
        for kf_number, label in keyframe_results:
            frame_labels[kf_number - 1] = label
    
    # ==========================================
    # 3. PROCESS INTERVALS
    # ==========================================
    
    print(f"\nProcessing {len(keyframe_numbers) - 1} intervals...")
    
    # Choose processing strategy: batch or sequential
    if batch_params.enable_batch_processing:
        # BATCH PROCESSING MODE (Efficient!)
        from .api_request_batcher import APIBatchRequest
        
        if 'api_batcher' not in locals():
            from .api_request_batcher import create_api_request_batcher
            api_batcher = create_api_request_batcher(batch_params)
        else:
            api_batcher.clear()  # Reuse processor, clear previous results
        
        # Collect all interval requests
        for idx in range(len(keyframe_numbers) - 1):
            start = keyframe_numbers[idx]
            end = keyframe_numbers[idx + 1]
            
            # Sample frames from interval
            frames_to_send = sample_interval_frames(
                frame_cache, start, end, batch_params.num_frames_per_interval
            )
            
            # Include neighbor frames if enabled
            if batch_params.include_neighbor_frames:
                if start > 1:
                    frames_to_send.insert(0, get_frame_from_cache(frame_cache, max(1, start - 1)))
                if end < video_props.frame_count:
                    frames_to_send.append(get_frame_from_cache(frame_cache, min(video_props.frame_count, end + 1)))
            
            # Calculate motion score
            motion = calculate_motion_score(frames_to_send, batch_params.motion_ignore_threshold)
            
            # Detect objects if enabled
            detected_objects = None
            if cv_service:
                middle_frame = frames_to_send[len(frames_to_send) // 2]
                detections = cv_service.detect_objects(middle_frame, batch_params.cv_confidence_threshold)
                detected_objects = [(name, conf) for name, conf, box in detections]
            
            # Build prompt
            prompt_text = prompt_builder.build_action_classification_prompt(
                motion_score=motion,
                detected_objects=detected_objects
            )
            
            # Add to batch
            api_batcher.add_request(APIBatchRequest(
                request_id=f"interval_{start}_{end}",
                frames=frames_to_send,
                prompt_text=prompt_text,
                context={'motion': motion, 'detected_objects': detected_objects}
            ))
        
        # Process all intervals in batches
        results = api_batcher.process_all(llm_service, prompt_builder)
        
        # Fill in interval labels
        for idx in range(len(keyframe_numbers) - 1):
            start = keyframe_numbers[idx]
            end = keyframe_numbers[idx + 1]
            label = results.get(f"interval_{start}_{end}", "idle")
            for f in range(start, end):
                frame_labels[f] = label
    
    else:
        # SEQUENTIAL PROCESSING MODE (Original)
        def process_interval(idx: int) -> tuple:
            """Process interval between keyframes"""
            # Apply rate limiting before API call
            if rate_limiter:
                rate_limiter.wait_if_needed()
            
            start = keyframe_numbers[idx]
            end = keyframe_numbers[idx + 1]
            
            # Sample frames from interval
            frames_to_send = sample_interval_frames(
                frame_cache, start, end, batch_params.num_frames_per_interval
            )
            
            # Include neighbor frames if enabled
            if batch_params.include_neighbor_frames:
                if start > 1:
                    frames_to_send.insert(0, get_frame_from_cache(frame_cache, max(1, start - 1)))
                if end < video_props.frame_count:
                    frames_to_send.append(get_frame_from_cache(frame_cache, min(video_props.frame_count, end + 1)))
            
            # Calculate motion score
            motion = calculate_motion_score(frames_to_send, batch_params.motion_ignore_threshold)
            
            # Detect objects if enabled
            detected_objects = None
            if cv_service:
                # Get objects from middle frame
                middle_frame = frames_to_send[len(frames_to_send) // 2]
                detections = cv_service.detect_objects(middle_frame, batch_params.cv_confidence_threshold)
                detected_objects = [(name, conf) for name, conf, box in detections]
            
            # Classify action
            label = classify_action(
                frames=frames_to_send,
                batch_params=batch_params,
                llm_service=llm_service,
                prompt_builder=prompt_builder,
                motion_score=motion,
                detected_objects=detected_objects
            )
            
            return start, end, label
        
        # Process intervals in parallel
        with ThreadPoolExecutor(max_workers=batch_params.max_workers_intervals) as executor:
            interval_results = list(executor.map(process_interval, range(len(keyframe_numbers) - 1)))
        
        # Fill in interval labels
        for start, end, label in interval_results:
            for f in range(start, end):
                frame_labels[f] = label
    
    # Fill after last keyframe
    last_kf = keyframe_numbers[-1]
    for f in range(last_kf, actual_frame_count):
        frame_labels[f] = frame_labels[last_kf - 1]
    
    # ==========================================
    # 4. TEMPORAL SMOOTHING
    # ==========================================
    
    if batch_params.enable_temporal_smoothing:
        print("\nApplying temporal smoothing...")
        frame_labels = apply_temporal_smoothing(
            frame_labels,
            batch_params.allowed_actions,
            batch_params.temporal_smoothing_window
        )
    
    # ==========================================
    # 5. GENERATE LABELED VIDEO (OPTIONAL)
    # ==========================================
    
    output_files = {}
    
    if batch_params.generate_labeled_video:
        print("\nGenerating labeled video...")
        
        video_output_path = os.path.join(
            batch_params.video_output_directory,
            f"{video_name}.mp4"
        )
        
        writer = create_video_writer(
            video_output_path,
            video_props.fps,
            video_props.width,
            video_props.height,
            batch_params.video_codec
        )
        
        latest_yolo_results = None
        latest_detections = []
        current_tool = None
        
        for i in range(actual_frame_count):
            frame = get_frame_from_cache(frame_cache, i + 1)
            
            # Run CV detection every N frames
            if cv_service and i % batch_params.cv_detection_frequency == 0:
                latest_yolo_results = cv_service.get_results_object(
                    frame, batch_params.cv_confidence_threshold
                )
                
                # Extract detections for relationship tracking
                if latest_yolo_results:
                    latest_detections = []
                    for box, cls_id, conf in zip(
                        latest_yolo_results.boxes.xyxy,
                        latest_yolo_results.boxes.cls,
                        latest_yolo_results.boxes.conf
                    ):
                        class_name = latest_yolo_results.names[int(cls_id)]
                        box_tuple = tuple(map(int, box.cpu().numpy()))
                        latest_detections.append((box_tuple, class_name, float(conf)))
                    
                    # Detect tool if enabled
                    if batch_params.enable_tool_detection:
                        detected_objects = [(name, conf) for box, name, conf in latest_detections]
                        current_tool = detect_tool(
                            frames=[frame],
                            batch_params=batch_params,
                            llm_service=llm_service,
                            prompt_builder=prompt_builder,
                            detected_objects=detected_objects
                        )
            
            # Track relationships if enabled
            line_info = []
            if relationship_tracker and latest_detections:
                relationships, line_info = relationship_tracker.find_relationships(
                    latest_detections, video_props.width, i + 1
                )
                relationship_tracker.update(relationships, i + 1)
            
            # Prepare action label
            action_label = frame_labels[i]
            if action_label == "using tool" and current_tool:
                action_label = f"using {current_tool}"
            
            # Create visualization
            frame_with_viz = create_visualization(
                frame=frame,
                action_label=action_label,
                yolo_results=latest_yolo_results,
                relationship_lines=line_info,
                batch_params=batch_params
            )
            
            out.write(annotated_frame)
        
        out.release()
        print(f"Saved labeled video: {video_output_path}")
        output_files['video'] = video_output_path
        
        # Finalize relationships
        if relationship_tracker:
            relationship_tracker.finalize(actual_frame_count)
    
    # ==========================================
    # 6. SAVE OUTPUTS
    # ==========================================
    
    print("\nSaving outputs...")
    
    # Save actions CSV
    if batch_params.save_actions_csv:
        actions_csv_path = save_actions_csv(
            video_name,
            frame_labels,
            video_props.fps,
            actual_frame_count,
            batch_params
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
        print(f"Total relationships: {len(relationship_tracker.get_relationships_csv_data())}")
    
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
    
    return output_files
