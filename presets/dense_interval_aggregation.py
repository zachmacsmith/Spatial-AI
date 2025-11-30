from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "14. Dense Interval Aggregation (Sonnet)"

def get_description() -> str:
    return "Interval Aggregation with DENSE keyframe sampling (Gap=4, Thresh=0.8)."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="dense_interval_aggregation_sonnet",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-sonnet-4-5-20250929", # Using Sonnet for best reasoning
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Standard State Check
        state_check_method=StateCheckMethod.LEGACY_TESTING_CLASS,
        
        # Aggregated Object Detection
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LLM_WITH_INTERVAL_AGGREGATION,
        
        # CRITICAL: Run CV on every frame so we have data to aggregate
        cv_detection_frequency=5,
        
        # DENSE KEYFRAME SETTINGS
        # Default is usually gap=20, thresh=1.0
        # We want MORE keyframes -> Lower gap, Lower threshold
        keyframe_min_gap=15,
        keyframe_max_gap=150, # Force keyframe every 150 frames (5 sec) even if no motion
        keyframe_threshold_multiplier=0.8,
        
        # Temporal Majority for Unknowns
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
        
        # Enable Temporal Smoothing
        enable_temporal_smoothing=True,
        temporal_smoothing_window=9
    )
