from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "15. Sensitive Strict Enhanced (Haiku)"

def get_description() -> str:
    return "Strict Enhanced logic with Haiku + DENSE keyframe sampling (Gap=5, MaxGap=150, Thresh=0.8)."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="sensitive_strict_enhanced_haiku",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-3-haiku-20240307", # Using Haiku for speed/cost
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Standard State Check
        state_check_method=StateCheckMethod.LEGACY_TESTING_CLASS,
        
        # Strict Object Detection
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LLM_STRICT,
        
        # DENSE KEYFRAME SETTINGS
        # Sensitive to motion (0.8) and frequent sampling (min 5, max 150)
        keyframe_min_gap=15,
        keyframe_max_gap=150, # Force keyframe every 5 seconds
        keyframe_threshold_multiplier=0.8,
        
        # Temporal Majority for Unknowns
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
        
        # Enable Temporal Smoothing
        enable_temporal_smoothing=True,
        temporal_smoothing_window=9
    )
