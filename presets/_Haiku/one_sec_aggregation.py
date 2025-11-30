from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "07. 1-Second Aggregation"

def get_description() -> str:
    return "Aggregates object detections over the past 1 second of video."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="one_sec_aggregation",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-haiku-4-5-20251001",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Standard State Check
        state_check_method=StateCheckMethod.LEGACY_TESTING_CLASS,
        
        # Aggregated Object Detection
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LLM_WITH_1SEC_AGGREGATION,
        
        # CRITICAL: Run CV on every frame so we have data to aggregate
        cv_detection_frequency=5,
        
        # Temporal Majority for Unknowns
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
        
        # Enable Temporal Smoothing
        enable_temporal_smoothing=True,
        temporal_smoothing_window=9
    )
