from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "04. Strict Enhanced (Sonnet)"

def get_description() -> str:
    return "Enhanced Temporal with strict prompting constraints (no reasoning, exact option match only)."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="strict_enhanced_sonnet",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-sonnet-4-5-20250929",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Standard State Check
        state_check_method=StateCheckMethod.LEGACY_TESTING_CLASS,
        
        # Strict Object Detection
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LLM_STRICT,
        
        # Temporal Majority for Unknowns
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
        
        # Enable Temporal Smoothing
        enable_temporal_smoothing=True,
        temporal_smoothing_window=9
    )
