from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "05. Relationship Aware (Spatial Context)"

def get_description() -> str:
    return "Uses spatial relationships (e.g., 'hand close to drill') to improve object identification."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="relationship_aware",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-haiku-4-5-20251001",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Standard State Check
        state_check_method=StateCheckMethod.LEGACY_TESTING_CLASS,
        
        # Relationship-Aware Object Detection
        enable_object_detection=True,
        enable_relationship_tracking=True,  # CRITICAL: Must be enabled
        object_check_method=ObjectCheckMethod.LLM_WITH_RELATIONSHIPS,
        
        # Temporal Majority for Unknowns
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
        
        # Enable Temporal Smoothing
        enable_temporal_smoothing=True,
        temporal_smoothing_window=9
    )
