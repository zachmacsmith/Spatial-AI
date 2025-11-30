from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "13. Super Preset (Multi-Frame + Aggregation)"

def get_description() -> str:
    return "Combines Multi-Frame visual context with Interval Aggregation for maximum robustness."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="super_preset_sonnet",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-sonnet-4-5-20250929",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Multi-Frame State Check (Best for understanding action)
        state_check_method=StateCheckMethod.LLM_MULTIFRAME,
        multi_frame_count=1,
        multi_frame_gap=3,
        
        # Interval Aggregation Object Check (Best for not missing tools)
        # Note: We might want to combine Multi-Frame + Aggregation in the future,
        # but for now, Aggregation is the proven winner for object recall.
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LLM_WITH_INTERVAL_AGGREGATION,
        
        # CRITICAL: Run CV on every frame so we have data to aggregate
        cv_detection_frequency=5,
        
        # Temporal Majority for Unknowns
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
        
        # Enable Temporal Smoothing
        enable_temporal_smoothing=True,
        temporal_smoothing_window=9
    )
