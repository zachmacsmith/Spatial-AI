from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "11. Multi-Frame State (Sonnet)"

def get_description() -> str:
    return "Uses multi-frame context (prev/next frames) for state detection. Standard object detection."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="multiframe_state_sonnet",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-sonnet-4-5-20250929",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Multi-Frame State Check
        state_check_method=StateCheckMethod.LLM_MULTIFRAME,
        
        # Standard Object Detection
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LLM_STRICT,
        
        # Multi-Frame Settings
        multi_frame_count=1,  # 1 frame before, 1 frame after
        multi_frame_gap=3,    # Gap of 3 frames
        
        # Temporal Majority for Unknowns
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
        
        # Enable Temporal Smoothing
        enable_temporal_smoothing=True,
        temporal_smoothing_window=5
    )
