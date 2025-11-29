from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "Enhanced Temporal (CV + Smoothing)"

def get_description() -> str:
    return "Combines CV-based object detection (TestingClass2) with Temporal Smoothing and Majority Voting (Legacy Temporal)."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="enhanced_temporal",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-sonnet-4-5-20250929",
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Legacy State Check (Motion + LLM)
        state_check_method=StateCheckMethod.LEGACY_TESTING_CLASS,
        
        # CV Object Detection (TestingClass2 style)
        # Now upgraded with ActionMapper support (measuring, etc.)
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LLM_WITH_CV_HINT,
        
        # Temporal Majority for Unknowns (Legacy Temporal style)
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
        
        # Enable Temporal Smoothing
        enable_temporal_smoothing=True,
        temporal_smoothing_window=9
    )
