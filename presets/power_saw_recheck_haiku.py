from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "18. Power Saw Recheck (Haiku)"

def get_description() -> str:
    return "Two-step object check: Strict (softened) -> Aggregation Recheck on unknown. Uses Haiku."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="power_saw_recheck_haiku",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-3-haiku-20240307",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Standard State Check (Reverted to Strict Enhanced default)
        state_check_method=StateCheckMethod.LEGACY_TESTING_CLASS,
        
        # Two-Step Object Detection
        # 1. Strict Check (Current Frame)
        # 2. Recheck with Aggregation (if unknown)
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LLM_TWO_STEP_RECHECK,
        
        # Modified Tool List
        # 1. Replace "saw" with "power saw"
        # 2. Remove "pencil"
        allowed_tools=[
            "brick trowel", "caulk gun", "drill", "power saw", "brick", "measuring", "hammer", "nail gun", "unknown"
        ],
        
        # Skip standard unknown check because the Two-Step method handles it
        unknown_object_check_method=UnknownObjectCheckMethod.SKIP,
        
        # Enable Temporal Smoothing
        enable_temporal_smoothing=True,
        temporal_smoothing_window=9
    )
