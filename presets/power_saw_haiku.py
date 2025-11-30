from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "17. Power Saw Strict (Haiku)"

def get_description() -> str:
    return "Strict Enhanced with 'power saw' option, softened uncertainty prompt, and Aggregation. Uses Haiku."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="power_saw_haiku",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-3-haiku-20240307",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Standard State Check with Softened Prompt
        state_check_method=StateCheckMethod.LEGACY_SOFTENED,
        
        # Strict Softened Object Detection with Aggregation
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LLM_AGGREGATION_SOFTENED,
        
        # Modified Tool List
        # 1. Replace "saw" with "power saw"
        # 2. Remove "pencil"
        allowed_tools=[
            "brick trowel", "caulk gun", "drill", "power saw", "brick", "measuring", "hammer", "nail gun", "unknown"
        ],
        
        # Temporal Majority for Unknowns
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
        
        # Enable Temporal Smoothing
        enable_temporal_smoothing=True,
        temporal_smoothing_window=9
    )
