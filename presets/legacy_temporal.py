from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "Legacy + Temporal Majority"

def get_description() -> str:
    return "Legacy logic but uses temporal majority voting for unknown objects."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="legacy_temporal",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-sonnet-4-5-20250929",
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Legacy State Check (Motion + LLM)
        state_check_method=StateCheckMethod.LEGACY_TESTING_CLASS,
        
        # YOLO Object Detection (Like TestingClass3)
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LEGACY_TESTING_CLASS,
        
        # Temporal Majority for Unknowns
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
    )
