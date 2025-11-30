from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "10. Hybrid Efficiency"

def get_description() -> str:
    return "Trusts extreme motion scores (Very Idle/Active) to save costs; asks LLM only for ambiguous frames."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="hybrid_efficiency",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-haiku-4-5-20251001",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # CRITICAL CHANGE: Hybrid Logic
        # Logic: 0 API calls for clear motion, 1 API call for edge cases
        state_check_method=StateCheckMethod.HYBRID_MOTION_THEN_LLM,
        
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LLM_STRICT,
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
    )
