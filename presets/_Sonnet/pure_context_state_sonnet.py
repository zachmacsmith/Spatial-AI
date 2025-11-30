from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "09. Pure Context (Sonnet)"

def get_description() -> str:
    return "Ignores motion thresholds completely. Asks LLM to classify state based purely on visual context."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="pure_context_state_sonnet",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-sonnet-4-5-20250929",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # CRITICAL CHANGE: Direct LLM call for state
        # Logic: 1 API call per frame, no motion heuristics
        state_check_method=StateCheckMethod.LLM_DIRECT,
        
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LLM_STRICT,
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
    )
