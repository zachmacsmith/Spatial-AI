from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "08. Action Mapped State (Sonnet)"

def get_description() -> str:
    return "Prioritizes deterministic rules from ActionMapper (e.g., 'drill' -> 'using tool') before asking LLM."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="action_mapped_state_sonnet",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-sonnet-4-5-20250929",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # CRITICAL CHANGE: Use Action Mapper logic
        # Logic: Checks strict object->action maps first, then hints LLM
        state_check_method=StateCheckMethod.ACTION_MAPPING, 
        
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.LLM_STRICT,
        unknown_object_check_method=UnknownObjectCheckMethod.TEMPORAL_MAJORITY,
    )
