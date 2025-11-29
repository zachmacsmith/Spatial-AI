from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "11. Visual Heuristic (Zero Cost)"

def get_description() -> str:
    return "Zero LLM cost for state. Relies entirely on CV presence of tools and motion thresholds."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="visual_heuristic",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-haiku-4-5-20251001",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # CRITICAL CHANGE: CV + Motion only
        state_check_method=StateCheckMethod.CV_OBJECTS_ONLY,
        
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.CV_DETECTION, # Match with cheap object check
        unknown_object_check_method=UnknownObjectCheckMethod.CV_CLASS_NAME,
    )
