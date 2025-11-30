from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "16. Raw LLM (No CV/Motion)"

def get_description() -> str:
    return "Pure LLM analysis. No object detection hints, no motion thresholds. Just the raw image."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="raw_llm_sonnet",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-sonnet-4-5-20250929",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Pure LLM State Check
        state_check_method=StateCheckMethod.LLM_DIRECT,
        
        # Pure LLM Object Check
        # We still enable object detection to allow the pipeline to run, 
        # but LLM_DIRECT ignores the CV results in the prompt.
        enable_object_detection=True, 
        object_check_method=ObjectCheckMethod.LLM_DIRECT,
        
        # Unknowns
        unknown_object_check_method=UnknownObjectCheckMethod.LLM_GUESS,
        
        # Disable temporal smoothing to see raw raw results
        enable_temporal_smoothing=False
    )
