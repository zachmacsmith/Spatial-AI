from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "99.Baseline: Majority Class (ZeroR)"

def get_description() -> str:
    return "Static baseline. Always guesses the most common class (ZeroR). The true benchmark for utility."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="baseline_majority",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-haiku-4-5-20251001",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Static Logic (Majority Class)
        # Assumes 'using tool' and 'hammer' (or configured tool) are the modes
        state_check_method=StateCheckMethod.STATIC_USING_TOOL,
        
        enable_object_detection=True,
        object_check_method=ObjectCheckMethod.STATIC_COMMON,
        unknown_object_check_method=UnknownObjectCheckMethod.SKIP,
        
        enable_temporal_smoothing=True # Smoothing helps static predictions stay static
    )