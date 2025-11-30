from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "98. Baseline: Random Guess"

def get_description() -> str:
    return "Stochastic baseline. Guesses randomly from allowed lists. Used to determine the statistical floor."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="baseline_random",
        llm_provider=LLMProvider.CLAUDE, # Unused but required
        llm_model="claude-haiku-4-5-20251001",
        api_requests_per_minute=1000,
        prompting_protocol=PromptingProtocolType.CASCADE,
        
        # Random Logic
        state_check_method=StateCheckMethod.RANDOM,
        
        enable_object_detection=True, # "True" so the pipeline runs, but logic is random
        object_check_method=ObjectCheckMethod.RANDOM,
        unknown_object_check_method=UnknownObjectCheckMethod.SKIP,
        
        # Disable smoothing so we see raw random noise
        enable_temporal_smoothing=False
    )