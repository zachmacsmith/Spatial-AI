from video_processing import (
    BatchParameters, 
    PromptingProtocolType, 
    StateCheckMethod, 
    ObjectCheckMethod, 
    UnknownObjectCheckMethod,
    LLMProvider
)

def get_name() -> str:
    return "Legacy (TestingClass.py)"

def get_description() -> str:
    return "Exact replication of the original TestingClass.py logic (specific prompts & motion thresholds)."

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="legacy",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-sonnet-4-5-20250929",
        prompting_protocol=PromptingProtocolType.CASCADE,
        state_check_method=StateCheckMethod.LEGACY_TESTING_CLASS,
        object_check_method=ObjectCheckMethod.LEGACY_TESTING_CLASS,
        unknown_object_check_method=UnknownObjectCheckMethod.LLM_GUESS, # Legacy didn't have explicit unknown logic, likely just guessed or failed
        
        # Ensure context builder is compatible if needed, but legacy functions build their own prompts mostly
        # The legacy functions use ctx.frame_context.motion_score directly
    )
