from video_processing import BatchParameters
from video_processing.batch_parameters import (
    LLMProvider,
    ToolDetectionMethod,
    ActionClassificationMethod
)

def get_name() -> str:
    return "Anthropic Haiku"

def get_description() -> str:
    return "Relationships with Claude 4.5 Haiku (Faster/Cheaper)"

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="relationships_anthropic_haiku",
        config_description=get_description(),
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-haiku-4-5-20251001",
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_productivity_analysis=False,
        tool_detection_method=ToolDetectionMethod.LLM_DIRECT,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME,
        cv_model_path="weights.pt",
        api_requests_per_minute=50,
        pricing_tier="pay_as_you_go"
    )
