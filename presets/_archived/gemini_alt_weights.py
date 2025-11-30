from video_processing import BatchParameters
from video_processing.batch_parameters import (
    LLMProvider,
    ToolDetectionMethod,
    ActionClassificationMethod
)

def get_name() -> str:
    return "Gemini Alt Weights"

def get_description() -> str:
    return "Relationships with Gemini and alternative CV model (weights_v2.pt)"

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="relationships_gemini_alt_weights",
        config_description=get_description(),
        llm_provider=LLMProvider.GEMINI,
        llm_model="gemini-2.0-flash-lite",
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_productivity_analysis=False,
        tool_detection_method=ToolDetectionMethod.LLM_DIRECT,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME,
        cv_model_path="weights_v2.pt",
        api_requests_per_minute=30,
        pricing_tier="free"
    )
