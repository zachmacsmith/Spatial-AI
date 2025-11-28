from video_processing import BatchParameters
from video_processing.batch_parameters import (
    LLMProvider,
    ToolDetectionMethod,
    ActionClassificationMethod
)

def get_name() -> str:
    return "Gemini Relationships (Baseline)"

def get_description() -> str:
    return "Relationships tracking with Gemini 2.0 Flash Lite (Standard)"

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="relationships_gemini",
        config_description=get_description(),
        llm_provider=LLMProvider.GEMINI,
        llm_model="gemini-2.0-flash-lite",
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_productivity_analysis=False,
        tool_detection_method=ToolDetectionMethod.LLM_DIRECT,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME,
        cv_model_path="weights.pt",
        api_requests_per_minute=30,
        pricing_tier="free"
    )
