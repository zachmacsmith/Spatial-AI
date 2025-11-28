from video_processing import BatchParameters
from video_processing.batch_parameters import (
    LLMProvider,
    ToolDetectionMethod,
    ActionClassificationMethod
)

def get_name() -> str:
    return "Gemini 2.5 Flash"

def get_description() -> str:
    return "Relationships with Gemini 2.5 Flash (Experimental/Faster)"

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="relationships_gemini_flash",
        config_description=get_description(),
        llm_provider=LLMProvider.GEMINI,
        llm_model="gemini-2.5-flash",
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_productivity_analysis=False,
        tool_detection_method=ToolDetectionMethod.LLM_DIRECT,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME,
        cv_model_path="weights.pt",
        api_requests_per_minute=10,
        pricing_tier="free"
    )
