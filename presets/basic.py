from video_processing import BatchParameters
from video_processing.batch_parameters import (
    ToolDetectionMethod,
    ActionClassificationMethod
)

def get_name() -> str:
    return "Basic (Action Only)"

def get_description() -> str:
    return "Basic action classification only (TestingClass.py equivalent)"

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="basic",
        config_description=get_description(),
        enable_object_detection=False,
        enable_relationship_tracking=False,
        enable_productivity_analysis=False,
        tool_detection_method=ToolDetectionMethod.LLM_DIRECT,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME
    )
