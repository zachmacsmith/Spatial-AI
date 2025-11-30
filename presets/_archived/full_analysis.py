from video_processing import BatchParameters
from video_processing.batch_parameters import (
    ToolDetectionMethod,
    ActionClassificationMethod,
    ProductivityAnalysisFormat
)

def get_name() -> str:
    return "Full Analysis"

def get_description() -> str:
    return "Full analysis with static charts (TestingClassFINAL.py equivalent)"

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="full",
        config_description=get_description(),
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_productivity_analysis=True,
        productivity_analysis_format=ProductivityAnalysisFormat.IMAGES,
        tool_detection_method=ToolDetectionMethod.CV_INFERENCE,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME
    )
