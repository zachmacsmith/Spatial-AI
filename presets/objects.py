from video_processing import BatchParameters
from video_processing.batch_parameters import (
    ToolDetectionMethod,
    ActionClassificationMethod
)

def get_name() -> str:
    return "Objects (YOLO)"

def get_description() -> str:
    return "With YOLO object detection (TestingClass2.py equivalent)"

def get_batch_params() -> BatchParameters:
    return BatchParameters(
        config_name="objects",
        config_description=get_description(),
        enable_object_detection=True,
        enable_relationship_tracking=False,
        enable_productivity_analysis=False,
        tool_detection_method=ToolDetectionMethod.CV_INFERENCE,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME
    )
