from video_processing import BatchParameters, PRESET_BALANCED

def get_name() -> str:
    return "Balanced (Hybrid)"

def get_description() -> str:
    return "Good trade-off. Uses Motion/CV first, falls back to LLM for complex cases."

def get_batch_params() -> BatchParameters:
    return PRESET_BALANCED
