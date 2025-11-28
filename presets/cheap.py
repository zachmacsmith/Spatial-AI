from video_processing import BatchParameters, PRESET_CHEAP

def get_name() -> str:
    return "Cheap (CV + Motion)"

def get_description() -> str:
    return "Lowest cost. Uses Motion Threshold and CV only. No LLM costs."

def get_batch_params() -> BatchParameters:
    return PRESET_CHEAP
