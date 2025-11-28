from video_processing import BatchParameters, PRESET_THOROUGH

def get_name() -> str:
    return "Thorough (LLM Heavy)"

def get_description() -> str:
    return "Maximum accuracy. Uses LLM for all decisions, with CV hints. High cost."

def get_batch_params() -> BatchParameters:
    return PRESET_THOROUGH
