from video_processing import BatchParameters, PRESET_BASELINE

def get_name() -> str:
    return "Baseline (Single Shot)"

def get_description() -> str:
    return "Legacy behavior. Single prompt for action classification. High cost, good accuracy."

def get_batch_params() -> BatchParameters:
    return PRESET_BASELINE
