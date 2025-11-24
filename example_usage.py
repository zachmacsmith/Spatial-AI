"""
Example Usage - Demonstrates how to use the new modular architecture

This shows how to process videos with different configurations.
"""

from video_processing import (
    BatchParameters,
    PRESET_BASIC,
    PRESET_OBJECTS,
    PRESET_RELATIONSHIPS,
    PRESET_FULL
)
from video_processing.video_processor import process_video


# ==========================================
# Example 1: Use a preset configuration
# ==========================================

def example_basic():
    """Process video with basic configuration (no object detection)"""
    params = PRESET_BASIC
    outputs = process_video("video_01", params)
    print(f"Outputs: {outputs}")


def example_full():
    """Process video with full configuration"""
    params = PRESET_FULL
    outputs = process_video("video_01", params)
    print(f"Outputs: {outputs}")


# ==========================================
# Example 2: Custom configuration
# ==========================================

def example_custom():
    """Create custom configuration"""
    params = BatchParameters(
        config_name="custom_experiment",
        config_description="Testing Gemini with YOLO v8",
        
        # Use Gemini instead of Claude
        llm_provider="gemini",
        llm_model="gemini-1.5-pro",
        
        # Use YOLO v8
        cv_model="yolo_v8",
        cv_model_path="weights_v8.pt",
        
        # Enable all features
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_productivity_analysis=False,  # Not yet implemented
        
        # Custom parameters
        cv_confidence_threshold=0.7,  # Higher threshold
        max_workers_keyframes=16,  # More parallelism
    )
    
    outputs = process_video("video_01", params)
    print(f"Batch ID: {params.batch_id}")
    print(f"Outputs: {outputs}")


# ==========================================
# Example 3: Batch processing multiple videos
# ==========================================

def example_batch():
    """Process multiple videos with same configuration"""
    params = PRESET_FULL
    
    videos = ["video_01", "video_02", "video_03"]
    
    for video_name in videos:
        print(f"\nProcessing {video_name}...")
        outputs = process_video(video_name, params)
        print(f"Completed {video_name}: {outputs['actions_csv']}")
    
    print(f"\nAll videos processed with batch_id: {params.batch_id}")


# ==========================================
# Example 4: Compare different CV models
# ==========================================

def example_compare_cv_models():
    """Compare different CV models with everything else the same"""
    from video_processing import CVModel
    
    cv_models = [CVModel.YOLO_CURRENT, CVModel.YOLO_V8, CVModel.YOLO_V9]
    
    for cv_model in cv_models:
        params = PRESET_FULL.copy()  # Creates new batch_id
        params.cv_model = cv_model
        params.cv_model_path = f"weights_{cv_model.value}.pt"
        params.config_name = f"cv_comparison_{cv_model.value}"
        params.experiment_id = "cv_model_comparison_2025_11"
        
        # Save configuration for later analysis
        params.save_batch_config()
        
        outputs = process_video("video_01", params)
        print(f"Processed with {cv_model.value}, batch_id: {params.batch_id}")


# ==========================================
# Example 5: Load configuration from file
# ==========================================

def example_load_config():
    """Load configuration from JSON file"""
    # First, save a configuration
    params = PRESET_FULL
    params.config_name = "my_experiment"
    params.to_json("configs/custom/my_experiment.json")
    
    # Later, load it back
    loaded_params = BatchParameters.from_json("configs/custom/my_experiment.json")
    outputs = process_video("video_01", loaded_params)
    print(f"Processed with loaded config: {outputs}")


# ==========================================
# Example 6: Reproduce exact run from batch_id
# ==========================================

def example_reproduce_run():
    """Reproduce exact configuration from previous run"""
    # Load configuration by batch_id
    batch_id = "batch_20251122_210000_a1b2c3d4"  # Example batch_id
    
    params = BatchParameters.from_batch_id(batch_id)
    outputs = process_video("video_01", params)
    print(f"Reproduced run with batch_id: {batch_id}")


# ==========================================
# Example 7: Process only specific videos
# ==========================================

def example_specific_videos():
    """Process only specific videos from directory"""
    params = PRESET_FULL
    
    # Option 1: Specify videos in parameters
    params.videos_to_process = ["video_01", "video_05", "video_10"]
    
    for video_name in params.get_videos_to_process():
        outputs = process_video(video_name, params)
        print(f"Processed {video_name}")


if __name__ == "__main__":
    # Run an example
    print("Running example: Full configuration")
    example_full()
    
    # To run other examples, uncomment:
    # example_custom()
    # example_batch()
    # example_compare_cv_models()
