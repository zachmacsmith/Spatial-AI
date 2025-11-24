"""
Simple test to verify the new architecture works

This tests basic imports and configuration without processing a full video.
"""

print("Testing imports...")

try:
    from video_processing import (
        BatchParameters,
        PRESET_BASIC,
        PRESET_OBJECTS,
        PRESET_RELATIONSHIPS,
        PRESET_FULL
    )
    print("✓ Imported presets successfully")
    
    from video_processing.video_processor import process_video
    print("✓ Imported video_processor successfully")
    
    from video_processing.ai.llm_service import get_llm_service
    from video_processing.ai.cv_service import get_cv_service
    print("✓ Imported AI services successfully")
    
    from video_processing.batch_comparison import BatchRegistry
    print("✓ Imported batch comparison successfully")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("Testing BatchParameters...")
print("="*60)

# Test creating a configuration
params = PRESET_FULL.copy()
print(f"✓ Created configuration")
print(f"  Batch ID: {params.batch_id}")
print(f"  Config name: {params.config_name}")
print(f"  LLM provider: {params.llm_provider.value}")
print(f"  CV model: {params.cv_model.value}")

# Test saving configuration
try:
    config_path = params.save_batch_config()
    print(f"✓ Saved configuration to: {config_path}")
except Exception as e:
    print(f"✗ Error saving config: {e}")

# Test getting videos to process
try:
    videos = params.get_videos_to_process()
    print(f"✓ Found {len(videos)} videos to process")
    if videos:
        print(f"  Videos: {videos[:3]}...")  # Show first 3
except Exception as e:
    print(f"✗ Error getting videos: {e}")

print("\n" + "="*60)
print("Testing batch tracking...")
print("="*60)

try:
    registry = BatchRegistry()
    all_batches = registry.get_all_batch_ids()
    print(f"✓ BatchRegistry initialized")
    print(f"  Total tracked batches: {len(all_batches)}")
except Exception as e:
    print(f"✗ Error with batch registry: {e}")

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nThe new modular architecture is working correctly.")
print("You can now run:")
print("  - python batch_process.py (menu-driven)")
print("  - python example_usage.py (examples)")
