"""
Quick test with a real video using BASIC preset

This will process video_01 with the BASIC preset to verify everything works.
"""

from video_processing import PRESET_BASIC
from video_processing.video_processor import process_video

print("="*60)
print("QUICK TEST - Processing video_01 with BASIC preset")
print("="*60)

# Use BASIC preset (no object detection, just action classification)
params = PRESET_BASIC.copy()

print(f"\nBatch ID: {params.batch_id}")
print(f"Config: {params.config_name}")
print(f"LLM: {params.llm_provider.value}")
print(f"Object detection: {params.enable_object_detection}")
print(f"Relationship tracking: {params.enable_relationship_tracking}")

try:
    print("\nProcessing video_01...")
    outputs = process_video("video_01", params)
    
    print("\n" + "="*60)
    print("✓ SUCCESS!")
    print("="*60)
    print(f"Actions CSV: {outputs['actions_csv']}")
    print(f"Metadata: {outputs['metadata']}")
    if 'video' in outputs:
        print(f"Labeled video: {outputs['video']}")
    
except Exception as e:
    print("\n" + "="*60)
    print("✗ ERROR")
    print("="*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
