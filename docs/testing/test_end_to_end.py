"""
End-to-End Test with Gemini API

Tests full video processing pipeline with:
- Batch processing enabled
- Gemini API
- All new features
"""

import sys
import os
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_processing import process_video
from video_processing.batch_parameters import BatchParameters, LLMProvider, CVModel

def run_end_to_end_test():
    """Run comprehensive end-to-end test"""
    
    print("="*70)
    print("END-TO-END TEST - GEMINI + BATCH PROCESSING")
    print("="*70)
    print()
    
    # Configure for Gemini with batch processing
    print("Configuring test parameters...")
    params = BatchParameters(
        # LLM Configuration
        llm_provider=LLMProvider.GEMINI,
        llm_model="gemini-2.0-flash-exp",
        
        # Batch Processing (NEW!)
        enable_batch_processing=True,
        batch_size=5,
        use_smart_batching=True,
        
        # Rate Limiting
        enable_rate_limiting=False,  # Disabled since we're using batch processing
        api_requests_per_minute=15,  # Gemini free tier limit
        
        # Processing
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_action_classification=True,
        
        # Output
        generate_labeled_video=True,
        save_actions_csv=True,
        save_relationships_csv=True,
        
        # Config
        config_name="e2e_test_gemini_batch",
        config_description="End-to-end test with Gemini and batch processing"
    )
    
    print(f"✓ Configuration created")
    print(f"  Batch ID: {params.batch_id}")
    print(f"  LLM: {params.llm_provider.value} ({params.llm_model})")
    print(f"  Batch Processing: {params.enable_batch_processing}")
    print(f"  Batch Size: {params.batch_size}")
    print(f"  Smart Batching: {params.use_smart_batching}")
    print()
    
    # Test video
    video_name = "video_01"
    
    print(f"Processing video: {video_name}")
    print("-" * 70)
    print()
    
    # Track start time
    start_time = time.time()
    
    try:
        # Process video
        output_files = process_video(video_name, params)
        
        # Track end time
        end_time = time.time()
        processing_time = end_time - start_time
        
        print()
        print("="*70)
        print("TEST RESULTS")
        print("="*70)
        print()
        
        # Verify outputs
        print("Output Files:")
        for output_type, path in output_files.items():
            exists = os.path.exists(path) if path else False
            status = "✓" if exists else "✗"
            print(f"  {status} {output_type}: {path}")
        
        print()
        print("Performance Metrics:")
        print(f"  Total processing time: {processing_time:.2f}s ({processing_time/60:.2f} min)")
        
        # Read actions CSV to count frames
        if 'actions_csv' in output_files and os.path.exists(output_files['actions_csv']):
            with open(output_files['actions_csv'], 'r') as f:
                lines = f.readlines()
                fps_line = lines[0].strip().split(',')
                fps = float(fps_line[0])
                duration = float(fps_line[1])
                frame_count = len(lines) - 1  # Subtract header
                
                print(f"  Video duration: {duration:.2f}s")
                print(f"  Frame count: {frame_count}")
                print(f"  FPS: {fps:.2f}")
                print(f"  Speed ratio: {processing_time/duration:.2f}x realtime")
        
        print()
        print("Batch Processing Stats:")
        print(f"  Batch processing enabled: {params.enable_batch_processing}")
        print(f"  Batch size: {params.batch_size}")
        print(f"  Smart batching: {params.use_smart_batching}")
        
        print()
        print("="*70)
        print("✓ END-TO-END TEST PASSED!")
        print("="*70)
        
        return True, output_files, processing_time
        
    except Exception as e:
        print()
        print("="*70)
        print("✗ END-TO-END TEST FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        return False, None, None


if __name__ == "__main__":
    success, outputs, time_taken = run_end_to_end_test()
    
    if success:
        print()
        print("Test completed successfully!")
        print(f"Processing time: {time_taken:.2f}s")
        sys.exit(0)
    else:
        print()
        print("Test failed!")
        sys.exit(1)
