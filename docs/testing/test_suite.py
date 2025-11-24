"""
Comprehensive Test Suite for Modular Architecture

Tests all components with Gemini API:
1. Imports and configuration
2. Video loading and frame caching
3. Keyframe detection
4. LLM service (Gemini)
5. Action classification
6. Output generation
7. Batch tracking
"""

import sys
import os

# Test results tracker
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_header(name):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

def test_pass(name):
    print(f"✓ PASS: {name}")
    test_results['passed'].append(name)

def test_fail(name, error):
    print(f"✗ FAIL: {name}")
    print(f"  Error: {error}")
    test_results['failed'].append((name, error))

def test_warn(name, message):
    print(f"⚠ WARNING: {name}")
    print(f"  {message}")
    test_results['warnings'].append((name, message))

# ==========================================
# TEST 1: Imports
# ==========================================
test_header("Module Imports")

try:
    from video_processing import (
        BatchParameters,
        PRESET_BASIC,
        PRESET_OBJECTS,
        PRESET_RELATIONSHIPS,
        PRESET_FULL
    )
    test_pass("Import presets")
except Exception as e:
    test_fail("Import presets", e)
    sys.exit(1)

try:
    from video_processing.video_processor import process_video
    test_pass("Import video_processor")
except Exception as e:
    test_fail("Import video_processor", e)
    sys.exit(1)

try:
    from video_processing.ai.llm_service import get_llm_service, GeminiService
    from video_processing.ai.cv_service import get_cv_service
    from video_processing.ai.prompt_builder import PromptBuilder
    test_pass("Import AI services")
except Exception as e:
    test_fail("Import AI services", e)
    sys.exit(1)

try:
    from video_processing.analysis.action_classifier import classify_action
    from video_processing.analysis.tool_detector import detect_tool
    test_pass("Import analysis modules")
except Exception as e:
    test_fail("Import analysis modules", e)
    sys.exit(1)

try:
    from video_processing.batch_comparison import BatchRegistry
    test_pass("Import batch tracking")
except Exception as e:
    test_fail("Import batch tracking", e)
    sys.exit(1)

# ==========================================
# TEST 2: Configuration with Gemini
# ==========================================
test_header("Configuration with Gemini")

try:
    from video_processing import LLMProvider
    
    # Create Gemini configuration
    params = PRESET_BASIC.copy()
    params.llm_provider = LLMProvider.GEMINI
    params.llm_model = "gemini-2.5-flash"  # Use stable Gemini 2.5 Flash
    
    test_pass("Create Gemini configuration")
    print(f"  Batch ID: {params.batch_id}")
    print(f"  LLM: {params.llm_provider.value}")
    print(f"  Model: {params.llm_model}")
except Exception as e:
    test_fail("Create Gemini configuration", e)
    sys.exit(1)

# ==========================================
# TEST 3: API Key Loading
# ==========================================
test_header("API Key Loading")

try:
    llm_service = get_llm_service(params)
    test_pass("Load Gemini API key from config")
    print(f"  Provider: {llm_service.get_provider_name()}")
except Exception as e:
    test_fail("Load Gemini API key", e)
    sys.exit(1)

# ==========================================
# TEST 4: Video Properties
# ==========================================
test_header("Video Properties")

try:
    from video_processing.utils.video_utils import get_video_properties
    
    video_path = "videos/video_01.mp4"
    if not os.path.exists(video_path):
        test_warn("Video file", f"{video_path} not found, skipping video tests")
    else:
        props = get_video_properties(video_path)
        test_pass("Extract video properties")
        print(f"  Resolution: {props.width}x{props.height}")
        print(f"  FPS: {props.fps:.2f}")
        print(f"  Frames: {props.frame_count}")
        print(f"  Duration: {props.duration:.2f}s")
except Exception as e:
    test_fail("Extract video properties", e)

# ==========================================
# TEST 5: Keyframe Loading
# ==========================================
test_header("Keyframe Loading")

try:
    from video_processing.utils.video_utils import load_keyframe_numbers
    
    keyframes_folder = "keyframes/keyframesvideo_01"
    if not os.path.exists(keyframes_folder):
        test_warn("Keyframes", f"{keyframes_folder} not found")
    else:
        keyframes = load_keyframe_numbers(keyframes_folder)
        test_pass("Load keyframe numbers")
        print(f"  Keyframes found: {len(keyframes)}")
        print(f"  Keyframe numbers: {keyframes}")
except Exception as e:
    test_fail("Load keyframe numbers", e)

# ==========================================
# TEST 6: Frame Caching (Small Test)
# ==========================================
test_header("Frame Caching")

try:
    from video_processing.utils.video_utils import load_all_frames
    
    if os.path.exists(video_path):
        # Only load first 50 frames for testing
        import cv2
        cap = cv2.VideoCapture(video_path)
        test_cache = {}
        for i in range(1, 51):
            ret, frame = cap.read()
            if not ret:
                break
            test_cache[i] = frame
        cap.release()
        
        test_pass("Frame caching (50 frames)")
        print(f"  Cached frames: {len(test_cache)}")
except Exception as e:
    test_fail("Frame caching", e)

# ==========================================
# TEST 7: Prompt Building
# ==========================================
test_header("Prompt Building")

try:
    prompt_builder = PromptBuilder(params)
    
    # Test action classification prompt
    prompt = prompt_builder.build_action_classification_prompt(
        motion_score=0.25,
        detected_objects=[("hammer", 0.95), ("nail", 0.87)]
    )
    test_pass("Build action classification prompt")
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Preview: {prompt[:100]}...")
except Exception as e:
    test_fail("Build prompt", e)

# ==========================================
# TEST 8: Gemini API Call (Single Frame)
# ==========================================
test_header("Gemini API Call")

try:
    if os.path.exists(video_path) and len(test_cache) > 0:
        # Test with single frame
        test_frame = test_cache[1]
        
        response = llm_service.send_multiframe_prompt(
            frames=[test_frame],
            prompt_text="Describe what you see in this image in one sentence.",
            max_tokens=100
        )
        
        test_pass("Gemini API call")
        print(f"  Response: {response[:100]}...")
    else:
        test_warn("Gemini API call", "No frames available for testing")
except Exception as e:
    test_fail("Gemini API call", e)

# ==========================================
# TEST 9: Action Classification
# ==========================================
test_header("Action Classification")

try:
    if os.path.exists(video_path) and len(test_cache) > 0:
        action = classify_action(
            frames=[test_cache[1]],
            batch_params=params,
            llm_service=llm_service,
            prompt_builder=prompt_builder,
            motion_score=0.15
        )
        
        test_pass("Action classification")
        print(f"  Classified action: {action}")
        print(f"  Valid action: {action in params.allowed_actions}")
    else:
        test_warn("Action classification", "No frames available")
except Exception as e:
    test_fail("Action classification", e)

# ==========================================
# TEST 10: Batch Tracking
# ==========================================
test_header("Batch Tracking")

try:
    # Save batch config
    config_path = params.save_batch_config()
    test_pass("Save batch configuration")
    print(f"  Config saved to: {config_path}")
    
    # Load it back
    loaded_params = BatchParameters.from_batch_id(params.batch_id)
    test_pass("Load batch configuration")
    print(f"  Loaded batch ID: {loaded_params.batch_id}")
    print(f"  Config matches: {loaded_params.batch_id == params.batch_id}")
except Exception as e:
    test_fail("Batch tracking", e)

# ==========================================
# TEST 11: Batch Registry
# ==========================================
test_header("Batch Registry")

try:
    registry = BatchRegistry()
    all_batches = registry.get_all_batch_ids()
    test_pass("Batch registry")
    print(f"  Total tracked batches: {len(all_batches)}")
    if all_batches:
        print(f"  Latest batch: {all_batches[-1]}")
except Exception as e:
    test_fail("Batch registry", e)

# ==========================================
# TEST 12: End-to-End Processing (Small Video)
# ==========================================
test_header("End-to-End Processing")

try:
    if os.path.exists(video_path) and os.path.exists(keyframes_folder):
        print("  Processing video_01 with Gemini...")
        print("  (This may take a few minutes)")
        
        # Use BASIC preset with Gemini
        test_params = PRESET_BASIC.copy()
        test_params.llm_provider = LLMProvider.GEMINI
        test_params.llm_model = "gemini-2.5-flash"  # Use stable Gemini 2.5 Flash
        test_params.generate_labeled_video = False  # Skip video generation for speed
        
        outputs = process_video("video_01", test_params)
        
        test_pass("End-to-end processing")
        print(f"  Actions CSV: {outputs['actions_csv']}")
        print(f"  Metadata: {outputs['metadata']}")
        
        # Verify outputs exist
        if os.path.exists(outputs['actions_csv']):
            test_pass("Actions CSV created")
            # Read first few lines
            with open(outputs['actions_csv'], 'r') as f:
                lines = f.readlines()[:5]
                print(f"  CSV preview ({len(lines)} lines):")
                for line in lines:
                    print(f"    {line.strip()}")
        else:
            test_fail("Actions CSV created", "File not found")
            
    else:
        test_warn("End-to-end processing", "Video or keyframes not found, skipping")
except Exception as e:
    test_fail("End-to-end processing", e)
    import traceback
    traceback.print_exc()

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print(f"✓ Passed: {len(test_results['passed'])}")
print(f"✗ Failed: {len(test_results['failed'])}")
print(f"⚠ Warnings: {len(test_results['warnings'])}")

if test_results['passed']:
    print("\nPassed tests:")
    for test in test_results['passed']:
        print(f"  ✓ {test}")

if test_results['failed']:
    print("\nFailed tests:")
    for test, error in test_results['failed']:
        print(f"  ✗ {test}: {error}")

if test_results['warnings']:
    print("\nWarnings:")
    for test, msg in test_results['warnings']:
        print(f"  ⚠ {test}: {msg}")

print("\n" + "="*60)
if len(test_results['failed']) == 0:
    print("✓ ALL TESTS PASSED!")
else:
    print(f"✗ {len(test_results['failed'])} TESTS FAILED")
print("="*60)
