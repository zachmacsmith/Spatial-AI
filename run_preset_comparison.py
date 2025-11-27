#!/usr/bin/env python3
"""
Preset Comparison Runner
Runs multiple preset configurations for comparison, including:
- Different LLM providers (Gemini, Anthropic)
- Different CV model weights
- Different preset configurations
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Import video processing components
from video_processing import BatchParameters
from video_processing.video_processor import process_video
from video_processing.batch_parameters import (
    LLMProvider,
    ToolDetectionMethod,
    ActionClassificationMethod
)

# ============================================================================
# PRESET DEFINITIONS
# ============================================================================

def create_preset_gemini_relationships():
    """PRESET 1: Relationships with Gemini (baseline)"""
    return BatchParameters(
        config_name="relationships_gemini",
        config_description="Relationships tracking with Gemini 2.0 Flash Lite",
        llm_provider=LLMProvider.GEMINI,
        llm_model="gemini-2.0-flash-lite",
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_productivity_analysis=False,
        tool_detection_method=ToolDetectionMethod.LLM_DIRECT,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME,
        cv_model_path="weights.pt",  # Default weights
        api_requests_per_minute=30,
        pricing_tier="free"
    )

def create_preset_anthropic_relationships():
    """PRESET 2: Relationships with Anthropic Claude"""
    return BatchParameters(
        config_name="relationships_anthropic",
        config_description="Relationships tracking with Claude 4.5 Sonnet",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-sonnet-4-5-20250929",
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_productivity_analysis=False,
        tool_detection_method=ToolDetectionMethod.LLM_DIRECT,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME,
        cv_model_path="weights.pt",
        api_requests_per_minute=50,  # Anthropic has higher limits
        pricing_tier="pay_as_you_go"
    )

def create_preset_gemini_alt_weights():
    """PRESET 3: Gemini with alternative CV weights"""
    return BatchParameters(
        config_name="relationships_gemini_alt_weights",
        config_description="Relationships with Gemini and alternative CV model",
        llm_provider=LLMProvider.GEMINI,
        llm_model="gemini-2.0-flash-lite",
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_productivity_analysis=False,
        tool_detection_method=ToolDetectionMethod.LLM_DIRECT,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME,
        cv_model_path="weights_v2.pt",  # Alternative weights file
        api_requests_per_minute=30,
        pricing_tier="free"
    )

def create_preset_anthropic_alt_weights():
    """PRESET 4: Anthropic with alternative CV weights"""
    return BatchParameters(
        config_name="relationships_anthropic_alt_weights",
        config_description="Relationships with Claude 4.5 Sonnet and alternative CV model",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-sonnet-4-5-20250929",
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_productivity_analysis=False,
        tool_detection_method=ToolDetectionMethod.LLM_DIRECT,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME,
        cv_model_path="weights_v2.pt",
        api_requests_per_minute=50,
        pricing_tier="pay_as_you_go"
    )

def create_preset_gemini_flash():
    """PRESET 5: Gemini 2.5 Flash (faster model)"""
    return BatchParameters(
        config_name="relationships_gemini_flash",
        config_description="Relationships with Gemini 2.5 Flash",
        llm_provider=LLMProvider.GEMINI,
        llm_model="gemini-2.5-flash",
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_productivity_analysis=False,
        tool_detection_method=ToolDetectionMethod.LLM_DIRECT,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME,
        cv_model_path="weights.pt",
        api_requests_per_minute=10,  # Lower limit for 2.5-flash
        pricing_tier="free"
    )

def create_preset_anthropic_haiku():
    """PRESET 6: Anthropic Claude Haiku (faster, cheaper)"""
    return BatchParameters(
        config_name="relationships_anthropic_haiku",
        config_description="Relationships with Claude 4.5 Haiku",
        llm_provider=LLMProvider.CLAUDE,
        llm_model="claude-haiku-4-5-20251001",
        enable_object_detection=True,
        enable_relationship_tracking=True,
        enable_productivity_analysis=False,
        tool_detection_method=ToolDetectionMethod.LLM_DIRECT,
        action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME,
        cv_model_path="weights.pt",
        api_requests_per_minute=50,
        pricing_tier="pay_as_you_go"
    )

# ============================================================================
# PRESET REGISTRY
# ============================================================================

PRESET_REGISTRY = {
    "1_gemini_relationships": create_preset_gemini_relationships,
    "2_anthropic_relationships": create_preset_anthropic_relationships,
    "3_gemini_alt_weights": create_preset_gemini_alt_weights,
    "4_anthropic_alt_weights": create_preset_anthropic_alt_weights,
    "5_gemini_flash": create_preset_gemini_flash,
    "6_anthropic_haiku": create_preset_anthropic_haiku,
}

# ============================================================================
# MAIN RUNNER
# ============================================================================

def check_prerequisites():
    """Check if required files exist"""
    issues = []
    
    # Check for API keys
    try:
        from config import GEMINI_API_KEY, ANTHROPIC_API_KEY
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your-api-key-here":
            issues.append("⚠ Gemini API key not configured")
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your-api-key-here":
            issues.append("⚠ Anthropic API key not configured")
    except ImportError:
        issues.append("✗ config.py not found or missing API keys")
    
    # Check for CV weights
    if not Path("weights.pt").exists():
        issues.append("⚠ weights.pt not found (default CV model)")
    if not Path("weights_v2.pt").exists():
        issues.append("⚠ weights_v2.pt not found (alternative CV model)")
    
    # Check for videos
    if not Path("videos").exists() or not list(Path("videos").glob("*.mp4")):
        issues.append("✗ No videos found in videos/ directory")
    
    return issues

def select_presets() -> List[str]:
    """Let user select which presets to run"""
    print("\n" + "=" * 70)
    print("AVAILABLE PRESETS")
    print("=" * 70)
    
    for key, func in PRESET_REGISTRY.items():
        preset = func()
        print(f"\n{key}:")
        print(f"  Name: {preset.config_name}")
        print(f"  Description: {preset.config_description}")
        print(f"  LLM: {preset.llm_provider.value}/{preset.llm_model}")
        print(f"  CV Weights: {preset.cv_model_path}")
        print(f"  Rate Limit: {preset.api_requests_per_minute} RPM")
    
    print("\n" + "=" * 70)
    print("SELECTION OPTIONS")
    print("=" * 70)
    print("  all    - Run all presets")
    print("  gemini - Run all Gemini presets (1, 3, 5)")
    print("  anthropic - Run all Anthropic presets (2, 4, 6)")
    print("  1,2,3  - Run specific presets by number")
    print("  q      - Quit")
    print()
    
    selection = input("Select presets: ").strip().lower()
    
    if selection == 'q':
        return []
    elif selection == 'all':
        return list(PRESET_REGISTRY.keys())
    elif selection == 'gemini':
        return [k for k in PRESET_REGISTRY.keys() if 'gemini' in k]
    elif selection == 'anthropic':
        return [k for k in PRESET_REGISTRY.keys() if 'anthropic' in k]
    else:
        # Parse comma-separated numbers
        try:
            numbers = [int(n.strip()) for n in selection.split(',')]
            return [f"{n}_{list(PRESET_REGISTRY.keys())[n-1].split('_', 1)[1]}" 
                    for n in numbers if 1 <= n <= len(PRESET_REGISTRY)]
        except:
            print(f"Invalid selection: {selection}")
            return []

def select_videos() -> List[str]:
    """Automatically select all videos in videos/ directory"""
    video_dir = Path("videos")
    if not video_dir.exists():
        print("✗ videos/ directory not found")
        return []
    
    videos = sorted([f.stem for f in video_dir.glob("*.mp4")])
    
    if not videos:
        print("✗ No videos found in videos/")
        return []
    
    print("\n" + "=" * 70)
    print(f"AUTO-SELECTED {len(videos)} VIDEOS")
    print("=" * 70)
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video}")
    print()
    
    return videos

def run_preset_comparison():
    """Main function to run preset comparison"""
    print("\n" + "=" * 70)
    print("PRESET COMPARISON RUNNER")
    print("=" * 70)
    print()
    
    # Check prerequisites
    issues = check_prerequisites()
    if issues:
        print("Prerequisites check:")
        for issue in issues:
            print(f"  {issue}")
        print()
        if any("✗" in issue for issue in issues):
            print("Critical issues found. Cannot proceed.")
            return
    
    # Select presets
    selected_presets = select_presets()
    if not selected_presets:
        print("No presets selected. Exiting.")
        return
    
    # Select videos
    selected_videos = select_videos()
    if not selected_videos:
        print("No videos selected. Exiting.")
        return
    
    # Confirm
    print("\n" + "=" * 70)
    print("CONFIRMATION")
    print("=" * 70)
    print(f"Presets: {len(selected_presets)}")
    for preset_key in selected_presets:
        print(f"  - {preset_key}")
    print(f"\nVideos: {len(selected_videos)}")
    for video in selected_videos:
        print(f"  - {video}")
    print(f"\nTotal runs: {len(selected_presets) * len(selected_videos)}")
    print()
    
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Run comparisons
    results = []
    start_time = datetime.now()
    
    for preset_key in selected_presets:
        preset_func = PRESET_REGISTRY[preset_key]
        preset = preset_func()
        
        for video_name in selected_videos:
            print("\n" + "=" * 70)
            print(f"Processing: {preset.config_name} / {video_name}")
            print("=" * 70)
            
            try:
                result = process_video(
                    video_name=video_name,
                    batch_params=preset
                )
                results.append({
                    'preset': preset.config_name,
                    'video': video_name,
                    'batch_id': preset.batch_id,
                    'status': 'success',
                    'error': None
                })
                print(f"✓ Completed: {preset.batch_id}")
            except Exception as e:
                print(f"✗ Error: {e}")
                results.append({
                    'preset': preset.config_name,
                    'video': video_name,
                    'batch_id': None,
                    'status': 'failed',
                    'error': str(e)
                })
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Total time: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
    print()
    
    # Save results summary
    summary_file = f"preset_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w') as f:
        f.write("PRESET COMPARISON RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration:.1f}s\n\n")
        
        for result in results:
            f.write(f"\nPreset: {result['preset']}\n")
            f.write(f"Video: {result['video']}\n")
            f.write(f"Status: {result['status']}\n")
            if result['batch_id']:
                f.write(f"Batch ID: {result['batch_id']}\n")
            if result['error']:
                f.write(f"Error: {result['error']}\n")
    
    print(f"Results saved to: {summary_file}")
    print("\nNext steps:")
    print("  1. Review outputs in outputs/data/ and outputs/vid_objs/")
    print("  2. Run benchmark_existing.py to compare accuracy")
    print("  3. Compare processing times and quality")

if __name__ == "__main__":
    try:
        run_preset_comparison()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
