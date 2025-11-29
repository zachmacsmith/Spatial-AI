#!/usr/bin/env python3
"""
Unified Batch Runner
Dynamically loads presets from presets/ directory and runs video processing.
"""
import os
import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_processing.video_processor import process_video
from video_processing import BatchParameters

def load_presets() -> Dict[str, Any]:
    """Dynamically load all presets from presets/ directory"""
    presets = {}
    presets_dir = Path(__file__).parent / "presets"
    
    if not presets_dir.exists():
        print(f"Error: presets directory not found at {presets_dir}")
        return {}
        
    for file_path in presets_dir.glob("*.py"):
        if file_path.name == "__init__.py":
            continue
            
        module_name = file_path.stem
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Verify required functions exist
            if (hasattr(module, 'get_batch_params') and 
                hasattr(module, 'get_name') and 
                hasattr(module, 'get_description')):
                presets[module_name] = module
            else:
                # print(f"Skipping {file_path.name}: missing required functions")
                pass
        except Exception as e:
            print(f"Error loading preset {file_path.name}: {e}")
            
    return presets

def select_preset(presets: Dict[str, Any]) -> Optional[Any]:
    """Display menu and get user selection"""
    sorted_keys = sorted(presets.keys())
    
    print("\n" + "=" * 70)
    print("AVAILABLE PRESETS")
    print("=" * 70)
    
    for i, key in enumerate(sorted_keys, 1):
        module = presets[key]
        name = module.get_name()
        desc = module.get_description()
        print(f"{i}. {name} ({key})")
        print(f"   {desc}")
        print()
        
    print("q. Quit")
    print("=" * 70)
    
    while True:
        choice = input("\nSelect preset (1-{}, q): ".format(len(sorted_keys))).strip().lower()
        
        if choice == 'q':
            return None
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_keys):
                return presets[sorted_keys[idx]]
            else:
                print("Invalid selection number.")
        except ValueError:
            print("Invalid input.")

def select_videos() -> List[str]:
    """Select videos to process"""
    video_dir = Path("videos")
    if not video_dir.exists():
        print("Error: videos/ directory not found")
        return []
        
    videos = sorted([f.stem for f in video_dir.glob("*.mp4")])
    if not videos:
        print("No videos found in videos/")
        return []
        
    print("\n" + "=" * 70)
    print(f"FOUND {len(videos)} VIDEOS")
    print("=" * 70)
    for i, v in enumerate(videos, 1):
        print(f"{i}. {v}")
    
    print("\nOptions:")
    print("  all - Process all videos")
    print("  1,2 - Process specific videos (comma separated)")
    print("  q   - Quit")
    
    choice = input("\nSelect videos: ").strip().lower()
    
    if choice == 'q':
        return []
    elif choice == 'all':
        return videos
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected = []
            for idx in indices:
                if 0 <= idx < len(videos):
                    selected.append(videos[idx])
            return selected
        except:
            print("Invalid selection")
            return []

def main():
    print("\n" + "=" * 70)
    print("VIDEO PROCESSING RUNNER")
    print("=" * 70)
    
    # 1. Load Presets
    presets = load_presets()
    if not presets:
        print("No presets found in presets/ directory.")
        return
        
    # 2. Select Preset
    selected_module = select_preset(presets)
    if not selected_module:
        print("Exiting.")
        return
        
    # 3. Get Parameters
    batch_params = selected_module.get_batch_params()
    print(f"\nSelected: {selected_module.get_name()}")
    
    # 4. Configure Video Generation (Optimization)
    print("\n" + "-" * 40)
    print("OPTIMIZATION SETTINGS")
    print("-" * 40)
    gen_video = input("Generate labeled video? (y/N) [default: N]: ").strip().lower()
    if gen_video == 'y':
        batch_params.generate_labeled_video = True
        print("✓ Video generation ENABLED (slower)")
    else:
        batch_params.generate_labeled_video = False
        print("✓ Video generation DISABLED (faster)")
        
    # 5. Select Videos
    videos = select_videos()
    if not videos:
        print("No videos selected. Exiting.")
        return
        
    # 6. Confirm and Run
    print("\n" + "=" * 70)
    print("CONFIRMATION")
    print("=" * 70)
    print(f"Preset: {batch_params.config_name}")
    print(f"Videos: {len(videos)}")
    print(f"Generate Video: {batch_params.generate_labeled_video}")
    
    # Prompt for batch note
    batch_note = input("\nEnter a note for this batch (optional): ").strip()
    if batch_note:
        batch_params.config_description = batch_note
    
    if input("\nStart processing? (y/n): ").strip().lower() != 'y':
        print("Cancelled.")
        return
        
    start_time = datetime.now()
    results = []
    
    # Capture the initial batch_id to ensure all videos go to the same folder
    # batch_params.batch_id is auto-generated in __post_init__ if not set
    # We want to use this SAME batch_id for all videos in this run
    run_batch_id = batch_params.batch_id
    print(f"\nBatch ID: {run_batch_id}")
    
    for video_name in videos:
        print(f"\nProcessing: {video_name}...")
        try:
            # Create a copy of params for each video to avoid shared state issues
            current_params = batch_params.copy()
            
            # FORCE the batch_id to be the same for all videos in this run
            # The .copy() method normally clears batch_id to generate a new one,
            # so we must explicitly set it back to our run_batch_id
            current_params.batch_id = run_batch_id
            
            # Ensure description is passed through
            current_params.config_description = batch_params.config_description
            
            output_files = process_video(video_name, current_params)
            results.append({"video": video_name, "status": "success", "output": output_files})
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            results.append({"video": video_name, "status": "failed", "error": str(e)})
            
    # Create README.txt in the batch folder
    try:
        # Determine batch folder path (using csv_directory as base, consistent with output_manager)
        batch_folder = Path(batch_params.csv_directory) / run_batch_id
        batch_folder.mkdir(parents=True, exist_ok=True)
        
        readme_path = batch_folder / "readme.txt"
        with open(readme_path, "w") as f:
            f.write(f"Batch ID: {run_batch_id}\n")
            f.write(f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Preset: {batch_params.config_name}\n")
            f.write(f"Note: {batch_params.config_description}\n")
            f.write(f"Videos Processed: {len(videos)}\n")
            f.write("-" * 40 + "\n")
            f.write("Videos:\n")
            for v in videos:
                f.write(f"- {v}\n")
                
        print(f"\n✓ Created batch README at: {readme_path}")
    except Exception as e:
        print(f"\n⚠ Error creating README: {e}")

    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print(f"Total time: {duration:.1f}s")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
    print(f"Output Folder: outputs/data/{run_batch_id}/")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
