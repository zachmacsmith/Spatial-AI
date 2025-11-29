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

def select_presets(presets: Dict[str, Any]) -> List[Any]:
    """Display menu and get user selection (supports multiple)"""
    # Sort by the preset name (which will now include numbering)
    sorted_keys = sorted(presets.keys(), key=lambda k: presets[k].get_name())
    
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
        choice = input("\nSelect presets (e.g. 1,3 or 1-3, q): ").strip().lower()
        
        if choice == 'q':
            return []
            
        try:
            selected_modules = []
            parts = [p.strip() for p in choice.split(',')]
            
            for part in parts:
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    indices = range(start - 1, end)
                else:
                    indices = [int(part) - 1]
                    
                for idx in indices:
                    if 0 <= idx < len(sorted_keys):
                        selected_modules.append(presets[sorted_keys[idx]])
                    else:
                        print(f"Warning: Invalid selection number {idx + 1}")
            
            if selected_modules:
                return selected_modules
            else:
                print("No valid presets selected.")
                
        except ValueError:
            print("Invalid input format. Use numbers separated by commas.")

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

class DualLogger:
    """Writes to both a file and a stream (stdout/stderr)"""
    def __init__(self, filename, stream):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stream = stream

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write to file

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()

def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("VIDEO PROCESSING RUNNER")
    print("=" * 70)
    
    # 1. Load Presets
    presets = load_presets()
    if not presets:
        print("No presets found in presets/ directory.")
        return
        
    # 2. Select Presets
    selected_modules = select_presets(presets)
    if not selected_modules:
        print("Exiting.")
        return
        
    print(f"\nSelected {len(selected_modules)} presets:")
    for m in selected_modules:
        print(f" - {m.get_name()}")
    
    # 3. Configure Video Generation (Unified)
    print("\n" + "-" * 40)
    print("OPTIMIZATION SETTINGS")
    print("-" * 40)
    gen_video = input("Generate labeled video? (y/N) [default: N]: ").strip().lower()
    generate_labeled_video = (gen_video == 'y')
    if generate_labeled_video:
        print("✓ Video generation ENABLED (slower)")
    else:
        print("✓ Video generation DISABLED (faster)")
        
    # 4. Select Videos (Unified)
    videos = select_videos()
    if not videos:
        print("No videos selected. Exiting.")
        return
        
    # 5. Batch Note (Unified)
    batch_note = input("\nEnter a note for these batches (optional): ").strip()
    
    # 6. Confirm and Run
    print("\n" + "=" * 70)
    print("CONFIRMATION")
    print("=" * 70)
    print(f"Presets to Run: {len(selected_modules)}")
    print(f"Videos per Preset: {len(videos)}")
    print(f"Generate Video: {generate_labeled_video}")
    if batch_note:
        print(f"Note: {batch_note}")
    
    if input("\nStart processing? (y/n): ").strip().lower() != 'y':
        print("Cancelled.")
        return
        
    total_start_time = datetime.now()
    
    # Loop through each selected preset
    for i, module in enumerate(selected_modules, 1):
        # Instantiate parameters for this preset to get batch_id
        batch_params = module.get_batch_params()
        run_batch_id = batch_params.batch_id
        
        # Determine batch folder path
        batch_folder = Path(batch_params.csv_directory) / run_batch_id
        batch_folder.mkdir(parents=True, exist_ok=True)
        
        # Setup Dual Logging
        log_file = batch_folder / "full_analysis.txt"
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        logger_out = DualLogger(log_file, original_stdout)
        logger_err = DualLogger(log_file, original_stderr) # Write stderr to same file
        
        sys.stdout = logger_out
        sys.stderr = logger_err
        
        try:
            print("\n" + "#" * 70)
            print(f"RUNNING PRESET {i}/{len(selected_modules)}: {module.get_name()}")
            print("#" * 70)
            
            # Apply unified settings
            batch_params.generate_labeled_video = generate_labeled_video
            if batch_note:
                batch_params.config_description = batch_note
                
            print(f"Batch ID: {run_batch_id}")
            print(f"Full Analysis Log: {log_file}")
            
            start_time = datetime.now()
            results = []
            
            for video_name in videos:
                print(f"\nProcessing: {video_name}...")
                try:
                    # Create a copy of params for each video to avoid shared state issues
                    current_params = batch_params.copy()
                    
                    # FORCE the batch_id to be the same for all videos in this run
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
                readme_path = batch_folder / "readme.txt"
                with open(readme_path, "w") as f:
                    f.write(f"Batch ID: {run_batch_id}\n")
                    f.write(f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Preset: {batch_params.config_name}\n")
                    f.write(f"Note: {batch_params.config_description}\n")
                    f.write(f"Videos Processed: {len(videos)}\n")
                    f.write("-" * 40 + "\n")
                    f.write("Videos:\n")
                    for res in results:
                        status_icon = "✓" if res["status"] == "success" else "✗"
                        f.write(f"{status_icon} {res['video']} ({res['status']})\n")
                        if res["status"] == "failed":
                            f.write(f"  Error: {res['error']}\n")
                        elif res["status"] == "success" and "output" in res:
                            # Check for partial failures (e.g. video generation failed but data saved)
                            if "video_error" in res["output"]:
                                f.write(f"  ⚠ Video Generation Failed: {res['output']['video_error']}\n")
                        
                print(f"\n✓ Created batch README at: {readme_path}")

                # Also save README to video output folder if it exists or if we are generating videos
                # Even if generate_labeled_video is False, it's good practice to have the readme there if the folder exists
                video_batch_folder = Path(batch_params.video_output_directory) / run_batch_id
                if video_batch_folder.exists() or batch_params.generate_labeled_video:
                    video_batch_folder.mkdir(parents=True, exist_ok=True)
                    video_readme_path = video_batch_folder / "readme.txt"
                    # Read the content we just wrote and write it to the new location
                    with open(readme_path, "r") as src, open(video_readme_path, "w") as dst:
                        dst.write(src.read())
                    print(f"✓ Duplicated README to: {video_readme_path}")

            except Exception as e:
                print(f"\n⚠ Error creating README: {e}")
        
            # Batch Summary
            duration = (datetime.now() - start_time).total_seconds()
            print(f"\nPreset Complete: {module.get_name()}")
            print(f"Time: {duration:.1f}s")
            print(f"Output Folder: outputs/data/{run_batch_id}/")
            
        finally:
            # Restore stdout/stderr and close loggers
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            logger_out.close()
            logger_err.close()

    total_duration = (datetime.now() - total_start_time).total_seconds()
    print("\n" + "=" * 70)
    print("ALL BATCHES COMPLETE")
    print("=" * 70)
    print(f"Total Execution Time: {total_duration:.1f}s")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
