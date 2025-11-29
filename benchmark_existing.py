#!/usr/bin/env python3
"""
Benchmark Existing Batches
Run benchmarks on previously processed batches without re-processing videos.
"""
import os
import sys
from pathlib import Path
from datetime import datetime

def find_batch_folders():
    """Find all batch folders in outputs/data/"""
    data_dir = Path("outputs/data")
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return []
    
    batch_folders = [f for f in data_dir.iterdir() if f.is_dir() and f.name.startswith("batch_")]
    return sorted(batch_folders, reverse=True)  # Newest first

def get_videos_from_batch(batch_folder):
    """Extract video names from a batch folder"""
    videos = set()
    for file in batch_folder.glob("*.csv"):
        if not file.name.endswith("_metadata.json") and not file.name.endswith("_relationships.csv"):
            # Extract video name from filename (e.g., "video_01.csv" -> "video_01")
            video_name = file.stem
            videos.add(video_name)
    return sorted(videos)

def main():
    print("=" * 70)
    print("BENCHMARK EXISTING BATCHES")
    print("=" * 70)
    print()
    
    # Find all batch folders
    batch_folders = find_batch_folders()
    
    if not batch_folders:
        print("No batch folders found in outputs/data/")
        return
    
    print(f"Found {len(batch_folders)} batch folders:\n")
    for i, folder in enumerate(batch_folders, 1):
        videos = get_videos_from_batch(folder)
        print(f"{i}. {folder.name}")
        print(f"   Videos: {', '.join(videos) if videos else 'None'}")
        print()
    
    # Let user select batch(es)
    print("Options:")
    print("  - Enter batch number (e.g., '1')")
    print("  - Enter 'all' to benchmark all batches")
    print("  - Enter 'q' to quit")
    print()
    
    choice = input("Select batch: ").strip().lower()
    
    if choice == 'q':
        print("Cancelled.")
        return
    
    batches_to_benchmark = []
    
    if choice == 'all':
        batches_to_benchmark = batch_folders
    else:
        # Support comma-separated list
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            for idx in indices:
                if 0 <= idx < len(batch_folders):
                    batches_to_benchmark.append(batch_folders[idx])
                else:
                    print(f"⚠ Invalid index ignored: {idx + 1}")
        except ValueError:
            print(f"Invalid input: {choice}")
            return
            
    if not batches_to_benchmark:
        print("No valid batches selected.")
        return
    
    # Ask for performance analysis options
    print("\nPerformance Analysis:")
    print("  0 - None (Skip performance charts)")
    print("  1 - Averages Only (Summary tables/charts)")
    print("  2 - Full Breakdown (Per-video charts + Averages)")
    print("  3 - Over/Under Analysis (Includes Full Breakdown)")
    perf_choice = input("Select option (0/1/2/3): ").strip()
    
    performance_mode = "none"
    generate_over_under = False
    if perf_choice == '1':
        performance_mode = "averages"
    elif perf_choice == '2':
        performance_mode = "full"
    elif perf_choice == '3':
        performance_mode = "full"
        generate_over_under = True
        
    # Import benchmark function (only after user has made selection)
    try:
        from post_processing.accuracy_benchmark import run_benchmark, generate_comparison_charts, generate_performance_comparison_charts
    except ImportError as e:
        print(f"❌ Error importing benchmark: {e}")
        print("Make sure seaborn is installed: pip install seaborn")
        return
    
    # Benchmark each selected batch
    all_benchmark_results = []
    
    for batch_folder in batches_to_benchmark:
        batch_id = batch_folder.name
        videos = get_videos_from_batch(batch_folder)
        
        if not videos:
            print(f"\n⚠ Skipping {batch_id} - no videos found")
            continue
        
        print("\n" + "=" * 70)
        print(f"Benchmarking: {batch_id}")
        print(f"Videos: {', '.join(videos)}")
        print("=" * 70)
        
        # Get batch metadata
        batch_name = batch_id
        model_version = "unknown"
        notes = ""
        
        # Try to load from batch tracking
        try:
            import json
            tracking_path = Path("outputs/batch_tracking") / f"{batch_id}.json"
            if tracking_path.exists():
                with open(tracking_path, 'r') as f:
                    config = json.load(f)
                    if 'llm_model' in config:
                        model_version = config['llm_model']
                        print(f"✓ Detected model version: {model_version}")
                    if 'config_description' in config:
                        notes = config['config_description']
            else:
                print(f"⚠ No batch config found at {tracking_path}")
        except Exception as e:
            print(f"⚠ Error loading batch config: {e}")

        print(f"Batch Name: {batch_name}")
        print(f"Model Version: {model_version}")
        if notes:
            print(f"Notes: {notes}")

        
        # Run benchmark
        try:
            result = run_benchmark(
                videos_to_process=videos,
                batch_name=batch_name,
                model_version=model_version,
                notes=notes,
                model_data_dir=str(batch_folder) + "/",
                batch_id=batch_id,
                performance_mode=performance_mode,
                generate_over_under=generate_over_under
            )
            if result:
                all_benchmark_results.append(result)
                print(f"\n✓ Benchmark complete for {batch_id}")
        except Exception as e:
            print(f"\n❌ Error benchmarking {batch_id}: {e}")
            import traceback
            traceback.print_exc()
            
    # Generate Comparison Charts if multiple batches processed
    if len(all_benchmark_results) > 1:
        print("\n" + "=" * 70)
        print("GENERATING COMPARISON CHARTS")
        print("=" * 70)
        
        comparison_dir = "benchmark_results/comparisons/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            generate_comparison_charts(all_benchmark_results, comparison_dir)
            print(f"\n✓ Comparison charts saved to: {comparison_dir}")
            
            if performance_mode != "none":
                generate_performance_comparison_charts(all_benchmark_results, comparison_dir)
                print(f"✓ Performance comparison charts saved to: {comparison_dir}")
                
        except Exception as e:
            print(f"❌ Error generating comparison charts: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
