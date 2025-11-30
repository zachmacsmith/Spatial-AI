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
            video_name = file.stem
            videos.add(video_name)
    return sorted(videos)

def get_batch_note(batch_folder):
    """Get the description/note for a batch."""
    import json
    try:
        tracking_path = Path("outputs/batch_tracking") / f"{batch_folder.name}.json"
        if tracking_path.exists():
            with open(tracking_path, 'r') as f:
                config = json.load(f)
                if 'config_description' in config: return config['config_description']
    except: pass
    try:
        readme_path = batch_folder / "readme.txt"
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                for line in f:
                    if line.startswith("Note:"): return line.replace("Note:", "").strip()
    except: pass
    return ""

def main():
    print("=" * 70); print("BENCHMARK EXISTING BATCHES"); print("=" * 70); print()
    
    batch_folders = find_batch_folders()
    if not batch_folders: print("No batch folders found."); return
    
    print(f"Found {len(batch_folders)} batch folders:\n")
    for i in range(len(batch_folders) - 1, -1, -1):
        folder = batch_folders[i]
        note = get_batch_note(folder)
        note_str = f" - {note}" if note else ""
        print(f"{i+1}. {folder.name}{note_str}")
        
    print("\nOptions: Number, 'all', or 'q'")
    choice = input("Select batch: ").strip().lower()
    if choice == 'q': return
    
    batches_to_benchmark = []
    if choice == 'all': batches_to_benchmark = batch_folders
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            for idx in indices:
                if 0 <= idx < len(batch_folders): batches_to_benchmark.append(batch_folders[idx])
        except: print("Invalid input"); return

    if not batches_to_benchmark: return

    # --- NEW: Metric Selection ---
    print("\nSelect Accuracy Measure:")
    print("  1 - Traditional (Standard Accuracy)")
    print("  2 - F1-Weighted (Macro F1 Score - Recommended for Imbalanced Data)")
    metric_choice = input("Select option (1/2): ").strip()
    
    metric_mode = "traditional"
    if metric_choice == '2':
        metric_mode = "f1_weighted"
        print(">> Selected: F1-Weighted Mode")
    else:
        print(">> Selected: Traditional Accuracy Mode")
    
    print("\nPerformance Analysis:")
    print("  0 - None"); print("  1 - Averages Only"); print("  2 - Full Breakdown"); print("  3 - Over/Under Analysis")
    perf_choice = input("Select option: ").strip()
    
    performance_mode = "none"
    generate_over_under = False
    if perf_choice == '1': performance_mode = "averages"
    elif perf_choice in ['2', '3']: 
        performance_mode = "full"
        if perf_choice == '3': generate_over_under = True
        
    try:
        from post_processing.accuracy_benchmark import run_benchmark, generate_comparison_charts, generate_performance_comparison_charts
    except ImportError as e:
        print(f"❌ Error importing benchmark: {e}"); return
    
    all_benchmark_results = []
    
    for batch_folder in batches_to_benchmark:
        batch_id = batch_folder.name
        videos = get_videos_from_batch(batch_folder)
        if not videos: continue
        
        # Get metadata
        model_version = "unknown"
        notes = ""
        try:
            import json
            tracking_path = Path("outputs/batch_tracking") / f"{batch_id}.json"
            if tracking_path.exists():
                with open(tracking_path, 'r') as f:
                    config = json.load(f)
                    if 'llm_model' in config: model_version = config['llm_model']
                    if 'config_description' in config: notes = config['config_description']
        except: pass

        print("\n" + "=" * 70)
        print(f"Benchmarking: {batch_id} ({metric_mode})")
        print("=" * 70)
        
        try:
            result = run_benchmark(
                videos_to_process=videos,
                batch_name=batch_id,
                model_version=model_version,
                notes=notes,
                model_data_dir=str(batch_folder) + "/",
                batch_id=batch_id,
                performance_mode=performance_mode,
                generate_over_under=generate_over_under,
                metric_mode=metric_mode # <--- Pass the mode
            )
            if result: all_benchmark_results.append(result)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback; traceback.print_exc()
            
    if len(all_benchmark_results) > 1:
        print("\n" + "=" * 70); print("GENERATING COMPARISON CHARTS"); print("=" * 70)
        # Separate output folder for F1 comparisons
        base = "benchmark_results_f1" if metric_mode == "f1_weighted" else "benchmark_results"
        comp_dir = f"{base}/comparisons/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            generate_comparison_charts(all_benchmark_results, comp_dir, metric_label=all_benchmark_results[0]['metric_label'])
            print(f"\n✓ Comparison charts saved to: {comp_dir}")
            if performance_mode != "none":
                generate_performance_comparison_charts(all_benchmark_results, comp_dir)
        except Exception as e: print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
