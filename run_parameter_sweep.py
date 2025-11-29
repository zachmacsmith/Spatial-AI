#!/usr/bin/env python3
"""
Parameter Sweep Analysis Tool
Runs multiple batches with varying parameters and generates performance graphs.
"""
import os
import sys
import time
import importlib.util
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import fields

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_processing import BatchParameters
from video_processing.video_processor import process_video
from run_batch import load_presets, select_presets, select_videos, DualLogger
from post_processing.accuracy_benchmark import run_benchmark

def get_tunable_parameters() -> Dict[str, type]:
    """
    Returns a dictionary of tunable parameters from BatchParameters
    and their types. Excludes internal fields.
    """
    excluded = {
        'batch_id', 'video_output_directory', 'csv_directory', 
        'config_name', 'config_description', 'allowed_actions', 
        'allowed_tools', 'allowed_objects'
    }
    
    params = {}
    for field in fields(BatchParameters):
        if field.name not in excluded:
            params[field.name] = field.type
            
    return params

def select_parameter(params: Dict[str, type]) -> str:
    """Let user select a parameter to tune"""
    sorted_params = sorted(params.keys())
    
    print("\n" + "=" * 50)
    print("AVAILABLE PARAMETERS")
    print("=" * 50)
    
    for i, param in enumerate(sorted_params, 1):
        print(f"{i}. {param} ({params[param].__name__})")
        
    while True:
        try:
            choice = input("\nSelect parameter number: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_params):
                return sorted_params[idx]
            print("Invalid selection.")
        except ValueError:
            print("Please enter a number.")

def parse_values(param_name: str, param_type: type) -> List[Any]:
    """Parse comma-separated values based on type"""
    print(f"\nEnter values for '{param_name}' (comma-separated).")
    print(f"Type: {param_type.__name__}")
    
    while True:
        try:
            raw_input = input("Values: ").strip()
            parts = [p.strip() for p in raw_input.split(',')]
            
            values = []
            for p in parts:
                if param_type == bool:
                    if p.lower() in ('true', 't', 'yes', 'y', '1'):
                        values.append(True)
                    elif p.lower() in ('false', 'f', 'no', 'n', '0'):
                        values.append(False)
                    else:
                        raise ValueError(f"Invalid boolean: {p}")
                elif param_type == int:
                    values.append(int(p))
                elif param_type == float:
                    values.append(float(p))
                else:
                    values.append(p)
            
            if not values:
                print("No values entered.")
                continue
                
            return values
        except ValueError as e:
            print(f"Error parsing values: {e}")

def select_metrics() -> List[str]:
    """Let user select metrics to plot"""
    metrics = {
        'state_accuracy': 'State Accuracy (%)',
        'object_accuracy': 'Object Accuracy (%)',
        'total_time': 'Total Time (s)',
        'speed_ratio': 'Speed Ratio (x)',
        'api_calls': 'Total API Calls'
    }
    keys = list(metrics.keys())
    
    print("\n" + "=" * 50)
    print("AVAILABLE METRICS")
    print("=" * 50)
    
    for i, key in enumerate(keys, 1):
        print(f"{i}. {metrics[key]}")
        
    print("all. Select All")
    
    while True:
        choice = input("\nSelect metrics (comma-separated, e.g. 1,3): ").strip().lower()
        if choice == 'all':
            return keys
            
        try:
            selected = []
            parts = [p.strip() for p in choice.split(',')]
            for p in parts:
                idx = int(p) - 1
                if 0 <= idx < len(keys):
                    selected.append(keys[idx])
            
            if selected:
                return selected
            print("No valid metrics selected.")
        except ValueError:
            print("Invalid input.")

def run_sweep():
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP ANALYSIS")
    print("=" * 70)
    
    # 1. Select Preset
    presets = load_presets()
    if not presets:
        print("No presets found.")
        return
        
    selected_modules = select_presets(presets)
    if not selected_modules or len(selected_modules) > 1:
        print("Please select exactly ONE preset for the sweep.")
        return
    
    preset_module = selected_modules[0]
    base_params = preset_module.get_batch_params()
    
    print(f"\nSelected Preset: {preset_module.get_name()}")
    
    # 2. Select Parameter
    tunable_params = get_tunable_parameters()
    param_name = select_parameter(tunable_params)
    param_type = tunable_params[param_name]
    
    # 3. Input Values
    values = parse_values(param_name, param_type)
    print(f"Values to sweep: {values}")
    
    # 4. Select Videos
    videos = select_videos()
    if not videos:
        return
        
    # 5. Select Metrics
    metrics = select_metrics()
    print(f"Metrics to plot: {metrics}")
    
    # 6. Confirm
    print("\n" + "=" * 70)
    print("SWEEP CONFIGURATION")
    print("=" * 70)
    print(f"Preset: {preset_module.get_name()}")
    print(f"Parameter: {param_name}")
    print(f"Values: {values}")
    print(f"Videos: {len(videos)}")
    print(f"Metrics: {metrics}")
    
    if input("\nStart sweep? (y/n): ").strip().lower() != 'y':
        print("Cancelled.")
        return
        
    # 7. Execution Loop
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("sweep_results") / f"{experiment_id}_{param_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    sweep_data = {m: [] for m in metrics}
    successful_values = []
    
    print(f"\nStarting sweep... Results will be saved to {results_dir}")
    
    for val in values:
        print(f"\n" + "-" * 50)
        print(f"Testing {param_name} = {val}")
        print("-" * 50)
        
        # Create params for this run
        current_params = base_params.copy()
        setattr(current_params, param_name, val)
        
        # Unique batch ID
        batch_id = f"sweep_{experiment_id}_{param_name}_{val}"
        current_params.batch_id = batch_id
        current_params.config_description = f"Sweep: {param_name}={val}"
        
        # Setup logging
        batch_folder = Path(current_params.csv_directory) / batch_id
        batch_folder.mkdir(parents=True, exist_ok=True)
        
        log_file = batch_folder / "full_analysis.txt"
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        logger_out = DualLogger(log_file, original_stdout)
        logger_err = DualLogger(log_file, original_stderr)
        
        sys.stdout = logger_out
        sys.stderr = logger_err
        
        try:
            # Run processing
            for video_name in videos:
                print(f"Processing {video_name}...")
                process_video(video_name, current_params)
            
            # Run benchmark
            print("\nRunning benchmark...")
            # We need to capture the return value of run_benchmark or read the CSV
            # run_benchmark returns nothing, but saves files.
            # We will read the summary csv.
            
            run_benchmark(batch_id, videos)
            
            # Read benchmark results
            stats_path = Path("benchmark_results") / batch_id / "benchmark_statistics.csv"
            if stats_path.exists():
                # Simple parsing of the stats file
                stats = {}
                with open(stats_path, 'r') as f:
                    for line in f:
                        if ',' in line:
                            k, v = line.strip().split(',')
                            try:
                                stats[k] = float(v.replace('%', ''))
                            except ValueError:
                                pass
                
                # Map stats to our metrics
                # Note: benchmark_statistics.csv keys might differ from our internal keys
                # Let's check what run_benchmark saves.
                # It saves: Average State Accuracy, Average Object Accuracy, Total Time, Speed Ratio, Total API Calls
                
                # Mapping
                val_map = {
                    'state_accuracy': stats.get('Average State Accuracy', 0),
                    'object_accuracy': stats.get('Average Object Accuracy', 0),
                    'total_time': stats.get('Total Time', 0),
                    'speed_ratio': stats.get('Average Speed Ratio', 0),
                    'api_calls': stats.get('Total API Calls', 0)
                }
                
                for m in metrics:
                    sweep_data[m].append(val_map[m])
                
                successful_values.append(val)
                print(f"✓ Results for {val}: {val_map}")
            else:
                print(f"⚠ Benchmark stats not found for {val}")
                
        except Exception as e:
            print(f"⚠ Error in sweep run for {val}: {e}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            logger_out.close()
            logger_err.close()
            
    # 8. Visualization
    if not successful_values:
        print("\nNo successful runs to plot.")
        return
        
    print("\nGenerating plots...")
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(successful_values, sweep_data[metric], marker='o', linestyle='-', linewidth=2)
        
        plt.title(f"Effect of {param_name} on {metric.replace('_', ' ').title()}")
        plt.xlabel(f"{param_name}")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plot_path = results_dir / f"{metric}_vs_{param_name}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
        
    print(f"\nSweep Complete! Results in {results_dir}")

if __name__ == "__main__":
    run_sweep()
