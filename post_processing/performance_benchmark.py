"""
Performance Benchmark - Timing and Performance Analysis

Analyzes timing data from batch_process.py to:
- Calculate performance metrics (processing time, speed ratio)
- Compare performance across different batch configurations
- Generate performance visualization charts
"""

import pandas as pd
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any


def analyze_timing_data(timing_csv_path: str = "Outputs/timing_results.csv") -> pd.DataFrame:
    """
    Load and analyze timing data from batch_process.py
    
    Args:
        timing_csv_path: Path to timing results CSV
    
    Returns:
        DataFrame with timing data and calculated metrics
    """
    if not os.path.exists(timing_csv_path):
        raise FileNotFoundError(f"Timing data not found: {timing_csv_path}")
    
    # Load timing data
    df = pd.read_csv(timing_csv_path)
    
    # Calculate additional metrics
    if 'processing_time_sec' in df.columns and 'video_duration_sec' in df.columns:
        # Speed ratio: how many seconds of processing per second of video
        df['speed_ratio'] = df['processing_time_sec'] / df['video_duration_sec']
        
        # Efficiency: inverse of speed ratio (higher is better)
        df['efficiency'] = 1 / df['speed_ratio']
    
    # Load BatchParameters for each batch_id if available
    if 'batch' in df.columns:
        batch_configs = []
        for batch_id in df['batch']:
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from video_processing.batch_parameters import BatchParameters
                params = BatchParameters.from_batch_id(batch_id)
                batch_configs.append({
                    'batch_id': batch_id,
                    'llm_provider': params.llm_provider.value,
                    'llm_model': params.llm_model,
                    'cv_model': params.cv_model.value
                })
            except Exception as e:
                batch_configs.append({
                    'batch_id': batch_id,
                    'llm_provider': 'unknown',
                    'llm_model': 'unknown',
                    'cv_model': 'unknown'
                })
        
        # Add config info to dataframe
        config_df = pd.DataFrame(batch_configs)
        df = df.merge(config_df, left_on='batch', right_on='batch_id', how='left')
    
    return df


def compare_batch_performance(batch_ids: List[str], timing_csv_path: str = "Outputs/timing_results.csv") -> Dict[str, Any]:
    """
    Compare performance across multiple batches
    
    Args:
        batch_ids: List of batch_ids to compare
        timing_csv_path: Path to timing results CSV
    
    Returns:
        Dictionary with comparison data and statistics
    """
    # Load timing data
    timing_data = analyze_timing_data(timing_csv_path)
    
    # Filter to selected batches
    if batch_ids:
        timing_data = timing_data[timing_data['batch'].isin(batch_ids)]
    
    if len(timing_data) == 0:
        return {
            'error': 'No timing data found for specified batch_ids',
            'summary': None
        }
    
    # Calculate summary statistics per batch
    summary = timing_data.groupby('batch').agg({
        'processing_time_sec': ['mean', 'std', 'min', 'max'],
        'video_duration_sec': 'sum',
        'speed_ratio': ['mean', 'std'],
        'video_name': 'count'
    }).round(2)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'video_name_count': 'videos_processed'})
    summary = summary.reset_index()
    
    # Find fastest and slowest batches
    avg_times = timing_data.groupby('batch')['processing_time_sec'].mean()
    fastest_batch = avg_times.idxmin()
    slowest_batch = avg_times.idxmax()
    
    return {
        'summary': summary,
        'full_data': timing_data,
        'fastest_batch': fastest_batch,
        'slowest_batch': slowest_batch,
        'fastest_time': avg_times.min(),
        'slowest_time': avg_times.max()
    }


def generate_performance_charts(timing_data: pd.DataFrame, output_dir: str) -> List[str]:
    """
    Generate performance visualization charts
    
    Args:
        timing_data: DataFrame with timing data
        output_dir: Directory to save charts
    
    Returns:
        List of chart file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Chart 1: Processing time per video (grouped by batch)
    if 'batch' in timing_data.columns and len(timing_data['batch'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Group by batch
        batches = timing_data['batch'].unique()
        x = range(len(timing_data))
        width = 0.8 / len(batches)
        
        for i, batch in enumerate(batches):
            batch_data = timing_data[timing_data['batch'] == batch]
            positions = [j + i * width for j in range(len(batch_data))]
            ax.bar(positions, batch_data['processing_time_sec'], 
                   width=width, label=f'Batch: {batch[:12]}...', alpha=0.8)
        
        ax.set_xlabel('Video', fontsize=12, fontweight='bold')
        ax.set_ylabel('Processing Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Processing Time by Video and Batch', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        chart1_path = os.path.join(output_dir, 'processing_time_comparison.png')
        plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart1_path)
    
    # Chart 2: Speed ratio distribution (box plot)
    if 'speed_ratio' in timing_data.columns and 'batch' in timing_data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create box plot
        batches = timing_data['batch'].unique()
        data_to_plot = [timing_data[timing_data['batch'] == batch]['speed_ratio'].values 
                        for batch in batches]
        
        bp = ax.boxplot(data_to_plot, labels=[b[:20] + '...' if len(b) > 20 else b for b in batches],
                        patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(range(len(batches)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Batch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speed Ratio (processing_time / video_duration)', fontsize=12, fontweight='bold')
        ax.set_title('Speed Ratio Distribution by Batch', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        # Add reference line at 1.0 (real-time processing)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Real-time (1.0)')
        ax.legend()
        
        plt.tight_layout()
        chart2_path = os.path.join(output_dir, 'speed_ratio_distribution.png')
        plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart2_path)
    
    # Chart 3: Average performance comparison (bar chart)
    if 'batch' in timing_data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate averages per batch
        avg_data = timing_data.groupby('batch').agg({
            'processing_time_sec': 'mean',
            'speed_ratio': 'mean'
        }).reset_index()
        
        x_pos = range(len(avg_data))
        bars = ax.bar(x_pos, avg_data['processing_time_sec'], 
                      color='#2196F3', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Batch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Processing Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Average Processing Time by Batch', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([b[:20] + '...' if len(b) > 20 else b for b in avg_data['batch']], 
                           rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, time, ratio in zip(bars, avg_data['processing_time_sec'], avg_data['speed_ratio']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}s\n({ratio:.2f}x)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        chart3_path = os.path.join(output_dir, 'batch_performance_comparison.png')
        plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart3_path)
    
    # Chart 4: Performance timeline (if timestamp available)
    if 'timestamp' in timing_data.columns and len(timing_data) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert timestamp to datetime if it's a string
        if timing_data['timestamp'].dtype == 'object':
            timing_data['timestamp_dt'] = pd.to_datetime(timing_data['timestamp'])
        else:
            timing_data['timestamp_dt'] = timing_data['timestamp']
        
        # Plot processing time over time
        for batch in timing_data['batch'].unique():
            batch_data = timing_data[timing_data['batch'] == batch].sort_values('timestamp_dt')
            ax.plot(batch_data['timestamp_dt'], batch_data['processing_time_sec'],
                    marker='o', label=f'Batch: {batch[:12]}...', linewidth=2)
        
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Processing Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Processing Time Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        chart4_path = os.path.join(output_dir, 'performance_timeline.png')
        plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart4_path)
    
    return chart_paths


def run_performance_analysis(batch_ids: List[str] = None, 
                            timing_csv_path: str = "Outputs/timing_results.csv",
                            output_dir: str = "benchmark_results/performance") -> Dict[str, Any]:
    """
    Run complete performance analysis
    
    Args:
        batch_ids: Optional list of batch_ids to analyze (None = all batches)
        timing_csv_path: Path to timing results CSV
        output_dir: Directory to save results and charts
    
    Returns:
        Dictionary with analysis results and chart paths
    """
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Load and analyze timing data
    print(f"\nLoading timing data from: {timing_csv_path}")
    timing_data = analyze_timing_data(timing_csv_path)
    
    # Filter to selected batches if specified
    if batch_ids:
        timing_data = timing_data[timing_data['batch'].isin(batch_ids)]
        print(f"Filtered to {len(batch_ids)} batch(es): {batch_ids}")
    
    print(f"Total records: {len(timing_data)}")
    print(f"Unique batches: {timing_data['batch'].nunique()}")
    print(f"Unique videos: {timing_data['video_name'].nunique()}")
    
    # Compare batch performance
    comparison = compare_batch_performance(batch_ids, timing_csv_path)
    
    if 'error' in comparison:
        print(f"\n✗ Error: {comparison['error']}")
        return comparison
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"\nFastest batch: {comparison['fastest_batch']}")
    print(f"  Average time: {comparison['fastest_time']:.2f} seconds")
    print(f"\nSlowest batch: {comparison['slowest_batch']}")
    print(f"  Average time: {comparison['slowest_time']:.2f} seconds")
    
    # Save summary to CSV
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'performance_summary.csv')
    comparison['summary'].to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Generate charts
    charts_dir = os.path.join(output_dir, 'charts')
    print(f"\nGenerating performance charts...")
    chart_paths = generate_performance_charts(timing_data, charts_dir)
    
    print(f"✓ Generated {len(chart_paths)} charts:")
    for chart_path in chart_paths:
        print(f"  - {chart_path}")
    
    print("\n" + "="*60)
    print("Performance analysis complete!")
    print("="*60)
    
    return {
        'summary': comparison['summary'],
        'timing_data': timing_data,
        'fastest_batch': comparison['fastest_batch'],
        'slowest_batch': comparison['slowest_batch'],
        'charts': chart_paths,
        'summary_path': summary_path
    }


if __name__ == "__main__":
    # Interactive CLI
    print("Performance Analysis Tool")
    print("="*60)
    
    timing_path = input("Timing CSV path (default: Outputs/timing_results.csv): ").strip()
    if not timing_path:
        timing_path = "Outputs/timing_results.csv"
    
    # Check if file exists
    if not os.path.exists(timing_path):
        print(f"Error: File not found: {timing_path}")
        exit(1)
    
    # Load data to show available batches
    timing_data = analyze_timing_data(timing_path)
    available_batches = timing_data['batch'].unique().tolist()
    
    print(f"\nFound {len(available_batches)} batch(es):")
    for i, batch in enumerate(available_batches, 1):
        print(f"  {i}. {batch}")
    
    # Ask which batches to analyze
    selection = input("\nAnalyze all batches? (Y/N): ").strip().upper()
    
    if selection == 'Y':
        batch_ids = None
        print("Will analyze all batches")
    else:
        batch_nums = input("Enter batch numbers to analyze (comma-separated): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in batch_nums.split(',')]
            batch_ids = [available_batches[i] for i in indices]
            print(f"Will analyze: {batch_ids}")
        except:
            print("Invalid selection, analyzing all batches")
            batch_ids = None
    
    # Run analysis
    results = run_performance_analysis(
        batch_ids=batch_ids,
        timing_csv_path=timing_path
    )
