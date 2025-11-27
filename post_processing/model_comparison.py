"""
Model Comparison - Compare Different Model Configurations

Combines accuracy and performance data to:
- Compare multiple batch configurations
- Analyze accuracy vs performance tradeoffs
- Generate comprehensive comparison reports
- Provide recommendations
"""

import pandas as pd
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import json


def compare_models(batch_ids: List[str], 
                  include_accuracy: bool = True,
                  include_performance: bool = True) -> Dict[str, Any]:
    """
    Comprehensive model comparison across multiple batch configurations
    
    Args:
        batch_ids: List of batch_ids to compare
        include_accuracy: Include accuracy benchmarking data
        include_performance: Include performance timing data
    
    Returns:
        Dictionary with comparison data, charts, and recommendations
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"\nComparing {len(batch_ids)} batch configurations...")
    
    comparison_data = []
    
    # Load data for each batch
    for batch_id in batch_ids:
        batch_info = {'batch_id': batch_id}
        
        # Load BatchParameters
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from video_processing.batch_parameters import BatchParameters
            params = BatchParameters.from_batch_id(batch_id)
            
            batch_info['llm_provider'] = params.llm_provider.value
            batch_info['llm_model'] = params.llm_model
            batch_info['cv_model'] = params.cv_model.value
            batch_info['config_name'] = params.config_name
        except Exception as e:
            print(f"⚠ Could not load BatchParameters for {batch_id}: {e}")
            batch_info['llm_provider'] = 'unknown'
            batch_info['llm_model'] = 'unknown'
            batch_info['cv_model'] = 'unknown'
            batch_info['config_name'] = 'unknown'
        
        # Load accuracy data if requested
        if include_accuracy:
            try:
                # Check unified results file
                accuracy_file = "benchmark_results/all_results.csv"
                if os.path.exists(accuracy_file):
                    acc_df = pd.read_csv(accuracy_file)
                    batch_acc = acc_df[acc_df['batch_id'] == batch_id]
                    
                    if len(batch_acc) > 0:
                        batch_info['avg_state_accuracy'] = batch_acc['state_accuracy'].mean()
                        batch_info['avg_object_accuracy'] = batch_acc['object_accuracy'].dropna().mean() if batch_acc['object_accuracy'].notna().any() else None
                        batch_info['videos_benchmarked'] = len(batch_acc)
                    else:
                        batch_info['avg_state_accuracy'] = None
                        batch_info['avg_object_accuracy'] = None
                        batch_info['videos_benchmarked'] = 0
                else:
                    batch_info['avg_state_accuracy'] = None
                    batch_info['avg_object_accuracy'] = None
                    batch_info['videos_benchmarked'] = 0
            except Exception as e:
                print(f"⚠ Could not load accuracy data for {batch_id}: {e}")
                batch_info['avg_state_accuracy'] = None
                batch_info['avg_object_accuracy'] = None
                batch_info['videos_benchmarked'] = 0
        
        # Load performance data if requested
        if include_performance:
            try:
                timing_file = "outputs/timing_results.csv"
                if os.path.exists(timing_file):
                    timing_df = pd.read_csv(timing_file)
                    batch_timing = timing_df[timing_df['batch'] == batch_id]
                    
                    if len(batch_timing) > 0:
                        batch_info['avg_processing_time'] = batch_timing['processing_time_sec'].mean()
                        batch_info['avg_speed_ratio'] = (batch_timing['processing_time_sec'] / batch_timing['video_duration_sec']).mean()
                        batch_info['videos_processed'] = len(batch_timing)
                    else:
                        batch_info['avg_processing_time'] = None
                        batch_info['avg_speed_ratio'] = None
                        batch_info['videos_processed'] = 0
                else:
                    batch_info['avg_processing_time'] = None
                    batch_info['avg_speed_ratio'] = None
                    batch_info['videos_processed'] = 0
            except Exception as e:
                print(f"⚠ Could not load performance data for {batch_id}: {e}")
                batch_info['avg_processing_time'] = None
                batch_info['avg_speed_ratio'] = None
                batch_info['videos_processed'] = 0
        
        comparison_data.append(batch_info)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    # Analyze tradeoffs
    tradeoff_analysis = analyze_tradeoffs(comparison_df)
    
    # Generate recommendations
    recommendations = generate_recommendations(comparison_df, tradeoff_analysis)
    
    return {
        'comparison_table': comparison_df,
        'tradeoff_analysis': tradeoff_analysis,
        'recommendations': recommendations
    }


def analyze_tradeoffs(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze accuracy vs performance tradeoffs
    
    Args:
        comparison_df: DataFrame with comparison data
    
    Returns:
        Dictionary with tradeoff analysis
    """
    analysis = {}
    
    # Find best accuracy
    if 'avg_state_accuracy' in comparison_df.columns:
        valid_acc = comparison_df[comparison_df['avg_state_accuracy'].notna()]
        if len(valid_acc) > 0:
            best_acc_idx = valid_acc['avg_state_accuracy'].idxmax()
            analysis['best_accuracy'] = {
                'batch_id': valid_acc.loc[best_acc_idx, 'batch_id'],
                'accuracy': valid_acc.loc[best_acc_idx, 'avg_state_accuracy'],
                'config': valid_acc.loc[best_acc_idx, 'config_name'] if 'config_name' in valid_acc.columns else 'unknown'
            }
    
    # Find best performance (lowest processing time)
    if 'avg_processing_time' in comparison_df.columns:
        valid_perf = comparison_df[comparison_df['avg_processing_time'].notna()]
        if len(valid_perf) > 0:
            best_perf_idx = valid_perf['avg_processing_time'].idxmin()
            analysis['best_performance'] = {
                'batch_id': valid_perf.loc[best_perf_idx, 'batch_id'],
                'processing_time': valid_perf.loc[best_perf_idx, 'avg_processing_time'],
                'speed_ratio': valid_perf.loc[best_perf_idx, 'avg_speed_ratio'],
                'config': valid_perf.loc[best_perf_idx, 'config_name'] if 'config_name' in valid_perf.columns else 'unknown'
            }
    
    # Find best balance (if both metrics available)
    if 'avg_state_accuracy' in comparison_df.columns and 'avg_speed_ratio' in comparison_df.columns:
        valid_both = comparison_df[
            comparison_df['avg_state_accuracy'].notna() & 
            comparison_df['avg_speed_ratio'].notna()
        ].copy()
        
        if len(valid_both) > 0:
            # Normalize metrics (0-1 scale)
            valid_both['norm_accuracy'] = (valid_both['avg_state_accuracy'] - valid_both['avg_state_accuracy'].min()) / (valid_both['avg_state_accuracy'].max() - valid_both['avg_state_accuracy'].min())
            valid_both['norm_speed'] = 1 - ((valid_both['avg_speed_ratio'] - valid_both['avg_speed_ratio'].min()) / (valid_both['avg_speed_ratio'].max() - valid_both['avg_speed_ratio'].min()))
            
            # Combined score (equal weight)
            valid_both['combined_score'] = (valid_both['norm_accuracy'] + valid_both['norm_speed']) / 2
            
            best_balance_idx = valid_both['combined_score'].idxmax()
            analysis['best_balance'] = {
                'batch_id': valid_both.loc[best_balance_idx, 'batch_id'],
                'accuracy': valid_both.loc[best_balance_idx, 'avg_state_accuracy'],
                'speed_ratio': valid_both.loc[best_balance_idx, 'avg_speed_ratio'],
                'combined_score': valid_both.loc[best_balance_idx, 'combined_score'],
                'config': valid_both.loc[best_balance_idx, 'config_name'] if 'config_name' in valid_both.columns else 'unknown'
            }
    
    return analysis


def generate_recommendations(comparison_df: pd.DataFrame, tradeoff_analysis: Dict) -> List[str]:
    """
    Generate recommendations based on comparison data
    
    Args:
        comparison_df: DataFrame with comparison data
        tradeoff_analysis: Tradeoff analysis results
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Recommendation 1: Best for accuracy
    if 'best_accuracy' in tradeoff_analysis:
        best_acc = tradeoff_analysis['best_accuracy']
        recommendations.append(
            f"For highest accuracy ({best_acc['accuracy']:.1%}), use batch: {best_acc['batch_id'][:20]}... "
            f"(Config: {best_acc['config']})"
        )
    
    # Recommendation 2: Best for speed
    if 'best_performance' in tradeoff_analysis:
        best_perf = tradeoff_analysis['best_performance']
        recommendations.append(
            f"For fastest processing ({best_perf['processing_time']:.1f}s avg, {best_perf['speed_ratio']:.2f}x speed), "
            f"use batch: {best_perf['batch_id'][:20]}... (Config: {best_perf['config']})"
        )
    
    # Recommendation 3: Best balance
    if 'best_balance' in tradeoff_analysis:
        best_bal = tradeoff_analysis['best_balance']
        recommendations.append(
            f"For best balance (accuracy: {best_bal['accuracy']:.1%}, speed: {best_bal['speed_ratio']:.2f}x), "
            f"use batch: {best_bal['batch_id'][:20]}... (Config: {best_bal['config']})"
        )
    
    # Recommendation 4: Model-specific insights
    if 'llm_provider' in comparison_df.columns:
        llm_groups = comparison_df.groupby('llm_provider').agg({
            'avg_state_accuracy': 'mean',
            'avg_processing_time': 'mean'
        }).dropna()
        
        if len(llm_groups) > 1:
            best_llm_acc = llm_groups['avg_state_accuracy'].idxmax()
            best_llm_speed = llm_groups['avg_processing_time'].idxmin()
            
            recommendations.append(
                f"LLM Provider Analysis: {best_llm_acc} has best average accuracy, "
                f"{best_llm_speed} has best average speed"
            )
    
    return recommendations


def generate_tradeoff_chart(comparison_df: pd.DataFrame, output_path: str) -> str:
    """
    Generate accuracy vs performance tradeoff scatter plot
    
    Args:
        comparison_df: DataFrame with comparison data
        output_path: Path to save chart
    
    Returns:
        Path to saved chart
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Filter to rows with both metrics
    valid_data = comparison_df[
        comparison_df['avg_state_accuracy'].notna() & 
        comparison_df['avg_speed_ratio'].notna()
    ]
    
    if len(valid_data) == 0:
        plt.close()
        return None
    
    # Create scatter plot
    scatter = ax.scatter(
        valid_data['avg_speed_ratio'],
        valid_data['avg_state_accuracy'],
        s=200,
        alpha=0.6,
        c=range(len(valid_data)),
        cmap='viridis',
        edgecolors='black',
        linewidth=2
    )
    
    # Add labels for each point
    for idx, row in valid_data.iterrows():
        ax.annotate(
            row['config_name'] if 'config_name' in row and pd.notna(row['config_name']) else row['batch_id'][:8],
            (row['avg_speed_ratio'], row['avg_state_accuracy']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold'
        )
    
    ax.set_xlabel('Speed Ratio (lower is better)', fontsize=12, fontweight='bold')
    ax.set_ylabel('State Accuracy (higher is better)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Performance Tradeoff', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add quadrant lines
    median_speed = valid_data['avg_speed_ratio'].median()
    median_acc = valid_data['avg_state_accuracy'].median()
    ax.axvline(x=median_speed, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=median_acc, color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax.text(ax.get_xlim()[0] + 0.02, ax.get_ylim()[1] - 0.02, 
            'High Accuracy\nSlow', fontsize=9, alpha=0.5, va='top')
    ax.text(ax.get_xlim()[1] - 0.02, ax.get_ylim()[1] - 0.02,
            'High Accuracy\nFast', fontsize=9, alpha=0.5, va='top', ha='right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_comparison_report(batch_ids: List[str],
                              output_dir: str = "benchmark_results/comparisons") -> str:
    """
    Generate comprehensive HTML comparison report
    
    Args:
        batch_ids: List of batch_ids to compare
        output_dir: Directory to save report
    
    Returns:
        Path to HTML report
    """
    # Run comparison
    results = compare_models(batch_ids, include_accuracy=True, include_performance=True)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = os.path.join(output_dir, f"comparison_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Save comparison data
    comparison_csv = os.path.join(report_dir, "comparison_data.csv")
    results['comparison_table'].to_csv(comparison_csv, index=False)
    
    # Generate tradeoff chart
    chart_path = os.path.join(report_dir, "accuracy_vs_speed.png")
    generate_tradeoff_chart(results['comparison_table'], chart_path)
    
    # Generate HTML report
    html_path = os.path.join(report_dir, "comparison_report.html")
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; background: white; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .recommendation {{ background: #e3f2fd; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; }}
        .chart {{ text-align: center; margin: 30px 0; }}
        .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>Model Comparison Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Comparing {len(batch_ids)} batch configurations</p>
    
    <h2>Comparison Table</h2>
    {results['comparison_table'].to_html(index=False)}
    
    <h2>Recommendations</h2>
    {''.join(f'<div class="recommendation">• {rec}</div>' for rec in results['recommendations'])}
    
    <h2>Accuracy vs Performance Tradeoff</h2>
    <div class="chart">
        <img src="accuracy_vs_speed.png" alt="Tradeoff Chart">
    </div>
    
    <h2>Tradeoff Analysis</h2>
    <pre>{json.dumps(results['tradeoff_analysis'], indent=2)}</pre>
</body>
</html>
"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n✓ Comparison report saved to: {html_path}")
    print(f"✓ Comparison data saved to: {comparison_csv}")
    if os.path.exists(chart_path):
        print(f"✓ Tradeoff chart saved to: {chart_path}")
    
    return html_path


if __name__ == "__main__":
    # Interactive CLI
    print("Model Comparison Tool")
    print("="*60)
    
    # Load available batches
    from video_processing.batch_comparison import BatchRegistry
    
    registry = BatchRegistry()
    all_batches = registry.get_all_batch_ids()
    
    if not all_batches:
        print("No batches found in batch tracking directory")
        exit(1)
    
    print(f"\nFound {len(all_batches)} tracked batches:")
    for i, batch_id in enumerate(all_batches, 1):
        print(f"  {i}. {batch_id}")
    
    # Select batches to compare
    selection = input("\nEnter batch numbers to compare (comma-separated): ").strip()
    try:
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        batch_ids = [all_batches[i] for i in indices]
    except:
        print("Invalid selection")
        exit(1)
    
    print(f"\nComparing {len(batch_ids)} batches...")
    
    # Generate report
    report_path = generate_comparison_report(batch_ids)
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)
