"""
Batch Comparison Utilities

Tools for comparing and analyzing different batch runs based on parameters.
Integrates with existing batch_process.py timing data and Benchmark.py results.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict


class BatchRegistry:
    """
    Registry for tracking and comparing batch runs.
    
    Provides easy access to:
    - All batch configurations
    - Grouping batches by parameter subsets
    - Comparing performance across configurations
    """
    
    def __init__(self, tracking_directory: str = "outputs/batch_tracking/"):
        self.tracking_directory = Path(tracking_directory)
        self.tracking_directory.mkdir(parents=True, exist_ok=True)
    
    def get_all_batch_ids(self) -> List[str]:
        """Get list of all batch IDs"""
        batch_files = list(self.tracking_directory.glob("batch_*.json"))
        return [f.stem for f in batch_files]
    
    def get_batch_config(self, batch_id: str) -> Dict[str, Any]:
        """Load configuration for a specific batch"""
        config_path = self.tracking_directory / f"{batch_id}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No configuration found for batch_id: {batch_id}")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all batch configurations as dict {batch_id: config}"""
        configs = {}
        for batch_id in self.get_all_batch_ids():
            configs[batch_id] = self.get_batch_config(batch_id)
        return configs
    
    def group_by_parameters(
        self,
        params_to_group: List[str],
        batch_ids: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Group batches by specific parameter values.
        
        Example:
            # Group all batches by CV model
            groups = registry.group_by_parameters(['cv_model'])
            # Returns: {
            #   'cv_model=yolo_current': ['batch_001', 'batch_002'],
            #   'cv_model=yolo_v8': ['batch_003', 'batch_004']
            # }
        
        Args:
            params_to_group: List of parameter names to group by
            batch_ids: Optional list of batch_ids to consider (default: all)
        
        Returns:
            Dict mapping parameter combination -> list of batch_ids
        """
        if batch_ids is None:
            batch_ids = self.get_all_batch_ids()
        
        groups = defaultdict(list)
        
        for batch_id in batch_ids:
            config = self.get_batch_config(batch_id)
            
            # Build key from parameters
            parts = []
            for param in sorted(params_to_group):
                if param in config:
                    value = config[param]
                    # Handle enum values
                    if isinstance(value, dict) and 'value' in value:
                        value = value['value']
                    parts.append(f"{param}={value}")
            
            key = "|".join(parts) if parts else "ungrouped"
            groups[key].append(batch_id)
        
        return dict(groups)
    
    def compare_batches(
        self,
        batch_ids: List[str],
        params_to_compare: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create comparison table of batches.
        
        Args:
            batch_ids: List of batch IDs to compare
            params_to_compare: Optional list of specific parameters to show
                              (default: show all parameters that differ)
        
        Returns:
            DataFrame with batch_ids as rows and parameters as columns
        """
        configs = [self.get_batch_config(bid) for bid in batch_ids]
        
        # If params not specified, find all params that differ
        if params_to_compare is None:
            all_params = set()
            for config in configs:
                all_params.update(config.keys())
            
            # Find params that differ across batches
            params_to_compare = []
            for param in all_params:
                values = [config.get(param) for config in configs]
                # Convert to strings for comparison
                values_str = [str(v) for v in values]
                if len(set(values_str)) > 1:  # Differs across batches
                    params_to_compare.append(param)
        
        # Build comparison table
        data = []
        for batch_id, config in zip(batch_ids, configs):
            row = {'batch_id': batch_id}
            for param in params_to_compare:
                value = config.get(param, None)
                # Handle enum values
                if isinstance(value, dict) and 'value' in value:
                    value = value['value']
                row[param] = value
            data.append(row)
        
        return pd.DataFrame(data)
    
    def find_batches_with_params(self, **param_filters) -> List[str]:
        """
        Find all batches matching specific parameter values.
        
        Example:
            # Find all batches using YOLO v8 and Claude
            batch_ids = registry.find_batches_with_params(
                cv_model='yolo_v8',
                llm_provider='claude'
            )
        
        Args:
            **param_filters: Parameter name -> value pairs to match
        
        Returns:
            List of matching batch_ids
        """
        matching_batches = []
        
        for batch_id in self.get_all_batch_ids():
            config = self.get_batch_config(batch_id)
            
            # Check if all filters match
            matches = True
            for param, expected_value in param_filters.items():
                actual_value = config.get(param)
                # Handle enum values
                if isinstance(actual_value, dict) and 'value' in actual_value:
                    actual_value = actual_value['value']
                
                if actual_value != expected_value:
                    matches = False
                    break
            
            if matches:
                matching_batches.append(batch_id)
        
        return matching_batches
    
    def export_comparison_csv(
        self,
        output_path: str,
        batch_ids: Optional[List[str]] = None,
        params_to_include: Optional[List[str]] = None
    ):
        """
        Export batch comparison to CSV for analysis.
        
        Args:
            output_path: Path to save CSV
            batch_ids: Optional list of batch_ids (default: all)
            params_to_include: Optional list of parameters to include
        """
        if batch_ids is None:
            batch_ids = self.get_all_batch_ids()
        
        df = self.compare_batches(batch_ids, params_to_include)
        df.to_csv(output_path, index=False)
        return output_path


def merge_with_timing_data(
    batch_registry: BatchRegistry,
    timing_csv_path: str = "Outputs/timing_results.csv"
) -> pd.DataFrame:
    """
    Merge batch configurations with timing data from batch_process.py.
    
    This allows analyzing how different parameters affect processing speed.
    
    Args:
        batch_registry: BatchRegistry instance
        timing_csv_path: Path to timing_results.csv from batch_process.py
    
    Returns:
        DataFrame with batch configs + timing metrics
    """
    # Load timing data
    timing_df = pd.read_csv(timing_csv_path)
    
    # Get all batch configs
    configs = batch_registry.get_all_configs()
    
    # Merge based on batch_id or timestamp matching
    # (This assumes timing data includes batch_id or we match by timestamp)
    
    # For now, return timing data with note to add batch_id to timing_results.csv
    print("Note: To enable automatic merging, add 'batch_id' column to timing_results.csv")
    print("This can be done by passing batch_params.batch_id to batch_process.py")
    
    return timing_df


def merge_with_benchmark_data(
    batch_registry: BatchRegistry,
    benchmark_csv_path: str = "Outputs/benchmark_results.csv"
) -> pd.DataFrame:
    """
    Merge batch configurations with benchmark accuracy data.
    
    This allows analyzing how different parameters affect model quality.
    
    Args:
        batch_registry: BatchRegistry instance
        benchmark_csv_path: Path to benchmark results from Benchmark.py
    
    Returns:
        DataFrame with batch configs + accuracy metrics
    """
    # Load benchmark data
    benchmark_df = pd.read_csv(benchmark_csv_path)
    
    # Similar to timing data, needs batch_id in benchmark results
    print("Note: To enable automatic merging, add 'batch_id' column to benchmark results")
    print("This can be done by passing batch_params.batch_id to Benchmark.py")
    
    return benchmark_df


# Example usage functions

def example_compare_cv_models():
    """Example: Compare different CV models with everything else the same"""
    registry = BatchRegistry()
    
    # Group by CV model
    groups = registry.group_by_parameters(['cv_model'])
    
    print("Batches grouped by CV model:")
    for key, batch_ids in groups.items():
        print(f"  {key}: {len(batch_ids)} batches")
    
    # Compare within each group
    for key, batch_ids in groups.items():
        if len(batch_ids) > 1:
            print(f"\nComparing {key}:")
            df = registry.compare_batches(batch_ids)
            print(df)


def example_find_best_config():
    """Example: Find best performing configuration"""
    registry = BatchRegistry()
    
    # This would merge with timing/benchmark data
    # For now, just show how to filter
    
    # Find all batches using specific settings
    fast_batches = registry.find_batches_with_params(
        preload_all_frames=True,
        max_workers_keyframes=16
    )
    
    print(f"Found {len(fast_batches)} batches with fast settings")
    
    # Compare them
    if fast_batches:
        df = registry.compare_batches(fast_batches)
        print(df)
