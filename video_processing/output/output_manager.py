"""
Output Manager - CSV generation with batch tracking

Handles saving all output files (CSVs) with proper batch tracking metadata.
"""

from pathlib import Path
from typing import List, Dict, Optional
import csv


def save_actions_csv(
    video_name: str,
    frame_labels: List[str],
    fps: float,
    frame_count: int,
    batch_params,
    output_path: Optional[str] = None
) -> str:
    """
    Save action classifications to CSV with batch tracking.
    
    Outputs are organized in batch-specific folders to prevent overwrites.
    
    Format:
    - Line 1: fps,total_duration,batch_id
    - Subsequent lines: frame_number,action_label (only when action changes)
    
    Args:
        video_name: Name of video
        frame_labels: List of action labels (one per frame)
        fps: Frames per second
        frame_count: Total number of frames
        batch_params: BatchParameters instance
        output_path: Optional custom output path
    
    Returns:
        Path to saved CSV file
    """
    if output_path is None:
        # Organize outputs by batch ID to prevent overwrites
        batch_folder = Path(batch_params.csv_directory) / batch_params.batch_id
        output_path = batch_folder / f"{video_name}.csv"
    else:
        output_path = Path(output_path)
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_duration = frame_count / fps
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header line with batch tracking
        if batch_params.track_model_versions:
            writer.writerow([fps, total_duration, batch_params.batch_id])
        else:
            writer.writerow([fps, total_duration])
        
        # Write state changes only
        if frame_count > 0:
            current_label = frame_labels[0]
            writer.writerow([1, current_label])
            
            for i in range(1, frame_count):
                if frame_labels[i] != current_label:
                    current_label = frame_labels[i]
                    writer.writerow([i + 1, current_label])
    
    return str(output_path)


def save_relationships_csv(
    video_name: str,
    relationships_data: List[Dict],
    fps: float,
    batch_params,
    output_path: Optional[str] = None
) -> str:
    """
    Save object relationships to CSV with batch tracking.
    
    Outputs are organized in batch-specific folders to prevent overwrites.
    
    Format:
    - Line 1: batch_id (if tracking enabled)
    - Line 2: start_frame,end_frame,start_time,end_time,duration,objects
    - Subsequent lines: relationship data
    
    Args:
        video_name: Name of video
        relationships_data: List of relationship dicts from RelationshipTracker
        fps: Frames per second
        batch_params: BatchParameters instance
        output_path: Optional custom output path
    
    Returns:
        Path to saved CSV file
    """
    if output_path is None:
        # Organize outputs by batch ID to prevent overwrites
        batch_folder = Path(batch_params.csv_directory) / batch_params.batch_id
        output_path = batch_folder / f"{video_name}_relationships.csv"
    else:
        output_path = Path(output_path)
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Optional batch tracking header
        if batch_params.track_model_versions:
            writer.writerow(['batch_id', batch_params.batch_id])
        
        # Column headers
        writer.writerow(['start_frame', 'end_frame', 'start_time', 'end_time', 'duration', 'objects'])
        
        # Write relationship data
        for rel in relationships_data:
            start_time = (rel['start_frame'] - 1) / fps
            end_time = (rel['end_frame'] - 1) / fps
            duration = end_time - start_time
            objects_str = ', '.join(sorted(rel['objects']))
            
            writer.writerow([
                rel['start_frame'],
                rel['end_frame'],
                f"{start_time:.2f}",
                f"{end_time:.2f}",
                f"{duration:.2f}",
                objects_str
            ])
    
    return str(output_path)


def save_batch_metadata(
    video_name: str,
    batch_params,
    processing_time: Optional[float] = None,
    output_files: Optional[Dict[str, str]] = None
) -> str:
    """
    Save batch metadata for this video processing run.
    
    Outputs are organized in batch-specific folders to prevent overwrites.
    
    Creates a JSON file with:
    - Batch parameters
    - Model versions
    - Processing time
    - Output file paths
    
    Args:
        video_name: Name of video
        batch_params: BatchParameters instance
        processing_time: Optional processing time in seconds
        output_files: Optional dict of output_type -> file_path
    
    Returns:
        Path to saved metadata file
    """
    import json
    from datetime import datetime
    
    # Organize outputs by batch ID to prevent overwrites
    batch_folder = Path(batch_params.csv_directory) / batch_params.batch_id
    metadata_path = batch_folder / f"{video_name}_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'video_name': video_name,
        'batch_id': batch_params.batch_id,
        'config_name': batch_params.config_name,
        'experiment_id': batch_params.experiment_id,
        'processed_at': datetime.now().isoformat(),
        'model_versions': batch_params.get_model_versions(),
        'parameters': batch_params.to_dict(),
    }
    
    if processing_time is not None:
        metadata['processing_time_seconds'] = processing_time
    
    if output_files is not None:
        metadata['output_files'] = output_files
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return str(metadata_path)
