## Configuration - batch processing
import pandas as pd
import os
from datetime import datetime

# Directory for model outputs and results
MODEL_DATA_DIR = "Outputs/Data2/"
RESULTS_FILE = "benchmark_summary.csv"

# Unified results files
UNIFIED_RESULTS_FILE = "benchmark_results/all_results.csv"
UNIFIED_SUMMARY_FILE = "benchmark_results/run_summary.csv"

## Parses the time stamp
def parse_timestamp(time_str):
    """Convert 'M:SS' to seconds as float."""
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])
    return minutes * 60 + seconds

def load_ground_truth(csv_path='TestData/LabelledData.csv'):
    """Load ground truth data and get available video IDs."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['Timestamp'] = df['Timestamp'].apply(parse_timestamp)
    
    # Replace NaN, empty strings, or None with "Unknown"
    df['Object'] = df['Object'].fillna('Unknown')
    df['Object'] = df['Object'].replace('', 'Unknown')
    df['Object'] = df['Object'].astype(str).replace('nan', 'Unknown')
    
    available_video_ids = sorted(df['Video ID'].unique())
    
    return df[['Video ID', 'Timestamp', 'State', 'Object']], available_video_ids

# Load ground truth data and get available video IDs
print("Loading ground truth data...")
testdata, gt_video_ids = load_ground_truth('TestData/LabelledData.csv')
print(f"Ground truth contains {len(gt_video_ids)} videos: {gt_video_ids}")

# Scan for available model output videos
def get_available_videos(directory, valid_video_ids):
    """Scan directory for video CSV files that exist in ground truth."""
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found!")
        return []
    
    files = [f for f in os.listdir(directory) if f.endswith('.csv') and f.startswith('video_')]
    video_names = []
    
    for f in files:
        video_name = f.replace('.csv', '')
        video_id = int(video_name.split('_')[1])
        
        # Only include if video ID exists in ground truth
        if video_id in valid_video_ids:
            video_names.append(video_name)
    
    # Sort by video number
    video_names.sort(key=lambda x: int(x.split('_')[1]))
    
    return video_names

# Get available videos that have both model outputs and ground truth
available_videos = get_available_videos(MODEL_DATA_DIR, gt_video_ids)

if not available_videos:
    print(f"\nNo video files found in {MODEL_DATA_DIR} that match ground truth data")
    exit(1)

print(f"\nFound {len(available_videos)} videos with both model outputs and ground truth:")
for video in available_videos:
    print(f"  {video}")

# Ask user for selection method
print("\nSelection options:")
print("  A - Process all available videos")
print("  Y/N - Ask for each video individually")
selection_mode = input("Select mode (A/Y): ").strip().upper()

videos_to_process = []

if selection_mode == 'A':
    videos_to_process = available_videos
    print(f"\nWill process all {len(videos_to_process)} videos")
else:
    # Ask Y/N for each video
    print("\nSelect videos to process:")
    for video_name in available_videos:
        response = input(f"Include {video_name}? (Y/N): ").strip().upper()
        if response == 'Y':
            videos_to_process.append(video_name)
    
    if not videos_to_process:
        print("No videos selected")
        exit(1)

# Show what will be processed
print(f"\nWill benchmark {len(videos_to_process)} videos: {videos_to_process}")
confirm = input("Continue? (Y/N): ").strip().upper()

if confirm != 'Y':
    print("Cancelled")
    exit(1)

batch_name = input("Batch_Name: ")
model_version = input("Model version (e.g., TestingClass2_v1): ").strip()
if not model_version:
    model_version = "unknown"
notes = input("Run notes (optional): ").strip()

RESULTS_DIR = "benchmark_results/" + batch_name

# Get next run_id
def get_next_run_id():
    """Get the next run_id by reading existing unified results."""
    if os.path.exists(UNIFIED_RESULTS_FILE):
        try:
            df = pd.read_csv(UNIFIED_RESULTS_FILE)
            if len(df) > 0 and 'run_id' in df.columns:
                return df['run_id'].max() + 1
        except:
            pass
    return 1

run_id = get_next_run_id()
run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"\nRun ID: {run_id}")
print(f"Model version: {model_version}")
if notes:
    print(f"Notes: {notes}")

## Parse prediction label
def parse_prediction_label(label):
    """Parse model label - currently just returns state."""
    return label  # Model outputs: 'using tool', 'idle', 'moving'

#Functions for loading data
def load_labeled_data(csv_path):
    """
    Load labeled data from CSV and convert frames to timestamps.
    
    Returns:
        tuple: (DataFrame, total_duration)
    """
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip().split(',')
        fps = float(first_line[0])
        total_duration = float(first_line[1])
    
    df = pd.read_csv(csv_path, skiprows=1, names=['frame', 'label', 'object'])
    df['timestamp'] = (df['frame'] - 1) / fps
    
    # Replace NaN, empty strings, or None with "Unknown"
    df['object'] = df['object'].fillna('Unknown')
    df['object'] = df['object'].replace('', 'Unknown')
    df['object'] = df['object'].astype(str).replace('nan', 'Unknown')
    
    return df, total_duration

def calculate_overlap(predicted_df, ground_truth_df, video_id, total_duration):
    """
    Calculate overlap metrics for state classification and object matching.
    """
    gt = ground_truth_df[ground_truth_df['Video ID'] == video_id].copy()
    gt = gt.sort_values('Timestamp').reset_index(drop=True)
    
    resolution = 0.1
    timeline_length = int(total_duration / resolution)
    
    pred_state = [''] * timeline_length
    pred_object = [''] * timeline_length
    gt_state = [''] * timeline_length
    gt_object = [''] * timeline_length
    
    # Fill predicted timeline
    for i in range(len(predicted_df)):
        start_time = predicted_df.iloc[i]['timestamp']
        end_time = predicted_df.iloc[i+1]['timestamp'] if i+1 < len(predicted_df) else total_duration
        
        start_idx = int(start_time / resolution)
        end_idx = int(end_time / resolution)
        
        state = parse_prediction_label(predicted_df.iloc[i]['label'])
        obj = predicted_df.iloc[i]['object']
        if pd.isna(obj) or obj == '' or str(obj) == 'nan':
            obj = 'Unknown'
        
        for idx in range(start_idx, min(end_idx, timeline_length)):
            pred_state[idx] = state
            pred_object[idx] = obj
    
    # Fill ground truth timeline
    for i in range(len(gt)):
        start_time = gt.iloc[i]['Timestamp']
        end_time = gt.iloc[i+1]['Timestamp'] if i+1 < len(gt) else total_duration
        
        start_idx = int(start_time / resolution)
        end_idx = int(end_time / resolution)
        
        state = gt.iloc[i]['State']
        obj = gt.iloc[i]['Object']
        if pd.isna(obj) or obj == '' or str(obj) == 'nan':
            obj = 'Unknown'
        
        for idx in range(start_idx, min(end_idx, timeline_length)):
            gt_state[idx] = state
            gt_object[idx] = obj
    
    # Calculate state accuracy
    state_correct = sum(1 for i in range(timeline_length) if pred_state[i] == gt_state[i])
    
    # Calculate object accuracy only when both are "using tool"
    using_tool_indices = [i for i in range(timeline_length) 
                          if pred_state[i] == 'using tool' and gt_state[i] == 'using tool']
    
    if using_tool_indices:
        object_correct = sum(1 for i in using_tool_indices 
                           if pred_object[i] == gt_object[i])
        object_accuracy = object_correct / len(using_tool_indices)
    else:
        object_accuracy = None
    
    return {
        'state_accuracy': state_correct / timeline_length,
        'object_accuracy': object_accuracy,
        'using_tool_time': len(using_tool_indices) * resolution,
        'total_time': total_duration
    }

# ----------------------------
# Main benchmark function
# ----------------------------
def run_benchmark(videos_to_process, batch_name, model_version, notes="", model_data_dir="Outputs/Data2/"):
    """
    Run benchmark on a list of videos.
    
    Args:
        videos_to_process: List of video names (e.g., ['video_01', 'video_05'])
        batch_name: Name for this benchmark batch
        model_version: Model version identifier
        notes: Optional notes about this run
        model_data_dir: Directory containing model output CSVs
    
    Returns:
        dict: Benchmark results including avg_state_accuracy, avg_object_accuracy
    """
    # Load ground truth
    testdata, gt_video_ids = load_ground_truth()
    
    # Filter to only videos with ground truth
    videos_with_gt = [v for v in videos_to_process if int(v.split('_')[1]) in gt_video_ids]
    
    if not videos_with_gt:
        print("No videos have ground truth data")
        return None
    
    # Setup directories
    RESULTS_DIR = "benchmark_results/" + batch_name
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Get run_id
    def get_next_run_id():
        if os.path.exists(UNIFIED_RESULTS_FILE):
            try:
                df = pd.read_csv(UNIFIED_RESULTS_FILE)
                if len(df) > 0 and 'run_id' in df.columns:
                    return df['run_id'].max() + 1
            except:
                pass
        return 1
    
    run_id = get_next_run_id()
    run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\nRun ID: {run_id}")
    print(f"Model version: {model_version}")
    if notes:
        print(f"Notes: {notes}")
    
    # Process videos
    all_results = {}
    results_records = []
    
    print("\nProcessing videos...")
    for i, video_name in enumerate(videos_with_gt, 1):
        print(f"\n{'='*50}")
        print(f"Processing {i}/{len(videos_with_gt)}: {video_name}")
        print(f"{'='*50}")
        
        video_id = int(video_name.split('_')[1])
        modelDataPath = f"{model_data_dir}{video_name}.csv"
        
        try:
            modeldata, total_duration = load_labeled_data(modelDataPath)
            results = calculate_overlap(modeldata, testdata, video_id=video_id, total_duration=total_duration)
            all_results[video_name] = results
            
            results_records.append({
                'run_id': run_id,
                'timestamp': run_timestamp,
                'batch_name': batch_name,
                'model_version': model_version,
                'video_name': video_name,
                'video_id': video_id,
                'state_accuracy': results['state_accuracy'],
                'object_accuracy': results['object_accuracy'] if results['object_accuracy'] is not None else None,
                'using_tool_time': results['using_tool_time'],
                'total_time': results['total_time'],
                'notes': notes
            })
            
            obj_str = f"{results['object_accuracy']:.2%}" if results['object_accuracy'] is not None else "N/A"
            print(f"✓ {video_name}: state={results['state_accuracy']:.2%}, object={obj_str}")
        
        except Exception as e:
            print(f"✗ ERROR processing {video_name}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if not all_results:
        print("No results to save")
        return None
    
    avg_state_accuracy = sum(r['state_accuracy'] for r in all_results.values()) / len(all_results)
    print(f"Average state accuracy: {avg_state_accuracy:.2%}")
    
    object_accuracies = [r['object_accuracy'] for r in all_results.values() if r['object_accuracy'] is not None]
    if object_accuracies:
        avg_object_accuracy = sum(object_accuracies) / len(object_accuracies)
        print(f"Average object accuracy (during 'using tool'): {avg_object_accuracy:.2%}")
        print(f"Videos with 'using tool' overlap: {len(object_accuracies)}/{len(all_results)}")
    else:
        avg_object_accuracy = None
        print("No 'using tool' overlap found for object comparison")
    
    # Save outputs to per-run folder
    results_df = pd.DataFrame(results_records)
    results_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Detailed results saved to: {results_path}")
    
    # Append to unified results file
    if os.path.exists(UNIFIED_RESULTS_FILE):
        results_df.to_csv(UNIFIED_RESULTS_FILE, mode='a', header=False, index=False)
    else:
        results_df.to_csv(UNIFIED_RESULTS_FILE, mode='w', header=True, index=False)
    print(f"✓ Results appended to unified file: {UNIFIED_RESULTS_FILE}")
    
    # Save summary statistics
    summary_data = {
        'metric': ['average_state_accuracy', 'average_object_accuracy', 'videos_with_tool_overlap', 'total_videos', 'total_duration'],
        'value': [
            avg_state_accuracy,
            avg_object_accuracy if avg_object_accuracy is not None else 'N/A',
            len(object_accuracies) if object_accuracies else 0,
            len(all_results),
            sum(r['total_time'] for r in all_results.values())
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(RESULTS_DIR, "benchmark_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary statistics saved to: {summary_path}")
    
    # Save metadata
    metadata = {
        'run_timestamp': [run_timestamp],
        'videos_processed': [', '.join(videos_with_gt)],
        'average_state_accuracy': [avg_state_accuracy],
        'average_object_accuracy': [avg_object_accuracy if avg_object_accuracy is not None else 'N/A']
    }
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(RESULTS_DIR, "benchmark_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Run metadata saved to: {metadata_path}")
    
    # Append to unified run summary
    summary_record = pd.DataFrame([{
        'run_id': run_id,
        'timestamp': run_timestamp,
        'batch_name': batch_name,
        'model_version': model_version,
        'videos_processed': len(videos_with_gt),
        'avg_state_accuracy': avg_state_accuracy,
        'avg_object_accuracy': avg_object_accuracy if avg_object_accuracy is not None else None,
        'total_duration': sum(r['total_time'] for r in all_results.values()),
        'notes': notes
    }])
    
    if os.path.exists(UNIFIED_SUMMARY_FILE):
        summary_record.to_csv(UNIFIED_SUMMARY_FILE, mode='a', header=False, index=False)
    else:
        summary_record.to_csv(UNIFIED_SUMMARY_FILE, mode='w', header=True, index=False)
    print(f"✓ Run summary appended to: {UNIFIED_SUMMARY_FILE}")
    
    return {
        'avg_state_accuracy': avg_state_accuracy,
        'avg_object_accuracy': avg_object_accuracy,
        'videos_processed': len(videos_with_gt),
        'results_path': results_path
    }

# ----------------------------
# Interactive CLI (when run standalone)
# ----------------------------
if __name__ == "__main__":
    # Load ground truth data and get available video IDs
    print("Loading ground truth data...")
    testdata, gt_video_ids = load_ground_truth('TestData/LabelledData.csv')
    print(f"Ground truth contains {len(gt_video_ids)} videos: {gt_video_ids}")
    
    # Get available videos that have both model outputs and ground truth
    available_videos = get_available_videos(MODEL_DATA_DIR, gt_video_ids)
    
    if not available_videos:
        print(f"\nNo video files found in {MODEL_DATA_DIR} that match ground truth data")
        exit(1)
    
    print(f"\nFound {len(available_videos)} videos with both model outputs and ground truth:")
    for video in available_videos:
        print(f"  {video}")
    
    # Ask user for selection method
    print("\nSelection options:")
    print("  A - Process all available videos")
    print("  Y/N - Ask for each video individually")
    selection_mode = input("Select mode (A/Y): ").strip().upper()
    
    videos_to_process = []
    
    if selection_mode == 'A':
        videos_to_process = available_videos
        print(f"\nWill process all {len(videos_to_process)} videos")
    else:
        # Ask Y/N for each video
        print("\nSelect videos to process:")
        for video_name in available_videos:
            response = input(f"Include {video_name}? (Y/N): ").strip().upper()
            if response == 'Y':
                videos_to_process.append(video_name)
        
        if not videos_to_process:
            print("No videos selected")
            exit(1)
    
    # Show what will be processed
    print(f"\nWill benchmark {len(videos_to_process)} videos: {videos_to_process}")
    confirm = input("Continue? (Y/N): ").strip().upper()
    
    if confirm != 'Y':
        print("Cancelled")
        exit(1)
    
    batch_name = input("Batch_Name: ")
    model_version = input("Model version (e.g., TestingClass2_v1): ").strip()
    if not model_version:
        model_version = "unknown"
    notes = input("Run notes (optional): ").strip()
    
    # Run benchmark
    results = run_benchmark(
        videos_to_process=videos_to_process,
        batch_name=batch_name,
        model_version=model_version,
        notes=notes,
        model_data_dir=MODEL_DATA_DIR
    )
    
    print("\n" + "="*50)
    print("Benchmark processing complete!")
    print("="*50)