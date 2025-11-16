## Configuration - only change this
video_names = ["video_01", "video_04", "video_05", "video_21", "video_22"]
arr = ["video_11", "video_16", ]

# Directory for CSV results
RESULTS_FILE = "benchmark_summary.csv"  # Output filename
batch_name = input("Batch_Name: ")

RESULTS_DIR = "benchmark_results/" + batch_name 

## Converts frames to time  
import pandas as pd
import os
from datetime import datetime

## Parses the time stamp
def parse_timestamp(time_str):
    """Convert 'M:SS' to seconds as float."""
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])
    return minutes * 60 + seconds

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
    
    df = pd.read_csv(csv_path, skiprows=1, names=['frame', 'label'])
    df['timestamp'] = (df['frame'] - 1) / fps
    
    return df, total_duration

def load_ground_truth(csv_path):
    """
    Load ground truth data and convert timestamps to seconds.
    
    Returns:
        DataFrame with columns: Video ID, Timestamp, State, Object
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['Timestamp'] = df['Timestamp'].apply(parse_timestamp)
    

    # States are filtered, maybe adjust later. 
    return df[['Video ID', 'Timestamp', 'State']] 

# Load ground truth data
testdata = load_ground_truth('TestData/LabelledData.csv')

def calculate_overlap(predicted_df, ground_truth_df, video_id, total_duration):
    """
    Calculate overlap metrics for state classification only.
    """
    gt = ground_truth_df[ground_truth_df['Video ID'] == video_id].copy()
    gt = gt.sort_values('Timestamp').reset_index(drop=True)
    
    resolution = 0.1
    timeline_length = int(total_duration / resolution)
    
    pred_state = [''] * timeline_length
    gt_state = [''] * timeline_length
    
    # Fill predicted timeline
    for i in range(len(predicted_df)):
        start_time = predicted_df.iloc[i]['timestamp']
        end_time = predicted_df.iloc[i+1]['timestamp'] if i+1 < len(predicted_df) else total_duration
        
        start_idx = int(start_time / resolution)
        end_idx = int(end_time / resolution)
        
        state = parse_prediction_label(predicted_df.iloc[i]['label'])
        
        for idx in range(start_idx, min(end_idx, timeline_length)):
            pred_state[idx] = state
    
    # Fill ground truth timeline
    for i in range(len(gt)):
        start_time = gt.iloc[i]['Timestamp']
        end_time = gt.iloc[i+1]['Timestamp'] if i+1 < len(gt) else total_duration
        
        start_idx = int(start_time / resolution)
        end_idx = int(end_time / resolution)
        
        state = gt.iloc[i]['State']
        
        for idx in range(start_idx, min(end_idx, timeline_length)):
            gt_state[idx] = state
    
    # Calculate metrics
    state_correct = sum(1 for i in range(timeline_length) if pred_state[i] == gt_state[i])
    
    return {
        'state_accuracy': state_correct / timeline_length,
        'total_time': total_duration
    }

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Iterate over all videos
all_results = {}
results_records = []  # Store detailed results for CSV

for video_name in video_names:
    # Extract video_id from name
    video_id = int(video_name.split('_')[1])
    
    # Load model data for this video
    modelDataPath = f"Outputs/Data/{video_name}.csv"
    modeldata, total_duration = load_labeled_data(modelDataPath)
    
    # Calculate overlap
    results = calculate_overlap(modeldata, testdata, video_id=video_id, total_duration=total_duration)
    all_results[video_name] = results
    
    # TODO: add more calculations here.

    # Store for CSV
    results_records.append({
        'video_name': video_name,
        'video_id': video_id,
        'state_accuracy': results['state_accuracy'],
        'total_time': results['total_time']
    })
    
    print(f"{video_name}: {results}")

# Print summary
print("\n=== SUMMARY ===")
avg_accuracy = sum(r['state_accuracy'] for r in all_results.values()) / len(all_results)
print(f"Average state accuracy: {avg_accuracy:.2%}")

# Save detailed results to CSV
results_df = pd.DataFrame(results_records)
results_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
results_df.to_csv(results_path, index=False)
print(f"\n✓ Detailed results saved to: {results_path}")

# Save summary statistics to separate CSV
summary_data = {
    'metric': ['average_state_accuracy', 'total_videos', 'total_duration'],
    'value': [
        avg_accuracy,
        len(all_results),
        sum(r['total_time'] for r in all_results.values())
    ]
}
summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(RESULTS_DIR, "benchmark_statistics.csv")
summary_df.to_csv(summary_path, index=False)
print(f"✓ Summary statistics saved to: {summary_path}")

# Save run metadata
metadata = {
    'run_timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
    'videos_processed': [', '.join(video_names)],
    'average_accuracy': [avg_accuracy]
}
metadata_df = pd.DataFrame(metadata)
metadata_path = os.path.join(RESULTS_DIR, "benchmark_metadata.csv")
metadata_df.to_csv(metadata_path, index=False)
print(f"✓ Run metadata saved to: {metadata_path}")