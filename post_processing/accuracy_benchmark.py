## Configuration - batch processing
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Directory for model outputs and results
MODEL_DATA_DIR = "outputs/data/"
RESULTS_FILE = "benchmark_summary.csv"

# Unified results files
UNIFIED_RESULTS_FILE = "benchmark_results/all_results.csv"
UNIFIED_SUMMARY_FILE = "benchmark_results/run_summary.csv"

def repair_and_load_csv(file_path):
    """
    Robust CSV loader that handles ParserError by padding rows with missing fields.
    Useful for files where schema has evolved (new columns added).
    """
    try:
        return pd.read_csv(file_path)
    except pd.errors.ParserError:
        print(f"⚠ ParserError detected in {file_path}. Attempting to repair...")
        
        # Read raw lines
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return pd.DataFrame()
            
        # Parse header
        header = lines[0].strip().split(',')
        
        # Find max columns
        max_cols = len(header)
        data_rows = []
        
        for line in lines[1:]:
            row = line.strip().split(',')
            max_cols = max(max_cols, len(row))
            data_rows.append(row)
            
        # If max_cols > len(header), we have unnamed columns or data overflow
        # But usually it's the other way around or mixed.
        # Let's assume the header is correct for the OLD schema, but some rows have NEW schema (more cols)
        # OR header is NEW schema, but some rows are OLD schema (fewer cols)
        
        # Actually, the error "Expected 12 fields in line 10, saw 17" implies 
        # the header has 12 columns, but some data row has 17.
        # So we need to extend the header if possible, or just treat extra cols as unknown.
        
        # Better approach: Read with python csv module to handle quoting, then pad
        import csv
        import io
        
        # Re-read properly
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        if not rows:
            return pd.DataFrame()
            
        header = rows[0]
        max_len = max(len(r) for r in rows)
        
        # If header is shorter than max_len, generate dummy column names
        if len(header) < max_len:
            for i in range(len(header), max_len):
                header.append(f"extra_col_{i}")
                
        # Pad all rows to max_len
        padded_rows = []
        for r in rows[1:]:
            if len(r) < max_len:
                r.extend([None] * (max_len - len(r)))
            padded_rows.append(r)
            
        return pd.DataFrame(padded_rows, columns=header)
    except Exception as e:
        print(f"⚠ Failed to load/repair {file_path}: {e}")
        return pd.DataFrame()

## Parses the time stamp
def parse_timestamp(time_str):
    """Convert 'M:SS' to seconds as float."""
    if pd.isna(time_str):
        return 0.0
    
    if isinstance(time_str, (int, float)):
        return float(time_str)
        
    try:
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except (ValueError, IndexError):
        return 0.0

def load_ground_truth(csv_path='TestData/LabelledData.csv'):
    """Load ground truth data and get available video IDs."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['Timestamp'] = df['Timestamp'].apply(parse_timestamp)
    
    # Replace NaN, empty strings, or None with "Unknown"
    df['Object'] = df['Object'].fillna('Unknown')
    df['Object'] = df['Object'].replace('', 'Unknown')
    df['Object'] = df['Object'].astype(str).replace('nan', 'Unknown')
    
    # Ensure Video ID is integer
    # Filter out non-numeric IDs if any
    df['Video ID'] = pd.to_numeric(df['Video ID'], errors='coerce')
    df = df.dropna(subset=['Video ID'])
    df['Video ID'] = df['Video ID'].astype(int)
    
    available_video_ids = sorted(df['Video ID'].unique())
    
    # Ensure Unknown Object column exists
    if 'Unknown Object' not in df.columns:
        df['Unknown Object'] = 'Unknown'
    else:
        df['Unknown Object'] = df['Unknown Object'].fillna('Unknown')
        df['Unknown Object'] = df['Unknown Object'].replace('', 'Unknown')
        df['Unknown Object'] = df['Unknown Object'].astype(str).replace('nan', 'Unknown')
    
    return df[['Video ID', 'Timestamp', 'State', 'Object', 'Unknown Object']], available_video_ids

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

# This section only runs when script is executed directly, not when imported
if __name__ == "__main__":
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
    Handles both legacy and new formats.
    
    Returns:
        tuple: (DataFrame, total_duration)
    """
    # Read metadata
    with open(csv_path, 'r') as f:
        header_line = f.readline().strip().split(',')
        metadata_line = f.readline().strip().split(',')
        
        try:
            # Check if first line is values (legacy) or header (new)
            # If first line has 'fps' string, it's a header, so use second line values
            if 'fps' in header_line[0].lower():
                fps = float(metadata_line[0])
                total_duration = float(metadata_line[1])
            else:
                # Legacy: first line is values
                fps = float(header_line[0])
                total_duration = float(header_line[1])
        except ValueError:
            # Fallback
            fps = 30.0
            total_duration = 0.0
            
    # Try to determine header row
    # New format has header on line 3 (index 2)
    # Legacy format has no header, just data starting line 2
    
    try:
        # Try reading with header on line 3 (0-indexed index 2)
        df = pd.read_csv(csv_path, header=2)
        
        # Check if it looks like the new format
        if 'frame' in df.columns and 'action' in df.columns:
            # New format
            df = df.rename(columns={'action': 'label', 'tool': 'object'})
            if 'tool_guess' not in df.columns:
                df['tool_guess'] = 'Unknown'
        else:
            # Fallback/Legacy check
            raise ValueError("Not new format")
            
    except Exception:
        # Fallback to legacy read (skip 1 line of metadata, then data)
        df = pd.read_csv(csv_path, skiprows=1, names=['frame', 'label', 'object'])
        df['tool_guess'] = 'Unknown'

    # Ensure frame is numeric
    df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
    df = df.dropna(subset=['frame'])
    
    df['timestamp'] = (df['frame'] - 1) / fps
    
    # Replace NaN, empty strings, or None with "Unknown"
    for col in ['object', 'tool_guess']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            df[col] = df[col].replace('', 'Unknown')
            df[col] = df[col].astype(str).replace('nan', 'Unknown')
    
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
        
        # Effective Object Logic (Prediction)
        obj = predicted_df.iloc[i]['object']
        guess = predicted_df.iloc[i]['tool_guess']
        
        if pd.isna(obj) or obj == '' or str(obj) == 'nan':
            obj = 'Unknown'
        if pd.isna(guess) or guess == '' or str(guess) == 'nan':
            guess = 'Unknown'
            
        # If primary object is unknown/using unknown, use guess
        effective_obj = obj
        if (obj.lower() == 'unknown' or obj.lower() == 'using unknown') and guess.lower() != 'unknown':
            effective_obj = guess
        
        for idx in range(start_idx, min(end_idx, timeline_length)):
            pred_state[idx] = state
            pred_object[idx] = effective_obj
    
    # Fill ground truth timeline
    for i in range(len(gt)):
        start_time = gt.iloc[i]['Timestamp']
        end_time = gt.iloc[i+1]['Timestamp'] if i+1 < len(gt) else total_duration
        
        start_idx = int(start_time / resolution)
        end_idx = int(end_time / resolution)
        
        state = gt.iloc[i]['State']
        
        # Effective Object Logic (Ground Truth)
        obj = gt.iloc[i]['Object']
        unknown_obj = gt.iloc[i]['Unknown Object'] if 'Unknown Object' in gt.columns else 'Unknown'
        
        if pd.isna(obj) or obj == '' or str(obj) == 'nan':
            obj = 'Unknown'
        if pd.isna(unknown_obj) or unknown_obj == '' or str(unknown_obj) == 'nan':
            unknown_obj = 'Unknown'
            
        # If primary object is unknown, use specific unknown object
        effective_gt_obj = obj
        if obj.lower() == 'unknown' and unknown_obj.lower() != 'unknown':
            effective_gt_obj = unknown_obj
        
        for idx in range(start_idx, min(end_idx, timeline_length)):
            gt_state[idx] = state
            gt_object[idx] = effective_gt_obj
    
    # Calculate state accuracy
    state_correct_count = sum(1 for i in range(timeline_length) if pred_state[i] == gt_state[i])
    
    # Calculate object accuracy only when both are "using tool"
    using_tool_indices = [i for i in range(timeline_length) 
                          if pred_state[i] == 'using tool' and gt_state[i] == 'using tool']
    
    if using_tool_indices:
        object_correct = sum(1 for i in using_tool_indices 
                           if pred_object[i] == gt_object[i])
        object_accuracy = object_correct / len(using_tool_indices)
    else:
        object_accuracy = None
        
    # Calculate Guess Metrics
    # 1. Identify times where model was unsure (Unknown)
    # 2. Check if it made a guess
    # 3. Check if guess was correct
    
    unknown_pred_indices = []
    guesses_made_indices = []
    correct_guesses_indices = []
    
    for i in range(timeline_length):
        # Check original prediction (before effective object logic)
        # We need to look at the raw data, but here we only have the timeline arrays which already used effective logic.
        # However, we can infer "Unknown" prediction if the effective object came from a guess, 
        # OR we can re-check the source dataframe. 
        # Simpler approach: Check if the *primary* object in prediction was Unknown.
        # Since we don't have the raw primary object in the timeline arrays (we overwrote it with effective),
        # we'll approximate: If pred_state is 'using tool' and pred_object is NOT Unknown, 
        # but the logic used a guess... wait, we need to track this during the loop.
        pass

    # Better approach: Re-iterate or track during filling.
    # Let's track counts during the filling loop? No, resolution might duplicate.
    # Let's re-calculate using the dataframe for exactness, or just use the timeline if we store more info.
    
    # Let's redo the timeline filling to store "is_guess" flag
    pass
    
    # Actually, let's just do it on the dataframe level for simplicity? 
    # No, we need time-weighted overlap.
    
    # Let's add a "guess_made" and "raw_object" timeline
    guess_made = [False] * timeline_length
    
    # Refill loop for guess tracking (or merge with above, but for now separate for clarity/safety in replacement)
    for i in range(len(predicted_df)):
        start_time = predicted_df.iloc[i]['timestamp']
        end_time = predicted_df.iloc[i+1]['timestamp'] if i+1 < len(predicted_df) else total_duration
        start_idx = int(start_time / resolution)
        end_idx = int(end_time / resolution)
        
        obj = predicted_df.iloc[i]['object']
        guess = predicted_df.iloc[i]['tool_guess']
        
        # Check if this was a guess scenario: Primary is Unknown, Guess is valid
        is_guess = False
        if pd.isna(obj) or obj == '' or str(obj).lower() == 'nan' or str(obj).lower() == 'unknown' or str(obj).lower() == 'using unknown':
            if not (pd.isna(guess) or guess == '' or str(guess).lower() == 'nan' or str(guess).lower() == 'unknown'):
                is_guess = True
        
        if is_guess:
            for idx in range(start_idx, min(end_idx, timeline_length)):
                guess_made[idx] = True

    # Now calculate metrics
    # Denominator: Time where primary prediction was Unknown (we can infer this: if guess_made is True, it was Unknown. 
    # But what if it was Unknown and NO guess was made? We need to track that too.)
    
    # Let's simplify:
    # Guess Rate = (Time with Guess) / (Time with Unknown Primary)
    # Guess Accuracy = (Time with Correct Guess) / (Time with Guess)
    
    time_unknown_primary = 0
    time_with_guess = 0
    time_correct_guess = 0
    
    for i in range(len(predicted_df)):
        start_time = predicted_df.iloc[i]['timestamp']
        end_time = predicted_df.iloc[i+1]['timestamp'] if i+1 < len(predicted_df) else total_duration
        duration = end_time - start_time
        
        obj = predicted_df.iloc[i]['object']
        guess = predicted_df.iloc[i]['tool_guess']
        
        is_unknown_primary = False
        if pd.isna(obj) or obj == '' or str(obj).lower() == 'nan' or str(obj).lower() == 'unknown' or str(obj).lower() == 'using unknown':
            is_unknown_primary = True
            
        has_guess = False
        if not (pd.isna(guess) or guess == '' or str(guess).lower() == 'nan' or str(guess).lower() == 'unknown'):
            has_guess = True
            
        if is_unknown_primary:
            time_unknown_primary += duration
            if has_guess:
                time_with_guess += duration
                
                # Check correctness (need to check against GT timeline for this duration)
                # This is tricky because GT changes during this segment.
                # We should use the timeline arrays.
                pass

    # Timeline based calculation for correctness
    count_unknown_primary = 0
    count_with_guess = 0
    count_correct_guess = 0
    
    for idx in range(timeline_length):
        # Re-evaluate "Unknown Primary" at this timestep? 
        # We didn't store it. 
        # Let's assume if guess_made[idx] is True, then it was Unknown Primary.
        # But we miss the "Unknown Primary but NO guess" case.
        
        # To properly do this without massive refactor, let's just look at the "guess_made" slots.
        # If we made a guess, was it right?
        if guess_made[idx]:
            count_with_guess += 1
            if pred_object[idx] == gt_object[idx]: # pred_object is already the "effective" object (the guess)
                count_correct_guess += 1
                
    guess_accuracy = count_correct_guess / count_with_guess if count_with_guess > 0 else None
    
    # Calculate Over/Under Prediction Stats
    tool_over = defaultdict(float)
    tool_under = defaultdict(float)
    tool_correct = defaultdict(float)
    state_over = defaultdict(float)
    state_under = defaultdict(float)
    state_correct = defaultdict(float)
    
    for i in range(timeline_length):
        p_obj = str(pred_object[i]).lower() if pred_object[i] else 'unknown'
        g_obj = str(gt_object[i]).lower() if gt_object[i] else 'unknown'
        p_state = str(pred_state[i]).lower() if pred_state[i] else 'unknown'
        g_state = str(gt_state[i]).lower() if gt_state[i] else 'unknown'
        
        # Tool Over/Under/Correct
        if p_obj != g_obj:
            # If predicted is not unknown, it's an overprediction of that tool (since it's not in GT)
            if p_obj != 'unknown':
                tool_over[p_obj] += resolution
            # If GT is not unknown, it's an underprediction of that tool (since it wasn't predicted)
            if g_obj != 'unknown':
                tool_under[g_obj] += resolution
        else:
            # Correct identification
            if p_obj != 'unknown':
                tool_correct[p_obj] += resolution
                
        # State Over/Under/Correct
        if p_state != g_state:
            if p_state != 'unknown':
                state_over[p_state] += resolution
            if g_state != 'unknown':
                state_under[g_state] += resolution
        else:
            # Correct identification
            if p_state != 'unknown':
                state_correct[p_state] += resolution

    return {
        'state_accuracy': state_correct_count / timeline_length, # Note: state_correct variable name collision, using state_correct_count from earlier
        'object_accuracy': object_accuracy,
        'guess_accuracy': guess_accuracy,
        'using_tool_time': len(using_tool_indices) * resolution,
        'total_time': total_duration,
        'guess_time': count_with_guess * resolution,
        'over_under': {
            'tool_over': dict(tool_over),
            'tool_under': dict(tool_under),
            'tool_correct': dict(tool_correct),
            'state_over': dict(state_over),
            'state_under': dict(state_under),
            'state_correct': dict(state_correct)
        }
    }

def generate_accuracy_charts(results_df, output_dir):
    """
    Generate accuracy visualization charts
    
    Args:
        results_df: DataFrame with benchmark results
        output_dir: Directory to save charts
    
    Returns:
        List of chart paths
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Chart 1: Per-video accuracy bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = range(len(results_df))
    
    ax.bar([i - 0.2 for i in x_pos], results_df['state_accuracy'], 
           width=0.4, label='State Accuracy', color='#4CAF50', edgecolor='black')
    
    # Only plot object accuracy where it exists
    object_acc = results_df['object_accuracy'].fillna(0)
    ax.bar([i + 0.2 for i in x_pos], object_acc,
           width=0.4, label='Object Accuracy', color='#2196F3', edgecolor='black')
    
    ax.set_xlabel('Video', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy by Video', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['video_name'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, 'accuracy_by_video.png')
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart1_path)
    
    # Chart 2: Average accuracy summary
    fig, ax = plt.subplots(figsize=(8, 6))
    avg_state = results_df['state_accuracy'].mean()
    avg_object = results_df['object_accuracy'].dropna().mean() if results_df['object_accuracy'].notna().any() else 0
    
    categories = ['State Accuracy', 'Object Accuracy\n(when using tool)']
    values = [avg_state, avg_object]
    colors = ['#4CAF50', '#2196F3']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Average Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, 'average_accuracy.png')
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart2_path)
    
    return chart_paths

def generate_comparison_charts(results_list, output_dir):
    """
    Generate comparison charts for multiple batches.
    
    Args:
        results_list: List of result dictionaries from run_benchmark
        output_dir: Directory to save comparison charts
    """
    if not results_list:
        print("No results to compare")
        return []
        
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []
    
    # Prepare data for plotting
    data = []
    for res in results_list:
        batch_name = res.get('batch_id', 'Unknown')
        
        # Construct label: "ConfigName (Note)"
        label = batch_name
        if res.get('batch_params'):
            label = res['batch_params'].config_name
            
        # Add note if present
        note = res.get('notes', '')
        if note:
            label = f"{label} ({note})"
            
        # Store label in result for reuse
        res['label'] = label
            
        data.append({
            'Batch': label,
            'Batch ID': batch_name,
            'State Accuracy': res['avg_state_accuracy'],
            'Object Accuracy': res['avg_object_accuracy'] if res['avg_object_accuracy'] is not None else 0,
            'Guess Accuracy': res.get('avg_guess_accuracy', 0) if res.get('avg_guess_accuracy') is not None else 0
        })
        
    df = pd.DataFrame(data)
    
    # Save comparison CSV
    csv_path = os.path.join(output_dir, "batch_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Comparison CSV saved to: {csv_path}")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Chart 1: Comparative Accuracy
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Melt for grouped bar chart
    df_melted = df.melt(id_vars=['Batch', 'Batch ID'], 
                        value_vars=['State Accuracy', 'Object Accuracy', 'Guess Accuracy'],
                        var_name='Metric', value_name='Accuracy')
    
    # Plot
    sns.barplot(data=df_melted, x='Batch', y='Accuracy', hue='Metric', ax=ax, palette='viridis', edgecolor='black')
    
    ax.set_title('Batch Comparison: Average Accuracy', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Batch', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)
        
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, 'batch_comparison_accuracy.png')
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart1_path)
    
    # ---------------------------------------------------------
    # NEW: Per-Video Comparison Charts
    # ---------------------------------------------------------
    
    # Collect per-video data
    video_data = []
    for res in results_list:
        batch_label = res.get('label', res.get('batch_id', 'Unknown'))
        
        if 'per_video_results' in res:
            for vid_name, vid_res in res['per_video_results'].items():
                video_data.append({
                    'Video': vid_name,
                    'Batch': batch_label,
                    'State Accuracy': vid_res['state_accuracy'],
                    'Object Accuracy': vid_res['object_accuracy'] if vid_res['object_accuracy'] is not None else 0
                })
    
    if video_data:
        df_video = pd.DataFrame(video_data)
        
        # Chart 2: Per-Video State Accuracy
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=df_video, x='Video', y='State Accuracy', hue='Batch', ax=ax, palette='viridis', edgecolor='black')
        
        ax.set_title('Per-Video State Accuracy Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Video', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        chart2_path = os.path.join(output_dir, 'comparison_by_video_state.png')
        plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart2_path)
        
        # Chart 3: Per-Video Object Accuracy
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=df_video, x='Video', y='Object Accuracy', hue='Batch', ax=ax, palette='viridis', edgecolor='black')
        
        ax.set_title('Per-Video Object Accuracy Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Video', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        chart3_path = os.path.join(output_dir, 'comparison_by_video_object.png')
        plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart3_path)
    
    print(f"✓ Generated comparison charts in: {output_dir}")
    return chart_paths

def generate_performance_charts(results_df, output_dir):
    """
    Generate performance visualization charts.
    
    Args:
        results_df: DataFrame with benchmark results (including performance stats)
        output_dir: Directory to save charts
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Check if we have performance data
    if 'processing_time' not in results_df.columns or results_df['processing_time'].isnull().all():
        print("No performance data available for charts.")
        return []
        
    # Chart 1: Processing Time per Video
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=results_df, x='video_name', y='processing_time', ax=ax, color='#FF9800', edgecolor='black')
    
    ax.set_title('Processing Time per Video', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Video', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1fs', padding=3)
        
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, 'performance_time_per_video.png')
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart1_path)
    
    # Chart 2: Speed Ratio (Processing Time / Video Duration)
    if 'speed_ratio' in results_df.columns and results_df['speed_ratio'].notna().any():
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=results_df, x='video_name', y='speed_ratio', ax=ax, color='#9C27B0', edgecolor='black')
        
        ax.set_title('Speed Ratio (Lower is Faster)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Ratio (Proc Time / Vid Duration)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Video', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add reference line for 1.0 (Real-time)
        ax.axhline(1.0, color='red', linestyle='--', label='Real-time (1.0x)')
        ax.legend()
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2fx', padding=3)
            
        plt.tight_layout()
        chart2_path = os.path.join(output_dir, 'performance_speed_ratio.png')
        plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart2_path)
        
    return chart_paths

def generate_performance_comparison_charts(results_list, output_dir):
    """
    Generate performance comparison charts for multiple batches.
    """
    if not results_list:
        return []
        
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []
    
    # Prepare data
    data = []
    for res in results_list:
        label = res.get('label', res.get('batch_id', 'Unknown'))
        
        # Calculate aggregates if not present (though run_benchmark should provide them)
        # We'll use what's in the result dict
        
        # We need average processing time per video, or total?
        # Let's look at per-video results to get averages
        if 'per_video_results' in res:
            times = [v['processing_time'] for v in res['per_video_results'].values() if v.get('processing_time')]
            ratios = [v['speed_ratio'] for v in res['per_video_results'].values() if v.get('speed_ratio')]
            
            avg_time = sum(times) / len(times) if times else 0
            avg_ratio = sum(ratios) / len(ratios) if ratios else 0
            total_time = sum(times)
            
            data.append({
                'Batch': label,
                'Avg Processing Time (s)': avg_time,
                'Avg Speed Ratio': avg_ratio,
                'Total Time (s)': total_time
            })
            
    if not data:
        return []
        
    df = pd.DataFrame(data)
    
    # Chart 1: Average Speed Ratio Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Batch', y='Avg Speed Ratio', hue='Batch', ax=ax, palette='magma', edgecolor='black', legend=False)
    
    ax.set_title('Batch Comparison: Average Speed Ratio', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speed Ratio (Lower is Faster)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Batch', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2fx', padding=3)
        
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, 'comparison_performance_speed.png')
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(chart1_path)
    
    return chart_paths

# ----------------------------
# Main benchmark function
# ----------------------------
def run_benchmark(videos_to_process, batch_name, model_version, notes="", model_data_dir="outputs/data/", batch_id=None, performance_mode="none", generate_over_under=False):
    """
    Run benchmark on a list of videos.
    
    Args:
        videos_to_process: List of video names (e.g., ['video_01', 'video_05'])
        batch_name: Name for this benchmark batch
        model_version: Model version identifier
        notes: Optional notes about this run
        model_data_dir: Directory containing model output CSVs
        batch_id: Optional batch_id to link results to BatchParameters configuration
    
    Returns:
        dict: Benchmark results including avg_state_accuracy, avg_object_accuracy, batch_params
    """
    # Load BatchParameters if batch_id provided
    batch_params = None
    if batch_id:
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from video_processing.batch_parameters import BatchParameters
            batch_params = BatchParameters.from_batch_id(batch_id)
            print(f"✓ Loaded BatchParameters for batch_id: {batch_id}")
            print(f"  LLM: {batch_params.llm_provider.value}, Model: {batch_params.llm_model}")
            print(f"  CV: {batch_params.cv_model.value}")
        except Exception as e:
            print(f"⚠ Could not load BatchParameters for {batch_id}: {e}")
            batch_params = None
    
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
                df = repair_and_load_csv(UNIFIED_RESULTS_FILE)
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
    
    # Aggregation for Over/Under
    agg_tool_over = defaultdict(float)
    agg_tool_under = defaultdict(float)
    agg_tool_correct = defaultdict(float)
    agg_state_over = defaultdict(float)
    agg_state_under = defaultdict(float)
    agg_state_correct = defaultdict(float)
    total_agg_duration = 0
    
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
            
            # Aggregate over/under stats
            if 'over_under' in results:
                ou = results['over_under']
                for k, v in ou['tool_over'].items(): agg_tool_over[k] += v
                for k, v in ou['tool_under'].items(): agg_tool_under[k] += v
                for k, v in ou['tool_correct'].items(): agg_tool_correct[k] += v
                for k, v in ou['state_over'].items(): agg_state_over[k] += v
                for k, v in ou['state_under'].items(): agg_state_under[k] += v
                for k, v in ou['state_correct'].items(): agg_state_correct[k] += v
            total_agg_duration += results['total_time']
            
            # Load performance stats from metadata if available
            perf_stats = {}
            try:
                import json
                metadata_path = os.path.join(model_data_dir, f"{video_name}_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                        if 'performance' in meta:
                            perf_stats = meta['performance']
            except Exception as e:
                print(f"  ⚠ Could not load metadata for {video_name}: {e}")

            # Update all_results with performance data for per-video comparison
            if perf_stats and video_name in all_results:
                all_results[video_name]['processing_time'] = perf_stats.get('total_time_seconds')
                all_results[video_name]['speed_ratio'] = perf_stats.get('speed_ratio')
            
            results_records.append({
                'run_id': run_id,
                'timestamp': run_timestamp,
                'batch_name': batch_name,
                'model_version': model_version,
                'batch_id': batch_id if batch_id else 'N/A',
                'video_name': video_name,
                'video_id': video_id,
                'state_accuracy': results['state_accuracy'],
                'object_accuracy': results['object_accuracy'] if results['object_accuracy'] is not None else None,
                'guess_accuracy': results['guess_accuracy'] if results['guess_accuracy'] is not None else None,
                'using_tool_time': results['using_tool_time'],
                'guess_time': results['guess_time'],
                'total_time': results['total_time'],
                'processing_time': perf_stats.get('total_time_seconds', meta.get('processing_time_seconds', None)),
                'speed_ratio': perf_stats.get('speed_ratio', None),
                'api_calls': perf_stats.get('api_calls_total', None),
                'notes': notes
            })
            
            obj_str = f"{results['object_accuracy']:.2%}" if results['object_accuracy'] is not None else "N/A"
            guess_str = f"{results['guess_accuracy']:.2%}" if results['guess_accuracy'] is not None else "N/A"
            print(f"✓ {video_name}: state={results['state_accuracy']:.2%}, object={obj_str}, guess={guess_str}")
        
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
        
    guess_accuracies = [r['guess_accuracy'] for r in all_results.values() if r['guess_accuracy'] is not None]
    if guess_accuracies:
        avg_guess_accuracy = sum(guess_accuracies) / len(guess_accuracies)
        print(f"Average guess accuracy: {avg_guess_accuracy:.2%}")
    else:
        avg_guess_accuracy = None
        
    # Performance Summary
    total_processing_time = sum(r['processing_time'] for r in results_records if r['processing_time'] is not None)
    total_api_calls = sum(r['api_calls'] for r in results_records if r['api_calls'] is not None)
    avg_speed_ratio = sum(r['speed_ratio'] for r in results_records if r['speed_ratio'] is not None) / len(results_records) if results_records else 0
    
    print(f"Total Processing Time: {total_processing_time:.2f}s")
    print(f"Average Speed Ratio: {avg_speed_ratio:.2f}x")
    
    # Save results to CSV

    print(f"Total API Calls: {total_api_calls}")
    
    # Save outputs to per-run folder
    results_df = pd.DataFrame(results_records)
    results_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Detailed results saved to: {results_path}")
    
    # Append to unified results file
    if os.path.exists(UNIFIED_RESULTS_FILE):
        try:
            # Read existing to check columns
            existing_df = repair_and_load_csv(UNIFIED_RESULTS_FILE)
            
            # Identify new columns in results_df that are not in existing_df
            new_cols = [c for c in results_df.columns if c not in existing_df.columns]
            
            # Identify missing columns in results_df that are in existing_df
            missing_cols = [c for c in existing_df.columns if c not in results_df.columns]
            
            if new_cols or missing_cols:
                # Schema mismatch - need to align
                print(f"⚠ Schema mismatch in {UNIFIED_RESULTS_FILE}. Aligning columns...")
                
                # Add new columns to existing_df (fill with NaN)
                for col in new_cols:
                    existing_df[col] = None
                    
                # Add missing columns to results_df (fill with NaN)
                for col in missing_cols:
                    results_df[col] = None
                
                # Reorder results_df to match existing_df (plus new columns at end)
                combined_cols = list(existing_df.columns)
                results_df = results_df[combined_cols]
                
                # Append and write back entire file to update header
                final_df = pd.concat([existing_df, results_df], ignore_index=True)
                final_df.to_csv(UNIFIED_RESULTS_FILE, index=False)
                print(f"✓ Updated unified file schema and appended results.")
            else:
                # Schemas match, just append
                results_df.to_csv(UNIFIED_RESULTS_FILE, mode='a', header=False, index=False)
                print(f"✓ Results appended to unified file: {UNIFIED_RESULTS_FILE}")
                
        except Exception as e:
            print(f"⚠ Error updating unified file: {e}")
            # Fallback to append (might fail if columns don't match)
            results_df.to_csv(UNIFIED_RESULTS_FILE, mode='a', header=False, index=False)
    else:
        results_df.to_csv(UNIFIED_RESULTS_FILE, mode='w', header=True, index=False)
        print(f"✓ Results appended to unified file: {UNIFIED_RESULTS_FILE}")
    
    # Save summary statistics
    summary_data = {
        'metric': ['average_state_accuracy', 'average_object_accuracy', 'average_guess_accuracy', 'videos_with_tool_overlap', 'total_videos', 'total_duration', 'total_processing_time', 'avg_speed_ratio', 'total_api_calls'],
        'value': [
            avg_state_accuracy,
            avg_object_accuracy if avg_object_accuracy is not None else 'N/A',
            avg_guess_accuracy if avg_guess_accuracy is not None else 'N/A',
            len(object_accuracies) if object_accuracies else 0,
            len(all_results),
            sum(r['total_time'] for r in all_results.values()),
            total_processing_time,
            avg_speed_ratio,
            total_api_calls
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
        'average_object_accuracy': [avg_object_accuracy if avg_object_accuracy is not None else 'N/A'],
        'average_guess_accuracy': [avg_guess_accuracy if avg_guess_accuracy is not None else 'N/A'],
        'avg_speed_ratio': [avg_speed_ratio]
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
        'avg_guess_accuracy': avg_guess_accuracy if avg_guess_accuracy is not None else None,
        'avg_speed_ratio': avg_speed_ratio,
        'total_duration': sum(r['total_time'] for r in all_results.values()),
        'notes': notes
    }])
    
    if os.path.exists(UNIFIED_SUMMARY_FILE):
        summary_record.to_csv(UNIFIED_SUMMARY_FILE, mode='a', header=False, index=False)
    else:
        summary_record.to_csv(UNIFIED_SUMMARY_FILE, mode='w', header=True, index=False)
    print(f"✓ Run summary appended to: {UNIFIED_SUMMARY_FILE}")
    
    # Generate accuracy charts
    charts_dir = os.path.join(RESULTS_DIR, "charts")
    try:
        chart_paths = generate_accuracy_charts(results_df, charts_dir)
        print(f"\n✓ Generated {len(chart_paths)} accuracy charts:")
        for chart_path in chart_paths:
            print(f"  - {chart_path}")
            
        # Generate performance charts if requested
        if performance_mode == "full":
            print("Generating performance charts...")
            perf_charts = generate_performance_charts(results_df, charts_dir)
            chart_paths.extend(perf_charts)
            for path in perf_charts:
                print(f"  - {path}")
    except Exception as e:
        print(f"⚠ Error generating charts: {e}")
        chart_paths = []
        
    if generate_over_under:
        print("\nGenerating over/under prediction charts...")
        
        # Get allowed tools if available
        allowed_tools = None
        if batch_params and hasattr(batch_params, 'allowed_tools'):
            allowed_tools = [t.lower() for t in batch_params.allowed_tools]
            
        try:
            generate_over_under_charts({
                'tool_over': agg_tool_over,
                'tool_under': agg_tool_under,
                'tool_correct': agg_tool_correct,
                'state_over': agg_state_over,
                'state_under': agg_state_under,
                'state_correct': agg_state_correct
            }, total_agg_duration, charts_dir, allowed_tools=allowed_tools)
            print(f"✓ Over/under charts saved to: {charts_dir}")
        except Exception as e:
            print(f"⚠ Error generating over/under charts: {e}")
            
    # Generate Benchmark Readme
    try:
        # Path to original readme
        original_readme_path = os.path.join(model_data_dir, "readme.txt")
        new_readme_path = os.path.join(RESULTS_DIR, "readme.txt")
        
        readme_content = ""
        if os.path.exists(original_readme_path):
            with open(original_readme_path, 'r') as f:
                readme_content = f.read()
        else:
            readme_content = f"Batch ID: {batch_id}\n(Original readme not found)\n"
            
        with open(new_readme_path, 'w') as f:
            f.write(readme_content)
            f.write("\n" + "=" * 40 + "\n")
            f.write("BENCHMARK RESULTS\n")
            f.write("=" * 40 + "\n")
            f.write(f"Date: {run_timestamp}\n")
            f.write(f"Videos Processed: {len(videos_with_gt)}\n")
            f.write(f"Average State Accuracy: {avg_state_accuracy:.2%}\n")
            if avg_object_accuracy is not None:
                f.write(f"Average Object Accuracy: {avg_object_accuracy:.2%}\n")
            else:
                f.write("Average Object Accuracy: N/A\n")
            f.write(f"Total Processing Time: {total_processing_time:.2f}s\n")
            f.write(f"Average Speed Ratio: {avg_speed_ratio:.2f}x\n")
            
        print(f"✓ Created benchmark README at: {new_readme_path}")
    except Exception as e:
        print(f"⚠ Error creating benchmark README: {e}")
    
    return {
        'avg_state_accuracy': avg_state_accuracy,
        'avg_object_accuracy': avg_object_accuracy,
        'videos_processed': len(videos_with_gt),
        'results_path': results_path,
        'batch_id': batch_id,
        'batch_params': batch_params,  # Include loaded BatchParameters
        'charts': chart_paths,  # Add chart paths
        'per_video_results': all_results, # Return detailed results for comparison
        'notes': notes
    }



def generate_over_under_charts(aggregated_over_under, total_duration, output_dir, allowed_tools=None):
    """
    Generate charts for over/under prediction of tools and states.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper to plot
    # Helper to plot
    def plot_over_under(data_over, data_under, data_correct, title, filename, is_proportion=False, filter_keys=None):
        # Combine keys
        all_keys = sorted(list(set(data_over.keys()) | set(data_under.keys()) | set(data_correct.keys())))
        
        # Filter keys if provided
        if filter_keys:
            all_keys = [k for k in all_keys if k in filter_keys]
            
        if not all_keys:
            return
            
        over_vals = [data_over.get(k, 0) for k in all_keys]
        under_vals = [-data_under.get(k, 0) for k in all_keys] # Negative for under
        correct_vals = [data_correct.get(k, 0) for k in all_keys]
        
        if is_proportion and total_duration > 0:
            over_vals = [v / total_duration * 100 for v in over_vals]
            under_vals = [v / total_duration * 100 for v in under_vals]
            correct_vals = [v / total_duration * 100 for v in correct_vals]
            ylabel = "Percentage of Total Time (%)"
        else:
            ylabel = "Duration (seconds)"
            
        fig, ax = plt.subplots(figsize=(14, 7))
        x_pos = range(len(all_keys))
        width = 0.35
        
        # Plot Over/Under on the left (offset)
        ax.bar([x - width/2 for x in x_pos], over_vals, width=width, color='#FFC107', label='Overprediction (FP)', edgecolor='black')
        ax.bar([x - width/2 for x in x_pos], under_vals, width=width, color='#F44336', label='Underprediction (FN)', edgecolor='black')
        
        # Plot Correct on the right (offset)
        ax.bar([x + width/2 for x in x_pos], correct_vals, width=width, color='#4CAF50', label='Correct (TP)', edgecolor='black')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_keys, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.axhline(0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
    # 1. Tool Over/Under (Seconds)
    plot_over_under(aggregated_over_under['tool_over'], aggregated_over_under['tool_under'], aggregated_over_under['tool_correct'],
                   "Tool Prediction Analysis (Seconds)", "tool_over_under_seconds.png", filter_keys=allowed_tools)
                   
    # 2. Tool Over/Under (Proportion)
    plot_over_under(aggregated_over_under['tool_over'], aggregated_over_under['tool_under'], aggregated_over_under['tool_correct'],
                   "Tool Prediction Analysis (%)", "tool_over_under_percent.png", is_proportion=True, filter_keys=allowed_tools)
                   
    # 3. State Over/Under (Seconds)
    plot_over_under(aggregated_over_under['state_over'], aggregated_over_under['state_under'], aggregated_over_under['state_correct'],
                   "State Prediction Analysis (Seconds)", "state_over_under_seconds.png")
                   
    # 4. State Over/Under (Proportion)
    plot_over_under(aggregated_over_under['state_over'], aggregated_over_under['state_under'], aggregated_over_under['state_correct'],
                   "State Prediction Analysis (%)", "state_over_under_percent.png", is_proportion=True)

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