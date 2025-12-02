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
    """Robust CSV loader that handles ParserError by padding rows."""
    try:
        return pd.read_csv(file_path)
    except pd.errors.ParserError:
        try:
            import csv
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows: return pd.DataFrame()
            header = rows[0]
            max_len = max(len(r) for r in rows)
            if len(header) < max_len:
                header += [f"extra_col_{i}" for i in range(len(header), max_len)]
            padded_rows = []
            for r in rows[1:]:
                if len(r) < max_len: r.extend([None] * (max_len - len(r)))
                padded_rows.append(r)
            return pd.DataFrame(padded_rows, columns=header)
        except Exception: return pd.DataFrame()
    except Exception: return pd.DataFrame()

def parse_timestamp(time_str):
    """Convert 'M:SS' to seconds as float."""
    if pd.isna(time_str): return 0.0
    if isinstance(time_str, (int, float)): return float(time_str)
    try:
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        return float(time_str)
    except: return 0.0

def load_ground_truth(csv_path='TestData/LabelledData.csv'):
    """Load ground truth data and get available video IDs."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['Timestamp'] = df['Timestamp'].apply(parse_timestamp)
    df['Object'] = df['Object'].fillna('Unknown').replace('', 'Unknown').astype(str).replace('nan', 'Unknown')
    df['Video ID'] = pd.to_numeric(df['Video ID'], errors='coerce')
    df = df.dropna(subset=['Video ID'])
    df['Video ID'] = df['Video ID'].astype(int)
    
    if 'Unknown Object' not in df.columns: df['Unknown Object'] = 'Unknown'
    else: df['Unknown Object'] = df['Unknown Object'].fillna('Unknown').replace('', 'Unknown').astype(str).replace('nan', 'Unknown')
    
    return df[['Video ID', 'Timestamp', 'State', 'Object', 'Unknown Object']], sorted(df['Video ID'].unique())

def get_available_videos(directory, valid_video_ids):
    """Scan directory for video CSV files."""
    if not os.path.exists(directory): return []
    files = [f for f in os.listdir(directory) if f.endswith('.csv') and f.startswith('video_')]
    video_names = []
    for f in files:
        vid_id = int(f.replace('.csv', '').split('_')[1])
        if vid_id in valid_video_ids: video_names.append(f.replace('.csv', ''))
    video_names.sort(key=lambda x: int(x.split('_')[1]))
    return video_names

def parse_prediction_label(label):
    return label

def load_labeled_data(csv_path):
    """Load labeled data from CSV."""
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        meta = f.readline().strip().split(',')
        if 'fps' in header[0].lower():
            fps = float(meta[0]); duration = float(meta[1])
        else:
            fps = float(header[0]); duration = float(header[1])
    try:
        df = pd.read_csv(csv_path, header=2)
        if 'frame' in df.columns and 'action' in df.columns:
            df = df.rename(columns={'action': 'label', 'tool': 'object'})
            if 'tool_guess' not in df.columns: df['tool_guess'] = 'Unknown'
        else: raise ValueError()
    except:
        df = pd.read_csv(csv_path, skiprows=1, names=['frame', 'label', 'object'])
        df['tool_guess'] = 'Unknown'

    df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
    df = df.dropna(subset=['frame'])
    df['timestamp'] = (df['frame'] - 1) / fps
    for col in ['object', 'tool_guess']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').replace('', 'Unknown').astype(str).replace('nan', 'Unknown')
    return df, duration

def calculate_overlap(predicted_df, ground_truth_df, video_id, total_duration, gt_only=False):
    """
    Calculate metrics (Accuracy, F1, Precision, Recall) using Over/Under time data.
    """
    gt = ground_truth_df[ground_truth_df['Video ID'] == video_id].copy()
    gt = gt.sort_values('Timestamp').reset_index(drop=True)
    
    resolution = 0.1
    timeline_length = int(total_duration / resolution)
    
    pred_state = [''] * timeline_length; pred_object = [''] * timeline_length
    gt_state = [''] * timeline_length; gt_object = [''] * timeline_length
    
    # Alias Mapping
    def normalize_obj(name):
        n = str(name).lower()
        if n == 'power saw': return 'saw'
        if n in ['tape', 'measuring tape']: return 'measuring'
        return n

    # 1. Fill Prediction Timeline
    for i in range(len(predicted_df)):
        start_t = predicted_df.iloc[i]['timestamp']
        end_t = predicted_df.iloc[i+1]['timestamp'] if i+1 < len(predicted_df) else total_duration
        start_idx = int(start_t / resolution); end_idx = int(end_t / resolution)
        state = parse_prediction_label(predicted_df.iloc[i]['label'])
        obj = predicted_df.iloc[i]['object']
        guess = predicted_df.iloc[i]['tool_guess']
        effective_obj = obj
        if (str(obj).lower() in ['unknown', 'using unknown']) and str(guess).lower() != 'unknown':
            effective_obj = guess
        
        # Normalize
        effective_obj = normalize_obj(effective_obj)
        
        for idx in range(start_idx, min(end_idx, timeline_length)):
            pred_state[idx] = state; pred_object[idx] = effective_obj
            
    # 2. Fill GT Timeline
    for i in range(len(gt)):
        start_t = gt.iloc[i]['Timestamp']
        end_t = gt.iloc[i+1]['Timestamp'] if i+1 < len(gt) else total_duration
        start_idx = int(start_t / resolution); end_idx = int(end_t / resolution)
        state = gt.iloc[i]['State']; obj = gt.iloc[i]['Object']
        effective_obj = obj
        # REMOVED Fallback to Unknown Object column
        # unk = gt.iloc[i]['Unknown Object']
        # if str(obj).lower() == 'unknown' and str(unk).lower() != 'unknown': effective_obj = unk
        
        # Normalize
        effective_obj = normalize_obj(effective_obj)
        
        for idx in range(start_idx, min(end_idx, timeline_length)):
            gt_state[idx] = state; gt_object[idx] = effective_obj

    # 3. Calculate Over/Under & Accuracy
    state_correct_dur = defaultdict(float); state_over_dur = defaultdict(float); state_under_dur = defaultdict(float)
    tool_correct_dur = defaultdict(float); tool_over_dur = defaultdict(float); tool_under_dur = defaultdict(float)
    
    total_state_matches = 0; using_tool_matches = 0; total_using_tool_opportunities = 0
    all_observed_states = set()
    gt_observed_states = set()
    gt_observed_tools = set()
    
    for i in range(timeline_length):
        p_state = str(pred_state[i]).lower() if pred_state[i] else 'unknown'
        g_state = str(gt_state[i]).lower() if gt_state[i] else 'unknown'
        p_obj = str(pred_object[i]).lower() if pred_object[i] else 'unknown'
        g_obj = str(gt_object[i]).lower() if gt_object[i] else 'unknown'
        
        if p_state != 'unknown': all_observed_states.add(p_state)
        if g_state != 'unknown': 
            all_observed_states.add(g_state)
            gt_observed_states.add(g_state)
        
        if g_obj != 'unknown': gt_observed_tools.add(g_obj)
        
        # State Logic
        if p_state == g_state:
            total_state_matches += 1
            if p_state != 'unknown': state_correct_dur[p_state] += resolution
        else:
            if p_state != 'unknown': state_over_dur[p_state] += resolution
            if g_state != 'unknown': state_under_dur[g_state] += resolution
                
        # Object Logic (Global)
        if p_obj == g_obj:
            if p_obj != 'unknown': tool_correct_dur[p_obj] += resolution
        else:
            if p_obj != 'unknown': tool_over_dur[p_obj] += resolution
            if g_obj != 'unknown': tool_under_dur[g_obj] += resolution

        # Object Accuracy (Traditional: Conditioned on "Using Tool")
        if p_state == 'using tool' and g_state == 'using tool':
            total_using_tool_opportunities += 1
            if p_obj == g_obj: using_tool_matches += 1

    # 4. Compute Metrics
    # State F1
    valid_classes = [s for s in all_observed_states if s != 'unknown']
    f1_sum = 0; count_classes = 0; idle_recall = 0.0
    state_f1_per_class = {}
    
    # Micro F1 Accumulators
    state_tp_sum = 0; state_fp_sum = 0; state_fn_sum = 0
    
    for state in valid_classes:
        tp = state_correct_dur[state]; fp = state_over_dur[state]; fn = state_under_dur[state]
        
        # Macro F1
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*(prec*rec)/(prec+rec) if (prec+rec)>0 else 0
        state_f1_per_class[state] = f1
        f1_sum += f1; count_classes += 1
        
        # Micro F1
        state_tp_sum += tp; state_fp_sum += fp; state_fn_sum += fn
        
        if state == 'idle': idle_recall = rec
        
    macro_f1 = f1_sum / count_classes if count_classes > 0 else 0
    
    # Micro F1 Calculation
    state_micro_prec = state_tp_sum / (state_tp_sum + state_fp_sum) if (state_tp_sum + state_fp_sum) > 0 else 0
    state_micro_rec = state_tp_sum / (state_tp_sum + state_fn_sum) if (state_tp_sum + state_fn_sum) > 0 else 0
    micro_f1 = 2 * (state_micro_prec * state_micro_rec) / (state_micro_prec + state_micro_rec) if (state_micro_prec + state_micro_rec) > 0 else 0
    
    # Object F1
    valid_tools = set(tool_correct_dur.keys()) | set(tool_over_dur.keys())    # GT-Only Filter
    # GT-Only Filter
    if gt_only:
        valid_tools = [t for t in valid_tools if t in gt_observed_tools]
    
    obj_f1_sum = 0; obj_count_classes = 0
    object_f1_per_class = {}
    
    # Micro F1 Accumulators
    obj_tp_sum = 0; obj_fp_sum = 0; obj_fn_sum = 0
    
    for tool in valid_tools:
        tp = tool_correct_dur[tool]; fp = tool_over_dur[tool]; fn = tool_under_dur[tool]
        
        # Macro F1
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*(prec*rec)/(prec+rec) if (prec+rec)>0 else 0
        object_f1_per_class[tool] = f1
        obj_f1_sum += f1; obj_count_classes += 1
        
        # Micro F1
        obj_tp_sum += tp; obj_fp_sum += fp; obj_fn_sum += fn
            
    object_macro_f1 = obj_f1_sum / obj_count_classes if obj_count_classes > 0 else 0
    
    # Micro F1 Calculation
    obj_micro_prec = obj_tp_sum / (obj_tp_sum + obj_fp_sum) if (obj_tp_sum + obj_fp_sum) > 0 else 0
    obj_micro_rec = obj_tp_sum / (obj_tp_sum + obj_fn_sum) if (obj_tp_sum + obj_fn_sum) > 0 else 0
    object_micro_f1 = 2 * (obj_micro_prec * obj_micro_rec) / (obj_micro_prec + obj_micro_rec) if (obj_micro_prec + obj_micro_rec) > 0 else 0

    state_accuracy = total_state_matches / timeline_length if timeline_length > 0 else 0
    object_accuracy_traditional = using_tool_matches / total_using_tool_opportunities if total_using_tool_opportunities > 0 else None

    return {
        'state_accuracy': state_accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1, # <--- NEW
        'idle_recall': idle_recall,
        'object_accuracy_traditional': object_accuracy_traditional,
        'object_macro_f1': object_macro_f1,
        'object_micro_f1': object_micro_f1, # <--- NEW
        'state_f1_per_class': state_f1_per_class,
        'object_f1_per_class': object_f1_per_class,
        'guess_accuracy': None,
        'using_tool_time': total_using_tool_opportunities * resolution,
        'total_time': total_duration,
        'over_under': {
            'tool_over': dict(tool_over_dur), 'tool_under': dict(tool_under_dur), 'tool_correct': dict(tool_correct_dur),
            'state_over': dict(state_over_dur), 'state_under': dict(state_under_dur), 'state_correct': dict(state_correct_dur)
        },
        'raw_timelines': {
            'pred_state': pred_state, 'gt_state': gt_state,
            'pred_object': pred_object, 'gt_object': gt_object
        }
    }

def generate_accuracy_charts(results_df, output_dir, metric_label="Accuracy"):
    """Generate charts with dynamic label."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    chart_paths = []
    
    # 1. Per-Video
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = range(len(results_df))
    # 'state_accuracy' column holds Swapped Score
    ax.bar([i-0.2 for i in x_pos], results_df['state_accuracy'], width=0.4, label=f'State {metric_label}', color='#4CAF50', edgecolor='black')
    
    # 'object_accuracy' column holds Swapped Score
    obj_acc = results_df['object_accuracy'].fillna(0)
    # Update Label to match metric
    ax.bar([i+0.2 for i in x_pos], obj_acc, width=0.4, label=f'Object {metric_label}', color='#2196F3', edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_label} by Video', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['video_name'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, 'accuracy_by_video.png')
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(p1)
    
    # 2. Average (Weighted by Duration)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Weighted State Average
    total_dur = results_df['total_time'].sum()
    if total_dur > 0:
        avg_state = (results_df['state_accuracy'] * results_df['total_time']).sum() / total_dur
    else:
        avg_state = 0
        
    # Weighted Object Average
    obj_df = results_df.dropna(subset=['object_accuracy'])
    obj_total_dur = obj_df['total_time'].sum()
    if obj_total_dur > 0:
        avg_obj = (obj_df['object_accuracy'] * obj_df['total_time']).sum() / obj_total_dur
    else:
        avg_obj = 0
    
    cats = [f'State {metric_label}', f'Object {metric_label}']
    vals = [avg_state, avg_obj]
    
    bars = ax.bar(cats, vals, color=['#4CAF50', '#2196F3'], edgecolor='black')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Average {metric_label}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    p2 = os.path.join(output_dir, 'average_accuracy.png')
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(p2)
    
    return chart_paths

def generate_f1_breakdown_chart(f1_data, output_dir, title, filename):
    """Generate bar chart for per-class F1 scores."""
    if not f1_data: return
    
    # Sort by score descending
    sorted_items = sorted(f1_data.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_items]
    scores = [x[1] for x in sorted_items]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, scores, color='#9C27B0', edgecolor='black')
    
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.xticks(rotation=45, ha='right')
    
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.1%}', ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_charts(results_list, output_dir, metric_label="Accuracy"):
    """Generate comparison charts with dynamic label."""
    if not results_list: return []
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    for res in results_list:
        label = res.get('batch_id', 'Unknown')
        if res.get('batch_params'): label = res['batch_params'].config_name
        note = res.get('notes', ''); 
        if note: label = f"{label} ({note})"
        
        data.append({
            'Batch': label,
            f'State {metric_label}': res['avg_state_accuracy'],
            f'Object {metric_label}': res['avg_object_accuracy'] if res['avg_object_accuracy'] is not None else 0
        })
        
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "batch_comparison.csv")
    df.to_csv(csv_path, index=False)
    
    df_melt = df.melt(id_vars=['Batch'], var_name='Metric', value_name='Score')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=df_melt, x='Batch', y='Score', hue='Metric', ax=ax, palette='viridis', edgecolor='black')
    ax.set_title(f'Batch Comparison: {metric_label}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.xticks(rotation=45, ha='right')
    for c in ax.containers:
        labels = [f'{v:.1%}' for v in c.datavalues]
        ax.bar_label(c, labels=labels, padding=3)
    
    path = os.path.join(output_dir, 'batch_comparison.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return [path]

# ... [generate_performance_charts & generate_performance_comparison_charts kept as is] ...
def generate_performance_charts(results_df, output_dir):
    """Generate performance visualization charts."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    if 'processing_time' not in results_df.columns or results_df['processing_time'].isnull().all():
        return []
        
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=results_df, x='video_name', y='processing_time', ax=ax, color='#FF9800', edgecolor='black')
    ax.set_title('Processing Time per Video')
    ax.set_ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    p1 = os.path.join(output_dir, 'performance_time.png')
    plt.savefig(p1, dpi=300)
    plt.close()
def generate_performance_comparison_charts(results_list, output_dir):
    """Generate performance comparison charts."""
    if not results_list: return []
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    for res in results_list:
        if 'per_video_results' in res:
            ratios = [v.get('speed_ratio') for v in res['per_video_results'].values() if v.get('speed_ratio')]
            avg_ratio = sum(ratios)/len(ratios) if ratios else 0
            label = res.get('batch_id', 'Unknown')
            if res.get('batch_params'): label = res['batch_params'].config_name
            data.append({'Batch': label, 'Avg Speed Ratio': avg_ratio})
            
    if not data: return []
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Batch', y='Avg Speed Ratio', hue='Batch', ax=ax, palette='magma', edgecolor='black', legend=False)
    ax.set_title('Batch Comparison: Average Speed Ratio')
    plt.xticks(rotation=45, ha='right')
    for c in ax.containers: ax.bar_label(c, fmt='%.2fx', padding=3)
    
    p = os.path.join(output_dir, 'comparison_performance_speed.png')
    plt.tight_layout()
    plt.savefig(p, dpi=300)
    plt.close()
    return [p]

def generate_over_under_charts(aggregated_over_under, total_duration, output_dir, allowed_tools=None):
    """Generate charts for over/under prediction."""
    os.makedirs(output_dir, exist_ok=True)
    
    def plot(data_over, data_under, data_correct, title, filename, is_prop=False, filters=None):
        keys = sorted(list(set(data_over.keys()) | set(data_under.keys()) | set(data_correct.keys())))
        if filters: keys = [k for k in keys if k in filters]
        if not keys: return
        
        over = [data_over.get(k,0) for k in keys]
        under = [-data_under.get(k,0) for k in keys]
        correct = [data_correct.get(k,0) for k in keys]
        
        if is_prop and total_duration > 0:
            over = [v/total_duration*100 for v in over]
            under = [v/total_duration*100 for v in under]
            correct = [v/total_duration*100 for v in correct]
            ylabel = "Percentage (%)"
        else: ylabel = "Seconds"
        
        fig, ax = plt.subplots(figsize=(14, 7))
        x = range(len(keys))
        w = 0.35
        ax.bar([i-w/2 for i in x], over, width=w, color='#FFC107', label='Overprediction (FP)', edgecolor='black')
        ax.bar([i-w/2 for i in x], under, width=w, color='#F44336', label='Underprediction (FN)', edgecolor='black')
        ax.bar([i+w/2 for i in x], correct, width=w, color='#4CAF50', label='Correct (TP)', edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.axhline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    plot(aggregated_over_under['tool_over'], aggregated_over_under['tool_under'], aggregated_over_under['tool_correct'],
         "Tool Prediction (Seconds)", "tool_over_under_seconds.png", filters=allowed_tools)
    plot(aggregated_over_under['tool_over'], aggregated_over_under['tool_under'], aggregated_over_under['tool_correct'],
         "Tool Prediction (%)", "tool_over_under_percent.png", is_prop=True, filters=allowed_tools)
    plot(aggregated_over_under['state_over'], aggregated_over_under['state_under'], aggregated_over_under['state_correct'],
         "State Prediction (Seconds)", "state_over_under_seconds.png")
    plot(aggregated_over_under['state_over'], aggregated_over_under['state_under'], aggregated_over_under['state_correct'],
         "State Prediction (%)", "state_over_under_percent.png", is_prop=True)

def run_benchmark(videos_to_process, batch_name, model_version, notes="", model_data_dir="outputs/data/", batch_id=None, performance_mode="none", generate_over_under=False, metric_mode="traditional", preset_name="N/A"):
    """
    Run benchmark.
    metric_mode: "traditional" (Accuracy), "macro_f1", "micro_f1", or "macro_f1_gt_only"
    """
    # 1. Setup based on metric mode
    gt_only = False
    
    if metric_mode == "macro_f1":
        metric_name = "Macro-F1"
        RESULTS_DIR = "benchmark_results_f1/" + batch_name
        UNIFIED_RESULTS_FILE_ACTUAL = "benchmark_results_f1/all_results.csv"
        UNIFIED_SUMMARY_FILE_ACTUAL = "benchmark_results_f1/run_summary.csv"
    elif metric_mode == "macro_f1_gt_only":
        metric_name = "Macro-F1 (GT Only)"
        RESULTS_DIR = "benchmark_results_f1_gt_only/" + batch_name
        UNIFIED_RESULTS_FILE_ACTUAL = "benchmark_results_f1_gt_only/all_results.csv"
        UNIFIED_SUMMARY_FILE_ACTUAL = "benchmark_results_f1_gt_only/run_summary.csv"
        gt_only = True
    elif metric_mode == "micro_f1":
        metric_name = "Micro-F1"
        RESULTS_DIR = "benchmark_results_micro_f1/" + batch_name
        UNIFIED_RESULTS_FILE_ACTUAL = "benchmark_results_micro_f1/all_results.csv"
        UNIFIED_SUMMARY_FILE_ACTUAL = "benchmark_results_micro_f1/run_summary.csv"
    else:
        metric_name = "Accuracy"
        RESULTS_DIR = "benchmark_results/" + batch_name
        UNIFIED_RESULTS_FILE_ACTUAL = UNIFIED_RESULTS_FILE
        UNIFIED_SUMMARY_FILE_ACTUAL = UNIFIED_SUMMARY_FILE
        
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load GT
    testdata, gt_video_ids = load_ground_truth()
    videos_with_gt = [v for v in videos_to_process if int(v.split('_')[1]) in gt_video_ids]
    if not videos_with_gt: print("No videos with GT found."); return None

    # Load Params
    batch_params = None
    if batch_id:
        try:
            from video_processing.batch_parameters import BatchParameters
            batch_params = BatchParameters.from_batch_id(batch_id)
        except: pass

    def get_run_id(filepath):
        if os.path.exists(filepath):
            try:
                df = repair_and_load_csv(filepath)
                if 'run_id' in df.columns: return df['run_id'].max() + 1
            except: pass
        return 1
    
    run_id = get_run_id(UNIFIED_RESULTS_FILE_ACTUAL)
    run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    all_results = {}
    results_records = []
    
    # Aggregation for Over/Under
    agg_over_under = {
        'tool_over': defaultdict(float), 'tool_under': defaultdict(float), 'tool_correct': defaultdict(float),
        'state_over': defaultdict(float), 'state_under': defaultdict(float), 'state_correct': defaultdict(float)
    }
    total_agg_dur = 0
    
    # Aggregation for Confusion Matrix
    agg_confusion = {
        'pred_state': [], 'gt_state': [],
        'pred_object': [], 'gt_object': []
    }
    
    print(f"\nRunning Benchmark ({metric_name} Mode)...")
    
    for video_name in videos_with_gt:
        vid_id = int(video_name.split('_')[1])
        try:
            modeldata, duration = load_labeled_data(f"{model_data_dir}{video_name}.csv")
            # Calculate ALL metrics
            metrics = calculate_overlap(modeldata, testdata, vid_id, duration, gt_only=gt_only)
            
            # SWAP LOGIC: Choose which metric acts as the "State Score" AND "Object Score"
            if metric_mode == "macro_f1" or metric_mode == "macro_f1_gt_only":
                primary_state = metrics['macro_f1']
                primary_obj = metrics['object_macro_f1']
            elif metric_mode == "micro_f1":
                primary_state = metrics['micro_f1']
                primary_obj = metrics['object_micro_f1']
            else:
                primary_state = metrics['state_accuracy']
                primary_obj = metrics['object_accuracy_traditional']
            
            # Performance stats
            perf = {}
            try:
                import json
                with open(f"{model_data_dir}{video_name}_metadata.json", 'r') as f:
                    m = json.load(f)
                    if 'performance' in m: perf = m['performance']
            except: pass

            # Store CHOSEN metrics for downstream graphs
            metrics['state_accuracy'] = primary_state 
            metrics['object_accuracy'] = primary_obj
            metrics['speed_ratio'] = perf.get('speed_ratio')
            metrics['processing_time'] = perf.get('total_time_seconds')
            
            all_results[video_name] = metrics
            
            results_records.append({
                'run_id': run_id, 'timestamp': run_timestamp, 'batch_name': batch_name,
                'model_version': model_version, 'batch_id': batch_id if batch_id else 'N/A',
                'video_name': video_name, 'video_id': vid_id,
                'state_accuracy': primary_state, # F1 or Acc
                'object_accuracy': primary_obj,  # F1 or Acc
                'guess_accuracy': metrics['guess_accuracy'],
                'using_tool_time': metrics['using_tool_time'],
                'total_time': metrics['total_time'],
                'metric_type': metric_name,
                'idle_recall': metrics['idle_recall'],
                'processing_time': perf.get('total_time_seconds'),
                'speed_ratio': perf.get('speed_ratio'),
                'api_calls': perf.get('api_calls_total'),
                'notes': notes
            })
            
            # Aggregate Over/Under
            if 'over_under' in metrics:
                for k in ['tool_over', 'tool_under', 'tool_correct', 'state_over', 'state_under', 'state_correct']:
                    for item, val in metrics['over_under'][k].items(): agg_over_under[k][item] += val
            total_agg_dur += duration
            
            # Aggregate Confusion Data (Sampled or Full)
            # calculate_overlap returns timelines? No, it calculates metrics.
            # We need to modify calculate_overlap to return timelines OR we reconstruct them here?
            # Reconstructing is hard. Let's modify calculate_overlap to return raw lists if needed.
            # OR we just extract them from metrics if we add them there.
            # Let's check calculate_overlap return.
            if 'raw_timelines' in metrics:
                agg_confusion['pred_state'].extend(metrics['raw_timelines']['pred_state'])
                agg_confusion['gt_state'].extend(metrics['raw_timelines']['gt_state'])
                agg_confusion['pred_object'].extend(metrics['raw_timelines']['pred_object'])
                agg_confusion['gt_object'].extend(metrics['raw_timelines']['gt_object'])
                
            obj_s = f"{primary_obj:.1%}" if primary_obj is not None else "N/A"
            print(f"✓ {video_name}: State {metric_name}={primary_state:.2%}, Object {metric_name}={obj_s}")
            
        except Exception as e:
            print(f"✗ Error {video_name}: {e}")

    if not all_results: return None

    # Save Results
    results_df = pd.DataFrame(results_records)
    results_df.to_csv(os.path.join(RESULTS_DIR, RESULTS_FILE), index=False)
    
    # Update Unified File
    try:
        if os.path.exists(UNIFIED_RESULTS_FILE_ACTUAL):
            existing = repair_and_load_csv(UNIFIED_RESULTS_FILE_ACTUAL)
            all_cols = list(set(existing.columns) | set(results_df.columns))
            results_df = results_df.reindex(columns=all_cols)
            existing = existing.reindex(columns=all_cols)
            final = pd.concat([existing, results_df], ignore_index=True)
            final.to_csv(UNIFIED_RESULTS_FILE_ACTUAL, index=False)
        else:
            os.makedirs(os.path.dirname(UNIFIED_RESULTS_FILE_ACTUAL), exist_ok=True)
            results_df.to_csv(UNIFIED_RESULTS_FILE_ACTUAL, index=False)
    except Exception as e: print(f"Warning: Unified file update failed: {e}")
    
    # Generate Charts
    charts_dir = os.path.join(RESULTS_DIR, "charts")
    generate_accuracy_charts(results_df, charts_dir, metric_label=metric_name)
    
    if performance_mode == "full":
        generate_performance_charts(results_df, charts_dir)

    # Calculate Global F1 Per Class (Aggregated)
    # Re-calculate F1 using the aggregated over/under counts
    global_state_f1 = {}
    global_object_f1 = {}
    
    # State F1
    valid_states = set(agg_over_under['state_correct'].keys()) | set(agg_over_under['state_over'].keys()) | set(agg_over_under['state_under'].keys())
    valid_states = [s for s in valid_states if s != 'unknown']
    
    # GT-Only Filter for Global Aggregation
    # We need to know which classes were in GT across ALL videos.
    # agg_over_under['state_under'] keys are definitely in GT (False Negatives).
    # agg_over_under['state_correct'] keys are definitely in GT (True Positives).
    # agg_over_under['state_over'] keys MIGHT be in GT (if they were also under/correct elsewhere), or purely hallucinated.
    # To be strictly correct, we should track the set of ALL GT classes encountered.
    # But for now, let's approximate: if a class has ANY 'correct' or 'under' duration, it exists in GT.
    # If it ONLY has 'over' duration, it MIGHT be a pure hallucination (or just overpredicted in a video where it didn't exist, but existed in another).
    # Wait, if we are doing "GT Only", we want to exclude classes that NEVER appeared in GT in the batch?
    # Or exclude them on a per-video basis?
    # The per-video calculation `macro_f1` returned earlier handled it per-video.
    # The global aggregation here is for the "Breakdown Chart".
    # If we want the breakdown chart to match the "GT Only" philosophy, we should probably show ALL classes but maybe highlight which ones were hallucinated?
    # Or just show all. The user cares about the *Average Score* being dragged down.
    # The breakdown chart showing a 0% bar for "drill" is actually useful information.
    # So I will NOT filter the breakdown chart, only the average score calculation (which is already done per-video).
    
    for s in valid_states:
        tp = agg_over_under['state_correct'][s]
        fp = agg_over_under['state_over'][s]
        fn = agg_over_under['state_under'][s]
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*(prec*rec)/(prec+rec) if (prec+rec)>0 else 0
        global_state_f1[s] = f1
        
    # Object F1
    valid_tools = set(agg_over_under['tool_correct'].keys()) | set(agg_over_under['tool_over'].keys()) | set(agg_over_under['tool_under'].keys())
    valid_tools = [t for t in valid_tools if t != 'unknown']
    for t in valid_tools:
        tp = agg_over_under['tool_correct'][t]
        fp = agg_over_under['tool_over'][t]
        fn = agg_over_under['tool_under'][t]
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*(prec*rec)/(prec+rec) if (prec+rec)>0 else 0
        global_object_f1[t] = f1
        
    # Generate F1 Breakdown Charts
    generate_f1_breakdown_chart(global_state_f1, charts_dir, f"State F1 Breakdown ({metric_name})", "f1_breakdown_state.png")
    generate_f1_breakdown_chart(global_object_f1, charts_dir, f"Object F1 Breakdown ({metric_name})", "f1_breakdown_object.png")
    
    # Generate Confusion Matrices
    generate_confusion_matrix(agg_confusion['gt_state'], agg_confusion['pred_state'], 
                              charts_dir, "State Confusion Matrix (Recall Normalized)", "confusion_matrix_state.png")
    generate_confusion_matrix(agg_confusion['gt_object'], agg_confusion['pred_object'], 
                              charts_dir, "Object Confusion Matrix (Recall Normalized)", "confusion_matrix_object.png")

    if generate_over_under:
        allowed = [t.lower() for t in batch_params.allowed_tools] if batch_params else None
        generate_over_under_charts(agg_over_under, total_agg_dur, charts_dir, allowed_tools=allowed)
    
    # Calculate Weighted Averages for Summary
    total_dur = results_df['total_time'].sum()
    weighted_avg_state = (results_df['state_accuracy'] * results_df['total_time']).sum() / total_dur if total_dur > 0 else 0
    
    obj_df = results_df.dropna(subset=['object_accuracy'])
    obj_total_dur = obj_df['total_time'].sum()
    weighted_avg_obj = (obj_df['object_accuracy'] * obj_df['total_time']).sum() / obj_total_dur if obj_total_dur > 0 else 0

    # Weighted Speed Ratio
    speed_df = results_df.dropna(subset=['speed_ratio'])
    speed_total_dur = speed_df['total_time'].sum()
    weighted_avg_speed = (speed_df['speed_ratio'] * speed_df['total_time']).sum() / speed_total_dur if speed_total_dur > 0 else 0

    # Readme
    with open(os.path.join(RESULTS_DIR, "readme.txt"), 'w') as f:
        f.write(f"BENCHMARK RESULTS ({metric_name.upper()})\n")
        f.write("="*30 + "\n")
        f.write(f"Batch Name: {batch_name}\n")
        f.write(f"Model Version: {model_version}\n")
        f.write(f"Preset: {preset_name}\n")
        if notes: f.write(f"Notes: {notes}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Metric Used: {metric_name}\n")
        f.write(f"Average State Score (Weighted): {weighted_avg_state:.2%}\n")
        f.write(f"Average Object Score (Weighted): {weighted_avg_obj:.2%}\n")
        if metric_mode == "macro_f1":
            f.write("Note: Scores refer to Macro-F1 (State) and Global Macro-F1 (Object).\n")
        elif metric_mode == "macro_f1_gt_only":
            f.write("Note: Scores refer to Macro-F1 (GT Classes Only). Hallucinations of non-existent classes are ignored in the average.\n")
        elif metric_mode == "micro_f1":
            f.write("Note: Scores refer to Micro-F1 (Global Average of TP/FP/FN).\n")
        f.write(f"Videos Processed: {len(videos_with_gt)}\n\n")
        
        f.write("PER-CLASS PERFORMANCE (Global F1)\n")
        f.write("-" * 30 + "\n")
        f.write("States:\n")
        for s, score in sorted(global_state_f1.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {s}: {score:.1%}\n")
        f.write("\nObjects:\n")
        for t, score in sorted(global_object_f1.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {t}: {score:.1%}\n")

    print(f"\n✓ Saved {metric_name} results to: {RESULTS_DIR}")
    
    return {
        'avg_state_accuracy': weighted_avg_state,
        'avg_object_accuracy': weighted_avg_obj,
        'avg_speed_ratio': weighted_avg_speed,
        'total_duration': total_dur,
        'batch_id': batch_id,
        'batch_params': batch_params,
        'metric_label': metric_name,
        'per_video_results': all_results,
        'per_video_results': all_results,
        'notes': notes,
        'agg_over_under': agg_over_under, # Return for grouped batch
        'total_agg_dur': total_agg_dur,
        'agg_confusion': agg_confusion
    }

def generate_accuracy_charts(results_df, output_dir, metric_label="Accuracy"):
    """Generate charts with dynamic label."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    chart_paths = []
    
    # 1. Per-Video
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = range(len(results_df))
    # 'state_accuracy' column holds Swapped Score
    ax.bar([i-0.2 for i in x_pos], results_df['state_accuracy'], width=0.4, label=f'State {metric_label}', color='#4CAF50', edgecolor='black')
    
    # 'object_accuracy' column holds Swapped Score
    obj_acc = results_df['object_accuracy'].fillna(0)
    # Update Label to match metric
    ax.bar([i+0.2 for i in x_pos], obj_acc, width=0.4, label=f'Object {metric_label}', color='#2196F3', edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_label} by Video', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['video_name'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    p1 = os.path.join(output_dir, 'accuracy_by_video.png')
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(p1)
    
    # 2. Average (Weighted by Duration)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Weighted State Average
    total_dur = results_df['total_time'].sum()
    if total_dur > 0:
        avg_state = (results_df['state_accuracy'] * results_df['total_time']).sum() / total_dur
    else:
        avg_state = 0
        
    # Weighted Object Average
    obj_df = results_df.dropna(subset=['object_accuracy'])
    obj_total_dur = obj_df['total_time'].sum()
    if obj_total_dur > 0:
        avg_obj = (obj_df['object_accuracy'] * obj_df['total_time']).sum() / obj_total_dur
    else:
        avg_obj = 0
    
    cats = [f'State {metric_label}', f'Object {metric_label}']
    vals = [avg_state, avg_obj]
    
    bars = ax.bar(cats, vals, color=['#4CAF50', '#2196F3'], edgecolor='black')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Average {metric_label}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    p2 = os.path.join(output_dir, 'average_accuracy.png')
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(p2)
    
    return chart_paths

def generate_f1_breakdown_chart(f1_data, output_dir, title, filename):
    """Generate bar chart for per-class F1 scores."""
    if not f1_data: return
    
    # Sort by score descending
    sorted_items = sorted(f1_data.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_items]
    scores = [x[1] for x in sorted_items]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, scores, color='#9C27B0', edgecolor='black')
    
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.xticks(rotation=45, ha='right')
    
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.1%}', ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_charts(results_list, output_dir, metric_label="Accuracy"):
    """Generate comparison charts with dynamic label."""
    if not results_list: return []
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    for res in results_list:
        label = res.get('batch_id', 'Unknown')
        if res.get('batch_params'): label = res['batch_params'].config_name
        note = res.get('notes', ''); 
        if note: label = f"{label} ({note})"
        
        data.append({
            'Batch': label,
            f'State {metric_label}': res['avg_state_accuracy'],
            f'Object {metric_label}': res['avg_object_accuracy'] if res['avg_object_accuracy'] is not None else 0
        })
        
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "batch_comparison.csv")
    df.to_csv(csv_path, index=False)
    
    df_melt = df.melt(id_vars=['Batch'], var_name='Metric', value_name='Score')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=df_melt, x='Batch', y='Score', hue='Metric', ax=ax, palette='viridis', edgecolor='black')
    ax.set_title(f'Batch Comparison: {metric_label}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.xticks(rotation=45, ha='right')
    for c in ax.containers:
        labels = [f'{v:.1%}' for v in c.datavalues]
        ax.bar_label(c, labels=labels, padding=3)
    
    path = os.path.join(output_dir, 'batch_comparison.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return [path]

# ... [generate_performance_charts & generate_performance_comparison_charts kept as is] ...
def generate_performance_charts(results_df, output_dir):
    """Generate performance visualization charts."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    if 'processing_time' not in results_df.columns or results_df['processing_time'].isnull().all():
        return []
        
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=results_df, x='video_name', y='processing_time', ax=ax, color='#FF9800', edgecolor='black')
    ax.set_title('Processing Time per Video')
    ax.set_ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    p1 = os.path.join(output_dir, 'performance_time.png')
    plt.savefig(p1, dpi=300)
    plt.close()
    
    if 'speed_ratio' in results_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=results_df, x='video_name', y='speed_ratio', ax=ax, color='#9C27B0', edgecolor='black')
        ax.set_title('Speed Ratio (Lower is Faster)')
        ax.axhline(1.0, color='red', linestyle='--')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        p2 = os.path.join(output_dir, 'performance_speed.png')
        plt.savefig(p2, dpi=300)
        plt.close()
        return [p1, p2]
    return [p1]

def generate_performance_comparison_charts(results_list, output_dir):
    """Generate performance comparison charts."""
    if not results_list: return []
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    for res in results_list:
        if 'per_video_results' in res:
            ratios = [v.get('speed_ratio') for v in res['per_video_results'].values() if v.get('speed_ratio')]
            avg_ratio = sum(ratios)/len(ratios) if ratios else 0
            label = res.get('batch_id', 'Unknown')
            if res.get('batch_params'): label = res['batch_params'].config_name
            data.append({'Batch': label, 'Avg Speed Ratio': avg_ratio})
            
    if not data: return []
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Batch', y='Avg Speed Ratio', hue='Batch', ax=ax, palette='magma', edgecolor='black', legend=False)
    ax.set_title('Batch Comparison: Average Speed Ratio')
    plt.xticks(rotation=45, ha='right')
    for c in ax.containers: ax.bar_label(c, fmt='%.2fx', padding=3)
    
    p = os.path.join(output_dir, 'comparison_performance_speed.png')
    plt.tight_layout()
    plt.savefig(p, dpi=300)
    plt.close()
    return [p]

def generate_over_under_charts(aggregated_over_under, total_duration, output_dir, allowed_tools=None):
    """Generate charts for over/under prediction."""
    os.makedirs(output_dir, exist_ok=True)
    
    def plot(data_over, data_under, data_correct, title, filename, is_prop=False, filters=None):
        keys = sorted(list(set(data_over.keys()) | set(data_under.keys()) | set(data_correct.keys())))
        if filters: keys = [k for k in keys if k in filters]
        if not keys: return
        
        over = [data_over.get(k,0) for k in keys]
        under = [-data_under.get(k,0) for k in keys]
        correct = [data_correct.get(k,0) for k in keys]
        
        if is_prop and total_duration > 0:
            over = [v/total_duration*100 for v in over]
            under = [v/total_duration*100 for v in under]
            correct = [v/total_duration*100 for v in correct]
            ylabel = "Percentage (%)"
        else: ylabel = "Seconds"
        
        fig, ax = plt.subplots(figsize=(14, 7))
        x = range(len(keys))
        w = 0.35
        ax.bar([i-w/2 for i in x], over, width=w, color='#FFC107', label='Overprediction (FP)', edgecolor='black')
        ax.bar([i-w/2 for i in x], under, width=w, color='#F44336', label='Underprediction (FN)', edgecolor='black')
        ax.bar([i+w/2 for i in x], correct, width=w, color='#4CAF50', label='Correct (TP)', edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.axhline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    plot(aggregated_over_under['tool_over'], aggregated_over_under['tool_under'], aggregated_over_under['tool_correct'],
         "Tool Prediction (Seconds)", "tool_over_under_seconds.png", filters=allowed_tools)
    plot(aggregated_over_under['tool_over'], aggregated_over_under['tool_under'], aggregated_over_under['tool_correct'],
         "Tool Prediction (%)", "tool_over_under_percent.png", is_prop=True, filters=allowed_tools)
    plot(aggregated_over_under['state_over'], aggregated_over_under['state_under'], aggregated_over_under['state_correct'],
         "State Prediction (Seconds)", "state_over_under_seconds.png")
    plot(aggregated_over_under['state_over'], aggregated_over_under['state_under'], aggregated_over_under['state_correct'],
         "State Prediction (%)", "state_over_under_percent.png", is_prop=True)

def generate_per_video_accuracy_comparison_charts(results_list, output_dir, metric_label="Accuracy"):
    """Generate per-video comparison charts for State and Object metrics."""
    if not results_list: return []
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data
    state_data = []
    object_data = []
    
    for res in results_list:
        label = res.get('batch_id', 'Unknown')
        if res.get('batch_params'): label = res['batch_params'].config_name
        note = res.get('notes', ''); 
        if note: label = f"{label} ({note})"
        
        if 'per_video_results' in res:
            for vid_id, metrics in res['per_video_results'].items():
                s_val = 0; o_val = 0
                if "GT Only" in metric_label:
                    s_val = metrics.get('macro_f1', 0)
                    o_val = metrics.get('object_macro_f1', 0)
                elif "Macro-F1" in metric_label:
                    s_val = metrics.get('macro_f1', 0)
                    o_val = metrics.get('object_macro_f1', 0)
                elif "Micro-F1" in metric_label:
                    s_val = metrics.get('micro_f1', 0)
                    o_val = metrics.get('object_micro_f1', 0)
                else:
                    s_val = metrics.get('state_accuracy', 0)
                    o_val = metrics.get('object_accuracy', 0)
                
                state_data.append({'Batch': label, 'Video': vid_id, 'Score': s_val})
                object_data.append({'Batch': label, 'Video': vid_id, 'Score': o_val})

    if not state_data: return []

    # Plot State
    df_state = pd.DataFrame(state_data)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=df_state, x='Video', y='Score', hue='Batch', ax=ax, palette='viridis', edgecolor='black')
    ax.set_title(f'Per-Video State {metric_label} Comparison', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Batch')
    plt.tight_layout()
    p1 = os.path.join(output_dir, 'per_video_state_comparison.png')
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Object
    df_obj = pd.DataFrame(object_data)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=df_obj, x='Video', y='Score', hue='Batch', ax=ax, palette='viridis', edgecolor='black')
    ax.set_title(f'Per-Video Object {metric_label} Comparison', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Batch')
    plt.tight_layout()
    p2 = os.path.join(output_dir, 'per_video_object_comparison.png')
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()
    
    return [p1, p2]

def generate_confusion_matrix(y_true, y_pred, output_dir, title, filename):
    """Generate confusion matrix heatmap."""
    if not y_true or not y_pred: return
    
    try:
        from sklearn.metrics import confusion_matrix
        import numpy as np
    except ImportError:
        print("sklearn or numpy not found, skipping confusion matrix.")
        return

    # Filter labels to only those present in the data
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    # Remove empty/unknown if desired, but usually good to keep for completeness
    # unique_labels = [l for l in unique_labels if l != 'unknown']
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Normalize by row (True Label) to get Recall/Sensitivity
    # Add epsilon to avoid division by zero
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Plot
    # Adjust figure size based on number of labels
    size = max(10, len(unique_labels) * 0.8)
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels, ax=ax)
    
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
