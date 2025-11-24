# final version


import os
import base64
from anthropic import Anthropic
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from difflib import get_close_matches
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from threading import local
import colorsys
import json
import matplotlib

matplotlib.use('Agg')  #non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


### config ###
video_directory = "testing_vids/"
keyframes_directory = "keyframes/keyframes"
video_output_directory = "outputs/vid_objs"
csv_directory = "outputs/data/"
analysis_output_directory = "outputs/analysis/"
video_default = "video_01"

#post-processing LLM analysis
ENABLE_PRODUCTIVITY_ANALYSIS = True

#setup
from config import ANTHROPIC_API_KEY
client = Anthropic(api_key=ANTHROPIC_API_KEY)
_thread_local = local()
confidence_threshold = 0.60 #accept above 60% confidence

#relationship tracking config
proximity_threshold_percent = 0.18  # 20% of frame width

def get_local_model():
    if not hasattr(_thread_local, "model"):
        _thread_local.model = YOLO("weights.pt")
    return _thread_local.model


#color management for objs
def get_color_for_class(class_name, all_class_names):
    if not hasattr(get_color_for_class, 'color_map'):
        get_color_for_class.color_map = {}
    if class_name not in get_color_for_class.color_map:
        #generate colors w/ hsv
        num_classes = len(all_class_names)
        class_idx = list(all_class_names).index(class_name) if class_name in all_class_names else len(
            get_color_for_class.color_map)
        hue = (class_idx * 0.618033988749895) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        get_color_for_class.color_map[class_name] = tuple(int(c * 255) for c in rgb)

    return get_color_for_class.color_map[class_name]


#=================================#
#relationship tracking
class RelationshipTracker:
    def __init__(self, fps, proximity_threshold_percent):
        self.fps = fps
        self.proximity_threshold_percent = proximity_threshold_percent

        #active relationships: key = frozenset of object names, value = start_frame
        self.active_relationships = {}

        #completed relationships: list of {objects: set, start_frame, end_frame}
        self.completed_relationships = []

    def get_box_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_distance(self, center1, center2):
        #distance between two centers
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    def find_relationships(self, detections, frame_width, current_frame):
        proximity_threshold = frame_width * self.proximity_threshold_percent
        if len(detections) < 2:
            return [], []
        #calc centers for all detections
        centers = [(self.get_box_center(det[0]), det[1]) for det in detections]
        #adjacency: which objects are close to each other
        n = len(centers)
        adjacency = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.get_distance(centers[i][0], centers[j][0])
                if dist <= proximity_threshold:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        #find connected components/objs
        visited = [False] * n
        relationships = []
        def dfs(node, component):
            visited[node] = True
            component.add(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    dfs(neighbor, component)

        for i in range(n):
            if not visited[i] and adjacency[i]:  #at least 1 close neighbor
                component = set()
                dfs(i, component)
                if len(component) >= 2:
                    relationships.append(component)

        #convert indices to obj names + get line drawing info
        relationship_sets = []
        line_info = []
        for component in relationships:
            obj_names = frozenset(centers[idx][1] for idx in component)
            relationship_sets.append(obj_names)

            indices = list(component)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    center1 = centers[indices[i]][0]
                    center2 = centers[indices[j]][0]
                    line_info.append((center1, center2))
        return relationship_sets, line_info

    def update(self, relationships, current_frame):
        #upd relationship tracking
        current_relationship_keys = set(relationships)

        #start new relationships or continue existing ones
        for rel in relationships:
            if rel not in self.active_relationships:
                #new relationship detected - start tracking it
                self.active_relationships[rel] = current_frame
                print(f"Frame {current_frame}: Started tracking relationship: {set(rel)}")

        #check for ended relationships
        ended_keys = []
        for rel_key, start_frame in self.active_relationships.items():
            if rel_key not in current_relationship_keys:
                #relationship ended - record immediately
                self.completed_relationships.append({
                    'objects': rel_key,
                    'start_frame': start_frame,
                    'end_frame': current_frame - 1  #fast frame active
                })
                ended_keys.append(rel_key)
                print(
                    f"Frame {current_frame}: Ended relationship: {set(rel_key)} (lasted {current_frame - start_frame} frames)")
        for key in ended_keys:
            del self.active_relationships[key]

    def finalize(self, last_frame):
        for rel_key, start_frame in self.active_relationships.items():
            self.completed_relationships.append({
                'objects': rel_key,
                'start_frame': start_frame,
                'end_frame': last_frame
            })
            print(f"Finalized relationship: {set(rel_key)} (lasted {last_frame - start_frame + 1} frames)")
        self.active_relationships.clear()

    def get_relationships_csv_data(self):
        #interaction data into csv format
        return self.completed_relationships


def draw_relationship_lines(frame, line_info):
    for center1, center2 in line_info:
        pt1 = (int(center1[0]), int(center1[1]))
        pt2 = (int(center2[0]), int(center2[1]))
        cv2.line(frame, pt1, pt2, (255, 255, 0), 5)  # Yellow lines
        #circles at centers
        cv2.circle(frame, pt1, 10, (255, 255, 0), -1)
        cv2.circle(frame, pt2, 10, (255, 255, 0), -1)
    return frame

#user-defined actions
def set_actions(action_list):
    global ALLOWED_ACTIONS, ACTION_PROMPT
    ALLOWED_ACTIONS = [a.lower() for a in action_list]
    actions_joined = ", ".join(ALLOWED_ACTIONS)
    ACTION_PROMPT = (
        f"You must classify the person's behavior into ONE of the following categories: "
        f"{actions_joined}. Respond ONLY with one word from this list, no explanations."
    )
set_actions(["using tool", "idle", "moving"])


#=============================#
#encode frames for claude
def encode_frames(frames, width_small=640, height_small=360):
    encoded_list = []
    for frame in frames:
        frame_small = cv2.resize(frame, (width_small, height_small))
        _, buffer_img = cv2.imencode('.jpg', frame_small)
        encoded_list.append(base64.b64encode(buffer_img.tobytes()).decode("utf-8"))
    return encoded_list

def get_objects(frame):
    model = get_local_model()
    results = model(frame, conf=confidence_threshold)
    output = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            object_name = model.names[class_id]
            confidence = float(box.conf[0])
            output.append((object_name, confidence))
    return output

def ask_claude_multiframe(frames, prompt_text):
    content_list = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded}
        }
        for encoded in encode_frames(frames)
    ]
    content_list.append({"type": "text", "text": prompt_text})
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": content_list}]
    )
    text = response.content[0].text.strip().lower()
    match = get_close_matches(text, ALLOWED_ACTIONS, n=1)
    if match:
        return match[0]
    for action in ALLOWED_ACTIONS:
        if action in text:
            return action
    return ALLOWED_ACTIONS[0]

#==============#
#video helpers

frame_cache = {}
def get_frame(frame_number):
    #get frame from cache
    if frame_number in frame_cache:
        return frame_cache[frame_number]
    #frame not in cache - shouldn't happen with pre-loaded frames
    raise ValueError(f"Frame {frame_number} not in cache")

def sample_interval_frames(start, end, num_frames=5):
    return [get_frame(int(start + i * (end - start) / (num_frames - 1))) for i in range(num_frames)]

def motion_score(frames, ignore_threshold=5):
    #compute motion scores
    if len(frames) < 2:
        return 0.0
    total_motion = 0.0
    for i in range(1, len(frames)):
        gray_prev = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_curr, gray_prev)

        # ignore tiny changes
        diff[diff < ignore_threshold] = 0
        total_motion += np.mean(diff) / 255.0  # normalize 0-1

    # normalize by number of frame differences
    motion_score_value = total_motion / (len(frames) - 1)
    return motion_score_value

#txt overlay
try:
    font = ImageFont.truetype("arial.ttf", 36)
except:
    font = ImageFont.load_default()

def overlay_action(frame, label):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text((30, 50), label, font=font, fill=(255, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_boxes_on_frame(frame, results):
    #yolo result
    model = get_local_model()
    all_class_names = set(model.names.values())
    for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        class_name = results.names[int(cls_id)]
        label = f"{class_name} {conf:.2f}"
        color = get_color_for_class(class_name, all_class_names)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# Productivity Analysis - Static Images
#=======================================#
def generate_productivity_analysis_images(video_name, actions_csv_path, relationships_csv_path):
    print("GENERATING PRODUCTIVITY ANALYSIS")
    print("=" * 60)

    #make output directory exist
    os.makedirs(analysis_output_directory, exist_ok=True)
    #read CSV
    with open(actions_csv_path, 'r') as f:
        actions_data = f.read()
    with open(relationships_csv_path, 'r') as f:
        relationships_data = f.read()

    #claude prompt
    analysis_prompt = f"""You are analyzing worker productivity data from a video analysis system. I'm providing you with two CSV files:

1. ACTIONS CSV (shows what the worker was doing over time):
```
{actions_data}
```
The first line contains: fps,total_duration
Subsequent lines show: frame_number,action_label
Actions include: using tool, idle, moving

2. RELATIONSHIPS CSV (shows object interactions):
```
{relationships_data}
```
Format: start_frame,end_frame,start_time,end_time,duration,objects
This shows when specific objects were in proximity (e.g., hand + tape measure + pencil indicates measuring activity)

TASK:
Analyze the worker's productivity and provide structured data for creating visualizations.

Your response must be ONLY valid JSON (no markdown, no explanations) with this structure:
{{
  "summary": {{
    "total_duration_seconds": <number>,
    "productive_time_seconds": <number>,
    "idle_time_seconds": <number>,
    "moving_time_seconds": <number>,
    "productivity_percentage": <number>
  }},
  "time_breakdown": {{
    "using tool": <seconds>,
    "idle": <seconds>,
    "moving": <seconds>
  }},
  "inferred_tasks": [
    {{
      "task_name": "string (e.g., 'Measuring', 'Marking', 'Assembly')",
      "duration_seconds": <number>,
      "objects_involved": ["obj1", "obj2"],
      "is_productive": <boolean>
    }}
  ],
  "timeline_data": [
    {{
      "time_seconds": <number>,
      "action": "string",
      "is_productive": <boolean>
    }}
  ],
  "insights": [
    "string: key insight 1",
    "string: key insight 2",
    "string: key insight 3"
  ]
}}

Infer productivity based on:
- "using tool" is generally productive
- "idle" is unproductive
- "moving" can be productive if associated with tool/object interactions
- Specific object combinations indicate concrete tasks (e.g., hand+tape measure = measuring)"""

    # Call Claude API
    print("Sending data to Claude for analysis...")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": analysis_prompt
        }]
    )

    json_text = response.content[0].text.strip()
    # clean up JSON
    if json_text.startswith("```json"):
        json_text = json_text.replace("```json", "", 1)
    if json_text.startswith("```"):
        json_text = json_text.replace("```", "", 1)
    if json_text.endswith("```"):
        json_text = json_text.rsplit("```", 1)[0]
    json_text = json_text.strip()

    try:
        analysis_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Received text: {json_text[:500]}")
        raise

    print("✓ Analysis data received, generating charts...")

    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    #output paths
    output_prefix = os.path.join(analysis_output_directory, video_name)

    # chart 1: pie chart of time breakdown
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    time_breakdown = analysis_data['time_breakdown']
    labels = list(time_breakdown.keys())
    sizes = list(time_breakdown.values())
    colors = ['#4CAF50', '#FF5252', '#FFC107']  # Green, Red, Yellow
    explode = (0.05, 0.05, 0.05)

    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors, textprops={'fontsize': 12})
    ax1.set_title(f'Time Breakdown - {video_name}', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    chart1_path = f"{output_prefix}_time_breakdown.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {chart1_path}")

    #chart 2: productivity summary bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    summary = analysis_data['summary']
    categories = ['Productive Time', 'Idle Time', 'Moving Time']
    values = [
        summary['productive_time_seconds'],
        summary['idle_time_seconds'],
        summary['moving_time_seconds']
    ]
    colors_bar = ['#4CAF50', '#FF5252', '#FFC107']

    bars = ax2.barh(categories, values, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Productivity Summary - {video_name}\n' +
                  f'Overall Productivity: {summary["productivity_percentage"]:.1f}%',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(axis='x', alpha=0.3)

    #add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax2.text(value, bar.get_y() + bar.get_height() / 2,
                 f' {value:.1f}s', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    chart2_path = f"{output_prefix}_productivity_summary.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {chart2_path}")

    #chart 3: timeline visualization
    if 'timeline_data' in analysis_data and analysis_data['timeline_data']:
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        timeline = analysis_data['timeline_data']

        times = [point['time_seconds'] for point in timeline]
        actions = [point['action'] for point in timeline]
        productive = [point['is_productive'] for point in timeline]

        #create color mapping
        action_colors = {
            'using tool': '#4CAF50',
            'idle': '#FF5252',
            'moving': '#FFC107'
        }
        colors_timeline = [action_colors.get(action, '#999999') for action in actions]

        #plot timeline
        for i in range(len(times) - 1):
            ax3.add_patch(Rectangle((times[i], 0), times[i + 1] - times[i], 1,
                                    color=colors_timeline[i], alpha=0.7))

        ax3.set_xlim(0, summary['total_duration_seconds'])
        ax3.set_ylim(0, 1)
        ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax3.set_title(f'Activity Timeline - {video_name}', fontsize=16, fontweight='bold', pad=20)
        ax3.set_yticks([])
        ax3.grid(axis='x', alpha=0.3)

        #add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4CAF50', label='Using Tool'),
            Patch(facecolor='#FF5252', label='Idle'),
            Patch(facecolor='#FFC107', label='Moving')
        ]
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
        chart3_path = f"{output_prefix}_timeline.png"
        plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {chart3_path}")

    #chart 4: inferred tasks breakdown
    if 'inferred_tasks' in analysis_data and analysis_data['inferred_tasks']:
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        tasks = analysis_data['inferred_tasks']

        task_names = [task['task_name'] for task in tasks]
        durations = [task['duration_seconds'] for task in tasks]
        is_productive = [task['is_productive'] for task in tasks]

        colors_tasks = ['#4CAF50' if prod else '#FF5252' for prod in is_productive]

        bars = ax4.barh(task_names, durations, color=colors_tasks, edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Duration (seconds)', fontsize=12, fontweight='bold')
        ax4.set_title(f'Inferred Tasks - {video_name}', fontsize=16, fontweight='bold', pad=20)
        ax4.grid(axis='x', alpha=0.3)

        #value labels
        for bar, duration in zip(bars, durations):
            ax4.text(duration, bar.get_y() + bar.get_height() / 2,
                     f' {duration:.1f}s', va='center', fontsize=10, fontweight='bold')

        #legend
        legend_elements = [
            Patch(facecolor='#4CAF50', label='Productive'),
            Patch(facecolor='#FF5252', label='Non-productive')
        ]
        ax4.legend(handles=legend_elements, loc='lower right', fontsize=10)

        plt.tight_layout()
        chart4_path = f"{output_prefix}_tasks.png"
        plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {chart4_path}")

    #insights to text file
    insights_path = f"{output_prefix}_insights.txt"
    with open(insights_path, 'w') as f:
        f.write(f"PRODUCTIVITY ANALYSIS - {video_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Duration: {summary['total_duration_seconds']:.2f} seconds\n")
        f.write(f"Productivity Rate: {summary['productivity_percentage']:.1f}%\n\n")
        f.write("KEY INSIGHTS:\n")
        f.write("-" * 60 + "\n")
        for i, insight in enumerate(analysis_data['insights'], 1):
            f.write(f"{i}. {insight}\n")
    print(f"✓ Saved: {insights_path}")

    print("=" * 60)
    print(f"✓ Analysis complete! Generated charts:")
    print(f"  - {chart1_path}")
    print(f"  - {chart2_path}")
    if 'timeline_data' in analysis_data and analysis_data['timeline_data']:
        print(f"  - {chart3_path}")
    if 'inferred_tasks' in analysis_data and analysis_data['inferred_tasks']:
        print(f"  - {chart4_path}")
    print(f"  - {insights_path}")
    print("=" * 60 + "\n")

    return analysis_data



# main processing function
def process_video(video_name):
    # Clear frame cache for new video
    global frame_cache
    frame_cache = {}
    video_path = video_directory + video_name + ".mp4"
    keyframes_folder = keyframes_directory + video_name
    keyframe_files = sorted(os.listdir(keyframes_folder), key=lambda x: int(x.rstrip('.jpg')))

    #pre-load all frames into cache
    cap = cv2.VideoCapture(video_path)
    frame_number = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_cache[frame_number] = frame
        frame_number += 1
    cap.release()

    frame_count = frame_number - 1
    frame_labels = [""] * frame_count

    # video properties
    cap_temp = cv2.VideoCapture(video_path)
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_temp.release()

    #initialize relationship tracker
    relationship_tracker = RelationshipTracker(
        fps,
        proximity_threshold_percent
    )


    #process keyframes (multithreaded)

    def process_keyframe(kf_file):
        kf_number = int(kf_file.rstrip(".jpg"))
        frame = get_frame(kf_number)
        prompt = f"This is a POV of a person. Check if hands are visible. {ACTION_PROMPT}"
        label = ask_claude_multiframe([frame], prompt)
        print(f"Keyframe {kf_number}: {label}")
        return kf_number, label

    with ThreadPoolExecutor(max_workers=8) as executor:
        keyframe_results = list(executor.map(process_keyframe, keyframe_files))

    keyframe_results.sort(key=lambda x: x[0])
    keyframe_numbers = []
    for kf_number, label in keyframe_results:
        frame_labels[kf_number - 1] = label
        keyframe_numbers.append(kf_number)

    #fill intervals with multi-frame + motion hints (multithreaded)
    def process_interval(idx):
        start = keyframe_numbers[idx]
        end = keyframe_numbers[idx + 1]
        frames_to_send = sample_interval_frames(start, end, num_frames=5)

        #include neighbor frames for context
        if start > 1:
            frames_to_send.insert(0, get_frame(max(1, start - 1)))
        if end < frame_count:
            frames_to_send.append(get_frame(min(frame_count, end + 1)))

        motion = motion_score(frames_to_send)

        #get objects from frames
        objects = []
        for frame in frames_to_send:
            all_objects = get_objects(frame)
            for obj in all_objects:
                seen = False
                for i in range(len(objects)):
                    if objects[i][0] == obj[0]:
                        seen = True
                        if obj[1] > objects[i][1]:
                            objects[i][1] = obj[1]
                if not seen:
                    objects.append([obj[0], obj[1]])
        print(motion)

        if objects:
            objects_str = ', '.join([str(obj) for obj in objects[0:2]]).replace("'", "")
            found = f"Found objects with their respective confidence: {objects_str}. "
        else:
            found = ""
        prompt = f"This is a POV of a person. Motion score: {motion:.2f}, a motion score of 0 to 0.16 suggests the person is idle, 0.16 or above suggests they are moving. " + found + "Check if hands are visible and classify behavior. {ACTION_PROMPT}"
        label = ask_claude_multiframe(frames_to_send, prompt)
        return start, end, label

    with ThreadPoolExecutor(max_workers=4) as executor:
        interval_results = list(executor.map(process_interval, range(len(keyframe_numbers) - 1)))
    for start, end, label in interval_results:
        for f in range(start, end):
            frame_labels[f] = label

    #fill after last keyframe
    last_kf = keyframe_numbers[-1]
    for f in range(last_kf, frame_count):
        frame_labels[f] = frame_labels[last_kf - 1]

    #temporal smoothing
    window = 9
    smoothed_labels = frame_labels.copy()
    for i in range(frame_count):
        start = max(0, i - window)
        end = min(frame_count, i + window)
        counts = {a: 0 for a in ALLOWED_ACTIONS}
        for j in range(start, end):
            label = frame_labels[j]
            if label in counts.keys():
                counts[label] += 1
        print(counts)
        if sum(counts.values()) == 0:
            smoothed_labels[i] = ALLOWED_ACTIONS[0]
        else:
            smoothed_labels[i] = max(counts, key=counts.get)
    frame_labels = smoothed_labels

    #save labeled video with relationship tracking
    cap_playback = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = video_output_directory + video_name + ".mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    model = get_local_model()
    latest_boxes = None
    latest_detections = []
    tool = None

    for i in range(frame_count):
        ret, frame = cap_playback.read()
        if not ret:
            break

        #run YOLO every 5 frames
        if i % 5 == 0:
            results = model(frame, conf=confidence_threshold)[0]
            latest_boxes = results

            #extract detections with boxes for relationship tracking
            latest_detections = []
            for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                class_name = results.names[int(cls_id)]
                latest_detections.append((box.cpu().numpy(), class_name, float(conf)))
            objects = get_objects(frame)
            tool = None
            for o in objects:
                if o[0] not in ["safety vest", "person", "hardhat", "hand"]:
                    tool = o[0]
                    break

        #track relationships - every frame
        if latest_detections:
            relationships, line_info = relationship_tracker.find_relationships(
                latest_detections, width, i + 1
            )
            relationship_tracker.update(relationships, i + 1)
        else:
            #no detectionsn- end any active relationships
            relationship_tracker.update([], i + 1)
            line_info = []

        #draw latest boxes
        frame_with_boxes = frame.copy()
        if latest_boxes is not None:
            frame_with_boxes = draw_boxes_on_frame(frame_with_boxes, latest_boxes)

        #relationship lines
        if line_info:
            frame_with_boxes = draw_relationship_lines(frame_with_boxes, line_info)

        #overlay action label
        label = frame_labels[i]
        if label:
            if label.lower() == "using tool":
                label = "using an unknown tool"
                if tool:
                    label = f"using {tool}"
            frame_with_boxes = overlay_action(frame_with_boxes, label)
        out.write(frame_with_boxes)

    cap_playback.release()
    out.release()
    print(f"Saved labeled video to {out_path}")

    relationship_tracker.finalize(frame_count)

    #save CSV w/ state changes
    csv_path = csv_directory + video_name + ".csv"
    total_duration = frame_count / fps
    with open(csv_path, 'w') as f:
        f.write(f"{fps},{total_duration}\n")

        if frame_count > 0:
            current_label = frame_labels[0]
            f.write(f"1,{current_label}\n")

            for i in range(1, frame_count):
                if frame_labels[i] != current_label:
                    current_label = frame_labels[i]
                    f.write(f"{i + 1},{current_label}\n")

    print(f"Saved labels to {csv_path}")

    #save relationships CSV
    relationships_csv_path = csv_directory + video_name + "_relationships.csv"
    relationships_data = relationship_tracker.get_relationships_csv_data()

    with open(relationships_csv_path, 'w') as f:
        f.write("start_frame,end_frame,start_time,end_time,duration,objects\n")
        for rel in relationships_data:
            start_time = (rel['start_frame'] - 1) / fps
            end_time = (rel['end_frame'] - 1) / fps
            duration = end_time - start_time
            objects_str = ', '.join(sorted(rel['objects']))
            f.write(
                f"{rel['start_frame']},{rel['end_frame']},{start_time:.2f},{end_time:.2f},{duration:.2f},{objects_str}\n")

    print(f"Saved relationships to {relationships_csv_path}")
    print(f"Total relationships recorded: {len(relationships_data)}")

    # post-processing productivity analysis
    if ENABLE_PRODUCTIVITY_ANALYSIS:
        try:
            analysis_data = generate_productivity_analysis_images(
                video_name,
                csv_path,
                relationships_csv_path
            )
            print(f"✓ Complete! Analysis images saved to: {analysis_output_directory}")
        except Exception as e:
            print(f"✗ Error generating productivity analysis: {e}")
            import traceback
            traceback.print_exc()
    return csv_path

if __name__ == "__main__":
    video_name = video_default
    process_video(video_name)
