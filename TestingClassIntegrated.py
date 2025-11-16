"""
Integrated testing class
uses data from action classification and object detection and interaction classification
feeds it all to LLM to analyse
"""



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
import webbrowser

# ----------------------------
# Configuration
# ----------------------------
video_directory = "testing_vids/"
keyframes_directory = "keyframes/keyframes"
video_output_directory = "outputs/vid_objs"
csv_directory = "outputs/data/"
analysis_output_directory = "outputs/analysis/"
video_default = "video_01"

# Toggle productivity analysis
ENABLE_PRODUCTIVITY_ANALYSIS = True

# ----------------------------
# Module-level setup
# ----------------------------
from config import ANTHROPIC_API_KEY

client = Anthropic(api_key=ANTHROPIC_API_KEY)

_thread_local = local()
confidence_threshold = 0.60

# Relationship tracking configuration
proximity_threshold_percent = 0.20  # 20% of frame width


def get_local_model():
    if not hasattr(_thread_local, "model"):
        _thread_local.model = YOLO("weights.pt")
    return _thread_local.model


# ----------------------------
# Color management for object classes
# ----------------------------
def get_color_for_class(class_name, all_class_names):
    """Generate consistent color for each object class."""
    if not hasattr(get_color_for_class, 'color_map'):
        get_color_for_class.color_map = {}

    if class_name not in get_color_for_class.color_map:
        # Generate evenly distributed colors using HSV
        num_classes = len(all_class_names)
        class_idx = list(all_class_names).index(class_name) if class_name in all_class_names else len(
            get_color_for_class.color_map)
        hue = (class_idx * 0.618033988749895) % 1.0  # Golden ratio for distribution
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        get_color_for_class.color_map[class_name] = tuple(int(c * 255) for c in rgb)

    return get_color_for_class.color_map[class_name]


# ----------------------------
# Relationship tracking
# ----------------------------
class RelationshipTracker:
    def __init__(self, fps, proximity_threshold_percent):
        self.fps = fps
        self.proximity_threshold_percent = proximity_threshold_percent

        # Active relationships: key = frozenset of object names, value = start_frame
        self.active_relationships = {}

        # Completed relationships: list of {objects: set, start_frame, end_frame}
        self.completed_relationships = []

    def get_box_center(self, box):
        """Get center point of bounding box."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_distance(self, center1, center2):
        """Calculate Euclidean distance between two centers."""
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    def find_relationships(self, detections, frame_width, current_frame):
        """
        Find relationships between objects in current frame.
        detections: list of (box, class_name, confidence)
        Returns: list of relationships (sets of object names) and line drawing info
        """
        proximity_threshold = frame_width * self.proximity_threshold_percent

        if len(detections) < 2:
            return [], []

        # Calculate centers for all detections
        centers = [(self.get_box_center(det[0]), det[1]) for det in detections]

        # Build adjacency: which objects are close to each other
        n = len(centers)
        adjacency = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.get_distance(centers[i][0], centers[j][0])
                if dist <= proximity_threshold:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        # Find connected components (groups of close objects)
        visited = [False] * n
        relationships = []

        def dfs(node, component):
            visited[node] = True
            component.add(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    dfs(neighbor, component)

        for i in range(n):
            if not visited[i] and adjacency[i]:  # Has at least one close neighbor
                component = set()
                dfs(i, component)
                if len(component) >= 2:
                    relationships.append(component)

        # Convert indices to object names and prepare line drawing info
        relationship_sets = []
        line_info = []

        for component in relationships:
            obj_names = frozenset(centers[idx][1] for idx in component)
            relationship_sets.append(obj_names)

            # Prepare lines to draw between all pairs in component
            indices = list(component)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    center1 = centers[indices[i]][0]
                    center2 = centers[indices[j]][0]
                    line_info.append((center1, center2))

        return relationship_sets, line_info

    def update(self, relationships, current_frame):
        """Update relationship tracking with current frame's relationships."""
        current_relationship_keys = set(relationships)

        # Start new relationships or continue existing ones
        for rel in relationships:
            if rel not in self.active_relationships:
                # New relationship detected - start tracking it
                self.active_relationships[rel] = current_frame
                print(f"Frame {current_frame}: Started tracking relationship: {set(rel)}")

        # Check for ended relationships
        ended_keys = []
        for rel_key, start_frame in self.active_relationships.items():
            if rel_key not in current_relationship_keys:
                # Relationship ended - record it immediately
                self.completed_relationships.append({
                    'objects': rel_key,
                    'start_frame': start_frame,
                    'end_frame': current_frame - 1  # Last frame where it was active
                })
                ended_keys.append(rel_key)
                print(
                    f"Frame {current_frame}: Ended relationship: {set(rel_key)} (lasted {current_frame - start_frame} frames)")

        for key in ended_keys:
            del self.active_relationships[key]

    def finalize(self, last_frame):
        """Finalize all active relationships at end of video."""
        for rel_key, start_frame in self.active_relationships.items():
            self.completed_relationships.append({
                'objects': rel_key,
                'start_frame': start_frame,
                'end_frame': last_frame
            })
            print(f"Finalized relationship: {set(rel_key)} (lasted {last_frame - start_frame + 1} frames)")
        self.active_relationships.clear()

    def get_relationships_csv_data(self):
        """Get relationship data formatted for CSV output."""
        return self.completed_relationships


def draw_relationship_lines(frame, line_info):
    """Draw lines between related objects."""
    for center1, center2 in line_info:
        pt1 = (int(center1[0]), int(center1[1]))
        pt2 = (int(center2[0]), int(center2[1]))
        cv2.line(frame, pt1, pt2, (255, 255, 0), 5)  # Yellow lines
        # Draw circles at centers
        cv2.circle(frame, pt1, 10, (255, 255, 0), -1)
        cv2.circle(frame, pt2, 10, (255, 255, 0), -1)
    return frame


# ----------------------------
# User-defined actions
# ----------------------------
def set_actions(action_list):
    global ALLOWED_ACTIONS, ACTION_PROMPT
    ALLOWED_ACTIONS = [a.lower() for a in action_list]
    actions_joined = ", ".join(ALLOWED_ACTIONS)
    ACTION_PROMPT = (
        f"You must classify the person's behavior into ONE of the following categories: "
        f"{actions_joined}. Respond ONLY with one word from this list, no explanations."
    )


set_actions(["using tool", "idle", "moving"])


# ----------------------------
# Encode frames for Claude
# ----------------------------
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


# ----------------------------
# Ask Claude using multiple frames
# ----------------------------
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


# ----------------------------
# Video helpers
# ----------------------------
frame_cache = {}


def get_frame(frame_number):
    """Get frame from cache (cache is cleared per video)"""
    if frame_number in frame_cache:
        return frame_cache[frame_number]

    # Frame not in cache - this shouldn't happen with pre-loaded frames
    raise ValueError(f"Frame {frame_number} not in cache")


def sample_interval_frames(start, end, num_frames=5):
    """Sample evenly spaced frames in an interval."""
    return [get_frame(int(start + i * (end - start) / (num_frames - 1))) for i in range(num_frames)]


def motion_score(frames, ignore_threshold=5):
    """
    Compute a motion score for a sequence of frames.

    frames: list of consecutive BGR frames
    ignore_threshold: pixel difference below this is ignored (camera noise / jitter)

    Returns: normalized motion score (0.0 - 1.0)
    """
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


# ----------------------------
# Overlay text
# ----------------------------
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
    """Draw YOLO results on a given frame manually with class-specific colors."""
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


# ----------------------------
# Productivity Analysis
# ----------------------------
def generate_productivity_analysis(video_name, actions_csv_path, relationships_csv_path):
    """
    Generate productivity analysis by sending action and relationship data to Claude API.
    Claude will create interactive HTML visualizations.
    """
    print("\n" + "=" * 60)
    print("GENERATING PRODUCTIVITY ANALYSIS")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(analysis_output_directory, exist_ok=True)

    # Read the CSV files
    with open(actions_csv_path, 'r') as f:
        actions_data = f.read()

    with open(relationships_csv_path, 'r') as f:
        relationships_data = f.read()

    # Create the analysis prompt
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
Based on the data and context given, represent the worker's productivity in a visual manner through graphs and pie charts. 

Your analysis should:
1. Infer which activities are productive based on context and object combinations
2. Calculate time spent on different activity types
3. Identify specific tasks performed (inferred from object combinations)
4. Show productivity trends over time
5. Highlight key insights about the worker's behavior

Create a COMPLETE, SELF-CONTAINED HTML file with:
- Interactive charts (use Chart.js from CDN)
- Professional styling
- Clear section headers
- Summary statistics
- All necessary CSS and JavaScript inline or from CDN
- The HTML should work when opened directly in a browser

Respond ONLY with the complete HTML code, no explanations before or after."""

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

    # Extract HTML content
    html_content = response.content[0].text

    # Clean up HTML if it's wrapped in markdown code blocks
    if html_content.startswith("```html"):
        html_content = html_content.replace("```html", "", 1)
    if html_content.startswith("```"):
        html_content = html_content.replace("```", "", 1)
    if html_content.endswith("```"):
        html_content = html_content.rsplit("```", 1)[0]
    html_content = html_content.strip()

    # Save HTML file
    html_output_path = os.path.join(analysis_output_directory, f"{video_name}_productivity_report.html")
    with open(html_output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ Productivity analysis saved to: {html_output_path}")

    # Try to open in browser
    try:
        webbrowser.open('file://' + os.path.abspath(html_output_path))
        print("✓ Analysis opened in browser")
    except:
        print("✗ Could not automatically open browser")
        print(f"  Please open manually: {html_output_path}")

    print("=" * 60 + "\n")

    return html_output_path


# ----------------------------
# Main processing function
# ----------------------------
def process_video(video_name):
    """Process a single video and generate labeled output."""

    # Clear frame cache for new video
    global frame_cache
    frame_cache = {}

    video_path = video_directory + video_name + ".mp4"
    keyframes_folder = keyframes_directory + video_name
    keyframe_files = sorted(os.listdir(keyframes_folder), key=lambda x: int(x.rstrip('.jpg')))

    # Pre-load all frames into cache
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

    # Get video properties
    cap_temp = cv2.VideoCapture(video_path)
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_temp.release()

    # Initialize relationship tracker
    relationship_tracker = RelationshipTracker(
        fps,
        proximity_threshold_percent
    )

    # ----------------------------
    # Process keyframes (multithreaded)
    # ----------------------------
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

    # ----------------------------
    # Fill intervals with multi-frame + motion hints (multithreaded)
    # ----------------------------
    def process_interval(idx):
        start = keyframe_numbers[idx]
        end = keyframe_numbers[idx + 1]
        frames_to_send = sample_interval_frames(start, end, num_frames=5)

        # Include neighbor frames for context
        if start > 1:
            frames_to_send.insert(0, get_frame(max(1, start - 1)))
        if end < frame_count:
            frames_to_send.append(get_frame(min(frame_count, end + 1)))

        motion = motion_score(frames_to_send)

        # Get objects from frames
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

    # Fill after last keyframe
    last_kf = keyframe_numbers[-1]
    for f in range(last_kf, frame_count):
        frame_labels[f] = frame_labels[last_kf - 1]

    # ----------------------------
    # Temporal smoothing
    # ----------------------------
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

    # ----------------------------
    # Save labeled video with relationship tracking
    # ----------------------------
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

        # Run YOLO detection every 5 frames
        if i % 5 == 0:
            results = model(frame, conf=confidence_threshold)[0]
            latest_boxes = results

            # Extract detections with boxes for relationship tracking
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

        # Track relationships - now processes every frame
        if latest_detections:
            relationships, line_info = relationship_tracker.find_relationships(
                latest_detections, width, i + 1
            )
            relationship_tracker.update(relationships, i + 1)
        else:
            # No detections, so end any active relationships
            relationship_tracker.update([], i + 1)
            line_info = []

        # Draw boxes from latest detection
        frame_with_boxes = frame.copy()
        if latest_boxes is not None:
            frame_with_boxes = draw_boxes_on_frame(frame_with_boxes, latest_boxes)

        # Draw relationship lines
        if line_info:
            frame_with_boxes = draw_relationship_lines(frame_with_boxes, line_info)

        # Overlay action label
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

    # Finalize relationship tracking
    relationship_tracker.finalize(frame_count)

    # Save CSV with state changes
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

    # Save relationships CSV
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

    # ----------------------------
    # Generate Productivity Analysis (if enabled)
    # ----------------------------
    if ENABLE_PRODUCTIVITY_ANALYSIS:
        try:
            analysis_path = generate_productivity_analysis(
                video_name,
                csv_path,
                relationships_csv_path
            )
            print(f"✓ Complete! Analysis available at: {analysis_path}")
        except Exception as e:
            print(f"✗ Error generating productivity analysis: {e}")
            import traceback
            traceback.print_exc()

    return csv_path


if __name__ == "__main__":
    video_name = video_default
    process_video(video_name)
