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

video_directory = "videos/"
keyframes_directory = "keyframes/keyframes"
video_output_directory = "Outputs/vid_objs"
csv_directory = "Outputs/Data2/"
video_default = "video_05"

# ----------------------------
# Module-level setup
# ----------------------------
from config import ANTHROPIC_API_KEY

client = Anthropic(api_key=ANTHROPIC_API_KEY)

_thread_local = local()
confidence_threshold = 0.60

def get_local_model():
    if not hasattr(_thread_local, "model"):
        _thread_local.model = YOLO("weights.pt")
    return _thread_local.model

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
    """Draw YOLO results on a given frame manually."""
    for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"{results.names[int(cls_id)]} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

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
    # Save labeled video
    # ----------------------------
    cap_playback = cv2.VideoCapture(video_path)
    fps = cap_playback.get(cv2.CAP_PROP_FPS)
    width = int(cap_playback.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_playback.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = video_output_directory + video_name + ".mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    model = get_local_model()
    latest_boxes = None
    tool = None

    for i in range(frame_count):
        ret, frame = cap_playback.read()
        if not ret:
            break

        # Run YOLO detection every 15 frames
        if i % 15 == 0:
            results = model(frame, conf=confidence_threshold)[0]
            latest_boxes = results

            objects = get_objects(frame)
            tool = None
            for o in objects:
                if o[0] not in ["safety vest", "person", "hardhat", "hand"]:
                    tool = o[0]
                    break

        # Draw boxes from latest detection
        frame_with_boxes = frame.copy()
        if latest_boxes is not None:
            frame_with_boxes = draw_boxes_on_frame(frame_with_boxes, latest_boxes)

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

    return csv_path2

if __name__ == "__main__":
    video_name = video_default
    process_video(video_name)