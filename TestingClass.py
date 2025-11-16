import os
import base64
from anthropic import Anthropic
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from difflib import get_close_matches
from concurrent.futures import ThreadPoolExecutor


# ----------------------------
# 1. Claude client
# ----------------------------
from config import ANTHROPIC_API_KEY
client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ----------------------------
# 2. User-defined actions
# ----------------------------
def set_actions(action_list):


   global ALLOWED_ACTIONS, ACTION_PROMPT
   ALLOWED_ACTIONS = [a.lower() for a in action_list]
   actions_joined = ", ".join(ALLOWED_ACTIONS)
   ACTION_PROMPT = (
       f"You must classify the person's behavior into ONE of the following categories: "
       f"{actions_joined}. Respond ONLY with one word from this list, no explanations."
   )


# Example: Replace with user input or CLI
set_actions(["using tool", "idle", "moving"])


# ----------------------------
# 3. Encode frames for Claude
# ----------------------------
def encode_frames(frames, width_small=640, height_small=360):
   encoded_list = []
   for frame in frames:
       frame_small = cv2.resize(frame, (width_small, height_small))
       _, buffer_img = cv2.imencode('.jpg', frame_small)
       encoded_list.append(base64.b64encode(buffer_img.tobytes()).decode("utf-8"))
   return encoded_list


# ----------------------------
# 4. Ask Claude using multiple frames
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
       model="claude-sonnet-4-5-20250929",
       max_tokens=200,
       messages=[{"role": "user", "content": content_list}]
   )
   text = response.content[0].text.strip().lower()


   # Match closest action
   match = get_close_matches(text, ALLOWED_ACTIONS, n=1)
   if match:
       return match[0]
   for action in ALLOWED_ACTIONS:
       if action in text:
           return action
   return ALLOWED_ACTIONS[0]  # fallback


# ----------------------------
# 5. Video helpers
# ----------------------------
frame_cache = {}
def get_frame(video_path, frame_number):
   if frame_number in frame_cache:
       return frame_cache[frame_number]
   cap = cv2.VideoCapture(video_path)
   cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
   ret, frame = cap.read()
   cap.release()
   if not ret:
       raise ValueError(f"Cannot read frame {frame_number}")
   frame_cache[frame_number] = frame
   return frame


# CHANGE: Added video_path parameter
def sample_interval_frames(video_path, start, end, num_frames=5):
   """Sample evenly spaced frames in an interval."""
   return [get_frame(video_path, int(start + i*(end-start)/(num_frames-1))) for i in range(num_frames)]




def motion_score(frames, ignore_threshold=5):
   """
   Compute a motion score for a sequence of frames (no hands needed).


   frames: list of consecutive BGR frames
   ignore_threshold: pixel difference below this is ignored (camera noise / jitter)


   Returns: normalized motion score (0.0 - 1.0)
   """
   if len(frames) < 2:
       return 0.0


   total_motion = 0.0
   h, w = frames[0].shape[:2]
   diag = np.sqrt(h ** 2 + w ** 2)


   for i in range(1, len(frames)):
       gray_prev = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
       gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
       diff = cv2.absdiff(gray_curr, gray_prev)


       # ignore tiny changes
       diff[diff < ignore_threshold] = 0


       total_motion += np.mean(diff) / 255.0  # normalize 0-1


   # normalize by number of frame differences
   motion_score = total_motion / (len(frames) - 1)


   return motion_score
# ----------------------------
# 6. Overlay text
# ----------------------------
try:
    #TODO: This isn't running for some reason?
   font = ImageFont.truetype("arial.ttf", 500)
   print('AAAAA')
except:
   font = ImageFont.load_default(500)


def overlay_action(frame, label):
   img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
   draw = ImageDraw.Draw(img)
   draw.text((30, 50), label, font=font, fill=(255, 0, 0))
   return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ----------------------------
# CHANGE: Wrapped sections 7-10 in process_video function
# ----------------------------
def process_video(video_name):
    """Process a single video and generate labeled output."""
    
    # ----------------------------
    # 7. Process keyframes (multithreaded)
    # ----------------------------
    video_path = "videos/" + video_name + ".mp4"
    keyframes_folder = "keyframes/keyframes" + video_name
    keyframe_files = sorted(os.listdir(keyframes_folder), key=lambda x: int(x.rstrip('.jpg')))
    frame_count = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    frame_labels = [""] * frame_count


    def process_keyframe(kf_file):
       kf_number = int(kf_file.rstrip(".jpg"))
       frame = get_frame(video_path, kf_number)
       prompt = f"This is a POV of a person. Check if hands are visible. {ACTION_PROMPT}"
       label = ask_claude_multiframe([frame], prompt)
       print(f"Keyframe {kf_number}: {label}")
       return kf_number, label


    with ThreadPoolExecutor(max_workers=4) as executor:
       keyframe_results = list(executor.map(process_keyframe, keyframe_files))


    keyframe_results.sort(key=lambda x: x[0])
    keyframe_numbers = []
    for kf_number, label in keyframe_results:
       frame_labels[kf_number - 1] = label
       keyframe_numbers.append(kf_number)


    # ----------------------------
    # 8. Fill intervals with multi-frame + motion hints (multithreaded)
    # ----------------------------
    def process_interval(idx):
       start = keyframe_numbers[idx]
       end = keyframe_numbers[idx + 1]
       # CHANGE: Pass video_path to sample_interval_frames
       frames_to_send = sample_interval_frames(video_path, start, end, num_frames=5)


       # Include neighbor frames for context
       if start > 1:
           frames_to_send.insert(0, get_frame(video_path, max(1, start-1)))
       if end < frame_count:
           frames_to_send.append(get_frame(video_path, min(frame_count, end+1)))


       motion = motion_score(frames_to_send)
       print(motion)
       prompt = f"This is a POV of a person. Motion score: {motion:.2f}, a motion score of 0-0.16 suggests the person is idle, 10 or above suggests they are moving. Check if hands are visible and classify behavior. {ACTION_PROMPT}"
       label = ask_claude_multiframe(frames_to_send, prompt)
       return start, end, label


    with ThreadPoolExecutor(max_workers=2) as executor:
       interval_results = list(executor.map(process_interval, range(len(keyframe_numbers)-1)))


    for start, end, label in interval_results:
       for f in range(start, end):
           frame_labels[f] = label


    # Fill after last keyframe
    last_kf = keyframe_numbers[-1]
    for f in range(last_kf, frame_count):
       frame_labels[f] = frame_labels[last_kf - 1]


    # ----------------------------
    # 9. Temporal smoothing
    # ----------------------------
    window = 9
    smoothed_labels = frame_labels.copy()
    for i in range(frame_count):
       start = max(0, i - window)
       end = min(frame_count, i + window)
       counts = {a: 0 for a in ALLOWED_ACTIONS}
       for j in range(start, end):
           label = frame_labels[j]
           if label in ALLOWED_ACTIONS:
               counts[label] += 1
       if sum(counts.values()) == 0:
           smoothed_labels[i] = ALLOWED_ACTIONS[0]
       else:
           smoothed_labels[i] = max(counts, key=counts.get)
    frame_labels = smoothed_labels


    # ----------------------------
    # 10. Save labeled video
    # ----------------------------
    cap_playback = cv2.VideoCapture(video_path)
    fps = cap_playback.get(cv2.CAP_PROP_FPS)
    width = int(cap_playback.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_playback.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = "Outputs/labeled_output" + video_path
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))


    for i in range(frame_count):
       ret, frame = cap_playback.read()
       if not ret:
           break
       if frame_labels[i]:
           frame = overlay_action(frame, frame_labels[i])
       out.write(frame)


    cap_playback.release()
    out.release()
    print(f"Saved labeled video to {out_path}")

    # Save CSV with state changes (frame, label)
    csv_path = "Outputs/Data/" + video_name + ".csv"
    total_duration = frame_count / fps  # Calculate total duration
    with open(csv_path, 'w') as f:
        f.write(f"{fps},{total_duration}\n")
        
        if frame_count > 0:
            current_label = frame_labels[0]
            f.write(f"1,{current_label}\n")  # First frame
            
            for i in range(1, frame_count):
                if frame_labels[i] != current_label:
                    current_label = frame_labels[i]
                    f.write(f"{i+1},{current_label}\n")  # Frame where change occurs
            
    print(f"Saved labels to {csv_path}")
    
    return csv_path

# CHANGE: Added if __name__ block
if __name__ == "__main__":
    video_name = "video_03"
    process_video(video_name)