import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# -----------------------------
# CONFIG
# -----------------------------
VIDEO_NAME = "video_10"
VIDEO_PATH = VIDEO_NAME + ".mp4"
SAVE_DIR = "keyframes/keyframes" + VIDEO_NAME
os.makedirs(SAVE_DIR, exist_ok=True)
SCALE = 0.5          # resize frames for speed
MIN_GAP = 20         # min frames between keyframes
K = 1              # threshold multiplier


# -----------------------------
# SETUP
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
success, prev_frame = cap.read()
if not success:
   print("Error: Cannot read video")
   exit()


os.makedirs(SAVE_DIR, exist_ok=True)


prev_frame_small = cv2.resize(prev_frame, (0, 0), fx=SCALE, fy=SCALE)
prev_gray = cv2.cvtColor(prev_frame_small, cv2.COLOR_BGR2GRAY)


frame_index = 0
last_saved = -9999
diff_values = []


# -----------------------------
# PROCESS VIDEO
# -----------------------------
while True:
   success, frame = cap.read()
   if not success:
       break


   frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
   frame_small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
   frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)


   # ---- Fast frame difference ----
   diff = cv2.absdiff(prev_gray, frame_gray)
   diff_val = np.mean(diff)
   diff_values.append(diff_val)


   # Adaptive threshold
   if len(diff_values) > 5:
       mean_val = np.mean(diff_values)
       std_val = np.std(diff_values)
       threshold = mean_val + K * std_val
   else:
       threshold = diff_val * 2


   # Save keyframe if difference exceeds threshold and MIN_GAP satisfied
   if diff_val > threshold and frame_index - last_saved > MIN_GAP:
       filename = os.path.join(SAVE_DIR, f"{frame_index}.jpg")
       cv2.imwrite(filename, frame)
       last_saved = frame_index
       print(f"Saved keyframe at frame {frame_index}")


   prev_gray = frame_gray


cap.release()


# -----------------------------
# PLOT
# -----------------------------
# plt.figure(figsize=(14, 4))
# plt.plot(diff_values, label="Frame Difference")
# plt.axhline(threshold, color='r', linestyle='--', label="Adaptive Threshold")
# plt.title("Fast Keyframe Detection Metric")
# plt.xlabel("Frame")
# plt.ylabel("Difference Value")
# plt.legend()
# plt.grid(True)
# plt.show()
