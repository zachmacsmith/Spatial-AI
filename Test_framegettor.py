import cv2
import os
import sys
import random
import numpy as np

def verify_seeking(video_path, output_dir="debug_frames"):
    """
    Diagnoses frame seeking accuracy by saving frames with stamped timing data.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # specific properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Analysis:")
    print(f"  Path: {video_path}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames (Metadata): {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print("-" * 40)

    # Select 5 test points spread across the video
    # We avoid the very beginning (0) as that usually works even in broken videos
    test_indices = sorted([
        int(total_frames * 0.1),
        int(total_frames * 0.3),
        int(total_frames * 0.5),
        int(total_frames * 0.7),
        int(total_frames * 0.9)
    ])

    print(f"Testing seeking at frames: {test_indices}\n")

    for target_frame_idx in test_indices:
        # 1. SEEK: This is the critical operation that might fail
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        
        # 2. READ
        ret, frame = cap.read()
        
        if not ret:
            print(f"FAILED to read frame {target_frame_idx}")
            continue

        # 3. VERIFY
        # Calculate where this frame SHOULD be in time
        expected_time = target_frame_idx / fps
        
        # Current position reported by OpenCV after reading
        actual_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Visual overlay
        h, w = frame.shape[:2]
        # Draw black background rectangle for text
        cv2.rectangle(frame, (10, 10), (500, 120), (0, 0, 0), -1)
        
        text_info = [
            f"Target Frame: {target_frame_idx}",
            f"Actual Pos:   {actual_pos}",
            f"Target Time:  {expected_time:.3f}s",
            f"FPS: {fps:.2f}"
        ]
        
        y = 40
        for line in text_info:
            cv2.putText(frame, line, (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30

        # Save to disk
        filename = f"{output_dir}/frame_{target_frame_idx}_time_{expected_time:.2f}s.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

    cap.release()
    print(f"\nDone. Check the '{output_dir}' folder.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_frames.py <path_to_video>")
    else:
        verify_seeking(sys.argv[1])