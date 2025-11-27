"""
Video Utilities - Frame handling, motion detection, video properties

Common utilities for video processing.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class VideoProperties:
    """Container for video properties"""
    def __init__(self, fps: float, width: int, height: int, frame_count: int):
        self.fps = fps
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.duration = frame_count / fps if fps > 0 else 0


def get_video_properties(video_path: str) -> VideoProperties:
    """
    Extract video properties (fps, dimensions, frame count)
    
    Args:
        video_path: Path to video file
    
    Returns:
        VideoProperties object
    """
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return VideoProperties(fps, width, height, frame_count)


def load_all_frames(video_path: str) -> Dict[int, np.ndarray]:
    """
    Load all frames from video into memory.
    
    Note: Some videos report incorrect frame counts in metadata.
    This function loads all actually-readable frames.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dict mapping frame_number (1-indexed) -> frame array
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video cannot be opened or no frames can be read
    """
    # Check if file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV failed to open video: {video_path}. "
                         f"File may be corrupted or use an unsupported codec.")
    
    frame_cache = {}
    frame_number = 1
    
    # Get reported frame count (may be inaccurate)
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Try to read first frame to verify video is readable
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Cannot read frames from video: {video_path}. "
                         f"Video file may be corrupted or empty.")
    
    # Store first frame
    frame_cache[1] = first_frame
    frame_number = 2
    
    # Read remaining frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_cache[frame_number] = frame
        frame_number += 1
    
    cap.release()
    
    actual_frames = len(frame_cache)
    if actual_frames != reported_frames:
        print(f"  ⚠ Video metadata mismatch: reported {reported_frames} frames, "
              f"actually loaded {actual_frames} frames")
    
    if actual_frames == 0:
        raise RuntimeError(f"No frames could be loaded from video: {video_path}")
    
    return frame_cache


def get_frame_from_cache(frame_cache: Dict[int, np.ndarray], frame_number: int) -> np.ndarray:
    """
    Retrieve frame from cache.
    
    Args:
        frame_cache: Frame cache dict
        frame_number: Frame number (1-indexed)
    
    Returns:
        Frame array
    
    Raises:
        ValueError: If frame not in cache
    """
    if frame_number not in frame_cache:
        raise ValueError(f"Frame {frame_number} not in cache")
    return frame_cache[frame_number]


def sample_interval_frames(
    frame_cache: Dict[int, np.ndarray],
    start_frame: int,
    end_frame: int,
    num_frames: int = 5
) -> List[np.ndarray]:
    """
    Sample evenly-spaced frames from an interval.
    
    Args:
        frame_cache: Frame cache dict
        start_frame: Start frame number (inclusive)
        end_frame: End frame number (inclusive)
        num_frames: Number of frames to sample
    
    Returns:
        List of sampled frames
    """
    frames = []
    for i in range(num_frames):
        frame_num = int(start_frame + i * (end_frame - start_frame) / (num_frames - 1))
        frames.append(get_frame_from_cache(frame_cache, frame_num))
    return frames


def calculate_motion_score(
    frames: List[np.ndarray],
    ignore_threshold: int = 5
) -> float:
    """
    Calculate motion score across frames.
    
    Computes average pixel difference between consecutive frames.
    Higher score = more motion.
    
    Args:
        frames: List of frames
        ignore_threshold: Ignore pixel differences below this value
    
    Returns:
        Motion score (0.0 to 1.0)
    """
    if len(frames) < 2:
        return 0.0
    
    total_motion = 0.0
    
    for i in range(1, len(frames)):
        gray_prev = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_curr, gray_prev)
        
        # Ignore tiny changes
        diff[diff < ignore_threshold] = 0
        total_motion += np.mean(diff) / 255.0  # Normalize to 0-1
    
    # Average across frame pairs
    motion_score = total_motion / (len(frames) - 1)
    return motion_score


def load_keyframe_numbers(keyframes_folder: str) -> List[int]:
    """
    Load keyframe numbers from keyframes directory.
    
    Args:
        keyframes_folder: Path to keyframes directory
    
    Returns:
        Sorted list of keyframe numbers
    """
    import os
    
    if not os.path.exists(keyframes_folder):
        raise FileNotFoundError(f"Keyframes folder not found: {keyframes_folder}")
    
    keyframe_files = os.listdir(keyframes_folder)
    keyframe_numbers = []
    
    for filename in keyframe_files:
        if filename.endswith('.jpg'):
            # Extract number from filename (e.g., "123.jpg" -> 123)
            try:
                frame_num = int(filename.rstrip('.jpg'))
                keyframe_numbers.append(frame_num)
            except ValueError:
                continue
    
    return sorted(keyframe_numbers)


def create_video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
    codec: str = 'mp4v'
) -> cv2.VideoWriter:
    """
    Create video writer for output.
    
    Args:
        output_path: Path for output video
        fps: Frames per second
        width: Frame width
        height: Frame height
        codec: Video codec (default: mp4v)
    
    Returns:
        cv2.VideoWriter object
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return writer
