"""
Video Utilities - Frame handling, motion detection, video properties

Common utilities for video processing.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import OrderedDict
from threading import Lock
from typing import Iterator, Tuple, List, Dict, Optional



class FrameLoader:
    """
    Lazy frame loader with LRU cache.
    
    Provides dict-like interface for backward compatibility.
    Frames are 1-indexed to match existing codebase convention.
    
    Memory usage: O(max_frames) instead of O(total_frames)
    
    Thread-safe: Yes, uses Lock for cache operations.
    
    Usage:
        loader = FrameLoader("video.mp4", max_frames=500)
        frame = loader[100]      # dict-style access
        frame = loader.get(100)  # .get() access
        loader.close()           # Release video file
    
    Context manager usage:
        with FrameLoader("video.mp4") as loader:
            frame = loader[100]
    """
    
    def __init__(self, video_path: str, max_frames: int = 500):
        self.video_path = video_path
        self.max_frames = max_frames
        
        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Extract properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize cache
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._pinned: Dict[int, np.ndarray] = {}  # Pinned frames never evicted
        self._lock = Lock()
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Track last accessed frame for sequential optimization
        self._last_read_position = 0

    def get(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get frame by number (1-indexed).
        
        Returns None if frame_number is out of bounds or read fails.
        Thread-safe.
        """
        # Bounds check
        if frame_number < 1 or frame_number > self.frame_count:
            return None
        
        with self._lock:
            # Check pinned frames first
            if frame_number in self._pinned:
                self.cache_hits += 1
                return self._pinned[frame_number]

            # Cache hit
            if frame_number in self._cache:
                self.cache_hits += 1
                self._cache.move_to_end(frame_number)  # Mark as recently used
                return self._cache[frame_number]
            
            # Cache miss
            self.cache_misses += 1
            frame = self._load_frame(frame_number)
            
            if frame is not None:
                self._cache[frame_number] = frame
                self._cache.move_to_end(frame_number)
                
                # Evict oldest frames if over limit
                while len(self._cache) > self.max_frames:
                    self._cache.popitem(last=False)
            
            return frame

    def _load_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Load single frame from video file.
        
        Internal method - caller must hold _lock.
        Converts BGR to RGB to match existing codebase convention.
        """
        # OpenCV is 0-indexed, our API is 1-indexed
        target_position = frame_number - 1
        
        # Retry logic for robust reading
        max_retries = 3
        for attempt in range(max_retries):
            # Seek to frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_position)
            
            # Read frame
            ret, frame = self.cap.read()
            
            if ret:
                # Return BGR to match existing convention
                return frame
                
            # If failed, wait briefly and retry
            import time
            time.sleep(0.1)
            
        return None

    def preload(self, frame_numbers: List[int]) -> None:
        """
        Preload specific frames into cache.
        
        Useful for preloading keyframes before processing.
        Silently skips invalid frame numbers.
        """
        for fn in frame_numbers:
            if 1 <= fn <= self.frame_count:
                self.get(fn)  # Populates cache as side effect

    def pin(self, frame_numbers: List[int]) -> None:
        """
        Load and pin frames in memory. Pinned frames are never evicted.
        Useful for keyframes that must remain available.
        """
        with self._lock:
            for fn in frame_numbers:
                if 1 <= fn <= self.frame_count:
                    if fn not in self._pinned:
                        frame = self._load_frame(fn)
                        if frame is not None:
                            self._pinned[fn] = frame
                            # Remove from LRU if present to save space
                            if fn in self._cache:
                                del self._cache[fn]

    def iter_frames(self, start: int = 1, end: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterate frames sequentially without caching.
        
        More efficient than random access for video generation.
        Does NOT populate cache (to avoid evicting useful frames).
        
        Args:
            start: First frame number (1-indexed, default 1)
            end: Last frame number (1-indexed, default frame_count)
        
        Yields:
            Tuple of (frame_number, frame_array)
        
        WARNING: Not thread-safe. Do not use concurrently with get().
        """
        if end is None:
            end = self.frame_count
        
        # Seek to start position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
        
        for fn in range(start, end + 1):
            ret, frame = self.cap.read()
            if not ret:
                break
            # Return BGR to match existing convention
            yield fn, frame

    def get_stats(self) -> Dict:
        """Return cache performance statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'current_cache_size': len(self._cache),
            'max_cache_size': self.max_frames,
            'total_frames': self.frame_count
        }

    def clear_cache(self) -> None:
        """Clear frame cache, keep video file open"""
        with self._lock:
            self._cache.clear()
            self._pinned.clear()
            self.cache_hits = 0
            self.cache_misses = 0

    def close(self) -> None:
        """Release video file and clear cache"""
        with self._lock:
            self._cache.clear()
            self._pinned.clear()
        self.cap.release()

    def __enter__(self) -> 'FrameLoader':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __len__(self) -> int:
        """Return total frame count (for len(loader))"""
        return self.frame_count

    def __getitem__(self, frame_number: int) -> np.ndarray:
        """
        Dict-style access: loader[frame_number]
        
        Raises KeyError if frame doesn't exist (matches dict behavior).
        """
        frame = self.get(frame_number)
        if frame is None:
            raise KeyError(f"Frame {frame_number} not found or could not be read")
        return frame

    def __contains__(self, frame_number: int) -> bool:
        """Support 'frame_number in loader' syntax"""
        return 1 <= frame_number <= self.frame_count


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
    if hasattr(frame_cache, 'get'):
        frame = frame_cache.get(frame_number)
        if frame is None:
            raise ValueError(f"Frame {frame_number} not in cache or could not be read")
        return frame
    
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


def extract_keyframes(
    video_path: str,
    output_dir: str,
    scale: float = 0.5,
    min_gap: int = 20,
    max_gap: int = 300,
    threshold_multiplier: float = 1.0
) -> int:
    """
    Extract keyframes from video using adaptive thresholding.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save keyframes
        scale: Resize factor for processing speed (default: 0.5)
        min_gap: Minimum frames between keyframes (default: 20)
        max_gap: Maximum frames between keyframes (default: 300)
        threshold_multiplier: Multiplier for adaptive threshold (default: 1.0)
    
    Returns:
        Number of keyframes extracted
    """
    import os
    import cv2
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    success, prev_frame = cap.read()
    if not success:
        cap.release()
        raise RuntimeError(f"Failed to read first frame: {video_path}")
    
    # Initial processing
    prev_frame_small = cv2.resize(prev_frame, (0, 0), fx=scale, fy=scale)
    prev_gray = cv2.cvtColor(prev_frame_small, cv2.COLOR_BGR2GRAY)
    
    frame_index = 0
    last_saved = -9999
    diff_values = []
    saved_count = 0
    
    # Always save first frame
    cv2.imwrite(os.path.join(output_dir, "1.jpg"), prev_frame)
    last_saved = 1
    saved_count += 1
    print(f"  Saved keyframe at frame 1")
    
    last_frame_captured = None
    last_frame_index = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Use 1-based indexing to match load_all_frames
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        last_frame_captured = frame
        last_frame_index = frame_index
        
        # Resize for speed
        frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(prev_gray, frame_gray)
        diff_val = np.mean(diff)
        diff_values.append(diff_val)
        
        # Adaptive threshold
        if len(diff_values) > 5:
            mean_val = np.mean(diff_values)
            std_val = np.std(diff_values)
            threshold = mean_val + threshold_multiplier * std_val
        else:
            threshold = diff_val * 2
            
        # Determine if we should save
        should_save = False
        reason = ""
        
        # 1. Motion threshold check
        if diff_val > threshold and frame_index - last_saved > min_gap:
            should_save = True
            reason = "motion"
            
        # 2. Max gap check (force keyframe if too long since last one)
        if frame_index - last_saved >= max_gap:
            should_save = True
            reason = "max_gap"
            
        if should_save:
            filename = os.path.join(output_dir, f"{frame_index}.jpg")
            cv2.imwrite(filename, frame)
            last_saved = frame_index
            saved_count += 1
            if saved_count % 10 == 0 or reason == "max_gap":
                print(f"  Saved keyframe at frame {frame_index} ({reason})")
        
        prev_gray = frame_gray
    
    cap.release()
    print(f"  Extracted {saved_count} keyframes to {output_dir}")
    return saved_count


def apply_temporal_smoothing(
    frame_labels: List[str],
    allowed_actions: List[str],
    window_size: int = 9
) -> List[str]:
    """
    Apply temporal smoothing to reduce jitter in classifications.
    
    For each frame, looks at surrounding frames and picks most common action.
    
    Args:
        frame_labels: List of action labels (one per frame)
        allowed_actions: List of allowed action names
        window_size: Size of smoothing window
    
    Returns:
        Smoothed list of action labels
    """
    frame_count = len(frame_labels)
    smoothed_labels = frame_labels.copy()
    
    for i in range(frame_count):
        start = max(0, i - window_size)
        end = min(frame_count, i + window_size)
        
        # Count actions in window
        counts = {action: 0 for action in allowed_actions}
        for j in range(start, end):
            label = frame_labels[j]
            if label in counts:
                counts[label] += 1
        
        # Pick most common
        if sum(counts.values()) > 0:
            smoothed_labels[i] = max(counts, key=counts.get)
        else:
            smoothed_labels[i] = allowed_actions[0]  # Fallback
    
    return smoothed_labels
