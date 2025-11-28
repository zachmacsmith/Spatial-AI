
import sys
import os
import cv2
import numpy as np
from video_processing.utils.video_utils import FrameLoader, load_all_frames

def test_frameloader():
    # Find a video file
    video_dir = "videos"
    if not os.path.exists(video_dir):
        print("No videos directory found")
        return
    
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not videos:
        print("No videos found")
        return
        
    video_path = os.path.join(video_dir, videos[0])
    print(f"Testing with video: {video_path}")
    
    # 1. Test Basic Loading
    print("\n1. Testing Basic Loading...")
    loader = FrameLoader(video_path, max_frames=10)
    print(f"Frame count: {loader.frame_count}")
    print(f"FPS: {loader.fps}")
    
    frame1 = loader.get(1)
    if frame1 is None:
        print("FAIL: Could not load frame 1")
    else:
        print(f"PASS: Loaded frame 1, shape: {frame1.shape}")
        
    # 2. Test 1-based Indexing
    print("\n2. Testing Indexing...")
    frame0 = loader.get(0)
    if frame0 is None:
        print("PASS: Frame 0 returned None (correctly 1-indexed)")
    else:
        print("FAIL: Frame 0 returned data (should be None)")
        
    # 3. Test Cache Eviction
    print("\n3. Testing Cache Eviction...")
    # Load 10 frames (fill cache)
    for i in range(1, 11):
        loader.get(i)
    
    stats = loader.get_stats()
    print(f"Cache size: {stats['current_cache_size']}")
    if stats['current_cache_size'] == 10:
        print("PASS: Cache filled to max")
    else:
        print(f"FAIL: Cache size {stats['current_cache_size']} != 10")
        
    # Load 11th frame (should evict frame 1)
    loader.get(11)
    if 1 not in loader._cache:
        print("PASS: Frame 1 evicted")
    else:
        print("FAIL: Frame 1 still in cache")
        
    if 11 in loader._cache:
        print("PASS: Frame 11 in cache")
    else:
        print("FAIL: Frame 11 not in cache")
        
    # 4. Test RGB/BGR Consistency
    print("\n4. Testing Color Format...")
    # Compare with load_all_frames
    # Note: load_all_frames returns BGR (raw cv2 read)
    # FrameLoader should also return BGR now
    # Manually read first frame to compare (avoid loading entire video with load_all_frames)
    cap = cv2.VideoCapture(video_path)
    ret, frame1_ref_bgr = cap.read()
    cap.release()
    
    if not ret:
        print("FAIL: Could not read reference frame")
        return

    frame1_loader_bgr = loader.get(1)
    
    if np.array_equal(frame1_ref_bgr, frame1_loader_bgr):
        print("PASS: Color format matches load_all_frames (BGR)")
    else:
        print("FAIL: Color format mismatch")
        
    loader.close()
    print("\nTest Complete")

if __name__ == "__main__":
    test_frameloader()
