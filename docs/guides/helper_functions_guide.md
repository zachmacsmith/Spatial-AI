# Helper Functions & Utilities Guide

This guide documents the core utility functions and helper classes that power the video processing pipeline.

## 🛠️ Video Utilities (`video_processing/utils/video_utils.py`)

These functions handle low-level video operations, frame loading, and motion analysis.

### `FrameLoader` (Class)
A memory-efficient, thread-safe frame loader with LRU caching.
*   **`__init__(video_path, max_frames=500)`**: Opens video and initializes cache.
*   **`get(frame_number)`**: Returns a frame (1-indexed) from cache or disk. Returns `None` if invalid.
*   **`iter_frames(start, end)`**: Efficiently yields frames sequentially without polluting the cache. Best for video generation.
*   **`preload(frame_numbers)`**: Pre-fetches specific frames (e.g., keyframes) into the cache.

### `load_all_frames(video_path)`
*   **Returns:** `Dict[int, np.ndarray]`
*   **Purpose:** Loads the *entire* video into memory.
*   **Warning:** High memory usage. Use `FrameLoader` for large videos.

### `calculate_motion_score(frames, ignore_threshold=5)`
*   **Returns:** `float` (0.0 - 1.0)
*   **Purpose:** Computes the average pixel difference between consecutive frames.
*   **Logic:**
    1.  Converts frames to grayscale.
    2.  Computes absolute difference.
    3.  Ignores pixels with difference < `ignore_threshold` (noise filtering).
    4.  Normalizes to 0-1 range.
*   **Usage:** Used by `state_check_motion_threshold` to detect "idle" vs "moving".

### `extract_keyframes(video_path, output_dir, ...)`
*   **Purpose:** Uses adaptive thresholding to extract distinct keyframes from a video.
*   **Logic:** Saves a frame if it differs significantly from the previous keyframe (based on mean + std dev of differences).

---

## 🎨 Visualization (`video_processing/utils/visualization.py`)

Functions for drawing overlays, bounding boxes, and relationship lines on frames.

### `create_visualization(frame, ...)`
The main entry point for rendering.
*   **Args:** `frame`, `action_label`, `detections`, `yolo_results`, `relationship_lines`, `batch_params`.
*   **Purpose:** Applies all enabled visualizations based on `BatchParameters`.

### `overlay_action_label(frame, label, ...)`
*   **Purpose:** Draws the action text (e.g., "using drill") in the top-left corner with a high-contrast outline.

### `draw_bounding_boxes(frame, detections, ...)`
*   **Purpose:** Draws bounding boxes for detected objects.
*   **Features:** Uses consistent colors for each class using golden ratio hashing.

### `draw_relationship_lines(frame, line_info, ...)`
*   **Purpose:** Draws lines connecting related objects (e.g., Hand -> Tool).

---

## 🧠 Decision Functions (`video_processing/decision_functions.py`)

These functions implement the core logic for the "Pluggable Decision System". They are registered via decorators and selected via `BatchParameters`.

### State Checks (`@register_state_check`)
Determine the worker's state (`idle`, `moving`, `using tool`).
*   **`motion_threshold`**: Uses `motion_score` < 0.16 for idle. Checks for tool detections for "using tool". (0 API calls)
*   **`llm_direct`**: Asks the LLM directly. (1 API call)
*   **`hybrid_motion_then_llm`**: Uses motion for obvious cases, falls back to LLM for ambiguous ones. (0-1 API calls)
*   **`legacy_testing_class`**: Replicates the exact prompt and logic from the original `TestingClass.py`.

### Object Checks (`@register_object_check`)
Identify the specific tool being used.
*   **`cv_detection`**: Returns the highest-confidence object detected by YOLO. (0 API calls)
*   **`llm_direct`**: Asks the LLM "What tool is being used?". (1 API call)
*   **`cv_then_llm`**: Uses YOLO result if available, otherwise asks LLM. (0-1 API calls)
*   **`llm_with_cv_hint`**: Asks LLM but provides YOLO detections as a "hint" in the prompt. (1 API call)

---

## ⚙️ Processing Strategies (`video_processing/processing_strategies.py`)

High-level strategies for processing the video timeline.

### `strategy_classify_all`
*   **Logic:** Classifies *every* defined interval (e.g., every 1 second).
*   **Best For:** Maximum temporal resolution.

### `strategy_keyframes_only`
*   **Logic:** Only classifies the extracted keyframes.
*   **Best For:** Quick summaries.

### `strategy_smart`
*   **Logic:**
    1.  Classifies keyframes first.
    2.  Checks intervals between keyframes.
    3.  If motion is low, interpolates the previous state.
    4.  If motion is high, triggers a new classification.
*   **Best For:** Efficiency (skips redundant processing of static scenes).
