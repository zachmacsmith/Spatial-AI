# Video Processing Presets Guide

This guide explains the available presets for the video processing pipeline. These presets allow you to quickly switch between different configurations for A/B testing, cost optimization, and model comparison.

## 🎯 Decision System Presets (New)

These presets utilize the new pluggable decision system to control how actions and tools are classified.

### 1. Baseline (Single Shot)
*   **Preset ID:** `baseline`
*   **Description:** The "standard" approach. Sends a single prompt to the LLM for every interval to classify action and tool simultaneously.
*   **Model:** Claude Sonnet 4.5
*   **Pros:** High accuracy, simple logic.
*   **Cons:** Higher API cost (sends images for every interval), no early exit.
*   **Best For:** Establishing a ground truth for accuracy comparison.

### 2. Cheap (CV + Motion)
*   **Preset ID:** `cheap`
*   **Description:** A cost-minimized approach that avoids LLM calls entirely.
*   **Logic:**
    *   **State:** Uses Motion Score (if < 0.16 -> idle) and CV detections (if tool present -> using tool).
    *   **Tool:** Uses YOLO detection results directly.
*   **Model:** Claude Sonnet 4.5 (configured but unused by default logic).
*   **Pros:** Extremely fast, zero LLM cost.
*   **Cons:** Lower accuracy for complex actions; cannot identify tools not in the YOLO class list.
*   **Best For:** Processing large volumes of video where approximate results are acceptable.

### 3. Balanced (Hybrid)
*   **Preset ID:** `balanced`
*   **Description:** A smart trade-off between cost and accuracy.
*   **Logic:**
    *   **State:** Checks Motion/CV first. If motion is very high/low or tools are clearly visible, it decides immediately. Only calls LLM for ambiguous cases.
    *   **Tool:** Checks CV first. If a tool is detected, it accepts it. Only calls LLM if no tool is detected by CV.
*   **Model:** Claude Sonnet 4.5
*   **Pros:** Significantly cheaper than Baseline while maintaining high accuracy for clear scenarios.
*   **Cons:** Slightly more complex logic path.
*   **Best For:** General production usage.

### 4. Thorough (LLM Heavy)
*   **Preset ID:** `thorough`
*   **Description:** Maximizes reasoning capability by using the LLM for every step, with CV as a "hint".
*   **Logic:**
    *   **State:** Asks LLM directly.
    *   **Tool:** Asks LLM, providing the CV detection list as a hint ("Object detection suggests: drill").
*   **Model:** Claude Sonnet 4.5
*   **Pros:** Highest potential accuracy; leverages CV to reduce LLM hallucinations without being limited by it.
*   **Cons:** Highest cost and latency (multiple API calls per interval).
*   **Best For:** Complex scenes where CV often fails or context is crucial.

### 5. Legacy (TestingClass.py)
*   **Preset ID:** `legacy`
*   **Description:** An exact replication of the original `TestingClass.py` logic.
*   **Logic:** Uses the specific legacy prompt format (*"Motion score: X... 0-0.16 suggests idle..."*) and specific threshold logic.
*   **Model:** Claude Sonnet 4.5
*   **Best For:** Reproducing historical results.

---

## 🤖 Model-Specific Presets

These presets are designed for comparing different LLM providers and models.

### 6. Anthropic Relationships
*   **Preset ID:** `anthropic_relationships`
*   **Model:** Claude 3.5 Sonnet
*   **Features:** Enabled relationship tracking and object detection.
*   **Best For:** High-quality relationship analysis.

### 7. Anthropic Haiku
*   **Preset ID:** `anthropic_haiku`
*   **Model:** Claude 3 Haiku
*   **Features:** Faster and cheaper than Sonnet.
*   **Best For:** Cost-sensitive relationship tracking.

### 8. Gemini Relationships
*   **Preset ID:** `gemini_relationships`
*   **Model:** Gemini 2.0 Flash Lite
*   **Features:** Standard Gemini configuration.
*   **Best For:** Testing Google's latest lightweight model.

### 9. Gemini Flash
*   **Preset ID:** `gemini_flash`
*   **Model:** Gemini 1.5 Flash
*   **Features:** Previous generation fast model.
*   **Best For:** Comparison with 2.0 Flash Lite.

---

## 🧪 Legacy Testing Presets

These presets replicate the configurations from the original `TestingClass*.py` files.

### 10. Basic (Action Only)
*   **Preset ID:** `basic`
*   **Equivalent:** `TestingClass.py`
*   **Features:** Action classification only. No object detection or relationships.

### 11. Objects (YOLO)
*   **Preset ID:** `objects`
*   **Equivalent:** `TestingClass2.py`
*   **Features:** Adds YOLO object detection.

### 12. Full Analysis
*   **Preset ID:** `full_analysis`
*   **Equivalent:** `TestingClassFINAL.py`
*   **Features:** Full suite including static charts and productivity analysis.

---

## 🛠️ Specialized & New Presets

### 13. Legacy + Temporal Majority
*   **Preset ID:** `legacy_temporal`
*   **Description:** Replicates the original logic but uses temporal majority voting to identify unknown objects.
*   **Best For:** General purpose baseline with improved stability.

### 14. Enhanced Temporal
*   **Preset ID:** `enhanced_temporal`
*   **Description:** Uses CV-first logic with temporal majority voting and smart classification.
*   **Best For:** Efficiency and stability.

### 15. Strict Enhanced
*   **Preset ID:** `strict_enhanced`
*   **Description:** Enhanced temporal logic with strict prompting constraints (no reasoning, exact option match only).
*   **Best For:** Reducing hallucinations.

### 16. Relationship Aware
*   **Preset ID:** `relationship_aware`
*   **Description:** Uses LLM with relationship context for object identification.
*   **Best For:** Complex scenes where object interaction matters.

### 17. Power Saw Strict (Haiku)
*   **Preset ID:** `power_saw_haiku`
*   **Description:** Strict object detection with aggregation, optimized for power saws using Claude 3 Haiku.
*   **Features:** "Power saw" in tool list, softened uncertainty prompt, aggregation.

### 18. Power Saw Recheck (Haiku)
*   **Preset ID:** `power_saw_recheck_haiku`
*   **Description:** Specialized for power saws. Uses a two-step recheck process:
    1.  Strict initial check on current frame.
    2.  If "unknown", triggers a recheck using 2 seconds of aggregated object identification and multiframe visual context (10 frames).
*   **Best For:** High accuracy on difficult-to-detect tools like power saws.

### 19. Raw LLM (Sonnet)
*   **Preset ID:** `raw_llm_sonnet`
*   **Description:** Direct LLM calls for state and object check, bypassing most heuristics.
*   **Best For:** Testing raw model capabilities.

### 20. Dense Interval Aggregation
*   **Preset ID:** `dense_interval_aggregation`
*   **Description:** Uses dense keyframe sampling (every 5 frames) for high-resolution interval aggregation.
*   **Best For:** Capturing rapid changes.
