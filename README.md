# Video Processing System

A modular video analysis system for construction site videos with AI-powered action classification, object detection, and relationship tracking.

---

## Quick Start

### 1. Setup Environment

```bash
conda env create -f environment.yml
conda activate egoenv
```

### 2. Configure API Keys

Create `config.py` in the root directory:

```python
GEMINI_API_KEY = "your-gemini-api-key"
ANTHROPIC_API_KEY = "your-claude-api-key"  # Optional
OPENAI_API_KEY = "your-openai-api-key"     # Optional
```

### 3. Run Processing

The system now uses a unified runner that supports presets.

**Interactive Runner:**

```bash
python run_batch.py
```

This will launch a menu where you can:
1. Select a **Preset** (e.g., "Legacy + Temporal Majority")
2. Toggle **Video Generation** (Enable for visualization, Disable for speed)
3. Select **Videos** to process

### 4. Benchmark Results

To evaluate accuracy against ground truth:

```bash
python benchmark_existing.py
```

This tool will:
1. List all processed batches
2. Calculate accuracy metrics (State, Object, Guess)
3. Generate accuracy charts
4. Output detailed CSV reports to `benchmark_results/`

---

## Configuration & Customization

The system is designed to be highly extensible. You can customize:
1.  **Batch Parameters**: Over 40 options to control every aspect of processing.
2.  **Presets**: Create your own reusable configurations.
3.  **Decision Logic**: Plug in your own Python functions for state/object classification.
4.  **Processing Strategies**: Define custom timeline processing logic.

### 📚 Guides

*   **[Extensibility Guide](docs/guides/EXTENSIBILITY_GUIDE.md)**: The complete reference for customizing the system. Covers Batch Parameters, Custom Presets, and Custom Decision Functions.
*   **[Presets Guide](docs/guides/presets_guide.md)**: Detailed explanation of included presets.
*   **[Helper Functions Guide](docs/guides/helper_functions_guide.md)**: Reference for utility functions.

### 🚀 Example Usage

Check `example_usage.py` for practical code examples:

```bash
python example_usage.py
```

It demonstrates:
*   Using standard presets
*   Creating custom configurations programmatically
*   Batch processing multiple videos
*   Comparing different CV models

---

## Available Presets

The system comes with several tuned presets:

- **Legacy + Temporal Majority** (`legacy_temporal`): **(Recommended)** Replicates the original logic but uses temporal majority voting to identify unknown objects.
- **Legacy** (`legacy`): Exact replication of the original `TestingClass.py` logic.
- **Balanced** (`balanced`): Hybrid approach using Motion/CV first, falling back to LLM. Good trade-off.
- **Thorough** (`thorough`): Maximum accuracy using LLM for all decisions. High cost.
- **Cheap** (`cheap`): Lowest cost using only Motion Threshold and CV. No LLM costs.

---

## Output Structure

All outputs are organized by batch ID:

```
outputs/
├── data/
│   └── batch_20251128_.../
│       ├── video_01.csv                  # Action classifications
│       ├── video_01_relationships.csv    # Object relationships
│       └── video_01_metadata.json        # Metadata & Performance stats
├── vid_objs/
│   └── batch_20251128_.../
│       └── video_01.mp4                  # Labeled video
└── batch_tracking/
    └── batch_20251128_...json            # Batch configuration
```

---

## Support

For questions or issues, check:
1. `presets/` - Example configurations
2. `docs/guides/` - Detailed guides
3. `video_processing/batch_parameters.py` - All configuration options

---

## Quick Start

### 1. Setup Environment

```bash
conda env create -f environment.yml
conda activate egoenv
```

### 2. Configure API Keys

Create `config.py` in the root directory:

```python
GEMINI_API_KEY = "your-gemini-api-key"
ANTHROPIC_API_KEY = "your-claude-api-key"  # Optional
OPENAI_API_KEY = "your-openai-api-key"     # Optional
```

### 3. Run Processing

**Option A: Use Presets (Recommended)**

```python
from video_processing import process_video, PRESET_FULL

outputs = process_video("video_01", PRESET_FULL)
print(f"Results saved to: {outputs['actions_csv']}")
```

**Option B: Use Batch Processor (Interactive)**

```bash
python batch_process.py
```

---

## System Architecture

### Core Components

```
video_processing/          # Main processing pipeline
├── ai/                    # LLM and CV model interfaces
│   ├── llm_service.py    # Claude, Gemini, OpenAI support
│   ├── cv_service.py     # YOLO object detection
│   └── prompt_builder.py # Prompt engineering
├── analysis/              # Analysis methods
│   ├── action_classifier.py      # Action classification
│   ├── tool_detector.py          # Tool detection
│   └── relationship_tracker.py   # Object relationships
├── utils/                 # Utilities
│   ├── video_utils.py    # Video I/O, frame handling
│   └── visualization.py  # Overlay generation
└── output/                # Output management
    └── output_manager.py # CSV/metadata generation

post_processing/           # Analysis tools
├── accuracy_benchmark.py  # Accuracy evaluation
├── performance_benchmark.py # Performance analysis
└── model_comparison.py    # Model comparison
```

### Data Flow

```
Input Video → Keyframe Extraction → Action Classification → Output CSVs
                                  ↓
                            Object Detection (optional)
                                  ↓
                         Relationship Tracking (optional)
                                  ↓
                          Labeled Video (optional)
```

---

## Configuration System

All processing is controlled via `BatchParameters`. Use presets or customize:

### Available Presets

```python
from video_processing import (
    PRESET_BASIC,           # Action classification only
    PRESET_OBJECTS,         # + Object detection
    PRESET_RELATIONSHIPS,   # + Relationship tracking
    PRESET_HTML_ANALYSIS,   # + HTML reports
    PRESET_FULL             # All features enabled
)
```

### Custom Configuration

```python
from video_processing import BatchParameters

params = BatchParameters(
    # AI Models
    llm_provider="gemini",              # claude, gemini, or openai
    llm_model="gemini-2.0-flash-exp",
    cv_model="yolo_current",
    
    # Features
    enable_object_detection=True,
    enable_relationship_tracking=True,
    enable_action_classification=True,
    
    # Processing
    num_frames_per_interval=5,
    enable_temporal_smoothing=True,
    
    # Output
    generate_labeled_video=True,
    save_actions_csv=True,
    save_relationships_csv=True
)

outputs = process_video("video_01", params)
```

### Key Parameters

**AI Configuration:**
- `llm_provider` - LLM provider (gemini, claude, openai)
- `llm_model` - Model version
- `cv_model` - YOLO variant (yolo_current, yolo_v8, yolo_v9)

**Analysis Methods:**
- `action_classification_method` - How to classify actions (llm_multiframe, llm_singleframe, cv_based, hybrid)
- `tool_detection_method` - How to detect tools (llm_direct, llm_with_context, cv_inference, hybrid)

**Performance:**
- `enable_batch_processing` - Group API requests for efficiency (default: True)
- `api_requests_per_minute` - Rate limit based on your API tier
- `max_workers_keyframes` - Parallel processing threads

**Output:**
- `csv_directory` - Where to save CSVs (default: `outputs/data/`)
- `video_output_directory` - Where to save videos (default: `outputs/vid_objs/`)

See `video_processing/batch_parameters.py` for all 40+ parameters.

---

## Output Structure

All outputs are organized by batch ID to prevent overwrites:

```
outputs/
├── data/
│   └── batch_20251124_015521_abc123/
│       ├── video_01.csv                  # Action classifications
│       ├── video_01_relationships.csv    # Object relationships
│       └── video_01_metadata.json        # Processing metadata
├── vid_objs/
│   └── batch_20251124_015521_abc123/
│       └── video_01.mp4                  # Labeled video
└── batch_tracking/
    └── batch_20251124_015521_abc123.json # Batch configuration
```

### Actions CSV Format

```csv
fps,total_duration,batch_id
23.98,21.02,batch_20251124_015521_abc123
1,using tool
450,idle
890,moving
```

### Relationships CSV Format

```csv
batch_id,batch_20251124_015521_abc123
start_frame,end_frame,start_time,end_time,duration,objects
100,250,4.17,10.42,6.25,hammer,nail
```

---

## Common Tasks

### Compare Different Models

```python
from video_processing import BatchParameters, CVModel

for cv_model in [CVModel.YOLO_V8, CVModel.YOLO_V9]:
    params = BatchParameters(
        cv_model=cv_model,
        experiment_id="cv_comparison"
    )
    outputs = process_video("video_01", params)
```

### Process Multiple Videos

```python
params = PRESET_FULL
videos = ["video_01", "video_02", "video_03"]

for video in videos:
    outputs = process_video(video, params)
    print(f"Completed: {video}")
```

### Reproduce Previous Run

```python
from video_processing import BatchParameters

# Load exact configuration from previous run
params = BatchParameters.from_batch_id("batch_20251124_015521_abc123")
outputs = process_video("video_01", params)
```

### Benchmark Accuracy

```python
from post_processing.accuracy_benchmark import run_benchmark

results, charts = run_benchmark(
    batch_id="batch_20251124_015521_abc123",
    model_data_dir="ModelData"
)
```

---

## Directory Structure

```
CompleteModel_Agentic/
├── README.md                  # This file
├── config.py                  # API keys (create this)
├── batch_process.py           # Interactive batch processor
├── example_usage.py           # Usage examples
│
├── video_processing/          # Core processing code
├── post_processing/           # Analysis tools
├── shared/                    # Shared utilities
│
├── docs/                      # Documentation
│   ├── examples/              # Utility scripts
│   ├── guides/                # How-to guides
│   └── testing/               # Test scripts
│
├── videos/                    # Input videos
├── keyframes/                 # Generated keyframes
├── TestData/                  # Ground truth (for benchmarking)
└── outputs/                   # All outputs (organized by batch_id)
```

---

## API Rate Limits

Configure based on your API tier:

```python
# Free Tier (Conservative)
params = BatchParameters(
    api_requests_per_minute=10,
    enable_batch_processing=True,  # Groups requests
    batch_size=10
)

# Paid Tier
params = BatchParameters(
    api_requests_per_minute=60,
    enable_batch_processing=True,
    batch_size=5
)
```

**API Request Batching** (enabled by default) groups multiple requests together to reduce total API calls and avoid rate limits.

---

## Extending the System

The system uses a registry pattern for easy extension. See `docs/guides/EXTENSIBILITY_GUIDE.md` for details.

**Add custom action classifier:**

```python
from video_processing.analysis.action_classifier import register_action_classifier

@register_action_classifier("my_method")
def my_classifier(frames, batch_params, llm_service, prompt_builder, **kwargs):
    # Your logic here
    return "action_label"

# Use it
params = BatchParameters(action_classification_method="my_method")
```

**Add custom prompt template:**

```python
params = BatchParameters(
    prompt_template="custom",
    custom_prompt_path="my_prompts/template.txt"
)
```

---

## Troubleshooting

### Rate Limit Errors

Wait 60 seconds, then reduce `api_requests_per_minute`:

```python
params.api_requests_per_minute = 5
```

### Frame Cache Warnings

```
⚠ Video metadata mismatch: reported 569 frames, actually loaded 504 frames
```

This is normal for some videos. The system handles it automatically.

### Import Errors

Ensure you're in the project directory:

```bash
cd /path/to/CompleteModel_Agentic
python example_usage.py
```

---

## Testing

```bash
# Run all tests
python docs/testing/test_comprehensive.py

# Test specific preset
python -c "from video_processing import process_video, PRESET_BASIC; process_video('video_01', PRESET_BASIC)"
```

---

## Documentation

- `docs/guides/EXTENSIBILITY_GUIDE.md` - How to extend the system
- `docs/guides/API_REQUEST_BATCHING.md` - API batching details
- `docs/guides/RATE_LIMITING.md` - API rate limit reference
- `example_usage.py` - Code examples

---

## Supported Objects

The system's YOLO model (`weights.pt`) is trained to detect the following 11 classes:

| Class ID | Object Name |
|----------|-------------|
| 0 | brick |
| 1 | brick trowel |
| 2 | caulk gun |
| 3 | drill |
| 4 | hand |
| 5 | hardhat |
| 6 | pencil |
| 7 | person |
| 8 | safety vest |
| 9 | saw |
| 10 | tape |

---

## Support

For questions or issues, check:
1. `example_usage.py` - Working examples
2. `docs/guides/` - Detailed guides
3. `video_processing/batch_parameters.py` - All configuration options
