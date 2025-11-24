"""
BatchParameters Configuration System

Comprehensive configuration for video processing with support for:
- Multiple AI models (Claude, Gemini, OpenAI)
- Multiple CV models (YOLO variants)
- Distributed processing
- Incremental/checkpoint processing
- Model version tracking
- Extensible analysis methods
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum
import json
from pathlib import Path
import datetime


# ==========================================
# ENUMS FOR TYPE SAFETY
# ==========================================

class LLMProvider(str, Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"
    OPENAI = "openai"
    CUSTOM = "custom"


class CVModel(str, Enum):
    YOLO_CURRENT = "yolo_current"
    YOLO_V8 = "yolo_v8"
    YOLO_V9 = "yolo_v9"
    CUSTOM_CV = "custom_cv"


class ToolDetectionMethod(str, Enum):
    LLM_DIRECT = "llm_direct"
    LLM_WITH_CONTEXT = "llm_with_context"
    CV_INFERENCE = "cv_inference"
    HYBRID = "hybrid"


class ActionClassificationMethod(str, Enum):
    LLM_MULTIFRAME = "llm_multiframe"
    LLM_SINGLEFRAME = "llm_singleframe"
    CV_BASED = "cv_based"
    HYBRID = "hybrid"


class PromptTemplate(str, Enum):
    STANDARD = "standard"
    DETAILED = "detailed"
    MINIMAL = "minimal"
    CUSTOM = "custom"


class ProductivityAnalysisFormat(str, Enum):
    NONE = "none"
    HTML = "html"
    IMAGES = "images"
    BOTH = "both"


class StorageBackend(str, Enum):
    """Challenge 5: Distributed Processing"""
    LOCAL = "local"
    SHARED_DISK = "shared_disk"
    REDIS = "redis"
    S3 = "s3"


# ==========================================
# BATCH PARAMETERS
# ==========================================

@dataclass
class BatchParameters:
    """
    Comprehensive configuration for video processing pipeline.
    
    Supports 40+ parameters across 9 categories:
    1. AI Models (LLM + CV)
    2. Prompt Engineering
    3. Analysis Methods
    4. Processing Parameters
    5. Feature Toggles
    6. Output Options
    7. Directories
    8. Performance & Distribution
    9. Metadata & Versioning
    """
    
    # ==========================================
    # 1. AI MODEL CONFIGURATION
    # ==========================================
    
    # LLM Configuration
    llm_provider: LLMProvider = LLMProvider.GEMINI  # Changed to Gemini (working API)
    llm_model: str = "gemini-2.0-flash-exp"
    llm_api_key: Optional[str] = None
    llm_max_tokens: int = 1000
    llm_temperature: float = 0.0
    
    # Computer Vision Configuration
    cv_model: CVModel = CVModel.YOLO_CURRENT
    cv_model_path: str = "weights.pt"
    cv_confidence_threshold: float = 0.60
    
    # ==========================================
    # 2. PROMPT ENGINEERING
    # ==========================================
    
    prompt_template: PromptTemplate = PromptTemplate.STANDARD
    custom_prompt_path: Optional[str] = None
    frame_encoding_width: int = 640
    frame_encoding_height: int = 360
    frame_encoding_quality: int = 85
    include_motion_score: bool = True
    include_object_list: bool = True
    include_object_confidence: bool = True
    max_objects_in_prompt: int = 5
    
    # ==========================================
    # 3. ANALYSIS METHODS
    # ==========================================
    
    action_classification_method: ActionClassificationMethod = ActionClassificationMethod.LLM_MULTIFRAME
    allowed_actions: List[str] = field(default_factory=lambda: ["using tool", "idle", "moving"])
    tool_detection_method: ToolDetectionMethod = ToolDetectionMethod.CV_INFERENCE
    allowed_tools: List[str] = field(default_factory=lambda: ["pencil", "saw", "measuring tape", "caulk gun", "unknown"])
    enable_relationship_tracking: bool = True
    proximity_threshold_percent: float = 0.18
    
    # ==========================================
    # 4. PROCESSING PARAMETERS
    # ==========================================
    
    num_frames_per_interval: int = 5
    include_neighbor_frames: bool = True
    cv_detection_frequency: int = 5
    enable_temporal_smoothing: bool = True
    temporal_smoothing_window: int = 9
    motion_score_threshold_idle: float = 0.16
    motion_ignore_threshold: int = 5
    
    # ==========================================
    # 5. FEATURE TOGGLES
    # ==========================================
    
    enable_object_detection: bool = True
    enable_tool_detection: bool = True
    enable_action_classification: bool = True
    enable_productivity_analysis: bool = True
    
    # ==========================================
    # 6. OUTPUT OPTIONS
    # ==========================================
    
    generate_labeled_video: bool = True
    video_codec: str = "mp4v"
    draw_bounding_boxes: bool = True
    draw_relationship_lines: bool = True
    draw_action_labels: bool = True
    overlay_font_size: int = 36
    save_actions_csv: bool = True
    save_relationships_csv: bool = True
    productivity_analysis_format: ProductivityAnalysisFormat = ProductivityAnalysisFormat.IMAGES
    
    # ==========================================
    # 7. DIRECTORIES
    # ==========================================
    
    video_directory: str = "videos/"
    keyframes_directory: str = "keyframes/"  # Actual folders are keyframes/keyframesvideo_XX
    video_output_directory: str = "outputs/vid_objs/"
    csv_directory: str = "outputs/data/"
    analysis_output_directory: str = "outputs/analysis/"
    
    # ==========================================
    # 8. PERFORMANCE & DISTRIBUTION
    # ==========================================
    
    max_workers_keyframes: int = 8
    max_workers_intervals: int = 4
    preload_all_frames: bool = True
    
    # API Rate Limiting
    enable_rate_limiting: bool = True  # Enable API rate limiting
    api_requests_per_minute: int = 10  # Max API requests per minute (adjust based on your tier)
    rate_limit_buffer: float = 0.1  # Safety buffer (10% slower than limit)
    
    # API Request Batching (NEW - Better than rate limiting!)
    # Groups multiple API requests into batches to reduce total API calls
    enable_batch_processing: bool = True  # Enable API request batching
    batch_size: int = 5  # Number of API requests per batch
    use_smart_batching: bool = True  # Intelligently group similar requests
    
    # Challenge 5: Distributed Processing
    storage_backend: StorageBackend = StorageBackend.LOCAL
    shared_storage_path: Optional[str] = None
    
    # Challenge 6: Incremental Processing
    enable_checkpointing: bool = False
    checkpoint_interval: int = 100  # frames
    checkpoint_directory: str = "checkpoints/"
    resume_from_checkpoint: bool = False
    checkpoint_file: Optional[str] = None
    
    # ==========================================
    # 9. METADATA & VERSIONING
    # ==========================================
    
    # Challenge 7: Enhanced Batch Tracking
    batch_id: Optional[str] = None  # Auto-generated if None
    config_name: str = "default"
    config_description: str = ""
    experiment_id: Optional[str] = None
    track_model_versions: bool = True
    save_config_with_outputs: bool = True
    
    # Batch tracking directory
    batch_tracking_directory: str = "outputs/batch_tracking/"
    
    # Video selection
    videos_to_process: Optional[List[str]] = None  # If None, process all in video_directory
    
    # Auto-populated metadata
    created_at: Optional[str] = None
    created_by: Optional[str] = None
    
    # ==========================================
    # METHODS
    # ==========================================
    
    def __post_init__(self):
        """Auto-populate metadata on creation"""
        import uuid
        from pathlib import Path
        
        if self.created_at is None:
            self.created_at = datetime.datetime.now().isoformat()
        
        # Auto-generate batch_id if not provided
        if self.batch_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            self.batch_id = f"batch_{timestamp}_{unique_id}"
        
        # Create batch tracking directory
        Path(self.batch_tracking_directory).mkdir(parents=True, exist_ok=True)
    
    def save_batch_config(self):
        """
        Save this batch configuration to tracking directory.
        Creates a JSON file with batch_id as filename for easy lookup.
        """
        from pathlib import Path
        
        config_path = Path(self.batch_tracking_directory) / f"{self.batch_id}.json"
        self.to_json(str(config_path))
        return str(config_path)
    
    def to_json(self, filepath: str):
        """Save configuration to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Convert string enums back to enum objects
        if 'llm_provider' in data and isinstance(data['llm_provider'], str):
            data['llm_provider'] = LLMProvider(data['llm_provider'])
        if 'cv_model' in data and isinstance(data['cv_model'], str):
            data['cv_model'] = CVModel(data['cv_model'])
        if 'tool_detection_method' in data and isinstance(data['tool_detection_method'], str):
            data['tool_detection_method'] = ToolDetectionMethod(data['tool_detection_method'])
        if 'action_classification_method' in data and isinstance(data['action_classification_method'], str):
            data['action_classification_method'] = ActionClassificationMethod(data['action_classification_method'])
        if 'prompt_template' in data and isinstance(data['prompt_template'], str):
            data['prompt_template'] = PromptTemplate(data['prompt_template'])
        if 'productivity_analysis_format' in data and isinstance(data['productivity_analysis_format'], str):
            data['productivity_analysis_format'] = ProductivityAnalysisFormat(data['productivity_analysis_format'])
        if 'storage_backend' in data and isinstance(data['storage_backend'], str):
            data['storage_backend'] = StorageBackend(data['storage_backend'])
        return cls(**data)
    
    @classmethod
    def from_batch_id(cls, batch_id: str, tracking_directory: str = "outputs/batch_tracking/"):
        """
        Load configuration from batch_id.
        Useful for reproducing exact configuration from previous run.
        """
        from pathlib import Path
        config_path = Path(tracking_directory) / f"{batch_id}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No configuration found for batch_id: {batch_id}")
        return cls.from_json(str(config_path))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def copy(self):
        """Create a copy of this configuration"""
        data = self.to_dict()
        # Remove batch_id so a new one is generated
        data.pop('batch_id', None)
        data.pop('created_at', None)
        return BatchParameters(**data)
    
    def validate(self):
        """Validate configuration"""
        if self.cv_confidence_threshold < 0 or self.cv_confidence_threshold > 1:
            raise ValueError("cv_confidence_threshold must be between 0 and 1")
        if self.temporal_smoothing_window < 1:
            raise ValueError("temporal_smoothing_window must be >= 1")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be >= 1")
        if self.resume_from_checkpoint and not self.checkpoint_file:
            raise ValueError("checkpoint_file must be specified when resume_from_checkpoint=True")
    
    def get_model_versions(self) -> Dict[str, str]:
        """
        Challenge 7: Get model versions for tracking
        Returns dict of model_type -> version
        """
        return {
            "batch_id": self.batch_id,
            "llm_provider": self.llm_provider.value,
            "llm_model": self.llm_model,
            "cv_model": self.cv_model.value,
            "cv_model_path": self.cv_model_path,
            "config_name": self.config_name,
            "experiment_id": self.experiment_id or "none",
            "created_at": self.created_at
        }
    
    def get_videos_to_process(self) -> List[str]:
        """
        Get list of videos to process.
        If videos_to_process is specified, use that.
        Otherwise, scan video_directory for all .mp4 files.
        """
        import os
        
        if self.videos_to_process is not None:
            return self.videos_to_process
        
        # Scan directory for all videos
        video_files = []
        if os.path.exists(self.video_directory):
            for f in os.listdir(self.video_directory):
                if f.endswith('.mp4'):
                    video_files.append(f.replace('.mp4', ''))
        
        return sorted(video_files)
    
    def get_comparison_key(self, params_to_compare: List[str]) -> str:
        """
        Generate a key for comparing batches based on specific parameters.
        
        Example:
            key = batch_params.get_comparison_key(['cv_model', 'llm_provider'])
            # Returns: "cv_model=yolo_current|llm_provider=claude"
        
        Useful for grouping batches that differ only in certain parameters.
        """
        parts = []
        data = self.to_dict()
        for param in sorted(params_to_compare):
            if param in data:
                value = data[param]
                # Convert enums to string
                if hasattr(value, 'value'):
                    value = value.value
                parts.append(f"{param}={value}")
        return "|".join(parts)


# ==========================================
# PRESET CONFIGURATIONS
# ==========================================

# Equivalent to TestingClass.py
PRESET_BASIC = BatchParameters(
    config_name="basic",
    config_description="Basic action classification (TestingClass.py equivalent)",
    enable_object_detection=False,
    enable_relationship_tracking=False,
    enable_productivity_analysis=False,
    tool_detection_method=ToolDetectionMethod.LLM_DIRECT,
    action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME
)

# Equivalent to TestingClass2.py
PRESET_OBJECTS = BatchParameters(
    config_name="objects",
    config_description="With YOLO object detection (TestingClass2.py equivalent)",
    enable_object_detection=True,
    enable_relationship_tracking=False,
    enable_productivity_analysis=False,
    tool_detection_method=ToolDetectionMethod.CV_INFERENCE,
    action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME
)

# Equivalent to TestingClass3.py
PRESET_RELATIONSHIPS = BatchParameters(
    config_name="relationships",
    config_description="With relationship tracking (TestingClass3.py equivalent)",
    enable_object_detection=True,
    enable_relationship_tracking=True,
    enable_productivity_analysis=False,
    tool_detection_method=ToolDetectionMethod.CV_INFERENCE,
    action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME
)

# Equivalent to TestingClassIntegrated.py
PRESET_HTML_ANALYSIS = BatchParameters(
    config_name="html_analysis",
    config_description="With HTML productivity reports (TestingClassIntegrated.py equivalent)",
    enable_object_detection=True,
    enable_relationship_tracking=True,
    enable_productivity_analysis=True,
    productivity_analysis_format=ProductivityAnalysisFormat.HTML,
    tool_detection_method=ToolDetectionMethod.CV_INFERENCE,
    action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME
)

# Equivalent to TestingClassFINAL.py
PRESET_FULL = BatchParameters(
    config_name="full",
    config_description="Full analysis with static charts (TestingClassFINAL.py equivalent)",
    enable_object_detection=True,
    enable_relationship_tracking=True,
    enable_productivity_analysis=True,
    productivity_analysis_format=ProductivityAnalysisFormat.IMAGES,
    tool_detection_method=ToolDetectionMethod.CV_INFERENCE,
    action_classification_method=ActionClassificationMethod.LLM_MULTIFRAME
)
