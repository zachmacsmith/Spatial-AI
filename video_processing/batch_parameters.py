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


class ContextStrategy(str, Enum):
    """Strategy for building prompt context"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    TEMPORAL = "temporal"
    FULL = "full"


class ProcessingStrategy(str, Enum):
    """Strategy for processing video frames"""
    CLASSIFY_ALL = "classify_all"
    KEYFRAMES_ONLY = "keyframes_only"
    SMART = "smart"
    KEYFRAMES_SAMPLED = "keyframes_sampled"


class PromptingProtocolType(str, Enum):
    """How to structure the classification query"""
    SINGLE_SHOT = "single_shot"
    CASCADE = "cascade"


class StateCheckMethod(str, Enum):
    """How to determine action state"""
    MOTION_THRESHOLD = "motion_threshold"
    LLM_DIRECT = "llm_direct"
    LEGACY_TESTING_CLASS = "legacy_testing_class"
    HYBRID_MOTION_THEN_LLM = "hybrid_motion_then_llm"
    ACTION_MAPPING = "action_mapping"
    LEGACY_SOFTENED = "legacy_softened"
    LLM_MULTIFRAME = "llm_multiframe"
    CV_OBJECTS_ONLY = "cv_objects_only"
    RANDOM = "random"
    STATIC_USING_TOOL = "static_using_tool"


class ObjectCheckMethod(str, Enum):
    """How to identify tools"""
    CV_DETECTION = "cv_detection"
    LLM_DIRECT = "llm_direct"
    CV_THEN_LLM = "cv_then_llm"
    LLM_WITH_CV_HINT = "llm_with_cv_hint"
    LLM_WITH_RELATIONSHIPS = "llm_with_relationships"
    LLM_STRICT = "llm_strict"
    LLM_STRICT_SOFTENED = "llm_strict_softened"
    LLM_STRICT_CONFIDENT = "llm_strict_confident"
    LLM_TWO_STEP_RECHECK = "llm_two_step_recheck"
    LLM_AGGREGATION_SOFTENED = "llm_aggregation_softened"
    LLM_WITH_INTERVAL_AGGREGATION = "llm_with_interval_aggregation"
    LLM_WITH_1SEC_AGGREGATION = "llm_with_1sec_aggregation"
    LEGACY_TESTING_CLASS = "legacy_testing_class"
    LEGACY_SOFTENED = "legacy_softened"
    ACTION_MAPPING = "action_mapping"
    LLM_MULTIFRAME = "llm_multiframe"
    RANDOM = "random"
    STATIC_COMMON = "static_common"


class UnknownObjectCheckMethod(str, Enum):
    """How to guess unknown tools"""
    LLM_GUESS = "llm_guess"
    LLM_GUESS_WITH_OPTIONS = "llm_guess_with_options"
    CV_CLASS_NAME = "cv_class_name"
    TEMPORAL_MAJORITY = "temporal_majority"
    SKIP = "skip"
    LLM_RECHECK = "llm_recheck"


class ToolDetectionMethod(str, Enum):
    """Legacy - kept for backward compatibility"""
    LLM_DIRECT = "llm_direct"
    LLM_WITH_CONTEXT = "llm_with_context"
    CV_INFERENCE = "cv_inference"
    HYBRID = "hybrid"


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
    llm_model: str = "gemini-2.0-flash-lite"
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
    context_strategy: ContextStrategy = ContextStrategy.STANDARD
    
    # Prompting protocol configuration
    prompting_protocol: PromptingProtocolType = PromptingProtocolType.SINGLE_SHOT
    
    # Decision function configuration (used when prompting_protocol=CASCADE)
    state_check_method: StateCheckMethod = StateCheckMethod.LLM_DIRECT
    object_check_method: ObjectCheckMethod = ObjectCheckMethod.CV_THEN_LLM
    unknown_object_check_method: UnknownObjectCheckMethod = UnknownObjectCheckMethod.LLM_GUESS
    
    # ==========================================
    # 3. ANALYSIS METHODS
    # ==========================================
    
    action_classification_method: ActionClassificationMethod = ActionClassificationMethod.LLM_MULTIFRAME
    allowed_actions: List[str] = field(default_factory=lambda: ["using tool", "idle", "moving"])
    tool_detection_method: ToolDetectionMethod = ToolDetectionMethod.CV_INFERENCE
    allowed_tools: List[str] = field(default_factory=lambda: [
        "brick trowel", "caulk gun", "drill", "pencil", "saw", "brick", "measuring", "hammer", "nail gun", "unknown"
    ])
    enable_relationship_tracking: bool = True
    proximity_threshold_percent: float = 0.18
    
    # ==========================================
    # 4. PROCESSING PARAMETERS
    # ==========================================
    
    num_frames_per_interval: int = 5
    include_neighbor_frames: bool = True
    cv_detection_frequency: int = 0
    yolo_vid_frequency: int = 5  # Frequency of YOLO detection for video
    
    # Temporal Smoothing
    enable_temporal_smoothing: bool = True
    temporal_smoothing_window: int = 9
    
    # Multi-Frame Context
    multi_frame_count: int = 0  # Number of frames to include before/after (0 = single frame)
    multi_frame_gap: int = 1    # Gap between context frames (1 = adjacent)
    motion_score_threshold_idle: float = 0.16
    motion_ignore_threshold: int = 5
    
    # Strategy parameters
    processing_strategy: ProcessingStrategy = ProcessingStrategy.CLASSIFY_ALL
    keyframe_sample_rate: int = 1  # For KEYFRAMES_SAMPLED strategy
    smart_classification_motion_threshold: float = 0.15  # For SMART strategy
    smart_classification_max_gap: int = 50  # Max frames between forced classifications
    
    # Keyframe Extraction Parameters
    keyframe_min_gap: int = 20
    keyframe_max_gap: int = 500  # Force a keyframe every N frames even if no motion
    keyframe_threshold_multiplier: float = 1.0
    
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
    overlay_font_size: int = 200
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
    
    # Frame loading strategy
    preload_all_frames: bool = False  # True: load all frames (high RAM). False: streaming with LRU cache
    frame_cache_size: int = 500       # Max frames in LRU cache when preload_all_frames=False
    preload_keyframes: bool = True    # Preload keyframes into cache for faster access
    
    # API Rate Limiting
    enable_rate_limiting: bool = True  # Enable API rate limiting
    pricing_tier: str = "free"  # "free" or "pay_as_you_go"
    api_requests_per_minute: Optional[int] = None  # None = auto-configure based on model/tier
    rate_limit_buffer: float = 0.1  # Safety buffer (10% slower than limit)
    
    # API Request Batching (NEW - Better than rate limiting!)
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
        """Auto-populate metadata and configure rate limits"""
        import uuid
        import datetime
        import json
        from pathlib import Path
        
        # Auto-configure rate limits if not manually set
        if self.api_requests_per_minute is None:
            self._configure_rate_limit()
        
        if self.created_at is None:
            self.created_at = datetime.datetime.now().isoformat()
        
        # Auto-generate batch_id if not provided
        if self.batch_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            self.batch_id = f"batch_{timestamp}_{unique_id}"
            
        # Create batch tracking directory
        Path(self.batch_tracking_directory).mkdir(parents=True, exist_ok=True)
            
    def _configure_rate_limit(self):
        """Load rate limits from config file based on model and tier"""
        import json
        from pathlib import Path
        try:
            config_path = Path(__file__).parent / 'config' / 'google_model_limits.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    limits = json.load(f)
                
                model_limits = limits.get(self.llm_model, limits.get("default"))
                if model_limits:
                    limit = model_limits.get(self.pricing_tier, 5)
                    self.api_requests_per_minute = limit
                    # print(f"✓ Auto-configured rate limit for {self.llm_model} ({self.pricing_tier}): {limit} RPM")
                else:
                    self.api_requests_per_minute = 10
            else:
                self.api_requests_per_minute = 10
        except Exception as e:
            print(f"⚠ Error loading rate limits: {e}")
            self.api_requests_per_minute = 10
        
    
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
        import json
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str):
        """Load configuration from JSON file"""
        import json
        # Enums are available in module scope
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
        if 'context_strategy' in data and isinstance(data['context_strategy'], str):
            data['context_strategy'] = ContextStrategy(data['context_strategy'])
        if 'processing_strategy' in data and isinstance(data['processing_strategy'], str):
            data['processing_strategy'] = ProcessingStrategy(data['processing_strategy'])
            
        # Handle new Enums
        if 'prompting_protocol' in data and isinstance(data['prompting_protocol'], str):
            data['prompting_protocol'] = PromptingProtocolType(data['prompting_protocol'])
        if 'state_check_method' in data and isinstance(data['state_check_method'], str):
            data['state_check_method'] = StateCheckMethod(data['state_check_method'])
        if 'object_check_method' in data and isinstance(data['object_check_method'], str):
            data['object_check_method'] = ObjectCheckMethod(data['object_check_method'])
        if 'unknown_object_check_method' in data and isinstance(data['unknown_object_check_method'], str):
            data['unknown_object_check_method'] = UnknownObjectCheckMethod(data['unknown_object_check_method'])
            
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
        from dataclasses import asdict
        from typing import Dict, Any
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


# ==========================================
# NEW PRESET CONFIGURATIONS (DECISION SYSTEM)
# ==========================================

# Baseline: Current behavior
PRESET_BASELINE = BatchParameters(
    batch_id="baseline",
    config_name="baseline",
    llm_provider=LLMProvider.CLAUDE,
    llm_model="claude-sonnet-4-5-20250929",
    prompting_protocol=PromptingProtocolType.SINGLE_SHOT,
    # Decision methods not used in single_shot
)

# Cheap: Minimize API calls
PRESET_CHEAP = BatchParameters(
    batch_id="cheap",
    config_name="cheap",
    llm_provider=LLMProvider.CLAUDE,
    llm_model="claude-sonnet-4-5-20250929",
    prompting_protocol=PromptingProtocolType.CASCADE,
    state_check_method=StateCheckMethod.MOTION_THRESHOLD,
    object_check_method=ObjectCheckMethod.CV_DETECTION,
    unknown_object_check_method=UnknownObjectCheckMethod.CV_CLASS_NAME,
)

# Balanced: Hybrid approach
PRESET_BALANCED = BatchParameters(
    batch_id="balanced",
    config_name="balanced",
    llm_provider=LLMProvider.CLAUDE,
    llm_model="claude-sonnet-4-5-20250929",
    prompting_protocol=PromptingProtocolType.CASCADE,
    state_check_method=StateCheckMethod.HYBRID_MOTION_THEN_LLM,
    object_check_method=ObjectCheckMethod.CV_THEN_LLM,
    unknown_object_check_method=UnknownObjectCheckMethod.LLM_GUESS,
)

# Thorough: Maximum accuracy
PRESET_THOROUGH = BatchParameters(
    batch_id="thorough",
    config_name="thorough",
    llm_provider=LLMProvider.CLAUDE,
    llm_model="claude-sonnet-4-5-20250929",
    prompting_protocol=PromptingProtocolType.CASCADE,
    state_check_method=StateCheckMethod.LLM_DIRECT,
    object_check_method=ObjectCheckMethod.LLM_WITH_CV_HINT,
    unknown_object_check_method=UnknownObjectCheckMethod.LLM_GUESS_WITH_OPTIONS,
)
