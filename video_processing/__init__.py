"""Video Processing Package

Main entry point for video processing pipeline.
"""

from .batch_parameters import (
    BatchParameters,
    LLMProvider,
    CVModel,
    ToolDetectionMethod,
    ActionClassificationMethod,
    PromptTemplate,
    ProductivityAnalysisFormat,
    StorageBackend,
    ContextStrategy,
    ProcessingStrategy,
    PRESET_BASIC,
    PRESET_OBJECTS,
    PRESET_RELATIONSHIPS,
    PRESET_HTML_ANALYSIS,
    PRESET_FULL,
    PromptingProtocolType,
    StateCheckMethod,
    ObjectCheckMethod,
    UnknownObjectCheckMethod,
    PRESET_BASELINE,
    PRESET_CHEAP,
    PRESET_BALANCED,
    PRESET_THOROUGH
)

from .video_processor import process_video

__all__ = [
    'BatchParameters',
    'LLMProvider',
    'CVModel',
    'process_video',
    'PRESET_BASIC',
    'PRESET_OBJECTS',
    'PRESET_RELATIONSHIPS',
    'PRESET_HTML_ANALYSIS',
    'StorageBackend',
    'ContextStrategy',
    'ProcessingStrategy',
    'PromptingProtocolType',
    'StateCheckMethod',
    'ObjectCheckMethod',
    'UnknownObjectCheckMethod',
    'PRESET_BASELINE',
    'PRESET_CHEAP',
    'PRESET_BALANCED',
    'PRESET_THOROUGH'
]
