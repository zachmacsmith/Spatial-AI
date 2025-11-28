"""
Performance Tracker
Tracks timing and metrics for video processing pipeline.
"""
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

@dataclass
class PerformanceTracker:
    """
    Tracks performance metrics for a processing batch.
    """
    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    section_starts: Dict[str, float] = field(default_factory=dict)
    section_times: Dict[str, float] = field(default_factory=dict)
    
    # Counters
    api_calls: int = 0
    frames_processed: int = 0
    
    # Video stats
    video_duration: float = 0.0
    video_fps: float = 0.0
    
    def start_section(self, name: str):
        """Start timing a named section."""
        self.section_starts[name] = time.time()
        
    def end_section(self, name: str):
        """End timing a named section and record duration."""
        if name in self.section_starts:
            duration = time.time() - self.section_starts[name]
            # Accumulate if section is visited multiple times
            self.section_times[name] = self.section_times.get(name, 0.0) + duration
            del self.section_starts[name]
            
    def increment_api_calls(self, count: int = 1):
        """Increment API call counter."""
        self.api_calls += count
        
    def set_video_stats(self, duration: float, fps: float, frames: int):
        """Set video statistics."""
        self.video_duration = duration
        self.video_fps = fps
        self.frames_processed = frames
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics dictionary."""
        total_time = (self.end_time or time.time()) - self.start_time
        
        # Calculate speed ratio (processing time / video duration)
        # Lower is better (faster). < 1.0 means faster than realtime.
        speed_ratio = 0.0
        if self.video_duration > 0:
            speed_ratio = total_time / self.video_duration
            
        # Processing speed (fps)
        processing_fps = 0.0
        if total_time > 0:
            processing_fps = self.frames_processed / total_time
            
        return {
            "total_time_seconds": round(total_time, 2),
            "video_duration_seconds": round(self.video_duration, 2),
            "speed_ratio": round(speed_ratio, 2),
            "processing_fps": round(processing_fps, 2),
            "realtime_factor": round(1.0 / speed_ratio, 2) if speed_ratio > 0 else 0,
            "api_calls_total": self.api_calls,
            "section_times_seconds": {k: round(v, 2) for k, v in self.section_times.items()},
            "frames_processed": self.frames_processed
        }
