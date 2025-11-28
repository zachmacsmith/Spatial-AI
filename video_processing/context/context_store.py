from collections import deque
from typing import Dict, List, Optional
from .frame_context import FrameContext

class ContextStore:
    """Accumulates frame contexts with temporal queries"""
    
    def __init__(self, fps: float, window_seconds: float = 5.0):
        self.fps = fps
        self.window_size = int(fps * window_seconds)
        self.frames: deque[FrameContext] = deque(maxlen=self.window_size)
        
        # Note: This grows unbounded. For very long videos, we might need a cleanup strategy.
        # For typical construction clips (minutes), this is fine (a few MBs).
        self._by_number: Dict[int, FrameContext] = {}
    
    def add(self, ctx: FrameContext):
        self.frames.append(ctx)
        self._by_number[ctx.frame_number] = ctx
    
    def get(self, frame_number: int) -> Optional[FrameContext]:
        return self._by_number.get(frame_number)
    
    def all_contexts(self) -> List[FrameContext]:
        """Get all stored contexts sorted by frame number"""
        return [self._by_number[k] for k in sorted(self._by_number.keys())]
    
    def all_frame_numbers(self) -> List[int]:
        """Get all stored frame numbers in order"""
        return sorted(self._by_number.keys())
        
    def clear(self):
        """Clear all stored contexts"""
        self.frames.clear()
        self._by_number.clear()
    
    @property
    def latest(self) -> Optional[FrameContext]:
        return self.frames[-1] if self.frames else None
    
    # Temporal queries
    def get_recent_actions(self, n: int = 5) -> List[str]:
        """Last N classified actions"""
        # Only consider frames that actually have an action assigned
        actions = [f.action for f in self.frames if f.action]
        return actions[-n:]
    
    def get_object_history(self, object_name: str) -> dict:
        """When did this object appear? How long visible?"""
        appearances = [f.frame_number for f in self.frames 
                       if object_name in f.object_names]
        if not appearances:
            return {"present": False}
        
        first_seen = appearances[0]
        last_seen = appearances[-1]
        
        # Calculate duration based on span (first to last)
        # Note: In sparse keyframe mode, this is the "span" of time the object was present,
        # not necessarily continuous visibility.
        duration_seconds = (last_seen - first_seen) / self.fps
        
        return {
            "present": True,
            "first_frame": first_seen,
            "last_frame": last_seen,
            "duration_seconds": duration_seconds,
            "observation_count": len(appearances),
            # Continuous if we have an observation for every frame in the span
            # (Only accurate if we are processing every frame, not just keyframes)
            "continuous": len(appearances) == (last_seen - first_seen + 1)
        }
    
    def get_new_objects(self, since_n_frames: int = 30) -> List[str]:
        """Objects that appeared recently"""
        if len(self.frames) < 2:
            return []
            
        # If we don't have enough history to look back 'since_n_frames',
        # we can't reliably say what's "new" vs just "what we have".
        # Return empty to be safe, or we could return everything if we want to be aggressive.
        # Safe approach: return empty.
        if len(self.frames) <= since_n_frames:
            return []
        
        recent_frames = list(self.frames)[-since_n_frames:]
        older_frames = list(self.frames)[:-since_n_frames]
        
        recent_objects = set()
        for f in recent_frames:
            recent_objects.update(f.object_names)
        
        older_objects = set()
        for f in older_frames:
            older_objects.update(f.object_names)
        
        return list(recent_objects - older_objects)
    
    def get_persistent_objects(self, min_frames: int = 10) -> List[str]:
        """Objects visible for at least N frames"""
        from collections import Counter
        counts = Counter()
        for f in self.frames:
            counts.update(f.object_names)
        return [obj for obj, count in counts.items() if count >= min_frames]
    
    def get_relationship_summary(self) -> List[dict]:
        """Recent relationship events"""
        # Track when relationships started/ended
        events = []
        active = {}  # frozenset -> start_frame
        
        for f in self.frames:
            current = set(f.relationships)
            
            # New relationships
            for rel in current:
                if rel not in active:
                    active[rel] = f.frame_number
                    events.append({
                        "type": "started",
                        "objects": list(rel),
                        "frame": f.frame_number
                    })
            
            # Ended relationships
            ended = [r for r in active if r not in current]
            for rel in ended:
                events.append({
                    "type": "ended",
                    "objects": list(rel),
                    "frame": f.frame_number,
                    "duration_frames": f.frame_number - active[rel]
                })
                del active[rel]
        
        return events
