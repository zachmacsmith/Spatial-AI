from typing import Callable, List, Protocol
from .context_store import ContextStore
from .frame_context import FrameContext

class PromptComponent(Protocol):
    def __call__(self, store: ContextStore, current: FrameContext) -> str:
        ...

# Individual components
def motion_component(store: ContextStore, current: FrameContext) -> str:
    if current.motion_score is None:
        return ""
    score = current.motion_score
    if score < 0.16:
        hint = "suggests idle"
    else:
        hint = "suggests movement"
    return f"Motion score: {score:.2f} ({hint})."

def objects_basic_component(store: ContextStore, current: FrameContext) -> str:
    if not current.detections:
        return "No objects detected."
    names = [d.class_name for d in current.detections[:5]]
    return f"Objects: {', '.join(names)}."

def objects_with_confidence_component(store: ContextStore, current: FrameContext) -> str:
    if not current.detections:
        return "No objects detected."
    parts = [f"{d.class_name} ({d.confidence:.0%})" for d in current.detections[:5]]
    return f"Objects: {', '.join(parts)}."

def temporal_objects_component(store: ContextStore, current: FrameContext) -> str:
    """Rich temporal context about objects"""
    if not current.detections:
        return "No objects detected."
    
    lines = []
    for det in current.detections[:5]:
        history = store.get_object_history(det.class_name)
        if history["present"] and history["duration_seconds"] > 0:
            lines.append(f"{det.class_name}: visible for {history['duration_seconds']:.1f}s")
        else:
            lines.append(f"{det.class_name}: just appeared")
    
    new_objects = store.get_new_objects(since_n_frames=15)
    if new_objects:
        lines.append(f"Recently appeared: {', '.join(new_objects)}")
    
    return "Objects:\n" + "\n".join(f"  - {line}" for line in lines)

def action_history_component(store: ContextStore, current: FrameContext) -> str:
    recent = store.get_recent_actions(5)
    if not recent:
        return ""
    # Filter out None values just in case
    valid_actions = [a for a in recent if a]
    if not valid_actions:
        return ""
    return f"Recent actions: {' → '.join(valid_actions)}"

def relationships_component(store: ContextStore, current: FrameContext) -> str:
    if not current.relationships:
        return ""
    rel_strs = [f"{' + '.join(sorted(r))}" for r in current.relationships]
    return f"Object proximity: {'; '.join(rel_strs)}"


# Component registry
COMPONENTS = {
    "motion": motion_component,
    "objects_basic": objects_basic_component,
    "objects_confidence": objects_with_confidence_component,
    "objects_temporal": temporal_objects_component,
    "action_history": action_history_component,
    "relationships": relationships_component,
}


class ComposableContextBuilder:
    """Builds prompt context from selected components"""
    
    def __init__(self, component_names: List[str]):
        self.components = []
        for name in component_names:
            if name not in COMPONENTS:
                raise ValueError(f"Unknown component: {name}. Available: {list(COMPONENTS.keys())}")
            self.components.append(COMPONENTS[name])
    
    def build(self, store: ContextStore, current: FrameContext) -> str:
        parts = []
        for component in self.components:
            result = component(store, current)
            if result:
                parts.append(result)
        return "\n".join(parts)


# Preset strategies
STRATEGIES = {
    "minimal": ["motion", "objects_basic"],
    "standard": ["motion", "objects_confidence"],
    "temporal": ["motion", "objects_temporal", "action_history"],
    "full": ["motion", "objects_temporal", "action_history", "relationships"],
}

def get_context_builder(strategy: str) -> ComposableContextBuilder:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown context strategy: {strategy}. Available: {list(STRATEGIES.keys())}")
    return ComposableContextBuilder(STRATEGIES[strategy])
