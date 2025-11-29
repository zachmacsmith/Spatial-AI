"""
Prompting Protocols
Orchestration logic for classification workflows.
"""
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass
import numpy as np

from .context import FrameContext, ContextStore
from .batch_parameters import BatchParameters
from .decision_functions import (
    DecisionContext,
    get_state_check,
    get_object_check,
    get_unknown_object_check
)

@dataclass
class ClassificationResult:
    """Complete result from a prompting protocol"""
    action: str                           # "idle", "moving", "using tool"
    tool: Optional[str] = None            # "drill", "saw", etc. or "unknown"
    tool_guess: Optional[str] = None      # Freeform guess when tool="unknown"
    api_calls_used: int = 0               # For cost tracking
    confidence: Optional[float] = None    # Optional confidence score

class PromptingProtocol:
    """Base class for prompting protocols"""
    
    def classify(
        self,
        frames: List[np.ndarray],
        frame_context: FrameContext,
        context_store: ContextStore,
        context_text: str,
        llm_service: Any,
        batch_params: BatchParameters,
        frame_cache: Optional[Dict[int, np.ndarray]] = None,
        cv_service: Any = None
    ) -> ClassificationResult:
        """
        Perform classification using this protocol.
        
        Args:
            frames: Image frames to analyze
            frame_context: Current frame's context data
            context_store: Full store for temporal queries
            context_text: Pre-rendered context from ContextBuilder
            llm_service: LLM service for API calls
            batch_params: Configuration parameters
        
        Returns:
            ClassificationResult with action, tool, guess, and API call count
        """
        raise NotImplementedError

# ==========================================
# REGISTRY
# ==========================================

PROMPTING_PROTOCOL_REGISTRY: Dict[str, Callable[[], PromptingProtocol]] = {}

def register_prompting_protocol(name: str):
    def decorator(cls):
        PROMPTING_PROTOCOL_REGISTRY[name] = cls
        return cls
    return decorator

def get_prompting_protocol(name: str) -> PromptingProtocol:
    if name not in PROMPTING_PROTOCOL_REGISTRY:
        raise ValueError(f"Unknown prompting protocol: {name}. Available: {list(PROMPTING_PROTOCOL_REGISTRY.keys())}")
    return PROMPTING_PROTOCOL_REGISTRY[name]()

# ==========================================
# IMPLEMENTATIONS
# ==========================================

@register_prompting_protocol("single_shot")
class SingleShotProtocol(PromptingProtocol):
    """
    Classic behavior: One prompt asks for action classification.
    Does not identify tools separately (unless part of action string).
    """
    
    def classify(
        self,
        frames: List[np.ndarray],
        frame_context: FrameContext,
        context_store: ContextStore,
        context_text: str,
        llm_service: Any,
        batch_params: BatchParameters,
        frame_cache: Optional[Dict[int, np.ndarray]] = None,
        cv_service: Any = None
    ) -> ClassificationResult:
        
        actions_joined = ", ".join(batch_params.allowed_actions)
        prompt_text = (
            f"This is a POV of a construction worker.\n\n"
            f"{context_text}\n\n"
            f"Classify the action: {actions_joined}.\n"
            f"If the action is 'using tool', specify the tool (e.g., 'using tool: hammer').\n"
            f"Otherwise, respond with ONE word only."
        )
        
        response = llm_service.send_multiframe_prompt(
            frames=frames,
            prompt_text=prompt_text,
            max_tokens=50
        ).strip().lower()
        
        # Parse response
        action = response
        tool = None
        
        if "using tool" in response:
            action = "using tool"
            if ":" in response:
                tool = response.split(":", 1)[1].strip()
            else:
                tool = "unknown"
        elif "moving" in response:
            action = "moving"
        elif "idle" in response:
            action = "idle"
            
        return ClassificationResult(
            action=action,
            tool=tool,
            api_calls_used=1
        )

@register_prompting_protocol("cascade")
class CascadeProtocol(PromptingProtocol):
    """
    Orchestrates decision functions in sequence with early termination.
    """
    
    def classify(
        self,
        frames: List[np.ndarray],
        frame_context: FrameContext,
        context_store: ContextStore,
        context_text: str,
        llm_service: Any,
        batch_params: BatchParameters,
        frame_cache: Optional[Dict[int, np.ndarray]] = None,
        cv_service: Any = None
    ) -> ClassificationResult:
        
        # 1. Build DecisionContext
        ctx = DecisionContext(
            frames=frames,
            frame_context=frame_context,
            context_store=context_store,
            context_text=context_text,
            llm_service=llm_service,
            batch_params=batch_params,
            frame_cache=frame_cache,
            cv_service=cv_service
        )
        
        # 2. Get configured decision functions from registries
        state_check = get_state_check(batch_params.state_check_method.value)
        object_check = get_object_check(batch_params.object_check_method.value)
        unknown_check = get_unknown_object_check(batch_params.unknown_object_check_method.value)
        
        # 3. Stage 1: Check state
        state = state_check(ctx)
        
        if state == "uncertain":
            state = "idle"  # Default fallback
        
        # Early termination for idle/moving
        if state in ["idle", "moving"]:
            return ClassificationResult(
                action=state,
                api_calls_used=ctx.api_calls_used
            )
        
        # 4. Stage 2: Check object (only reached if using_tool)
        tool = object_check(ctx)
        
        # Early termination for known tool
        if tool != "unknown":
            return ClassificationResult(
                action="using tool",
                tool=tool,
                api_calls_used=ctx.api_calls_used
            )
        
        # 5. Stage 3: Check unknown object
        tool_guess = unknown_check(ctx)
        
        return ClassificationResult(
            action="using tool",
            tool="unknown",
            tool_guess=tool_guess,
            api_calls_used=ctx.api_calls_used
        )
