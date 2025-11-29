from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np

from .frame_context import FrameContext
from .context_store import ContextStore
from ..batch_parameters import BatchParameters

@dataclass
class DecisionContext:
    """
    All information available for making a decision.
    Decision functions receive this and use whatever subset they need.
    """
    frames: List[np.ndarray]
    frame_context: FrameContext
    context_store: ContextStore
    context_text: str
    llm_service: Any
    batch_params: BatchParameters
    frame_cache: Optional[Dict[int, np.ndarray]] = None
    cv_service: Any = None
    api_calls_used: int = 0
    
    def call_llm(self, prompt: str, max_tokens: int = 50, valid_options: Optional[List[str]] = None, log_label: str = "LLM Call") -> str:
        """
        Wrapper for LLM calls that tracks usage.
        Increments self.api_calls_used.
        Returns stripped response text.
        """
        self.api_calls_used += 1
        
        print(f"\n=== {log_label} ===")
        print(f"LLM Prompt: {prompt}")
        if valid_options:
            print(f"Valid Options: {valid_options}")
            
        # Call LLM service
        # We want to see the RAW response before validation for debugging
        # But send_multiframe_prompt does validation internally.
        # So we'll rely on the service to return the validated result,
        # but we'll print it clearly.
        
        result = self.llm_service.send_multiframe_prompt(
            frames=self.frames,
            prompt_text=prompt,
            max_tokens=max_tokens,
            valid_options=valid_options
        ).strip()
        
        print(f"LLM Response (Validated): {result}")
        print("==================\n")
        
        return result
        
    def call_llm_no_image(self, prompt: str, max_tokens: int = 50, valid_options: Optional[List[str]] = None) -> str:
        """
        LLM call without sending images (cheaper).
        For decisions that only need text context.
        Increments self.api_calls_used.
        """
        # TODO: Implement text-only call if service supports it
        # For now, just use same method but maybe empty frames? 
        # Or just pass frames anyway since our interface requires them.
        return self.call_llm(prompt, max_tokens, valid_options)
