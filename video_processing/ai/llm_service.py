"""
LLM Service - Language Model Abstraction

Supports multiple LLM providers:
- Claude (Anthropic)
- Gemini (Google)
- OpenAI
- Custom providers

Separated from CV service for clean architecture.
"""

import base64
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import cv2
import numpy as np


class LLMService(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def send_multiframe_prompt(
        self,
        frames: List[np.ndarray],
        prompt_text: str,
        max_tokens: int = 1000,
        temperature: float = 0.0
    ) -> str:
        """Send frames + text prompt to LLM, return response"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name for logging/tracking"""
        pass


class ClaudeService(LLMService):
    """Claude (Anthropic) LLM Service"""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def send_multiframe_prompt(
        self,
        frames: List[np.ndarray],
        prompt_text: str,
        max_tokens: int = 1000,
        temperature: float = 0.0
    ) -> str:
        """Send frames to Claude with prompt"""
        # Encode frames
        encoded_frames = self._encode_frames(frames)
        
        # Build content list
        content_list = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded}
            }
            for encoded in encoded_frames
        ]
        content_list.append({"type": "text", "text": prompt_text})
        
        # Call Claude API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content_list}]
        )
        
        return response.content[0].text.strip()
    
    def _encode_frames(self, frames: List[np.ndarray], width: int = 640, height: int = 360) -> List[str]:
        """Encode frames as base64 JPEG"""
        encoded_list = []
        for frame in frames:
            frame_small = cv2.resize(frame, (width, height))
            _, buffer_img = cv2.imencode('.jpg', frame_small)
            encoded_list.append(base64.b64encode(buffer_img.tobytes()).decode("utf-8"))
        return encoded_list
    
    def get_provider_name(self) -> str:
        return f"claude_{self.model}"


class GeminiService(LLMService):
    """Gemini (Google) LLM Service"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            self.model_name = model
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    def send_multiframe_prompt(
        self,
        frames: List[np.ndarray],
        prompt_text: str,
        max_tokens: int = 1000,
        temperature: float = 0.0
    ) -> str:
        """Send frames to Gemini with prompt"""
        import google.generativeai as genai
        from PIL import Image
        
        # Convert frames to PIL Images
        pil_images = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame_small = cv2.resize(frame_rgb, (640, 360))
            pil_images.append(Image.fromarray(frame_small))
        
        # Build content list (images + text)
        content = pil_images + [prompt_text]
        
        # Call Gemini API
        response = self.model.generate_content(
            content,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        )
        
        return response.text.strip()
    
    def get_provider_name(self) -> str:
        return f"gemini_{self.model_name}"


class OpenAIService(LLMService):
    """OpenAI (GPT-4 Vision) LLM Service"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def send_multiframe_prompt(
        self,
        frames: List[np.ndarray],
        prompt_text: str,
        max_tokens: int = 1000,
        temperature: float = 0.0
    ) -> str:
        """Send frames to OpenAI with prompt"""
        # Encode frames
        encoded_frames = self._encode_frames(frames)
        
        # Build messages
        content = []
        for encoded in encoded_frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}"
                }
            })
        content.append({"type": "text", "text": prompt_text})
        
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content.strip()
    
    def _encode_frames(self, frames: List[np.ndarray], width: int = 640, height: int = 360) -> List[str]:
        """Encode frames as base64 JPEG"""
        encoded_list = []
        for frame in frames:
            frame_small = cv2.resize(frame, (width, height))
            _, buffer_img = cv2.imencode('.jpg', frame_small)
            encoded_list.append(base64.b64encode(buffer_img.tobytes()).decode("utf-8"))
        return encoded_list
    
    def get_provider_name(self) -> str:
        return f"openai_{self.model}"


def get_llm_service(batch_params) -> LLMService:
    """
    Factory function to get LLM service based on BatchParameters
    
    Args:
        batch_params: BatchParameters instance
    
    Returns:
        LLMService instance for the configured provider
    """
    from ..batch_parameters import LLMProvider
    
    # Get API key from batch_params or config
    api_key = batch_params.llm_api_key
    
    if api_key is None:
        # Try to import from config.py (in root directory, not shared/)
        try:
            import sys
            import os
            # Add parent directory to path to import config
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import config
            
            # Only import the key we need
            if batch_params.llm_provider.value == "claude":
                api_key = getattr(config, 'ANTHROPIC_API_KEY', None)
            elif batch_params.llm_provider.value == "gemini":
                api_key = getattr(config, 'GEMINI_API_KEY', None)
            elif batch_params.llm_provider.value == "openai":
                api_key = getattr(config, 'OPENAI_API_KEY', None)
            
            if api_key is None:
                raise ValueError(f"API key for {batch_params.llm_provider.value} not found in config.py")
        except (ImportError, AttributeError) as e:
            raise ValueError(f"No API key provided and config.py not found or incomplete: {e}")
    
    # Create appropriate service
    if batch_params.llm_provider.value == "claude":
        return ClaudeService(api_key, batch_params.llm_model)
    elif batch_params.llm_provider.value == "gemini":
        return GeminiService(api_key, batch_params.llm_model)
    elif batch_params.llm_provider.value == "openai":
        return OpenAIService(api_key, batch_params.llm_model)
    else:
        raise ValueError(f"Unsupported LLM provider: {batch_params.llm_provider}")
