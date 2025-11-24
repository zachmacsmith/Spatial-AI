"""
Prompt Builder - Flexible prompt engineering

Supports multiple prompt templates and dynamic prompt construction.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class PromptBuilder:
    """Builds prompts for LLM based on configuration and context"""
    
    def __init__(self, batch_params):
        self.batch_params = batch_params
        self.prompt_templates = {
            'standard': self._build_standard_prompt,
            'detailed': self._build_detailed_prompt,
            'minimal': self._build_minimal_prompt,
            'custom': self._build_custom_prompt
        }
    
    def build_action_classification_prompt(
        self,
        motion_score: Optional[float] = None,
        detected_objects: Optional[List[Tuple[str, float]]] = None
    ) -> str:
        """
        Build prompt for action classification.
        
        Args:
            motion_score: Optional motion score
            detected_objects: Optional list of (object_name, confidence)
        
        Returns:
            Prompt string
        """
        template_name = self.batch_params.prompt_template.value
        builder_func = self.prompt_templates.get(template_name, self._build_standard_prompt)
        
        return builder_func(
            task='action_classification',
            motion_score=motion_score,
            detected_objects=detected_objects
        )
    
    def build_tool_detection_prompt(
        self,
        detected_objects: Optional[List[Tuple[str, float]]] = None
    ) -> str:
        """
        Build prompt for tool detection.
        
        Args:
            detected_objects: Optional list of (object_name, confidence)
        
        Returns:
            Prompt string
        """
        template_name = self.batch_params.prompt_template.value
        builder_func = self.prompt_templates.get(template_name, self._build_standard_prompt)
        
        return builder_func(
            task='tool_detection',
            detected_objects=detected_objects
        )
    
    def _build_standard_prompt(
        self,
        task: str,
        motion_score: Optional[float] = None,
        detected_objects: Optional[List[Tuple[str, float]]] = None
    ) -> str:
        """Standard prompt template (current approach)"""
        
        if task == 'action_classification':
            # Base prompt
            actions_joined = ", ".join(self.batch_params.allowed_actions)
            prompt = (
                f"This is a POV of a person. "
                f"Check if hands are visible. "
            )
            
            # Add motion context if enabled
            if self.batch_params.include_motion_score and motion_score is not None:
                prompt += (
                    f"Motion score: {motion_score:.2f}, "
                    f"a motion score of 0 to 0.16 suggests the person is idle, "
                    f"0.16 or above suggests they are moving. "
                )
            
            # Add object context if enabled
            if self.batch_params.include_object_list and detected_objects:
                objects_to_show = detected_objects[:self.batch_params.max_objects_in_prompt]
                if self.batch_params.include_object_confidence:
                    objects_str = ', '.join([f"{obj[0]} ({obj[1]:.2f})" for obj in objects_to_show])
                else:
                    objects_str = ', '.join([obj[0] for obj in objects_to_show])
                prompt += f"Found objects: {objects_str}. "
            
            # Add classification instruction
            prompt += (
                f"You must classify the person's behavior into ONE of the following categories: "
                f"{actions_joined}. Respond ONLY with one word from this list, no explanations."
            )
            
            return prompt
        
        elif task == 'tool_detection':
            tools_joined = ", ".join(self.batch_params.allowed_tools)
            prompt = (
                f"Which tool is the person using? "
                f"Choose ONLY ONE from this list: {tools_joined}. "
                f"Respond with ONLY the tool name, no explanations."
            )
            return prompt
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _build_detailed_prompt(
        self,
        task: str,
        motion_score: Optional[float] = None,
        detected_objects: Optional[List[Tuple[str, float]]] = None
    ) -> str:
        """Detailed prompt with more context"""
        
        if task == 'action_classification':
            actions_joined = ", ".join(self.batch_params.allowed_actions)
            prompt = (
                f"You are analyzing a first-person video of a worker. "
                f"Your task is to classify their current activity.\n\n"
            )
            
            if self.batch_params.include_motion_score and motion_score is not None:
                motion_desc = "low (idle)" if motion_score < 0.16 else "high (active)"
                prompt += f"Motion level: {motion_score:.2f} ({motion_desc})\n"
            
            if self.batch_params.include_object_list and detected_objects:
                prompt += "Detected objects in scene:\n"
                for obj_name, conf in detected_objects[:self.batch_params.max_objects_in_prompt]:
                    prompt += f"  - {obj_name} (confidence: {conf:.0%})\n"
            
            prompt += (
                f"\nClassify the worker's activity as ONE of: {actions_joined}\n"
                f"Respond with ONLY the category name, nothing else."
            )
            
            return prompt
        
        elif task == 'tool_detection':
            tools_joined = ", ".join(self.batch_params.allowed_tools)
            prompt = (
                f"You are analyzing a first-person video of a worker.\n"
                f"Identify which tool they are currently using.\n\n"
                f"Available tools: {tools_joined}\n\n"
                f"Respond with ONLY the tool name from the list above."
            )
            return prompt
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _build_minimal_prompt(
        self,
        task: str,
        motion_score: Optional[float] = None,
        detected_objects: Optional[List[Tuple[str, float]]] = None
    ) -> str:
        """Minimal prompt (concise)"""
        
        if task == 'action_classification':
            actions_joined = ", ".join(self.batch_params.allowed_actions)
            return f"Classify activity: {actions_joined}. One word only."
        
        elif task == 'tool_detection':
            tools_joined = ", ".join(self.batch_params.allowed_tools)
            return f"Which tool: {tools_joined}?"
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _build_custom_prompt(
        self,
        task: str,
        motion_score: Optional[float] = None,
        detected_objects: Optional[List[Tuple[str, float]]] = None
    ) -> str:
        """Load custom prompt from file"""
        
        if self.batch_params.custom_prompt_path is None:
            raise ValueError("custom_prompt_path not specified in BatchParameters")
        
        # Load custom prompt template
        with open(self.batch_params.custom_prompt_path, 'r') as f:
            template = f.read()
        
        # Replace placeholders
        template = template.replace("{task}", task)
        template = template.replace("{actions}", ", ".join(self.batch_params.allowed_actions))
        template = template.replace("{tools}", ", ".join(self.batch_params.allowed_tools))
        
        if motion_score is not None:
            template = template.replace("{motion_score}", f"{motion_score:.2f}")
        
        if detected_objects:
            objects_str = ", ".join([obj[0] for obj in detected_objects[:self.batch_params.max_objects_in_prompt]])
            template = template.replace("{objects}", objects_str)
        
        return template
